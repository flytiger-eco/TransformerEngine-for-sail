# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pack routes directly into the owners' buffers for two-rank expert parallelism."""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import transformer_engine_torch as tex


# A fixed-width collective also lets peers report incompatible expert partitions.
_MAX_LOCAL_EXPERTS = 64
_BOUNDS = 2 * _MAX_LOCAL_EXPERTS + 1
_STRIDE = _BOUNDS + 6
_INT_MAX = 2**31 - 1


@triton.jit
def _route_header(
    ids,
    local_positions,
    header,
    block_counts,
    slots,
    experts,
    hidden,
    dtype,
    topk: tl.constexpr,
    row_stride,
    slot_stride: tl.constexpr,
    bounds: tl.constexpr,
    check_ids: tl.constexpr,
    stable: tl.constexpr,
    bins: tl.constexpr,
    block: tl.constexpr,
):
    if check_ids:
        slot = tl.program_id(0) * block + tl.arange(0, block)
        token = (slot // topk).to(tl.int64)
        expert = tl.load(
            ids + token * row_stride + (slot % topk) * slot_stride, mask=slot < slots, other=-1
        )
        valid = (expert >= 0) & (expert < 2 * experts)
        if stable:
            # Backward must keep a reproducible reduction order for weight gradients.
            counts = tl.histogram(tl.where(valid, expert, bins - 1), bins)
            g = tl.arange(0, bins)
            tl.store(
                block_counts + g * tl.num_programs(0) + tl.program_id(0),
                counts,
                mask=g < 2 * experts,
            )
            tl.atomic_add(header + g, counts, mask=(g < 2 * experts) & (counts > 0), sem="relaxed")
            tl.store(local_positions + slot, -1, mask=(slot < slots) & ~valid)
        else:
            pos = tl.atomic_add(header + tl.where(valid, expert, 0), 1, mask=valid, sem="relaxed")
            tl.store(local_positions + slot, tl.where(valid, pos, -1), mask=slot < slots)
        if tl.sum(((expert != -1) & ~valid).to(tl.int32), 0):
            tl.atomic_or(header + bounds + 5, 1, sem="relaxed")
    if tl.program_id(0) == 0:
        tl.store(header + bounds, slots)
        tl.store(header + bounds + 1, experts)
        tl.store(header + bounds + 2, hidden)
        tl.store(header + bounds + 3, dtype)
        tl.store(header + bounds + 4, check_ids)


@triton.jit
def _expert_layout(
    metadata,
    src_bases,
    offsets,
    experts: tl.constexpr,
    rank: tl.constexpr,
    stride: tl.constexpr,
    block: tl.constexpr,
):
    expert = tl.arange(0, block)
    a = tl.load(metadata + expert, mask=expert < 2 * experts, other=0)
    b = tl.load(metadata + stride + expert, mask=expert < 2 * experts, other=0)
    total = a + b
    first_owner = tl.sum(tl.where(expert < experts, total, 0), 0)
    within = tl.cumsum(total, 0) - total - tl.where(expert >= experts, first_owner, 0)
    tl.store(src_bases + expert, within + (a if rank == 1 else 0), mask=expert < 2 * experts)
    local = expert - rank * experts
    tl.store(offsets + local, within, mask=(local >= 0) & (local < experts))
    local_count = first_owner if rank == 0 else tl.sum(tl.where(expert >= experts, total, 0), 0)
    tl.store(offsets + experts, local_count)


@triton.jit
def _route_positions(
    ids,
    positions,
    src_bases,
    probs,
    prob0,
    prob1,
    slots,
    experts: tl.constexpr,
    topk: tl.constexpr,
    id_row_stride,
    id_slot_stride: tl.constexpr,
    prob_row_stride,
    prob_slot_stride: tl.constexpr,
    block: tl.constexpr,
):
    slot = tl.program_id(0) * block + tl.arange(0, block)
    token = (slot // topk).to(tl.int64)
    expert = tl.load(
        ids + token * id_row_stride + (slot % topk) * id_slot_stride, mask=slot < slots, other=-1
    )
    valid = expert >= 0
    base = tl.load(src_bases + tl.maximum(expert, 0))
    local_pos = tl.load(positions + slot, mask=valid, other=0)
    index = base + local_pos
    weight = tl.load(
        probs + token * prob_row_stride + (slot % topk) * prob_slot_stride, mask=valid, other=0
    )
    tl.store(tl.where(expert < experts, prob0, prob1) + index, weight, mask=valid)
    tl.store(positions + slot, index, mask=valid)


def _bulk_plan(x, ids, fc1, fc2, group, state, backward=False):
    """Exchange counts and construct the packed owner layout for this invocation."""
    topk = ids.size(1)
    hidden = x.size(1)
    intermediate = fc2[0].size(1)
    experts = len(fc1)
    slots = ids.numel()
    can_pack = (
        x.device == state.device
        and (x.dtype == torch.bfloat16 or (backward and x.dtype == torch.float16))
        and hidden >= 256
        and hidden % 8 == 0
        and intermediate >= 128
        and intermediate % 8 == 0
        and experts <= _MAX_LOCAL_EXPERTS
        and topk <= 8
        and slots <= _INT_MAX
    )
    header = ids.new_zeros(_STRIDE)
    positions = ids.new_empty(slots) if can_pack else header
    blocks = max(1, triton.cdiv(slots, 256)) if can_pack else 1
    counts = ids.new_empty((2 * experts, blocks)) if can_pack and backward else header
    _route_header[(blocks,)](
        ids if slots else header,
        positions if slots else header,
        header,
        counts,
        min(slots, _INT_MAX),
        experts,
        hidden,
        int(x.dtype == torch.float16),
        topk,
        *ids.stride(),
        _BOUNDS,
        can_pack,
        backward,
        triton.next_power_of_2(2 * experts + 1) if can_pack and backward else 1,
        256,
    )
    both = ids.new_empty(2 * _STRIDE)
    dist.all_gather_into_tensor(both, header, group=group)
    peers = both.view(2, _STRIDE).tolist()
    h0, h1 = (p[_BOUNDS:] for p in peers)
    if h0[2:4] != h1[2:4]:
        raise ValueError("Expert-parallel input width and dtype must match on every rank")
    if h0[1] != h1[1] or h0[1] <= 0 or h0[5] or h1[5]:
        raise ValueError(
            "Expected equal positive local expert counts and expert IDs in range or -1"
        )
    if not (h0[4] and h1[4]) or max(h0[0], h1[0]) < 4096:
        return None
    rank = group.rank()
    sizes = [sum(sum(p[r * experts : (r + 1) * experts]) for p in peers) for r in range(2)]
    if sum(sizes) > _INT_MAX:
        return None
    offsets = ids.new_empty(experts + 1)
    src_bases = ids.new_empty(2 * experts)
    _expert_layout[(1,)](
        both, src_bases, offsets, experts, rank, _STRIDE, triton.next_power_of_2(2 * experts)
    )

    return sizes, positions, src_bases, offsets, counts


def bulk_forward(x, ids, probs, fc1, fc2, group, state):
    """Return None collectively when the two ranks should use the generic peer path."""
    plan = _bulk_plan(x, ids, fc1, fc2, group, state)
    if plan is None:
        return None
    sizes, positions, src_bases, offsets, _ = plan
    if not max(sizes):
        return torch.zeros_like(x, memory_format=torch.contiguous_format)
    rank, slots, topk, experts, hidden = group.rank(), ids.numel(), ids.size(1), len(fc1), x.size(1)
    # The same tile-aligned capacity on both ranks keeps rendezvous collective when
    # routing changes. FC2 writes back over the packed scratch after FC1 consumes it.
    capacity = triton.cdiv(max(sizes), 1024) * 1024
    prob_rows = triton.cdiv(capacity * 4, hidden * x.element_size())
    buffers = state.buffers(group, (capacity + prob_rows, hidden), x)
    # A preceding small-batch call can still be reading its local combine slots
    # on a different stream. Wait for both peers before writing their scratch.
    state.handle.barrier(channel=3)
    scales = [buf[capacity:].view(torch.float32).view(-1) for buf in buffers]
    if slots:
        _route_positions[(triton.cdiv(slots, 256),)](
            ids,
            positions,
            src_bases,
            probs,
            *scales,
            slots,
            experts,
            topk,
            *ids.stride(),
            *probs.stride(),
            256,
        )
        tex.mega_moe_bulk_push(x, ids, positions, *buffers, experts, rank)
    state.handle.barrier(channel=0)
    packed = buffers[rank][: sizes[rank]]
    tex.mega_moe_local_forward(packed, scales[rank][: sizes[rank]], offsets, fc1, fc2, out=packed)
    state.handle.barrier(channel=1)
    result = tex.mega_moe_bulk_pull(ids, positions, *buffers, experts, rank)
    # Later calls may reuse the same symmetric addresses on another stream or rank.
    state.handle.barrier(channel=2)
    state.done.record(torch.cuda.current_stream(x.device))
    return result
