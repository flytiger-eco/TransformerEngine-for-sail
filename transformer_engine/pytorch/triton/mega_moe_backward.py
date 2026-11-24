# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Two-rank backward dispatch and combine in contiguous expert-major storage."""

import torch
import triton
import triton.language as tl
import transformer_engine_torch as tex

from .mega_moe_bulk import _bulk_plan


@triton.jit
def _backward_positions(
    ids,
    probs,
    block_ends,
    src_bases,
    positions,
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
    # Rank order followed by increasing source slot order matches the collective
    # route plan, including the two occurrences of a duplicated expert.
    chunk, expert = tl.program_id(0), tl.program_id(1)
    slot = chunk * block + tl.arange(0, block)
    token, k = (slot // topk).to(tl.int64), slot % topk
    actual = tl.load(ids + token * id_row_stride + k * id_slot_stride, mask=slot < slots, other=-1)
    selected = actual == expert
    before = tl.load(block_ends + expert * tl.num_programs(0) + chunk - 1, mask=chunk > 0, other=0)
    row = tl.load(src_bases + expert) + before + tl.cumsum(selected.to(tl.int32), 0) - 1
    tl.store(positions + slot, row, mask=selected)
    scale = tl.load(probs + token * prob_row_stride + k * prob_slot_stride, mask=selected, other=0)
    tl.store(tl.where(expert < experts, prob0, prob1) + row, scale, mask=selected)


@triton.jit
def _backward_push(
    x,
    dy,
    ids,
    positions,
    peer0,
    peer1,
    capacity,
    hidden: tl.constexpr,
    experts: tl.constexpr,
    topk: tl.constexpr,
    x_row_stride,
    x_col_stride: tl.constexpr,
    dy_row_stride,
    dy_col_stride: tl.constexpr,
    id_row_stride,
    id_slot_stride: tl.constexpr,
    columns: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * columns + tl.arange(0, columns)
    value = tl.load(x + token * x_row_stride + col * x_col_stride, mask=col < hidden, other=0)
    grad = tl.load(dy + token * dy_row_stride + col * dy_col_stride, mask=col < hidden, other=0)
    for k in tl.static_range(topk):
        expert = tl.load(ids + token * id_row_stride + k * id_slot_stride)
        if expert >= 0:
            row = tl.load(positions + token * topk + k).to(tl.int64)
            dst = tl.where(expert < experts, peer0, peer1) + row * hidden + col
            tl.store(dst, value, mask=col < hidden)
            tl.store(dst + capacity.to(tl.int64) * hidden, grad, mask=col < hidden)


@triton.jit
def _backward_pull(
    ids,
    positions,
    dx0,
    dx1,
    dp0,
    dp1,
    dx,
    dp,
    hidden: tl.constexpr,
    experts: tl.constexpr,
    topk: tl.constexpr,
    id_row_stride,
    id_slot_stride: tl.constexpr,
    columns: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * columns + tl.arange(0, columns)
    total = tl.full((columns,), 0, tl.float32)
    for k in tl.static_range(topk):
        expert = tl.load(ids + token * id_row_stride + k * id_slot_stride)
        row = tl.load(positions + token * topk + k).to(tl.int64)
        src = tl.where(expert < experts, dx0, dx1) + row * hidden + col
        total += tl.load(src, mask=(expert >= 0) & (col < hidden), other=0).to(tl.float32)
        if tl.program_id(1) == 0:
            v = tl.load(tl.where(expert < experts, dp0, dp1) + row, mask=expert >= 0, other=0)
            tl.store(dp + token * topk + k, v)
    tl.store(dx + token * hidden + col, total, mask=col < hidden)


@triton.jit
def _backward_pull_probs(
    ids,
    positions,
    dp0,
    dp1,
    dp,
    slots,
    experts: tl.constexpr,
    topk: tl.constexpr,
    id_row_stride,
    id_slot_stride: tl.constexpr,
    block: tl.constexpr,
):
    slot = tl.program_id(0) * block + tl.arange(0, block)
    token, k = (slot // topk).to(tl.int64), slot % topk
    expert = tl.load(ids + token * id_row_stride + k * id_slot_stride, mask=slot < slots, other=-1)
    row = tl.load(positions + slot, mask=slot < slots, other=0)
    grad = tl.load(tl.where(expert < experts, dp0, dp1) + row, mask=expert >= 0, other=0)
    tl.store(dp + slot, grad, mask=slot < slots)


def bulk_backward(dy, x, ids, probs, fc1, fc2, group, state):
    """Return token gradients on the source and expert weight gradients on the owner."""
    plan = _bulk_plan(x, ids, fc1, fc2, group, state, backward=True)
    if plan is None:
        return None
    sizes, positions, src_bases, offsets, counts = plan
    rank = group.rank()
    tokens, topk = ids.shape
    experts, hidden = len(fc1), x.size(1)
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    dp = torch.empty_like(probs, memory_format=torch.contiguous_format)
    if not max(sizes):
        return (
            dx.zero_(),
            dp.zero_(),
            [torch.zeros_like(w, memory_format=torch.contiguous_format) for w in fc1],
            [torch.zeros_like(w, memory_format=torch.contiguous_format) for w in fc2],
        )

    # The same row capacity on both ranks permits direct P2P writes. dx occupies
    # a separate slice because FC1 wgrad consumes the original x after FC1 dgrad.
    capacity = triton.cdiv(max(sizes), 1024) * 1024
    scalar_rows = triton.cdiv(capacity * 4, hidden * x.element_size())
    buffers = state.buffers(group, (3 * capacity + 2 * scalar_rows, hidden), x)
    ds = [b[2 * capacity : 3 * capacity] for b in buffers]
    scales = [
        b[3 * capacity : 3 * capacity + scalar_rows].view(torch.float32).view(-1) for b in buffers
    ]
    dscales = [b[3 * capacity + scalar_rows :].view(torch.float32).view(-1) for b in buffers]
    state.handle.barrier(channel=3)
    columns = min(2048, triton.next_power_of_2(hidden))
    grid = (tokens, triton.cdiv(hidden, columns))
    vector_route = x.dtype == torch.bfloat16 and topk == 2 and hidden % 256 == 0
    if tokens:
        ends = torch.cumsum(counts, 1, dtype=torch.int32)
        _backward_positions[(counts.size(1), 2 * experts)](
            ids,
            probs,
            ends,
            src_bases,
            positions,
            *scales,
            ids.numel(),
            experts,
            topk,
            *ids.stride(),
            *probs.stride(),
            256,
        )
        if vector_route and all(
            t.stride(1) == 1 and t.stride(0) % 256 == 0 and t.data_ptr() % 512 == 0 for t in (x, dy)
        ):
            tex.mega_moe_bulk_push(
                x, ids, positions, *(b[:capacity] for b in buffers), experts, rank
            )
            tex.mega_moe_bulk_push(
                dy, ids, positions, *(b[capacity : 2 * capacity] for b in buffers), experts, rank
            )
        else:
            _backward_push[grid](
                x,
                dy,
                ids,
                positions,
                *buffers,
                capacity,
                hidden,
                experts,
                topk,
                *x.stride(),
                *dy.stride(),
                *ids.stride(),
                columns,
            )
    state.handle.barrier(channel=0)
    rows = sizes[rank]
    block = buffers[rank]
    main = torch.cuda.current_stream(x.device)
    if state.grad_stream is None:
        state.grad_stream = torch.cuda.Stream(device=x.device)
        # Persistent weight-gradient CTAs use a full CU's shared memory.
        # Leave two CUs for the peer barrier and gradient reads on SM89.
        state.grad_ready = tex.MegaMoEGradReady(reserve_sms=2)
    _, _, dw1, dw2 = tex.mega_moe_local_backward(
        block[capacity : capacity + rows],
        block[:rows],
        scales[rank][:rows],
        offsets,
        fc1,
        fc2,
        ds[rank][:rows],
        dscales[rank][:rows],
        state.grad_ready,
    )
    # Each rank publishes its local route gradients once FC1 dgrad completes.
    # Its two weight-gradient GEMMs can still be running on the owner's stream.
    with torch.cuda.stream(state.grad_stream):
        state.grad_ready.wait()
        state.handle.barrier(channel=1)
        if tokens:
            if vector_route and dx.data_ptr() % 512 == 0:
                tex.mega_moe_bulk_pull(ids, positions, *ds, experts, rank, dx)
                _backward_pull_probs[(triton.cdiv(ids.numel(), 256),)](
                    ids, positions, *dscales, dp, ids.numel(), experts, topk, *ids.stride(), 256
                )
            else:
                _backward_pull[grid](
                    ids,
                    positions,
                    *ds,
                    *dscales,
                    dx,
                    dp,
                    hidden,
                    experts,
                    topk,
                    *ids.stride(),
                    columns,
                )
            ids.record_stream(state.grad_stream)
    # The last peer reader must retire before the next invocation can overwrite
    # its owner's dx/dp. Queue this barrier after the weight GEMMs: a spinning
    # return-stream barrier can otherwise wait a full GEMM for the other PPU
    # to admit its barrier CTA. The pull itself still overlaps both GEMMs.
    main.wait_stream(state.grad_stream)
    state.handle.barrier(channel=2)
    state.done.record(main)
    return dx, dp, dw1, dw2
