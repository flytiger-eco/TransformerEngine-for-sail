# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Build stable expert-major training rows from source slots on the training EP group."""

from dataclasses import dataclass
from weakref import finalize

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
import triton.language as tl

from ._mega_moe_peer import _workspace


_MAX_LOCAL_EXPERTS = 48


@triton.jit
def _count_blocks(
    ids, hist, slots, k, s0, s1, experts, WORLD: tl.constexpr, BINS: tl.constexpr, B: tl.constexpr
):
    block = tl.program_id(0)
    slot = block * B + tl.arange(0, B)
    value = tl.load(ids + slot // k * s0 + slot % k * s1, slot < slots, -1)
    valid = (value >= 0) & (value < experts * WORLD)
    bins = tl.arange(0, BINS)
    counts = tl.histogram(tl.where(valid, value, 0), BINS)
    counts -= tl.where(bins == 0, tl.sum((~valid).to(tl.int32), 0), 0)
    row = hist + block * (BINS + 2)
    tl.store(row + bins, counts)
    tl.store(row + BINS, tl.sum(((value != -1) & ~valid).to(tl.int32), 0))
    tl.store(row + (BINS + 1), tl.sum(((value == -1) & (slot < slots)).to(tl.int32), 0))


@triton.jit
def _scan_blocks(
    hist,
    prefix,
    header,
    blocks,
    experts,
    hidden,
    width,
    tokens,
    k,
    eligible,
    BINS: tl.constexpr,
    B: tl.constexpr,
):
    expert = tl.program_id(0)
    index = tl.arange(0, B)
    count = tl.load(hist + index * (BINS + 2) + expert, index < blocks, 0)
    if expert < BINS:
        tl.store(prefix + index * BINS + expert, tl.cumsum(count) - count, index < blocks)
    tl.store(header + expert, tl.sum(count, 0))
    if expert == 0:
        tl.store(header + (BINS + 2), experts)
        tl.store(header + (BINS + 3), hidden)
        tl.store(header + (BINS + 4), width)
        tl.store(header + (BINS + 5), tokens)
        tl.store(header + (BINS + 6), k)
        tl.store(header + (BINS + 7), eligible)


@triton.jit
def _bases(
    reports,
    bases,
    offsets,
    sizes,
    report,
    rank,
    E: tl.constexpr,
    NE: tl.constexpr,
    WORLD: tl.constexpr,
    BINS: tl.constexpr,
):
    owner = tl.program_id(0)
    expert = tl.arange(0, NE)
    source = tl.arange(0, WORLD)
    count = tl.load(
        reports + source[:, None] * (BINS + 8) + owner * E + expert[None, :],
        expert[None, :] < E,
        0,
    )
    total = tl.sum(count, 0)
    padded = tl.cdiv(total, 256) * 256
    begin = tl.cumsum(padded) - padded
    previous = tl.sum(tl.where(source[:, None] < rank, count, 0), 0)
    tl.store(bases + owner * E + expert, begin + previous, expert < E)
    tl.store(report + owner, tl.sum(padded, 0))
    if owner == rank:
        counts = reports + source * (BINS + 8)
        bad = tl.max(tl.load(counts + BINS), 0) != 0
        dropped = tl.load(counts + (BINS + 1))
        experts = tl.load(counts + (BINS + 2))
        hidden = tl.load(counts + (BINS + 3))
        width = tl.load(counts + (BINS + 4))
        tokens = tl.load(counts + (BINS + 5))
        topk = tl.load(counts + (BINS + 6))
        eligible = tl.load(counts + (BINS + 7))
        cap = tl.max(tokens, 0)
        mismatch = (tl.min(experts, 0) != tl.max(experts, 0)) | (
            tl.min(hidden, 0) != tl.max(hidden, 0)
        )
        fallback = (
            (tl.min(eligible, 0) == 0)
            | (tl.min(width, 0) != tl.max(width, 0))
            | (tl.min(topk, 0) != tl.max(topk, 0))
            | (cap < tl.where(tl.max(topk, 0) == 6, 1024, 2048))
            | (tl.sum(tokens.to(tl.int64) * topk, 0) + E * 256 >= 2147483648)
        )
        status = tl.where(bad, 2, tl.where(mismatch, 3, tl.where(fallback, 1, 0)))
        tl.store(report + WORLD, status)
        tl.store(report + WORLD + 1, cap)
        tl.store(report + WORLD + 2, tl.load(reports + rank * (BINS + 8) + (BINS + 1)))
        tl.store(report + WORLD + 3, tl.max(dropped, 0))
        tl.store(sizes + expert, total, expert < E)
        tl.store(offsets + expert, begin, expert < E)
        tl.store(offsets + E, tl.sum(padded, 0))


@triton.jit
def _peer(peers, rank):
    ptr = peers[0]
    for i in tl.static_range(1, len(peers)):
        ptr = tl.where(rank == i, peers[i], ptr)
    return ptr


@triton.jit
def _max_index(a, b):
    return tl.maximum(a, b)


@triton.jit
def _push(
    ids,
    probs,
    prefix,
    bases,
    peers,
    n,
    is0,
    is1,
    ps0,
    ps1,
    rank,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BINS: tl.constexpr,
    B: tl.constexpr,
):
    block = tl.program_id(0)
    lane = tl.arange(0, B)
    slot = block * B + lane
    exp = tl.load(ids + slot // TOPK * is0 + slot % TOPK * is1, slot < n, -1)
    # Sort only one CTA's slots. The prefix table preserves the order across CTAs.
    key = tl.sort(tl.where(exp >= 0, exp, BINS) * B + lane, descending=False)
    expert = key // B
    previous = tl.gather(expert, tl.maximum(lane - 1, 0), 0)
    first = tl.associative_scan(tl.where(expert != previous, lane, 0), 0, _max_index)
    valid = expert < E * len(peers)
    row = (
        tl.load(bases + expert, valid, 0)
        + tl.load(prefix + block * BINS + expert, valid, 0)
        + lane
        - first
    )
    source_slot = block * B + key % B
    prob = tl.load(probs + source_slot // TOPK * ps0 + source_slot % TOPK * ps1, valid, 0)
    target = _peer(peers, expert // E) + 2 * row.to(tl.int64)
    route = ((tl.full((1,), 0, tl.int64) + rank) << 32) | source_slot.to(tl.int64)
    tl.store(target, route, valid)
    tl.store(target + 1, prob.to(tl.int32, bitcast=True).to(tl.int64), valid)


@triton.jit
def _unpack(buf, offsets, sizes, route, prob, tiles, E: tl.constexpr, NE: tl.constexpr):
    tile = tl.program_id(0)
    row = tile * 256 + tl.arange(0, 256)
    index = tl.arange(0, NE)
    starts = tl.load(offsets + index, index < E, 2147483647)
    expert = tl.sum((starts <= tile * 256).to(tl.int32), 0) - 1
    valid = row < tl.load(offsets + expert) + tl.load(sizes + expert)
    tl.store(tiles + tile, expert)
    source = buf + 2 * row.to(tl.int64)
    tl.store(route + row, tl.load(source, valid, -1))
    bits = tl.load(source + 1, valid, 0).to(tl.int32)
    tl.store(prob + row, bits.to(tl.float32, bitcast=True))


@triton.jit
def _push_grad(
    route,
    dp,
    peers,
    n,
    B: tl.constexpr,
):
    row = tl.program_id(0) * B + tl.arange(0, B)
    entry = tl.load(route + row, row < n, -1)
    valid = entry >= 0
    peer = _peer(peers, (entry >> 32).to(tl.int32))
    slot = (entry & 0xFFFFFFFF).to(tl.int32)
    tl.store(peer + slot, tl.load(dp + row, valid, 0), valid)


class _Scratch:
    def __init__(self, group, device, count):
        # A rank that fails to allocate must still reach its peers before rendezvous.
        failed = torch.zeros(1, dtype=torch.int32, device=device)
        error = None
        try:
            self.storage = symm.empty(count, device=device, dtype=torch.int64)
        except RuntimeError as exc:
            error = exc
            failed.fill_(2 if isinstance(exc, torch.OutOfMemoryError) else 1)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
        status = failed.item()
        if status:
            if error is not None:
                raise error
            if status == 2:
                raise torch.OutOfMemoryError("A peer could not allocate the MegaMoE route buffer")
            raise RuntimeError("A peer could not allocate the MegaMoE route buffer")
        self.handle = symm.rendezvous(self.storage, group)
        self.peers = [
            self.handle.get_buffer(rank, self.storage.shape, self.storage.dtype)
            for rank in range(group.size())
        ]
        self.grad_peers = [p.view(torch.float32) for p in self.peers]
        self.done = torch.cuda.Event()


def _scratch(group, device, count):
    state = _workspace(group)
    scratch = getattr(state, "training_routes", None)
    if scratch is None or scratch.storage.numel() < count:
        if scratch is not None:
            scratch.done.synchronize()
        scratch = _Scratch(group, device, triton.next_power_of_2(count))
        if not hasattr(state, "training_routes"):
            finalize(group, _finish, state).atexit = False
        state.training_routes = scratch
    torch.cuda.current_stream(device).wait_event(scratch.done)
    return scratch


def _finish(state):
    state.training_routes.done.synchronize()


@dataclass
class TrainingRoute:
    group: dist.ProcessGroup
    cap: int
    tokens: int
    topk: int
    dropped: bool
    any_dropped: bool


def stage_source(local, source, cap, topk, dropped):
    tokens, hidden = source.shape
    local[:tokens].copy_(source)
    slots = local[cap : cap + tokens * topk].view(tokens, topk, hidden)
    if dropped:
        slots.zero_()
    return slots


def prepare(x, ids, probs, experts, width, aligned, group):
    """Agree on dimensions, then publish each (slot, probability) in owner-row order."""
    world = group.size()
    bins = triton.next_power_of_2(world * _MAX_LOCAL_EXPERTS)
    tokens, k = ids.shape
    hidden = x.size(1)
    slots = ids.numel()
    eligible = (
        k in (6, 8)
        and 0 < width <= 16384
        and width % 256 == 0
        and hidden > 0
        and hidden % 256 == 0
        and 0 < experts <= _MAX_LOCAL_EXPERTS
        and aligned
        and slots + experts * 256 < 2**31
    )
    kwargs = dict(device=x.device, dtype=torch.int32)
    blocks = max(1, triton.cdiv(slots, 256)) if eligible else 1
    hist = torch.empty((blocks, bins + 2), **kwargs)
    prefix = torch.empty((blocks, bins), **kwargs)
    header = torch.empty(bins + 8, **kwargs)
    _count_blocks[(blocks,)](
        ids,
        hist,
        slots if eligible else 0,
        k,
        *ids.stride(),
        experts,
        world,
        bins,
        256,
    )
    scan_blocks = blocks if blocks <= 4096 else 1
    _scan_blocks[(bins + 2,)](
        hist,
        prefix,
        header,
        scan_blocks,
        experts,
        hidden,
        width,
        tokens,
        k,
        int(eligible),
        bins,
        triton.next_power_of_2(scan_blocks),
    )
    if blocks > scan_blocks:
        cumulative = torch.cumsum(hist, dim=0, dtype=torch.int32)
        torch.sub(cumulative[:, :bins], hist[:, :bins], out=prefix)
        header[: bins + 2].copy_(cumulative[-1])
    reports = torch.empty((world, bins + 8), **kwargs)
    dist.all_gather_into_tensor(reports.view(-1), header, group=group)
    rank = group.rank()
    # Read only the fixed-size summary. Bases and segment sizes stay on the device.
    e = min(experts, _MAX_LOCAL_EXPERTS)
    bases = torch.empty(world * e, **kwargs)
    offsets = torch.empty(e + 1, **kwargs)
    sizes = torch.empty(e, **kwargs)
    summary = torch.empty(world + 4, **kwargs)
    _bases[(world,)](
        reports, bases, offsets, sizes, summary, rank, e, triton.next_power_of_2(e), world, bins
    )
    host = summary.cpu().tolist()
    status, cap, dropped, any_dropped = host[world:]
    if status == 2:
        raise ValueError("Expected expert IDs in range or -1")
    if status == 3:
        raise ValueError("Expert-parallel input width and expert count must match on every rank")
    if status:
        return None
    rows = host[rank]
    # Storage is shared across steps. The results saved for backward have independent lifetimes.
    state = _workspace(group)
    with state.lock:
        scratch = _scratch(group, x.device, max(2 * max(host[:world]), (cap * k + 1) // 2))
        peers = state.buffers(group, ((1 + k) * cap, hidden), x)
        # Both workspace completion events are queued before this cross-rank barrier.
        # It protects the previous generation's local reads of routes and output slots.
        scratch.handle.barrier(channel=0)
        output_slots = stage_source(peers[rank], x, cap, k, bool(dropped))
        if slots:
            _push[(blocks,)](
                ids,
                probs,
                prefix,
                bases,
                tuple(scratch.peers),
                slots,
                *ids.stride(),
                *probs.stride(),
                rank,
                experts,
                k,
                bins,
                256,
            )
        scratch.handle.barrier(channel=1)
        route = torch.empty(rows, dtype=torch.int64, device=x.device)
        packed_prob = torch.empty(rows, dtype=torch.float32, device=x.device)
        tiles = torch.empty(rows // 256, **kwargs)
        if rows:
            _unpack[(rows // 256,)](
                scratch.peers[rank],
                offsets,
                sizes,
                route,
                packed_prob,
                tiles,
                experts,
                triton.next_power_of_2(experts),
            )
        scratch.done.record(torch.cuda.current_stream(x.device))
    plan = TrainingRoute(group, cap, tokens, k, bool(dropped), bool(any_dropped))
    return plan, route, tiles, offsets, packed_prob, (peers, output_slots)


def return_grad(dp, route, plan):
    """Write the owner gradient to its unique source slot, leaving skipped slots at zero."""
    state = _workspace(plan.group)
    with state.lock:
        scratch = _scratch(plan.group, dp.device, (plan.cap * plan.topk + 1) // 2)
        rank = plan.group.rank()
        values = scratch.grad_peers[rank][: plan.tokens * plan.topk].view(plan.tokens, plan.topk)
        scratch.handle.barrier(channel=0)
        if plan.any_dropped:
            if plan.dropped:
                values.zero_()
            scratch.handle.barrier(channel=1)
        if route.numel():
            _push_grad[(triton.cdiv(route.numel(), 256),)](
                route, dp, tuple(scratch.grad_peers), route.numel(), 256
            )
        scratch.handle.barrier(channel=2)
        result = values.clone()
        scratch.done.record(torch.cuda.current_stream(dp.device))
    return result
