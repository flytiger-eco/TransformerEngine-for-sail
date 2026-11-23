# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Build stable expert-major training rows from source slots on eight peers."""

from dataclasses import dataclass
from weakref import finalize

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
import triton.language as tl

from ._mega_moe_peer import _workspace


@triton.jit
def _count_blocks(ids, hist, slots, k, s0, s1, experts, B: tl.constexpr):
    block = tl.program_id(0)
    slot = block * B + tl.arange(0, B)
    value = tl.load(ids + slot // k * s0 + slot % k * s1, slot < slots, -1)
    valid = (value >= 0) & (value < experts * 8)
    bins = tl.arange(0, 128)
    counts = tl.histogram(tl.where(valid, value, 0), 128)
    counts -= tl.where(bins == 0, tl.sum((~valid).to(tl.int32), 0), 0)
    row = hist + block * 130
    tl.store(row + tl.arange(0, 128), counts)
    tl.store(row + 128, tl.sum(((value != -1) & ~valid).to(tl.int32), 0))
    tl.store(row + 129, tl.sum(((value == -1) & (slot < slots)).to(tl.int32), 0))


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
    B: tl.constexpr,
):
    expert = tl.program_id(0)
    index = tl.arange(0, B)
    count = tl.load(hist + index * 130 + expert, index < blocks, 0)
    if expert < 128:
        tl.store(prefix + index * 128 + expert, tl.cumsum(count) - count, index < blocks)
    tl.store(header + expert, tl.sum(count, 0))
    if expert == 0:
        tl.store(header + 130, experts)
        tl.store(header + 131, hidden)
        tl.store(header + 132, width)
        tl.store(header + 133, tokens)
        tl.store(header + 134, k)
        tl.store(header + 135, eligible)


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
):
    owner = tl.program_id(0)
    expert = tl.arange(0, NE)
    source = tl.arange(0, 8)
    count = tl.load(
        reports + source[:, None] * 136 + owner * E + expert[None, :],
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
        counts = reports + source * 136
        bad = tl.max(tl.load(counts + 128), 0) != 0
        dropped = tl.load(counts + 129)
        experts = tl.load(counts + 130)
        hidden = tl.load(counts + 131)
        width = tl.load(counts + 132)
        tokens = tl.load(counts + 133)
        eligible = tl.load(counts + 135)
        cap = tl.max(tokens, 0)
        mismatch = (tl.min(experts, 0) != tl.max(experts, 0)) | (
            tl.min(hidden, 0) != tl.max(hidden, 0)
        )
        fallback = (
            (tl.min(eligible, 0) == 0)
            | (tl.min(width, 0) != tl.max(width, 0))
            | (cap < 2048)
            | (tl.sum(tokens.to(tl.int64), 0) * 8 + 4096 >= 2147483648)
        )
        status = tl.where(bad, 2, tl.where(mismatch, 3, tl.where(fallback, 1, 0)))
        tl.store(report + 8, status)
        tl.store(report + 9, cap)
        tl.store(report + 10, tl.load(reports + rank * 136 + 129))
        tl.store(report + 11, tl.max(dropped, 0))
        tl.store(sizes + expert, total, expert < E)
        tl.store(offsets + expert, begin, expert < E)
        tl.store(offsets + E, tl.sum(padded, 0))


@triton.jit
def _peer(P0, P1, P2, P3, P4, P5, P6, P7, rank):
    p01 = tl.where(rank == 0, P0, P1)
    p23 = tl.where(rank == 2, P2, P3)
    p45 = tl.where(rank == 4, P4, P5)
    p67 = tl.where(rank == 6, P6, P7)
    return tl.where(rank < 4, tl.where(rank < 2, p01, p23), tl.where(rank < 6, p45, p67))


@triton.jit
def _max_index(a, b):
    return tl.maximum(a, b)


@triton.jit
def _push(
    ids,
    probs,
    prefix,
    bases,
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    n,
    is0,
    is1,
    ps0,
    ps1,
    rank,
    E: tl.constexpr,
    B: tl.constexpr,
):
    block = tl.program_id(0)
    lane = tl.arange(0, B)
    slot = block * B + lane
    exp = tl.load(ids + slot // 8 * is0 + slot % 8 * is1, slot < n, -1)
    # Sort only one CTA's slots. The prefix table preserves the order across CTAs.
    key = tl.sort(tl.where(exp >= 0, exp, 128) * B + lane, descending=False)
    expert = key // B
    previous = tl.gather(expert, tl.maximum(lane - 1, 0), 0)
    first = tl.associative_scan(tl.where(expert != previous, lane, 0), 0, _max_index)
    valid = expert < E * 8
    row = (
        tl.load(bases + expert, valid, 0)
        + tl.load(prefix + block * 128 + expert, valid, 0)
        + lane
        - first
    )
    source_slot = block * B + key % B
    prob = tl.load(probs + source_slot // 8 * ps0 + source_slot % 8 * ps1, valid, 0)
    target = _peer(P0, P1, P2, P3, P4, P5, P6, P7, expert // E) + 2 * row.to(tl.int64)
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
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    n,
    B: tl.constexpr,
):
    row = tl.program_id(0) * B + tl.arange(0, B)
    entry = tl.load(route + row, row < n, -1)
    valid = entry >= 0
    peer = _peer(P0, P1, P2, P3, P4, P5, P6, P7, (entry >> 32).to(tl.int32))
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
    dropped: bool
    any_dropped: bool


def stage_source(local, source, cap, dropped):
    tokens, hidden = source.shape
    local[:tokens].copy_(source)
    slots = local[cap : cap + tokens * 8].view(tokens, 8, hidden)
    if dropped:
        slots.zero_()
    return slots


def prepare(x, ids, probs, experts, width, aligned, group):
    """Agree on dimensions, then publish each (slot, probability) in owner-row order."""
    tokens, k = ids.shape
    hidden = x.size(1)
    slots = ids.numel()
    eligible = (
        k == 8
        and 0 < width <= 16384
        and width % 256 == 0
        and hidden > 0
        and hidden % 256 == 0
        and 0 < experts <= 16
        and aligned
        and slots + 4096 < 2**31
    )
    kwargs = dict(device=x.device, dtype=torch.int32)
    blocks = max(1, triton.cdiv(slots, 256)) if eligible else 1
    hist = torch.empty((blocks, 130), **kwargs)
    prefix = torch.empty((blocks, 128), **kwargs)
    header = torch.empty(136, **kwargs)
    _count_blocks[(blocks,)](
        ids,
        hist,
        slots if eligible else 0,
        k,
        *ids.stride(),
        experts,
        256,
    )
    scan_blocks = blocks if blocks <= 4096 else 1
    _scan_blocks[(130,)](
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
        triton.next_power_of_2(scan_blocks),
    )
    if blocks > scan_blocks:
        cumulative = torch.cumsum(hist, dim=0, dtype=torch.int32)
        torch.sub(cumulative[:, :128], hist[:, :128], out=prefix)
        header[:130].copy_(cumulative[-1])
    reports = torch.empty((8, 136), **kwargs)
    dist.all_gather_into_tensor(reports.view(-1), header, group=group)
    rank = group.rank()
    # Read only the fixed-size summary. Bases and segment sizes stay on the device.
    e = min(experts, 16)
    bases = torch.empty(8 * e, **kwargs)
    offsets = torch.empty(e + 1, **kwargs)
    sizes = torch.empty(e, **kwargs)
    summary = torch.empty(12, **kwargs)
    _bases[(8,)](reports, bases, offsets, sizes, summary, rank, e, triton.next_power_of_2(e))
    host = summary.cpu().tolist()
    status, cap, dropped, any_dropped = host[8:]
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
        scratch = _scratch(group, x.device, max(2 * max(host[:8]), cap * 4))
        peers = state.buffers(group, (9 * cap, hidden), x)
        # Both workspace completion events are queued before this cross-rank barrier.
        # It protects the previous generation's local reads of routes and output slots.
        scratch.handle.barrier(channel=0)
        output_slots = stage_source(peers[rank], x, cap, bool(dropped))
        if slots:
            _push[(blocks,)](
                ids,
                probs,
                prefix,
                bases,
                *scratch.peers,
                slots,
                *ids.stride(),
                *probs.stride(),
                rank,
                experts,
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
    plan = TrainingRoute(group, cap, tokens, bool(dropped), bool(any_dropped))
    return plan, route, tiles, offsets, packed_prob, (peers, output_slots)


def return_grad(dp, route, plan):
    """Write the owner gradient to its unique source slot, leaving skipped slots at zero."""
    state = _workspace(plan.group)
    with state.lock:
        scratch = _scratch(plan.group, dp.device, plan.cap * 4)
        rank = plan.group.rank()
        values = scratch.grad_peers[rank][: plan.tokens * 8].view(plan.tokens, 8)
        scratch.handle.barrier(channel=0)
        if plan.any_dropped:
            if plan.dropped:
                values.zero_()
            scratch.handle.barrier(channel=1)
        if route.numel():
            _push_grad[(triton.cdiv(route.numel(), 256),)](
                route, dp, *scratch.grad_peers, route.numel(), 256
            )
        scratch.handle.barrier(channel=2)
        result = values.clone()
        scratch.done.record(torch.cuda.current_stream(dp.device))
    return result
