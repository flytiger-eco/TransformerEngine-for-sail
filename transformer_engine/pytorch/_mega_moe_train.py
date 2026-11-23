# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Saved owner-local activations for eager BF16 EP8/top-8 training."""
import torch
import triton
import triton.language as tl
import transformer_engine_torch as tex

from ._mega_moe_peer import _workspace
from ._mega_moe_routing import make_route_plan, return_routes


@triton.jit
def _activation(z, prob, raw, y, I: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    j = tl.program_id(1) * B + tl.arange(0, B)
    base = r.to(tl.int64) * I
    g = tl.load(z + 2 * base + j, j < I, 0).to(tl.float32)
    u = tl.load(z + 2 * base + I + j, j < I, 0).to(tl.float32)
    h = g * tl.sigmoid(g) * u
    tl.store(raw + base + j, h, j < I)
    tl.store(y + base + j, h * tl.load(prob + r), j < I)


@triton.jit
def _activation_grad(dy, z, prob, raw, dz, dp, I: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, B)
    base = r.to(tl.int64) * I
    g = tl.load(z + 2 * base + j, j < I, 0).to(tl.float32)
    u = tl.load(z + 2 * base + I + j, j < I, 0).to(tl.float32)
    grad = tl.load(dy + base + j, j < I, 0).to(tl.float32)
    s = tl.sigmoid(g)
    scaled = grad * tl.load(prob + r)
    tl.store(dz + 2 * base + j, scaled * u * s * (1.0 + g * (1.0 - s)), j < I)
    tl.store(dz + 2 * base + I + j, scaled * (g * s), j < I)
    tl.store(dp + r, tl.sum(grad * tl.load(raw + base + j, j < I, 0), 0))


def _owner_plan(x, probs, experts, plan, local_id):
    # Exchange the expert, source slot and exact FP32 probability bits together.
    slot = plan.slots | (plan.group.rank() << 32)
    prob_bits = probs.reshape(-1).index_select(0, plan.slots).view(torch.int32)
    metadata = plan.to_experts(torch.stack((local_id, slot, prob_bits), dim=1))
    owner_ids, owner_slot, owner_prob_bits = metadata.unbind(1)
    counts = torch.bincount(owner_ids, minlength=experts)
    padded = (counts + 255) // 256 * 256
    rows = int(padded.sum())
    offsets = torch.cat((padded.new_zeros(1), padded.cumsum(0))).int()
    order = torch.argsort(owner_ids, stable=True)
    starts = (padded - counts).cumsum(0) - (padded - counts)
    places = torch.empty_like(order)
    places[order] = torch.arange(order.numel(), device=x.device) + torch.repeat_interleave(
        starts, counts, output_size=order.numel()
    )
    route = torch.full((rows,), -1, device=x.device, dtype=torch.long)
    route[places] = owner_slot
    packed_prob = probs.new_zeros((rows,))
    packed_prob[places] = owner_prob_bits.int().view(torch.float32)
    tiles = torch.repeat_interleave(
        torch.arange(experts, device=x.device, dtype=torch.int32),
        padded // 256,
        output_size=rows // 256,
    )
    return route, tiles, offsets, places, packed_prob


def _stage(state, group, source, cap):
    tokens, hidden = source.shape
    peers = state.buffers(group, (9 * cap, hidden), source)
    # Other ranks can still be combining the previous invocation's slots.
    state.handle.barrier(channel=0)
    local = peers[group.rank()]
    local[:tokens].copy_(source)
    slots = local[cap : cap + tokens * 8].view(tokens, 8, hidden)
    slots.zero_()
    state.handle.barrier(channel=1)
    return peers, slots


class _Train(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probs, plan, local_id, cap, *weights):
        group = plan.group
        e = len(weights) // 2
        w1, w2 = weights[:e], weights[e:]
        route, tiles, offsets, places, prob = _owner_plan(x, probs, e, plan, local_id)
        state = _workspace(group)
        rows, width = route.numel(), w2[0].size(1)
        packed = x.new_empty((rows, x.size(1)))
        with state.lock:
            peers, slots = _stage(state, group, x, cap)
            args = (route, tiles, peers, group.rank(), cap)
            z = tex._mega_moe_train_linear(packed, w1, *args, True, True)
            y = x.new_empty((rows, width))
            raw = prob.new_empty((rows, width))
            if rows:
                _activation[(rows, triton.cdiv(width, 256))](z, prob, raw, y, width, 256)
            tex._mega_moe_train_linear(y, w2, *args, False, True)
            state.handle.barrier(channel=2)
            out = tex.mega_moe_combine(slots)
            state.done.record(torch.cuda.current_stream(x.device))
        ctx.save_for_backward(packed, z, raw, y, prob, route, tiles, offsets, places, *weights)
        ctx.plan, ctx.cap, ctx.experts = plan, cap, e
        return out

    @staticmethod
    def backward(ctx, dy):
        x, z, raw, y, prob, route, tiles, offsets, places, *weights = ctx.saved_tensors
        plan, cap, e = ctx.plan, ctx.cap, ctx.experts
        group = plan.group
        w1, w2 = weights[:e], weights[e:]
        state = _workspace(group)
        routed_dy = torch.empty_like(x)
        with state.lock:
            peers, slots = _stage(state, group, dy, cap)
            args = (route, tiles, peers, group.rank(), cap)
            dh = tex._mega_moe_train_linear(routed_dy, w2, *args, True, False)
            dz, dp = torch.empty_like(z), torch.empty_like(prob)
            if x.size(0):
                width = y.size(1)
                _activation_grad[(x.size(0),)](
                    dh, z, prob, raw, dz, dp, width, triton.next_power_of_2(width)
                )
            tex._mega_moe_train_linear(dz, w1, *args, False, False)
            state.handle.barrier(channel=2)
            dx = tex.mega_moe_combine(slots)
            state.done.record(torch.cuda.current_stream(x.device))
        dw1 = [torch.empty_like(w) for w in w1]
        dw2 = [torch.empty_like(w) for w in w2]
        tex.mega_moe_wgrad(routed_dy, y, offsets, dw2, False)
        tex.mega_moe_wgrad(dz, x, offsets, dw1, False)
        return dx, return_routes(dp[places], plan), None, None, None, *dw1, *dw2


def training_forward(x, ids, probs, w1, w2, group):
    i = w2[0].size(1)
    aligned = all(w.is_contiguous() and w.data_ptr() % 16 == 0 for w in (*w1, *w2))
    fields = (ids.size(1), i, int(aligned), x.size(0))
    plan, local_id, reports = make_route_plan(x, ids, len(w1), group, fields)
    cap = max(r[3] for r in reports)
    if (
        ids.size(1) != 8
        or not 0 < i <= 16384
        or i % 256
        or x.size(1) % 256
        or not x.size(1)
        or len(w1) > 16
        or not aligned
        or any(tuple(r[:3]) != fields[:3] for r in reports)
        or cap < 2048
        or sum(r[3] for r in reports) * 8 + 4096 >= 2**31
    ):
        return None
    return _Train.apply(x, probs, plan, local_id, cap, *w1, *w2)
