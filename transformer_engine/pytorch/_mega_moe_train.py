# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Saved owner-local activations for eager BF16 peer-accessible EP training."""
import torch
import triton
import triton.language as tl
import transformer_engine_torch as tex

from ._mega_moe_peer import _workspace
from ._mega_moe_train_route import prepare, return_grad, stage_source


@triton.jit
def _activation(z, prob, raw, y, I: tl.constexpr, B: tl.constexpr):
    r = tl.program_id(0)
    j = tl.program_id(1) * B + tl.arange(0, B)
    base = r.to(tl.int64) * I
    g = tl.load(z + 2 * base + j, j < I, 0).to(tl.float32)
    u = tl.load(z + 2 * base + I + j, j < I, 0).to(tl.float32)
    h = g * tl.sigmoid(g) * u
    if raw is not None:
        tl.store(raw + base + j, h, j < I)
    tl.store(y + base + j, h * tl.load(prob + r), j < I)


def _save_activation(z, prob, raw, y):
    rows, width = y.shape
    if rows:
        block = 1024 if width == 3072 else 256
        _activation[(rows, triton.cdiv(width, block))](z, prob, raw, y, width, block)


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
    h = g * s * u if raw is None else tl.load(raw + base + j, j < I, 0)
    tl.store(dp + r, tl.sum(grad * h, 0))


def _stage(state, group, source, cap, topk, dropped):
    peers = state.buffers(group, ((1 + topk) * cap, source.size(1)), source)
    # Other ranks can still be combining the previous invocation's slots.
    state.handle.barrier(channel=0)
    slots = stage_source(peers[group.rank()], source, cap, topk, dropped)
    state.handle.barrier(channel=1)
    return peers, slots


class _Train(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probs, plan, route, tiles, offsets, prob, staged, reduction_rows, *weights):
        cap = plan.cap
        group = plan.group
        e = len(weights) // 2
        w1, w2 = weights[:e], weights[e:]
        state = _workspace(group)
        rows, width = route.numel(), w2[0].size(1)
        packed = x.new_empty((rows, x.size(1)))
        with state.lock:
            peers, slots = staged
            args = (route, tiles, peers, group.rank(), cap)
            z = tex._mega_moe_train_linear(packed, w1, *args, True, True, topk=plan.topk)
            y = x.new_empty((rows, width))
            _save_activation(z, prob, None, y)
            tex._mega_moe_train_linear(y, w2, *args, False, True, True, topk=plan.topk)
            state.handle.barrier(channel=2)
            out = tex.mega_moe_combine(slots)
        ctx.save_for_backward(packed, z, y, prob, route, tiles, offsets, reduction_rows, *weights)
        ctx.plan, ctx.cap, ctx.experts = plan, cap, e
        return out

    @staticmethod
    def backward(ctx, dy):
        x, z, y, prob, route, tiles, offsets, reduction_rows, *weights = ctx.saved_tensors
        plan, cap, e = ctx.plan, ctx.cap, ctx.experts
        group = plan.group
        w1, w2 = weights[:e], weights[e:]
        state = _workspace(group)
        routed_dy = torch.empty_like(x)
        with state.lock:
            peers, slots = _stage(state, group, dy, cap, plan.topk, plan.dropped)
            args = (route, tiles, peers, group.rank(), cap)
            dh = tex._mega_moe_train_linear(routed_dy, w2, *args, True, False, topk=plan.topk)
            dz, dp = torch.empty_like(z), torch.empty_like(prob)
            if x.size(0):
                width = y.size(1)
                _activation_grad[(x.size(0),)](
                    dh, z, prob, None, dz, dp, width, triton.next_power_of_2(width)
                )
            tex._mega_moe_train_linear(dz, w1, *args, False, False, True, topk=plan.topk)
            state.handle.barrier(channel=2)
            dx = tex.mega_moe_combine(slots)
            state.done.record(torch.cuda.current_stream(x.device))
        dw1 = [torch.empty_like(w) for w in w1]
        dw2 = [torch.empty_like(w) for w in w2]
        tex.mega_moe_wgrad(routed_dy, y, offsets, dw2, False, reduction_rows)
        tex.mega_moe_wgrad(dz, x, offsets, dw1, False, reduction_rows)
        return (
            dx,
            return_grad(dp, route, plan),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *dw1,
            *dw2,
        )


def training_forward(x, ids, probs, w1, w2, group):
    i = w2[0].size(1)
    aligned = all(w.is_contiguous() and w.data_ptr() % 16 == 0 for w in (*w1, *w2))
    state = _workspace(group)
    # Keep the payload exclusively staged until forward records its completion event.
    with state.lock:
        with torch.no_grad():
            result = prepare(x, ids, probs, len(w1), i, aligned, group)
        if result is None:
            return None
        try:
            return _Train.apply(x, probs, *result, *w1, *w2)
        finally:
            state.done.record(torch.cuda.current_stream(x.device))
