# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Time one balanced EP8 training step at H=3584, I=3072 and 2048 rows per expert.

Each SAMPLES row gives forward, backward and total milliseconds, taking the
maximum across ranks for each interval. The router network and optimizer are
outside the measured region.
"""

import json
import argparse
import os
import statistics

from benchmark_mega_moe_ep8 import Baseline, routes
from benchmark_mega_moe_deepep import DeepEPLegacyForward, profile_kernels

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch._mega_moe_train import _activation_grad, _save_activation
from transformer_engine.pytorch.mega_moe import _bf16_forward
from deep_ep import topk_idx_t


@triton.jit
def _merge(part, reverse, dst, H: tl.constexpr, B: tl.constexpr, K: tl.constexpr = 8):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for k in tl.static_range(K):
        src = tl.load(reverse + row * K + k)
        acc += tl.load(part + src.to(tl.int64) * H + col, (src >= 0) & (col < H), 0).to(tl.float32)
    tl.store(dst + row.to(tl.int64) * H + col, acc, col < H)


class DeepTraining(torch.autograd.Function):
    """DeepEP dispatch with cached backward routing and expert-local grouped GEMMs."""

    @staticmethod
    def forward(ctx, x, ids, probs, base, *weights):
        e, k = base.experts, ids.size(1)
        w1, w2 = weights[:e], weights[e:]
        idx = ids.to(topk_idx_t)
        experts = dist.get_world_size() * e
        legacy = isinstance(base, DeepEPLegacyForward)
        if legacy:
            per_rank, per_rdma, per_expert, in_rank, _ = base.buffer.get_dispatch_layout(
                idx, experts
            )
            rx, ri, rp, counts, handle, _ = base.buffer.dispatch(
                x,
                topk_idx=idx,
                topk_weights=probs,
                num_tokens_per_rank=per_rank,
                num_tokens_per_rdma_rank=per_rdma,
                num_tokens_per_expert=per_expert,
                is_token_in_rank=in_rank,
            )
        else:
            rx, ri, rp, handle, _ = base.buffer.dispatch(
                x,
                topk_idx=idx,
                topk_weights=probs,
                num_experts=experts,
                num_sms=base.num_sms,
                expert_alignment=1,
                do_cpu_sync=True,
                do_expand=False,
                do_handle_copy=False,
            )
            counts = handle.num_recv_tokens_per_expert_list
        # Full training uses the deduplicated receive layout so the cached handle
        # can return both input gradients and FP32 router-weight gradients.
        flat = ri.flatten()
        slot = (flat >= 0).nonzero(as_tuple=True)[0]
        slot = slot[flat[slot].argsort(stable=True)]
        rows = slot // k
        reverse = torch.full_like(flat, -1, dtype=torch.int32)
        reverse[slot] = torch.arange(slot.numel(), device=x.device, dtype=torch.int32)
        owner_x = rx[rows]
        p = rp.flatten()[slot]
        z = base.linear(owner_x, w1, counts)
        y = x.new_empty((len(rows), w2[0].size(1)))
        raw = p.new_empty(y.shape)
        _save_activation(z, p, raw, y)
        partial = base.linear(y, w2, counts)
        reduced = torch.empty_like(rx)
        if len(rx):
            _merge[(len(rx), triton.cdiv(x.size(1), 256))](
                partial, reverse, reduced, x.size(1), 256, k
            )
        out, _, _ = base.buffer.combine(reduced, handle)
        ctx.save_for_backward(owner_x, z, raw, y, p, rows, slot, reverse, *weights)
        ctx.base, ctx.handle = base, handle
        ctx.k, ctx.counts = k, counts
        return out

    @staticmethod
    def backward(ctx, dy):
        x, z, raw, y, p, rows, slot, reverse, *weights = ctx.saved_tensors
        base, handle = ctx.base, ctx.handle
        w1, w2 = weights[: base.experts], weights[base.experts :]
        if isinstance(base, DeepEPLegacyForward):
            recv, *_ = base.buffer.dispatch(dy.contiguous(), handle=handle)
            offsets = x.new_tensor([0, *ctx.counts], dtype=torch.int32).cumsum(0, dtype=torch.int32)
        else:
            recv, *_ = base.buffer.dispatch(dy.contiguous(), handle=handle, num_sms=base.num_sms)
            offsets = torch.nn.functional.pad(handle.psum_num_recv_tokens_per_expert, (1, 0)).int()
        owner_dy = recv[rows]
        dh = tex.mega_moe_dgrad(owner_dy, offsets, w2)
        dz, dp = torch.empty_like(z), torch.empty_like(p)
        if len(rows):
            _activation_grad[(len(rows),)](
                dh, z, p, raw, dz, dp, y.size(1), triton.next_power_of_2(y.size(1))
            )
        partial = tex.mega_moe_dgrad(dz, offsets, w1)
        reduced = torch.empty_like(recv)
        if len(recv):
            _merge[(len(recv), triton.cdiv(dy.size(1), 256))](
                partial, reverse, reduced, dy.size(1), 256, ctx.k
            )
        rdp = dp.new_zeros((len(recv), ctx.k))
        rdp.flatten()[slot] = dp
        dx, dp_src, _ = base.buffer.combine(reduced, handle, topk_weights=rdp)
        dw1 = [torch.empty_like(w) for w in w1]
        dw2 = [torch.empty_like(w) for w in w2]
        tex.mega_moe_wgrad(owner_dy, y, offsets, dw2, False)
        tex.mega_moe_wgrad(dz, x, offsets, dw1, False)
        return dx, None, dp_src, None, *dw1, *dw2


@triton.jit
def _locate_dy(metadata, source_to_recv, rows, STRIDE: tl.constexpr, B: tl.constexpr):
    recv = tl.program_id(0) * B + tl.arange(0, B)
    source = tl.load(metadata + recv * STRIDE, recv < rows, 0)
    tl.store(source_to_recv + source, recv, recv < rows)


@triton.jit
def _map_slots(meta, source_to_recv, reverse, gather, rows, K: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    recv_forward, k = idx // K, idx % K
    valid = recv_forward < rows
    source = tl.load(meta + recv_forward * (2 + K), valid, 0)
    owner = tl.load(meta + recv_forward * (2 + K) + 2 + k, valid, -1)
    recv_backward = tl.load(source_to_recv + source, valid, 0)
    tl.store(reverse + recv_backward * K + k, owner, valid)
    tl.store(gather + owner, recv_backward, valid & (owner >= 0))


@triton.jit
def _pack_dp(dp, reverse, rdp, size, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    owner = tl.load(reverse + idx, idx < size, -1)
    value = tl.load(dp + owner, (idx < size) & (owner >= 0), 0)
    tl.store(rdp + idx, value, idx < size)


class DeepExpandedTraining(torch.autograd.Function):
    """Direct expert-major dispatch and combine with saved activations for backward."""

    @staticmethod
    def forward(ctx, x, ids, probs, base, *weights):
        e = base.experts
        w1, w2 = weights[:e], weights[e:]
        owner_x, _, p, handle, _ = base.buffer.dispatch(
            x,
            topk_idx=ids.to(topk_idx_t),
            topk_weights=probs,
            num_experts=base.buffer.num_ranks * e,
            num_sms=base.num_sms,
            expert_alignment=1,
            do_cpu_sync=True,
            do_expand=True,
            do_handle_copy=False,
        )
        counts = handle.num_recv_tokens_per_expert_list
        z = base.linear(owner_x, w1, counts)
        rows, width = owner_x.size(0), w2[0].size(1)
        y = x.new_empty((rows, width))
        raw = p.new_empty(y.shape)
        _save_activation(z, p, raw, y)
        partial = base.linear(y, w2, counts)
        out, _, _ = base.buffer.combine(partial, handle)
        ctx.save_for_backward(owner_x, z, raw, y, p, *weights)
        ctx.base, ctx.handle = base, handle
        return out

    @staticmethod
    def backward(ctx, dy):
        x, z, raw, y, p, *weights = ctx.saved_tensors
        base, forward = ctx.base, ctx.handle
        e = base.experts
        w1, w2 = weights[:e], weights[e:]
        # Expanded combine accepts BF16 expert outputs only. A fresh deduplicated
        # handle can return dx and the separate FP32 gradient for every top-k slot.
        meta = forward.recv_src_metadata
        k = forward.topk_idx.size(1)
        gather = meta.new_empty(x.size(0))
        recv, _, _, handle, _ = base.buffer.dispatch(
            dy.contiguous(),
            topk_idx=forward.topk_idx,
            num_experts=forward.num_experts,
            num_sms=base.num_sms,
            expert_alignment=1,
            do_cpu_sync=True,
            do_expand=False,
            do_handle_copy=False,
        )
        recv_count = recv.size(0)
        source_to_recv = meta.new_empty(base.buffer.num_ranks * handle.num_max_tokens_per_rank)
        reverse = meta.new_empty((recv_count, k))
        if recv_count:
            _locate_dy[(triton.cdiv(recv_count, 256),)](
                handle.recv_src_metadata, source_to_recv, recv_count, 2 + k, 256
            )
            _map_slots[(triton.cdiv(meta.size(0) * k, 256),)](
                meta, source_to_recv, reverse, gather, meta.size(0), k, 256
            )
        owner_dy = torch.index_select(recv, 0, gather)
        offsets = torch.nn.functional.pad(forward.psum_num_recv_tokens_per_expert, (1, 0)).int()
        dh = tex.mega_moe_dgrad(owner_dy, offsets, w2)
        dz, dp = torch.empty_like(z), torch.empty_like(p)
        if x.size(0):
            width = y.size(1)
            _activation_grad[(x.size(0),)](
                dh, z, p, raw, dz, dp, width, triton.next_power_of_2(width)
            )
        partial = tex.mega_moe_dgrad(dz, offsets, w1)
        reduced = torch.empty_like(recv)
        rdp = dp.new_empty((recv_count, k))
        if recv_count:
            _merge[(recv_count, triton.cdiv(dy.size(1), 256))](
                partial, reverse, reduced, dy.size(1), 256, k
            )
            _pack_dp[(triton.cdiv(reverse.numel(), 256),)](dp, reverse, rdp, reverse.numel(), 256)
        dx, dp_src, _ = base.buffer.combine(reduced, handle, topk_weights=rdp)
        dw1 = [torch.empty_like(w) for w in w1]
        dw2 = [torch.empty_like(w) for w in w2]
        tex.mega_moe_wgrad(owner_dy, y, offsets, dw2, False)
        tex.mega_moe_wgrad(dz, x, offsets, dw1, False)
        return dx, None, dp_src, None, *dw1, *dw2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-recompute",
        action="store_true",
        help="Also time the previous MegaMoE training path",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect backward kernel counts outside the timed loop",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1073],
        help="Cycle these balanced routes between rounds",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        parser.error("expected nonnegative warmup and positive iteration count")
    rank = int(os.environ["LOCAL_RANK"])
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev)
    group = dist.group.WORLD
    assert group.size() == 8, "expected eight ranks"
    torch.manual_seed(1500 + rank)
    h, i, t = 3584, 3072, 2048
    w1 = [
        (torch.randn(2 * i, h, device=dev) / h**0.5).bfloat16().requires_grad_() for _ in range(8)
    ]
    w2 = [(torch.randn(h, i, device=dev) / i**0.5).bfloat16().requires_grad_() for _ in range(8)]
    x = torch.randn(t, h, device=dev).bfloat16().requires_grad_()
    dy = torch.randn_like(x)
    batches = [routes(t, 64, 8, seed, rank, dev) for seed in args.seeds]
    for _, p in batches:
        p.requires_grad_()
    ids, p = batches[0]
    params = (x, p, *w1, *w2)

    def saved():
        out = te.bf16_mega_moe(x, ids, p, w1, w2, ep_group=group)
        assert "TrainBackward" in type(out.grad_fn).__name__
        return out

    base = Baseline(group, t, h, 8, None, w1, w2, 39)
    methods = {
        "saved": saved,
        "deepep": lambda: DeepTraining.apply(x, ids, p, base, *w1, *w2),
        "deepep_expanded": lambda: DeepExpandedTraining.apply(x, ids, p, base, *w1, *w2),
    }
    if args.include_recompute:
        methods["recompute"] = lambda: _bf16_forward(x, ids, p, w1, w2, group.group_name)
    # Cross-check every implementation before accepting its samples.
    ref = None
    for name in methods:
        if not rank:
            print("CHECK", name, flush=True)
        out = methods[name]()
        torch.cuda.synchronize()
        if not rank:
            print("CHECK_BWD", name, flush=True)
        tensors = (out.detach(), *torch.autograd.grad(out, params, dy))
        torch.cuda.synchronize()
        if not rank:
            print("CHECK_COMPLETE", name, flush=True)
        if ref is None:
            ref = tensors
        else:
            drift = torch.stack(
                [
                    (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
                    for a, b in zip(tensors, ref)
                ]
            )
            dist.all_reduce(drift, dist.ReduceOp.MAX)
            if not rank:
                print("REL_L2", name, drift.tolist(), flush=True)
            assert drift.max().item() < 0.012
    del ref, tensors
    if args.profile:
        out = methods["saved"]()
        kernels, _ = profile_kernels(lambda: torch.autograd.grad(out, params, dy))
        if not rank:
            print("BWD_KERNELS", json.dumps(dict(kernels)), flush=True)
    data = {name: [] for name in methods}
    a, b, c = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    for round_idx in range(args.warmup + args.iters):
        ids, p = batches[round_idx % len(batches)]
        params = (x, p, *w1, *w2)
        order = list(methods)
        order = order[round_idx % len(methods) :] + order[: round_idx % len(methods)]
        for name in order:
            dist.barrier()
            torch.cuda.synchronize()
            a.record()
            out = methods[name]()
            b.record()
            grad = torch.autograd.grad(out, params, dy)
            c.record()
            c.synchronize()
            elapsed = torch.tensor(
                [a.elapsed_time(b), b.elapsed_time(c), a.elapsed_time(c)], device=dev
            )
            dist.all_reduce(elapsed, dist.ReduceOp.MAX)
            sample = elapsed.tolist()
            if round_idx >= args.warmup:
                data[name].append(sample)
            del grad, out
        if not rank:
            print("ROUND", round_idx, flush=True)
    if not rank:
        print(
            "MEDIAN",
            json.dumps({k: list(map(statistics.median, zip(*v))) for k, v in data.items()}),
            flush=True,
        )
        print("SAMPLES", json.dumps(data), flush=True)
    base.buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
