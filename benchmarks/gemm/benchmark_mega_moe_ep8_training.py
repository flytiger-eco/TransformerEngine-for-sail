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
from benchmark_mega_moe_deepep import profile_kernels

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch._mega_moe_train import _activation, _activation_grad
from transformer_engine.pytorch.mega_moe import _bf16_forward
from deep_ep import topk_idx_t


@triton.jit
def _merge(part, reverse, dst, H: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for k in tl.static_range(8):
        src = tl.load(reverse + row * 8 + k)
        acc += tl.load(part + src.to(tl.int64) * H + col, (src >= 0) & (col < H), 0).to(tl.float32)
    tl.store(dst + row.to(tl.int64) * H + col, acc, col < H)


class DeepTraining(torch.autograd.Function):
    """DeepEP dispatch with cached backward routing and expert-local grouped GEMMs."""

    @staticmethod
    def forward(ctx, x, ids, probs, base, *weights):
        w1, w2 = weights[:8], weights[8:]
        rx, ri, rp, handle, _ = base.buffer.dispatch(
            x,
            topk_idx=ids.to(topk_idx_t),
            topk_weights=probs,
            num_experts=64,
            num_sms=39,
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
        rows = slot // 8
        reverse = torch.full_like(flat, -1, dtype=torch.int32)
        reverse[slot] = torch.arange(slot.numel(), device=x.device, dtype=torch.int32)
        owner_x = rx[rows]
        p = rp.flatten()[slot]
        z = base.linear(owner_x, w1, counts)
        y = x.new_empty((len(rows), 3072))
        raw = p.new_empty(y.shape)
        _activation[(len(rows), 12)](z, p, raw, y, 3072, 256)
        partial = base.linear(y, w2, counts)
        reduced = torch.empty_like(rx)
        _merge[(len(rx), 14)](partial, reverse, reduced, 3584, 256)
        out, _, _ = base.buffer.combine(reduced, handle)
        ctx.save_for_backward(owner_x, z, raw, y, p, rows, slot, reverse, *weights)
        ctx.base, ctx.handle = base, handle
        return out

    @staticmethod
    def backward(ctx, dy):
        x, z, raw, y, p, rows, slot, reverse, *weights = ctx.saved_tensors
        w1, w2 = weights[:8], weights[8:]
        base, handle = ctx.base, ctx.handle
        recv, _, _, _, _ = base.buffer.dispatch(dy.contiguous(), handle=handle, num_sms=39)
        owner_dy = recv[rows]
        offsets = torch.nn.functional.pad(handle.psum_num_recv_tokens_per_expert, (1, 0)).int()
        dh = tex.mega_moe_dgrad(owner_dy, offsets, w2)
        dz, dp = torch.empty_like(z), torch.empty_like(p)
        _activation_grad[(len(rows),)](dh, z, p, raw, dz, dp, 3072, 4096)
        partial = tex.mega_moe_dgrad(dz, offsets, w1)
        reduced = torch.empty_like(recv)
        _merge[(len(recv), 14)](partial, reverse, reduced, 3584, 256)
        rdp = dp.new_zeros((len(recv), 8))
        rdp.flatten()[slot] = dp
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
    ids, p = routes(t, 64, 8, 1073, rank, dev)
    p.requires_grad_()
    params = (x, p, *w1, *w2)

    def saved():
        out = te.bf16_mega_moe(x, ids, p, w1, w2, ep_group=group)
        assert "TrainBackward" in type(out.grad_fn).__name__
        return out

    base = Baseline(group, t, h, 8, None, w1, w2, 39)
    methods = {
        "saved": saved,
        "deepep": lambda: DeepTraining.apply(x, ids, p, base, *w1, *w2),
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
    for round_idx in range(args.warmup + args.iters):
        order = list(methods)
        order = order[round_idx % len(methods) :] + order[: round_idx % len(methods)]
        for name in order:
            dist.barrier()
            torch.cuda.synchronize()
            a, b, c = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
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
