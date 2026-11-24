#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Replay expert row counts through the public EP2 MegaMoE autograd API.

Run with torchrun --standalone --nproc-per-node=2. Each input TAG_LINE:fc1
or TAG_LINE:fc2 names a captured forward projection. Times include backward
routing, FC1 recomputation, both dgrads, both wgrads and returning dx and dp.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from transformer_engine.common import _get_shared_object_file

from benchmark_mega_moe_deepep import error, make_weights
from replay_tag import extent, parse_tag


def case_shape(lines, spec):
    index, stage = spec.split(":")
    index = int(index)
    if not 1 <= index <= len(lines):
        raise ValueError(f"Tag line {index} is outside the input file")
    op, fields = parse_tag(lines[index - 1])
    if op != "GroupedGemmRtc" or fields["layout"] != "NT" or stage not in ("fc1", "fc2"):
        raise ValueError(f"{spec} must identify an RTC NT forward projection")
    groups = int(fields["groups"])
    counts = extent(fields["m"])
    if not fields["m"].startswith("["):
        counts *= groups
    n, k = int(fields["n"]), int(fields["k"])
    if len(counts) != groups or (stage == "fc1" and n % 2):
        raise ValueError(f"Invalid group or gated dimension in {spec}")
    return counts, (k if stage == "fc1" else n), (n // 2 if stage == "fc1" else k)


def routes(counts, rank, generator, device):
    # Swap the rank receiving the odd extra slot for each expert on owner 1.
    # Both owners see the captured counts exactly; both sources emit sum(counts) routes.
    n = len(counts)
    chunks = []
    for owner in range(2):
        for expert, count in enumerate(counts):
            slots = count // 2 + int(count % 2 and (owner + expert) % 2 == rank)
            chunks.append(torch.full((slots,), owner * n + expert, dtype=torch.int32))
    pool = torch.cat(chunks)
    pool = pool[torch.randperm(pool.numel(), generator=generator)]
    if pool.numel() % 2:
        pool = torch.cat([pool, pool.new_tensor([-1])])
    return pool.view(-1, 2).to(device)


def gather(t):
    peers = [torch.empty_like(t) for _ in range(2)]
    dist.all_gather(peers, t.contiguous())
    return torch.cat(peers)


@torch.no_grad()
def check_gradients(params, ids, dy, weights, grads, rank):
    # Independent per-expert dense FP32 GEMMs preserve the BF16 cast points
    # of the public training operator and reduce both routes into each token.
    x, probs = params[:2]
    local_tokens, hidden = x.shape
    ax, ap, ai, ady = [gather(t.detach()) for t in (x, probs, ids, dy)]
    gslot = torch.zeros((*ai.shape, hidden), dtype=x.dtype, device=x.device)
    pslot = torch.zeros_like(ap)
    w1, w2 = weights
    g = len(w1)
    errors = {"dW1": [], "dW2": []}
    for e in range(g):
        rows, slots = torch.where(ai == rank * g + e)
        batch, grad = ax[rows].float(), ady[rows].float()
        gate, up = (batch @ w1[e].detach().float().T).bfloat16().float().chunk(2, dim=1)
        sig = torch.sigmoid(gate)
        silu = gate * sig
        scale = ap[rows, slots, None]
        act = (silu * up * scale).bfloat16().float()
        dh = (grad @ w2[e].detach().float()).bfloat16().float()
        pslot[rows, slots] = (dh * (silu * up)).sum(1)
        scaled = dh * scale
        dz = torch.cat([scaled * up * sig * (1 + gate * (1 - sig)), scaled * silu], 1)
        dz = dz.bfloat16().float()
        gslot[rows, slots] = (dz @ w1[e].detach().float()).bfloat16()
        errors["dW1"].append(error(grads[2 + e], (dz.T @ batch).bfloat16()))
        errors["dW2"].append(error(grads[2 + g + e], (grad.T @ act).bfloat16()))
    # Exactly one contributing owner per slot, including duplicate expert IDs.
    dist.all_reduce(gslot)
    dist.all_reduce(pslot)
    sl = slice(rank * local_tokens, (rank + 1) * local_tokens)
    errors["dX"] = [error(grads[0], gslot[sl].float().sum(1).bfloat16())]
    errors["dP"] = [error(grads[1], pslot[sl])]
    stats = {name: max(x["relative_l2"] for x in samples) for name, samples in errors.items()}
    if max(stats.values()) >= 0.012 or not all(torch.isfinite(t).all() for t in grads):
        raise AssertionError(stats)
    return stats


def run_case(lines, spec, args, rank):
    counts, hidden, intermediate = case_shape(lines, spec)
    device = torch.device("cuda", rank)
    ids = routes(counts, rank, torch.Generator().manual_seed(args.seed + rank), device)
    gen = torch.Generator(device=device).manual_seed(args.seed + 2 + rank)
    x = torch.randn((ids.size(0), hidden), device=device, generator=gen).bfloat16().requires_grad_()
    probs = torch.randn(ids.shape, device=device, generator=gen).softmax(1).requires_grad_()
    w1, w2 = make_weights(hidden, intermediate, len(counts), device, args.seed + 17 + rank)
    for weight in [*w1, *w2]:
        weight.requires_grad_()
    params = (x, probs, *w1, *w2)
    out = te.bf16_mega_moe(x, ids, probs, w1, w2, ep_group=dist.group.WORLD)
    dy = torch.randn(out.shape, device=device, generator=gen).bfloat16()

    def step():
        return torch.autograd.grad(out, params, dy, retain_graph=True)

    grads = step()
    errors = check_gradients(params, ids, dy, (w1, w2), grads, rank)
    del grads
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as trace:
        step()
        torch.cuda.synchronize()
    events = [
        {
            "name": e.name,
            "start_us": e.time_range.start,
            "us": e.device_time,
            "stream": e.device_resource_id,
        }
        for e in trace.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    elapsed = []
    for _ in range(args.rounds):
        timers = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(args.iters)
        ]
        dist.barrier()
        torch.cuda.synchronize()
        for a, b in timers:
            a.record()
            step()
            b.record()
        torch.cuda.synchronize()
        elapsed.extend(a.elapsed_time(b) for a, b in timers)
    result = {"rank": rank, "relative_l2": errors, "events": events, "times_ms": elapsed}
    peers = [None, None]
    dist.all_gather_object(peers, result)
    slow = [max(v) for v in zip(*(p["times_ms"] for p in peers), strict=True)]
    if rank == 0:
        print(spec, "median backward ms", statistics.median(slow), "errors", errors, flush=True)
    return {
        "case": spec,
        "owner_counts": counts,
        "hidden": hidden,
        "intermediate": intermediate,
        "median_ms": statistics.median(slow),
        "ranks": peers,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-file", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["26:fc1", "29:fc1", "151:fc2", "1:fc1"])
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or min(args.rounds, args.iters) < 1:
        parser.error("Expected nonnegative warmup and positive rounds and iterations")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    if dist.get_world_size() != 2 or int(os.environ["LOCAL_WORLD_SIZE"]) != 2:
        raise ValueError("This benchmark uses two EP ranks on one node")
    torch.set_float32_matmul_precision("highest")
    lines = args.tag_file.read_text().splitlines()
    results = []
    for spec in args.cases:
        results.append(run_case(lines, spec, args, rank))
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "torch": torch.__version__,
                        "core": str(_get_shared_object_file("core")),
                        "config": vars(args),
                        "cases": results,
                    },
                    indent=2,
                    default=str,
                )
                + "\n"
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
