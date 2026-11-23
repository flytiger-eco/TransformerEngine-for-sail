# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""EP8/top-8 forward and saved-activation backward regressions for public MegaMoE; run with eight torchrun ranks.

Includes empty/ragged source ranks, skipped/duplicate/zero-weight slots, empty
expert owners, and workspace growth/reuse across independently scheduled streams.
This is correctness-only and performs no timing or profiling.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks/gemm"))
from benchmark_mega_moe_deepep import error, make_weights, reference


def make_case(tokens, rank, seed, device, special=False, empty_owners=False):
    gen = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(tokens, 512, device=device, generator=gen).bfloat16()
    ids = torch.rand(tokens, 16, device=device, generator=gen).argsort(1)[:, :8].int()
    probs = torch.randn(tokens, 8, device=device, generator=gen).softmax(1)
    if empty_owners:
        # Only global experts 0..7 are used; ranks 4..7 own zero expert rows.
        ids = torch.rand(tokens, 8, device=device, generator=gen).argsort(1).int()
    if special:
        ids[::11, 1] = ids[::11, 0]
        ids[::13, 3] = -1
        # An expert is valid even if its route probability is zero.
        probs[::7, 4] = 0
    return x, ids, probs


def independent_reference(case, weights1, weights2, duplicates=False):
    x, ids, probs = case
    if not duplicates:
        return reference(x, ids, probs, weights1, weights2)
    # The upstream helper uses advanced-index +=, which assumes no repeated
    # source row within one expert. Evaluate each slot separately so duplicate
    # expert slots remain independent; then accumulate once in FP32.
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.size(1)):
        result += reference(
            x, ids[:, slot : slot + 1], probs[:, slot : slot + 1], weights1, weights2
        ).float()
    return result.bfloat16()


def check_collectively(label, actual, expected, device):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        values = [float("inf"), float("inf"), 1.0]
    elif not actual.numel():
        values = [0.0, 0.0, 0.0]
    else:
        metrics = error(actual, expected)
        values = [
            metrics["relative_l2"],
            metrics["max_abs"],
            float(not torch.isfinite(actual).all().item()),
        ]
    worst = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    rel, maximum, nonfinite = worst.tolist()
    passed = rel < 0.012 and maximum < 0.15 and nonfinite == 0
    record = {
        "case": label,
        "relative_l2_max_rank": rel,
        "max_abs_max_rank": maximum,
        "passed": passed,
    }
    if dist.get_rank() == 0:
        print(json.dumps(record), flush=True)
    # All ranks execute the same reduction and reach this assertion together.
    assert passed, record
    return record


@torch.no_grad()
def training_reference(x, ids, probs, dy, all1, all2):
    # Compute each route on its source, independently of MegaMoE's owner layout.
    out = torch.zeros_like(x, dtype=torch.float32)
    dx = torch.zeros_like(out)
    dp = torch.zeros_like(probs)
    dw1 = torch.empty((len(all1), *all1[0].shape), dtype=torch.float32, device=x.device)
    dw2 = torch.empty((len(all2), *all2[0].shape), dtype=torch.float32, device=x.device)
    for expert, (w1, w2) in enumerate(zip(all1, all2)):
        row, slot = (ids == expert).nonzero(as_tuple=True)
        inp, grad = x[row], dy[row]
        gate, up = (inp @ w1.T).float().chunk(2, dim=1)
        sigmoid = torch.sigmoid(gate)
        raw = gate * sigmoid * up
        prob = probs[row, slot, None]
        y = (raw * prob).to(x.dtype)
        out.index_add_(0, row, (y @ w2.T).float())
        dh = (grad @ w2).float()
        dp[row, slot] = (dh * raw).sum(dim=1)
        scaled = dh * prob
        dz = torch.cat(
            (scaled * up * sigmoid * (1 + gate * (1 - sigmoid)), scaled * (gate * sigmoid)), dim=1
        ).to(x.dtype)
        dx.index_add_(0, row, (dz @ w1).float())
        dw1[expert] = dz.float().T @ inp.float()
        dw2[expert] = grad.float().T @ y.float()
    dist.all_reduce(dw1)
    dist.all_reduce(dw2)
    owner = slice(2 * dist.get_rank(), 2 * dist.get_rank() + 2)
    return (out.to(x.dtype), dx.to(x.dtype), dp, *dw1[owner].to(x.dtype), *dw2[owner].to(x.dtype))


@torch.enable_grad()
def check_training(rank, device):
    import transformer_engine.pytorch as te

    group = dist.group.WORLD
    all1, all2 = make_weights(512, 256, 16, device, 87001)
    w1 = [w.requires_grad_() for w in all1[2 * rank : 2 * rank + 2]]
    w2 = [w.requires_grad_() for w in all2[2 * rank : 2 * rank + 2]]
    cases = []
    for step, tokens in enumerate((2048, 0 if rank == 0 else 2051 + rank)):
        x, ids, p = make_case(
            tokens, rank, 87035 + step, device, special=True, empty_owners=bool(step)
        )
        x.requires_grad_()
        p.requires_grad_()
        dy = torch.randn((x.size(1), tokens), device=device, dtype=x.dtype).T
        expected = training_reference(x, ids, p, dy, all1, all2)
        out = te.bf16_mega_moe(x, ids, p, w1, w2, ep_group=group)
        assert "TrainBackward" in type(out.grad_fn).__name__
        cases.append((x, p, dy, out, expected))
    for x, p, dy, out, expected in reversed(cases):
        actual = torch.autograd.grad(out, (x, p, *w1, *w2), dy)
        drift = torch.stack(
            [
                (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-10)
                for a, b in zip((out, *actual), expected)
            ]
        )
        dist.all_reduce(drift, op=dist.ReduceOp.MAX)
        if not rank:
            print("EP8 training relative L2", drift.tolist(), flush=True)
        assert drift.max().item() < 0.01, drift
    if not rank:
        print("EP8 saved-activation gradient checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-fused", action="store_true")
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    dist.init_process_group("nccl", device_id=device)
    assert dist.get_world_size() == 8
    if args.training:
        check_training(rank, device)
        dist.destroy_process_group()
        return
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch._mega_moe_peer import _workspace

    w1, w2 = make_weights(512, 256, 16, device, 87001)
    local1, local2 = w1[rank * 2 : (rank + 1) * 2], w2[rank * 2 : (rank + 1) * 2]
    baseline = make_case(2048, rank, 87011, device)
    ragged = make_case(0 if rank == 0 else 2048 + 97 * rank, rank, 87021, device, special=True)
    growth = make_case(4096 - 31 * rank, rank, 87031, device)
    empty_owners = make_case(2048 + 13 * rank, rank, 87041, device, empty_owners=True)
    cases = {
        "baseline": baseline,
        "ragged_skipped_duplicate_zero": ragged,
        "growth": growth,
        "empty_expert_owners": empty_owners,
    }
    expected = {
        name: independent_reference(case, w1, w2, duplicates=name.startswith("ragged"))
        for name, case in cases.items()
    }
    torch.cuda.synchronize()
    dist.barrier()

    def forward(case):
        return te.bf16_mega_moe(*case, local1, local2, ep_group=dist.group.WORLD)

    records = []
    for name in ("baseline", "ragged_skipped_duplicate_zero", "empty_expert_owners"):
        if rank == 0:
            print(f"Running {name}", flush=True)
        actual = forward(cases[name])
        torch.cuda.synchronize()
        records.append(check_collectively(name, actual, expected[name], device))

    # Both streams wait only for input/reference preparation. They deliberately
    # do not wait for each other: the public operator must serialize its reusable
    # peer storage. Growth occurs between the first two calls, and later calls
    # shrink input geometry while retaining a larger allocated capacity.
    a, b = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    a.wait_stream(torch.cuda.current_stream())
    b.wait_stream(torch.cuda.current_stream())
    outputs = []
    for stream, name in (
        (a, "baseline"),
        (b, "growth"),
        (a, "baseline"),
        (b, "ragged_skipped_duplicate_zero"),
    ):
        with torch.cuda.stream(stream):
            outputs.append((name, forward(cases[name])))
    torch.cuda.synchronize()
    for iteration, (name, actual) in enumerate(outputs):
        records.append(
            check_collectively(f"cross_stream_{iteration}_{name}", actual, expected[name], device)
        )

    state = _workspace(dist.group.WORLD)
    path = {
        "peer_supported": state.supported,
        "bulk_supported": state.bulk_supported,
        "fused_supported": state.fused_supported,
        "fused_capacity": getattr(state.fused, "capacity", 0),
    }
    paths = [None] * 8
    dist.all_gather_object(paths, path)
    if args.require_fused:
        assert all(p["fused_supported"] and p["fused_capacity"] >= 4096 for p in paths), paths
    if rank == 0:
        report = {
            "configuration": {
                "world": 8,
                "topk": 8,
                "hidden": 512,
                "intermediate": 256,
                "local_experts": 2,
            },
            "checks": records,
            "paths": paths,
        }
        print(json.dumps(report, indent=2), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    dist.destroy_process_group()
    if rank == 0:
        print("EP8 fused edge checks passed (7 cases)", flush=True)


def run_eight_ranks(training):
    if training:
        pytest.importorskip("triton")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        pytest.skip("Eight PPUs required")
    if any(torch.cuda.get_device_capability(rank) != (8, 9) for rank in range(8)):
        pytest.skip("Eight SM89 PPUs required")
    if not hasattr(torch.cuda, "can_device_access_peer") or not all(
        src == dst or torch.cuda.can_device_access_peer(src, dst)
        for src in range(8)
        for dst in range(8)
    ):
        pytest.skip("Eight mutually addressable PPUs required")

    from transformer_engine.pytorch import _mega_moe_peer as peer

    for rank in range(8):
        with torch.cuda.device(rank):
            if not peer.tex.mega_moe_fused_available():
                pytest.skip("Fused MegaMoE device capabilities unavailable")

    # Exercise route batching and source-boundary handling together. Existing
    # EP2 tests retain coverage of the original 8/15/1 partition defaults.
    env = os.environ | {
        "NVTE_MOE_EP8_FUSED": "1",
        "NVTE_MOE_FUSED_COPY_SMS": "8",
        "NVTE_MOE_FUSED_COMBINE_SMS": "4",
        "NVTE_MOE_FUSED_ROUTE_BATCH": "4",
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            str(Path(__file__).resolve()),
            "--require-fused",
            *(["--training"] if training else []),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=650 if training else 420,
    )
    assert result.returncode == 0, result.stdout
    message = (
        "EP8 saved-activation gradient checks passed"
        if training
        else "EP8 fused edge checks passed (7 cases)"
    )
    assert message in result.stdout, result.stdout


def test_fused_ep8_top8_forward():
    run_eight_ranks(training=False)


def test_fused_ep8_top8_training():
    run_eight_ranks(training=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
