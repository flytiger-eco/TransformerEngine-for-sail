# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare EP8 BF16 MegaMoE with DeepEP on balanced randomized routes."""
import argparse
import json
import os
import random
import statistics
import subprocess
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

os.environ.setdefault("NVTE_USE_CUTLASS_GROUPED_GEMM", "1")
os.environ.setdefault("NVTE_USE_CUTLASS_GROUPED_GEMM_RTC", "1")
os.environ.setdefault("NVTE_FORCE_RTC_NO_FALLBACK", "1")
import torch
import torch.distributed as dist
import benchmark_mega_moe_deepep as upstream


def distribution_version(name):
    """A source checkout need not have installed distribution metadata."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def source_sha(root):
    """Record a checkout revision without failing for installed packages."""
    root = Path(root)
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def routes(tokens, total_experts, topk, seed, rank, device):
    """Exact per-source expert histogram; swaps remove duplicate expert slots."""
    gen = torch.Generator().manual_seed(seed + rank)
    assert tokens * topk % total_experts == 0
    pool = torch.arange(total_experts).repeat_interleave(tokens * topk // total_experts)
    rows = pool[torch.randperm(len(pool), generator=gen)].view(tokens, topk).tolist()
    rng = random.Random(seed + rank)
    for i, row in enumerate(rows):
        for slot in range(topk):
            old = row[slot]
            if old not in row[:slot]:
                continue
            while True:
                j, k = rng.randrange(tokens), rng.randrange(topk)
                other = rows[j][k]
                if j != i and other not in row and old not in rows[j]:
                    row[slot], rows[j][k] = other, old
                    break
    ids = torch.tensor(rows, dtype=torch.int32)
    assert all(len(set(row)) == topk for row in rows)
    assert torch.bincount(ids.to(torch.int64).flatten()).eq(tokens * topk // total_experts).all()
    probs = torch.randn(tokens, topk, generator=gen).softmax(1)
    return ids.to(device), probs.to(device)


class Baseline(upstream.DeepEPForward):
    topk = 8

    @staticmethod
    def make_buffer(group, tokens, hidden, num_sms):
        from deep_ep import ElasticBuffer

        return ElasticBuffer(
            group,
            num_max_tokens_per_rank=tokens,
            hidden=hidden,
            num_topk=Baseline.topk,
            allow_hybrid_mode=False,
            explicitly_destroy=True,
        )

    def __call__(self, x, ids, probs):
        from deep_ep import topk_idx_t

        rx, _, rp, handle, _ = self.buffer.dispatch(
            x,
            topk_idx=ids.to(topk_idx_t),
            topk_weights=probs,
            num_experts=dist.get_world_size() * self.experts,
            num_sms=self.num_sms,
            expert_alignment=1,
            do_cpu_sync=True,
            do_expand=True,
            do_handle_copy=False,
        )
        counts = handle.num_recv_tokens_per_expert_list
        partial = self.expert_mlp(rx, rp, counts)
        out, _, _ = self.buffer.combine(partial, handle)
        return out, counts


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--local-experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--mpe", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--route-variants", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--deep-ep-sms", type=int, default=39)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "hidden",
        "intermediate",
        "local_experts",
        "mpe",
        "route_variants",
        "rounds",
        "iters",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.topk <= 8 * args.local_experts:
        parser.error("--topk must be between 1 and the total expert count")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.mpe * args.local_experts % args.topk:
        parser.error("MPE times local experts must be divisible by top-k")
    if args.mpe % 8:
        parser.error("MPE must be divisible by 8 for exact per-source expert counts")
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", device_id=device)
    world = dist.get_world_size()
    assert world == 8
    assert args.mpe * args.local_experts % args.topk == 0
    tokens = args.mpe * args.local_experts // args.topk
    Baseline.topk = args.topk
    import transformer_engine.pytorch as te
    import deep_ep
    from transformer_engine.common import _get_shared_object_file

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    log("Creating weights and exact balanced top-k routes")
    w1, w2 = upstream.make_weights(
        args.hidden, args.intermediate, world * args.local_experts, device, args.seed
    )
    shard = slice(rank * args.local_experts, (rank + 1) * args.local_experts)
    local1, local2 = w1[shard], w2[shard]
    gen = torch.Generator(device=device).manual_seed(args.seed + 100 + rank)
    x = torch.randn(tokens, args.hidden, generator=gen, device=device).bfloat16()
    cases = [
        (
            x,
            *routes(
                tokens,
                world * args.local_experts,
                args.topk,
                args.seed + 1000 * (i + 1),
                rank,
                device,
            ),
        )
        for i in range(args.route_variants)
    ]
    baseline = Baseline(
        dist.group.WORLD,
        tokens,
        args.hidden,
        args.local_experts,
        upstream.ScaledSwiGLU(_get_shared_object_file("core")),
        local1,
        local2,
        args.deep_ep_sms,
    )
    methods = {
        "deepep": lambda x, ids, probs: baseline(x, ids, probs)[0],
        "mega": lambda x, ids, probs: te.bf16_mega_moe(
            x, ids, probs, local1, local2, ep_group=dist.group.WORLD
        ),
    }
    result = {
        "rank": rank,
        "args": vars(args) | {"output": str(args.output)},
        "optimization_env": {
            name: os.environ.get(name)
            for name in (
                "NVTE_MOE_EP8_FUSED",
                "NVTE_MOE_FUSED_COPY_SMS",
                "NVTE_MOE_FUSED_COMBINE_SMS",
                "NVTE_MOE_FUSED_ROUTE_BATCH",
            )
        },
        "tokens_per_rank": tokens,
        "total_experts": world * args.local_experts,
        "versions": {
            "torch": torch.__version__,
            "torch_distribution": distribution_version("torch"),
            "deep_ep": getattr(deep_ep, "__version__", None),
            "deep_ep_distribution_metadata": distribution_version("deep_ep"),
            "deep_ep_module": deep_ep.__file__,
            "deep_ep_source_sha": source_sha(Path(deep_ep.__file__).resolve().parents[1]),
            "te_source_sha": source_sha(Path(te.__file__).resolve().parents[2]),
            "te_core": str(_get_shared_object_file("core")),
            "te_torch": str(_get_shared_object_file("torch")),
        },
        "hardware": str(torch.cuda.get_device_properties(device)),
        "correctness": [],
        "route_stats": [],
        "counts": [],
    }
    for i, case in enumerate(cases):
        log(f"Correctness variant {i}: independent FP32 reference")
        ref = upstream.reference(*case, w1, w2)
        log(f"Correctness variant {i}: DeepEP")
        base, counts = baseline(*case)
        assert counts == [args.mpe] * args.local_experts, counts
        log(f"Correctness variant {i}: MegaMoE")
        actual = methods["mega"](*case)
        checks = {
            "deepep_vs_reference": upstream.error(base, ref),
            "mega_vs_reference": upstream.error(actual, ref),
            "mega_vs_deepep": upstream.error(actual, base),
        }
        for name, metrics in checks.items():
            assert metrics["relative_l2"] < 0.012 and metrics["max_abs"] < 0.15, (name, metrics)
        log(f"Correctness variant {i}: {checks}")
        result["correctness"].append(checks)
        result["counts"].append(counts)
        owners = case[1].to(torch.int64) // args.local_experts
        result["route_stats"].append(
            {
                "remote_route_fraction": (owners != rank).float().mean().item(),
                "expert_histogram_per_source": torch.bincount(
                    case[1].to(torch.int64).flatten()
                ).tolist(),
            }
        )
    from transformer_engine.pytorch._mega_moe_peer import _workspace

    state = _workspace(dist.group.WORLD)
    result["mega_path"] = {
        "peer_supported": state.supported,
        "bulk_supported": state.bulk_supported,
        "fused_supported": state.fused_supported,
    }
    log(f'MegaMoE path: {result["mega_path"]}')
    if args.profile:
        log("Profiling warmed kernel identities outside timed measurements")
        result["profiles"] = {}
        for name, fn in methods.items():
            counts, events = upstream.profile_kernels(lambda: fn(*cases[0]))
            result["profiles"][name] = {"kernel_names": dict(counts), "events": events}
        rtc = [
            name
            for name, count in result["profiles"]["deepep"]["kernel_names"].items()
            for _ in range(count)
            if name.startswith("nvte_grouped_gemm3_")
        ]
        assert len(rtc) == 2, rtc
    log("Warmup and alternating AB/BA measurement")
    for i in range(args.warmup):
        for fn in methods.values():
            fn(*cases[i % len(cases)])
    torch.cuda.synchronize()
    result["event_ms"], result["wall_ms_per_iter"] = upstream.measure(
        methods, cases, args.rounds, args.iters
    )
    gathered = [None] * world
    dist.all_gather_object(gathered, result)
    if rank == 0:
        summary = {}
        for name in methods:
            grid = [
                [max(r["event_ms"][name][i][j] for r in gathered) for j in range(args.iters)]
                for i in range(args.rounds)
            ]
            walls = [
                max(r["wall_ms_per_iter"][name][i] for r in gathered) for i in range(args.rounds)
            ]
            summary[name] = {
                "event_ms": upstream.percentiles([v for row in grid for v in row]),
                "event_ms_round_medians": [statistics.median(row) for row in grid],
                "wall_ms_per_iter": upstream.percentiles(walls),
            }
        summary["mega_speedup_vs_deepep"] = (
            summary["deepep"]["event_ms"]["median"] / summary["mega"]["event_ms"]["median"]
        )
        summary["mega_latency_reduction_pct"] = 100 * (1 - 1 / summary["mega_speedup_vs_deepep"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary_rank_max": summary, "ranks": gathered}, indent=2)
        )
        log(json.dumps(summary, indent=2))
    baseline.buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    with torch.no_grad():
        main()
