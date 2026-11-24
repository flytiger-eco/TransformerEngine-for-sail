# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compare public BF16 MegaMoE with DeepEP dispatch/combine and TE grouped RTC forward.

Run with two peer-connected devices, for example:

    CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc-per-node 2 \
        benchmarks/gemm/benchmark_mega_moe_deepep.py --output results.json

DeepEP's SM89 JIT needs the CUDA-compatible PCCL device headers. If the installation
does not provide them, run DeepEP's generate_cuda_header.py on the matching PCCL
headers and set EP_PCCL_ROOT_DIR to the directory containing include_cuda/.
"""

import argparse
import ctypes
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import torch
import torch.distributed as dist


class NVTEShape(ctypes.Structure):
    _fields_ = [("data", ctypes.c_size_t * 15), ("ndim", ctypes.c_size_t)]


class NVTEBasicTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("dtype", ctypes.c_int), ("shape", NVTEShape)]


class ScaledSwiGLU:
    """The core C API fuses FP32 SwiGLU, FP32 route scaling and one BF16 store."""

    def __init__(self, core_path):
        self.core = ctypes.CDLL(str(core_path))
        ptr = ctypes.c_void_p
        self.core.nvte_create_tensor.argtypes = [ctypes.c_int]
        self.core.nvte_create_tensor.restype = ptr
        self.core.nvte_set_tensor_param_v2.argtypes = [ptr, ctypes.c_int, ptr, ctypes.c_size_t]
        self.core.nvte_set_tensor_param_v2.restype = None
        self.core.nvte_scaled_swiglu.argtypes = [ptr, ptr, ptr, ctypes.c_int64, ptr]
        self.core.nvte_scaled_swiglu.restype = None
        self.core.nvte_destroy_tensor.argtypes = [ptr]
        self.core.nvte_destroy_tensor.restype = None

    def __call__(self, projection, probs):
        out = projection.new_empty((projection.size(0), projection.size(1) // 2))
        handles = []
        try:
            for tensor, dtype in ((projection, 6), (probs, 4), (out, 6)):
                shape = NVTEShape((ctypes.c_size_t * 15)(*tensor.shape), tensor.ndim)
                data = NVTEBasicTensor(tensor.data_ptr(), dtype, shape)
                handle = self.core.nvte_create_tensor(0)  # NVTE_DELAYED_TENSOR_SCALING
                handles.append(handle)
                self.core.nvte_set_tensor_param_v2(
                    handle, 0, ctypes.byref(data), ctypes.sizeof(data)
                )
            self.core.nvte_scaled_swiglu(*handles, 0, torch.cuda.current_stream().cuda_stream)
        finally:
            for handle in handles:
                self.core.nvte_destroy_tensor(handle)
        return out


def expected_counts(tokens, experts):
    base, extra = divmod(2 * tokens, experts)
    return [base + int(i < extra) for i in range(experts)]


def make_routes(tokens, experts, rank, seed, device, routing):
    """Preserve exact expert counts and a 50% remote-route fraction on each source."""
    generator = torch.Generator().manual_seed(seed + rank)
    counts = expected_counts(tokens, experts)
    odd = [i for i, count in enumerate(counts) if count % 2]
    bonus = set(odd[rank::2])
    per_source = torch.tensor([count // 2 + (i in bonus) for i, count in enumerate(counts)])
    assignment = torch.arange(experts).repeat_interleave(per_source)
    if routing == "split":
        destinations = [
            assignment[torch.randperm(tokens, generator=generator)] + owner * experts
            for owner in range(2)
        ]
        ids = torch.stack(destinations, dim=1)
        switch = torch.rand(tokens, generator=generator) < 0.5
        ids[switch] = ids[switch].flip(1)
    else:
        pool = torch.cat([assignment, assignment + experts])
        pairs = pool[torch.randperm(2 * tokens, generator=generator)].view(tokens, 2).tolist()
        rng = random.Random(seed + rank)
        # A second-column swap preserves the histogram. Each expert occurs at
        # most T times, so a pair avoiding the repeated expert always exists.
        for i, (first, second) in enumerate(pairs):
            if first == second:
                j = rng.randrange(tokens)
                while first in pairs[j]:
                    j = (j + 1) % tokens
                pairs[i][1], pairs[j][1] = pairs[j][1], pairs[i][1]
        ids = torch.tensor(pairs)
    assert (ids[:, 0] != ids[:, 1]).all()
    probs = torch.randn(tokens, 2, generator=generator).softmax(1)
    return ids.to(device, torch.int32), probs.to(device)


def route_stats(ids, rank, experts):
    owners = ids.long() // experts
    num_remote = (owners != rank).sum(dim=1)
    return {
        "remote_route_fraction": num_remote.float().mean().item() / 2,
        "tokens_by_num_remote_slots_0_1_2": torch.bincount(num_remote, minlength=3).tolist(),
        "tokens_touching_one_owner": (owners[:, 0] == owners[:, 1]).sum().item(),
        "tokens_touching_two_owners": (owners[:, 0] != owners[:, 1]).sum().item(),
    }


def make_weights(hidden, intermediate, experts, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)

    def draw(n, k):
        return (torch.randn(n, k, device=device, generator=generator) / math.sqrt(k)).bfloat16()

    return (
        [draw(2 * intermediate, hidden) for _ in range(experts)],
        [draw(hidden, intermediate) for _ in range(experts)],
    )


def reference(x, ids, probs, w1, w2):
    """Dense FP32 matmuls with the BF16 storage boundaries of the public forward."""
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert, (first, second) in enumerate(zip(w1, w2)):
        rows, slots = torch.where(ids == expert)
        if rows.numel():
            projection = (x[rows].float() @ first.float().T).bfloat16()
            gate, up = projection.float().chunk(2, dim=1)
            act = (torch.nn.functional.silu(gate) * up * probs[rows, slots, None]).bfloat16()
            partial = (act.float() @ second.float().T).bfloat16()
            # Each source row visits distinct global experts, once per expert.
            out[rows] += partial.float()
    return out.bfloat16()


def error(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError((actual.shape, expected.shape, actual.dtype, expected.dtype))
    diff = (actual.float() - expected.float()).abs()
    return {
        "relative_l2": (diff.norm() / expected.float().norm().clamp_min(1e-12)).item(),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
    }


class DeepEPForward:
    def __init__(self, group, tokens, hidden, experts, scaled_swiglu, w1, w2, num_sms):
        from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm

        self.buffer = self.make_buffer(group, tokens, hidden, num_sms)
        self.gemm = general_grouped_gemm
        self.scaled_swiglu = scaled_swiglu
        self.w1, self.w2 = w1, w2
        self.experts, self.num_sms = experts, num_sms

    @staticmethod
    def make_buffer(group, tokens, hidden, num_sms):
        from deep_ep import ElasticBuffer

        return ElasticBuffer(
            group,
            num_max_tokens_per_rank=tokens,
            hidden=hidden,
            num_topk=2,
            allow_hybrid_mode=False,
            explicitly_destroy=True,
        )

    def linear(self, x, weights, counts):
        out = x.new_empty((x.size(0), weights[0].size(0)))
        self.gemm(
            weights,
            x,
            [out],
            [None] * len(weights),
            x.dtype,
            layout="TN",
            m_splits=counts,
            single_output=True,
        )
        return out

    def __call__(self, x, ids, probs):
        from deep_ep import topk_idx_t

        rx, _, rp, handle, _ = self.buffer.dispatch(
            x,
            topk_idx=ids.to(topk_idx_t),
            topk_weights=probs,
            num_experts=2 * self.experts,
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

    def expert_mlp(self, x, probs, counts):
        projection = self.linear(x, self.w1, counts)
        activation = self.scaled_swiglu(projection, probs)
        return self.linear(activation, self.w2, counts)


class DeepEPLegacyForward(DeepEPForward):
    """Legacy DeepEP plus Megatron's fused index conversion and TE local permutation."""

    @staticmethod
    def make_buffer(group, tokens, hidden, num_sms):
        from deep_ep import Buffer

        if num_sms:
            Buffer.set_num_sms(num_sms)
        configs = (
            Buffer.get_intranode_dispatch_config(group.size(), hidden, "bf16"),
            Buffer.get_intranode_combine_config(group.size(), hidden),
        )
        nvl_bytes = max(c.get_nvl_buffer_size_hint(2 * hidden, group.size()) for c in configs)
        return Buffer(group, num_nvl_bytes=nvl_bytes, explicitly_destroy=True)

    def __call__(self, x, ids, probs):
        from deep_ep import Buffer, topk_idx_t
        from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
        from transformer_engine.pytorch import moe_permute_with_probs, moe_unpermute

        if self.num_sms:
            Buffer.set_num_sms(self.num_sms)
        ids = ids.to(topk_idx_t)
        per_rank, per_rdma, per_expert, in_rank, _ = self.buffer.get_dispatch_layout(
            ids, 2 * self.experts
        )
        rx, ri, rp, counts, handle, _ = self.buffer.dispatch(
            x,
            topk_idx=ids,
            topk_weights=probs,
            num_tokens_per_rank=per_rank,
            num_tokens_per_rdma_rank=per_rdma,
            num_tokens_per_expert=per_expert,
            is_token_in_rank=in_rank,
            expert_alignment=1,
        )
        mask, dense_probs = fused_indices_to_multihot(ri, rp, self.experts)
        packed, row_probs, row_map = moe_permute_with_probs(rx, dense_probs, mask, sum(counts))
        partial = self.expert_mlp(packed, row_probs, counts)
        restored = moe_unpermute(partial, row_map, restore_shape=rx.shape)
        out, _, _ = self.buffer.combine(restored, handle)
        return out, counts


def profile_kernels(fn):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        # Kineto initialization latency differs by rank. Exclude peer startup from
        # the first profiled communication kernel by meeting inside the trace.
        dist.barrier()
        torch.cuda.synchronize()
        with torch.profiler.record_function("measured_moe_forward"):
            fn()
            torch.cuda.synchronize()
    trace = prof.events()
    region = next(e.time_range for e in trace if e.name == "measured_moe_forward")
    events = [
        e
        for e in trace
        if e.device_type == torch.autograd.DeviceType.CUDA
        and region.start <= e.time_range.start <= region.end
    ]
    return Counter(e.name for e in events), [
        {"name": e.name, "device_us": e.device_time} for e in events
    ]


def percentiles(values):
    ordered = sorted(values)

    def q(fraction):
        index = (len(ordered) - 1) * fraction
        lo, hi = math.floor(index), math.ceil(index)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    return {"p10": q(0.1), "median": statistics.median(values), "p90": q(0.9)}


def measure(methods, cases, rounds, iterations):
    """AB / BA blocks; all per-rank event intervals include dispatch, compute and combine."""
    measured = {name: [] for name in methods}
    wall = {name: [] for name in methods}
    for run in range(rounds):
        order = list(methods) if run % 2 == 0 else list(reversed(methods))
        for name in order:
            events = [
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                for _ in range(iterations)
            ]
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for step, (before, after) in enumerate(events):
                before.record()
                methods[name](*cases[(run * iterations + step) % len(cases)])
                after.record()
            torch.cuda.synchronize()
            wall[name].append((time.perf_counter() - start) * 1e3 / iterations)
            measured[name].append([before.elapsed_time(after) for before, after in events])
    return measured, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=17959)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--local-experts", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--routing", choices=["mixed", "split"], default="mixed")
    parser.add_argument("--route-variants", type=int, default=2)
    parser.add_argument(
        "--deepep-implementation",
        choices=("v2", "legacy"),
        default="v2",
        help="legacy additionally requires Megatron-LM on PYTHONPATH",
    )
    parser.add_argument(
        "--deep-ep-sms",
        type=int,
        help=(
            "transport SMs (default: all available for sequential forward; "
            "0 selects DeepEP's overlap-oriented default)"
        ),
    )
    parser.add_argument(
        "--automatic-rtc",
        action="store_true",
        help="use the existing TE dispatch settings and verify the observed baseline kernel",
    )
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--verify-kernels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--expect-three-kernels",
        action="store_true",
        help="Check that the warmed MegaMoE forward launches only pre-dispatch, L1 and L2.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(
            args.tokens,
            args.hidden,
            args.intermediate,
            args.local_experts,
            args.route_variants,
            args.iters,
            args.rounds,
        )
        < 1
        or args.warmup < 0
    ):
        parser.error("invalid tensor dimensions or iteration counts")
    if not args.automatic_rtc:
        os.environ.update(
            NVTE_USE_CUTLASS_GROUPED_GEMM="1",
            NVTE_USE_CUTLASS_GROUPED_GEMM_RTC="1",
            NVTE_FORCE_RTC_NO_FALLBACK="1",
            NVTE_GG_FORCE_TILE="256,256,64,3",
        )
    import transformer_engine.pytorch as te
    from transformer_engine.common import _get_shared_object_file

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() != 2:
        raise ValueError("this benchmark requires two EP ranks")
    group = dist.group.WORLD
    available = torch.tensor(
        torch.cuda.get_device_properties(device).multi_processor_count,
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(available, op=dist.ReduceOp.MIN, group=group)
    if args.deep_ep_sms is None:
        args.deep_ep_sms = int(available)
        if args.deepep_implementation == "legacy":
            args.deep_ep_sms -= args.deep_ep_sms % 2
    torch.set_float32_matmul_precision("highest")
    w1, w2 = make_weights(args.hidden, args.intermediate, 2 * args.local_experts, device, args.seed)
    shard = slice(dist.get_rank() * args.local_experts, (dist.get_rank() + 1) * args.local_experts)
    # Share identical BF16 input and weight storage between the two methods.
    gen = torch.Generator(device=device).manual_seed(args.seed + 2 + dist.get_rank())
    x = torch.randn(args.tokens, args.hidden, device=device, generator=gen).bfloat16()
    cases = [
        (
            x,
            *make_routes(
                args.tokens,
                args.local_experts,
                dist.get_rank(),
                args.seed + 1000 * (i + 1),
                device,
                args.routing,
            ),
        )
        for i in range(args.route_variants)
    ]
    w1_local, w2_local = w1[shard], w2[shard]
    core = _get_shared_object_file("core")
    backend = DeepEPForward if args.deepep_implementation == "v2" else DeepEPLegacyForward
    baseline = backend(
        group,
        args.tokens,
        args.hidden,
        args.local_experts,
        ScaledSwiGLU(core),
        w1_local,
        w2_local,
        args.deep_ep_sms,
    )
    methods = {
        "deepep": lambda x, ids, probs: baseline(x, ids, probs)[0],
        "mega": lambda x, ids, probs: te.bf16_mega_moe(
            x, ids, probs, w1_local, w2_local, ep_group=group
        ),
    }
    result = {
        "args": vars(args) | {"output": str(args.output) if args.output else None},
        "versions": {
            "torch": torch.__version__,
            "deep_ep": version("deep_ep"),
            "te_core": str(core),
            "te_torch": str(_get_shared_object_file("torch")),
        },
        "hardware": torch.cuda.get_device_name(),
        "rank": dist.get_rank(),
        "device_visibility": os.getenv("CUDA_VISIBLE_DEVICES"),
        "route_stats": [
            route_stats(ids, dist.get_rank(), args.local_experts) for _, ids, _ in cases
        ],
        "local_counts": expected_counts(args.tokens, args.local_experts),
        "baseline_tile_fc1_fc2": None if args.automatic_rtc else [256, 256, 64, 3],
        "correctness": [],
    }
    try:
        for case in cases:
            ref = reference(*case, w1, w2)
            base, counts = baseline(*case)
            actual = methods["mega"](*case)
            if counts != result["local_counts"]:
                raise AssertionError(f"DeepEP received {counts}, expected {result['local_counts']}")
            checks = {
                "deepep_vs_reference": error(base, ref),
                "mega_vs_reference": error(actual, ref),
                "mega_vs_deepep": error(actual, base),
            }
            for pair, metrics in checks.items():
                if not (metrics["relative_l2"] < 0.012 and metrics["max_abs"] < 0.15):
                    raise AssertionError((pair, metrics))
            result["correctness"].append(checks)
        if args.verify_kernels or args.expect_three_kernels:
            profiles = {
                name: profile_kernels(lambda: fn(*cases[0])) for name, fn in methods.items()
            }
            kernels = {name: data[0] for name, data in profiles.items()}
            rtc = [
                name
                for name in kernels["deepep"]
                for _ in range(kernels["deepep"][name])
                if name.startswith("nvte_grouped_gemm3_")
            ]
            if len(rtc) != 2 or any(
                "bf16_bf16_bf16_acc_fp32_LA_RM_LB_CM" not in name
                or (not args.automatic_rtc and "_tb256x256x64_s3_" not in name)
                for name in rtc
            ):
                raise AssertionError(
                    f"expected exactly two NT BF16 RTC GEMMs matching the tile policy: {rtc}"
                )
            fragments = (
                ("elastic::dispatch_impl", "elastic::combine_impl")
                if args.deepep_implementation == "v2"
                else ("intranode::dispatch", "intranode::combine")
            )
            for fragment in (*fragments, "scaled_gated_forward_kernel"):
                if sum(count for name, count in kernels["deepep"].items() if fragment in name) != 1:
                    raise AssertionError(f"baseline did not launch one {fragment}")
            if args.expect_three_kernels:
                expected = [
                    f"nvte_mega_moe_fused_{stage}" for stage in ("pre_dispatch", "l1", "l2")
                ]
                actual = [
                    event["name"]
                    for event in profiles["mega"][1]
                    if event["name"] != "measured_moe_forward"
                ]
                if actual != expected:
                    raise AssertionError(f"expected the three fused MegaMoE launches, got {actual}")
            result["kernel_names"] = kernels
            result["profiled_device_events"] = {name: data[1] for name, data in profiles.items()}
        for i in range(args.warmup):
            for fn in methods.values():
                fn(*cases[i % len(cases)])
        torch.cuda.synchronize()
        result["event_ms"], result["wall_ms_per_iter"] = measure(
            methods, cases, args.rounds, args.iters
        )
        gathered = [None] * 2
        dist.all_gather_object(gathered, result)
        if dist.get_rank() == 0:
            summary = {}
            for name in methods:
                grid = [
                    [max(r["event_ms"][name][i][j] for r in gathered) for j in range(args.iters)]
                    for i in range(args.rounds)
                ]
                walls = [
                    max(r["wall_ms_per_iter"][name][i] for r in gathered)
                    for i in range(args.rounds)
                ]
                summary[name] = {
                    "event_ms": percentiles([t for row in grid for t in row]),
                    "event_ms_round_medians": [statistics.median(row) for row in grid],
                    "wall_ms_per_iter": percentiles(walls),
                }
            summary["mega_speedup_vs_deepep"] = (
                summary["deepep"]["event_ms"]["median"] / summary["mega"]["event_ms"]["median"]
            )
            summary["paired_round_speedups"] = [
                base / mega
                for base, mega in zip(
                    summary["deepep"]["event_ms_round_medians"],
                    summary["mega"]["event_ms_round_medians"],
                )
            ]
            for clock in ("event_ms", "wall_ms_per_iter"):
                base = summary["deepep"][clock]["median"]
                candidate = summary["mega"][clock]["median"]
                summary[f"{clock}_reduction_pct"] = 100 * (1 - candidate / base)
            summary["paired_round_reduction_pct"] = [
                100 * (1 - 1 / ratio) for ratio in summary["paired_round_speedups"]
            ]
            report = {"summary_rank_max": summary, "ranks": gathered}
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(summary, indent=2), flush=True)
    finally:
        # Teardown is collective. On a local failure let torchrun stop peers
        # that may still be inside the forward.
        if sys.exc_info()[0] is None:
            baseline.buffer.destroy()
            dist.destroy_process_group()


if __name__ == "__main__":
    with torch.no_grad():
        main()
