# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Device-kernel accounting for the six training GEMMs in a route replay."""

from collections import Counter
import statistics

import torch
import torch.distributed as dist


STAGES = ("fwd_l1", "fwd_l2", "dgrad_l2", "dgrad_l1", "wgrad_l2", "wgrad_l1")
MEGA = {
    "nvte_mega_moe_fwd_pull": 0,
    "nvte_mega_moe_fwd_scatter": 1,
    "nvte_mega_moe_fused_l1": 0,
    "nvte_mega_moe_fused_l2": 1,
    "nvte_mega_moe_peer_fc1": 0,
    "nvte_mega_moe_peer_fc2": 1,
    "nvte_mega_moe_fc1": 0,
    "nvte_mega_moe_fc2": 1,
    "nvte_mega_moe_dgrad_pull": 2,
    "nvte_mega_moe_dgrad_scatter": 3,
}


def attribute(kernels, backward, local_experts):
    """A phase contains one invocation. Repeated gradient symbols launch FC2 first."""
    groups = [[] for _ in STAGES]
    counts = Counter()
    # expert_gemm batches tensor gradients by MegaMoEGradArgs::kMaxExperts.
    # FC2 and FC1 each launch all their chunks, including a partial final chunk.
    tensor_chunks = (local_experts + 31) // 32
    for name, us in kernels:
        stage = MEGA.get(name)
        # FC1 can also appear as a separate recompute during generic backward.
        if stage is not None and backward != (stage >= 2):
            continue
        if stage is None:
            pair = None
            if not backward and name.startswith("nvte_grouped_gemm"):
                pair = 0
            elif backward and name in ("nvte_mega_moe_dgrad", "nvte_mega_moe_tensor_dgrad"):
                pair = 2
            elif backward and name in ("nvte_mega_moe_wgrad", "nvte_mega_moe_tensor_wgrad"):
                pair = 4
            if pair is not None:
                launches = tensor_chunks if name.startswith("nvte_mega_moe_tensor_") else 1
                if counts[pair] >= 2 * launches:
                    raise ValueError(f"ambiguous GEMM sequence: {name}")
                stage = pair + counts[pair] // launches
                counts[pair] += 1
            elif any(s in name for s in ("intranode::dispatch", "elastic::dispatch")):
                stage = 2 if backward else 0
            elif any(s in name for s in ("intranode::combine", "elastic::combine")):
                stage = 3 if backward else 1
            elif name in ("nvte_mega_moe_bulk_push", "_backward_push"):
                stage = 2 if backward else 0
            elif name in ("nvte_mega_moe_bulk_pull", "_backward_pull"):
                stage = 3 if backward else 1
        if stage is not None:
            groups[stage].append((name, us))
    return groups


def profile_steps(forward, backward, training, device, local_experts):
    # Replay one captured layer four times per logical step. Release each graph before
    # the next replay, so profiling does not require four layers of activation memory.
    layers, steps = 4, 3
    samples = []
    symbols = [set() for _ in STAGES]
    for _ in range(steps):
        dist.barrier()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            for layer in range(layers):
                with torch.profiler.record_function(f"replay_fwd_{layer}"):
                    y = forward()
                    torch.cuda.synchronize()
                if training:
                    with torch.profiler.record_function(f"replay_bwd_{layer}"):
                        backward(y)
                        torch.cuda.synchronize()
                del y
        regions = {}
        kernels = []
        for event in prof.events():
            span = event.time_range
            if event.device_type == torch.autograd.DeviceType.CUDA:
                kernels.append((span.start, event.name, span.end - span.start))
            elif event.name.startswith(("replay_fwd_", "replay_bwd_")):
                regions[event.name] = span
        kernels.sort()
        times = [0.0] * len(STAGES)
        for region, span in regions.items():
            events = [(n, us) for start, n, us in kernels if span.start <= start < span.end]
            for i, group in enumerate(
                attribute(events, region.startswith("replay_bwd_"), local_experts)
            ):
                times[i] += sum(us for _, us in group) / 1000
                symbols[i].update(n for n, _ in group)
        values = torch.tensor(times, device=device, dtype=torch.float64)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        samples.append(values.tolist())
    rank_symbols = [None] * dist.get_world_size()
    dist.all_gather_object(rank_symbols, symbols)
    names = [sorted(set.union(*entries)) for entries in zip(*rank_symbols)]
    return {
        "layers_per_step": layers,
        "steps": steps,
        "mean_ms": dict(zip(STAGES, map(statistics.mean, zip(*samples)))),
        "kernels": dict(zip(STAGES, names)),
    }
