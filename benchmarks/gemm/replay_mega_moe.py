# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Replay one captured EP invocation with torchrun, preserving each source slot."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
from unittest.mock import patch

import torch
import torch.distributed as dist
from mega_moe_tags import read_tags
import transformer_engine.pytorch as te
from transformer_engine.pytorch import _mega_moe_peer as peer


def restore(layout, device, values=None):
    shape, strides = layout["shape"], layout["stride"]
    dtype = getattr(torch, layout["dtype"])
    size = 1 + sum((n - 1) * s for n, s in zip(shape, strides)) if math.prod(shape) else 0
    storage = torch.empty(size + 512, device=device, dtype=dtype)
    start = ((layout["offset_bytes"] - storage.data_ptr()) % 512) // storage.element_size()
    tensor = storage.as_strided(shape, strides, start)
    if values is None:
        storage.normal_(std=max(1, shape[-1]) ** -0.5)
    else:
        if tuple(shape) != values.shape or values.dtype != dtype:
            raise ValueError("routing payload does not match its layout")
        # Index the backing storage explicitly for expanded/overlapping source views.
        offsets = torch.arange(shape[0])[:, None] * strides[0] + torch.arange(shape[1]) * strides[1]
        # Aliased entries contain identical values. Install them on CPU before the device copy.
        host = torch.empty(size, dtype=dtype)
        host[offsets.flatten()] = values.flatten()
        storage[start : start + size].copy_(host)
    return tensor.requires_grad_(layout["requires_grad"])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help="rank tag log, legacy manifest.json or v1 ID")
    parser.add_argument("--call", help="invocation to replay when a log contains multiple calls")
    parser.add_argument(
        "--directory", type=Path, default=Path("."), help="capture root for NVTX tags"
    )
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument(
        "--expect-ep8-training",
        action="store_true",
        help="require the EP8 saved-activation backend",
    )
    parser.add_argument("--backend", choices=("megamoe", "deepep"), default="megamoe")
    parser.add_argument(
        "--profile-ep16-path",
        action="store_true",
        help="record per-rank forward/backward kernel and collective counts for a two-node EP16 replay",
    )
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iters < 1:
        parser.error("expected warmup >= 0 and iters >= 1")
    return args


def replay(args, device):
    if args.backend == "deepep":
        os.environ.update(
            NVTE_USE_CUTLASS_GROUPED_GEMM="1",
            NVTE_USE_CUTLASS_GROUPED_GEMM_RTC="1",
            NVTE_FORCE_RTC_NO_FALLBACK="1",
        )
    match = re.fullmatch(r"te_mega_moe:v1:([a-f0-9]{32})", args.capture)
    path = args.directory / match[1] / "manifest.json" if match else Path(args.capture)
    legacy = match or path.name == "manifest.json"
    world = dist.get_world_size()
    os.environ.pop("NVTE_MOE_CAPTURE_TAG", None)
    rank = dist.get_rank()
    torch.manual_seed(args.seed + rank)
    error = None
    try:
        if legacy:
            manifest = json.loads(path.read_text())
            tag = manifest["tag"]
            ranks = manifest["global_ranks"]
            local = manifest["local"]
            if manifest["version"] != 1 or (match and args.capture != tag) or len(ranks) != world:
                raise ValueError("capture version, tag or torchrun world size mismatch")
            shard = torch.load(
                path.parent / f"rank-{rank}.pt", weights_only=True, map_location="cpu"
            )
            if (
                shard["rank"] != rank
                or shard["global_rank"] != ranks[rank]
                or tag != "te_mega_moe:v1:" + shard["id"]
            ):
                raise ValueError("shard does not belong to this invocation/rank")
        else:
            tag, local, shard = read_tags(path, args.call, world, rank)
        group = None if local else dist.group.WORLD
        e = shard["experts"]
        if len(shard["layouts"]) != 3 + 2 * e:
            raise ValueError("expert layout count mismatch")
        x, ids, probs, *weights = [
            restore(layout, device, values)
            for layout, values in zip(
                shard["layouts"], [None, shard["ids"], shard["probs"], *([None] * (2 * e))]
            )
        ]
        if not torch.equal(ids.cpu(), shard["ids"]) or not torch.equal(
            probs.detach().cpu().contiguous().view(torch.int32), shard["probs"].view(torch.int32)
        ):
            raise ValueError("restored source slots differ from captured routing data")
        for key, value in shard.get("settings", {}).items():
            if not key.startswith("NVTE_MOE_") or key.startswith("NVTE_MOE_CAPTURE_"):
                raise ValueError(f"invalid capture setting: {key}")
            os.environ[key] = value
    except Exception as exc:
        error = f"rank {rank}: {exc}"
    errors = [None] * world
    dist.all_gather_object(errors, error)
    if any(errors):
        raise ValueError(f"incomplete or invalid MegaMoE capture: {errors}")
    op = te.bf16_mega_moe if x.dtype == torch.bfloat16 else te.fp16_mega_moe
    params = [v for v in (x, probs, *weights) if v.requires_grad]
    dy = torch.randn_like(x)
    grad_enabled = shard["grad_enabled"]

    training = grad_enabled and bool(params)
    modes = [None] * world
    dist.all_gather_object(modes, training)
    if len(set(modes)) != 1:
        raise ValueError("only some EP ranks have differentiable inputs for backward")

    if args.profile_ep16_path and (
        world != 16 or group is None or not training or args.backend != "megamoe"
    ):
        raise ValueError("--profile-ep16-path requires a distributed EP16 MegaMoE training capture")

    def megamoe_forward():
        with torch.set_grad_enabled(grad_enabled):
            return op(x, ids, probs, weights[:e], weights[e:], ep_group=group)

    def backward(y):
        return torch.autograd.grad(y, params, dy) if grad_enabled and params else ()

    # The reference exchanges actual token rows with all-to-all, bypassing peer kernels.
    with patch.object(peer, "_peer_supported", return_value=False):
        ref = megamoe_forward()
        expected = (ref.detach(), *backward(ref))
    forward = megamoe_forward
    baseline = None
    if args.backend == "deepep":
        from benchmark_mega_moe_deepep import DeepEPLegacyForward
        from benchmark_mega_moe_ep8_training import DeepTraining

        if x.dtype != torch.bfloat16 or group is None:
            raise ValueError("DeepEP replay requires distributed BF16 inputs")
        baseline = DeepEPLegacyForward(
            group, x.size(0), x.size(1), e, None, weights[:e], weights[e:], 0
        )

        if world == 16 and os.environ.get("DEEPEP_NUM_MAX_ICN_PEERS") == "16":
            topology = (baseline.buffer.local_ranks, baseline.buffer.runtime.get_num_rdma_ranks())
            topologies = [None] * world
            dist.all_gather_object(topologies, topology)
            if any(t != (16, 1) for t in topologies):
                raise ValueError(f"DeepEP expected one 16-rank ICN domain, got {topologies}")

        def deepep_forward():
            with torch.set_grad_enabled(grad_enabled):
                if x.size(0) == 0:
                    # DeepEP interprets null data pointers as an absent optional input.
                    # An all-dropped row keeps zero dispatched routes on this source.
                    pad = torch.nn.functional.pad
                    result = DeepTraining.apply(
                        pad(x, (0, 0, 0, 1)),
                        pad(ids, (0, 0, 0, 1), value=-1),
                        pad(probs, (0, 0, 0, 1)),
                        baseline,
                        *weights,
                    )
                    return result[:0]
                return DeepTraining.apply(
                    x.contiguous(), ids.contiguous(), probs.contiguous(), baseline, *weights
                )

        forward = deepep_forward

    y = forward()
    backend = type(y.grad_fn).__name__ if y.grad_fn is not None else None
    if (args.expect_ep8_training or args.profile_ep16_path) and backend != "_TrainBackward":
        raise AssertionError(f"expected fused EP{world} training, got {backend}")
    actual = (y.detach(), *backward(y))
    errors = torch.stack(
        [
            (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
            for a, b in zip(actual, expected)
        ]
    ).max()
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    drift = errors.item()
    if not drift < 0.012:
        raise AssertionError(f"reference relative L2 error {drift}")
    del ref, expected, y, actual
    a, b, c = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    samples = []
    for i in range(args.warmup + args.iters):
        dist.barrier()
        torch.cuda.synchronize()
        a.record()
        y = forward()
        b.record()
        grads = backward(y)
        c.record()
        c.synchronize()
        if i >= args.warmup:
            times = torch.tensor(
                [
                    a.elapsed_time(b),
                    b.elapsed_time(c) if training else 0.0,
                    a.elapsed_time(c) if training else a.elapsed_time(b),
                ],
                device=device,
            )
            dist.all_reduce(times, op=dist.ReduceOp.MAX)
            samples.append(times.tolist())
        del grads, y
    if args.profile_stages:
        from mega_moe_kernel_profile import profile_steps

        profile = profile_steps(forward, backward, training, device, e)
    if args.profile_ep16_path:
        from mega_moe_kernel_profile import profile_ep16_call

        paths = profile_ep16_call(forward, backward)
    if rank == 0:
        result = {
            "tag": tag,
            "world": world,
            "backend": args.backend,
            "relative_l2": drift,
            "backward": training,
            "autograd_node": backend,
            "samples_ms": samples,
            "median_ms": dict(
                zip(("forward", "backward", "total"), map(statistics.median, zip(*samples)))
            ),
        }
        if args.profile_stages:
            result["kernel_profile"] = profile
        if args.profile_ep16_path:
            result["rank_paths"] = paths
        if "expert_rows" in shard:
            result["rows_per_global_expert"] = shard["expert_rows"]
        print(json.dumps(result), flush=True)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    if baseline is not None:
        baseline.buffer.destroy()


def run_replays(cases):
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    try:
        for args in cases:
            # A capture may change routing flags, and DeepEP forces TE RTC on its arm.
            with patch.dict(os.environ):
                replay(args, device)
    finally:
        dist.destroy_process_group()


def main():
    run_replays([parse_args()])


if __name__ == "__main__":
    main()
