# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare bulk EP2 push/pull against the public collective path on the same PPUs."""

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.pytorch import _mega_moe_peer as peer


def _case(rank, tokens, topk, hidden=256, intermediate=128, experts=2, dtype=torch.bfloat16):
    # Exercise the source strides of X, IDs and probabilities and transposed weights.
    device = torch.device("cuda", rank)

    def sample(rows, cols, precision, scale=1.0):
        data = torch.randn((rows, 2 * cols), device=device).mul_(scale).to(precision)
        return data[:, ::2].detach().requires_grad_()

    x = sample(tokens, hidden, dtype)
    ids = torch.randint(2 * experts, (tokens, topk * 2), device=device, dtype=torch.int32)[:, ::2]
    ids[::3, 1] = ids[::3, 0]
    ids[1::4, 0] = -1
    probs = sample(tokens, topk, torch.float32, 0.5)
    with torch.no_grad():
        probs[::7, 0] = 0
    w1 = [
        sample(hidden, 2 * intermediate, dtype, 0.04).T.detach().requires_grad_()
        for _ in range(experts)
    ]
    w2 = [
        sample(intermediate, hidden, dtype, 0.08).T.detach().requires_grad_()
        for _ in range(experts)
    ]
    return x, ids, probs, w1, w2


def _run(inputs):
    fn = te.bf16_mega_moe if inputs[0].dtype == torch.bfloat16 else te.fp16_mega_moe
    return fn(*inputs, ep_group=dist.group.WORLD)


def _compare(inputs, *, gradients=False, profile=False, fallback=False):
    x, _, probs, w1, w2 = inputs
    dy = torch.randn_like(x)

    def evaluate():
        y = _run(inputs)
        if not gradients:
            return (y,)
        return (y, *torch.autograd.grad(y, (x, probs, *w1, *w2), dy))

    with patch.object(peer, "_peer_supported", return_value=False):
        expected = evaluate()
    if profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            actual = evaluate()
            torch.cuda.synchronize()
        events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        kernels = [e.name for e in events]
        for name in ("_bulk_push", "_bulk_pull"):
            assert any(name in kernel for kernel in kernels), (name, kernels)
        if gradients:
            # The public gradient uses a stable per-source slot prefix, including
            # repeated expert IDs, and writes both gradients back through the peer mapping.
            assert any("_backward_positions" in name for name in kernels), kernels
            assert not any("AllToAll" in name or "all_to_all" in name for name in kernels), kernels
            compute = {
                e.device_resource_id for e in events if "mega_moe" in e.name and "wgrad" in e.name
            }
            returns = {e.device_resource_id for e in events if "_backward_pull" in e.name}
            assert compute and returns and compute.isdisjoint(returns), (compute, returns)
    else:
        actual = evaluate()
    for observed, reference in zip(actual, expected):
        # Ineligible shapes use the original scalar epilogue, whose GEMM
        # accumulation order can differ at a BF16 rounding boundary.
        if fallback:
            torch.testing.assert_close(observed, reference, rtol=0.016, atol=0.002)
        else:
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)


def _check_streams(inputs):
    x, ids, probs, w1, w2 = inputs
    with torch.no_grad():
        calls = [
            (
                (x * scale).requires_grad_(),
                torch.where(ids >= 0, (ids + i) % 4, ids),
                (probs * (-1) ** i).requires_grad_(),
                w1,
                w2,
            )
            for i, scale in enumerate((1.0, -1.0, 2.0, -2.0))
        ]
        dy = torch.randn_like(x)

    def evaluate(y, args, retain=False):
        return (y, *torch.autograd.grad(y, (args[0], args[2], *w1, *w2), dy, retain_graph=retain))

    with patch.object(peer, "_peer_supported", return_value=False):
        references = [evaluate(_run(args), args) for args in calls]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    results = []
    for i, args in enumerate(calls):
        with torch.cuda.stream(streams[i % 2]):
            results.append(evaluate(_run(args), args, retain=True))
    # The two caller streams alternately reuse the workspace and its return
    # stream while the earlier graphs and their gradient results stay live.
    for i in reversed(range(len(calls))):
        with torch.cuda.stream(streams[i % 2]):
            results.append(evaluate(results[i][0], calls[i]))
    torch.cuda.synchronize()
    for result, expected in zip(results, references + references[::-1]):
        for observed, reference in zip(result, expected):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    # Reuse the original tensor objects with changed routing and probabilities.
    with torch.no_grad():
        ids.copy_(calls[-1][1])
        probs.mul_(2)
        x.neg_()
        _compare(inputs)


def _worker():
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=80))
    try:
        if not peer._peer_supported(dist.group.WORLD, device):
            print("Skipping bulk EP2: peer access unavailable", flush=True)
            return
        torch.manual_seed(5901 + rank)
        tokens, topk = (2049, 2) if rank == 0 else (1025, 3)
        inputs = _case(rank, tokens, topk)
        _compare(inputs, gradients=True, profile=True)
        _check_streams(inputs)
        # Activation and output-gradient transport also accepts FP16; one source
        # has no tokens while its experts still receive routes from the other rank.
        _compare(_case(rank, 2049 if rank == 0 else 0, 2, dtype=torch.float16), gradients=True)

        # Only rank 1 has tokens, all experts they select belong to rank 0.
        empty = _case(rank, 0 if rank == 0 else 2050, 2)
        empty[1].fill_(0)
        _compare(empty, gradients=True)
        empty[1].fill_(-1)
        _compare(empty, gradients=True)

        # Fast-path admission must agree even when only one rank's local
        # intermediate width or top-k requires the generic path.
        _compare(_case(rank, tokens, topk, intermediate=129 if rank else 128), fallback=True)
        _compare(_case(rank, tokens, 9 if rank else topk), fallback=True)

        for kwargs in (
            {"hidden": 264 if rank else 256},
            {"experts": 3 if rank else 2},
            {"dtype": torch.float16 if rank else torch.bfloat16},
        ):
            mismatched = _case(rank, tokens, topk, **kwargs)
            mismatched[1].zero_()
            with pytest.raises(ValueError):
                _run(mismatched)
        for invalid in (-2, 4):
            if rank == 1:
                inputs[1][0, 0] = invalid
            with pytest.raises(ValueError):
                _run(inputs)
        inputs[1][0, 0] = 1
        _compare(inputs)
    finally:
        dist.destroy_process_group()


def test_bulk_peer_transport():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Two PPU devices required")
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("SM89 PPU required")
    pytest.importorskip("triton")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=320,
    )
    assert result.returncode == 0, result.stdout
    if "Skipping bulk EP2" in result.stdout:
        pytest.skip("Two mutually addressable PPUs required")


if __name__ == "__main__":
    _worker()
