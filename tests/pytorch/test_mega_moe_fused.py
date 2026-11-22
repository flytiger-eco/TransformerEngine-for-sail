# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Exercise the three-kernel BF16 peer path, including workspace transitions."""

import os
import subprocess
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.pytorch import _mega_moe_peer as peer


def _weights(rank, width):
    # Independent shifted diagonals give each projection a known result and
    # exercise the gate, value, and output weight addresses independently.
    col = torch.arange(width, device="cuda")
    w1, w2 = [], []
    for expert in (2 * rank, 2 * rank + 1):
        first = torch.zeros((2 * width, width), dtype=torch.bfloat16, device="cuda")
        second = torch.zeros_like(first[:width])
        first[col, (col - expert - 3) % width] = (expert + 1) / 4
        first[width + col, (col + expert + 1) % width] = 1 + expert / 4
        second[col, (col - expert - 7) % width] = 1 - expert / 8
        w1.append(first.requires_grad_())
        w2.append(second.requires_grad_())
    return w1, w2


def _inputs(tokens, width):
    def sample(shape, dtype):
        return torch.randn(shape, device="cuda", dtype=dtype).requires_grad_()

    x = sample((tokens, width), torch.bfloat16)
    # Includes duplicate remote slots, duplicate local slots, all-dropped
    # tokens, mixed local/remote slots, and an expert with no rows.
    pattern = torch.tensor([[2, 2], [0, 0], [-1, -1], [2, 1], [-1, 0]], device="cuda")
    ids = pattern.repeat((tokens + 4) // 5, 1)[:tokens].int()
    probs = sample((tokens, 2), torch.float32)
    with torch.no_grad():
        probs[::7, 0] = 0
    return x, ids, probs


def _check_output(output, x, ids, probs):
    x = x.float()
    slots = torch.zeros((2, *x.shape), device=x.device)
    for expert in range(4):
        gate = (x.roll(expert + 3, 1) * ((expert + 1) / 4)).bfloat16().float()
        up = (x.roll(-expert - 1, 1) * (1 + expert / 4)).bfloat16().float()
        for k in range(2):
            activation = (F.silu(gate) * up * probs[:, k, None]).bfloat16().float()
            result = (
                (activation.roll(expert + 7, 1) * (1 - expert / 8)).bfloat16().float()
            )
            slots[k] += torch.where(ids[:, k, None] == expert, result, 0.0)
    expected = slots.sum(0).bfloat16().float()
    difference = output.float() - expected
    # An activation rounded one BF16 unit differently can survive cancellation
    # of opposite-signed routes. Bound rounding both before and after the sum.
    tolerance = torch.finfo(torch.bfloat16).eps * (slots.abs().sum(0) + expected.abs())
    excess = (difference.abs() - tolerance).clamp_min(0)
    torch.testing.assert_close(excess, torch.zeros_like(excess), atol=0, rtol=0)
    assert difference.norm() <= 0.001 * expected.norm()


def _forward(args, weights):
    return te.bf16_mega_moe(*args, *weights, ep_group=dist.group.WORLD)


def _check(args, weights):
    with torch.no_grad():
        result = _forward(args, weights)
        _check_output(result, *args)
        return result


def _check_fc2_publication(rank):
    # At this shape each expert spans 18 FC2 M tiles. A source can already be
    # combining the first half of H while the peer projects the second half.
    tokens, hidden, intermediate = 17959, 2048, 512
    device = torch.device("cuda", rank)
    generator = torch.Generator(device=device).manual_seed(991 + rank)
    per_source = [2245] * 6 + ([2245, 2244] if rank == 0 else [2244, 2245])
    pool = torch.arange(16, dtype=torch.int32, device=device).repeat_interleave(
        torch.tensor(per_source * 2, device=device)
    )
    w1 = torch.zeros(2 * intermediate, hidden, dtype=torch.bfloat16, device=device)
    w2 = torch.zeros(hidden, intermediate, dtype=torch.bfloat16, device=device)
    w1[:intermediate, 0] = 1
    w1[intermediate:, 1] = 1
    w2[:, 0] = 1
    x = torch.zeros(tokens, hidden, dtype=torch.bfloat16, device=device)
    x[:, 0] = 64  # SiLU(64) rounds to exactly 64 in FP32.
    ids = pool.new_empty(tokens, 2)
    probs = torch.empty(tokens, 2, device=device)
    for _ in range(4):
        ids.copy_(
            pool[torch.randperm(2 * tokens, generator=generator, device=device)].view(
                -1, 2
            )
        )
        x[:, 1] = torch.randint(1, 128, (tokens,), device=device, generator=generator)
        probs.uniform_(0.25, 1, generator=generator)
        out = te.bf16_mega_moe(
            x, ids, probs, [w1] * 8, [w2] * 8, ep_group=dist.group.WORLD
        )
        routed = (64 * x[:, 1:2].float() * probs).bfloat16()
        expected = routed.float().sum(1, keepdim=True).bfloat16()
        torch.testing.assert_close(out, expected.expand_as(out), rtol=0, atol=0)


def _worker():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=120))
    try:
        if not (
            peer._peer_supported(dist.group.WORLD, device)
            and peer._workspace(dist.group.WORLD).fused_supported
        ):
            print(
                "Skipping fused MegaMoE: symmetric peer access unavailable", flush=True
            )
            dist.destroy_process_group()
            return
        torch.manual_seed(743 + rank)
        width = 512  # Multiple output tiles for both projections.
        weights = _weights(rank, width)
        args = _inputs((2305, 117)[rank], width)
        x, ids, probs = args
        _check(
            args, weights
        )  # Allocate, rendezvous, and compile before counting launches.

        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as trace:
            output = _forward(args, weights)
            torch.cuda.synchronize()
        launches = [
            e.name
            for e in trace.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert launches == [
            "nvte_mega_moe_fused_pre_dispatch",
            "nvte_mega_moe_fused_l1",
            "nvte_mega_moe_fused_l2",
        ], launches

        dy = torch.randn_like(output)
        params = (x, probs, *weights[0], *weights[1])
        grads = torch.autograd.grad(output, params, dy)
        with patch.object(peer, "_peer_supported", return_value=False):
            expected_grads = torch.autograd.grad(_forward(args, weights), params, dy)
        for grad, expected in zip(grads, expected_grads, strict=True):
            torch.testing.assert_close(grad, expected, atol=0, rtol=0)
        assert torch.count_nonzero(grads[1][ids < 0]) == 0

        with torch.no_grad():
            _check(tuple(t.T.contiguous().T for t in args), weights)
            # One rank's weight strides alone force the entire group to use
            # the generic peer path. The next call can use fused storage again.
            strided_weights = [[w.T.contiguous().T for w in layer] for layer in weights]
            _check(args, strided_weights if rank == 1 else weights)
            cases = [
                (
                    x * (step + 1),
                    torch.where(ids >= 0, (ids + step) % 4, ids),
                    probs / (step + 1),
                )
                for step in range(3)
            ]
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            for stream in streams:
                stream.wait_stream(torch.cuda.current_stream())
            results = []
            for i, inputs in enumerate(cases):
                with torch.cuda.stream(streams[i % 2]):
                    results.append(_forward(inputs, weights))
            torch.cuda.synchronize()
            for inputs, result in zip(cases, results, strict=True):
                _check_output(result, *inputs)
            # Mutating the same objects must rebuild slot lists and invalidate
            # dropped entries despite reusing the symmetric allocation.
            ids.copy_(cases[-1][1])
            probs.neg_()
            _check(args, weights)
            if rank == 1:
                ids[0, 0] = 4
            with pytest.raises(ValueError):
                _forward(args, weights)
            ids[0, 0] = 1
            _check(_inputs(13, width), weights)  # Small fallback between fused calls.
            large = _inputs((0, 8193)[rank], width)  # Grow on just one source.
            large[1].zero_()  # All routes go from rank 1 to expert 0.
            with patch.object(
                peer.symm,
                "empty",
                wraps=peer.symm.empty,
                side_effect=(
                    torch.OutOfMemoryError("simulated peer allocation failure")
                    if rank == 1
                    else None
                ),
            ), pytest.raises(torch.OutOfMemoryError):
                _forward(large, weights)
            _check(args, weights)  # The failed resize retained the smaller mapping.
            _check(large, weights)
            large[1].fill_(-1)
            _check(large, weights)
            _check(args, weights)
            # Three FC2 column tiles split into two tiles and one tile.
            _check(_inputs((2305, 117)[rank], 768), _weights(rank, 768))
            _check_fc2_publication(rank)
    except Exception:
        # torchrun terminates the peer after this worker exits. Collective
        # teardown can otherwise wait for a peer still inside the failed step.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    else:
        dist.destroy_process_group()


def test_fused_three_kernel_peer_forward():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Two PPUs required")
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("SM89 PPU required")
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
    if "Skipping fused MegaMoE" in result.stdout:
        pytest.skip("Two mutually addressable PPUs required")


if __name__ == "__main__":
    _worker()
