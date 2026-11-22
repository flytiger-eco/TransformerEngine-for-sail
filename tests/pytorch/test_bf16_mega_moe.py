# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare the routed RTC operator and its gradients against an independent CPU graph."""

import os
import subprocess
import sys
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

import transformer_engine.pytorch as te


DTYPES = (torch.bfloat16, torch.float16)


def _run(dtype):
    return te.bf16_mega_moe if dtype == torch.bfloat16 else te.fp16_mega_moe


def _inputs(ids, dtype, experts, hidden=37, intermediate=19):
    """Keep the same full, deterministic reference inputs on all ranks."""
    rng = torch.Generator().manual_seed(314159)

    def sample(shape, scale=1):
        return (torch.randn(shape, generator=rng) * scale).to(dtype)

    tokens = ids.size(0)
    x = sample((tokens, hidden))
    w1 = [sample((2 * intermediate, hidden), max(1, hidden) ** -0.5) for _ in range(experts)]
    w2 = [sample((hidden, intermediate), max(1, intermediate) ** -0.5) for _ in range(experts)]
    probs = torch.randn(ids.shape, generator=rng)
    if probs.numel():
        probs[0, 0] = 0  # A selected route with zero score still has a score gradient.
    dy = sample((tokens, hidden))
    return x, probs, w1, w2, dy


def _reference(ids, x, probs, w1, w2, dy):
    """FP32 CPU GEMMs with rounding at the routed operator's storage boundaries."""
    dtype = x.dtype
    x, probs, *weights = [t.float().detach().requires_grad_() for t in (x, probs, *w1, *w2)]
    result = torch.zeros_like(x)
    for expert, (fc1, fc2) in enumerate(zip(weights[: len(w1)], weights[len(w1) :])):
        rows, slots = torch.where(ids == expert)
        routed_x = x[rows].to(dtype).float()  # Each route returns its dX in the input dtype.
        gate, up = (routed_x @ fc1.T).to(dtype).float().chunk(2, dim=1)
        hidden = (F.silu(gate) * up * probs[rows, slots, None]).to(dtype).float()
        routed = (hidden @ fc2.T).to(dtype).float()
        result = result.index_add(0, rows, routed)
    result = result.to(dtype)
    grads = torch.autograd.grad(result, (x, probs, *weights), dy)
    return result.detach(), [g if i == 1 else g.to(dtype) for i, g in enumerate(grads)]


def _device(tensor, *, grad=False):
    # Noncontiguous x, router tensors, original expert parameters and dY exercise
    # the layout contract without changing the logical shape or values.
    value = tensor.T.contiguous().to("cuda").T
    return value.requires_grad_(grad)


def _check(
    ids,
    dtype,
    experts,
    *,
    token_slice=slice(None),
    expert_slice=slice(None),
    group=None,
    hidden=37,
    intermediate=19
):
    x, probs, w1, w2, dy = _inputs(ids, dtype, experts, hidden, intermediate)
    y_ref, (dx, dp, *dw) = _reference(ids, x, probs, w1, w2, dy)
    local_dx = dx[token_slice]
    local_dp = dp[token_slice]
    local_dw = dw[:experts][expert_slice] + dw[experts:][expert_slice]
    local_x = _device(x[token_slice], grad=True)
    local_p = _device(probs[token_slice], grad=True)
    local_w1 = [_device(w, grad=True) for w in w1[expert_slice]]
    local_w2 = [_device(w, grad=True) for w in w2[expert_slice]]
    y = _run(dtype)(local_x, _device(ids[token_slice]), local_p, local_w1, local_w2, ep_group=group)
    grads = torch.autograd.grad(
        y, (local_x, local_p, *local_w1, *local_w2), _device(dy[token_slice])
    )
    # Derivatives of SwiGLU and GEMMs may round one intermediate differently in
    # a fused backward. Compare each value (including every unused parameter).
    tol = dict(rtol=0.035, atol=0.025) if dtype == torch.bfloat16 else dict(rtol=0.005, atol=0.004)
    torch.testing.assert_close(y.cpu(), y_ref[token_slice], **tol)
    for actual, expected in zip(grads, (local_dx, local_dp, *local_dw), strict=True):
        torch.testing.assert_close(actual.cpu(), expected, **tol)
    # Check dropped and zero-weight routes exactly even for small score gradients.
    if ids[token_slice].numel():
        unused = ids[token_slice] == -1
        assert torch.count_nonzero(grads[1].cpu()[unused]) == 0
        assert torch.count_nonzero(y.cpu()[unused.all(dim=1)]) == 0
        assert torch.count_nonzero(grads[0].cpu()[unused.all(dim=1)]) == 0
        valid_zero = (probs[token_slice] == 0) & ~unused & (local_dp.abs() > 1e-4)
        assert torch.all(grads[1].cpu()[valid_zero] != 0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tokens", [0, 3])
@pytest.mark.parametrize("fake", [False, True], ids=["meta", "fake"])
def test_training_shapes(dtype, tokens, fake):
    with FakeTensorMode() if fake else nullcontext():
        device = "cpu" if fake else "meta"

        def param(*shape):
            return torch.empty(*shape, dtype=dtype, device=device, requires_grad=True)

        x = param(tokens, 16)
        ids = torch.empty(tokens, 2, device=device, dtype=torch.int32)
        p = torch.empty_like(ids, dtype=torch.float32, requires_grad=True)
        fc1 = [param(24, 16) for _ in range(2)]
        fc2 = [param(16, 12) for _ in range(2)]
        y = _run(dtype)(x, ids, p, fc1, fc2)
        assert y.shape == x.shape and y.dtype == dtype
        tensors = (x, p, *fc1, *fc2)
        for tensor, grad in zip(tensors, torch.autograd.grad(y.sum(), tensors), strict=True):
            assert grad.shape == tensor.shape and grad.dtype == tensor.dtype


def test_rejects_mismatched_weights():
    x = torch.empty(2, 16, dtype=torch.bfloat16, device="meta")
    ids = torch.zeros(2, 1, dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="weight"):
        w1 = torch.empty(4, 16, device="meta", dtype=torch.float32)
        te.bf16_mega_moe(x, ids, ids.float(), [w1], [x.T])


def _require_sm89():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("RTC MegaMoE requires a SM89 PPU")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scenario", ["mixed", "multiple_tiles", "all_dropped", "no_tokens"])
def test_rtc_training(dtype, scenario):
    _require_sm89()
    ids = torch.tensor([[1, 1, -1], [2, -1, 1], [-1, -1, -1], [0, 2, 0]], dtype=torch.int32)
    if scenario == "multiple_tiles":
        ids = ids.repeat(9, 1)
    elif scenario == "all_dropped":
        ids.fill_(-1)
    elif scenario == "no_tokens":
        ids = ids[:0]
    _check(ids, dtype, 4)


@pytest.mark.parametrize("hidden,intermediate", [(0, 7), (5, 0)])
def test_empty_feature_dimension(hidden, intermediate):
    _require_sm89()
    ids = torch.tensor([[0, 1], [-1, 1]], dtype=torch.int32)
    _check(ids, torch.bfloat16, 2, hidden=hidden, intermediate=intermediate)


@pytest.mark.parametrize("dtype,expected", [(torch.bfloat16, 0.375), (torch.float16, -0.015625)])
def test_route_weight_is_rounded_before_fc2(dtype, expected):
    _require_sm89()
    # Without routing, 32*33 - 33*32 cancels exactly. Multiplying the two
    # activations by 0.23 and rounding before FC2 gives a nonzero result.
    x = torch.ones(1, 1, dtype=dtype, device="cuda")
    fc1 = torch.tensor([[32], [32], [1], [1.03125]], dtype=dtype, device="cuda")
    fc2 = torch.tensor([[33, -32]], dtype=dtype, device="cuda")
    ids = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    probs = torch.full((1, 1), 0.23, dtype=torch.float32, device="cuda")
    output = _run(dtype)(x, ids, probs, [fc1], [fc2])
    assert output.item() == expected


@pytest.mark.parametrize("dtype", DTYPES)
def test_combines_routes_in_fp32(dtype):
    _require_sm89()
    # These three expert outputs sum to exactly one. Summing in the storage
    # dtype loses the unit term for at least one of the route permutations.
    x = torch.ones((3, 1), device="cuda", dtype=dtype)
    fc1 = [torch.tensor([[32], [1]], device="cuda", dtype=dtype) for _ in range(3)]
    fc2 = [torch.full((1, 1), v / 32, device="cuda", dtype=dtype) for v in (8192, 1, -8192)]
    ids = torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=torch.int32, device="cuda")
    y = _run(dtype)(x, ids, torch.ones_like(ids, dtype=torch.float32), fc1, fc2)
    torch.testing.assert_close(y, x, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_uses_rtc_kernels(dtype):
    _require_sm89()
    ids = torch.tensor([[1], [0]], dtype=torch.int32)
    x, probs, w1, w2, dy = _inputs(ids, dtype, 2)
    x, probs, *weights = [_device(t, grad=True) for t in (x, probs, *w1, *w2)]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        y = _run(dtype)(x, _device(ids), probs, weights[:2], weights[2:])
        torch.autograd.grad(y, (x, probs, *weights), _device(dy))
        torch.cuda.synchronize()
    _assert_rtc_symbols(prof, "fc1", "fc2", "combine", "dgrad", "wgrad", "swiglu_backward")


def _assert_rtc_symbols(prof, *names):
    kernels = {e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA}
    for name in names:
        symbol = "nvte_mega_moe_" + name
        assert any(symbol in kernel for kernel in kernels), (symbol, kernels)


def _check_peer_streams(ep, rank):
    # Reuse the same workspace shape with different probabilities on two streams.
    # Deferred reads of peer activations or writes into recycled output slots can
    # otherwise overwrite a previous call whose kernels are still executing.
    ids = torch.tensor([[2, 0], [1, 3], [1, 1], [3, -1]], dtype=torch.int32).repeat(17, 1)
    x, probs, w1, w2, dy = _inputs(ids, torch.float16, 4)
    reference, _ = _reference(ids, x, probs, w1, w2, dy)
    rows, experts = slice(34 * rank, 34 * (rank + 1)), slice(2 * rank, 2 * rank + 2)
    x, ids, probs = [_device(t[rows]) for t in (x, ids, probs)]
    w1, w2 = [[_device(w) for w in group[experts]] for group in (w1, w2)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    results = []
    for i, scale in enumerate((1, -1, 2, -2)):
        with torch.cuda.stream(streams[i % 2]):
            results.append(te.fp16_mega_moe(x, ids, probs * scale, w1, w2, ep_group=ep))
    torch.cuda.synchronize()
    for actual, scale in zip(results, (1, -1, 2, -2), strict=True):
        torch.testing.assert_close(actual.cpu(), reference[rows] * scale, rtol=0.005, atol=0.004)


def _ep_worker():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        from transformer_engine.pytorch import _mega_moe_peer as peer

        ep = dist.new_group([0, 1])
        supports_peer = peer._peer_supported(ep, torch.device("cuda", rank))
        token_slice = (slice(0, 3), slice(3, 5))[rank]
        expert_slice = slice(2 * rank, 2 * rank + 2)
        for dtype in DTYPES:
            # Uneven source token counts, duplicate and dropped slots, and routes
            # in both directions. Expert 3 receives no work.
            ids = torch.tensor([[2, 2], [1, -1], [-1, 2], [0, -1], [1, 0]], dtype=torch.int32)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                _check(ids, dtype, 4, token_slice=token_slice, expert_slice=expert_slice, group=ep)
                torch.cuda.synchronize()
            if supports_peer:
                _assert_rtc_symbols(prof, "peer_fc1", "peer_fc2", "combine")
            else:
                _assert_rtc_symbols(prof, "fc1", "fc2")
            with patch.object(peer, "_peer_supported", return_value=False):
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    _check(
                        ids, dtype, 4, token_slice=token_slice, expert_slice=expert_slice, group=ep
                    )
                    torch.cuda.synchronize()
                _assert_rtc_symbols(prof, "fc1", "fc2")
            # Rank 0 owns the only active expert and has zero source tokens;
            # rank 1 owns both source tokens and has zero local expert work.
            ids = torch.tensor([[0, -1], [0, 0]], dtype=torch.int32)
            _check(
                ids,
                dtype,
                4,
                token_slice=(slice(0, 0), slice(None))[rank],
                expert_slice=slice(2 * rank, 2 * rank + 2),
                group=ep,
            )
        if supports_peer:
            _check_peer_streams(ep, rank)
        # Every rank has only dropped routes, so all NCCL payloads are empty.
        _check(
            ids.fill_(-1),
            torch.float16,
            4,
            token_slice=slice(rank, rank + 1),
            expert_slice=slice(2 * rank, 2 * rank + 2),
            group=ep,
        )
        # Both ranks execute private EP groups. Rank 1's local expert 0 must
        # resolve within its group even though its global process rank is 1.
        singletons = [dist.new_group([r]) for r in range(2)]
        ids = torch.tensor([[0, -1]] * (rank + 1), dtype=torch.int32)
        _check(ids, torch.bfloat16, 1, group=singletons[rank])
    finally:
        dist.destroy_process_group()


def test_two_rank_expert_parallel():
    _require_sm89()
    if torch.cuda.device_count() < 2:
        pytest.skip("Expert parallel test requires two PPUs")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
        ],
        check=True,
        timeout=300,
    )


if __name__ == "__main__":
    _ep_worker()
