# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Exercise routed gradient GEMMs, including beta=1 and empty expert segments."""

import pytest
import torch
from torch.nn import functional as F

import transformer_engine.pytorch as te
import transformer_engine_torch as tex


@pytest.fixture(autouse=True)
def _sm89():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("Requires an SM89 PPU")


def _assert_gradient(actual, expected, tolerance):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if not expected.numel() or torch.count_nonzero(expected) == 0:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        return
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert error < tolerance, error.item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_expert_gradient_segments_and_accumulation(dtype):
    torch.manual_seed(149)
    in_features, out_features = 512, 256
    # 33 experts crosses the pointer-table window. The last window has real
    # work and several consecutive experts in the first window are empty.
    counts = [0, 3, 513] + [0] * 28 + [61, 577]
    rows = sum(counts)
    offsets = torch.tensor([0, *counts], device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    x = torch.randn(rows, in_features, device="cuda", dtype=dtype)
    dy = torch.randn(rows, out_features, device="cuda", dtype=dtype)
    weights = [torch.randn(out_features, in_features, device="cuda", dtype=dtype) for _ in counts]
    dx = tex.mega_moe_dgrad(dy, offsets, weights)
    reference = torch.cat(
        [(g.float() @ w.float()).to(dtype) for g, w in zip(dy.split(counts), weights)]
    )
    tolerance = 0.006 if dtype == torch.bfloat16 else 0.001
    _assert_gradient(dx, reference, tolerance)

    for target_dtype in (dtype, torch.float32):
        targets = [torch.randn(w.shape, device="cuda", dtype=target_dtype) for w in weights]
        products = [g.float().T @ a.float() for g, a in zip(dy.split(counts), x.split(counts))]
        for accumulate in (True, False, True):
            expected = [
                (t.float() + prod if accumulate else prod).to(target_dtype)
                for t, prod in zip(targets, products)
            ]
            tex.mega_moe_wgrad(dy, x, offsets, targets, accumulate)
            for actual, want in zip(targets, expected):
                _assert_gradient(
                    actual, want, tolerance if target_dtype != torch.float32 else 0.0001
                )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_routed_training_backward_against_independent_autograd(dtype):
    torch.manual_seed(271)
    tokens, hidden, intermediate, experts = 1213, 512, 256, 4
    ids = torch.randint(experts - 1, (tokens, 2), dtype=torch.int32, device="cuda")
    ids[::13, :] = -1
    ids[1::7, 1] = ids[1::7, 0]
    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype).requires_grad_()
    probs = torch.rand(tokens, 2, device="cuda")
    probs[2::9, 0] = 0
    probs.requires_grad_()
    w1 = [
        (torch.randn(2 * intermediate, hidden, device="cuda") / hidden**0.5)
        .to(dtype)
        .requires_grad_()
        for _ in range(experts)
    ]
    w2 = [
        (torch.randn(hidden, intermediate, device="cuda") / intermediate**0.5)
        .to(dtype)
        .requires_grad_()
        for _ in range(experts)
    ]
    params = (x, probs, *w1, *w2)
    dy = torch.randn(hidden, tokens, dtype=dtype, device="cuda").T
    fn = te.bf16_mega_moe if dtype == torch.bfloat16 else te.fp16_mega_moe
    actual = torch.autograd.grad(fn(x, ids, probs, w1, w2), params, dy)

    # Duplicate the token-to-slot view before differentiating so dX sums
    # rounded route gradients in FP32, just as the public operator specifies.
    route_x = x.detach().float()[:, None].expand(-1, 2, -1).contiguous().requires_grad_()
    result = torch.zeros(tokens, hidden, device=x.device)
    for e, (f1, f2) in enumerate(zip(w1, w2)):
        row, slot = torch.where(ids == e)
        z = (route_x[row, slot].to(dtype).float() @ f1.float().T).to(dtype).float()
        gate, up = z.chunk(2, dim=1)
        act = (F.silu(gate) * up * probs[row, slot, None]).to(dtype)
        result = result.index_add(0, row, (act.float() @ f2.float().T).to(dtype).float())
    ref = torch.autograd.grad(result.to(dtype), (route_x, *params[1:]), dy)
    for value, want in zip(actual, (ref[0].sum(1).to(dtype), *ref[1:])):
        _assert_gradient(value, want, 0.010 if dtype == torch.bfloat16 else 0.003)
    assert not torch.count_nonzero(actual[1][ids == -1])
    assert torch.count_nonzero(actual[1][(ids >= 0) & (probs == 0)])


def test_accumulating_an_empty_batch_preserves_existing_main_grad():
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
    dy = x[:, :128].contiguous()
    offsets = torch.zeros(3, device="cuda", dtype=torch.int32)
    targets = [torch.randn(128, 256, device="cuda") for _ in range(2)]
    targets[0][0, 0] = float("nan")
    before = [t.clone() for t in targets]
    tex.mega_moe_wgrad(dy, x, offsets, targets, True)
    for actual, expected in zip(targets, before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    tex.mega_moe_wgrad(dy, x, offsets, targets, False)
    assert all(torch.count_nonzero(t) == 0 for t in targets)


def test_wgrad_checks_each_experts_reduction_length():
    torch.manual_seed(299)
    # Both partitions give the launcher a candidate K of 512. Only the
    # experts with exactly 512 rows can use its four-column pipeline.
    counts = [0, 509, 515, 1024, 512, 512, 512, 512]
    rows, out_features, in_features = sum(counts), 1024, 3072
    x = torch.randn(rows, in_features, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(rows, out_features, device="cuda", dtype=x.dtype)
    grads = [x.new_empty((out_features, in_features)) for _ in counts]
    # Check both sides of every N-tile boundary, including the boundary
    # between two four-column work items.
    n = torch.arange(0, in_features, 256, device="cuda")
    n = torch.cat((n, n + 255))
    m = torch.tensor([0, 255, 256, 511, 512, 767, 768, 1023], device="cuda")
    for splits in ([512] * len(counts), counts):
        offsets = torch.tensor([0, *splits], device="cuda", dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        tex.mega_moe_wgrad(dy, x, offsets, grads)
        for grad, g, a in zip(grads, dy.split(splits), x.split(splits)):
            ref = (g[:, m].float().T @ a[:, n].float()).to(x.dtype)
            _assert_gradient(grad[m[:, None], n], ref, 0.006)


@pytest.mark.parametrize("m", [1024, 2048])
def test_wgrad_uses_live_rows_with_padded_offsets(m):
    torch.manual_seed(303)
    # The 33rd expert uses a second pointer window. The nine N tiles also
    # exercise the remainder after two four-column work items.
    counts = [0, 64, 128, 192, 256] * 6 + [64, 192, 129]
    counts[2] = 129  # Exercise a partial K tile in both pointer windows.
    device = "cuda"
    # M=2048 exercises the compact BF16 overwrite kernel in the first window.
    n, stride = 2304, 256
    lengths = torch.tensor(counts, device=device, dtype=torch.int32)
    offsets = torch.arange(len(counts) + 1, device=device, dtype=torch.int32) * stride
    x = torch.randn(len(counts) * stride, n, device=device, dtype=torch.bfloat16)
    dy = torch.randn(len(counts) * stride, m, device=device, dtype=x.dtype)
    grads = [x.new_full((m, n), 1) for _ in counts]
    # Nonzero padding makes accidental reductions beyond the live rows observable.
    rows = torch.tensor([0, 127, 128, 255, 256, m - 1], device=device)
    columns = torch.arange(0, n, 256, device=device)
    columns = torch.cat((columns, columns + 255))
    for accumulate in (True, False):
        tex.mega_moe_wgrad(dy, x, offsets, grads, accumulate, lengths)
        for i, count in enumerate(counts):
            start = i * stride
            active = slice(start, start + count)
            ref = dy[active, rows].float().T @ x[active, columns].float()
            if accumulate:
                ref = ref + 1
            _assert_gradient(grads[i][rows[:, None], columns], ref.to(x.dtype), 0.006)


def test_backward_rejects_overlapping_outputs_with_separate_storage():
    rows, hidden, intermediate = 5, 256, 128
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    storage = torch.randn(rows + 1, hidden, dtype=x.dtype, device=x.device)
    dy = storage[:rows]
    probs = torch.rand(rows, device=x.device)
    offsets = torch.tensor([0, rows], dtype=torch.int32, device=x.device)
    w1 = [x.new_empty((2 * intermediate, hidden))]
    w2 = [x.new_empty((hidden, intermediate))]
    saved_dy, saved_probs = dy.clone(), probs.clone()

    # A DLPack rewrap has its own StorageImpl even for the identical pointer.
    for target in (dy, storage[1:]):
        out = torch.from_dlpack(target)
        assert out.data_ptr() == target.data_ptr()
        with pytest.raises(RuntimeError, match="overlaps"):
            tex.mega_moe_local_backward(dy, x, probs, offsets, w1, w2, out_dx=out)
    with pytest.raises(RuntimeError, match="overlaps"):
        tex.mega_moe_local_backward(dy, x, probs, offsets, w1, w2, out_dp=torch.from_dlpack(probs))
    ids = torch.zeros(rows, 2, dtype=torch.int32, device=x.device)
    with pytest.raises(RuntimeError, match="overlaps"):
        tex.mega_moe_bulk_pull(ids, ids, dy, x, 1, 0, out=torch.from_dlpack(dy))
    torch.testing.assert_close(dy, saved_dy, atol=0, rtol=0)
    torch.testing.assert_close(probs, saved_probs, atol=0, rtol=0)


def test_routed_activation_skips_padded_rows():
    from transformer_engine.pytorch._mega_moe_train import _activation_grad, _save_activation

    # At width 2304 a 2048-element flat block crosses live/padded rows in both
    # directions. The adjacent padding rows also contain a fully padded block.
    width = 2304
    route = torch.tensor([0, -1, -1, 1 << 32], dtype=torch.int64, device="cuda")
    live = route >= 0
    prob = torch.tensor([0.0, float("nan"), float("nan"), 0.75], device=route.device)
    torch.manual_seed(733)
    z = torch.randn(route.numel(), 2 * width, dtype=torch.bfloat16, device=route.device)
    dy = torch.randn(route.numel(), width, dtype=z.dtype, device=route.device)
    z[0] = dy[0] = 1
    z[~live] = dy[~live] = float("nan")

    def run(routes):
        y = torch.full_like(dy, float("nan"))
        dz = torch.full_like(z, float("nan"))
        dp = torch.full_like(prob, float("nan"))
        _save_activation(z, prob, None, y, routes)
        _activation_grad[(route.numel(),)](dy, z, prob, None, dz, dp, width, 4096, routes)
        return y, dz, dp

    expected = run(None)
    y, dz, dp = run(route)
    for value, reference in zip((y, dz, dp), expected):
        torch.testing.assert_close(value[live], reference[live], atol=0, rtol=0)
        torch.testing.assert_close(value[~live], torch.zeros_like(value[~live]), atol=0, rtol=0)
    # A valid zero-probability route still has a derivative with respect to its weight.
    assert dp[0] > 0
