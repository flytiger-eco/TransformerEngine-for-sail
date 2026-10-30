# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Correctness tests for the CUTLASS RTC grouped GEMM path (sm80/sm89).

The per-GEMM M of a GroupedLinear is the per-expert token count, which MoE
routing makes non-uniform in every real training step. These tests hold the
RTC path to a pure-torch fp32 reference over uniform AND non-uniform split
patterns; the reference itself is validated by the static (non-RTC) path on
the same wheel.

NVTE_FORCE_RTC_NO_FALLBACK=1 makes an RTC path that declines to run raise
instead of silently falling back to cuBLAS, which would let these tests pass
vacuously.
"""

import os

import pytest
import torch
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm

NUM_GEMMS = 16
RTOL = 0.05
ATOL = 0.05

_SPLIT_PATTERNS = [
    pytest.param([1024] * NUM_GEMMS, id="uniform_1024"),
    pytest.param([1024 if i % 2 == 0 else 1023 for i in range(NUM_GEMMS)],
                 id="off_by_one_1024_1023"),
    pytest.param([1024 if i % 2 == 0 else 768 for i in range(NUM_GEMMS)],
                 id="tile_multiples_1024_768"),
    pytest.param([1024] * 5 + [1023] + [1024] * 10,
                 id="single_deviation_group5_1023"),
    pytest.param([1015, 1029, 1022] * 5 + [1015],
                 id="mixed_partial_tiles"),
    pytest.param([950, 1024, 900, 1000, 850, 1024, 990, 810] * 2,
                 id="jagged_same_tile_counts"),
]

_MODULE_SHAPES = [
    pytest.param(2048, 1024, id="fc1_2048_1024"),
    pytest.param(512, 2048, id="fc2_512_2048"),
]


@pytest.fixture(autouse=True)
def _rtc_env(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cap = torch.cuda.get_device_capability()
    if cap not in ((8, 0), (8, 9)):
        pytest.skip(f"RTC grouped GEMM targets sm80/sm89, got sm{cap[0]}{cap[1]}")
    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "1")
    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM_RTC", "1")
    monkeypatch.setenv("NVTE_FORCE_RTC_NO_FALLBACK", "1")
    monkeypatch.setenv("NVTE_GG_ALLOW_FP32_ACCUM", "1")
    monkeypatch.setenv("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", "1")
    # A wheel install puts the headers in <site-packages>/include/cutlass3
    # while the shared object lives in <site-packages>/transformer_engine/,
    # so the runtime resolver's dladdr fallback cannot reach them.
    site = os.path.dirname(os.path.dirname(transformer_engine.__file__))
    candidate = os.path.join(site, "include", "cutlass3")
    # Prefer the effective headers materialized by configure when running from a
    # source tree, so a stale installed copy cannot mask a fix.
    effective = os.path.join(site, "build", "cmake", "actlize_effective", "include")
    if os.path.isfile(os.path.join(effective, "ppu_include.hpp")):
        candidate = effective
    if os.path.isfile(os.path.join(candidate, "cutlass", "cutlass.h")):
        monkeypatch.setenv("NVTE_CUTLASS3_INCLUDE_DIR", candidate)
    yield
    torch.cuda.synchronize()


def _reference(inp, weights, grad_out, splits):
    """Forward + backward in fp32, one plain matmul per group."""
    x = inp.detach().clone().float().requires_grad_(True)
    ws = [w.detach().clone().float().requires_grad_(True) for w in weights]
    outs, off = [], 0
    for w, m in zip(ws, splits):
        outs.append(x[off:off + m] @ w.t())
        off += m
    out = torch.cat(outs, 0)
    out.backward(grad_out.detach().clone().float())
    return out.detach(), x.grad.detach(), [w.grad.detach() for w in ws]


def _split_rows(tensor, splits):
    """Views of a (sum(splits), .) tensor, one per group."""
    views, off = [], 0
    for m in splits:
        views.append(tensor[off:off + m])
        off += m
    return views


def _assert_group_list_close(actual, expected, name):
    """Per-group comparison; the failure message names the wrong groups."""
    bad = []
    for group, (a, e) in enumerate(zip(actual, expected)):
        a, e = a.float(), e.float()
        tol = ATOL + RTOL * e.abs()
        n_bad = int(((a - e).abs() > tol).sum())
        if n_bad or not bool(torch.isfinite(a).all()):
            bad.append((group, tuple(a.shape), n_bad, float((a - e).abs().max())))
    assert not bad, (
        f"{name}: grouped GEMM result wrong in {len(bad)}/{len(actual)} groups "
        f"(group, shape, bad_elems, max_abs_err): {bad}"
    )


@pytest.mark.parametrize("in_features,out_features", _MODULE_SHAPES)
@pytest.mark.parametrize("splits", _SPLIT_PATTERNS)
def test_grouped_linear_rtc_nonuniform_m(splits, in_features, out_features):
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    dtype = torch.bfloat16

    module = te.GroupedLinear(
        NUM_GEMMS, in_features, out_features,
        bias=False, params_dtype=dtype,
    ).cuda()
    wparams = [getattr(module, "weight{}".format(i)) for i in range(NUM_GEMMS)]
    with torch.no_grad():
        for p in wparams:
            p.copy_(torch.randn(p.shape, device="cuda", dtype=dtype))
    weights = [p.detach().clone() for p in wparams]

    total = sum(splits)
    inp = torch.randn(total, in_features, device="cuda", dtype=dtype)
    grad_out = torch.randn(total, out_features, device="cuda", dtype=dtype)

    module.zero_grad(set_to_none=True)
    x = inp.detach().clone().requires_grad_(True)
    out = module(x, splits)
    out.backward(grad_out)

    ref_out, ref_dx, ref_dw = _reference(inp, weights, grad_out, splits)

    label = f"in={in_features} out={out_features} splits[:4]={splits[:4]}"
    _assert_group_list_close(_split_rows(out.detach(), splits),
                             _split_rows(ref_out, splits), f"out [{label}]")
    _assert_group_list_close(_split_rows(x.grad.detach(), splits),
                             _split_rows(ref_dx, splits), f"dx [{label}]")
    dw = []
    for i, p in enumerate(wparams):
        assert p.grad is not None, f"wgrad missing for group {i}"
        dw.append(p.grad.detach())
    _assert_group_list_close(dw, ref_dw, f"dw [{label}]")


def _wgrad_call(a, b, out, splits, accumulate):
    general_grouped_gemm(
        a, b, out, [None] * len(a), torch.bfloat16, layout="NT",
        m_splits=splits, grad=True, accumulate=accumulate, single_output=False,
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("groups", [1, 2, 3, 4])
@pytest.mark.parametrize("mpe", [256, 200, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("accumulate", [False, True])
def test_grouped_wgrad_rtc_inplace(groups, mpe, dtype, accumulate, capfd):
    """Cover a real main_grad, the first/later M tile, a K tail and a repeated RMW."""
    torch.manual_seed(1234)
    a = [torch.randn(mpe, 1024, device="cuda", dtype=torch.bfloat16) for _ in range(groups)]
    b = [torch.randn(mpe, 512, device="cuda", dtype=torch.bfloat16) for _ in range(groups)]
    out = [torch.randn(512, 1024, device="cuda", dtype=dtype) for _ in range(groups)]
    products = [y.float().t() @ x.float() for x, y in zip(a, b)]
    references = [c.float().clone() for c in out]
    if not accumulate:
        # beta==0 must not read the old D, not even when it holds NaN.
        for c in out:
            c.fill_(float("nan"))
    for _ in range(2):
        _wgrad_call(a, b, out, [mpe] * groups, accumulate)
        for group, (actual, product) in enumerate(zip(out, products)):
            expected = product + references[group] if accumulate else product
            for rows in (slice(0, 256), slice(256, 512)):
                ref = expected[rows]
                rel = torch.linalg.vector_norm(actual[rows].float() - ref) / torch.linalg.vector_norm(ref)
                assert torch.isfinite(rel) and rel < (1e-6 if dtype == torch.float32 else 4e-3), (
                    group, rows, float(rel), dtype, accumulate,
                )
            # bf16 rounds at every step, so the reference for the second
            # accumulation has to start from the rounded main_grad.
            references[group] = expected.to(dtype).float()
    assert "fallback" not in capfd.readouterr().err.lower()


@pytest.mark.parametrize("groups", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_wgrad_rtc_mixed_tails(groups, dtype, capfd):
    """Cover differing per-group K, M/N tails and a distinct C/D pointer per group."""
    torch.manual_seed(5678)
    splits = [256, 200, 64, 248][:groups]
    rows = [512, 200, 64, 536][:groups]
    a = [torch.randn(k, 1032, device="cuda", dtype=torch.bfloat16) for k in splits]
    b = [torch.randn(k, m, device="cuda", dtype=torch.bfloat16) for k, m in zip(splits, rows)]
    out = [torch.randn(m, 1032, device="cuda", dtype=dtype) for m in rows]
    references = [y.float().t() @ x.float() + c.float() for x, y, c in zip(a, b, out)]
    _wgrad_call(a, b, out, splits, True)
    for actual, expected in zip(out, references):
        rel = torch.linalg.vector_norm(actual.float() - expected) / torch.linalg.vector_norm(expected)
        assert torch.isfinite(rel) and rel < (1e-6 if dtype == torch.float32 else 4e-3), float(rel)
    assert "fallback" not in capfd.readouterr().err.lower()


@pytest.mark.parametrize("rows", [512, 200, 64, 536])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_wgrad_rtc_source_map(rows, dtype):
    """A==0 and C==arange must be bit-identical; a norm would hide the original
    1536 misplaced elements."""
    a = [torch.zeros(200, 1032, device="cuda", dtype=torch.bfloat16)]
    b = [torch.randn(200, rows, device="cuda", dtype=torch.bfloat16)]
    source = torch.arange(rows * 1032, device="cuda", dtype=torch.float32).reshape(rows, 1032).to(dtype)
    out = [source.clone()]
    _wgrad_call(a, b, out, [200], True)
    assert torch.equal(out[0], source)


@pytest.mark.parametrize("prologue", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mpe", [64, 200, 2048])
def test_grouped_wgrad_rtc_accumulator_seed(monkeypatch, prologue, dtype, mpe):
    """An exactly representable non-zero product proves the seed is consumed
    once and never lost across K or persistent tiles."""
    monkeypatch.setenv("NVTE_GG_EXTRA_DEFINES", f"-DNVTE_GG_PROLOGUE_C={int(prologue)}")
    monkeypatch.setenv("NVTE_GG_WGRAD_C_FUSION", "1")
    rows = [64, 200, 536]
    a = [torch.full((mpe, 1032), 0.125, device="cuda", dtype=torch.bfloat16) for _ in rows]
    b = [torch.full((mpe, m), 0.0625, device="cuda", dtype=torch.bfloat16) for m in rows]
    out = [(torch.arange(m * 1032, device="cuda").reshape(m, 1032) % 16).to(dtype) / 8
           for m in rows]
    expected = [c.clone() for c in out]
    for _ in range(2):
        expected = [(c.float() + mpe / 128).to(dtype) for c in expected]
        _wgrad_call(a, b, out, [mpe] * len(rows), True)
        for actual, reference in zip(out, expected):
            assert torch.equal(actual, reference)


@pytest.mark.parametrize("atomic_options", ["", "-DNVTE_GG_ATOMIC_GLOBAL=1",
                                            "-DNVTE_GG_ATOMIC_GLOBAL=1 -DNVTE_GG_ATOMIC_INTERLEAVE=1"])
def test_grouped_wgrad_rtc_atomic_special_values(monkeypatch, atomic_options):
    """A zero product proves the atomic treats subnormals and Inf/NaN exactly
    like the fragment path does."""
    rows, cols, k = 64, 1032, 64
    a = [torch.zeros(k, cols, device="cuda", dtype=torch.bfloat16)]
    b = [torch.ones(k, rows, device="cuda", dtype=torch.bfloat16)]
    values = torch.tensor([0.0, -0.0, 1e-45, -1e-45, 1e-40, -1e-40,
                           torch.finfo(torch.float32).tiny,
                           -torch.finfo(torch.float32).tiny,
                           float("inf"), -float("inf"), float("nan")],
                          device="cuda", dtype=torch.float32)
    source = values.repeat((rows * cols + values.numel() - 1) // values.numel())
    source = source[:rows * cols].reshape(rows, cols)
    reference, actual = [source.clone()], [source.clone()]
    monkeypatch.setenv("NVTE_GG_EXTRA_DEFINES", "-DNVTE_GG_ATOMIC_C=0")
    _wgrad_call(a, b, reference, [k], True)
    monkeypatch.setenv("NVTE_GG_EXTRA_DEFINES", "-DNVTE_GG_ATOMIC_C=1 " + atomic_options)
    _wgrad_call(a, b, actual, [k], True)
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("atomic_options", ["", "-DNVTE_GG_ATOMIC_GLOBAL=1",
                                            "-DNVTE_GG_ATOMIC_GLOBAL=1 -DNVTE_GG_ATOMIC_INTERLEAVE=1"])
@pytest.mark.parametrize("prologue", [False, True])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_wgrad_rtc_atomic_c(monkeypatch, prologue, accumulate, dtype, atomic_options):
    """Compare the atomic experiment bit for bit against the original path,
    covering the tail boundary, back-to-back RMW, the bf16/noC bypass and
    mutual exclusion."""
    torch.manual_seed(9012)
    splits, rows, cols = [64, 200, 2048], [64, 200, 536], 1032
    a = [torch.randn(k, cols, device="cuda", dtype=torch.bfloat16) for k in splits]
    b = [torch.randn(k, m, device="cuda", dtype=torch.bfloat16)
         for k, m in zip(splits, rows)]
    sources = [torch.randn(m, cols, device="cuda", dtype=dtype) for m in rows]
    sentinel, guard = 97.0, 32
    storage = [torch.full((m * cols + 2 * guard,), sentinel, device="cuda", dtype=dtype)
               for m in rows]
    actual = [buf[guard:-guard].reshape(m, cols) for buf, m in zip(storage, rows)]
    reference = [c.clone() for c in sources]
    for out, source in zip(actual, sources):
        out.copy_(source)
    if not accumulate:
        for out in actual + reference:
            out.fill_(float("nan"))
    for _ in range(2):
        monkeypatch.setenv("NVTE_GG_EXTRA_DEFINES",
                           f"-DNVTE_GG_PROLOGUE_C={int(prologue)} -DNVTE_GG_ATOMIC_C=0")
        _wgrad_call(a, b, reference, splits, accumulate)
        monkeypatch.setenv("NVTE_GG_EXTRA_DEFINES",
                           f"-DNVTE_GG_PROLOGUE_C={int(prologue)} -DNVTE_GG_ATOMIC_C=1 {atomic_options}")
        _wgrad_call(a, b, actual, splits, accumulate)
        for out, expected, buf in zip(actual, reference, storage):
            assert torch.isfinite(out).all()
            assert torch.equal(out, expected)
            assert torch.all(buf[:guard] == sentinel)
            assert torch.all(buf[-guard:] == sentinel)


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_wgrad_rtc_single_fp32_launch(accumulate, dtype):
    """Guard against the regression that dispatched bf16 first and fp32 after,
    including the extra kernel that beta==0 used to hide."""
    a = [torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)]
    b = [torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)]
    out = [torch.randn(512, 1024, device="cuda", dtype=dtype)]
    _wgrad_call(a, b, out, [256], accumulate)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        _wgrad_call(a, b, out, [256], accumulate)
    kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
    assert len(kernels) == 1, kernels
    name = "fp32" if dtype == torch.float32 else "bf16"
    assert f"bf16_bf16_{name}_acc_fp32" in kernels[0], kernels