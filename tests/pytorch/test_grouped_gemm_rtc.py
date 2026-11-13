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
from unittest.mock import Mock

import pytest
import torch
import transformer_engine
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
import transformer_engine.pytorch.cpp_extensions.gemm as gemm_module
from transformer_engine.pytorch.cpp_extensions.gemm import (
    general_grouped_gemm,
    general_grouped_gemm_for_grouped_tensor,
)
from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import GroupedTensorStorage

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
def _rtc_env(monkeypatch, request):
    # Selection-only specs mock capability queries and never initialize CUDA.
    if request.node.name.startswith("test_grouped_gemm_policy_"):
        yield
        return
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cap = torch.cuda.get_device_capability()
    if cap not in ((8, 0), (8, 9)):
        pytest.skip(f"RTC grouped GEMM targets sm80/sm89, got sm{cap[0]}{cap[1]}")
    if request.node.name.startswith("test_contiguous") and cap != (8, 9):
        pytest.skip("contiguous grouped GEMM requires sm89")
    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "1")
    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM_RTC", "1")
    monkeypatch.setenv("NVTE_FORCE_RTC_NO_FALLBACK", "1")
    monkeypatch.setenv("NVTE_GG_ALLOW_FP32_ACCUM", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "0")
    monkeypatch.delenv("NVTE_GG_FORCE_TILE", raising=False)
    monkeypatch.delenv("NVTE_GG_CONTIGUOUS_N_EXPAND", raising=False)
    monkeypatch.delenv("NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX", raising=False)
    monkeypatch.delenv("NVTE_GG_CONTIGUOUS_BN", raising=False)
    monkeypatch.delenv("NVTE_GG_CONTIGUOUS_ALIGNED", raising=False)
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
    # 长回归绑定已完成构建的冻结树，避免 configure 物化期间混读头文件。
    frozen = os.getenv("NVTE_GG_TEST_INCLUDE_DIR")
    if frozen:
        assert os.path.isfile(os.path.join(frozen, "ppu_include.hpp")), frozen
        assert os.path.isfile(os.path.join(frozen, "cutlass", "cutlass.h")), frozen
        candidate = frozen
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mpe", [64, 200, 2048])
def test_grouped_wgrad_rtc_accumulator_seed(dtype, mpe):
    """An exactly representable non-zero product proves C is added
    once and never lost across K or persistent tiles."""
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


def test_grouped_wgrad_rtc_source_special_values():
    """A zero product preserves source subnormals and Inf/NaN numerically."""
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
    actual = [source.clone()]
    _wgrad_call(a, b, actual, [k], True)
    torch.testing.assert_close(actual[0], source, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_wgrad_rtc_guarded_tail_rmw(accumulate, dtype):
    """Independent products-plus-old oracle for irregular tails and repeated RMW."""
    torch.manual_seed(9012)
    splits, rows, cols = [64, 200, 2048], [64, 200, 536], 1032
    a = [torch.randn(k, cols, device="cuda", dtype=torch.bfloat16) for k in splits]
    b = [torch.randn(k, m, device="cuda", dtype=torch.bfloat16)
         for k, m in zip(splits, rows)]
    products = [y.float().t() @ x.float() for x, y in zip(a, b)]
    sources = [torch.randn(m, cols, device="cuda", dtype=dtype) for m in rows]
    sentinel, guard = 97.0, 32
    storage = [torch.full((m * cols + 2 * guard,), sentinel, device="cuda", dtype=dtype)
               for m in rows]
    actual = [buf[guard:-guard].reshape(m, cols) for buf, m in zip(storage, rows)]
    references = [c.float().clone() for c in sources]
    for out, source in zip(actual, sources):
        out.copy_(source)
    if not accumulate:
        for out in actual:
            out.fill_(float("nan"))
    for _ in range(2):
        _wgrad_call(a, b, actual, splits, accumulate)
        for group, (out, product, old, buf, k) in enumerate(
                zip(actual, products, references, storage, splits)):
            expected = product + old if accumulate else product
            rel = torch.linalg.vector_norm(out.float() - expected) / \
                torch.linalg.vector_norm(expected).clamp_min(1e-30)
            limit = max(1e-6, 1e-6 * k / 256) if dtype == torch.float32 else 4e-3
            assert torch.isfinite(out).all() and torch.isfinite(rel) and rel < limit, (
                group, float(rel), dtype, accumulate,
            )
            rounded = expected.to(dtype).float()
            torch.testing.assert_close(
                out.float(), rounded, rtol=1e-4 if dtype == torch.float32 else 0.05,
                atol=5e-4 * max(1, k / 256) if dtype == torch.float32 else 0.05,
            )
            # BF16 rounds each write; the next RMW starts from that precision.
            references[group] = rounded
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


def _packed_views(shapes, dtype, fill=None, prefix=32):
    """每个操作数独立 backing，只有 backing 首尾放 guard。"""
    sizes = [m * n for m, n in shapes]
    backing = torch.full((sum(sizes) + prefix + 32,), 97, device="cuda", dtype=dtype)
    views, offset = [], prefix
    for shape, count in zip(shapes, sizes):
        view = backing[offset:offset + count].view(shape)
        if fill is None:
            view.copy_(torch.randn(shape, device="cuda", dtype=dtype))
        else:
            view.fill_(fill)
        views.append(view)
        offset += count
    return views, backing


def _contiguous_shapes(layout, splits, m=512, n=512, k=256):
    if layout == "NT":
        return [(s, n) for s in splits], [(s, m) for s in splits], [(m, n)] * len(splits)
    return ([(n, k) if layout == "TN" else (k, n)] * len(splits),
            [(s, k) for s in splits], [(s, n) for s in splits])


def _packed_case(layout, dtype, splits, m=512, n=512, k=256):
    ashapes, bshapes, dshapes = _contiguous_shapes(layout, splits, m, n, k)
    a, ag = _packed_views(ashapes, torch.bfloat16)
    b, bg = _packed_views(bshapes, torch.bfloat16)
    d, dg = _packed_views(dshapes, dtype)
    return a, b, d, (ag, bg, dg)


def _products(a, b, layout):
    return [(y.float().t() if layout == "NT" else y.float()) @
            (x.float().t() if layout == "TN" else x.float()) for x, y in zip(a, b)]


def _assert_guards(backings, prefix=32):
    for backing in backings:
        assert torch.all(backing[:prefix] == 97)
        assert torch.all(backing[-32:] == 97)


def _contiguous_ending(dtype, with_c, mode=1, kmax=512):
    ending = f"_C{'fp32' if dtype == torch.float32 else 'bf16'}" if with_c else "_Cvoid"
    ending += f"_contiguous{mode}"
    if mode > 1 and kmax != 512:
        ending += f"_kmax{kmax}"
    ending += "_fulltile"
    return ending


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("pattern", ["aligned_ragged", "positive_k256", "many", "empty", "zero_k"])
def test_contiguous_grouped_base(monkeypatch, layout, dtype, accumulate, pattern, capfd):
    """Full-tile ragged groups, positive K=256, K=0 and repeated persistent RMW."""
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("contiguous 首轮仅 sm89")
    # More nonempty groups than the persistent grid has SMs.
    repetitions = torch.cuda.get_device_properties(0).multi_processor_count + 1
    splits = {"aligned_ragged": [256, 768, 0, 512], "positive_k256": [256, 0, 256],
              "many": [256, 0] * repetitions,
              "empty": [0, 0, 0], "zero_k": [0, 0, 0] if layout == "NT" else
              [256, 768, 0, 512]}[pattern]
    m = n = 256 if pattern == "many" else 512
    k = 0 if pattern == "zero_k" else 256
    if layout == "NT":
        ashapes = [(s, n) for s in splits]
        bshapes = [(s, m) for s in splits]
        dshapes = [(m, n)] * len(splits)
    else:
        ashapes = [(n, k) if layout == "TN" else (k, n)] * len(splits)
        bshapes = [(s, k) for s in splits]
        dshapes = [(s, n) for s in splits]
    torch.manual_seed(123)
    a, aguard = _packed_views(ashapes, torch.bfloat16)
    b, bguard = _packed_views(bshapes, torch.bfloat16)
    actual, dguard = _packed_views(dshapes, dtype)
    reference = [d.clone() for d in actual]
    if not accumulate:
        for d in actual + reference:
            d.fill_(float("nan"))
    products = [(y.float().t() if layout == "NT" else y.float()) @
                (x.float().t() if layout == "TN" else x.float()) for x, y in zip(a, b)]
    monkeypatch.setenv("NVTE_GG_FORCE_TILE", "256,128,64,3")
    for _ in range(2):
        old = [d.float().clone() for d in actual]
        monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "0")
        general_grouped_gemm(a, b, reference, [None] * len(a), dtype, layout=layout,
                             m_splits=splits, accumulate=accumulate, grad=layout == "NT")
        monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
        general_grouped_gemm(a, b, actual, [None] * len(a), dtype, layout=layout,
                             m_splits=splits, accumulate=accumulate, grad=layout == "NT")
        torch.cuda.synchronize()
        for d, ref, product, c in zip(actual, reference, products, old):
            if not accumulate:
                assert torch.equal(d, ref)
            expected = product + c if accumulate else product
            # Both implementations must meet the independent fp32 oracle;
            # C seeding order need not give bit-identical fp32 reductions.
            torch.testing.assert_close(d.float(), expected, rtol=0.005, atol=0.005)
            torch.testing.assert_close(ref.float(), expected, rtol=0.005, atol=0.005)
    for guard in (aguard, bguard, dguard):
        assert torch.all(guard[:32] == 97) and torch.all(guard[-32:] == 97)
    assert "fallback" not in capfd.readouterr().err.lower()


@pytest.mark.parametrize("invalid", ["separate", "gap", "reversed", "dtype", "shape", "fusion"])
def test_contiguous_reject_invalid(monkeypatch, invalid):
    a, b, d, guards = _packed_case("TN", torch.bfloat16, [256, 256], n=256)
    before = guards[-1].clone()
    kwargs = {}
    error = "must be packed adjacent views"
    if invalid == "separate": a = [x.clone() for x in a]
    elif invalid == "gap": a[1] = a[1][1:]
    elif invalid == "reversed": a.reverse()
    elif invalid == "dtype": b[1] = b[1].float()
    elif invalid == "shape":
        b[1] = b[1].view(128, 512)
        error = "group 1 incompatible matrix dimensions"
    elif invalid == "fusion":
        kwargs["gelu"] = True
        error = "does not support single_output or fusion"
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    with pytest.raises(ValueError, match=f"NVTE_GG_CONTIGUOUS.*{error}"):
        general_grouped_gemm(a, b, d, [None] * 2, torch.bfloat16, **kwargs)
    torch.cuda.synchronize()
    assert torch.equal(guards[-1], before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_contiguous_source_map(monkeypatch, dtype):
    a, _ = _packed_views([(256, 512), (0, 512), (256, 512)], torch.bfloat16, 0)
    b, _ = _packed_views([(256, 512), (0, 512), (256, 512)], torch.bfloat16, 1)
    d, guard = _packed_views([(512, 512)] * 3, dtype)
    for view in d:
        view.copy_((torch.arange(view.numel(), device="cuda") % 1024).view_as(view).to(dtype))
    before = [view.clone() for view in d]
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    for _ in range(2):
        general_grouped_gemm(a, b, d, [None] * 3, dtype, layout="NT",
                             m_splits=[256, 0, 256], accumulate=True)
        torch.cuda.synchronize()
        assert all(torch.equal(view, source) for view, source in zip(d, before))
    assert torch.all(guard[:32] == 97) and torch.all(guard[-32:] == 97)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
def test_contiguous_unaligned_tails(monkeypatch, layout, dtype, accumulate):
    """Former unaligned short-K successes now reject before modifying the full D backing."""
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("contiguous 首轮仅 sm89")
    n, m, k = 513, 269, 7
    splits = [7, 0, 1, 65] if layout == "NT" else [269, 0, 1, 17]
    ashapes = ([(s, n) for s in splits] if layout == "NT" else
               [(n, k) if layout == "TN" else (k, n)] * len(splits))
    bshapes = [(s, m) if layout == "NT" else (s, k) for s in splits]
    dshapes = [(m if layout == "NT" else s, n) for s in splits]
    a, aguard = _packed_views(ashapes, torch.bfloat16)
    b, bguard = _packed_views(bshapes, torch.bfloat16)
    d, dguard = _packed_views(dshapes, dtype)
    before = dguard.clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    with pytest.raises(ValueError, match="NVTE_GG_CONTIGUOUS.*requires 16-byte alignment"):
        general_grouped_gemm(a, b, d, [None] * len(a), dtype,
                             layout=layout, m_splits=splits, accumulate=accumulate)
    torch.cuda.synchronize()
    assert torch.equal(dguard, before)
    _assert_guards((aguard, bguard, dguard))


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("n,prefix", [(512, 32), (514, 32), (512, 33),
                                     (513, 32), (513, 33)])
def test_contiguous_epilogue_copy_alignment(monkeypatch, layout, dtype, accumulate, n, prefix):
    """完整 tile 的连续 pair、动态行步长和 packed 起点均独立验收。"""
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("contiguous 首轮仅 sm89")
    m, k = 512, 256
    splits = [256, 768, 0, 512]
    ashapes = ([(s, n) for s in splits] if layout == "NT" else
               [(n, k) if layout == "TN" else (k, n)] * len(splits))
    bshapes = [(s, m) if layout == "NT" else (s, k) for s in splits]
    dshapes = [(m if layout == "NT" else s, n) for s in splits]
    torch.manual_seed(321)
    a, aguard = _packed_views(ashapes, torch.bfloat16)
    b, bguard = _packed_views(bshapes, torch.bfloat16)
    d, dguard = _packed_views(dshapes, dtype, prefix=prefix)
    products = [(y.float().t() if layout == "NT" else y.float()) @
                (x.float().t() if layout == "TN" else x.float()) for x, y in zip(a, b)]
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_FORCE_TILE", "256,128,64,3")
    if n % 256 or prefix % (16 // d[0].element_size()):
        before = dguard.clone()
        error = "requires 16-byte alignment" if prefix != 32 else "N must be a multiple of 256"
        with pytest.raises(ValueError, match=f"NVTE_GG_CONTIGUOUS.*{error}"):
            general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout=layout,
                                 m_splits=splits, accumulate=accumulate)
        torch.cuda.synchronize()
        assert torch.equal(dguard, before)
        return
    expands = (1,)
    for expand in expands:
        monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", str(expand))
        if not accumulate:
            for view in d:
                view.fill_(float("nan"))
        for _ in range(2):
            before = [view.float().clone() for view in d]
            general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout=layout,
                                 m_splits=splits, accumulate=accumulate)
            torch.cuda.synchronize()
            for actual, product, old in zip(d, products, before):
                expected = product + old if accumulate else product
                torch.testing.assert_close(actual.float(), expected, rtol=0.005, atol=0.005)
                assert torch.isfinite(actual).all()
        for guard, start in ((aguard, 32), (bguard, 32), (dguard, prefix)):
            assert torch.all(guard[:start] == 97) and torch.all(guard[-32:] == 97)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout=layout,
                             m_splits=splits, accumulate=accumulate)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
    assert len(kernels) == 1 and kernels[0].endswith(_contiguous_ending(dtype, accumulate)), kernels
    assert "tb256x128x64_s3" in kernels[0], kernels


@pytest.mark.parametrize("layout,dtype,bn,n,prefix,accumulate,kmax", [
    pytest.param("TN", torch.bfloat16, 128, 512, 32, False, 512, id="tn-bf16-bn128-aligned"),
    pytest.param("NN", torch.bfloat16, 256, 1024, 32, False, 512, id="nn-bf16-bn256-aligned"),
    pytest.param("NT", torch.bfloat16, 256, 1024, 32, False, 1024, id="nt-bf16-mixed-kmax1024"),
    pytest.param("TN", torch.float32, 128, 512, 32, False, 512, id="tn-fp32-bn128-aligned"),
    pytest.param("NN", torch.float32, 256, 1024, 32, False, 512, id="nn-fp32-bn256-aligned"),
    pytest.param("NT", torch.float32, 128, 512, 32, False, 2048, id="nt-fp32-mixed-kmax2048"),
    pytest.param("TN", torch.bfloat16, 256, 1024, 33, False, 512, id="prefix-fallback"),
    pytest.param("NN", torch.float32, 128, 514, 32, False, 512, id="stride-fallback"),
    pytest.param("NT", torch.bfloat16, 128, 513, 32, False, 512, id="odd-n-fallback"),
    pytest.param("TN", torch.float32, 128, 513, 33, False, 512, id="both-fallback"),
    pytest.param("NT", torch.float32, 128, 512, 32, True, 512, id="withc-fp32-unchanged"),
    pytest.param("NT", torch.bfloat16, 128, 512, 32, True, 512, id="withc-bf16-unchanged"),
])
def test_contiguous_aligned_epilogue(monkeypatch, layout, dtype, bn, n, prefix, accumulate, kmax):
    """Strict aligned baseline rejects invalid stride/origin."""
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("contiguous 首轮仅 sm89")
    m, k = 512, 256
    splits = [0, 256, 512, kmax, kmax + 256] if layout == "NT" else [256, 768, 0, 512]
    ashapes = ([(s, n) for s in splits] if layout == "NT" else
               [(n, k) if layout == "TN" else (k, n)] * len(splits))
    bshapes = [(s, m) if layout == "NT" else (s, k) for s in splits]
    dshapes = [(m if layout == "NT" else s, n) for s in splits]
    torch.manual_seed(321)
    a, aguard = _packed_views(ashapes, torch.bfloat16)
    b, bguard = _packed_views(bshapes, torch.bfloat16)
    d, dguard = _packed_views(dshapes, dtype, prefix=prefix)
    initial = [view.clone() for view in d]
    products = [(y.float().t() if layout == "NT" else y.float()) @
                (x.float().t() if layout == "TN" else x.float()) for x, y in zip(a, b)]
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_BN", str(bn))
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX", str(kmax))
    if n % 256 or prefix != 32:
        before = dguard.clone()
        error = "requires 16-byte alignment" if prefix != 32 else "N must be a multiple of 256"
        with pytest.raises(ValueError, match=f"NVTE_GG_CONTIGUOUS.*{error}"):
            general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout=layout,
                                 m_splits=splits, accumulate=accumulate)
        torch.cuda.synchronize()
        assert torch.equal(dguard, before)
        return
    expected_bits = None
    expands = (1, 4) if not accumulate and bn == 256 else (1,)
    for expand in expands:
        monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", str(expand))
        for view, old in zip(d, initial):
            view.copy_(old) if accumulate else view.fill_(float("nan"))
        general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout=layout,
                             m_splits=splits, accumulate=accumulate)
        torch.cuda.synchronize()
        for view, product, old in zip(d, products, initial):
            torch.testing.assert_close(view.float(), product + old.float() if accumulate else product,
                                       rtol=0.005, atol=0.005)
            assert torch.isfinite(view).all()
        integer = torch.int32 if dtype == torch.float32 else torch.int16
        bits = [view.view(integer).clone() for view in d]
        if expected_bits is None:
            expected_bits = bits
        assert all(torch.equal(x, y) for x, y in zip(expected_bits, bits))
        for guard, start in ((aguard, 32), (bguard, 32), (dguard, prefix)):
            assert torch.all(guard[:start] == 97) and torch.all(guard[-32:] == 97)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout=layout,
                             m_splits=splits, accumulate=accumulate)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
    mode = 2 if expands[-1] == 4 and n % 1024 == 0 else 1
    assert len(kernels) == 1 and kernels[0].endswith(
        _contiguous_ending(dtype, accumulate, mode, kmax)), kernels
    assert f"tb256x{bn}x64_s3" in kernels[0], kernels


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
def test_contiguous_zero_k_bits(monkeypatch, dtype, accumulate):
    """K=0、beta=1 逐位保持 signed zero/Inf/NaN；beta=0 不读旧 D。"""
    a, _ = _packed_views([(0, 256)] * 2, torch.bfloat16)
    b, _ = _packed_views([(0, 256)] * 2, torch.bfloat16)
    d, guard = _packed_views([(256, 256)] * 2, dtype)
    values = torch.tensor([0., -0., float("inf"), -float("inf"), float("nan")],
                          device="cuda", dtype=dtype)
    for view in d:
        view.copy_(values.repeat((view.numel() + 4) // 5)[:view.numel()].view_as(view))
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    before = [view.view(bits).clone() for view in d]
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    general_grouped_gemm(a, b, d, [None] * 2, dtype, layout="NT",
                         m_splits=[0, 0], accumulate=accumulate)
    torch.cuda.synchronize()
    for view, old in zip(d, before):
        assert torch.equal(view.view(bits), old if accumulate else torch.zeros_like(old))
    assert torch.all(guard[:32] == 97) and torch.all(guard[-32:] == 97)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_contiguous_single_launch(monkeypatch, dtype):
    """必须命中专用符号，禁止用未执行或其他后端的正确结果冒充。"""
    a, _ = _packed_views([(256, 256)] * 2, torch.bfloat16)
    b, _ = _packed_views([(256, 256)] * 2, torch.bfloat16)
    d, _ = _packed_views([(256, 256)] * 2, dtype)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    general_grouped_gemm(a, b, d, [None] * 2, dtype, layout="NT",
                         m_splits=[256, 256], accumulate=True)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        general_grouped_gemm(a, b, d, [None] * 2, dtype, layout="NT",
                             m_splits=[256, 256], accumulate=True)
        torch.cuda.synchronize()
    kernels = [event.name for event in prof.events()
               if event.name.startswith("nvte_grouped_gemm3_")]
    assert len(kernels) == 1 and kernels[0].endswith(_contiguous_ending(dtype, True)), kernels
    assert "tb256x128x64_s3" in kernels[0], kernels


def _contiguous_raw_call(a, b, d, workspace, workspace_size, splits,
                         accumulate=False, layout="NN", dtype=None):
    """Call the extension directly, without Python admission or metadata repair."""
    empty = [torch.empty(0, device="cuda", dtype=torch.bfloat16) for _ in a]
    dtype = d[0].dtype if dtype is None else dtype
    d_type = {torch.bfloat16: tex.DType.kBFloat16, torch.float32: tex.DType.kFloat32}[dtype]
    return tex.te_general_grouped_gemm(
        a, layout[0] == "T", b, layout[1] == "T", d, d_type, splits,
        empty, tex.DType.kBFloat16, False, empty, False,
        workspace, workspace_size, accumulate, False, 0)


@pytest.mark.parametrize("option", [
    "NVTE_GG_RTC_LINEINFO", "NVTE_GG_DISABLE_RTC_CACHE", "NVTE_CUTLASS_GROUPED_GEMM_RTC_VERBOSE",
])
@pytest.mark.parametrize("rows", [0, 256], ids=["empty", "work"])
def test_contiguous_reject_invalid_rtc_option_before_upload(monkeypatch, option, rows):
    a, _ = _packed_views([(256, 256)], torch.bfloat16)
    b, _ = _packed_views([(rows, 256)], torch.bfloat16)
    d, guard = _packed_views([(rows, 256)], torch.bfloat16)
    workspace = torch.full((2048,), 73, dtype=torch.uint8, device="cuda")
    before_d, before_workspace = guard.clone(), workspace.clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv(option, "bad")
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
    ]) as prof:
        with pytest.raises(RuntimeError, match=f"NVTE_GG_CONTIGUOUS invalid {option}"):
            _contiguous_raw_call(a, b, d, [workspace], workspace.numel(), [rows])
        torch.cuda.synchronize()
    assert not [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
    assert torch.equal(guard, before_d)
    assert torch.equal(workspace, before_workspace)


@pytest.mark.parametrize("invalid", ["short", "inflated", "backing", "dtype", "rank", "cpu",
                                     "strided", "unaligned", "missing", "lists"])
def test_contiguous_host_workspace(monkeypatch, invalid):
    """真实容量优先于声明容量；非法 workspace/list 必须在读写前拒绝。"""
    a, _ = _packed_views([(256, 256)] * 2, torch.bfloat16)
    b, _ = _packed_views([(512, 256)] * 2, torch.bfloat16)
    d, guard = _packed_views([(512, 256)] * 2, torch.bfloat16)
    before = guard.clone()
    ws = torch.empty(2048, device="cuda", dtype=torch.uint8)
    size = ws.numel()
    if invalid == "short":
        ws, size = ws[:256], 256
    elif invalid == "inflated":
        size += 1
    elif invalid == "backing":
        ws.untyped_storage().resize_(128)
    elif invalid == "dtype":
        ws = ws.float()
    elif invalid == "rank":
        ws = ws.view(2, 1024)
    elif invalid == "cpu":
        ws = ws.cpu()
    elif invalid == "strided":
        ws, size = ws[::2], 1024
    elif invalid == "unaligned":
        ws, size = ws[1:], 2047
    elif invalid == "lists":
        b = []
    workspace = [] if invalid == "missing" else [ws]
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    error = {
        "short": "insufficient device workspace", "inflated": "exceeds the actual workspace capacity",
        "backing": "exceeds the actual workspace capacity",
        "dtype": "contiguous rank-1 CUDA byte workspace", "rank": "contiguous rank-1 CUDA byte workspace",
        "cpu": "contiguous rank-1 CUDA byte workspace", "strided": "contiguous rank-1 CUDA byte workspace",
        "unaligned": "128-byte aligned workspace", "missing": "requires device workspace",
        "lists": "matching nonempty tensor lists",
    }[invalid]
    with pytest.raises(RuntimeError, match=f"NVTE_GG_CONTIGUOUS.*{error}"):
        _contiguous_raw_call(a, b, d, workspace, size, [512, 512])
    torch.cuda.synchronize()
    assert torch.equal(guard, before)


def test_contiguous_host_dimension_overflow(monkeypatch):
    """零元素的大维度不分配巨大矩阵，也不得进入描述构造。"""
    rows = 2 ** 31
    a, _ = _packed_views([(0, 0)], torch.bfloat16)
    b, _ = _packed_views([(rows, 0)], torch.bfloat16)
    d, guard = _packed_views([(rows, 0)], torch.bfloat16)
    ws = torch.empty(128, device="cuda", dtype=torch.uint8)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    with pytest.raises(RuntimeError, match="NVTE_GG_CONTIGUOUS.*dimension exceeds INT_MAX-256"):
        _contiguous_raw_call(a, b, d, [ws], ws.numel(), [rows])
    assert torch.all(guard == 97)


@pytest.mark.parametrize("rows", [0, 256])
@pytest.mark.parametrize("boundary", ["list", "grouped", "direct"])
def test_contiguous_reject_graph_capture(monkeypatch, rows, boundary):
    a, aguard = _packed_views([(256, 256)], torch.bfloat16)
    b, bguard = _packed_views([(rows, 256)], torch.bfloat16)
    d, guard = _packed_views([(rows, 256)], torch.bfloat16)
    ws = torch.empty(2048, device="cuda", dtype=torch.uint8)
    if boundary == "direct":
        call = lambda: _contiguous_raw_call(a, b, d, [ws], ws.numel(), [rows])
    elif boundary == "list":
        call = lambda: general_grouped_gemm(a, b, d, [None], torch.bfloat16,
                                            layout="NN", m_splits=[rows])
    else:
        operands = [_grouped_storage(views, backing)
                    for views, backing in zip((a, b, d), (aguard, bguard, guard))]
        call = lambda: general_grouped_gemm_for_grouped_tensor(*operands, layout="NN")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    before, before_ws = guard.clone(), ws.clone()
    graph = torch.cuda.CUDAGraph()
    with pytest.raises((RuntimeError, ValueError), match="NVTE_GG_CONTIGUOUS.*graph capture"):
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.synchronize()
    assert torch.equal(guard, before)
    assert torch.equal(ws, before_ws)


@pytest.mark.parametrize("layout", ["TN", "NN"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k", [0, 256, 512, 768])
def test_contiguous_nexpand_small_k(monkeypatch, layout, dtype, k):
    """Full-tile K=256, K=0, empty experts and threshold bypass."""
    splits, n = [256, 768, 0, 512] * 3, 1024
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_BN", "256")
    ashapes = [(n, k) if layout == "TN" else (k, n)] * len(splits)
    a, aguard = _packed_views(ashapes, torch.bfloat16)
    b, bguard = _packed_views([(m, k) for m in splits], torch.bfloat16)
    d, dguard = _packed_views([(m, n) for m in splits], dtype, float("nan"))
    reference, _ = _packed_views([(m, n) for m in splits], dtype, float("nan"))
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "1")
    general_grouped_gemm(a, b, reference, [None] * len(a), dtype,
                         layout=layout, m_splits=splits, accumulate=False)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "4")
    for _ in range(2):
        general_grouped_gemm(a, b, d, [None] * len(a), dtype,
                             layout=layout, m_splits=splits, accumulate=False)
        torch.cuda.synchronize()
        for x, y, actual, expected in zip(a, b, d, reference):
            assert torch.equal(actual, expected)
            product = y.float() @ (x.float().t() if layout == "TN" else x.float())
            torch.testing.assert_close(actual.float(), product, rtol=0.005, atol=0.005)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        general_grouped_gemm(a, b, d, [None] * len(a), dtype,
                             layout=layout, m_splits=splits, accumulate=False)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
    mode = 2 if 0 < k <= 512 else 1
    assert len(kernels) == 1 and kernels[0].endswith(_contiguous_ending(dtype, False, mode)), kernels
    assert "tb256x256x64_s3_Cvoid" in kernels[0], kernels
    for guard in (aguard, bguard, dguard):
        assert torch.all(guard[:32] == 97) and torch.all(guard[-32:] == 97)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n", [1024, 768])
def test_contiguous_nexpand_mixed_k(monkeypatch, dtype, n):
    """Per-expert K=0/256/threshold bypass; N%256 but N%1024!=0 is legal NE1."""
    splits, m = [0, 256, 512, 768], 512
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_BN", "256")
    a, aguard = _packed_views([(k, n) for k in splits], torch.bfloat16)
    b, bguard = _packed_views([(k, m) for k in splits], torch.bfloat16)
    d, dguard = _packed_views([(m, n)] * len(splits), dtype, float("nan"))
    reference, _ = _packed_views([(m, n)] * len(splits), dtype, float("nan"))
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "1")
    general_grouped_gemm(a, b, reference, [None] * len(a), dtype, layout="NT",
                         m_splits=splits, accumulate=False)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "4")
    general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout="NT",
                         m_splits=splits, accumulate=False)
    torch.cuda.synchronize()
    for x, y, actual, expected in zip(a, b, d, reference):
        assert torch.equal(actual, expected)
        torch.testing.assert_close(actual.float(), y.float().t() @ x.float(),
                                   rtol=0.005, atol=0.005)
    for guard in (aguard, bguard, dguard):
        assert torch.all(guard[:32] == 97) and torch.all(guard[-32:] == 97)


@pytest.mark.parametrize("option", ["0", "2", "bad", "withC"])
def test_contiguous_nexpand_reject(monkeypatch, option):
    """探索开关只接受 1/4，首轮 noC 之外明确报错。"""
    a, _ = _packed_views([(512, 256)], torch.bfloat16)
    b, _ = _packed_views([(256, 256)], torch.bfloat16)
    d, guard = _packed_views([(256, 512)], torch.bfloat16)
    before = guard.clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "4" if option == "withC" else option)
    with pytest.raises(RuntimeError, match="NVTE_GG_CONTIGUOUS_N_EXPAND"):
        general_grouped_gemm(a, b, d, [None], torch.bfloat16, layout="TN",
                             m_splits=[256], accumulate=option == "withC")
    torch.cuda.synchronize()
    assert torch.equal(guard, before)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n,k", [(1024, 256), (2048, 256), (4096, 512), (1024, 768),
                                (512, 256), (768, 256), (1024, 0)])
def test_contiguous_bn256_no_c(monkeypatch, layout, dtype, n, k):
    """Wide noC NE1/NE4, N1024 gating, aligned ragged K and persistent work."""
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("contiguous 仅 sm89")
    m = 512
    # Mixed eligible/ineligible K never crosses expert boundaries.
    splits = ([0, k, 256, 768] * 3 if 0 < k <= 512 else [k] * 12) if layout == "NT" else \
        [256, 768, 0, 512] * 3
    ashapes = ([(s, n) for s in splits] if layout == "NT" else
               [(n, k) if layout == "TN" else (k, n)] * len(splits))
    bshapes = [(s, m) if layout == "NT" else (s, k) for s in splits]
    dshapes = [(m if layout == "NT" else s, n) for s in splits]
    torch.manual_seed(456)
    a, aguard = _packed_views(ashapes, torch.bfloat16, prefix=32)
    b, bguard = _packed_views(bshapes, torch.bfloat16, prefix=32)
    d, dguard = _packed_views(dshapes, dtype, float("nan"), prefix=32)
    reference, refguard = _packed_views(dshapes, dtype, float("nan"), prefix=32)
    products = [(y.float().t() if layout == "NT" else y.float()) @
                (x.float().t() if layout == "TN" else x.float()) for x, y in zip(a, b)]
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_BN", "256")
    monkeypatch.setenv("NVTE_GG_FORCE_TILE", "256,256,64,3")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "1")
    general_grouped_gemm(a, b, reference, [None] * len(a), dtype,
                         layout=layout, m_splits=splits, accumulate=False)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "4")
    for _ in range(2):
        general_grouped_gemm(a, b, d, [None] * len(a), dtype,
                             layout=layout, m_splits=splits, accumulate=False)
        torch.cuda.synchronize()
        for actual, expected, product in zip(d, reference, products):
            assert torch.equal(actual, expected)
            torch.testing.assert_close(actual.float(), product, rtol=0.005, atol=0.005)
            if k == 0:
                bits = torch.int32 if dtype == torch.float32 else torch.int16
                assert torch.equal(actual.view(bits), torch.zeros_like(actual.view(bits)))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        general_grouped_gemm(a, b, d, [None] * len(a), dtype,
                             layout=layout, m_splits=splits, accumulate=False)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
    mode = 2 if 0 < k <= 512 and n % 1024 == 0 else 1
    assert len(kernels) == 1 and kernels[0].endswith(_contiguous_ending(dtype, False, mode)), kernels
    assert "tb256x256x64_s3" in kernels[0], kernels
    for guard in (aguard, bguard, dguard, refguard):
        assert torch.all(guard[:32] == 97) and torch.all(guard[-32:] == 97)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("n", [768, 1024])
@pytest.mark.parametrize("k", [0, 256])
def test_contiguous_nexpand_bn128_reject(monkeypatch, layout, dtype, n, k):
    """Former BN128 NE4 is unsupported even when no expert would expand."""
    splits = [k] if layout == "NT" else [256]
    a, b, d, guards = _packed_case(layout, dtype, splits, n=n, k=k)
    before = guards[-1].clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_BN", "128")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "4")
    with pytest.raises(RuntimeError, match="NVTE_GG_CONTIGUOUS_N_EXPAND=4 requires noC BN256"):
        general_grouped_gemm(a, b, d, [None], dtype, layout=layout, m_splits=splits)
    torch.cuda.synchronize()
    assert torch.equal(guards[-1], before)


@pytest.mark.parametrize("option", ["0", "129", "bad", "withC", "tile"])
def test_contiguous_bn256_reject(monkeypatch, option):
    """非法 BN、宽有C或强制tile冲突均在写D前拒绝。"""
    a, _ = _packed_views([(1024, 256)], torch.bfloat16)
    b, _ = _packed_views([(256, 256)], torch.bfloat16)
    d, guard = _packed_views([(256, 1024)], torch.bfloat16)
    before = guard.clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_BN", "256" if option in ("withC", "tile") else option)
    monkeypatch.delenv("NVTE_GG_FORCE_TILE", raising=False)
    if option == "tile":
        monkeypatch.setenv("NVTE_GG_FORCE_TILE", "256,128,64,3")
    error = "fixes tile=256,256,64,3" if option == "tile" else \
        "BN=256 currently supports noC only" if option == "withC" else "BN must be 128 or 256"
    with pytest.raises(RuntimeError, match=f"NVTE_GG_CONTIGUOUS.*{error}"):
        general_grouped_gemm(a, b, d, [None], torch.bfloat16, layout="TN",
                             m_splits=[256], accumulate=option == "withC")
    torch.cuda.synchronize()
    assert torch.equal(guard, before)


def _check_nexpand_kmax(monkeypatch, layout, dtype, bn, kmax, splits, n, k):
    """NE1 bitwise comparison, fp32 oracle, actual symbol and packed guards."""
    m = 512
    ashapes = ([(s, n) for s in splits] if layout == "NT" else
               [(n, k) if layout == "TN" else (k, n)] * len(splits))
    bshapes = [(s, m) if layout == "NT" else (s, k) for s in splits]
    dshapes = [(m if layout == "NT" else s, n) for s in splits]
    torch.manual_seed(2026)
    a, ag = _packed_views(ashapes, torch.bfloat16, prefix=32)
    b, bg = _packed_views(bshapes, torch.bfloat16, prefix=32)
    d, dg = _packed_views(dshapes, dtype, float("nan"), prefix=32)
    reference, rg = _packed_views(dshapes, dtype, float("nan"), prefix=32)
    products = [(y.float().t() if layout == "NT" else y.float()) @
                (x.float().t() if layout == "TN" else x.float()) for x, y in zip(a, b)]
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_BN", str(bn))
    monkeypatch.delenv("NVTE_GG_FORCE_TILE", raising=False)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX", str(kmax))
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "1")
    general_grouped_gemm(a, b, reference, [None] * len(a), dtype,
                         layout=layout, m_splits=splits, accumulate=False)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "4" if bn == 256 else "1")
    for _ in range(2):
        general_grouped_gemm(a, b, d, [None] * len(a), dtype,
                             layout=layout, m_splits=splits, accumulate=False)
        torch.cuda.synchronize()
        for actual, expected, product in zip(d, reference, products):
            assert torch.equal(actual, expected)
            torch.testing.assert_close(actual.float(), product, rtol=0.005, atol=0.005)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        general_grouped_gemm(a, b, d, [None] * len(a), dtype,
                             layout=layout, m_splits=splits, accumulate=False)
        torch.cuda.synchronize()
    eligible = sum(0 < s <= kmax for s in splits) if layout == "NT" else \
        sum(s > 0 for s in splits) if 0 < k <= kmax else 0
    expanded = bn == 256 and eligible > 0 and n % 1024 == 0
    ending = _contiguous_ending(dtype, False, 2 if expanded else 1, kmax)
    kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
    assert len(kernels) == 1 and kernels[0].endswith(ending), kernels
    assert f"tb256x{bn}x64_s3" in kernels[0], kernels
    bits = torch.int32 if dtype == torch.float32 else torch.int16
    for index, actual in enumerate(d):
        if (splits[index] if layout == "NT" else k) == 0:
            assert torch.equal(actual.view(bits), torch.zeros_like(actual.view(bits)))
    for guard in (ag, bg, dg, rg):
        assert torch.all(guard[:32] == 97) and torch.all(guard[-32:] == 97)


@pytest.mark.parametrize("layout", ["TN", "NN"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bn", [128, 256])
@pytest.mark.parametrize("kmax", [512, 1024, 2048])
@pytest.mark.parametrize("delta", [-256, -1, 0, 1, 256])
def test_contiguous_nexpand_kmax_boundary(monkeypatch, layout, dtype, bn, kmax, delta):
    """Aligned threshold neighbours succeed; old +/-1 neighbours explicitly reject."""
    if abs(delta) == 1:
        a, b, d, guards = _packed_case(layout, dtype, [256, 0, 512],
                                       n=bn * 8, k=kmax + delta)
        before = guards[-1].clone()
        monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
        with pytest.raises(ValueError, match="NVTE_GG_CONTIGUOUS.*K must be a multiple of 256"):
            general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout=layout,
                                 m_splits=[256, 0, 512])
        torch.cuda.synchronize()
        assert torch.equal(guards[-1], before)
        return
    _check_nexpand_kmax(monkeypatch, layout, dtype, bn, kmax,
                       [256, 0, 512] * 4, bn * 8, kmax + delta)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bn", [128, 256])
@pytest.mark.parametrize("kmax", [512, 1024, 2048])
@pytest.mark.parametrize("n", [768, 1024, 2048, 2050])
def test_contiguous_nexpand_kmax_mixed(monkeypatch, dtype, bn, kmax, n):
    """K0/aligned neighbours and legal NE1 N; the old N=2050 rejects unchanged."""
    splits = [0, 256, 512, 768, 1024, 1280, 1536, 2048, 2304]
    if n % 256:
        a, b, d, guards = _packed_case("NT", dtype, splits, n=n)
        before = guards[-1].clone()
        monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
        with pytest.raises(ValueError, match="NVTE_GG_CONTIGUOUS.*N must be a multiple of 256"):
            general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout="NT", m_splits=splits)
        torch.cuda.synchronize()
        assert torch.equal(guards[-1], before)
        return
    _check_nexpand_kmax(monkeypatch, "NT", dtype, bn, kmax, splits, n, 0)


@pytest.mark.parametrize("option", ["0", "511", "513", "4096", "bad", "512tail",
                                    " 512", "+512", "0512", "1024.0", "2048\n"])
@pytest.mark.parametrize("ne", ["1", "4"])
def test_contiguous_nexpand_kmax_reject(monkeypatch, option, ne):
    """严格整串解析，NE1也拒绝非法Kmax；D在报错前逐位不变。"""
    a, _ = _packed_views([(1024, 256)], torch.bfloat16)
    b, _ = _packed_views([(256, 256)], torch.bfloat16)
    d, guard = _packed_views([(256, 1024)], torch.bfloat16)
    before = guard.clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_BN", "256")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", ne)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX", option)
    with pytest.raises(RuntimeError, match="NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX"):
        general_grouped_gemm(a, b, d, [None], torch.bfloat16, layout="TN", m_splits=[256])
    torch.cuda.synchronize()
    assert torch.equal(guard, before)


@pytest.mark.parametrize("kmax", ["", "512", "1024", "2048"])
def test_contiguous_nexpand_kmax_with_c_ne1(monkeypatch, kmax):
    """合法Kmax不影响有C普通符号；NE4申请无论shape均拒绝。"""
    a, _ = _packed_views([(256, 512)], torch.bfloat16, 0)
    b, _ = _packed_views([(256, 512)], torch.bfloat16, 1)
    d, guard = _packed_views([(512, 512)], torch.float32)
    before = guard.view(torch.int32).clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX", kmax)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "1")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        general_grouped_gemm(a, b, d, [None], torch.float32, layout="NT",
                             m_splits=[256], accumulate=True)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
    assert len(kernels) == 1 and kernels[0].endswith("_Cfp32_contiguous1_fulltile"), kernels
    assert torch.equal(guard.view(torch.int32), before)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "4")
    with pytest.raises(RuntimeError, match="NVTE_GG_CONTIGUOUS_N_EXPAND.*noC"):
        general_grouped_gemm(a, b, d, [None], torch.float32, layout="NT",
                             m_splits=[256], accumulate=True)
    assert torch.equal(guard.view(torch.int32), before)


def test_contiguous_nexpand_kmax_cache(monkeypatch, tmp_path):
    """三档同进程切换、独立进程磁盘命中与prebuilt exact-symbol。"""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    script = '''
import json, os, torch
from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
A = [torch.full((1024,256), 0.125, device="cuda", dtype=torch.bfloat16)]
B = [torch.full((256,256), 0.0625, device="cuda", dtype=torch.bfloat16)]
D = [torch.full((256,1024), float("nan"), device="cuda", dtype=torch.bfloat16)]
names = []
for kmax in (512,1024,2048,512,2048,1024):
    os.environ["NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX"] = str(kmax)
    general_grouped_gemm(A,B,D,[None],torch.bfloat16,layout="TN",m_splits=[256])
    torch.cuda.synchronize()
    assert torch.all(D[0] == 256/128)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        general_grouped_gemm(A,B,D,[None],torch.bfloat16,layout="TN",m_splits=[256])
        torch.cuda.synchronize()
    kernels = [e.name for e in p.events() if e.name.startswith("nvte_grouped_gemm3_")]
    ending = "_Cvoid_contiguous2" + (f"_kmax{kmax}" if kmax != 512 else "") + "_fulltile"
    assert len(kernels) == 1 and kernels[0].endswith(ending), kernels
    names.append(kernels[0])
os.environ["NVTE_GG_CONTIGUOUS"] = "0"
os.environ["NVTE_GG_CONTIGUOUS_N_EXPAND"] = "bad"
os.environ["NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX"] = "bad"
general_grouped_gemm(A,B,D,[None],torch.bfloat16,layout="TN",m_splits=[256])
torch.cuda.synchronize()
assert torch.all(D[0] == 256/128)
print("CACHE_NAMES=" + json.dumps(names), flush=True)
'''
    env = dict(os.environ, NVTE_GG_CONTIGUOUS="1", NVTE_GG_CONTIGUOUS_BN="256",
               NVTE_GG_CONTIGUOUS_N_EXPAND="4", NVTE_GG_RTC_CACHE_DIR=str(tmp_path / "cache"),
               NVTE_CUTLASS_RTC_DUMP_CUBIN_DIR=str(tmp_path / "dump"),
               NVTE_CUTLASS_GROUPED_GEMM_RTC_VERBOSE="1")
    for key in ("NVTE_GG_FORCE_TILE", "NVTE_GG_DISABLE_RTC_CACHE", "NVTE_GG_RTC_LOAD_CUBIN_DIR"):
        env.pop(key, None)
    logs = []
    for phase in ("compile", "disk", "prebuilt"):
        if phase == "prebuilt":
            env["NVTE_GG_RTC_CACHE_DIR"] = str(tmp_path / "prebuilt_cache")
            env["NVTE_GG_RTC_LOAD_CUBIN_DIR"] = str(tmp_path / "dump")
        proc = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              timeout=900)
        (tmp_path / f"{phase}.log").write_text(proc.stdout)
        assert proc.returncode == 0, proc.stdout[-6000:]
        names = json.loads(next(line.split("=", 1)[1] for line in proc.stdout.splitlines()
                                if line.startswith("CACHE_NAMES=")))
        assert len(set(names)) == 3
        for name in set(names):
            if phase != "compile":
                marker = "loaded cached cubin" if phase == "disk" else "loaded prebuilt cubin"
                assert any(marker in line and name in line for line in proc.stdout.splitlines())
            options = (tmp_path / "dump" / f"{name}.options.txt").read_text()
            macro = "-DNVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX="
            if "_kmax" in name:
                assert macro + name.rsplit("_kmax", 1)[1].split("_", 1)[0] in options
            else:
                assert macro not in options
        logs.append(names)
    assert logs[0] == logs[1] == logs[2]


def _check_contiguous_production_case(monkeypatch, layout, dtype, bn, kmax, splits, n, k,
                                      ne=4, with_c=False):
    """Production columns: repeat/cache identity, initialized accumulators, K0 and fallbacks."""
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("contiguous grouped GEMM requires sm89")
    m = 512
    ashapes = ([(s, n) for s in splits] if layout == "NT" else
               [(n, k) if layout == "TN" else (k, n)] * len(splits))
    bshapes = [(s, m) if layout == "NT" else (s, k) for s in splits]
    dshapes = [(m if layout == "NT" else s, n) for s in splits]
    prefix = 32
    ne = ne if bn == 256 else 1
    torch.manual_seed(1138)
    a, ag = _packed_views(ashapes, torch.bfloat16, prefix=prefix)
    b, bg = _packed_views(bshapes, torch.bfloat16, prefix=prefix)
    d, dg = _packed_views(dshapes, dtype, float("nan"), prefix=prefix)
    products = [(y.float().t() if layout == "NT" else y.float()) @
                (x.float().t() if layout == "TN" else x.float()) for x, y in zip(a, b)]
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_BN", str(bn))
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", str(ne))
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND_K_MAX", str(kmax))
    monkeypatch.delenv("NVTE_GG_FORCE_TILE", raising=False)
    eligible = sum(0 < s <= kmax for s in splits) if layout == "NT" else \
        sum(s > 0 for s in splits) if 0 < k <= kmax else 0
    expanded = not with_c and ne == 4 and eligible > 0 and n % (bn * 4) == 0
    baseline = None
    names = []
    for _ in range(4):
        for tensor in d:
            tensor.fill_(0.75 if with_c else float("nan"))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            general_grouped_gemm(a, b, d, [None] * len(a), dtype,
                                 layout=layout, m_splits=splits, accumulate=with_c)
            torch.cuda.synchronize()
        kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
        ending = _contiguous_ending(dtype, with_c, 2 if expanded else 1, kmax)
        assert len(kernels) == 1 and kernels[0].endswith(ending), kernels
        names.append(kernels[0])
        for actual, product in zip(d, products):
            torch.testing.assert_close(actual.float(), product + (0.75 if with_c else 0),
                                       rtol=0.005, atol=0.005)
        if baseline is None:
            baseline = [tensor.clone() for tensor in d]
        else:
            assert all(torch.equal(tensor, expected) for tensor, expected in zip(d, baseline))
        if not with_c:
            bits = torch.int32 if dtype == torch.float32 else torch.int16
            for index, tensor in enumerate(d):
                if (splits[index] if layout == "NT" else k) == 0:
                    assert torch.equal(tensor.view(bits), torch.zeros_like(tensor.view(bits)))
    assert len(set(names)) == 1
    for guard in (ag, bg, dg):
        assert torch.all(guard[:prefix] == 97) and torch.all(guard[-32:] == 97)


@pytest.mark.parametrize("layout", ["TN", "NN"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bn", [128, 256])
def test_contiguous_production_columns_repeat_cache(monkeypatch, layout, dtype, bn):
    """Aligned columns retain K0/K256, threshold/fallback and repeated symbol identity."""
    for k in (0, 256, 512, 768):
        _check_contiguous_production_case(monkeypatch, layout, dtype, bn, 512,
                                          [256, 0, 512] * 4, bn * 8, k)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_contiguous_production_mixed_kmax(monkeypatch, dtype):
    """Mixed K0/eligible/bypass experts cover all Kmax values with strict aligned D."""
    for kmax in (512, 1024, 2048):
        _check_contiguous_production_case(monkeypatch, "NT", dtype, 256, kmax,
                                          [0, 256, 512, 768, 1024, 1280, 2048, 2304], 1024, 0)


@pytest.mark.parametrize("mode", ["NE1", "N768", "WithC"])
def test_contiguous_production_fallback(monkeypatch, mode):
    """NE1, non-divisible N768 and WithC keep ordinary columns without a toggle."""
    if mode == "WithC":
        _check_contiguous_production_case(monkeypatch, "NT", torch.float32, 128, 512,
                                          [256, 256], 512, 0, ne=1, with_c=True)
    else:
        _check_contiguous_production_case(monkeypatch, "TN", torch.bfloat16, 256, 512,
                                          [256, 512], 768 if mode == "N768" else 1024, 256,
                                          ne=1 if mode == "NE1" else 4)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("splits", [
    pytest.param([256, 200, 0, 64, 248], id="legacy_tails"),
    pytest.param([1, 0, 17, 63], id="legacy_short"),
    pytest.param([257, 255, 1, 0] * 12, id="legacy_many"),
    pytest.param([8, 0, 1], id="legacy_source_map"),
])
def test_contiguous_legacy_irregular_and_ordinary(monkeypatch, layout, dtype, accumulate, splits):
    """Do not lose irregular coverage: strict rejection plus ordinary RTC numerical regression."""
    a, b, d, guards = _packed_case(layout, dtype, splits, m=264, n=520, k=64)
    before = guards[-1].clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    axis = "M" if layout == "NT" or splits[0] % 256 else "N"
    with pytest.raises(ValueError, match=f"NVTE_GG_CONTIGUOUS.*{axis} must be a multiple of 256"):
        general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout=layout,
                             m_splits=splits, accumulate=accumulate)
    torch.cuda.synchronize()
    assert torch.equal(guards[-1], before)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "0")
    products = _products(a, b, layout)
    for _ in range(2):
        old = [view.float().clone() for view in d]
        if not accumulate:
            for view in d:
                view.fill_(float("nan"))
        general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout=layout,
                             m_splits=splits, accumulate=accumulate, grad=layout == "NT")
        torch.cuda.synchronize()
        for actual, product, source in zip(d, products, old):
            torch.testing.assert_close(actual.float(), product + source if accumulate else product,
                                       rtol=0.005, atol=0.005)
            assert torch.isfinite(actual).all()
    _assert_guards(guards)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("m,n,k", [
    (512, 1024, 1), (512, 2048, 65), (512, 1024, 513), (512, 512, 65),
    (512, 513, 7), (512, 514, 65), (512, 520, 64),
    (257, 512, 65), (269, 513, 7), (512, 1024, 1025), (512, 1024, 2049),
])
def test_contiguous_legacy_partial_tiles_reject(monkeypatch, layout, dtype, accumulate, m, n, k):
    """Retain former short/tail/stride cases as explicit strict-contract rejections."""
    splits = [k] if layout == "NT" else [m]
    a, b, d, guards = _packed_case(layout, dtype, splits, m=m, n=n, k=k)
    axis = next(axis for axis, extent in zip("MNK", (m, n, k)) if extent % 256)
    before = guards[-1].clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    with pytest.raises(ValueError, match=f"NVTE_GG_CONTIGUOUS.*{axis} must be a multiple of 256"):
        general_grouped_gemm(a, b, d, [None], dtype, layout=layout,
                             m_splits=splits, accumulate=accumulate)
    torch.cuda.synchronize()
    assert torch.equal(guards[-1], before)
    _assert_guards(guards)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("boundary", ["python", "extension"])
@pytest.mark.parametrize("axis", ["M", "N", "K", "split"])
def test_contiguous_per_group_contract(monkeypatch, layout, dtype, accumulate, boundary, axis):
    """Totals being divisible by 256 never licenses an irregular active expert."""
    splits, m, n, k = [256, 768], 512, 512, 256
    if axis == "M":
        if layout == "NT": m = 128
        else: splits = [256, 128, 128]  # total 512, but groups 1/2 are invalid
    elif axis == "N": n = 128
    elif axis == "K":
        if layout == "NT": splits = [256, 128, 128]
        else: k = 128
    a, b, d, guards = _packed_case(layout, dtype, splits, m=m, n=n, k=k)
    provided = [512, 512] if axis == "split" else splits  # same total, wrong per expert
    before = guards[-1].clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    if boundary == "python":
        error = ("split 512 does not match logical " + ("K" if layout == "NT" else "M")) \
            if axis == "split" else f"{axis} must be a multiple of 256"
        with pytest.raises(ValueError, match=f"NVTE_GG_CONTIGUOUS.*{error}"):
            general_grouped_gemm(a, b, d, [None] * len(a), dtype, layout=layout,
                                 m_splits=provided, accumulate=accumulate)
    else:
        ws = torch.empty(4096, device="cuda", dtype=torch.uint8)
        error = "split must match K for NT or M otherwise" if axis == "split" else \
            "requires M, N, and K divisible by 256 for nonempty output"
        with pytest.raises(RuntimeError, match=f"NVTE_GG_CONTIGUOUS.*{error}"):
            _contiguous_raw_call(a, b, d, [ws], ws.numel(), provided, accumulate, layout, dtype)
    torch.cuda.synchronize()
    assert torch.equal(guards[-1], before)
    _assert_guards(guards)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("invalid", ["length", "negative", "bool", "float", "tensor_dtype", "tensor_rank"])
def test_contiguous_split_metadata_reject(monkeypatch, layout, invalid):
    a, b, d, guards = _packed_case(layout, torch.bfloat16, [256, 256])
    provided = {
        "length": [256], "negative": [256, -256], "bool": [256, True],
        "float": [256, 256.0], "tensor_dtype": torch.tensor([256., 256.]),
        "tensor_rank": torch.tensor([[256, 256]]),
    }[invalid]
    error = "splits must be nonnegative integers" if invalid in ("negative", "bool", "float") else \
        "requires one integer split per expert"
    before = guards[-1].clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    with pytest.raises(ValueError, match=f"NVTE_GG_CONTIGUOUS.*{error}"):
        general_grouped_gemm(a, b, d, [None] * 2, torch.bfloat16, layout=layout, m_splits=provided)
    torch.cuda.synchronize()
    assert torch.equal(guards[-1], before)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("representation", ["omitted", "tuple", "cpu_tensor", "cuda_tensor", "raw_omitted"])
def test_contiguous_split_metadata_success(monkeypatch, layout, representation):
    splits = [256, 768, 0, 512]
    a, b, d, guards = _packed_case(layout, torch.float32, splits)
    provided = None
    if representation == "tuple": provided = tuple(splits)
    elif representation.endswith("tensor"):
        provided = torch.tensor(splits, dtype=torch.int64,
                                device="cuda" if representation == "cuda_tensor" else "cpu")
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    if representation == "raw_omitted":
        ws = torch.empty(4096, device="cuda", dtype=torch.uint8)
        _contiguous_raw_call(a, b, d, [ws], ws.numel(), [], layout=layout)
    else:
        general_grouped_gemm(a, b, d, [None] * len(a), torch.float32, layout=layout, m_splits=provided)
    torch.cuda.synchronize()
    for actual, product in zip(d, _products(a, b, layout)):
        torch.testing.assert_close(actual.float(), product, rtol=0.005, atol=0.005)
    _assert_guards(guards)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("zero_k", [False, True])
def test_contiguous_raw_numerical(monkeypatch, layout, dtype, accumulate, zero_k):
    splits = [0, 0] if zero_k and layout == "NT" else [256, 768, 0, 512]
    a, b, d, guards = _packed_case(layout, dtype, splits, k=0 if zero_k else 256)
    products = _products(a, b, layout)
    inputs_before = [buf.clone() for buf in guards[:2]]
    ws = torch.empty(4096, device="cuda", dtype=torch.uint8)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    for _ in range(2):
        source = [t.float().clone() for t in d]
        if not accumulate:
            for t in d: t.fill_(float("nan"))
        _contiguous_raw_call(a, b, d, [ws], ws.numel(), splits, accumulate, layout, dtype)
        torch.cuda.synchronize()
        for actual, product, old in zip(d, products, source):
            torch.testing.assert_close(actual.float(), product + old if accumulate else product,
                                       rtol=0.005, atol=0.005)
    assert all(torch.equal(buf, old) for buf, old in zip(guards[:2], inputs_before))
    _assert_guards(guards)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("operand", ["A", "B", "D"])
@pytest.mark.parametrize("invalid", ["pointer", "stride", "separate", "gap", "empty_storage", "negative_view"])
def test_contiguous_host_packed_storage(monkeypatch, layout, dtype, accumulate, operand, invalid):
    """Bypass Python: every operand, including empty views, retains the native storage contract."""
    splits = [0, 0] if invalid == "empty_storage" else [256, 256]
    # NT needs M=0 to make D empty too, while its K remains a legitimate ragged axis.
    a, b, d, guards = _packed_case(layout, dtype, splits, m=0 if invalid == "empty_storage" else 512)
    tensors = {"A": a, "B": b, "D": d}[operand]
    if invalid == "pointer":
        tensors[:], backing = _packed_views([tuple(t.shape) for t in tensors], tensors[0].dtype, prefix=33)
        if operand == "D": guards = (*guards[:-1], backing)
        error = "nonnull 16-byte aligned data pointer"
    elif invalid == "stride":
        tensors[0] = tensors[0].t()
        error = "requires contiguous rank-2 CUDA storage"
    elif invalid == "negative_view":
        tensors[0] = torch._neg_view(tensors[0])
        error = "without lazy negative/conjugate bits"
    elif invalid == "gap":
        shapes = [tuple(t.shape) for t in tensors]
        views, backing = _packed_views([shapes[0], (1, 8), shapes[1]], tensors[0].dtype)
        tensors[:] = [views[0], views[2]]
        if operand == "D": guards = (*guards[:-1], backing)
        error = "must share one storage allocation with adjacent storage_offset values"
    else:
        tensors[1] = tensors[1].clone()
        error = "must share one storage allocation with adjacent storage_offset values"
    before = guards[-1].clone()
    ws = torch.empty(4096, device="cuda", dtype=torch.uint8)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    with pytest.raises(RuntimeError, match=f"NVTE_GG_CONTIGUOUS.*{error}"):
        _contiguous_raw_call(a, b, d, [ws], ws.numel(), splits, accumulate, layout, dtype)
    torch.cuda.synchronize()
    assert torch.equal(guards[-1], before)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("boundary", ["python", "extension"])
@pytest.mark.parametrize("empty_axis", ["M", "N"])
def test_contiguous_structurally_empty_output(monkeypatch, layout, dtype, accumulate, boundary, empty_axis):
    """Non-256 dimensions are legal when D has no elements, after structural validation."""
    m, n, k = (0, 128, 7) if empty_axis == "M" else (128, 0, 7)
    splits = [k, k] if layout == "NT" else [m, m]
    a, b, d, guards = _packed_case(layout, dtype, splits, m=m, n=n, k=k)
    before = [buf.clone() for buf in guards]
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    if boundary == "python":
        general_grouped_gemm(a, b, d, [None] * 2, dtype, layout=layout,
                             m_splits=splits, accumulate=accumulate)
    else:
        ws = torch.empty(4096, device="cuda", dtype=torch.uint8)
        _contiguous_raw_call(a, b, d, [ws], ws.numel(), splits, accumulate, layout, dtype)
    torch.cuda.synchronize()
    assert all(t.numel() == 0 for t in d)
    assert all(torch.equal(buf, old) for buf, old in zip(guards, before))


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("boundary", ["python", "extension"])
@pytest.mark.parametrize("invalid", ["shape", "fixed", "split", "storage", "dtype"])
def test_contiguous_empty_d_still_validates(monkeypatch, layout, boundary, invalid):
    """Empty D skips divisibility only, not logical shape, fixed axes, split or storage checks."""
    splits = [0, 0] if layout != "NT" else [7, 7]
    a, b, d, guards = _packed_case(layout, torch.bfloat16, splits, m=0, n=128, k=7)
    provided = list(splits)
    if invalid == "shape":
        d[1] = d[1].view(0, 129)
    elif invalid == "fixed":
        # Both outputs stay empty, but the fixed N differs.
        shapes = [(128, 7), (129, 7)] if layout == "TN" else \
            [(7, 128), (7, 129)]
        a, _ = _packed_views(shapes, torch.bfloat16)
        d[1] = d[1].view(0, 129)
    elif invalid == "split": provided[1] += 1
    elif invalid == "storage": d[1] = d[1].clone()
    elif invalid == "dtype": d[1] = d[1].float()
    before = guards[-1].clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    python_error = {"shape": "incompatible matrix dimensions", "fixed": "requires fixed M/N",
                    "split": "does not match logical", "storage": "must be packed adjacent views",
                    "dtype": "must be packed adjacent views"}[invalid]
    native_error = {"shape": "incompatible matrix dimensions", "fixed": "requires fixed M/N",
                    "split": "split must match", "storage": "must share one storage allocation",
                    "dtype": "requires contiguous rank-2 CUDA storage"}[invalid]
    if boundary == "python":
        with pytest.raises(ValueError, match=f"NVTE_GG_CONTIGUOUS.*{python_error}"):
            general_grouped_gemm(a, b, d, [None] * 2, torch.bfloat16,
                                 layout=layout, m_splits=provided)
    else:
        ws = torch.empty(4096, device="cuda", dtype=torch.uint8)
        with pytest.raises(RuntimeError, match=f"NVTE_GG_CONTIGUOUS.*{native_error}"):
            _contiguous_raw_call(a, b, d, [ws], ws.numel(), provided, layout=layout)
    torch.cuda.synchronize()
    assert torch.equal(guards[-1], before)


def _grouped_storage(views, backing):
    shapes = [tuple(t.shape) for t in views]
    offsets = [0]
    for t in views:
        offsets.append(offsets[-1] + t.numel())
    logical = (sum(s[0] for s in shapes), shapes[0][1])
    if any(s[1] != logical[1] for s in shapes): logical = (1, offsets[-1])
    return GroupedTensorStorage(
        logical, backing.dtype, num_tensors=len(views), shapes=shapes,
        data=backing[32:-32], offsets=list(offsets),
        first_dims=torch.tensor([s[0] for s in shapes], device="cuda", dtype=torch.int64),
        last_dims=torch.tensor([s[1] for s in shapes], device="cuda", dtype=torch.int64),
        tensor_offsets=torch.tensor(offsets, device="cuda", dtype=torch.int64),
    )


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("representation", ["ragged", "uniform_implicit", "parameters",
                                             "discrete_in", "discrete_out", "all_empty", "zero_k"])
def test_contiguous_grouped_storage_adapter(monkeypatch, layout, dtype, accumulate, representation):
    splits = [256, 768, 0, 512]
    if representation in ("uniform_implicit", "parameters"): splits = [256, 256]
    elif representation == "all_empty": splits = [0, 0]
    a, b, d, guards = _packed_case(layout, dtype, splits,
                                   m=0 if representation == "all_empty" else 512,
                                   k=0 if representation == "zero_k" else 256)
    if representation == "zero_k" and layout == "NT":
        a, b, d, guards = _packed_case(layout, dtype, [0, 0])
    operands = [_grouped_storage(views, backing) for views, backing in zip((a, b, d), guards)]
    if representation == "uniform_implicit":
        for operand in operands:
            operand.first_dims = operand.last_dims = operand.tensor_offsets = None
    elif representation == "parameters":
        operands[0] = [torch.nn.Parameter(t, requires_grad=False) for t in a]
        operands[2] = [torch.nn.Parameter(t, requires_grad=False) for t in d]
    elif representation == "discrete_in":
        operands[0] = a
    elif representation == "discrete_out":
        operands[2] = d
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    products = _products(a, b, layout)
    input_before = [guard.clone() for guard in guards[:2]]
    for _ in range(2):
        source = [t.float().clone() for t in d]
        if not accumulate:
            for t in d: t.fill_(float("nan"))
        result = general_grouped_gemm_for_grouped_tensor(*operands, layout=layout, accumulate=accumulate)
        torch.cuda.synchronize()
        assert result is operands[2]
        for actual, product, old in zip(d, products, source):
            torch.testing.assert_close(actual.float(), product + old if accumulate else product,
                                       rtol=0.005, atol=0.005)
        assert all(torch.equal(buf, old) for buf, old in zip(guards[:2], input_before))
    _assert_guards(guards)


@pytest.mark.parametrize("operand_index", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("damage", ["logical", "logical_rank", "logical_negative", "negative",
                                     "first_live", "last_live", "first_dtype", "last_count", "csr_dtype",
                                     "csr_cpu", "csr_start", "csr_end", "csr_empty", "offset_cache", "shape_cache",
                                     "actual_dtype", "data_rank", "data_size", "data_backing",
                                     "first_backing", "last_backing", "csr_backing",
                                     "first_lazy", "last_lazy", "csr_lazy"])
def test_contiguous_grouped_storage_corruption(monkeypatch, operand_index, dtype, damage):
    a, b, d, guards = _packed_case("NT", dtype, [256, 0, 512])
    operands = [_grouped_storage(views, backing) for views, backing in zip((a, b, d), guards)]
    target = operands[operand_index]
    error = "ragged shape/offset/storage mismatch"
    if damage == "logical": target.logical_shape = (target.logical_shape[0] + 1, target.logical_shape[1])
    elif damage == "logical_rank":
        target.logical_shape = (*target.logical_shape, 1)
        error = "invalid grouped logical shape"
    elif damage == "logical_negative":
        target.logical_shape = (-1, target.logical_shape[1])
        error = "invalid grouped logical shape"
    elif damage == "first_live": target.first_dims[0] += 256
    elif damage == "last_live": target.last_dims[0] += 256
    elif damage == "negative":
        target.first_dims[0] = -1
        error = "negative expert dimension"
    elif damage == "first_dtype":
        target.first_dims = target.first_dims.int()
        error = "invalid first_dims"
    elif damage == "last_count":
        target.last_dims = target.last_dims[:-1]
        error = "invalid last_dims"
    elif damage == "csr_dtype":
        target.tensor_offsets = target.tensor_offsets.int()
        error = "invalid tensor_offsets"
    elif damage == "csr_cpu":
        target.tensor_offsets = target.tensor_offsets.cpu()
        error = "invalid tensor_offsets"
    elif damage == "csr_start": target.tensor_offsets[0] = 1
    elif damage == "csr_end": target.tensor_offsets[-1] -= 1
    elif damage == "csr_empty": target.tensor_offsets[2] += 1
    elif damage == "offset_cache": target.offsets[1] += 1
    elif damage == "shape_cache": target.tensor_shapes[1] = (256, 512)
    elif damage == "actual_dtype":
        target.fake_dtype = torch.float32 if target.get_dtype() == torch.bfloat16 else torch.bfloat16
        error = "matching logical and actual dtype"
    elif damage == "data_rank":
        target.rowwise_data = target.rowwise_data.view(1, -1)
        error = "matching logical and actual dtype"
    elif damage == "data_size": target.rowwise_data = target.rowwise_data[:-1]
    elif damage == "data_backing":
        data = torch.empty(target.rowwise_data.numel() + 1, device="cuda", dtype=target.get_dtype())[1:]
        data.untyped_storage().resize_(0)
        target.rowwise_data = data
        error = "rowwise data exceeds storage capacity"
    elif damage.endswith(("_backing", "_lazy")):
        field = {"first": "first_dims", "last": "last_dims", "csr": "tensor_offsets"}[damage.split("_")[0]]
        metadata = getattr(target, field).clone()
        if damage.endswith("_backing"):
            metadata.untyped_storage().resize_(8)
            error = f"{field} exceeds storage capacity"
        else:
            metadata = torch._neg_view(metadata)
            error = f"invalid {field}"
        setattr(target, field, metadata)
    before = guards[-1].clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    with pytest.raises(ValueError, match=f"NVTE_GG_CONTIGUOUS.*{error}"):
        general_grouped_gemm_for_grouped_tensor(*operands, layout="NT", accumulate=True)
    torch.cuda.synchronize()
    assert torch.equal(guards[-1], before)


@pytest.mark.parametrize("option", ["bias", "bias_scale", "alpha", "beta", "grad"])
def test_contiguous_grouped_storage_reject_api_options(monkeypatch, option):
    a, b, d, guards = _packed_case("TN", torch.bfloat16, [256, 768, 0, 512])
    operands = [_grouped_storage(views, backing) for views, backing in zip((a, b, d), guards)]
    value = True if option == "grad" else torch.ones(1, device="cuda", dtype=torch.float32)
    before = guards[-1].clone()
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        with pytest.raises(ValueError, match="NVTE_GG_CONTIGUOUS.*fusion or custom alpha/beta"):
            general_grouped_gemm_for_grouped_tensor(*operands, layout="TN", **{option: value})
        torch.cuda.synchronize()
    assert not [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")]
    assert torch.equal(guards[-1], before)


@pytest.mark.parametrize("env", [None, "0", "1"], ids=["unset", "zero", "explicit"])
@pytest.mark.parametrize("ppu,cap,device,supported,default", [
    ("selection-spec", (8, 9), "cuda:3", True, True),
    ("", (8, 9), "cuda:3", True, True),  # Presence, not marker truthiness.
    (None, (8, 9), "cuda:3", True, False),
    ("selection-spec", (8, 0), "cuda:3", True, False),
    ("selection-spec", (9, 0), "cuda:3", True, False),
    ("selection-spec", (8, 9), "cpu", True, False),
    ("selection-spec", (8, 9), "cuda:3", False, False),
])
def test_grouped_gemm_policy_selection(monkeypatch, env, ppu, cap, device, supported, default):
    """Env=0 is not an opt-out; env=1 overrides even the supported/build/device gates."""
    if env is None:
        monkeypatch.delenv("NVTE_GG_CONTIGUOUS", raising=False)
    else:
        monkeypatch.setenv("NVTE_GG_CONTIGUOUS", env)
    monkeypatch.setattr(torch.version, "ppu", ppu, raising=False)
    capability = Mock(return_value=cap)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    device = torch.device(device)
    assert gemm_module._use_contiguous_grouped_gemm(device, supported=supported) is (
        env == "1" or default
    )
    if env != "1" and supported and ppu is not None and device.type == "cuda":
        capability.assert_called_once_with(device)
    else:
        capability.assert_not_called()


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("representation", ["grouped", "discrete_in", "discrete_out"])
@pytest.mark.parametrize("env,ppu,cap,choice,old_flags,contiguous", [
    pytest.param(None, "selection-spec", (8, 9), None, "0", True, id="ppu-unset-default"),
    pytest.param("0", "selection-spec", (8, 9), None, "0", True, id="ppu-zero-default"),
    pytest.param("0", None, (8, 9), None, "0", False, id="nonppu-ada-original"),
    pytest.param("0", "selection-spec", (8, 0), None, "0", False, id="ppu-non89-original"),
    pytest.param("0", None, (8, 9), True, None, True, id="true-old-flags-unset"),
    pytest.param("0", None, (8, 9), True, "0", True, id="true-old-flags-zero"),
    pytest.param("1", "selection-spec", (8, 9), False, "0", False, id="false-overrides-env"),
])
def test_contiguous_grouped_wrapper_dispatch(
    monkeypatch, layout, representation, env, ppu, cap, choice, old_flags, contiguous,
):
    """Spy on the exact ABI; never execute cuBLAS on a simulated architecture."""
    splits = [256, 512]
    a, b, d, guards = _packed_case(layout, torch.float32, splits, m=256, n=256)
    operands = [_grouped_storage(views, backing) for views, backing in zip((a, b, d), guards)]
    if representation == "discrete_in": operands[0] = a
    elif representation == "discrete_out": operands[2] = d
    if env is None:
        monkeypatch.delenv("NVTE_GG_CONTIGUOUS", raising=False)
    else:
        monkeypatch.setenv("NVTE_GG_CONTIGUOUS", env)
    for name in ("NVTE_USE_CUTLASS_GROUPED_GEMM", "NVTE_USE_CUTLASS_GROUPED_GEMM_RTC",
                 "NVTE_GG_ALLOW_FP32_ACCUM"):
        if old_flags is None: monkeypatch.delenv(name, raising=False)
        else: monkeypatch.setenv(name, old_flags)
    monkeypatch.setattr(torch.version, "ppu", ppu, raising=False)
    # A no-argument/current-device query deliberately reports a different architecture.
    capability = Mock(side_effect=lambda device=None: (10, 0) if device is None else cap)
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    monkeypatch.setattr(gemm_module, "get_sm_count", lambda: 16)
    monkeypatch.setenv("NVTE_EXT_MARGIN_SM", "3")
    workspace, setup = object(), object()
    setup_getter = Mock(return_value=setup)
    monkeypatch.setattr(gemm_module, "_get_grouped_gemm_setup_workspace", setup_getter)
    monkeypatch.setattr(gemm_module, "_get_grouped_cublas_workspace", Mock(return_value=workspace))
    alpha = torch.ones(2, device="cuda", dtype=torch.float32)
    monkeypatch.setattr(gemm_module, "_get_fp32_ones_tensor", Mock(return_value=alpha))
    dedicated = Mock()
    monkeypatch.setattr(tex, "te_contiguous_grouped_gemm", dedicated)
    original_names = {
        "grouped": "te_general_grouped_gemm_for_grouped_tensor",
        "discrete_in": "te_general_grouped_gemm_for_discrete_in",
        "discrete_out": "te_general_grouped_gemm_for_discrete_out",
    }
    originals = {key: Mock(return_value=operands[2]) for key in original_names}
    for key, name in original_names.items(): monkeypatch.setattr(tex, name, originals[key])
    list_api = Mock(side_effect=AssertionError("GroupedTensor must not trampoline through list API"))
    monkeypatch.setattr(tex, "te_general_grouped_gemm", list_api)
    result = general_grouped_gemm_for_grouped_tensor(
        *operands, layout=layout, accumulate=True, use_split_accumulator=True,
        use_contiguous_grouped_gemm=choice,
    )
    assert result is operands[2]
    list_api.assert_not_called()
    if contiguous:
        dedicated.assert_called_once()
        args = dedicated.call_args.args
        assert len(args) == 9 and not dedicated.call_args.kwargs
        for actual, expected in zip((args[0], args[2], args[4]), (a, b, d)):
            assert all(x.data_ptr() == y.data_ptr() for x, y in zip(actual, expected))
            assert len(actual) == len(expected)
        assert args[1] is (layout[0] == "T") and args[3] is (layout[1] == "T")
        assert args[5] == splits and args[6] is workspace and args[7] is True and args[8] == 13
        setup_getter.assert_not_called()
        for original in originals.values(): original.assert_not_called()
        if choice is None and env != "1":
            capability.assert_called_once_with(operands[1].rowwise_data.device)
        else:
            capability.assert_not_called()
    else:
        dedicated.assert_not_called()
        originals[representation].assert_called_once_with(
            operands[0], layout[0] == "T", operands[1], layout[1] == "T", operands[2],
            None, None, alpha, alpha, setup, workspace, True, 13,
        )
        for key, original in originals.items():
            if key != representation: original.assert_not_called()
        setup_getter.assert_called_once_with(operands[1].rowwise_data.device.index, len(splits))


@pytest.mark.parametrize("invalid,error", [
    ("grad", "grad is not supported"),
    ("both_discrete", "Both A and out are discrete"),
    ("bias_out", "Bias is not supported when out is a list"),
    ("bias_scale", "bias_scale requires bias"),
    ("cpu_alpha", "alpha and beta must be CUDA tensors"),
])
def test_contiguous_grouped_wrapper_false_api_guards(monkeypatch, invalid, error):
    """Explicit False retains original errors, even with both default selectors enabled."""
    a, b, d, guards = _packed_case("TN", torch.bfloat16, [256, 256], n=256)
    operands = [_grouped_storage(views, backing) for views, backing in zip((a, b, d), guards)]
    kwargs = {"use_contiguous_grouped_gemm": False}
    if invalid == "grad": kwargs["grad"] = True
    elif invalid == "both_discrete": operands[0], operands[2] = a, d
    elif invalid == "bias_out":
        operands[2], kwargs["bias"] = d, object()
    elif invalid == "bias_scale": kwargs["bias_scale"] = object()
    elif invalid == "cpu_alpha":
        kwargs["alpha"] = kwargs["beta"] = torch.ones(1, dtype=torch.float32)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "1")
    monkeypatch.setattr(torch.version, "ppu", "selection-spec", raising=False)
    dedicated = Mock()
    monkeypatch.setattr(tex, "te_contiguous_grouped_gemm", dedicated)
    before = guards[-1].clone()
    with pytest.raises(NotImplementedError if invalid == "grad" else ValueError, match=error):
        general_grouped_gemm_for_grouped_tensor(*operands, **kwargs)
    dedicated.assert_not_called()
    assert torch.equal(guards[-1], before)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("invalid", ["length", "negative", "mismatch", "bool", "float", "tensor"])
def test_contiguous_dedicated_split_guards(monkeypatch, layout, invalid):
    """The dedicated pybind entry validates splits without an env-selected legacy trampoline."""
    a, b, d, guards = _packed_case(layout, torch.float32, [256, 512], m=256, n=256)
    provided = {
        "length": [256], "negative": [256, -512], "mismatch": [512, 256],
        "bool": [256, True], "float": [256, 512.0], "tensor": torch.tensor([256, 512]),
    }[invalid]
    error = {
        "length": "one entry per group", "negative": "split must match",
        "mismatch": "split must match", "bool": "nonnegative integers, not bools",
        "float": "nonnegative integers, not bools", "tensor": "integer split list",
    }[invalid]
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "0")
    ws = torch.full((4096,), 73, device="cuda", dtype=torch.uint8)
    before, before_ws = guards[-1].clone(), ws.clone()
    with pytest.raises(RuntimeError, match=f"NVTE_GG_CONTIGUOUS.*{error}"):
        tex.te_contiguous_grouped_gemm(
            a, layout[0] == "T", b, layout[1] == "T", d, provided, ws, False, 0,
        )
    torch.cuda.synchronize()
    assert torch.equal(guards[-1], before) and torch.equal(ws, before_ws)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("operand", ["A", "B", "D"])
@pytest.mark.parametrize("invalid", ["separate", "gap"])
def test_contiguous_dedicated_packing_guards(monkeypatch, layout, operand, invalid):
    a, b, d, guards = _packed_case(layout, torch.float32, [256, 256], m=256, n=256)
    tensors = {"A": a, "B": b, "D": d}[operand]
    if invalid == "separate":
        tensors[1] = tensors[1].clone()
    else:
        views, backing = _packed_views(
            [tuple(tensors[0].shape), (1, 8), tuple(tensors[1].shape)], tensors[0].dtype,
        )
        tensors[:] = [views[0], views[2]]
        if operand == "D": guards = (*guards[:-1], backing)
    monkeypatch.delenv("NVTE_GG_CONTIGUOUS", raising=False)
    ws = torch.full((4096,), 73, device="cuda", dtype=torch.uint8)
    before, before_ws = guards[-1].clone(), ws.clone()
    outputs_before = [t.clone() for t in d]
    with pytest.raises(RuntimeError, match="NVTE_GG_CONTIGUOUS.*must share one storage allocation"):
        tex.te_contiguous_grouped_gemm(
            a, layout[0] == "T", b, layout[1] == "T", d, [256, 256], ws, True, 0,
        )
    torch.cuda.synchronize()
    assert torch.equal(guards[-1], before) and torch.equal(ws, before_ws)
    assert all(torch.equal(t, old) for t, old in zip(d, outputs_before))


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulate", [False, True])
def test_contiguous_dedicated_raw_numerical(monkeypatch, layout, dtype, accumulate):
    """Dedicated launch ignores old enable flags and preserves empty experts and packed guards."""
    splits = [256, 0, 512]
    a, b, d, guards = _packed_case(layout, dtype, splits, m=256, n=256)
    products = _products(a, b, layout)
    ws = torch.empty(4096, device="cuda", dtype=torch.uint8)
    inputs_before = [buf.clone() for buf in guards[:2]]
    for env in (None, "0"):
        if env is None: monkeypatch.delenv("NVTE_GG_CONTIGUOUS", raising=False)
        else: monkeypatch.setenv("NVTE_GG_CONTIGUOUS", env)
        for name in ("NVTE_USE_CUTLASS_GROUPED_GEMM", "NVTE_USE_CUTLASS_GROUPED_GEMM_RTC",
                     "NVTE_GG_ALLOW_FP32_ACCUM"):
            if env is None: monkeypatch.delenv(name, raising=False)
            else: monkeypatch.setenv(name, "0")
        source = [t.float().clone() for t in d]
        if not accumulate:
            for t in d: t.fill_(float("nan"))
        result = tex.te_contiguous_grouped_gemm(
            a, layout[0] == "T", b, layout[1] == "T", d, tuple(splits), ws, accumulate, 0,
        )
        torch.cuda.synchronize()
        assert result is None
        for actual, product, old in zip(d, products, source):
            torch.testing.assert_close(actual.float(), product + old if accumulate else product,
                                       rtol=0.005, atol=0.005)
    assert all(torch.equal(buf, old) for buf, old in zip(guards[:2], inputs_before))
    _assert_guards(guards)


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
def test_contiguous_grouped_wrapper_list_isolation(monkeypatch, layout):
    """Back-to-back dedicated/default wrappers cannot change the ordinary list API's selection."""
    splits = [256, 0, 512]
    a, b, d, guards = _packed_case(layout, torch.float32, splits, m=256, n=256)
    operands = [_grouped_storage(views, backing) for views, backing in zip((a, b, d), guards)]
    ordinary_splits = [256, 200, 64]
    la, lb, ld, _ = _packed_case(layout, torch.float32, ordinary_splits, m=256, n=256)
    # Independent allocations and non-256 splits must remain legal for the list API.
    la, lb, ld = ([t.clone() for t in tensors] for tensors in (la, lb, ld))
    products, ordinary_products = _products(a, b, layout), _products(la, lb, layout)
    monkeypatch.setenv("NVTE_GG_CONTIGUOUS", "0")
    monkeypatch.setattr(torch.version, "ppu", "selection-spec", raising=False)
    dedicated = Mock(wraps=tex.te_contiguous_grouped_gemm)
    list_api = Mock(wraps=tex.te_general_grouped_gemm)
    monkeypatch.setattr(tex, "te_contiguous_grouped_gemm", dedicated)
    monkeypatch.setattr(tex, "te_general_grouped_gemm", list_api)
    for choice in (True, None, True):
        monkeypatch.delenv("NVTE_GG_CONTIGUOUS_N_EXPAND", raising=False)
        before_env = dict(os.environ)
        for t in d: t.fill_(float("nan"))
        assert general_grouped_gemm_for_grouped_tensor(
            *operands, layout=layout, use_contiguous_grouped_gemm=choice,
        ) is operands[2]
        assert dict(os.environ) == before_env
        # Strict-only options must not leak into ordinary list launches either.
        monkeypatch.setenv("NVTE_GG_CONTIGUOUS_N_EXPAND", "bad")
        before_env = dict(os.environ)
        for t in ld: t.fill_(float("nan"))
        general_grouped_gemm(la, lb, ld, [None] * len(la), torch.float32,
                             layout=layout, m_splits=ordinary_splits, grad=layout == "NT")
        torch.cuda.synchronize()
        assert dict(os.environ) == before_env
        for tensors, references in ((d, products), (ld, ordinary_products)):
            for actual, expected in zip(tensors, references):
                torch.testing.assert_close(actual.float(), expected, rtol=0.005, atol=0.005)
        _assert_guards(guards)
    assert dedicated.call_count == 3 and list_api.call_count == 3
