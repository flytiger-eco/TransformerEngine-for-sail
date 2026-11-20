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

import importlib

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.cpp_extensions.gemm import general_grouped_gemm
from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import GroupedTensorStorage

gemm_ext = importlib.import_module("transformer_engine.pytorch.cpp_extensions.gemm")

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
    yield
    torch.cuda.synchronize()


def test_grouped_workspace_stream_cache(monkeypatch):
    """Pointer ranges, not timing, prove isolation even when launches serialize."""
    device = torch.cuda.current_device()
    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    setup = gemm_ext._get_grouped_gemm_setup_workspace
    setup.cache_clear()
    size_requests = []

    def setup_size(groups):
        size_requests.append(groups)
        return groups * 256

    # Test the cache independently of compile-time cuBLAS grouped-setup support.
    monkeypatch.setattr(gemm_ext, "get_grouped_gemm_setup_workspace_size", setup_size)
    snapshots, ordinary = [], []
    try:
        for stream in streams:
            with torch.cuda.stream(stream):
                ws = gemm_ext.get_cublas_workspace(device, False, True)
                assert ws is gemm_ext.get_cublas_workspace(device, False, True)
                assert len(ws) == gemm_ext.tex.get_num_cublas_streams()
                assert all(buf.numel() > 0 for buf in ws)
                buffers = list(ws)
                for groups in (3, 5):
                    buf = setup(device, groups, stream.cuda_stream)
                    assert buf is setup(device, groups, stream.cuda_stream)
                    buffers.append(buf)
                for layout in ("TN", "NN", "NT"):
                    buf = gemm_ext._get_grouped_cublas_workspace(device, layout, stream.cuda_stream)
                    assert buf.numel() > 0
                    assert buf is gemm_ext._get_grouped_cublas_workspace(
                        device, layout, stream.cuda_stream
                    )
                    buffers.append(buf)
                snapshots.append(buffers)
                ordinary.append([gemm_ext.get_cublas_workspace(device, ub, False)
                                 for ub in (False, True)])
        assert all(a is b for a, b in zip(*ordinary))  # Non-grouped/UB remain device-cached.
        assert size_requests == [3, 5, 3, 5]
        nonempty = [buf for buffers in snapshots for buf in buffers if buf.numel()]
        for i, left in enumerate(nonempty):
            for right in nonempty[i + 1:]:
                assert (left.data_ptr() + left.numel() <= right.data_ptr()
                        or right.data_ptr() + right.numel() <= left.data_ptr())
    finally:
        # Never leave mocked setup capacities in the production cache.
        setup.cache_clear()


@pytest.mark.parametrize("path", ["ordinary", "grouped_tensor"])
def test_grouped_gemm_two_streams(path):
    """Interleave groups and workspaces on two caller streams."""
    if path == "grouped_tensor" and torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("The grouped-tensor multi-tensor adapter requires PPU sm89")
    torch.manual_seed(5702)
    streams = [torch.cuda.Stream() for _ in range(2)]
    cases = []

    def pack(tensors):
        rows, cols = tensors[0].shape
        return GroupedTensorStorage(
            (len(tensors) * rows, cols), torch.bfloat16, num_tensors=len(tensors),
            data=torch.cat([t.flatten() for t in tensors]),
        )

    for sid in range(2):
        stream_cases = []
        for groups in (4, 3):
            m, n, k = 256 + sid * 128, 256 + sid * 256, 256
            a = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.125
                 for _ in range(groups)]
            b = [torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.125
                 for _ in range(groups)]
            refs = [y.float() @ x.float().t() for x, y in zip(a, b)]
            d = [torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
                 for _ in range(groups)]
            if path == "grouped_tensor":
                a, b, d = pack(a), pack(b), pack(d)
                outputs = list(d.rowwise_data.view(groups, m, n).unbind())
            else:
                outputs = d
            stream_cases.append((a, b, d, outputs, refs))
        cases.append(stream_cases)

    def launch(case):
        a, b, d, _, _ = case
        if path == "grouped_tensor":
            gemm_ext.general_grouped_gemm_for_grouped_tensor(a, b, d)
        else:
            general_grouped_gemm(a, b, d, [None] * len(a), torch.bfloat16, m_splits=[])

    ready = torch.cuda.Event()
    ready.record()  # Inputs and references were produced on the current stream.
    for stream, stream_cases in zip(streams, cases):
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            for case in stream_cases:
                launch(case)  # Warm every JIT specialization before the stress loop.
        stream.synchronize()
    for _ in range(4):
        for index in range(2):
            for stream, stream_cases in zip(streams, cases):
                with torch.cuda.stream(stream):
                    for out in stream_cases[index][3]:
                        out.fill_(float("nan"))
                    launch(stream_cases[index])
    done = [torch.cuda.Event() for _ in streams]
    for stream, event in zip(streams, done):
        event.record(stream)
        torch.cuda.current_stream().wait_event(event)
    # Keep all inputs, outputs, metadata and events alive until both streams finish.
    for stream_cases in cases:
        for _, _, _, outputs, refs in stream_cases:
            for actual, expected in zip(outputs, refs):
                torch.testing.assert_close(actual.float(), expected, rtol=0.005, atol=0.01)


def _direct_grouped_gemm(a, b, d, *, workspace=None, workspace_size=None,
                         bias=None, pre_gelu=None, single_output=False, m_splits=(),
                         accumulate=False, grad=False):
    """Exercise the C++ boundary without Python wrapper validation or allocation."""
    if workspace is None:
        workspace = gemm_ext.get_cublas_workspace(torch.cuda.current_device(), False, True)
    if workspace_size is None:
        workspace_size = workspace[0].numel()
    empty = torch.empty(0, dtype=torch.bfloat16)
    dtype = gemm_ext.TE_DType[torch.bfloat16]
    return gemm_ext.tex.te_general_grouped_gemm(
        a, True, b, False, d, dtype, list(m_splits),
        [empty] * len(a) if bias is None else bias, dtype, single_output,
        [empty] * len(a) if pre_gelu is None else pre_gelu, grad,
        workspace, workspace_size, accumulate, False, 0,
    )


@pytest.mark.parametrize("target,problem", [
    *[(target, problem) for target in ("a", "b", "d", "workspace")
      for problem in ("noncontiguous", "cpu", "mixed_device")],
    *[(target, problem) for target in ("a", "b", "d", "bias", "pre_gelu")
      for problem in ("short", "long")],
    ("a", "shape"), ("d", "shape"), ("d", "capacity"), ("d", "dtype"),
    ("workspace", "empty"), ("workspace", "short"), ("workspace", "rank"),
    ("workspace", "dtype"), ("workspace", "capacity"), ("workspace", "negative_size"),
])
def test_direct_grouped_gemm_rejects_invalid_batch(target, problem):
    if problem == "mixed_device" and torch.cuda.device_count() < 2:
        pytest.skip("Mixed-device validation requires two GPUs")
    # K=0 in group 0 would zero D/bias/pre-GELU before finding group 1 invalid.
    a = [torch.empty(256, 0, device="cuda", dtype=torch.bfloat16),
         torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) * 0.125]
    b = [torch.empty(256, 0, device="cuda", dtype=torch.bfloat16),
         torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) * 0.125]
    d = [torch.full((256, 256), 7.0, device="cuda", dtype=torch.bfloat16) for _ in a]
    empty = torch.empty(0, dtype=torch.bfloat16)  # Unused CPU sentinels remain legal.
    bias = [torch.full((256,), 7.0, device="cuda", dtype=torch.bfloat16), empty]
    pre_gelu = [d[0].clone(), empty]
    workspace = gemm_ext.get_cublas_workspace(torch.cuda.current_device(), False, True)
    args = dict(a=list(a), b=list(b), d=list(d), bias=list(bias), pre_gelu=list(pre_gelu),
                workspace=list(workspace), workspace_size=workspace[0].numel(), grad=True)
    values = args[target]
    tensor = values[-1]
    if problem == "noncontiguous":
        values[-1] = tensor[::2] if target == "workspace" else tensor.t()
    elif problem in ("cpu", "mixed_device"):
        device = "cpu" if problem == "cpu" else torch.device(
            "cuda", (torch.cuda.current_device() + 1) % torch.cuda.device_count()
        )
        values[-1] = tensor.to(device)
    elif problem == "short":
        values.pop()
    elif problem == "long":
        values.append(tensor)
    elif problem == "empty":
        values.clear()
    elif problem == "rank":
        values[-1] = tensor.view(1, -1)
    elif problem == "dtype":
        values[-1] = tensor.float()
    elif problem == "shape":
        values[-1] = tensor.view(128, 512)
    elif problem == "capacity":
        if target == "workspace":
            args["workspace_size"] += 1
        else:
            values[-1] = tensor[:-1]
    elif problem == "negative_size":
        args["workspace_size"] = -1
    watched = d + bias + pre_gelu + args["d"]
    before = [tensor.clone() for tensor in watched]
    with pytest.raises(TypeError if problem == "negative_size" else RuntimeError):
        _direct_grouped_gemm(**args)
    for actual, expected in zip(watched, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A failed descriptor refresh must not poison a later valid call on this thread.
    _direct_grouped_gemm(a, b, d, bias=bias, pre_gelu=pre_gelu, grad=True)
    assert torch.count_nonzero(d[0]) == 0
    assert torch.count_nonzero(bias[0]) == 0
    assert torch.count_nonzero(pre_gelu[0]) == 0
    torch.testing.assert_close(d[1].float(), b[1].float() @ a[1].float().t(),
                               rtol=0.005, atol=0.01)


@pytest.mark.parametrize("single_input", [False, True])
@pytest.mark.parametrize("problem", ["capacity", "empty_d", "extra_d", "split_length",
                                     "negative_split", "split_coverage"])
@pytest.mark.parametrize("splits", [(0, 256), (256, 256)])
def test_direct_grouped_gemm_single_output_bounds(single_input, problem, splits):
    a = [torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) * 0.125
         for _ in range(2)]
    rows = sum(splits)
    inp = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16) * 0.125
    b = inp if single_input else list(inp.split(splits))
    storage = torch.full((rows * 256 + 32,), 7.0, device="cuda", dtype=torch.bfloat16)
    # Spare capacity is allowed; it must not be written by either group.
    d = [storage]
    args = dict(a=a, b=b, d=d, single_output=True, m_splits=splits)
    if problem == "capacity":
        args["d"] = [storage[:rows * 256 - 1]]
    elif problem == "empty_d":
        args["d"] = []
    elif problem == "extra_d":
        args["d"] = d * 2
    else:
        # Split validation belongs to the single-input entry; list-B does not use splits.
        args["b"] = inp
        args["m_splits"] = {"split_length": [rows], "negative_split": [-1, rows + 1],
                            "split_coverage": [splits[0], splits[1] - 1]}[problem]
    with pytest.raises(RuntimeError):
        _direct_grouped_gemm(**args)
    assert torch.all(storage == 7)
    _direct_grouped_gemm(a, b, d, single_output=True, m_splits=splits)
    expected = torch.cat([x.float() @ w.float().t() for x, w in zip(inp.split(splits), a)])
    torch.testing.assert_close(storage[:rows * 256].view(rows, 256).float(), expected,
                               rtol=0.005, atol=0.01)
    assert torch.all(storage[rows * 256:] == 7)


@pytest.mark.parametrize("single_output", [False, True])
@pytest.mark.parametrize("accumulate", [False, True])
def test_direct_grouped_gemm_flatten_and_empty(single_output, accumulate, monkeypatch):
    # RTC is rank-2 only; the binding must retain cuBLAS's outer-dimension flattening.
    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "0")
    monkeypatch.setenv("NVTE_FORCE_RTC_NO_FALLBACK", "0")
    a = [torch.full((2, 128, 256), 0.125, device="cuda", dtype=torch.bfloat16)
         for _ in range(3)]
    b = [torch.full((2, 128, 256), 0.0625, device="cuda", dtype=torch.bfloat16)
         for _ in range(3)]
    a[0] = a[0][..., :0].contiguous()  # K=0 with a nonempty output.
    b[0] = b[0][..., :0].contiguous()
    b[1] = b[1][:0]  # Zero-token split.
    d = [torch.full((rows, 128, 256), 7.0, device="cuda", dtype=torch.bfloat16)
         for rows in (2, 0, 2)]
    if single_output:
        d = [torch.cat([t.flatten() for t in d])]
    _direct_grouped_gemm(a, b, d, single_output=single_output, accumulate=accumulate)
    actual = d[0].view(2, 256, 256) if single_output else torch.stack([d[0].view(256, 256),
                                                                    d[2].view(256, 256)])
    assert torch.all(actual[0] == (7 if accumulate else 0))
    assert torch.all(actual[1] == (9 if accumulate else 2))
    # An empty batch must not index A/B/D or the optional epilogue lists.
    _direct_grouped_gemm([], [], [])


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


@pytest.mark.parametrize("implementation,k,n", [
    ("module", 256, 128), ("module_main_grad", 256, 128), ("op", 256, 128),
    ("module", 255, 128), ("op", 256, 127),
])
def test_grouped_linear_rtc_packed_weight(implementation, k, n, monkeypatch):
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("Packed BF16 RTC requires PPU sm89")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    torch.manual_seed(57)
    splits = [257, 128, 64]
    aligned = k % 8 == 0 and n % 8 == 0
    if not aligned:
        monkeypatch.setenv("NVTE_FORCE_RTC_NO_FALLBACK", "0")
    dtype = torch.bfloat16
    args = dict(bias=False, single_grouped_weight=True, device="cuda")
    fused = implementation == "module_main_grad"
    if implementation.startswith("module"):
        layer = te.GroupedLinear(len(splits), k, n, params_dtype=dtype,
                                 fuse_wgrad_accumulation=fused, **args)
        if fused:
            layer.weight.main_grad = torch.ones(len(splits), n, k, device="cuda", dtype=torch.float32)
    else:
        layer = te.ops.GroupedLinear(len(splits), k, n, dtype=dtype, **args)
    x = torch.randn(sum(splits), k, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn(sum(splits), n, device="cuda", dtype=dtype)
    sizes = torch.tensor(splits, device="cuda", dtype=torch.int64)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = layer(x, sizes)
        out.backward(dy)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")
               and e.device_type == torch.autograd.DeviceType.CUDA]
    if aligned:
        assert len(kernels) == 3 and all("_grouped_tensor" in name for name in kernels)
    else:
        assert not any("_grouped_tensor" in name for name in kernels)
    weights = layer.weight.split_into_quantized_tensors()
    ref_out, ref_dx, ref_dw = _reference(x, weights, dy, splits)
    _assert_group_list_close([out, x.grad], [ref_out, ref_dx], implementation)
    if fused:
        _assert_group_list_close(layer.weight.main_grad.unbind(), [dw + 1 for dw in ref_dw], "main_grad")
    else:
        _assert_group_list_close(layer.weight.grad.view(len(splits), n, k).unbind(), ref_dw, "wgrad")


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


def _packed_storage(shapes, dtype, *, mirror):
    """Build a packed GroupedTensorStorage for per-expert (rows, cols) shapes.

    mirror=True uses make_grouped_tensor, which materializes the host offsets
    mirror and supports ragged first dims; mirror=False uses the bare constructor
    (uniform only, offsets=None) to exercise the offset-free uniform fast path.
    """
    groups = len(shapes)
    rows = [r for r, _ in shapes]
    cols = [c for _, c in shapes]
    total = sum(r * c for r, c in shapes)
    if mirror:
        first = (None if len(set(rows)) == 1
                 else torch.tensor(rows, device="cuda", dtype=torch.int64))
        logical0 = rows[0] * groups if first is None else sum(rows)
        storage = GroupedTensorStorage.make_grouped_tensor(
            groups, first, None, logical0, cols[0], dtype=dtype, device="cuda")
    else:
        assert len(set(rows)) == 1 and len(set(cols)) == 1, "bare constructor is uniform-only"
        storage = GroupedTensorStorage((rows[0] * groups, cols[0]), dtype, num_tensors=groups,
                                       data=torch.empty(total, dtype=dtype, device="cuda"))
    storage.rowwise_data.copy_(torch.randn(total, device="cuda", dtype=dtype) * 0.125)
    return storage


def _expert_views(storage, shapes):
    data = storage.rowwise_data
    views, offset = [], 0
    for rows, cols in shapes:
        views.append(data.narrow(0, offset, rows * cols).view(rows, cols))
        offset += rows * cols
    return views


@pytest.mark.parametrize("api", ["ordinary", "grouped_tensor"])
@pytest.mark.parametrize("layout,m,n,k,expand", [
    ("NN", 2048, 2048, 512, True),
    ("TN", 2048, 2048, 1024, True),
    ("TN", 259, 1536, 256, False),
    ("NN", 256, 1024, 2304, False),
])
def test_rtc_nexpand(api, layout, m, n, k, expand):
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("N_expand requires PPU sm89")
    torch.manual_seed(57)
    groups = 16
    ashape = (n, k) if layout == "TN" else (k, n)
    a, b, d = [_packed_storage([shape] * groups, torch.bfloat16, mirror=False)
               for shape in (ashape, (m, k), (m, n))]
    av, bv, dv = [_expert_views(t, [shape] * groups)
                  for t, shape in ((a, ashape), (b, (m, k)), (d, (m, n)))]
    if api == "ordinary":
        # Descending addresses rule out accidental adjacency in the allocator.
        av = [w.clone() for w in av]
        av.sort(key=lambda w: w.data_ptr(), reverse=True)

        def run():
            general_grouped_gemm(av, bv, dv, [None] * groups, torch.bfloat16,
                                 layout=layout, m_splits=[m] * groups)
    else:
        def run():
            gemm_ext.general_grouped_gemm_for_grouped_tensor(a, b, d, layout=layout)
    run()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.name.startswith("nvte_grouped_gemm3_")
               and e.device_type == torch.autograd.DeviceType.CUDA]
    assert len(kernels) == 1
    assert ("_ne4" in kernels[0]) == expand
    assert ("_grouped_tensor" in kernels[0]) == (api == "grouped_tensor")
    for weight, inp, out in zip(av, bv, dv):
        ref = inp.float() @ (weight.float().t() if layout == "TN" else weight.float())
        torch.testing.assert_close(out.float(), ref, rtol=0.005, atol=0.01)

    if api == "grouped_tensor" and layout == "NN" and expand:
        # Reuse the uniform launch with device scalars and the old D as C.
        previous = [out.clone() for out in dv]
        alpha = torch.tensor(0.5, device="cuda")
        beta = torch.tensor(-0.25, device="cuda")
        gemm_ext.general_grouped_gemm_for_grouped_tensor(a, b, d, layout=layout,
                                                         alpha=alpha, beta=beta)
        for x, y, out, old in zip(av, bv, dv, previous):
            ref = 0.5 * (y.float() @ x.float()) - 0.25 * old.float()
            torch.testing.assert_close(out.float(), ref, rtol=0.005, atol=0.01)


@pytest.mark.parametrize("mirror", [False, True])
def test_grouped_tensor_metadata_and_accumulation(mirror):
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("Native GroupedTensor requires PPU sm89")
    # Cover a second metadata chunk, empty experts, short M and fp32 C/D.
    rows = [0, 8, 257] * 11
    groups, k, n = len(rows), 128, 256
    shapes = [(m, k) for m in rows]
    a = _packed_storage([(n, k)] * groups, torch.bfloat16, mirror=False)
    b = _packed_storage(shapes, torch.bfloat16, mirror=True)
    d = _packed_storage([(m, n) for m in rows], torch.float32, mirror=True)
    if not mirror:
        b.tensor_offsets = d.tensor_offsets = None
        b.offsets = d.offsets = None
    av = _expert_views(a, [(n, k)] * groups)
    bv = _expert_views(b, shapes)
    dv = _expert_views(d, [(m, n) for m in rows])
    alpha = torch.tensor(0.5, device="cuda")
    beta = torch.tensor(-0.25, device="cuda")
    refs = [0.5 * (y.float() @ x.float().t()) - 0.25 * z for x, y, z in zip(av, bv, dv)]
    gemm_ext.general_grouped_gemm_for_grouped_tensor(a, b, d, alpha=alpha, beta=beta)
    for out, ref in zip(dv, refs):
        torch.testing.assert_close(out, ref, rtol=0.005, atol=0.01)
