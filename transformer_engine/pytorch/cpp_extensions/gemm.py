# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""

from typing import Iterable, Optional, Tuple, Union, List
import os
import functools
from numbers import Integral
import torch
import transformer_engine_torch as tex
from ..constants import TE_DType, DType
from ..utils import get_sm_count, _empty_tensor

from ..quantized_tensor import Quantizer
from ..tensor.float8_blockwise_tensor import Float8BlockQuantizer
from ..tensor.storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from ..tensor.storage.grouped_tensor_storage import GroupedTensorStorage
from ..tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage
from ..tensor.utils import is_custom
from ..custom_recipes.gemm import custom_gemm
from ...debug.pytorch.debug_quantization import DebugQuantizer


__all__ = [
    "general_gemm",
    "general_grouped_gemm",
    "general_grouped_gemm_for_grouped_tensor",
]


_NUM_MAX_UB_STREAMS = 3


def get_cublas_workspace_size_bytes() -> None:
    """Return 32 MiB if using hopper, 4 MiB for all other architectures."""
    if torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9:
        # 32 MiB for NVFP4 GEMM, plus additional 1024 B for alignment and misc scales
        return 32 * 1024 * 1024 + 1024
    return 4_194_304


@functools.lru_cache(maxsize=None)
def get_cublas_workspace(device: int, ub: bool, grouped_gemm: bool) -> torch.Tensor:
    """Returns workspace for cublas GEMM."""
    assert not (ub and grouped_gemm), "UB is unsupported for grouped GEMM."

    if ub:
        return torch.empty(
            get_cublas_workspace_size_bytes() * _NUM_MAX_UB_STREAMS,
            dtype=torch.uint8,
            device=device,
        )
    if grouped_gemm:
        _multi_stream_cublas_workspace = []
        for _ in range(tex.get_num_cublas_streams()):
            _multi_stream_cublas_workspace.append(
                torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)
            )
        return _multi_stream_cublas_workspace

    return torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)


def validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    """Validate whether a GEMM scaling factor is consistent with its usage"""
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0


def _is_nvfp4_row_scaled_tensor(tensor: torch.Tensor) -> bool:
    """Whether tensor carries row-scaled NVFP4 global amax metadata."""
    return isinstance(tensor, NVFP4TensorStorage) and tensor._row_scaled_nvfp4


def _nvfp4_row_scaled_gemm_inputs(
    A: NVFP4TensorStorage,
    B: NVFP4TensorStorage,
    *,
    transa: bool,
) -> Tuple[NVFP4TensorStorage, NVFP4TensorStorage, torch.Tensor]:
    """Return GEMM aliases and FP32 output scales for row-scaled NVFP4."""
    A_metadata = A.get_metadata()
    weight_amax = A._amax_rowwise if transa else A._amax_columnwise
    assert weight_amax is not None and weight_amax.numel() == 1
    A_metadata["amax_rowwise" if transa else "amax_columnwise"] = weight_amax.new_ones(1)
    A_metadata["row_scaled_nvfp4"] = False

    B_metadata = B.get_metadata()
    rhs_rowwise_amax = B._amax_rowwise
    assert rhs_rowwise_amax is not None
    B_metadata["amax_rowwise"] = rhs_rowwise_amax.new_ones(1)
    B_metadata["row_scaled_nvfp4"] = False

    assert rhs_rowwise_amax.dtype == torch.float32 and weight_amax.dtype == torch.float32
    return (
        NVFP4TensorStorage(**A_metadata),
        NVFP4TensorStorage(**B_metadata),
        (rhs_rowwise_amax * weight_amax).view(-1, 1),
    )


def general_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """GEMM supporting fp8 inputs."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    alpha = validate_gemm_scale(alpha, True)
    beta = validate_gemm_scale(beta, accumulate)
    workspace = get_cublas_workspace(A.device.index, ub is not None, False)

    if ub_type is not None:
        assert ub is not None, (
            f"{'AG+GEMM' if ub_type == tex.CommOverlapType.AG else 'GEMM+RS'} overlap requires"
            + "a valid `ub` communicator object."
        )

    if ub is not None:
        assert ub_type is not None, "Comm+GEMM overlap requires a valid `comm_type` argument."
        if ub_type == tex.CommOverlapType.RS:
            if not (bulk_overlap and not ub.is_fp8_ubuf()):
                assert extra_output is not None, "GEMM+RS overlap requires extra output tensor."

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    # If A or B are custom tensors -> dispatch to quantizers's qgemm implementation
    if is_custom(A) or is_custom(B):
        return custom_gemm(
            A,
            B,
            workspace,
            out_dtype,
            quantization_params,
            gelu,
            gelu_in,
            accumulate,
            layout,
            out,
            bias,
            use_split_accumulator,
            grad,
        )

    debug_quantizer = None
    if isinstance(quantization_params, DebugQuantizer):
        debug_quantizer = quantization_params
        quantization_params = quantization_params.parent_quantizer
        A = A.get_tensor(not transa)
        B = B.get_tensor(transb)

    # Use bfloat16 as default bias_dtype
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]

    if isinstance(A, Float8BlockwiseQTensorStorage) or isinstance(B, Float8BlockwiseQTensorStorage):
        # FP8 block-scaling requires split accumulator
        use_split_accumulator = True

    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params,
        TE_DType[out_dtype] if out_dtype is not None else None,
        bias,
        bias_dtype,
        gelu,
        gelu_in,
        grad,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )
    kwargs = {
        "comm_overlap": ub,
        "comm_type": ub_type,
        "extra_output": extra_output,
        "bulk_overlap": bulk_overlap,
        "alpha": alpha,
        "beta": beta,
    }

    if not _is_nvfp4_row_scaled_tensor(A) and not _is_nvfp4_row_scaled_tensor(B):
        out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)
    else:
        if _is_nvfp4_row_scaled_tensor(A):
            raise NotImplementedError("Row-scaled NVFP4 GEMM does not support row-scaled A.")
        assert layout[1] == "N", "Row-scaled NVFP4 GEMM currently supports N-layout B only."
        if grad:
            raise RuntimeError(
                "Row-scaled NVFP4 GEMM currently supports fprop only. "
                "Backward NVFP4 gradient quantizers should use scalar global amax."
            )
        assert not gelu, "Row-scaled NVFP4 GEMM currently does not support fused GELU."
        assert not accumulate, "Row-scaled NVFP4 GEMM currently does not support accumulation."
        assert (
            quantization_params is None
        ), "Row-scaled NVFP4 GEMM currently does not support output quantization."
        assert ub is None, "Row-scaled NVFP4 GEMM currently does not support CommOverlap."
        assert (
            extra_output is None
        ), "Row-scaled NVFP4 GEMM currently does not support extra output."
        assert not bulk_overlap, "Row-scaled NVFP4 GEMM currently does not support bulk overlap."
        assert out is None or (
            isinstance(out, torch.Tensor) and not is_custom(out)
        ), "Row-scaled NVFP4 GEMM currently supports only plain torch.Tensor outputs."
        assert isinstance(
            A, NVFP4TensorStorage
        ), "Row-scaled NVFP4 GEMM currently requires NVFP4 A."
        # cuBLAS folds NVFP4 global amax values into GEMM alpha. Keep the row-scaled
        # recipe's global scales out of alpha and apply them in FP32 below.
        gemm_A, gemm_B, rowwise_global_scales = _nvfp4_row_scaled_gemm_inputs(A, B, transa=transa)

        requested_out, requested_out_dtype = out, out_dtype
        fp32_out = (
            torch.empty_like(requested_out, dtype=torch.float32)
            if requested_out is not None
            else None
        )
        gemm_args = list(args)
        gemm_args[0] = gemm_A  # A
        gemm_args[2] = gemm_B  # B
        gemm_args[4] = fp32_out  # out
        gemm_args[5] = None  # quantization_params
        gemm_args[6] = TE_DType[torch.float32]  # out_dtype
        gemm_args[7] = None  # bias
        out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*gemm_args, **kwargs)
        out_2d = out.reshape(-1, out.shape[-1])

        assert rowwise_global_scales.dtype == torch.float32 and out.dtype == torch.float32
        assert rowwise_global_scales.numel() == out_2d.shape[0]

        out_2d.mul_(rowwise_global_scales)
        if bias is not None:
            out_2d.add_(bias.to(dtype=torch.float32))

        if requested_out is not None:
            requested_out.copy_(out.to(dtype=requested_out.dtype))
            out = requested_out
        elif requested_out_dtype is not None and requested_out_dtype != torch.float32:
            out = out.to(dtype=requested_out_dtype)

    if debug_quantizer is not None:
        out = debug_quantizer.process_gemm_output(out)

    return out, bias_grad, gelu_input, extra_output


def _use_contiguous_grouped_gemm(device: torch.device, *, supported: bool = True) -> bool:
    if os.getenv("NVTE_GG_CONTIGUOUS", "0") == "1":
        return True
    return (
        supported
        and getattr(torch.version, "ppu", None) is not None
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device) == (8, 9)
    )


def _validate_contiguous_splits(splits, groups, total_tokens=None):
    if isinstance(splits, torch.Tensor):
        if splits.ndim != 1 or splits.numel() != groups or splits.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        ):
            raise ValueError("NVTE_GG_CONTIGUOUS requires one integer split per expert")
        if splits.is_cuda:
            with torch.cuda.device(splits.device):
                if torch.cuda.is_current_stream_capturing():
                    raise ValueError("NVTE_GG_CONTIGUOUS does not support CUDA graph capture")
        if splits.is_neg() or splits.is_conj():
            raise ValueError("NVTE_GG_CONTIGUOUS splits have lazy negative/conjugate bits")
        span = (splits.numel() - 1) * splits.stride(0) + 1 if splits.numel() else 0
        if (splits.storage_offset() + span) * splits.element_size() > splits.untyped_storage().nbytes():
            raise ValueError("NVTE_GG_CONTIGUOUS splits exceed storage capacity")
        splits = splits.tolist()
    if not isinstance(splits, (list, tuple)) or len(splits) != groups:
        raise ValueError("NVTE_GG_CONTIGUOUS requires one integer split per expert")
    if any(isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in splits):
        raise ValueError("NVTE_GG_CONTIGUOUS splits must be nonnegative integers")
    if total_tokens is not None and sum(splits) != total_tokens:
        raise ValueError("NVTE_GG_CONTIGUOUS sum(m_splits) must match the ragged token length")
    return list(splits)


def _validate_contiguous_grouped_gemm(A, B, out, quantization_params, layout, m_splits=None):
    if layout not in ("TN", "NN", "NT"):
        raise ValueError("NVTE_GG_CONTIGUOUS requires TN, NN or NT layout")
    groups = len(A)
    if not groups or len(B) != groups or len(out) != groups:
        raise ValueError("NVTE_GG_CONTIGUOUS requires nonempty, equal-length A/B/out lists")
    if groups > 2**31 - 1:
        raise ValueError("NVTE_GG_CONTIGUOUS group count exceeds INT_MAX")
    if len(quantization_params) != groups or any(q is not None for q in quantization_params):
        raise ValueError("NVTE_GG_CONTIGUOUS does not support quantization")
    operands = (("A", A, (torch.bfloat16,)),
                ("B", B, (torch.bfloat16,)),
                ("out", out, (torch.bfloat16, torch.float32)))
    for name, tensors, _ in operands:
        if any(type(t) is not torch.Tensor for t in tensors):
            raise ValueError(f"NVTE_GG_CONTIGUOUS {name} requires ordinary Tensor views")
    device = A[0].device
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("NVTE_GG_CONTIGUOUS does not support CUDA graph capture")
    for name, tensors, dtypes in operands:
        storage = tensors[0].untyped_storage()
        offset = tensors[0].storage_offset()
        dtype = tensors[0].dtype
        for i, t in enumerate(tensors):
            if t.is_neg() or t.is_conj():
                raise ValueError(f"NVTE_GG_CONTIGUOUS {name} group {i} has lazy negative/conjugate bits")
            if (t.device != device or t.device.type != "cuda" or t.dtype != dtype
                    or t.dtype not in dtypes or t.ndim != 2 or not t.is_contiguous()
                    or t.untyped_storage()._cdata != storage._cdata
                    or t.storage_offset() != offset):
                raise ValueError(f"NVTE_GG_CONTIGUOUS {name} group {i} must be packed adjacent "
                                 "views of one storage, with uniform device/dtype")
            if any(dim > 2**31 - 1 - 256 for dim in t.shape):
                raise ValueError(f"NVTE_GG_CONTIGUOUS {name} group {i} dimension overflow")
            offset += t.numel()
            if offset * t.element_size() > storage.nbytes():
                raise ValueError(f"NVTE_GG_CONTIGUOUS {name} group {i} exceeds storage capacity")
            if t.numel() and t.data_ptr() % 16:
                raise ValueError(f"NVTE_GG_CONTIGUOUS {name} group {i} requires 16-byte alignment")
    shapes = []
    for i, (a, b, d) in enumerate(zip(A, B, out)):
        m, k = (b.shape[1], b.shape[0]) if layout[1] == "T" else b.shape
        n, ak = a.shape if layout[0] == "T" else (a.shape[1], a.shape[0])
        if ak != k or tuple(d.shape) != (m, n):
            raise ValueError(f"NVTE_GG_CONTIGUOUS {layout} group {i} incompatible matrix dimensions")
        if m and n:
            for axis, extent in zip("MNK", (m, n, k)):
                if extent % 256:
                    raise ValueError(f"NVTE_GG_CONTIGUOUS {layout} group {i} "
                                     f"M/N/K={m}/{n}/{k}: {axis} must be a multiple of 256")
        shapes.append((m, n, k))
    fixed = (0, 1) if layout == "NT" else (1, 2)
    if any(any(s[i] != shapes[0][i] for i in fixed) for s in shapes):
        raise ValueError("NVTE_GG_CONTIGUOUS requires fixed M/N for NT, fixed N/K otherwise")
    d_start = out[0].untyped_storage().data_ptr() + out[0].storage_offset() * out[0].element_size()
    d_end = d_start + sum(d.numel() for d in out) * out[0].element_size()
    for name, tensors in (("A", A), ("B", B)):
        start = tensors[0].untyped_storage().data_ptr() + tensors[0].storage_offset() * tensors[0].element_size()
        end = start + sum(t.numel() for t in tensors) * tensors[0].element_size()
        if d_start < d_end and start < end and d_start < end and start < d_end:
            raise ValueError(f"NVTE_GG_CONTIGUOUS out must not overlap {name} storage")
    split_axis = 2 if layout == "NT" else 0
    expected_splits = [s[split_axis] for s in shapes]
    if m_splits is not None:
        splits = _validate_contiguous_splits(m_splits, groups)
        for i, (actual, expected) in enumerate(zip(splits, expected_splits)):
            if actual != expected:
                raise ValueError(f"NVTE_GG_CONTIGUOUS {layout} group {i} split {actual} "
                                 f"does not match logical {'K' if split_axis == 2 else 'M'}={expected}")
    return expected_splits


def _contiguous_grouped_tensor_views(operand, name):
    if isinstance(operand, (list, tuple)):
        if any(type(t) not in (torch.Tensor, torch.nn.Parameter) for t in operand):
            raise ValueError(f"NVTE_GG_CONTIGUOUS {name} requires ordinary Tensor views")
        return [t.detach() if type(t) is torch.nn.Parameter else t for t in operand]
    if not isinstance(operand, GroupedTensorStorage) or operand.quantizer is not None:
        raise ValueError(f"NVTE_GG_CONTIGUOUS {name} requires unquantized GroupedTensor storage")
    data = operand.rowwise_data
    if (type(data) is not torch.Tensor or data.ndim != 1 or not data.is_cuda
            or not data.is_contiguous() or data.dtype != operand.get_dtype()
            or data.is_neg() or data.is_conj()):
        raise ValueError(f"NVTE_GG_CONTIGUOUS {name} requires contiguous rowwise storage "
                         "with matching logical and actual dtype and no lazy negative/conjugate bits")
    with torch.cuda.device(data.device):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("NVTE_GG_CONTIGUOUS does not support CUDA graph capture")
    if (data.storage_offset() + data.numel()) * data.element_size() > data.untyped_storage().nbytes():
        raise ValueError(f"NVTE_GG_CONTIGUOUS {name} rowwise data exceeds storage capacity")
    groups = operand.num_tensors
    logical = operand.logical_shape
    if (not isinstance(groups, Integral) or not 0 < groups <= 2**31 - 1
            or len(logical) != 2 or min(logical) < 0):
        raise ValueError(f"NVTE_GG_CONTIGUOUS {name} invalid grouped logical shape/count")

    def metadata(field, count):
        value = getattr(operand, field)
        if value is None:
            return None
        if (type(value) is not torch.Tensor or value.dtype != torch.int64
                or value.device != data.device or value.ndim != 1 or value.numel() != count
                or not value.is_contiguous() or value.is_neg() or value.is_conj()):
            raise ValueError(f"NVTE_GG_CONTIGUOUS {name} invalid {field}")
        if (value.storage_offset() + value.numel()) * value.element_size() > value.untyped_storage().nbytes():
            raise ValueError(f"NVTE_GG_CONTIGUOUS {name} {field} exceeds storage capacity")
        return value.tolist()

    first = metadata("first_dims", groups)
    last = metadata("last_dims", groups)
    offsets = metadata("tensor_offsets", groups + 1)
    if first is None:
        if last is None and logical[0] % groups:
            raise ValueError(f"NVTE_GG_CONTIGUOUS {name} logical rows do not divide experts")
        first = [logical[0] // groups if last is None else logical[0]] * groups
    if last is None:
        last = [logical[1]] * groups
    if min(first + last) < 0:
        raise ValueError(f"NVTE_GG_CONTIGUOUS {name} negative expert dimension")
    shapes = list(zip(first, last))
    expected = [0]
    for rows, cols in shapes:
        expected.append(expected[-1] + rows * cols)
    if (expected[-1] != data.numel() or expected[-1] != logical[0] * logical[1]
            or (offsets is not None and offsets != expected)
            or (operand.offsets is not None and list(operand.offsets) != expected)
            or (operand.tensor_shapes is not None
                and [tuple(s) for s in operand.tensor_shapes] != shapes)):
        raise ValueError(f"NVTE_GG_CONTIGUOUS {name} ragged shape/offset/storage mismatch")
    return [data.narrow(0, expected[i], rows * cols).view(rows, cols)
            for i, (rows, cols) in enumerate(shapes)]


def general_grouped_gemm(
    A: List[torch.Tensor],
    B: List[torch.Tensor],
    out: List[torch.Tensor],
    quantization_params: List[Optional[Quantizer]],
    out_dtype: torch.dtype,
    layout: str = "TN",
    m_splits: Optional[List[int]] = None,
    gelu: bool = False,
    grad=False,
    accumulate: bool = False,
    bias: Optional[List[torch.Tensor]] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[DType] = None,
    single_output=False,
) -> Tuple[List[torch.Tensor], ...]:
    """
    TN layout Grouped GEMM with fp8 inputs.
    """
    num_gemms = len(A)
    if os.getenv("NVTE_GG_CONTIGUOUS", "0") == "1":
        if single_output or use_bias or gelu or D_dtype is not None:
            raise ValueError("NVTE_GG_CONTIGUOUS does not support single_output or fusion")
        m_splits = _validate_contiguous_grouped_gemm(
            A, B, out, quantization_params, layout, m_splits
        )

    transa = layout[0] == "T"
    transb = layout[1] == "T"

    empty_tensor = _empty_tensor()
    empty_tensors = [empty_tensor] * num_gemms

    # Use bfloat16 as default bias_dtype
    gelu_input = empty_tensors
    out_dtype = TE_DType[out[0].dtype] if D_dtype is None else D_dtype

    sm_count = get_sm_count()
    workspaces = get_cublas_workspace(A[0].device.index, False, True)

    if grad and use_bias:
        grad_bias = [
            torch.empty(B[i].size(1), dtype=out[0].dtype, device="cuda") for i in range(num_gemms)
        ]
    else:
        grad_bias = empty_tensors
    bias = bias if use_bias else empty_tensors
    if use_bias:
        bias_dtype = TE_DType[grad_bias[0].dtype] if grad else TE_DType[bias[0].dtype]
    else:
        bias_dtype = TE_DType[torch.bfloat16]

    single_input = isinstance(B, torch.Tensor)
    if any(map(_is_nvfp4_row_scaled_tensor, A)):
        raise NotImplementedError("Row-scaled NVFP4 grouped GEMM does not support row-scaled A.")
    if not single_input and any(map(_is_nvfp4_row_scaled_tensor, B)):
        assert D_dtype is None, "Row-scaled NVFP4 grouped GEMM currently does not support D_dtype."
        if single_output:
            assert (
                m_splits is not None
            ), "Row-scaled NVFP4 grouped GEMM requires m_splits with single output."
        out_init = out[0] if single_output else None
        if single_output:
            start_idx = 0
            out_views = []
            for i in range(num_gemms):
                size = m_splits[i]
                out_views.append(out_init[start_idx : start_idx + size])
                start_idx += size
        else:
            out_views = out
        for i in range(num_gemms):
            if out_views[i].numel() == 0:
                continue
            general_gemm(
                A[i],
                B[i],
                quantization_params=quantization_params[i],
                out_dtype=out_views[i].dtype,
                out=out_views[i],
                gelu=gelu,
                accumulate=accumulate,
                layout=layout,
                bias=bias[i] if use_bias else None,
                use_split_accumulator=use_split_accumulator,
                grad=grad,
            )
        if single_output:
            out = out_init
        return out, grad_bias, gelu_input

    if isinstance(quantization_params[0], DebugQuantizer):
        assert not gelu, "GELU not supported in debug mode"
        if single_output:
            out_init = out[0]
            start_idx = 0
            out = [None] * num_gemms
            for i in range(num_gemms):
                size = m_splits[i]
                out[i] = out_init[start_idx : start_idx + size]
                start_idx += size
        for i in range(num_gemms):
            _, bias_or_grad, _, _ = general_gemm(
                A[i],
                B[i],
                quantization_params=quantization_params[i],
                out_dtype=out[0].dtype,
                layout=layout,
                accumulate=accumulate,
                out=out[i],
                bias=bias[i] if use_bias else None,
                use_split_accumulator=use_split_accumulator,
                grad=grad,
            )
            if grad and use_bias:
                grad_bias[i] = bias_or_grad
        if single_output:
            out = out_init

        return out, grad_bias if grad else bias, None

    if gelu:
        gelu_input = [
            torch.empty_like(o, dtype=bias_dtype, memory_format=torch.contiguous_format)
            for o in out
        ]  # this should differ with respect to single output

    gemm_func = (
        tex.te_general_grouped_gemm_single_input if single_input else tex.te_general_grouped_gemm
    )
    bias = gemm_func(
        A,
        transa,
        B,
        transb,
        out,
        out_dtype,
        m_splits,
        grad_bias if grad else bias,
        bias_dtype,
        single_output,
        gelu_input,  # this is pre_gelu_out
        grad,  # grad
        workspaces,
        workspaces[0].shape[0],
        accumulate,
        use_split_accumulator,
        sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
    )

    return out, bias, gelu_input


@functools.lru_cache(maxsize=None)
def get_grouped_gemm_setup_workspace_size(num_tensors: int) -> int:
    """Return workspace size for grouped GEMM pointer setup."""
    return tex.get_grouped_gemm_setup_workspace_size(num_tensors)


@functools.lru_cache(maxsize=None)
def _get_fp32_ones_tensor(num_tensors: int, device: torch.device) -> torch.Tensor:
    """Cached ones tensor."""
    return torch.ones(num_tensors, dtype=torch.float32, device=device)


@functools.lru_cache(maxsize=None)
def _get_fp32_zeros_tensor(num_tensors: int, device: torch.device) -> torch.Tensor:
    """Cached zeros tensor."""
    return torch.zeros(num_tensors, dtype=torch.float32, device=device)


@functools.lru_cache(maxsize=None)
def _get_grouped_gemm_setup_workspace(device: int, num_tensors: int) -> torch.Tensor:
    """Persistent setup workspace (per-group pointer/dim arrays) for grouped-tensor GEMM."""
    return torch.empty(
        get_grouped_gemm_setup_workspace_size(num_tensors),
        dtype=torch.uint8,
        device=device,
    )


@functools.lru_cache(maxsize=None)
def _get_grouped_cublas_workspace(device: int, layout: str) -> torch.Tensor:
    """Persistent cuBLAS workspace for the grouped-tensor GEMM path, one per GEMM layout.

    Grouped cuBlasLt GEMM kernels in cuBLAS versions <= 13.7 leave behind stale descriptors in the
    workspace that cause back-to-back GEMM kernels to crash/deadlock on 2nd CUDA-graph replay. As a
    workaround, we allocate a different workspace for each GEMM layout (TN, NN, NT) to avoid
    contamination between subsequent GEMM calls (when there is no other graph node between GEMM
    kernels).
    """
    assert layout in ("TN", "NN", "NT"), f"unexpected grouped GEMM layout {layout}"
    return torch.empty(get_cublas_workspace_size_bytes(), dtype=torch.uint8, device=device)


def general_grouped_gemm_for_grouped_tensor(
    A,
    B,
    out,
    *,
    layout: str = "TN",
    accumulate: bool = False,
    use_split_accumulator: bool = False,
    bias=None,
    bias_scale: Optional[torch.Tensor] = None,
    grad: bool = False,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    use_contiguous_grouped_gemm: Optional[bool] = None,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """GroupedTensor GEMM, defaulting to the strict contiguous backend on PPU sm89."""
    if use_contiguous_grouped_gemm is None:
        if os.getenv("NVTE_GG_CONTIGUOUS", "0") == "1":
            use_contiguous_grouped_gemm = True
        elif getattr(torch.version, "ppu", None) is not None:
            if isinstance(B, (list, tuple)):
                if not B or not isinstance(B[0], torch.Tensor):
                    raise ValueError("NVTE_GG_CONTIGUOUS requires nonempty Tensor B views")
                data = B[0]
            elif isinstance(B, GroupedTensorStorage):
                data = B.rowwise_data if B.rowwise_data is not None else B.columnwise_data
                if data is None:
                    raise ValueError("NVTE_GG_CONTIGUOUS requires B storage")
            else:
                raise ValueError("NVTE_GG_CONTIGUOUS requires GroupedTensor B storage")
            use_contiguous_grouped_gemm = _use_contiguous_grouped_gemm(data.device)
        else:
            use_contiguous_grouped_gemm = False
    if use_contiguous_grouped_gemm:
        if grad or bias is not None or bias_scale is not None or alpha is not None or beta is not None:
            raise ValueError("NVTE_GG_CONTIGUOUS does not support fusion or custom alpha/beta")
        a = _contiguous_grouped_tensor_views(A, "A")
        b = _contiguous_grouped_tensor_views(B, "B")
        d = _contiguous_grouped_tensor_views(out, "out")
        splits = _validate_contiguous_grouped_gemm(a, b, d, [None] * len(a), layout)
        with torch.cuda.device(a[0].device):
            workspace = _get_grouped_cublas_workspace(a[0].device.index, layout)
            sm_count = get_sm_count()
        tex.te_contiguous_grouped_gemm(
            a, layout[0] == "T", b, layout[1] == "T", d, splits,
            workspace, accumulate,
            sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count))),
        )
        return out
    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    if grad:
        raise NotImplementedError("grad is not supported for grouped_tensor GEMM yet.")
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    is_discrete_out = isinstance(out, list)
    is_discrete_in = isinstance(A, list)
    if is_discrete_in and is_discrete_out:
        raise ValueError("Both A and out are discrete. This is not supported yet.")

    if isinstance(A, GroupedTensorStorage) and A.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")
    if isinstance(B, GroupedTensorStorage) and B.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")
    if isinstance(out, GroupedTensorStorage) and out.row_scaled_nvfp4:
        raise NotImplementedError("Row-scaled NVFP4 GroupedTensor GEMM is not supported yet.")

    def _is_fp8_blockwise(operand) -> bool:
        if isinstance(operand, (list, tuple)):
            return any(isinstance(t, Float8BlockwiseQTensorStorage) for t in operand)
        if isinstance(operand, GroupedTensorStorage):
            return isinstance(operand.quantizer, Float8BlockQuantizer)
        return False

    if _is_fp8_blockwise(A) or _is_fp8_blockwise(B):
        # The fused grouped FP8 block-scaling GEMM only supports split accumulation,
        # so force it on and intentionally override any caller-supplied value. This
        # matches the Float8BlockScaling recipe, which fixes use_split_accumulator=True
        # for all of fprop/dgrad/wgrad, so no user-configurable setting is discarded.
        use_split_accumulator = True

    if is_discrete_out:
        # wgrad case.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_discrete_out
    elif is_discrete_in:
        # Use-case: forward pass with list of weights.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_discrete_in
    else:
        # Use-case: Single Grouped Parameter for Weight/ Weight Grads.
        grouped_gemm_impl = tex.te_general_grouped_gemm_for_grouped_tensor

    if is_discrete_out and bias is not None:
        raise ValueError(
            "Bias is not supported when out is a list (discrete_out mode) yet. "
            "Apply bias manually after the GEMM."
        )

    if bias_scale is not None and bias is None:
        raise ValueError("bias_scale requires bias to be provided.")

    num_tensors = B.num_tensors
    rowwise = B.rowwise_data
    device = rowwise.device if rowwise is not None else B.columnwise_data.device

    # Hopper (SM90) uses a single shared alpha/beta scalar;
    # Blackwell+ (SM100) supports per-group alpha/beta arrays.
    per_group = torch.cuda.get_device_capability() >= (10, 0)
    num_alphabeta = num_tensors if per_group else 1

    if alpha is None:
        alpha = _get_fp32_ones_tensor(num_alphabeta, device)
    if beta is None:
        if accumulate:
            beta = _get_fp32_ones_tensor(num_alphabeta, device)
        else:
            beta = _get_fp32_zeros_tensor(num_alphabeta, device)

    if not alpha.is_cuda or not beta.is_cuda:
        raise ValueError("alpha and beta must be CUDA tensors.")

    workspace_setup = _get_grouped_gemm_setup_workspace(device.index, num_tensors)
    # Each grouped-GEMM layout gets its own persistent cuBLAS workspace: two grouped
    # GEMMs sharing one workspace can deadlock under CUDA-graph replay (see
    # _get_grouped_cublas_workspace). wgrad (NT) is the case seen in TE; fprop (TN) and
    # dgrad (NN) have also been reported to conflict, so all three layouts are isolated.
    workspace_cublas = _get_grouped_cublas_workspace(device.index, layout)

    sm_count = get_sm_count()
    sm_count = sm_count - int(os.getenv("NVTE_EXT_MARGIN_SM", str(sm_count)))

    return grouped_gemm_impl(
        A,
        transa,
        B,
        transb,
        out,
        bias,
        bias_scale,
        alpha,
        beta,
        workspace_setup,
        workspace_cublas,
        use_split_accumulator,
        sm_count,
    )
