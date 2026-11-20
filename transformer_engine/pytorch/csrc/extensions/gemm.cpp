/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <pybind11/pybind11.h>

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>

#include "../extensions.h"
#include "common/common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "pybind.h"
#include "transformer_engine/transformer_engine.h"
#include "util.h"

namespace {

void* get_data_ptr(transformer_engine::pytorch::MaybeTensor tensor) {
  if (tensor.has_value()) return tensor->data_ptr();
  return nullptr;
}

size_t get_size(transformer_engine::pytorch::MaybeTensor tensor, int dim) {
  if (tensor.has_value()) return static_cast<size_t>(tensor->size(dim));
  return 0;
}

// Dense operands only need their rowwise descriptor. After a quantized operand,
// replace the wrapper as well so its scale and columnwise state cannot leak.
struct GroupedGemmHostScratch {
  struct QuantizedInputs {
    bool a = false, b = false;
  };
  std::vector<transformer_engine::TensorWrapper> a, b, d;
  std::vector<NVTETensor> a_v, b_v, d_v;
  std::vector<QuantizedInputs> quantized;

  void resize(size_t count) {
    a.resize(count);
    b.resize(count);
    d.resize(count);
    a_v.resize(count);
    b_v.resize(count);
    d_v.resize(count);
    quantized.resize(count);
  }
};

// Off switch for the descriptor cache, used to A/B the host cost of building
// every NVTETensor from scratch.
inline bool reuse_grouped_descriptors() {
  static const bool enabled = [] {
    const char* value = std::getenv("NVTE_GG_REUSE_DESCRIPTORS");
    return value == nullptr || std::string(value) != "0";
  }();
  return enabled;
}

// True for ordinary torch tensors, including nn.Parameter. Quantized tensor
// subclasses are excluded: their descriptors carry per-call scale pointers.
inline bool is_plain_torch_tensor(py::handle tensor) {
  PyObject* obj = tensor.ptr();
  if (!THPVariable_Check(obj)) {
    return false;
  }
  using namespace transformer_engine::pytorch::detail;
  return !(IsFloat8Tensor(obj) || IsMXFP8Tensor(obj) || IsFloat8BlockwiseQTensor(obj) ||
           IsNVFP4Tensor(obj));
}

void check_discrete_grouped_tensor(py::handle value, const char* name) {
  if (!is_plain_torch_tensor(value)) return;
  const auto tensor = py::cast<at::Tensor>(value);
  NVTE_CHECK(!tensor.is_quantized() && tensor.is_cuda() && tensor.is_contiguous() && tensor.dim() == 2 &&
                 tensor.get_device() == at::cuda::current_device(),
             "Grouped GEMM ", name, " must be a contiguous CUDA matrix on the current device.");
}

void check_grouped_bias_scale(const at::Tensor& scale) {
  NVTE_CHECK(scale.is_cuda() && scale.is_contiguous() && scale.dim() == 1 &&
                 scale.scalar_type() == at::kFloat &&
                 scale.get_device() == at::cuda::current_device(),
             "Grouped bias scale must be a contiguous CUDA float32 vector on the current device.");
}

}  // namespace

namespace transformer_engine::pytorch {

namespace detail {

bool is_low_precision(const DType type) {
  return type == DType::kFloat8E4M3 || type == DType::kFloat8E5M2;
}

Shape getGemmOutputShape(const NVTEShape& A_shape, const bool transa, const NVTEShape& B_shape,
                         const bool transb, size_t tp_size = 1, size_t tp_dim = 0) {
  // Flatten outer dims to get 2D matrices
  NVTE_CHECK(A_shape.ndim >= 1, "Tensor A needs to have at least 1 dimension");
  NVTE_CHECK(B_shape.ndim >= 1, "Tensor B needs to have at least 1 dimension");
  const auto [A0, A1] = get_2d_dims(A_shape);
  const auto [B0, B1] = get_2d_dims(B_shape);

  // Check matrix dims
  NVTE_CHECK((transa ? A1 : A0) == (transb ? B0 : B1), "Invalid matrix dimensions for GEMM (A=(",
             A0, ",", A1, "), transa=", transa, ", B=(", B0, ",", B1, "), transb=", transb, ")");

  // Preserve B's leading dimensions unless transposition or comm overlap flattens them.
  Shape ret = (transb || tp_size != 1) ? Shape{transb ? B1 : B0, 0} : Shape(B_shape);
  ret.back() = transa ? A0 : A1;

  // Correct output dims for comm+GEMM overlap if needed
  if (tp_size > 1) {
    if (tp_dim == 0) {
      // Outer dim is sharded, comm+GEMM overlap would need to do all-gather
      ret.front() *= tp_size;
    } else {
      // Inner dim is sharded, comm+GEMM overlap would need to do reduce-scatter
      ret.front() /= tp_size;
    }
  }
  return ret;
}

struct GroupedGemmConfig {
  TensorWrapper te_alpha;
  TensorWrapper te_beta;
  TensorWrapper te_workspace_setup;
  TensorWrapper te_workspace_cublas;
  std::optional<transformer_engine::GroupedMatmulConfigWrapper> matmul_config;
};

GroupedGemmConfig prepare_grouped_gemm_config(at::Tensor alpha, at::Tensor beta,
                                              at::Tensor workspace_setup,
                                              at::Tensor workspace_cublas, size_t num_tensors,
                                              int math_sm_count, bool use_split_accumulator) {
  auto check_buffer = [](const at::Tensor& tensor, at::ScalarType dtype, const char* name) {
    NVTE_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.scalar_type() == dtype &&
                   tensor.get_device() == at::cuda::current_device(),
               "Grouped GEMM ", name,
               " must be contiguous CUDA storage of the required dtype on the current device.");
  };
  check_buffer(alpha, at::kFloat, "alpha");
  check_buffer(beta, at::kFloat, "beta");
  check_buffer(workspace_setup, at::kByte, "workspace_setup");
  check_buffer(workspace_cublas, at::kByte, "workspace_cublas");
  const bool per_group = (alpha.numel() == static_cast<int64_t>(num_tensors));
  const bool scalar = (alpha.numel() == 1);
  NVTE_CHECK(per_group || scalar, "Grouped GEMM expects alpha to have 1 or num_tensors (",
             num_tensors, ") elements, got ", alpha.numel());
  NVTE_CHECK(beta.numel() == alpha.numel(),
             "Grouped GEMM expects beta to have the same number of elements as alpha (",
             alpha.numel(), "), got ", beta.numel());

  GroupedGemmConfig grouped_gemm_config{
      makeTransformerEngineTensor(alpha),
      makeTransformerEngineTensor(beta),
      makeTransformerEngineTensor(workspace_setup.data_ptr(),
                                  std::vector<size_t>{static_cast<size_t>(workspace_setup.numel())},
                                  DType::kByte),
      makeTransformerEngineTensor(
          workspace_cublas.data_ptr(),
          std::vector<size_t>{static_cast<size_t>(workspace_cublas.numel())}, DType::kByte),
      std::nullopt,
  };

  if (math_sm_count > 0 || use_split_accumulator) {
    grouped_gemm_config.matmul_config.emplace();
    if (math_sm_count > 0) {
      grouped_gemm_config.matmul_config->set_sm_count(math_sm_count);
    }
    grouped_gemm_config.matmul_config->set_use_split_accumulator(use_split_accumulator);
  }

  return grouped_gemm_config;
}

}  // namespace detail

std::pair<TensorWrapper, py::object> createOutputTensor(const std::vector<size_t>& shape,
                                                        DType dtype, py::handle quantizer) {
  std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
  return my_quantizer->create_tensor(shape, dtype);
}

std::vector<py::object> gemm(py::handle A, bool transa, py::handle B, bool transb, py::object D,
                             py::handle quantizer, std::optional<DType> out_dtype, MaybeTensor bias,
                             DType bias_type, bool gelu, MaybeTensor gelu_in, bool grad,
                             at::Tensor workspace, size_t workspaceSize, bool accumulate,
                             bool use_split_accumulator, CommOverlapCore* comm_overlap,
                             std::optional<CommOverlapType> comm_type, MaybeTensor extra_output,
                             bool bulk_overlap, float alpha, std::optional<float> beta) {
  using namespace transformer_engine::pytorch::detail;

  // Ensure that cublasLt handle is created on the correct device,
  // overriding torch.cuda.set_device calls from user side.
  // Assumes all tensors passed are on the same device.
  at::cuda::CUDAGuard device_guard(workspace.device());

  // Input tensors
  NVTE_CHECK(!A.is_none(), "Tensor A has not been provided");
  NVTE_CHECK(!B.is_none(), "Tensor B has not been provided");
  auto none = py::none();
  TensorWrapper A_tensor = makeTransformerEngineTensor(A, none);
  TensorWrapper B_tensor = makeTransformerEngineTensor(B, none);

  const bool low_precision =
      detail::is_low_precision(A_tensor.dtype()) || detail::is_low_precision(B_tensor.dtype());
  const bool fp8_block_scaling = A_tensor.scaling_mode() == NVTE_BLOCK_SCALING_1D ||
                                 A_tensor.scaling_mode() == NVTE_BLOCK_SCALING_2D ||
                                 B_tensor.scaling_mode() == NVTE_BLOCK_SCALING_1D ||
                                 B_tensor.scaling_mode() == NVTE_BLOCK_SCALING_2D;
  // Get TP info for comm+GEMM overlap
  size_t tp_size = 1;
  size_t tp_dim = 0;
  if (comm_overlap && !bulk_overlap && comm_overlap->with_cublasmp()) {
    tp_size = comm_overlap->get_tp_size();
    tp_dim = (comm_type.value() == CommOverlapType::AG) ? 0 : 1;
  }
  // Check tensor dimensions
  const auto& A_shape = A_tensor.shape();
  const auto& B_shape = B_tensor.shape();
  const std::vector<size_t> D_shape(
      detail::getGemmOutputShape(A_shape, transa, B_shape, transb, tp_size, tp_dim));

  // Check scaling factors
  if (accumulate) {
    if (!beta) {
      beta = 1.0f;
    }
  } else {
    if (!beta) {
      beta = 0.0f;
    }
    NVTE_CHECK(beta == 0.0, "Trying to use non-zero beta while not accumulating ",
               "into D tensor. Beta has nothing to be applied to.");
  }

  DType output_dtype = out_dtype ? *out_dtype : A_tensor.dtype();
  // Output tensor
  TensorWrapper D_tensor;
  if (D.is_none()) {
    std::tie(D_tensor, D) = createOutputTensor(D_shape, output_dtype, quantizer);
  } else {
    D_tensor = makeTransformerEngineTensor(D, quantizer);
    NVTE_CHECK(Shape(D_tensor.shape()) == D_shape, "GEMM output has invalid dims (expected ",
               std::to_string(D_shape), ", got ", std::to_string(D_tensor.shape()), ")");
    if (out_dtype) {
      NVTE_CHECK(*out_dtype == D_tensor.dtype(), "GEMM output has invalid dtype (expected ",
                 static_cast<int>(*out_dtype), ", found ", static_cast<int>(D_tensor.dtype()), ")");
    }
  }

  // maintain unquantized tensor in case we need unfused quantization support.
  TensorWrapper unquantized_D_tensor;
  py::object unquantized_out;
  // Unfused quantization is needed in the following cases
  // 1. Inputs: BF16, Output: FP8 (GEMM output has to be BF16, so FP8 quantization needed after that)
  // 2. Inputs: FP8, Output: FP8 (For any quantization apart from delayed scaling,
  // GEMM Output needs to be in BF16, to allow for unfused quantization)
  bool unfused_quantization_needed = !quantizer.is_none();
  if (low_precision) {
    // At the moment, only use-case for fused GEMM:
    // Delayed scaling quantizer with per-tensor scaling inputs
    bool is_per_tensor_scaling_input = IsFloat8Tensor(A.ptr()) || IsFloat8Tensor(B.ptr());
    if (IsFloat8Quantizers(quantizer.ptr()) && is_per_tensor_scaling_input)
      unfused_quantization_needed = false;
  }

  if (unfused_quantization_needed) {
    NoneQuantizer q{none};
    std::tie(unquantized_D_tensor, unquantized_out) = q.create_tensor(D_shape, output_dtype);
  }
  TensorWrapper& out_tensor = unfused_quantization_needed ? unquantized_D_tensor : D_tensor;

  // Bias tensor
  TensorWrapper bias_tensor;
  MaybeTensor bias_grad = std::nullopt;
  if (bias.has_value()) {
    if (grad) {
      auto opts =
          torch::TensorOptions().dtype(GetATenDType(out_tensor.dtype())).device(torch::kCUDA);
      bias_grad = at::empty({static_cast<int64_t>(B_shape.data[B_shape.ndim - 1])}, opts);
      bias_tensor = makeTransformerEngineTensor(*bias_grad);
    } else {
      if (!bias->is_contiguous()) {
        bias = bias->contiguous();
      }
      bias_tensor = makeTransformerEngineTensor(*bias);
    }
  }

  // Activation input tensor
  MaybeTensor pre_gelu_out = std::nullopt;
  DType gelu_type = low_precision ? bias_type : out_tensor.dtype();
  if (gelu) {
    if (!grad) {
      auto dtype = GetATenDType(gelu_type);
      auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
      std::vector<int64_t> torch_shape;
      for (auto v : D_shape) {
        torch_shape.push_back(v);
      }
      pre_gelu_out = at::empty(torch_shape, opts);
    } else {
      if (gelu_in.has_value()) {
        pre_gelu_out = *gelu_in;
      }
    }
  }
  const auto gelu_shape = gelu ? D_shape : std::vector<size_t>{0};

  auto te_pre_gelu_out =
      makeTransformerEngineTensor(get_data_ptr(pre_gelu_out), gelu_shape, gelu_type);

  // Workspace
  auto te_workspace = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  std::vector<size_t>{workspaceSize}, DType::kByte);

  // Set an external SM Margin to all the GEMMs.
  // This comes in handy when DP is overlapped with GEMMs
  const int device_id = at::cuda::current_device();
  const int sm_count = transformer_engine::cuda::sm_count(device_id);
  int num_math_sms = sm_count - transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", sm_count);

  // Construct GEMM config
  transformer_engine::MatmulConfigWrapper config;
  if (grad) {
    config.set_dbias_tensor(bias_tensor.data());
    config.set_with_dgelu_epilogue(gelu);
  } else {
    config.set_bias_tensor(bias_tensor.data());
    config.set_with_gelu_epilogue(gelu);
  }
  config.set_epilogue_aux_tensor(te_pre_gelu_out.data());
  config.set_use_split_accumulator(use_split_accumulator);
  config.set_sm_count(num_math_sms);

  // Keep the swizzled scaling factor tensors alive during the GEMM.
  std::vector<std::optional<at::Tensor>> swizzled_scale_inverses_list;
  auto main_stream = at::cuda::getCurrentCUDAStream();
  if (A_tensor.numel() != 0 && B_tensor.numel() != 0) {
    // Optionally swizzle the scaling factors
    auto [A_row_scales, A_col_scales] = swizzle_scales_for_gemm(A_tensor, transa, !transa);
    auto [B_row_scales, B_col_scales] = swizzle_scales_for_gemm(B_tensor, !transb, transb);
    swizzled_scale_inverses_list.emplace_back(std::move(A_row_scales));
    swizzled_scale_inverses_list.emplace_back(std::move(A_col_scales));
    swizzled_scale_inverses_list.emplace_back(std::move(B_row_scales));
    swizzled_scale_inverses_list.emplace_back(std::move(B_col_scales));

    // Emulate the FP8 block scaling recipe with MXFP8 on Blackwell and newer
    // as it is not natively supported by cublasLt
    if (fp8_block_scaling && transformer_engine::cuda::sm_arch() >= 100) {
      // Convert tensors to mxfp8 and swizzle their scaling factors
      swizzled_scale_inverses_list.emplace_back(
          std::move(convert_block_scaling_to_mxfp8_tensor(A_tensor, transa)));
      swizzled_scale_inverses_list.emplace_back(
          std::move(convert_block_scaling_to_mxfp8_tensor(B_tensor, !transb)));
      // Use TN GEMM to avoid having to transpose data.
      transa = true;
      transb = false;
    }

    if (comm_overlap) {
      // Prepare extra output tensor
      TensorWrapper extra_output_tensor;
      if (extra_output.has_value()) {
        extra_output_tensor = makeTransformerEngineTensor(*extra_output);
      } else {
        extra_output_tensor =
            makeTransformerEngineTensor(nullptr, std::vector<size_t>{0}, DType::kByte);
      }

      // Direct GEMM call to the correct overlap
      if (bulk_overlap) {
        NVTE_SCOPED_GIL_RELEASE({
          comm_overlap->bulk_overlap(A_tensor, transa, B_tensor, transb, out_tensor, bias_tensor,
                                     te_pre_gelu_out, te_workspace, grad, accumulate,
                                     use_split_accumulator, comm_type.value(), extra_output_tensor,
                                     main_stream);
        });
      } else if (comm_type.value() == CommOverlapType::AG) {
        if (comm_overlap->is_atomic_gemm()) {
          NVTE_SCOPED_GIL_RELEASE({
            comm_overlap->atomic_gemm_overlap_ag(A_tensor, transa, B_tensor, transb, out_tensor,
                                                 bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                                 accumulate, use_split_accumulator,
                                                 extra_output_tensor, main_stream);
          });
        } else {
          NVTE_SCOPED_GIL_RELEASE({
            comm_overlap->split_overlap_ag(A_tensor, transa, B_tensor, transb, out_tensor,
                                           bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                           accumulate, use_split_accumulator, extra_output_tensor,
                                           main_stream);
          });
        }
      } else {
        if (comm_overlap->is_atomic_gemm()) {
          NVTE_SCOPED_GIL_RELEASE({
            comm_overlap->atomic_gemm_overlap_rs(A_tensor, transa, B_tensor, transb, out_tensor,
                                                 bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                                 accumulate, use_split_accumulator,
                                                 extra_output_tensor, main_stream);
          });
        } else {
          NVTE_SCOPED_GIL_RELEASE({
            comm_overlap->split_overlap_rs(A_tensor, transa, B_tensor, transb, out_tensor,
                                           bias_tensor, te_pre_gelu_out, te_workspace, grad,
                                           accumulate, use_split_accumulator, extra_output_tensor,
                                           main_stream);
          });
        }
      }
    } else {
      // Launch GEMM
      NVTE_SCOPED_GIL_RELEASE({
        nvte_cublas_gemm_v2(transa, transb, &alpha, A_tensor.data(), B_tensor.data(), &beta.value(),
                            out_tensor.data(), out_tensor.data(), te_workspace.data(), config,
                            main_stream);
      });
    }
  } else {
    if (out_tensor.numel() != 0 && !accumulate) {
      out_tensor.zero_(main_stream);
    }
    if (bias.has_value()) {
      if (bias->numel() != 0 && grad) {
        bias_grad->zero_();
      }
    }
  }
  if (unfused_quantization_needed) {
    // Quantize the output
    std::unique_ptr<Quantizer> my_quantizer = convert_quantizer(quantizer);
    my_quantizer->quantize(unquantized_D_tensor, D_tensor);
  }
  // Pack outputs
  std::vector<py::object> out;
  out.emplace_back(std::move(D));
  out.emplace_back(py::cast(bias_grad));
  if (gelu && !grad) {
    out.emplace_back(py::cast(*pre_gelu_out));
  } else {
    out.emplace_back(py::none());
  }
  if (extra_output.has_value()) {
    out.emplace_back(py::cast(extra_output));
  } else {
    out.emplace_back(py::none());
  }
  return out;
}

void te_atomic_gemm(at::Tensor A, at::Tensor A_scale_inverse, DType A_type,
                    std::vector<int64_t> A_scaling_mode, bool transa, at::Tensor B,
                    at::Tensor B_scale_inverse, DType B_type, std::vector<int64_t> B_scaling_mode,
                    bool transb, at::Tensor D, at::Tensor D_scale, DType D_type, at::Tensor D_amax,
                    at::Tensor bias, DType bias_type, at::Tensor pre_gelu_out, bool grad,
                    at::Tensor workspace, size_t workspaceSize, bool accumulate,
                    bool use_split_accumulator, int math_sm_count, int m_split, int n_split,
                    bool gemm_producer, at::Tensor counter) {
  // Ensure that cublasLt handle is created on the correct device,
  // overriding torch.cuda.set_device calls from user side.
  // Assumes all tensors passed are on the same device.
  at::cuda::CUDAGuard device_guard(workspace.device());

  // TODO: Handle scaling modes
  NVTEScalingMode nvte_scaling_modeA = NVTE_DELAYED_TENSOR_SCALING;
  NVTEScalingMode nvte_scaling_modeB = NVTE_DELAYED_TENSOR_SCALING;

  auto te_A = makeTransformerEngineTensor(
      A.data_ptr(), {static_cast<size_t>(A.size(0)), static_cast<size_t>(A.size(1))}, A_type,
      nullptr, nullptr, A_scale_inverse.data_ptr(), getTensorShape(A_scale_inverse),
      nvte_scaling_modeA);
  auto te_B = makeTransformerEngineTensor(
      B.data_ptr(), {static_cast<size_t>(B.size(0)), static_cast<size_t>(B.size(1))}, B_type,
      nullptr, nullptr, B_scale_inverse.data_ptr(), getTensorShape(B_scale_inverse),
      nvte_scaling_modeB);
  // TODO: D_scale_inv cannot be nullptr when D_type is FP8.
  auto te_D = makeTransformerEngineTensor(
      D.data_ptr(),
      std::vector<size_t>{static_cast<size_t>(D.size(0)), static_cast<size_t>(D.size(1))}, D_type,
      D_amax.data_ptr(), D_scale.data_ptr(), nullptr);
  auto te_bias = makeTransformerEngineTensor(
      bias.data_ptr(), std::vector<size_t>{static_cast<size_t>(bias.size(0))}, bias_type);
  auto te_counter = makeTransformerEngineTensor(
      counter.data_ptr(), std::vector<size_t>{static_cast<size_t>(counter.size(0))}, DType::kInt32);

  const auto gelu_shape = pre_gelu_out.data_ptr() == nullptr
                              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0))}
                              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out.size(0)),
                                                    static_cast<size_t>(pre_gelu_out.size(1))};
  auto te_pre_gelu_out = makeTransformerEngineTensor(
      pre_gelu_out.data_ptr(), gelu_shape, GetTransformerEngineDType(pre_gelu_out.scalar_type()));
  auto te_workspace = makeTransformerEngineTensor(workspace.data_ptr(),
                                                  std::vector<size_t>{workspaceSize}, DType::kByte);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_cublas_atomic_gemm(te_A.data(), te_B.data(), te_D.data(), te_bias.data(),
                            te_pre_gelu_out.data(), transa, transb, grad, te_workspace.data(),
                            accumulate, use_split_accumulator, math_sm_count, m_split, n_split,
                            gemm_producer, te_counter.data(), at::cuda::getCurrentCUDAStream());
  });
}

std::optional<std::vector<at::Tensor>> te_general_grouped_gemm(
    std::vector<py::handle> A, bool transa, py::handle B, bool transb,
    std::optional<std::vector<at::Tensor>> D, DType D_type, std::vector<int64_t> m_splits,
    std::vector<at::Tensor> bias, DType bias_type, bool single_output,
    std::vector<at::Tensor> pre_gelu_out, bool grad, std::vector<at::Tensor> workspace,
    size_t workspaceSize, bool accumulate, bool use_split_accumulator, int math_sm_count) {
  if (single_output && D == std::nullopt) {
    NVTE_ERROR("not implemented, D should be allocated for single output case.");
  }

  NVTE_CHECK(!workspace.empty() &&
                 workspace.size() >= static_cast<size_t>(nvte_get_num_compute_streams()),
             "Grouped GEMM requires a workspace for every compute stream.");
  const auto device = workspace[0].device();
  for (const auto& ws : workspace) {
    NVTE_CHECK(ws.is_cuda() && ws.device() == device && ws.dim() == 1 &&
                   ws.is_contiguous() && ws.scalar_type() == at::kByte,
               "Grouped GEMM workspace must be contiguous CUDA uint8 vectors on one device.");
    NVTE_CHECK(workspaceSize <= static_cast<size_t>(ws.numel()),
               "Grouped GEMM workspaceSize exceeds workspace capacity.");
  }
  NVTE_CHECK(bias.size() == A.size() && pre_gelu_out.size() == A.size(),
             "Grouped GEMM requires one bias and pre_gelu_out entry per group.");
  NVTE_CHECK(!D.has_value() || D->size() == (single_output ? 1 : A.size()),
             "Grouped GEMM has an invalid number of D tensors.");

  // Pin cuBLAS to the workspace device, after validating the entire workspace list.
  at::cuda::CUDAGuard device_guard(device);
  const auto none = py::none();
  const auto dtype = GetATenDType(D_type);
  const DType te_D_dtype = GetTransformerEngineDType(dtype);
  auto check_tensor = [&](const at::Tensor& tensor, const char* name) {
    NVTE_CHECK(tensor.is_cuda() && tensor.device() == device && tensor.is_contiguous(),
               "Grouped GEMM ", name, " must be contiguous CUDA storage on the workspace device.");
  };
  if (D.has_value()) {
    for (const auto& out : *D) {
      check_tensor(out, "D");
      NVTE_CHECK(out.scalar_type() == dtype, "Grouped GEMM D has an invalid output dtype.");
    }
  }
  size_t output_offset = 0;
  static thread_local GroupedGemmHostScratch scratch;
  scratch.resize(A.size());
  const bool reuse = reuse_grouped_descriptors();
  auto set_dense = [&](TensorWrapper& desc, void* ptr, DType type, const NVTEShape& shape,
                       bool was_quantized = false) {
    if (reuse && !was_quantized) {
      desc.set_rowwise_data(ptr, type, shape);
    } else {
      desc = TensorWrapper(ptr, shape, type);
    }
  };
  auto set_input = [&](TensorWrapper& desc, bool& was_quantized, py::handle obj, const char* name) {
    if (is_plain_torch_tensor(obj)) {
      auto tensor = obj.cast<at::Tensor>();
      check_tensor(tensor, name);
      set_dense(desc, tensor.data_ptr(), GetTransformerEngineDType(tensor.scalar_type()),
                Shape(tensor.sizes()), was_quantized);
      was_quantized = false;
    } else {
      desc = makeTransformerEngineTensor(obj, none);
      was_quantized = true;
    }
  };
  std::vector<TensorWrapper> te_bias_wrappers, te_pre_gelu_out_wrappers;
  std::vector<NVTETensor> te_bias_vector, te_pre_gelu_out_vector;
  te_bias_vector.reserve(A.size());
  te_pre_gelu_out_vector.reserve(A.size());
  // Python passes one shared empty tensor for every unused bias / pre-GELU slot. A slot that
  // holds the same tensor as the previous GEMM's reuses its NVTETensor instead of a new one.
  const at::Tensor* prev_bias = nullptr;
  const at::Tensor* prev_pre_gelu_out = nullptr;
  const bool single_input = THPVariable_Check(B.ptr());
  at::Tensor input;
  std::vector<py::handle> input_parts;
  int64_t input_offset = 0;
  DType input_dtype = DType::kBFloat16;
  if (single_input) {
    input = B.cast<at::Tensor>();
    check_tensor(input, "B");
    NVTE_CHECK(single_output && !transb && input.dim() == 2 &&
                   (input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf),
               "Single input requires a contiguous BF16/FP16 CUDA matrix and single output.");
    NVTE_CHECK(m_splits.size() == A.size(), "Invalid number of input splits.");
    int64_t rows = 0;
    for (int64_t n : m_splits) {
      NVTE_CHECK(n >= 0 && n <= input.size(0) - rows, "Invalid input split size.");
      rows += n;
    }
    NVTE_CHECK(rows == input.size(0), "Input splits must cover all rows.");
    input_dtype = GetTransformerEngineDType(input.scalar_type());
  } else {
    input_parts = B.cast<std::vector<py::handle>>();
    NVTE_CHECK(input_parts.size() == A.size(), "Grouped GEMM requires matching A and B lengths.");
  }
  std::vector<at::Tensor> D_vectors;
  // Defer empty-group writes until every group has passed validation.
  std::vector<at::Tensor> tensors_to_zero;
  size_t used = 0;
  for (size_t i = 0; i < A.size(); i++) {
    auto& te_A = scratch.a[used];
    auto& te_B = scratch.b[used];
    auto& quantized = scratch.quantized[used];
    set_input(te_A, quantized.a, A[i], "A");
    if (single_input) {
      void* ptr = input.numel() == 0 ? nullptr
                                     : static_cast<char*>(input.data_ptr()) +
                                           input_offset * input.size(1) * input.element_size();
      set_dense(te_B, ptr, input_dtype,
                Shape{static_cast<size_t>(m_splits[i]), static_cast<size_t>(input.size(1))},
                quantized.b);
      quantized.b = false;
      input_offset += m_splits[i];
    } else {
      set_input(te_B, quantized.b, input_parts[i], "B");
    }

    at::Tensor out_tensor;
    // Describe the output slice directly: no per-expert torch tensor, hence no
    // cudaPointerGetAttributes to recover the device from the raw pointer.
    void* out_slice_ptr = nullptr;
    const auto out_shape = detail::getGemmOutputShape(te_A.shape(), transa, te_B.shape(), transb);
    size_t D_numel = 1;
    for (const size_t t : out_shape) {
      NVTE_CHECK(t == 0 || D_numel <= static_cast<size_t>(INT64_MAX) / t,
                 "Grouped GEMM output shape is too large.");
      D_numel *= t;
    }
    if (single_output) {
      NVTE_CHECK(D_numel <= static_cast<size_t>((*D)[0].numel()) - output_offset,
                 "Grouped GEMM D has insufficient single-output capacity.");
      // Count all dimensions, including the outer dimensions flattened by GEMM.
      if (D_numel != 0) {
        out_slice_ptr = static_cast<char*>((*D)[0].data_ptr()) +
                        output_offset * (*D)[0].element_size();
      }
      output_offset += D_numel;
    } else if (D.has_value()) {
      out_tensor = (*D)[i];
      NVTE_CHECK(out_shape == out_tensor.sizes(), "Grouped GEMM D has an invalid output shape.");
    } else {
      const std::vector<int64_t> sizes(out_shape.begin(), out_shape.end());
      out_tensor = at::empty(sizes, torch::TensorOptions().dtype(dtype).device(device));
      D_vectors.emplace_back(out_tensor);
    }

    if (te_A.numel() == 0 || te_B.numel() == 0) {
      if (!accumulate) {
        if (out_slice_ptr != nullptr) {
          tensors_to_zero.emplace_back(
              (*D)[0].view({-1}).narrow(0, output_offset - D_numel, D_numel));
        } else if (out_tensor.defined() && out_tensor.numel() != 0) {
          tensors_to_zero.emplace_back(out_tensor);
        }
      }
      if (bias[i].numel() != 0 && grad) {
        tensors_to_zero.emplace_back(bias[i]);
      }
      if (pre_gelu_out[i].numel() != 0) tensors_to_zero.emplace_back(pre_gelu_out[i]);
      continue;
    }

    set_dense(scratch.d[used], single_output ? out_slice_ptr : out_tensor.data_ptr(), te_D_dtype,
              out_shape);

    if (prev_bias != nullptr && bias[i].is_same(*prev_bias)) {
      te_bias_vector.emplace_back(te_bias_vector.back());
    } else {
      te_bias_wrappers.emplace_back(makeTransformerEngineTensor(bias[i]));
      te_bias_vector.emplace_back(te_bias_wrappers.back().data());
      prev_bias = &bias[i];
    }

    if (prev_pre_gelu_out != nullptr && pre_gelu_out[i].is_same(*prev_pre_gelu_out)) {
      te_pre_gelu_out_vector.emplace_back(te_pre_gelu_out_vector.back());
    } else {
      const auto gelu_shape =
          pre_gelu_out[i].data_ptr() == nullptr
              ? std::vector<size_t>{static_cast<size_t>(pre_gelu_out[i].size(0))}
              : std::vector<size_t>{static_cast<size_t>(pre_gelu_out[i].size(0)),
                                    static_cast<size_t>(pre_gelu_out[i].size(1))};

      te_pre_gelu_out_wrappers.emplace_back(
          makeTransformerEngineTensor(get_data_ptr(pre_gelu_out[i]), gelu_shape, bias_type));
      te_pre_gelu_out_vector.emplace_back(te_pre_gelu_out_wrappers.back().data());
      prev_pre_gelu_out = &pre_gelu_out[i];
    }

    scratch.a_v[used] = te_A.data();
    scratch.b_v[used] = te_B.data();
    scratch.d_v[used] = scratch.d[used].data();
    used++;
  }
  scratch.resize(used);
  for (auto& tensor : tensors_to_zero) tensor.zero_();

  if (scratch.a_v.empty()) {
    return bias;
  }

  // Keep the swizzled scaling factor tensors alive during the GEMM.
  std::vector<std::optional<at::Tensor>> swizzled_scale_inverses_list;

  // Optionally swizzle the scaling factors
  swizzled_scale_inverses_list.emplace_back(
      multi_tensor_swizzle_scales_for_gemm(scratch.a, transa, !transa));
  swizzled_scale_inverses_list.emplace_back(
      multi_tensor_swizzle_scales_for_gemm(scratch.b, !transb, transb));

  // Emulate the FP8 block scaling recipe with MXFP8 on Blackwell and newer
  // as it is not natively supported by cublasLt
  if (transformer_engine::cuda::sm_arch() >= 100) {
    // Check if is using FP8 block scaling
    bool exists_tensor_using_fp8_block_scaling = false;
    bool exists_tensor_not_using_fp8_block_scaling = false;
    for (const auto& tensor_wrappers : {&scratch.a, &scratch.b}) {
      for (const TensorWrapper& tensor : *tensor_wrappers) {
        const NVTEScalingMode scaling_mode = tensor.scaling_mode();
        if (scaling_mode == NVTE_BLOCK_SCALING_1D || scaling_mode == NVTE_BLOCK_SCALING_2D)
          exists_tensor_using_fp8_block_scaling = true;
        else
          exists_tensor_not_using_fp8_block_scaling = true;
      }
    }
    if (exists_tensor_using_fp8_block_scaling) {
      NVTE_CHECK(!exists_tensor_not_using_fp8_block_scaling,
                 "Either all tensors or no tensor must be FP8 block scaling tensors");
      // Convert tensors to mxfp8 and swizzle their scaling factors
      for (TensorWrapper& A_tensor : scratch.a) {
        swizzled_scale_inverses_list.emplace_back(
            convert_block_scaling_to_mxfp8_tensor(A_tensor, transa));
      }
      for (TensorWrapper& B_tensor : scratch.b) {
        swizzled_scale_inverses_list.emplace_back(
            convert_block_scaling_to_mxfp8_tensor(B_tensor, !transb));
      }
      // Use TN GEMM to avoid having to transpose data.
      transa = true;
      transb = false;
    }
  }

  std::vector<NVTETensor> te_workspace_vector;
  std::vector<TensorWrapper> te_workspace_wrappers;
  for (size_t i = 0; i < workspace.size(); i++) {
    auto wsp = makeTransformerEngineTensor(workspace[i].data_ptr(),
                                           std::vector<size_t>{workspaceSize}, DType::kByte);
    te_workspace_vector.emplace_back(wsp.data());
    te_workspace_wrappers.emplace_back(std::move(wsp));
  }

  // For now, we only have multi-stream cublas backend.
  NVTE_SCOPED_GIL_RELEASE({
    nvte_multi_tensor_gemm(scratch.a_v.data(), scratch.b_v.data(), scratch.d_v.data(),
                           te_bias_vector.data(), te_pre_gelu_out_vector.data(),
                           static_cast<int>(scratch.a_v.size()), transa, transb, grad,
                           te_workspace_vector.data(), accumulate, use_split_accumulator,
                           math_sm_count, at::cuda::getCurrentCUDAStream());
  });
  return bias;
}

py::object te_general_grouped_gemm_for_grouped_tensor(
    py::handle A, bool transa, py::handle B, bool transb, py::handle D, py::object bias,
    std::optional<at::Tensor> bias_scale, at::Tensor alpha, at::Tensor beta,
    at::Tensor workspace_setup, at::Tensor workspace_cublas, bool use_split_accumulator,
    int math_sm_count) {
  using namespace transformer_engine::pytorch::detail;

  init_extension();
  at::cuda::CUDAGuard device_guard(workspace_cublas.device());

  auto grouped_A = GroupedTensorFromPyTorchGroupedTensor(A);
  auto grouped_B = GroupedTensorFromPyTorchGroupedTensor(B);
  auto grouped_D = GroupedTensorFromPyTorchGroupedTensor(D);

  const size_t num_tensors = grouped_A.num_tensors();
  NVTE_CHECK(num_tensors > 0, "Grouped GEMM requires non-empty inputs.");
  NVTE_CHECK(grouped_B.num_tensors() == num_tensors,
             "Grouped GEMM requires A and B to have the same num_tensors.");
  NVTE_CHECK(grouped_D.num_tensors() == num_tensors,
             "Grouped GEMM requires D to have the same num_tensors as inputs.");

  auto gemm_config = prepare_grouped_gemm_config(alpha, beta, workspace_setup, workspace_cublas,
                                                 num_tensors, math_sm_count, use_split_accumulator);

  [[maybe_unused]] auto swizzled_scales_A =
      maybe_swizzle_grouped_tensor(grouped_A, transa, !transa);
  [[maybe_unused]] auto swizzled_scales_B =
      maybe_swizzle_grouped_tensor(grouped_B, transb, !transb);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_grouped_gemm(grouped_A.data(), transa, grouped_B.data(), transb, grouped_D.data(),
                      grouped_D.data(), gemm_config.te_alpha.data(), gemm_config.te_beta.data(),
                      gemm_config.te_workspace_setup.data(), gemm_config.te_workspace_cublas.data(),
                      gemm_config.matmul_config.has_value()
                          ? static_cast<NVTEGroupedMatmulConfig>(*gemm_config.matmul_config)
                          : nullptr,
                      at::cuda::getCurrentCUDAStream());
  });

  if (!bias.is_none()) {
    auto grouped_bias = GroupedTensorFromPyTorchGroupedTensor(bias);
    if (bias_scale.has_value()) {
      check_grouped_bias_scale(*bias_scale);
      auto te_bias_scale = makeTransformerEngineTensor(*bias_scale);
      NVTE_SCOPED_GIL_RELEASE({
        nvte_grouped_scaled_bias_add(grouped_D.data(), grouped_bias.data(), te_bias_scale.data(),
                                     at::cuda::getCurrentCUDAStream());
      });
    } else {
      NVTE_SCOPED_GIL_RELEASE({
        nvte_grouped_bias_add(grouped_D.data(), grouped_bias.data(),
                              at::cuda::getCurrentCUDAStream());
      });
    }
  }

  return py::reinterpret_borrow<py::object>(D);
}

py::object te_general_grouped_gemm_for_discrete_in(py::handle A, bool transa, py::handle B,
                                                   bool transb, py::handle D, py::object bias,
                                                   std::optional<at::Tensor> bias_scale,
                                                   at::Tensor alpha, at::Tensor beta,
                                                   at::Tensor workspace_setup,
                                                   at::Tensor workspace_cublas,
                                                   bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine::pytorch::detail;

  init_extension();

  // Ensure that cublasLt handle is created on the correct device,
  // overriding torch.cuda.set_device calls from user side.
  // Assumes all tensors passed are on the same device.
  at::cuda::CUDAGuard device_guard(workspace_cublas.device());

  auto grouped_B = GroupedTensorFromPyTorchGroupedTensor(B);
  auto grouped_D = GroupedTensorFromPyTorchGroupedTensor(D);

  const auto A_list = py::cast<std::vector<py::object>>(A);
  const size_t num_tensors = grouped_B.num_tensors();
  NVTE_CHECK(num_tensors > 0, "Grouped GEMM requires non-empty inputs.");
  NVTE_CHECK(A_list.size() == num_tensors,
             "Grouped GEMM requires A_list to have num_tensors elements.");
  NVTE_CHECK(grouped_D.num_tensors() == num_tensors,
             "Grouped GEMM requires D to have the same num_tensors as inputs.");

  auto gemm_config = prepare_grouped_gemm_config(alpha, beta, workspace_setup, workspace_cublas,
                                                 num_tensors, math_sm_count, use_split_accumulator);

  std::vector<TensorWrapper> te_A_wrappers;
  std::vector<NVTETensor> te_A_vector;
  te_A_wrappers.reserve(num_tensors);
  te_A_vector.reserve(num_tensors);
  const auto none = py::none();
  for (const auto& tensor : A_list) {
    check_discrete_grouped_tensor(tensor, "A");
    te_A_wrappers.emplace_back(makeTransformerEngineTensor(tensor, none));
    te_A_vector.emplace_back(te_A_wrappers.back().data());
  }

  std::vector<std::optional<at::Tensor>> swizzled_scale_inverses_list;
  swizzled_scale_inverses_list.emplace_back(
      multi_tensor_swizzle_scales_for_gemm(te_A_wrappers, transa, !transa));

  [[maybe_unused]] auto swizzled_scales_B =
      maybe_swizzle_grouped_tensor(grouped_B, transb, !transb);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_grouped_gemm_with_discrete_inputA(
        te_A_vector.data(), num_tensors, transa, grouped_B.data(), transb, grouped_D.data(),
        grouped_D.data(), gemm_config.te_alpha.data(), gemm_config.te_beta.data(),
        gemm_config.te_workspace_setup.data(), gemm_config.te_workspace_cublas.data(),
        gemm_config.matmul_config.has_value()
            ? static_cast<NVTEGroupedMatmulConfig>(*gemm_config.matmul_config)
            : nullptr,
        at::cuda::getCurrentCUDAStream());
  });

  if (!bias.is_none()) {
    auto grouped_bias = GroupedTensorFromPyTorchGroupedTensor(bias);
    if (bias_scale.has_value()) {
      check_grouped_bias_scale(*bias_scale);
      auto te_bias_scale = makeTransformerEngineTensor(*bias_scale);
      NVTE_SCOPED_GIL_RELEASE({
        nvte_grouped_scaled_bias_add(grouped_D.data(), grouped_bias.data(), te_bias_scale.data(),
                                     at::cuda::getCurrentCUDAStream());
      });
    } else {
      NVTE_SCOPED_GIL_RELEASE({
        nvte_grouped_bias_add(grouped_D.data(), grouped_bias.data(),
                              at::cuda::getCurrentCUDAStream());
      });
    }
  }

  return py::reinterpret_borrow<py::object>(D);
}

py::object te_general_grouped_gemm_for_discrete_out(py::handle A, bool transa, py::handle B,
                                                    bool transb, py::handle D, py::object bias,
                                                    std::optional<at::Tensor> bias_scale,
                                                    at::Tensor alpha, at::Tensor beta,
                                                    at::Tensor workspace_setup,
                                                    at::Tensor workspace_cublas,
                                                    bool use_split_accumulator, int math_sm_count) {
  using namespace transformer_engine::pytorch::detail;

  init_extension();

  // Ensure that cublasLt handle is created on the correct device,
  // overriding torch.cuda.set_device calls from user side.
  // Assumes all tensors passed are on the same device.
  at::cuda::CUDAGuard device_guard(workspace_cublas.device());

  NVTE_CHECK(bias.is_none(), "Bias is not supported for discrete output grouped GEMM.");

  auto grouped_A = GroupedTensorFromPyTorchGroupedTensor(A);
  auto grouped_B = GroupedTensorFromPyTorchGroupedTensor(B);

  const auto D_list = py::cast<std::vector<py::object>>(D);
  const size_t num_tensors = grouped_A.num_tensors();
  NVTE_CHECK(num_tensors > 0, "Grouped GEMM requires non-empty inputs.");
  NVTE_CHECK(grouped_B.num_tensors() == num_tensors,
             "Grouped GEMM requires A and B to have the same num_tensors.");
  NVTE_CHECK(D_list.size() == num_tensors,
             "Grouped GEMM requires D_list to have num_tensors elements.");

  auto gemm_config = prepare_grouped_gemm_config(alpha, beta, workspace_setup, workspace_cublas,
                                                 num_tensors, math_sm_count, use_split_accumulator);

  std::vector<TensorWrapper> te_D_wrappers;
  std::vector<NVTETensor> te_D_vector;
  te_D_wrappers.reserve(num_tensors);
  te_D_vector.reserve(num_tensors);
  const auto none = py::none();
  for (const auto& tensor : D_list) {
    check_discrete_grouped_tensor(tensor, "D");
    te_D_wrappers.emplace_back(makeTransformerEngineTensor(tensor, none));
    te_D_vector.emplace_back(te_D_wrappers.back().data());
  }

  [[maybe_unused]] auto swizzled_scales_A =
      maybe_swizzle_grouped_tensor(grouped_A, transa, !transa);
  [[maybe_unused]] auto swizzled_scales_B =
      maybe_swizzle_grouped_tensor(grouped_B, transb, !transb);

  NVTE_SCOPED_GIL_RELEASE({
    nvte_grouped_gemm_with_discrete_out(
        grouped_A.data(), transa, grouped_B.data(), transb, te_D_vector.data(), num_tensors,
        te_D_vector.data(), num_tensors, gemm_config.te_alpha.data(), gemm_config.te_beta.data(),
        gemm_config.te_workspace_setup.data(), gemm_config.te_workspace_cublas.data(),
        gemm_config.matmul_config.has_value()
            ? static_cast<NVTEGroupedMatmulConfig>(*gemm_config.matmul_config)
            : nullptr,
        at::cuda::getCurrentCUDAStream());
  });

  return py::reinterpret_borrow<py::object>(D);
}
}  // namespace transformer_engine::pytorch
