/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "cutlass/gemm/kernel/nvte_grouped_gemm_traits.hpp"
#include "cutlass_grouped_gemm_rtc.h"
#include "cutlass_grouped_gemm_rtc_kernel.h"
#include "ppu_adaptive_tile.hpp"
#include "te_profiling_interface.h"

namespace transformer_engine::grouped_gemm_rtc {
namespace {
constexpr size_t kCPUWorkSpaceSize = 4 * 1024 * 1024;

// A slot stays reserved while the host reads its shapes. Its event also
// protects the pinned memory until the asynchronous H2D copy has consumed it.
constexpr unsigned kHostWsSlots = 8;
constexpr size_t kHostWsSlotSize = kCPUWorkSpaceSize / kHostWsSlots;

struct StagingSlot {
  std::mutex mutex;
  cudaEvent_t copied = nullptr;
  bool recorded = false;
};

struct DeviceWorkspace {
  std::once_flag init;
  std::atomic<unsigned> next{0};
  StagingSlot slots[kHostWsSlots];
  char* pinned = nullptr;

  std::mutex cache_mutex;
  char* last_workspace = nullptr;
  cudaStream_t last_stream = nullptr;
  std::vector<char> last_params;

  void init_staging() {
    if (cudaHostAlloc(&pinned, kCPUWorkSpaceSize, cudaHostAllocDefault) != cudaSuccess) {
      (void)cudaGetLastError();
      return;  // Fall back to pageable memory when the locked-memory quota is exhausted.
    }
    for (unsigned i = 0; i < kHostWsSlots; ++i) {
      if (cudaEventCreateWithFlags(&slots[i].copied, cudaEventDisableTiming) != cudaSuccess) {
        (void)cudaGetLastError();
        for (unsigned j = 0; j < i; ++j) cudaEventDestroy(slots[j].copied);
        cudaFreeHost(pinned);
        pinned = nullptr;
        return;
      }
    }
  }
};

DeviceWorkspace& device_workspace(int device) {
  static std::mutex mutex;
  static std::map<int, DeviceWorkspace> workspaces;
  std::lock_guard<std::mutex> lock(mutex);
  return workspaces[device];
}

class HostStaging {
 public:
  HostStaging(int device, size_t param_bytes)
      : device_(device_workspace(device)), param_bytes_(param_bytes) {
    std::call_once(device_.init, [&] { device_.init_staging(); });
    if (device_.pinned && param_bytes <= kHostWsSlotSize) {
      const unsigned index = device_.next.fetch_add(1, std::memory_order_relaxed) % kHostWsSlots;
      slot_ = &device_.slots[index];
      lock_ = std::unique_lock<std::mutex>(slot_->mutex);
      if (slot_->recorded) NVTE_CHECK_CUDA(cudaEventSynchronize(slot_->copied));
      data_ = device_.pinned + index * kHostWsSlotSize;
    } else {
      heap_.resize(param_bytes);
      data_ = heap_.data();
    }
  }

  char* data() const { return data_; }

  void upload(char* dest, size_t total, bool scratch_unused, cudaStream_t stream) {
    if (slot_ && total > kHostWsSlotSize) {
      heap_.assign(data_, data_ + param_bytes_);
      slot_ = nullptr;
      lock_.unlock();
    }
    if (!slot_) {
      heap_.resize(total);
      data_ = heap_.data();
    }

    static const bool cache_enabled = getenv<bool>("NVTE_GG_CACHE_PARAMS", true);
    std::lock_guard<std::mutex> lock(device_.cache_mutex);
    const bool cacheable = cache_enabled && scratch_unused;
    if (cacheable && device_.last_workspace == dest && device_.last_stream == stream &&
        device_.last_params.size() == param_bytes_ &&
        std::memcmp(device_.last_params.data(), data_, param_bytes_) == 0)
      return;

    device_.last_workspace = nullptr;
    std::memset(data_ + param_bytes_, 0, total - param_bytes_);
    NVTE_CHECK_CUDA(cudaMemcpyAsync(dest, data_, total, cudaMemcpyHostToDevice, stream));
    if (slot_) {
      auto status = cudaEventRecord(slot_->copied, stream);
      if (status != cudaSuccess) {
        // The H2D was already submitted. Finish it before releasing the slot.
        NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
        NVTE_CHECK_CUDA(status);
      }
      slot_->recorded = true;
    }
    if (cacheable) {
      device_.last_params.assign(data_, data_ + param_bytes_);
      device_.last_stream = stream;
      device_.last_workspace = dest;
    }
  }

 private:
  DeviceWorkspace& device_;
  size_t param_bytes_;
  StagingSlot* slot_ = nullptr;
  std::unique_lock<std::mutex> lock_;
  std::vector<char> heap_;
  char* data_ = nullptr;
};

template <bool trans_a, bool trans_b, int SmArch, typename Element = cutlass::bfloat16_t,
          int TileM = 256, int TileN = 128, int TileK = 64, int Stages = 0, bool OutFp32 = false>
void launch_grouped_gemm_rtc(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                             NVTETensor* workspace, bool accumulate, int num_gemms, int device,
                             int num_sms, cudaStream_t stream, TileConfig const& cfg) {
  using Traits =
      GemmTraits<trans_a, trans_b, SmArch, Element, TileM, TileN, TileK, Stages, OutFp32>;
  using GemmKernel = typename Traits::GemmKernel;

  using ElementA = typename Traits::ElementA;
  using ElementB = typename Traits::ElementB;
  using ElementC = typename Traits::ElementC;
  using ElementD = typename Traits::ElementD;
  using LayoutA = typename Traits::LayoutA;
  using LayoutB = typename Traits::LayoutB;
  using LayoutC = typename Traits::LayoutC;
  using ProblemShape = typename Traits::ProblemShape;
  using ProblemShapeType = typename Traits::ProblemShapeType;
  using InternalStrideA = typename GemmKernel::InternalStrideA;
  using InternalStrideB = typename GemmKernel::InternalStrideB;
  using InternalStrideC = typename GemmKernel::InternalStrideC;
  using InternalStrideD = typename GemmKernel::InternalStrideD;

  // Device workspace holds shapes, three pointer arrays, three stride arrays,
  // then the aligned CUTLASS scratch region. C and D share the output arrays.
  const size_t gemm_coord_size = DIVUP_TO_MULTIPLE(num_gemms * sizeof(ProblemShapeType), 128UL);
  const size_t ptr_bytes = DIVUP_TO_MULTIPLE(num_gemms * sizeof(void*), 128UL);
  const size_t ldd_bytes = DIVUP_TO_MULTIPLE(num_gemms * sizeof(int64_t), 128UL);
  const size_t param_workspace_size = gemm_coord_size + 3 * ptr_bytes + 3 * ldd_bytes;
  // Keep a valid aligned kernel-workspace pointer even when the kernel reports
  // zero bytes; some PPU scheduler/epilogue code checks the pointer itself.
  constexpr size_t kMinKernelWorkspace = 128UL;

  Tensor* wspace = convertNVTETensor(workspace[0]);
  NVTE_CHECK(wspace != nullptr, "RTC grouped GEMM requires workspace[0].");
  char* dev_ws = reinterpret_cast<char*>(wspace->data.dptr);

  // The staging ring reuses pinned slots guarded by host-side events, and the
  // launch path issues driver-API calls (function attributes, module loads).
  // Capturing that into a CUDA graph would replay a slot whose contents the
  // next host-side launch already overwrote (and the driver calls would fail),
  // so reject capture explicitly instead of producing a silently wrong graph.
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture_status));
  NVTE_CHECK(capture_status == cudaStreamCaptureStatusNone,
             "CUTLASS RTC grouped GEMM does not support CUDA graph capture; "
             "disable the RTC arm (NVTE_USE_CUTLASS_GROUPED_GEMM_RTC=0) when "
             "capturing.");

  HostStaging staging(device, param_workspace_size);
  char* host_ws = staging.data();

  // C aliases D for the list API, including its leading dimension.
  const size_t ptr_a = gemm_coord_size;
  const size_t ptr_b = ptr_a + ptr_bytes;
  const size_t ptr_d = ptr_b + ptr_bytes;
  const size_t lda = ptr_d + ptr_bytes;
  const size_t ldb = lda + ldd_bytes;
  const size_t ldd = ldb + ldd_bytes;
  auto* h_shapes = reinterpret_cast<ProblemShapeType*>(host_ws);
  auto** h_ptr_A = reinterpret_cast<ElementA**>(host_ws + ptr_a);
  auto** h_ptr_B = reinterpret_cast<ElementB**>(host_ws + ptr_b);
  auto** h_ptr_D = reinterpret_cast<ElementD**>(host_ws + ptr_d);
  auto* h_lda = reinterpret_cast<int64_t*>(host_ws + lda);
  auto* h_ldb = reinterpret_cast<int64_t*>(host_ws + ldb);
  auto* h_ldd = reinterpret_cast<int64_t*>(host_ws + ldd);

  for (int i = 0; i < num_gemms; ++i) {
    const Tensor* inputA = convertNVTETensorCheck(A[i]);
    const Tensor* inputB = convertNVTETensorCheck(B[i]);
    Tensor* outputD = convertNVTETensor(D[i]);

    // Compute per-group (m, n, k) matching the static path
    // (cutlass_grouped_gemm.cuh). After the caller's A/B swap, inputA = ext B
    // and inputB = ext A. M and K come from inputA; N comes from inputB.
    const int m =
        trans_a ? static_cast<int>(inputA->data.shape[1]) : static_cast<int>(inputA->data.shape[0]);
    const int k =
        trans_a ? static_cast<int>(inputA->data.shape[0]) : static_cast<int>(inputA->data.shape[1]);
    const int n =
        trans_b ? static_cast<int>(inputB->data.shape[0]) : static_cast<int>(inputB->data.shape[1]);

    h_shapes[i] = ProblemShapeType(m, n, k);

    h_ptr_A[i] = reinterpret_cast<ElementA*>(inputA->data.dptr);
    h_ptr_B[i] = reinterpret_cast<ElementB*>(inputB->data.dptr);
    h_ptr_D[i] = reinterpret_cast<ElementD*>(outputD->data.dptr);

    // Per-group leading dimensions (matches static path).
    h_lda[i] = LayoutA::packed({m, k}).stride(0);
    h_ldb[i] = LayoutB::packed({k, n}).stride(0);
    h_ldd[i] = LayoutC::packed({m, n}).stride(0);
  }

  auto* d_shapes = reinterpret_cast<ProblemShapeType*>(dev_ws);
  auto** d_ptr_A = reinterpret_cast<ElementA const**>(dev_ws + ptr_a);
  auto** d_ptr_B = reinterpret_cast<ElementB const**>(dev_ws + ptr_b);
  auto** d_ptr_C = reinterpret_cast<ElementC const**>(dev_ws + ptr_d);
  auto** d_ptr_D = reinterpret_cast<ElementD**>(dev_ws + ptr_d);
  auto* d_lda = reinterpret_cast<InternalStrideA*>(dev_ws + lda);
  auto* d_ldb = reinterpret_cast<InternalStrideB*>(dev_ws + ldb);
  auto* d_ldc = reinterpret_cast<InternalStrideC*>(dev_ws + ldd);
  auto* d_ldd = reinterpret_cast<InternalStrideD*>(dev_ws + ldd);

  // Construct GroupProblemShape with device + host pointers.
  ProblemShape problem_shape;
  problem_shape.num_groups = num_gemms;
  problem_shape.problem_shapes = d_shapes;
  problem_shape.host_problem_shapes = h_shapes;

  typename GemmKernel::EpilogueArguments epilogue_args{};
  auto& fusion_args = epilogue_args.thread;
  fusion_args.alpha = 1.0f;
  // The FP32 direct epilogue consumes C from ptr_C. The ordinary EVT consumes beta.
  fusion_args.beta = accumulate && !OutFp32 ? 1.0f : 0.0f;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
  epilogue_args.ptr_C = accumulate ? d_ptr_C : nullptr;
  epilogue_args.dC = d_ldc;
  epilogue_args.ptr_D = d_ptr_D;
  epilogue_args.dD = d_ldd;

  typename GemmKernel::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
  args.problem_shape = problem_shape;
  args.mainloop = {d_ptr_A, d_lda, d_ptr_B, d_ldb};
  args.epilogue = epilogue_args;
  args.hw_info = cutlass::KernelHardwareInfo{};

  args.hw_info.cu_count = num_sms;

  if (!GemmKernel::can_implement(args)) {
    NVTE_ERROR(
        "CUTLASS RTC grouped GEMM: GemmKernel::can_implement failed "
        "for num_gemms=",
        num_gemms, ".");
  }

  const size_t required_kernel_workspace = GemmKernel::get_workspace_size(args);
  const size_t kernel_workspace_size =
      DIVUP_TO_MULTIPLE(std::max(kMinKernelWorkspace, required_kernel_workspace), 128UL);
  const size_t total_workspace_size = param_workspace_size + kernel_workspace_size;
  NVTE_CHECK(total_workspace_size <= kCPUWorkSpaceSize,
             "Insufficient CPU workspace for RTC grouped GEMM (required=", total_workspace_size,
             ", available=", kCPUWorkSpaceSize, ").");
  NVTE_CHECK(total_workspace_size <= static_cast<size_t>(wspace->numel()),
             "Insufficient device workspace for RTC grouped GEMM (required=",
             static_cast<int64_t>(total_workspace_size),
             ", available=", static_cast<int64_t>(wspace->numel()), ").");

  staging.upload(dev_ws, total_workspace_size, required_kernel_workspace == 0, stream);
  // Adding CUTLASS scratch can move a pageable staging buffer.
  h_shapes = reinterpret_cast<ProblemShapeType*>(staging.data());
  args.problem_shape.host_problem_shapes = h_shapes;

  // Pass the same sized and initialized workspace used by the kernel's
  // to_underlying_arguments() split.
  void* kernel_workspace = dev_ws + param_workspace_size;
  if (GemmKernel::initialize_workspace(args, kernel_workspace, stream) !=
      cutlass::Status::kSuccess) {
    NVTE_ERROR(
        "CUTLASS RTC grouped GEMM: workspace initialization failed "
        "for num_gemms=",
        num_gemms, ".");
  }
  typename GemmKernel::Params params = GemmKernel::to_underlying_arguments(args, kernel_workspace);

  constexpr auto input_dtype =
      std::is_same_v<Element, cutlass::half_t> ? DType::kFloat16 : DType::kBFloat16;
  const auto kernel = get_or_compile(
      device,
      {SmArch, trans_a, trans_b, input_dtype, OutFp32 ? DType::kFloat32 : input_dtype, false}, cfg,
      GemmKernel::SharedStorageSize);
  constexpr int threads = GemmKernel::MaxThreadsPerBlock;

  // Apply any worker-count override along the scheduler's raster direction.
  dim3 grid = GemmKernel::get_grid_shape(params);
  const bool raster_along_n =
      (params.scheduler.raster_order_ == decltype(params.scheduler)::RasterOrder::AlongN);
  // The scheduler is a grid-stride persistent loop: each CTA starts at its
  // linearized blockIdx and advance_to_next_work() adds total_grid_size_ (=the
  // launched gridDim). Any grid size <= total_ctas covers every tile exactly
  // once. Tuned worker counts and occupancy-derived grids must be applied on
  // the non-cluster-minor dimension; otherwise the scheduler can index past
  // the problem for AlongN/AlongM rasterization.
  auto set_grid_1d = [&](uint32_t n) {
    n = n > 0 ? n : 1;
    grid = raster_along_n ? dim3(1, n, 1) : dim3(n, 1, 1);
  };

  // Tuned worker counts are production data carried by the matched table row.
  // Limit them to the measured hardware/workload envelope; all sm89 and other
  // sm80 configurations retain their previous default/occupancy grid policy.
  int workers_per_cu = 0;
  if (SmArch == 80 && args.hw_info.cu_count == 64 && num_gemms == 16 && cfg.workers_per_cu > 0) {
    // MoE routing makes the per-expert token count dynamic. For TN/NN that is
    // the GEMM M dimension; for NT/wgrad it is K. Keep only the model dimensions
    // uniform so a row selected by average MPE also applies to imbalanced experts.
    bool compatible_shapes = true;
    for (int g = 1; g < num_gemms; ++g) {
      if constexpr (trans_a && !trans_b) {
        compatible_shapes = cute::get<0>(h_shapes[g]) == cute::get<0>(h_shapes[0]) &&
                            cute::get<1>(h_shapes[g]) == cute::get<1>(h_shapes[0]);
      } else {
        compatible_shapes = cute::get<1>(h_shapes[g]) == cute::get<1>(h_shapes[0]) &&
                            cute::get<2>(h_shapes[g]) == cute::get<2>(h_shapes[0]);
      }
      if (!compatible_shapes) break;
    }
    if (compatible_shapes) workers_per_cu = cfg.workers_per_cu;
  }
  if (workers_per_cu <= 0 && cfg.tile_m <= 128) {
    // Auto multi-CTA residency for 128-row tiles that do not carry a tuned
    // worker count. Filling available resident slots overlaps the next tile's
    // pipeline fill with the current tile's epilogue.
    int occ = 0;
    if (cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, kernel->function, static_cast<int>(threads),
            static_cast<size_t>(kernel->smem_size)) == CUDA_SUCCESS &&
        occ > 0) {
      workers_per_cu = occ;
    }
  }
  if (workers_per_cu > 0) {
    uint32_t total_ctas = 0;
    for (int g = 0; g < num_gemms; g++) {
      const int gm = cute::get<0>(h_shapes[g]);
      const int gn = cute::get<1>(h_shapes[g]);
      const int ctas_m = (gm + cfg.tile_m - 1) / cfg.tile_m;
      const int ctas_n = (gn + cfg.tile_n - 1) / cfg.tile_n;
      total_ctas += static_cast<uint32_t>(ctas_m * ctas_n);
    }
    uint64_t want = uint64_t(args.hw_info.cu_count) * uint64_t(workers_per_cu);
    if (total_ctas > 0 && want > total_ctas) want = total_ctas;
    set_grid_1d(static_cast<uint32_t>(want));
  }
  dim3 block(threads, 1, 1);

  // HGTX per-launch tag (DeepGemm-style three-piece). Group extents come from
  // the host problem shapes, so the tag carries the real per-group
  // distribution with no D2H sync. Push must pair with pop on the same host
  // thread around the launch; the guard keeps them paired across exceptions.
  te_profiling::TeProfParam te_prof;
  if (te_profiling::TeProfilingInterface::Instance().enabled()) {
    std::vector<int> te_m(num_gemms), te_n(num_gemms), te_k(num_gemms);
    for (int g = 0; g < num_gemms; ++g) {
      te_m[g] = cute::get<0>(h_shapes[g]);
      te_n[g] = cute::get<1>(h_shapes[g]);
      te_k[g] = cute::get<2>(h_shapes[g]);
    }
    const int te_tile[4] = {cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.stages};
    te_prof.set_grouped_gemm("GroupedGemmRtc", input_dtype == DType::kFloat16 ? "fp16" : "bf16",
                             num_gemms, trans_a ? (trans_b ? "TT" : "TN") : (trans_b ? "NT" : "NN"),
                             accumulate, te_m.data(), te_n.data(), te_k.data(), te_tile, device,
                             stream);
  }

  void* kargs[] = {&params};
  {
    te_profiling::TeHgtxGuard te_prof_guard(te_prof);
    NVTE_CALL_CHECK_CUDA_DRIVER(cuLaunchKernel, kernel->function, grid.x, grid.y, grid.z, block.x,
                                block.y, block.z, static_cast<unsigned>(kernel->smem_size),
                                static_cast<CUstream>(stream), kargs, nullptr);
  }
  if (verbose_tile_debug()) {
    NVTE_WARN("CUTLASS RTC grouped GEMM launched: function=0x",
              reinterpret_cast<uintptr_t>(kernel->function), " grid=(", grid.x, ",", grid.y, ",",
              grid.z, ")", " block=(", block.x, ",", block.y, ",", block.z, ")",
              " smem=", kernel->smem_size, "B  num_gemms=", num_gemms, " tile=(", cfg.tile_m, ",",
              cfg.tile_n, ",", cfg.tile_k, ")", " stages=", cfg.stages,
              " out_fp32=", std::is_same_v<typename Traits::ElementD, float>);
  }
}

// ---------------------------------------------------------------------------
// Public C++ entry point (called from cublaslt_gemm.cu)
// ---------------------------------------------------------------------------

bool dispatch(const NVTETensor* A, const NVTETensor* B, NVTETensor* D, int num_gemms, bool transa,
              bool transb, NVTETensor* workspace, bool accumulate, int device, int math_sm_count,
              cudaStream_t stream) {
  if (!A || !B || !D || num_gemms <= 0) return false;
  const Tensor* inputA = convertNVTETensorCheck(A[0]);
  const bool is_bf16 = (inputA->data.dtype == DType::kBFloat16);
  const bool is_fp16 = (inputA->data.dtype == DType::kFloat16);
  if (!is_bf16 && !is_fp16) return false;

  // Output element type. D carries the operand dtype in the common case; the
  // exception is gradient accumulation into an fp32 main_grad
  // (fuse_wgrad_accumulation), which needs a 32-bit C/D while A/B stay 16-bit.
  // C is aliased to D by this launcher (there is no separate C tensor), so a
  // single flag describes both.
  const DType out_dtype = convertNVTETensorCheck(D[0])->data.dtype;
  if (out_dtype != inputA->data.dtype && out_dtype != DType::kFloat32) {
    return false;
  }
  const bool out_fp32 = (out_dtype == DType::kFloat32);

  const int sm_arch = cuda::sm_arch(device);
  if (sm_arch != 80 && sm_arch != 89) return false;

  bool ne4 = sm_arch == 89 && is_bf16 && !transb && !accumulate;
  uint64_t expanded_tiles = 0;

  // CUTLASS sees the operands in the opposite order: B supplies M/K, A supplies N.
  const bool ta = transb;
  const bool tb = transa;
  int64_t sum_m = 0, sum_n = 0, sum_k = 0;
  for (int i = 0; i < num_gemms; ++i) {
    const Tensor* tin = convertNVTETensorCheck(B[i]);  // input  (M, K)
    const Tensor* twt = convertNVTETensorCheck(A[i]);  // weight (N)
    // The kernel compiles one output element type for every group, so a mixed
    // group set has to fall back instead of being silently mistyped.
    if (convertNVTETensorCheck(D[i])->data.dtype != out_dtype) return false;
    const int m = ta ? static_cast<int>(tin->data.shape[1]) : static_cast<int>(tin->data.shape[0]);
    const int k = ta ? static_cast<int>(tin->data.shape[0]) : static_cast<int>(tin->data.shape[1]);
    const int n = tb ? static_cast<int>(twt->data.shape[0]) : static_cast<int>(twt->data.shape[1]);
    if (ne4) {
      const auto* dest = convertNVTETensorCheck(D[i]);
      ne4 = tin->data.shape.size() == 2 && twt->data.shape.size() == 2 &&
            dest->data.shape.size() == 2 && ne4_shape(m, n, k) &&
            tin->dtype() == DType::kBFloat16 && twt->dtype() == DType::kBFloat16 &&
            (reinterpret_cast<uintptr_t>(tin->data.dptr) |
             reinterpret_cast<uintptr_t>(twt->data.dptr) |
             reinterpret_cast<uintptr_t>(dest->data.dptr)) %
                    16 ==
                0;
      expanded_tiles += uint64_t(m / 256) * (n / 1024);
    }
    sum_m += m;
    sum_n += n;
    sum_k += k;
  }
  const int avg_m = static_cast<int>(sum_m / num_gemms);
  const int avg_n = static_cast<int>(sum_n / num_gemms);
  const int avg_k = static_cast<int>(sum_k / num_gemms);

  // CU count for the adaptive heuristic: prefer the caller's math_sm_count,
  // then the device query, then the PPU SM89 default.
  int num_sms = math_sm_count;
  if (num_sms <= 0) num_sms = cuda::sm_count();
  if (num_sms <= 0) num_sms = 39;

  TileConfig cfg =
      select_grouped_tile(avg_m, avg_n, avg_k, num_gemms, transa, transb, sm_arch, num_sms);
  if (ne4) select_n_expand(cfg, expanded_tiles, num_sms);

  if (verbose_tile_debug()) {
    NVTE_WARN("CUTLASS RTC grouped GEMM dispatch: num_gemms=", num_gemms, " sm_arch=", sm_arch,
              " transa=", transa, " transb=", transb, " accumulate=", accumulate,
              " out_fp32=", out_fp32, " avg_shape=(", avg_m, ",", avg_n, ",", avg_k, ")", " tile=(",
              cfg.tile_m, ",", cfg.tile_n, ",", cfg.tile_k, ") stages=", cfg.stages);
  }

  // TE and CUTLASS use opposite operand order.
  auto run = [&](auto ta_const, auto tb_const, auto arch_const, auto out_const) {
    constexpr bool ta = decltype(ta_const)::value;
    constexpr bool tb = decltype(tb_const)::value;
    constexpr int Arch = decltype(arch_const)::value;
    constexpr bool OutFp32 = decltype(out_const)::value;
#define NVTE_GG_LAUNCH_TILE_STAGES(TM, TN, TK, ST)                                         \
  if (cfg.matches(TM, TN, TK, ST)) {                                                       \
    if (is_fp16) {                                                                         \
      launch_grouped_gemm_rtc<ta, tb, Arch, cutlass::half_t, TM, TN, TK, ST, OutFp32>(     \
          B, A, D, workspace, accumulate, num_gemms, device, num_sms, stream, cfg);        \
    } else {                                                                               \
      launch_grouped_gemm_rtc<ta, tb, Arch, cutlass::bfloat16_t, TM, TN, TK, ST, OutFp32>( \
          B, A, D, workspace, accumulate, num_gemms, device, num_sms, stream, cfg);        \
    }                                                                                      \
    return;                                                                                \
  }
    NVTE_GG_TILE_TABLE(NVTE_GG_LAUNCH_TILE_STAGES)
#undef NVTE_GG_LAUNCH_TILE_STAGES
    NVTE_ERROR("CUTLASS RTC grouped GEMM: no host instantiation for tile=(", cfg.tile_m, ",",
               cfg.tile_n, ",", cfg.tile_k, ") stages=", cfg.stages);
  };

  auto dispatch_layout = [&](auto arch_const, auto out_const) -> bool {
    using F = std::false_type;
    using T = std::true_type;
    if (!transa && !transb) {
      run(F{}, F{}, arch_const, out_const);
      return true;
    }
    if (!transb && transa) {
      run(F{}, T{}, arch_const, out_const);
      return true;
    }
    if (transb && !transa) {
      run(T{}, F{}, arch_const, out_const);
      return true;
    }
    return false;  // TT not supported
  };

  auto dispatch_output = [&](auto arch) {
    return out_fp32 ? dispatch_layout(arch, std::true_type{})
                    : dispatch_layout(arch, std::false_type{});
  };
  return sm_arch == 80 ? dispatch_output(std::integral_constant<int, 80>{})
                       : dispatch_output(std::integral_constant<int, 89>{});
}

}  // namespace

void invalidate_workspace(int device) {
  auto& workspace = device_workspace(device);
  std::lock_guard<std::mutex> lock(workspace.cache_mutex);
  workspace.last_workspace = nullptr;
}

}  // namespace transformer_engine::grouped_gemm_rtc

bool cutlass_grouped_gemm_rtc(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                              int num_gemms, bool transa, bool transb, NVTETensor* workspace,
                              bool accumulate, int device, int math_sm_count, cudaStream_t stream) {
  try {
    return transformer_engine::grouped_gemm_rtc::dispatch(
        A, B, D, num_gemms, transa, transb, workspace, accumulate, device, math_sm_count, stream);
  } catch (const std::exception& e) {
    if (transformer_engine::getenv<bool>("NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK", false)) {
      NVTE_WARN("CUTLASS RTC grouped GEMM failed: ", e.what());
    }
    return false;
  }
}
