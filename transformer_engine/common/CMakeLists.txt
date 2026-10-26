# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)

# Repo root, used by the actlize effective-tree materialization below.
set(NVTE_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/../..)

# Language options
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 89 90 100 120)
  elseif (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90 100 120)
  else ()
    set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
  endif()
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
endif()

# Hide non-necessary symbols in shared object.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libtransformer_engine.version")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libtransformer_engine.version")

# Transformer Engine library
project(transformer_engine LANGUAGES CUDA CXX)

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_VERSION VERSION_LESS 12.0)
  message(FATAL_ERROR "CUDA 12.0+ is required, but found CUDA ${CUDAToolkit_VERSION}")
endif()

# cuDNN frontend API
set(CUDNN_FRONTEND_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cudnn-frontend/include")
if(NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}")
    message(FATAL_ERROR
            "Could not find cuDNN frontend API at ${CUDNN_FRONTEND_INCLUDE_DIR}. "
            "Try running 'git submodule update --init --recursive' "
            "within the Transformer Engine source.")
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cudnn-frontend/cmake/cuDNN.cmake)

set(CUTLASS_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cutlass/include")
set(CUTLASS_TOOLS_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cutlass/tools/util/include")

# The RTC grouped-GEMM path was developed against a separate cutlass3 checkout.
# v2.16_dev uses the actlize submodule as the CUTLASS 3.x + PPU header source
# (actlize is a PPU-modified CUTLASS 3.x). Redirect the legacy names there.
set(ACTLIZE_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/actlize/include")
set(ACTLIZE_TOOLS_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/actlize/tools/util/include")

# TE-maintained overlay over the actlize submodule: CUDA-converted headers that
# carry the RTC grouped-GEMM mainloop and epilogue implementations.
# The submodule itself stays at pristine upstream v1.0.0 and is never modified
# in place.
set(ACTLIZE_OVERLAY_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/gemm/actlize_overlay/include")

# Materialize the single effective include tree that the whole build (the host
# nvcc TU and runtime NVRTC alike) compiles against: a copy of the pristine
# actlize headers with the overlay files copied over them. One physical tree
# is REQUIRED: these headers use #pragma once (per-path dedup), so a pure
# -I-order overlay would expand both the overlay copy and the submodule copy
# and fail with redefinition errors (some includes also resolve relatively
# from the include root before -I is consulted). Like the cudafy step below,
# this snapshot is taken at configure time -- re-run cmake after editing
# actlize or overlay headers.
set(NVTE_GG_EFFECTIVE_ACTLIZE "${CMAKE_BINARY_DIR}/actlize_effective")
set(NVTE_GG_EFFECTIVE_ACTLIZE_INCLUDE "${NVTE_GG_EFFECTIVE_ACTLIZE}/include")
set(NVTE_GG_EFFECTIVE_ACTLIZE_TOOLS_INCLUDE
  "${NVTE_GG_EFFECTIVE_ACTLIZE}/tools/util/include")
file(COPY "${ACTLIZE_INCLUDE_DIR}/"
     DESTINATION "${NVTE_GG_EFFECTIVE_ACTLIZE_INCLUDE}/")
file(COPY "${ACTLIZE_OVERLAY_INCLUDE_DIR}/cutlass/"
     DESTINATION "${NVTE_GG_EFFECTIVE_ACTLIZE_INCLUDE}/cutlass/")
# cudafy.py locates tools headers as <include_dir>/../tools/util/include, so
# the effective tree must mirror the actlize repo layout.
file(COPY "${ACTLIZE_TOOLS_INCLUDE_DIR}/"
     DESTINATION "${NVTE_GG_EFFECTIVE_ACTLIZE_TOOLS_INCLUDE}/")

# Run cudafy on the effective tree (hggc -> cuda mechanical conversion).
# Idempotent; the overlay files are already in CUDA form and pass through
# unchanged, and the submodule working tree is left pristine.
execute_process(
  COMMAND python3 ${NVTE_ROOT_DIR}/3rdparty/cudafy-for-sail/cudafy.py
          actlize --version=1.0.0 ${NVTE_GG_EFFECTIVE_ACTLIZE_INCLUDE}
  WORKING_DIRECTORY ${NVTE_ROOT_DIR}
  RESULT_VARIABLE CUDAFY_RESULT
  OUTPUT_VARIABLE CUDAFY_OUTPUT
  ERROR_VARIABLE CUDAFY_OUTPUT
)
if(NOT CUDAFY_RESULT EQUAL 0)
  message(FATAL_ERROR "cudafy.py failed (exit code ${CUDAFY_RESULT}):\n${CUDAFY_OUTPUT}")
endif()
message(STATUS "cudafy.py actlize (effective tree) done")

# The RTC grouped-GEMM path compiles against the effective tree: pristine
# actlize + TE overlay files, cudafy-converted. Runtime NVRTC resolves this
# same directory via the build-time baked path (string_path_cutlass3_include).
set(CUTLASS3_INCLUDE_DIR "${NVTE_GG_EFFECTIVE_ACTLIZE_INCLUDE}")
set(CUTLASS3_TOOLS_INCLUDE_DIR "${NVTE_GG_EFFECTIVE_ACTLIZE_TOOLS_INCLUDE}")

# Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# NVIDIA MathDX include directory (from Python package install location)
if(NOT DEFINED MATHDX_INCLUDE_DIR)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -m pip show nvidia-mathdx
    OUTPUT_VARIABLE _PIP_SHOW_MATHDX
    ERROR_VARIABLE _PIP_SHOW_MATHDX_ERR
    RESULT_VARIABLE _PIP_SHOW_MATHDX_RES
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _PIP_SHOW_MATHDX_RES EQUAL 0)
    message(FATAL_ERROR "Failed to query 'nvidia-mathdx' with pip (using ${Python_EXECUTABLE}): ${_PIP_SHOW_MATHDX_ERR}")
  endif()
  string(REGEX MATCH "Location: ([^\n\r]+)" _MATHDX_LOC_MATCH "${_PIP_SHOW_MATHDX}")
  if(NOT _MATHDX_LOC_MATCH)
    message(FATAL_ERROR "Could not parse installation location for 'nvidia-mathdx'. Output was:\n${_PIP_SHOW_MATHDX}")
  endif()
  set(MATHDX_LOCATION "${CMAKE_MATCH_1}")
  set(MATHDX_INCLUDE_DIR "${MATHDX_LOCATION}/nvidia/mathdx/include")
endif()
if(NOT EXISTS "${MATHDX_INCLUDE_DIR}")
  message(FATAL_ERROR "MATHDX include directory not found at ${MATHDX_INCLUDE_DIR}. Set MATHDX_INCLUDE_DIR or ensure 'nvidia-mathdx' is installed for ${Python_EXECUTABLE}.")
endif()

# Configure Transformer Engine library
include_directories(${PROJECT_SOURCE_DIR}/..)
set(transformer_engine_SOURCES)
list(APPEND transformer_engine_SOURCES
     cudnn_utils.cpp
     transformer_engine.cpp
     common.cu
     multi_tensor/adam.cu
     multi_tensor/compute_scale.cu
     multi_tensor/l2norm.cu
     multi_tensor/scale.cu
     multi_tensor/sgd.cu
     transpose/cast_transpose.cu
     transpose/transpose.cu
     transpose/cast_transpose_fusion.cu
     transpose/transpose_fusion.cu
     transpose/multi_cast_transpose.cu
     transpose/quantize_transpose_square_blockwise.cu
     transpose/quantize_transpose_vector_blockwise.cu
     transpose/swap_first_dims.cu
     transpose/quantize_transpose_vector_blockwise_fp4.cu
     activation/gelu.cu
     dropout/dropout.cu
     fused_attn/flash_attn.cu
     fused_attn/context_parallel.cu
     fused_attn/kv_cache.cu
     fused_attn/fused_attn_f16_max512_seqlen.cu
     fused_attn/fused_attn_f16_arbitrary_seqlen.cu
     activation/relu.cu
     activation/swiglu.cu
     fused_attn/fused_attn_fp8.cu
     fused_attn/fused_attn.cpp
     fused_attn/utils.cu
     gemm/config.cpp
     gemm/cublaslt_gemm.cu
     gemm/cutlass_grouped_gemm.cu
     gemm/cutlass_grouped_gemm_rtc.cu
     normalization/common.cpp
     normalization/layernorm/ln_api.cpp
     normalization/layernorm/ln_bwd_semi_cuda_kernel.cu
     normalization/layernorm/ln_fwd_cuda_kernel.cu
     normalization/rmsnorm/rmsnorm_api.cpp
     normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
     normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu
     permutation/permutation.cu
     util/cast.cu
     util/padding.cu
     util/cuda_driver.cpp
     util/cuda_nvml.cpp
     util/cuda_runtime.cpp
     util/multi_stream.cpp
     util/rtc.cpp
     swizzle/swizzle.cu
     fused_softmax/scaled_masked_softmax.cu
     fused_softmax/scaled_upper_triang_masked_softmax.cu
     fused_softmax/scaled_aligned_causal_masked_softmax.cu
     fused_rope/fused_rope.cu
     fused_router/fused_moe_aux_loss.cu
     fused_router/fused_score_for_moe_aux_loss.cu
     fused_router/fused_topk_with_score_function.cu
     recipe/current_scaling.cu
     recipe/delayed_scaling.cu
     recipe/fp8_block_scaling.cu
     recipe/nvfp4.cu
     hadamard_transform/hadamard_transform.cu
     hadamard_transform/hadamard_transform_cast_fusion.cu
     comm_gemm_overlap/userbuffers/ipcsocket.cc
     comm_gemm_overlap/userbuffers/userbuffers-host.cpp
     comm_gemm_overlap/userbuffers/userbuffers.cu
     comm_gemm_overlap/comm_gemm_overlap.cpp)

if (NVTE_WITH_CUBLASMP)
list(APPEND transformer_engine_SOURCES
     comm_gemm/comm_gemm.cpp)
endif()

# The RTC grouped-GEMM host TU mirrors PPU CUTLASS 3.x types (same Traits as
# the NVRTC kernel template at gemm/rtc/cutlass_grouped_gemm.cu), so it needs
# CUTLASS 3.x + ppu/ headers and USE_SAIL=1 just like the statically-built
# PPU paths. This is a per-source addition because the rtc host TU
# lives in the main `transformer_engine` target rather than
# `transformer_engine_groupgemm`.
set_property(
  SOURCE gemm/cutlass_grouped_gemm_rtc.cu
  APPEND PROPERTY INCLUDE_DIRECTORIES
  "${CUTLASS3_INCLUDE_DIR}"
  "${CUTLASS3_TOOLS_INCLUDE_DIR}")
# The PPU generation is selected entirely by the ArchTag template argument
# (cutlass::arch::PPU0010 for sm80, PPU0015 for sm89), which this TU picks at
# runtime from the device's compute capability -- exactly like the static
# cutlass_grouped_gemm_ppu.cu path. No preprocessor version macro is involved,
# so one host TU covers both PPU generations.
set_property(
  SOURCE gemm/cutlass_grouped_gemm_rtc.cu
  APPEND PROPERTY COMPILE_DEFINITIONS USE_SAIL=1)

# HGTX (PPU's NVTX equivalent) per-launch tag instrumentation for the grouped
# GEMM backends (RTC / static PPU CUTLASS / cuBLAS fallback), a port of
# DeepGemm's profiling_interface.cuh (see DeepGemm/nvtx_tag_replay_guide.md).
# The tag headers are header-only via the PPU SDK's hgtx3/hgToolsExt.h;
# when that header is absent (non-PPU hosts) the instrumentation compiles to
# a no-op and these sources build unchanged. Runtime switches (export before
# process start, same contract as DeepGemm):
#   PPU_LIB_PERF_INSTRUMENT=1  HGTX range push/pop around every launch
#   PPU_LIB_SHOW_PARAMS=1      print the full [TE] --format= tag to stdout
#   acu --hgtx --hgtx-include 'tengine@<range_name>/' filters by tag.
set(NVTE_HGTX_SDK_INCLUDE
    "/usr/local/PPU_SDK/targets/x86_64-linux/include"
    CACHE PATH "Directory providing hgtx3/hgToolsExt.h for grouped GEMM HGTX tags")
if(EXISTS "${NVTE_HGTX_SDK_INCLUDE}/hgtx3/hgToolsExt.h")
  message(STATUS "HGTX grouped GEMM tag instrumentation enabled (${NVTE_HGTX_SDK_INCLUDE})")
  set(NVTE_HGTX_SOURCES
      gemm/cublaslt_gemm.cu
      gemm/cutlass_grouped_gemm_ppu.cu
      gemm/cutlass_grouped_gemm_rtc.cu)
  # Include path points at the hgtx3 directory itself, NOT .../include: that
  # tree shadows same-named CUDA SDK headers (vector_types.h, host_defines.h,
  # driver_types.h, cub/, ...) and breaks the CUDA host compile of these TUs.
  set_property(SOURCE ${NVTE_HGTX_SOURCES} APPEND PROPERTY
               INCLUDE_DIRECTORIES "${NVTE_HGTX_SDK_INCLUDE}/hgtx3")
  set_property(SOURCE ${NVTE_HGTX_SOURCES} APPEND PROPERTY
               COMPILE_DEFINITIONS NVTE_ENABLE_HGTX=1)
else()
  message(STATUS "HGTX grouped GEMM tag instrumentation disabled (no hgtx3/hgToolsExt.h under ${NVTE_HGTX_SDK_INCLUDE})")
endif()

add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
target_include_directories(transformer_engine PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
  set_source_files_properties(
    "gemm/cutlass_grouped_gemm.cu"
    PROPERTIES
    COMPILE_FLAGS
    "-gencode arch=compute_90a,code=sm_90a")
else()
  message(FATAL_ERROR "cutlass gemm/cutlass_grouped_gemm.cu kernel required sm 90a")
endif()

# PPU (SM80/SM89) grouped GEMM is built into a separate static library so that it
# can be compiled against actlize headers without affecting the rest of the library.
set(transformer_engine_grouped_gemm_sources
    gemm/cutlass_grouped_gemm_ppu.cu
    )

add_library(transformer_engine_groupgemm_ppu STATIC ${transformer_engine_grouped_gemm_sources})
set_property(TARGET transformer_engine_groupgemm_ppu PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET transformer_engine_groupgemm_ppu PROPERTY CUDA_ARCHITECTURES OFF)

set_property(
  SOURCE gemm/cutlass_grouped_gemm_ppu.cu
  APPEND
  PROPERTY
  COMPILE_OPTIONS "--generate-code=arch=compute_80a,code=sm_80a;--generate-code=arch=compute_89,code=sm_89;-g0")

target_include_directories(transformer_engine_groupgemm_ppu PRIVATE
                          ${NVTE_GG_EFFECTIVE_ACTLIZE_INCLUDE}
                          ${NVTE_GG_EFFECTIVE_ACTLIZE_TOOLS_INCLUDE})
target_include_directories(transformer_engine_groupgemm_ppu PUBLIC
                          "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(transformer_engine_groupgemm_ppu PRIVATE
                          "${CUDNN_FRONTEND_INCLUDE_DIR}")
target_compile_definitions(transformer_engine_groupgemm_ppu PRIVATE
                          NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING
                          )
target_link_libraries(transformer_engine PUBLIC transformer_engine_groupgemm_ppu)

# Configure dependencies
target_link_libraries(transformer_engine PUBLIC
                      CUDA::cublas
                      CUDA::cudart
                      CUDNN::cudnn_all)

target_include_directories(transformer_engine PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(transformer_engine PRIVATE ${MATHDX_INCLUDE_DIR})
target_include_directories(transformer_engine SYSTEM PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/cccl)
target_include_directories(transformer_engine PRIVATE "${CUDNN_FRONTEND_INCLUDE_DIR}")
target_include_directories(transformer_engine PRIVATE
                          ${CUTLASS_INCLUDE_DIR}
                          ${CUTLASS_TOOLS_INCLUDE_DIR})

# Compiling Userbuffers with native MPI bootstrapping requires linking against MPI
option(NVTE_UB_WITH_MPI "Bootstrap Userbuffers with MPI" OFF)
if (NVTE_UB_WITH_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(transformer_engine PUBLIC MPI::MPI_CXX)
    target_include_directories(transformer_engine PRIVATE ${MPI_CXX_INCLUDES})
    target_compile_definitions(transformer_engine PUBLIC NVTE_UB_WITH_MPI)
endif()

option(NVTE_ENABLE_NVSHMEM "Compile with NVSHMEM library" OFF)
if (NVTE_ENABLE_NVSHMEM)
    add_subdirectory(nvshmem_api)
    target_link_libraries(transformer_engine PUBLIC nvshmemapi)
    target_include_directories(transformer_engine PUBLIC ${NVSHMEMAPI_INCLUDE_DIR})
endif()

option(NVTE_WITH_CUBLASMP "Use cuBLASMp for tensor parallel GEMMs" OFF)
if (NVTE_WITH_CUBLASMP)
    target_compile_definitions(transformer_engine PRIVATE NVTE_WITH_CUBLASMP)
    target_include_directories(transformer_engine PRIVATE ${CUBLASMP_DIR}/include ${NVSHMEM_DIR}/include)
    find_library(CUBLASMP_LIB
                 NAMES cublasmp libcublasmp
                 PATHS ${CUBLASMP_DIR}
                 PATH_SUFFIXES lib
                 REQUIRED)
    find_library(NVSHMEM_HOST_LIB
                 NAMES nvshmem_host libnvshmem_host.so.3
                 PATHS ${NVSHMEM_DIR}
                 PATH_SUFFIXES lib
                 REQUIRED)
  target_link_libraries(transformer_engine PUBLIC ${CUBLASMP_LIB} ${NVSHMEM_HOST_LIB})
  message(STATUS "Using cuBLASMp at: ${CUBLASMP_DIR}")
  message(STATUS "Using nvshmem at: ${NVSHMEM_DIR}")
endif()

# Hack to enable dynamic loading in cuDNN frontend
target_compile_definitions(transformer_engine PUBLIC NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)

# Helper functions to make header files with C++ strings
function(make_string_header STRING STRING_NAME)
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()
function(make_string_header_from_file file_ STRING_NAME)
    file(READ "${file_}" STRING)
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()

# Header files with C++ strings
list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 cuda_include_path)
make_string_header("${cuda_include_path}"
                   string_path_cuda_include)
make_string_header("${CUTLASS3_INCLUDE_DIR}"
                   string_path_cutlass3_include)
make_string_header_from_file(transpose/rtc/cast_transpose_fusion.cu
                             string_code_transpose_rtc_cast_transpose_fusion_cu)
make_string_header_from_file(transpose/rtc/cast_transpose.cu
                             string_code_transpose_rtc_cast_transpose_cu)
make_string_header_from_file(transpose/rtc/transpose.cu
                             string_code_transpose_rtc_transpose_cu)
make_string_header_from_file(transpose/rtc/swap_first_dims.cu
                             string_code_transpose_rtc_swap_first_dims_cu)
make_string_header_from_file(utils.cuh
                             string_code_utils_cuh)
make_string_header_from_file(util/math.h
                             string_code_util_math_h)
make_string_header_from_file(gemm/rtc/cutlass_grouped_gemm.cu
                             string_code_gemm_rtc_cutlass_grouped_gemm_cu)
target_include_directories(transformer_engine PRIVATE
                           "${CMAKE_CURRENT_BINARY_DIR}/string_headers")

# Compiler options
set_source_files_properties(fused_softmax/scaled_masked_softmax.cu
                            fused_softmax/scaled_upper_triang_masked_softmax.cu
                            fused_softmax/scaled_aligned_causal_masked_softmax.cu
                            multi_tensor/adam.cu
                            multi_tensor/compute_scale.cu
                            multi_tensor/l2norm.cu
                            multi_tensor/scale.cu
                            multi_tensor/sgd.cu
                            fused_attn/flash_attn.cu
                            fused_attn/context_parallel.cu
                            fused_attn/kv_cache.cu
                            PROPERTIES
                            COMPILE_OPTIONS "--use_fast_math")
option(NVTE_BUILD_ACTIVATION_WITH_FAST_MATH "Compile activation kernels with --use_fast_math option" OFF)
if (NVTE_BUILD_ACTIVATION_WITH_FAST_MATH)
  set_source_files_properties(activation/gelu.cu
                              activation/relu.cu
                              activation/swiglu.cu
                              util/cast.cu
                              PROPERTIES
                              COMPILE_OPTIONS "--use_fast_math")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Number of parallel build jobs
if(ENV{MAX_JOBS})
  set(BUILD_JOBS_STR "$ENV{MAX_JOBS}")
elseif(ENV{NVTE_BUILD_MAX_JOBS})
  set(BUILD_JOBS_STR "$ENV{NVTE_BUILD_MAX_JOBS}")
else()
  set(BUILD_JOBS_STR "max")
endif()
message(STATUS "Parallel build jobs: ${BUILD_JOBS_STR}")

# Number of threads per parallel build job
set(BUILD_THREADS_PER_JOB $ENV{NVTE_BUILD_THREADS_PER_JOB})
if (NOT BUILD_THREADS_PER_JOB)
  set(BUILD_THREADS_PER_JOB 1)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads ${BUILD_THREADS_PER_JOB}")
message(STATUS "Threads per parallel build job: ${BUILD_THREADS_PER_JOB}")

# Install library
install(TARGETS transformer_engine DESTINATION .)

# Install CUTLASS 3.x + PPU headers for runtime NVRTC compilation.
# At runtime, resolve_cutlass_include_dir() uses the build-time baked path
# (string_path_cutlass3_include) which points here for installed packages.
install(DIRECTORY "${CUTLASS3_INCLUDE_DIR}/cutlass"
                  "${CUTLASS3_INCLUDE_DIR}/cute"
                  "${CUTLASS3_INCLUDE_DIR}/ppu"
        DESTINATION include/cutlass3
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inl"
                       PATTERN "*.cuh")
# The PPU headers also live at the root of the include dir: the RTC source
# includes "ppu_include.hpp" directly (which in turn includes
# "accutlass.hpp"), and resolve_cutlass_include_dir() validates the installed
# tree by requiring cutlass/cutlass.h *and* ppu_include.hpp next to each
# other. Without them the installed package cannot resolve the include dir
# and the RTC arm is disabled at runtime.
foreach(_ppu_root_header ppu_include.hpp accutlass.hpp)
  if(EXISTS "${CUTLASS3_INCLUDE_DIR}/${_ppu_root_header}")
    install(FILES "${CUTLASS3_INCLUDE_DIR}/${_ppu_root_header}"
            DESTINATION include/cutlass3)
  endif()
endforeach()
