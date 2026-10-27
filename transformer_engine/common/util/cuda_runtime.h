/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_

#include <cuda_runtime_api.h>

#include <string>

namespace transformer_engine {

namespace cuda {

/* \brief Number of accessible devices */
int num_devices();

/* \brief Which device is currently being used */
int current_device();

/* \brief Compute capability of device
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \return Compute capability as int. Last digit is minor revision,
 *         remaining digits are major revision.
 */
int sm_arch(int device_id = -1);

/* \brief Number of multiprocessors on a device
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \return Number of multiprocessors
 */
int sm_count(int device_id = -1);

/* \brief Static device properties used by kernel heuristics
 *
 * Values are queried once per device and cached.
 */
struct DeviceProps {
  /* Number of multiprocessors */
  int sm_count;
  /* 32-bit registers per multiprocessor */
  int regs_per_sm;
  /* Max resident threads per multiprocessor */
  int max_threads_per_sm;
  /* Shared memory per multiprocessor (bytes) */
  size_t shared_mem_per_sm;
};

/* \brief Cached static device properties
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \return Device properties
 */
const DeviceProps &device_props(int device_id = -1);

/* \brief Resident blocks per SM achievable by a compiled kernel
 *
 * Uses the CUDA occupancy API, so the result reflects the kernel's real
 * register and shared memory footprint rather than an estimate. Kernels that
 * spill registers to local memory report 0, because spilling turns register
 * traffic into memory traffic and such a configuration is never worth
 * selecting.
 *
 * Values are queried once per (kernel, block size, device) and cached.
 *
 * \param[in] kernel_func Kernel to query
 * \param[in] block_size Threads per block
 * \param[in] device_id CUDA device (default is current device)
 *
 * \return Achievable resident blocks per SM, or 0 if the kernel spills to
 *         local memory or the query fails
 */
int kernel_blocks_per_sm(const void *kernel_func, int block_size, int device_id = -1);

/* \brief Minimum and maximum stream priorities supported on device
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \param[out] low_priority Lowest priority value on device.
 *
 * \param[out] high_priority Highest priority value on device.
 */
void stream_priority_range(int *low_priority, int *high_priority, int device_id = -1);

/* \brief CUDA Multicast support status for device
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \return CUDA multicast support flag
 */
bool supports_multicast(int device_id = -1);

/* \brief Path to CUDA Toolkit headers
 *
 * The path can be configured by setting NVTE_CUDA_INCLUDE_DIR in the
 * environment. Otherwise searches in common install paths.
 *
 * \param[in] required Whether to throw exception if not found
 *
 * \return Path to include directory, or an empty string if not found
 */
const std::string &include_directory(bool required = false);

/* \brief Version number of CUDA Toolkit headers
 *
 * The headers are accessed at run-time and its CUDA version may
 * differ from compile-time and from the CUDA Runtime. The header path
 * can be configured by setting NVTE_CUDA_INCLUDE_DIR in the
 * environment (default is to search in common install paths).
 *
 * \param[in] required Whether to throw exception if headers are not
 *                     found or if version cannot be determined.
 *
 * \return CUDA version encoded as major * 1000 + minor * 10, or -1 if
 *         it could not be determined.
 */
int include_directory_version(bool required = false);

/* \brief CUDA Runtime version number at run-time
 *
 * Versions may differ between compile-time and run-time.
 */
int cudart_version();

/* \brief cuBLAS version number at run-time
 *
 * Versions may differ between compile-time and run-time.
 */
size_t cublas_version();

}  // namespace cuda

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_
