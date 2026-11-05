/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_NORMALIZATION_TUNED_DISPATCH_H_
#define TRANSFORMER_ENGINE_COMMON_NORMALIZATION_TUNED_DISPATCH_H_

#include <cstddef>

namespace transformer_engine {
namespace normalization {

// Tuned norm kernels grid-stride over rows and only synchronize within a single
// row, so arbitrary row counts are supported. LayerNorm keeps the historical
// rows % 4 == 0 restriction because its multi-CTA presets have not been
// validated for arbitrary row counts.
constexpr bool tuned_dispatch_supported(bool is_rmsnorm, size_t rows, bool is_aligned) {
  return is_aligned && (is_rmsnorm || rows % 4 == 0);
}

}  // namespace normalization
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_NORMALIZATION_TUNED_DISPATCH_H_
