/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

namespace transformer_engine::rtc {

// Each launch handles a window of experts. Device offsets always index the
// original packed row space, including for later windows and empty experts.
struct MegaMoEGradArgs {
  static constexpr int kMaxExperts = 32;
  const int *offsets;
  const int *reduction_rows;  // Optional wgrad K lengths within the packed expert intervals.
  const void *a, *b;
  void *d;
  void *expert[kMaxExperts];  // Input weights for dgrad/recompute, destinations for wgrad.
  int rows, groups, input_width, output_width;
  bool accumulate;
  int reserve_sms;  // Host launch policy for concurrent EP gradient return.
};
}  // namespace transformer_engine::rtc
