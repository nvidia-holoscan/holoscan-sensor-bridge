/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <climits>
#include <stdexcept>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>

#include "iq_enc_op.hpp"

static const size_t thread_block_size = 1024;

__device__ float clamp(float value, float low, float high) {
  return fminf(fmaxf(value, low), high);
}

__global__ void cuda_encode_kernel(int16_t* encoded_output, const float* iq_components, size_t count, float scale) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  int swap_offset = -2 * static_cast<int>((i % 4)) + 3;
  size_t input_index = i + swap_offset;
  size_t i_index = (i / 4) * 8 + (i % 4);
  size_t q_index = i_index + 4;
  encoded_output[i_index] = static_cast<int16_t>(clamp(SHRT_MAX * iq_components[2 * input_index] / scale, SHRT_MIN, SHRT_MAX));
  encoded_output[q_index] = static_cast<int16_t>(clamp(SHRT_MAX * iq_components[2 * input_index + 1] / scale, SHRT_MIN, SHRT_MAX));
}

namespace hololink::operators {

void IQEncoderOp::cuda_iq_encode(int16_t* encoded_output, const float* iq_components, size_t count, float scale) {
  if (count % 4) throw std::runtime_error("count must be multiple of 4");
  cuda_encode_kernel<<<static_cast<size_t>(std::ceil(1.0 * count / thread_block_size)),
                   thread_block_size>>>(encoded_output, iq_components, count, scale);
}

}  // namespace hololink::operators