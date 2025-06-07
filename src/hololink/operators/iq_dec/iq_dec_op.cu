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

#include "iq_dec_op.hpp"

static const size_t thread_block_size = 1024;

__global__ void cuda_decode_kernel(float* iq_component, const int16_t* encoded_input, size_t count, float scale) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  int swap_offset = -2 * static_cast<int>((i % 4)) + 3;
  size_t output_index = 2 * (i + swap_offset);  // 2 * because the I and Q are interleaved
  size_t i_index = (i / 4) * 8 + (i % 4);
  size_t q_index = i_index + 4;
  iq_component[output_index] = static_cast<float>(encoded_input[i_index]) * scale / SHRT_MAX;
  iq_component[output_index + 1] = static_cast<float>(encoded_input[q_index]) * scale / SHRT_MAX;
}

namespace hololink::operators {

void IQDecoderOp::cuda_iq_decode(float* iq_component, const int16_t* encoded_input, size_t count, float scale) {
  if (count % 4) throw std::runtime_error("size must be multiple of 4");
  cuda_decode_kernel<<<static_cast<size_t>(std::ceil(1.0 * count / thread_block_size)),
                   thread_block_size>>>(iq_component, encoded_input, count, scale);
}

}  // namespace hololink::operators
