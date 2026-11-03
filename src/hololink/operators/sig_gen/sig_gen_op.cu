/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "sig_gen_op.cuh"

#include <cmath>
#include <stdexcept>

#include <hololink/common/cuda_error.hpp>
#include <hololink/core/logging_internal.hpp>

static const size_t thread_block_size = 1024;

__global__ void memset_kernel(float* output, int count, float value, unsigned stride)
{
  size_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= count) return; // Check to see of out-of-bounds

  output_index *= stride;
  output[output_index] = value;
}

namespace hololink::operators {

// Must match the Context struct the 'x' variable getter below declares.
struct SignalContext {
  size_t index_;
  int samples_per_period_;
  double sampling_interval_;
};

SignalContext* create_signal_context() {
  SignalContext* device_context = nullptr;
  CUDA_CHECK(cudaMalloc(&device_context, sizeof(SignalContext)));
  return device_context;
}

void destroy_signal_context(SignalContext* device_context) {
  if (device_context)
    cudaFree(device_context);
}

expr_eval::Expression create_signal_expression(expr_eval::Parser& parser, const std::string& signal_expression_string) {
  hololink::expr_eval::VariablesSymbolTable variables_symbol_table{
    // Variable X getter
    {"x", R"(
      struct Context {
        size_t index_;
        int samples_per_period_;
        double sampling_interval_;
      };

      auto context = reinterpret_cast<const Context*>(data);
      unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x + context->index_; // Doesn't work with uint64_t for some reason
      i = i % context->samples_per_period_;
      return static_cast<float>(i * context->sampling_interval_);
    )"}
  };
  hololink::expr_eval::ConstantsSymbolTable constants_symbol_table;
  expr_eval::add_predefined_constants(constants_symbol_table);
  return parser.compile(signal_expression_string, variables_symbol_table, constants_symbol_table);
}

size_t evaluate_expression(const expr_eval::Expression& expression, float* device_output, size_t count,
                        size_t samples_index, const Rational& sampling_interval, size_t stride,
                        SignalContext* device_context, cudaStream_t stream) {
  const SignalContext host_context{
    samples_index,
    sampling_interval.den_ / sampling_interval.gcd(), // Samples per Period calculation
    static_cast<double>(sampling_interval)};
  // From pageable memory, so host_context is consumed before this returns.
  CUDA_CHECK(cudaMemcpyAsync(device_context, &host_context, sizeof(host_context), cudaMemcpyHostToDevice, stream));

  if (expression)
    expression.evaluate(device_output, count, stride, device_context, stream);
  else if (count) {
    HSB_LOG_DEBUG("Failed to evaluate expression - Invalid Expression");
    memset_kernel<<<static_cast<size_t>(std::ceil(1.0 * count / thread_block_size)),
                    thread_block_size, 0, stream>>>(device_output, count, 0, stride);
  }

  return (host_context.index_ + count) % host_context.samples_per_period_;
}

}  // namespace hololink::operators
