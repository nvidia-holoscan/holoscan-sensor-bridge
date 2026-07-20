/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <hololink/core/csi_controller.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include <cuda.h>

void shift_and_cast_kernel(
    const uint16_t* in,
    uint8_t* out,
    int count,
    cudaStream_t stream);

void grayscale_kernel_launch(
    const uint16_t* input,
    uint8_t* rgb,
    int size,
    cudaStream_t stream,
    float max_val);

void jet_kernel_launch(
    const uint16_t* depth,
    uint8_t* rgb,
    int size,
    cudaStream_t stream);

void unpack_kernel_launch(
    const uint8_t* raw,
    uint16_t* depth,
    uint16_t* conf,
    uint16_t* ab,
    int width,
    int height,
    cudaStream_t stream);    
namespace hololink::operators {

/**
 * Operator that enables the HSB packetizer to output 10- and 12-bit packed pixel formats,
 * then converts the output to 16 bits per pixel.
 *   - 10-bit format is packed 3 pixels per 4 bytes as {2'b0, p3[9:0], p2[9:0], p1[9:0]}
 *   - 12-bit format is packed 2 pixels per 3 bytes as {p2[11:0], p1[11:0]}
 */
class ADTFUnpackOp : public holoscan::Operator {
public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ADTFUnpackOp);
  ADTFUnpackOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;

  void compute(
      holoscan::InputContext& op_input,
      holoscan::OutputContext& op_output,
      holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<int> width_;
  holoscan::Parameter<int> height_;
  holoscan::Parameter<int> num_planes_;

  holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;

  int frame_size_;
  int pixel_size_;

  holoscan::Parameter<int> cuda_device_ordinal_;
  std::shared_ptr<holoscan::Tensor> depth_tensor_;
  std::shared_ptr<holoscan::Tensor> conf_tensor_;
  std::shared_ptr<holoscan::Tensor> ab_tensor_;

  std::shared_ptr<holoscan::Tensor> depth_rgb_;
  std::shared_ptr<holoscan::Tensor> conf_rgb_;
  std::shared_ptr<holoscan::Tensor> ab_rgb_;    

    holoscan::Parameter<std::string> in_tensor_name_;
    holoscan::Parameter<std::string> out_tensor_name_;

    CUcontext cuda_context_ = nullptr;
    CUdevice cuda_device_ = 0;
    bool is_integrated_ = false;
    bool host_memory_warning_ = false;

    holoscan::CudaStreamHandler cuda_stream_handler_;

    //std::shared_ptr<hololink::common::CudaFunctionLauncher> cuda_function_launcher_;

};

} // namespace hololink::operators
