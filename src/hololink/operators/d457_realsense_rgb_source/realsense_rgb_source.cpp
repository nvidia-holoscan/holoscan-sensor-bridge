// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realsense_rgb_source.hpp"

#include <holoscan/holoscan.hpp>
#include <hololink/logging.hpp>
#include <hololink/native/cuda_helper.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <gxf/std/tensor.hpp>
#include <cuda.h>
#include <holoscan/utils/cuda_stream_handler.hpp>

namespace holoscan::operators {

void D457RealSenseRGBSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");
  spec.output<holoscan::gxf::Entity>("output");
  spec.param(allocator_, "allocator", "Allocator", "Memory allocator");
}

void D457RealSenseRGBSourceOp::start() {
  CudaCheck(cuInit(0));
  CudaCheck(cuDeviceGet(&cuda_device_, 0));
  CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
  context_guard_ = std::make_unique<hololink::native::CudaContextScopedPush>(cuda_context_);
}

void D457RealSenseRGBSourceOp::stop() {
  CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
  cuda_context_ = nullptr;
}

void D457RealSenseRGBSourceOp::compute(InputContext& input, OutputContext& output, ExecutionContext& context) {
  auto maybe_entity = input.receive<gxf::Entity>("input");
  if (!maybe_entity) {
    throw std::runtime_error("Failed to receive input message");
  }

  auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());
  auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
  if (!maybe_tensor) {
    throw std::runtime_error("No tensor in input message");
  }

  const auto input_tensor = maybe_tensor.value();

  std::cout << "[D457Op] shape: ";
  for (int i = 0; i < input_tensor->rank(); ++i)
    std::cout << input_tensor->shape().dimension(i) << " ";
  std::cout << ", element type: " << static_cast<int>(input_tensor->element_type()) << std::endl;

  // Just emit it as-is
  output.emit(entity);
}


}  // namespace holoscan::operators
