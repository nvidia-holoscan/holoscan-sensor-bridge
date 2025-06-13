// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realsense_rgb_source.hpp"

#include <librealsense2/rs.hpp>
#include <cuda_runtime.h>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/fragment.hpp>
#include <gxf/std/tensor.hpp>
#include <hololink/native/cuda_helper.hpp>
#include <iostream>

namespace holoscan::operators {

void D455RealSenseRGBSourceOp::setup(OperatorSpec& spec) {
  spec.output<gxf::Entity>("output");
  spec.param(allocator_, "allocator", "Allocator", "Memory allocator");
}

void D455RealSenseRGBSourceOp::start() {
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, 30);  // Full HD
  pipe_.start(cfg);

  CudaCheck(cuInit(0));
  CudaCheck(cuDeviceGet(&cuda_device_, 0));
  CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));

  context_guard_ = std::make_unique<hololink::native::CudaContextScopedPush>(cuda_context_);
}

void D455RealSenseRGBSourceOp::stop() {
  pipe_.stop();
  CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
  cuda_context_ = nullptr;
}

void D455RealSenseRGBSourceOp::compute(InputContext&, OutputContext& output, ExecutionContext& context) {
  rs2::frameset frames = pipe_.wait_for_frames();
  rs2::video_frame color_frame = frames.get_color_frame();

  const int width = color_frame.get_width();
  const int height = color_frame.get_height();
  const size_t num_elements = width * height * 3;

  nvidia::gxf::Shape shape{height, width, 3};
  const auto type = nvidia::gxf::PrimitiveType::kUnsigned8;
  const auto stride = nvidia::gxf::ComputeTrivialStrides(shape, nvidia::gxf::PrimitiveTypeSize(type));

  auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      fragment()->executor().context(), allocator_->gxf_cid());
  if (!allocator_handle) {
    throw std::runtime_error("Failed to get GXF allocator handle");
  }

  auto maybe_entity = nvidia::gxf::CreateTensorMap(context.context(), allocator_handle.value(),
    {{"output", nvidia::gxf::MemoryStorageType::kDevice, shape, type, 0, stride}}, false);
  if (!maybe_entity) {
    throw std::runtime_error("Failed to create output entity");
  }

  auto entity = std::move(maybe_entity.value());
  auto maybe_tensor = entity.get<nvidia::gxf::Tensor>("output");
  if (!maybe_tensor) {
    throw std::runtime_error("Failed to retrieve output tensor");
  }

  auto tensor = maybe_tensor.value();

  // Copy to GPU
  cudaMemcpy(tensor->pointer(), color_frame.get_data(), num_elements, cudaMemcpyHostToDevice);

  output.emit(entity);
}

}  // namespace holoscan::operators
