// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realsense_depth_source.hpp"

#include <holoscan/holoscan.hpp>
#include <hololink/logging.hpp>
#include <hololink/native/cuda_helper.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <gxf/std/tensor.hpp>
#include <cuda.h>
#include <holoscan/utils/cuda_stream_handler.hpp>

namespace holoscan::operators {

namespace {
const char* source = R"(
extern "C" {

__global__ void colorize_depth_kernel(const unsigned short* depth, unsigned char* rgb, int size, float min_d, float max_d, float depth_unit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    unsigned short d = depth[idx];

    if (d == 0) {
        rgb[3 * idx + 0] = 0;
        rgb[3 * idx + 1] = 0;
        rgb[3 * idx + 2] = 0;
        return;
    }

    float norm = (d * depth_unit - min_d) / (max_d - min_d);
    norm = fminf(fmaxf(norm, 0.0f), 1.0f);

    float r = 255.0f * norm;
    float g = 255.0f * (1.0f - fabsf(norm - 0.5f) * 2);
    float b = 255.0f * (1.0f - norm);

    rgb[3 * idx + 0] = static_cast<unsigned char>(b);
    rgb[3 * idx + 1] = static_cast<unsigned char>(g);
    rgb[3 * idx + 2] = static_cast<unsigned char>(r);
}

})";
} // namespace

void D457RealSenseDepthSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");
  spec.output<holoscan::gxf::Entity>("output");
  spec.param(allocator_, "allocator", "Allocator", "Memory allocator");
}

void D457RealSenseDepthSourceOp::start() {
  colorizer_ = std::make_unique<rs2::utils::Colorizer>();

  CudaCheck(cuInit(0));
  CudaCheck(cuDeviceGet(&cuda_device_, 0));
  CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
  context_guard_ = std::make_unique<hololink::native::CudaContextScopedPush>(cuda_context_);
  cuda_function_launcher_ = std::make_unique<hololink::native::CudaFunctionLauncher>(
      source, std::vector<std::string>{"colorize_depth_kernel"});
}

void D457RealSenseDepthSourceOp::stop() {
  cuda_function_launcher_.reset();
  CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
  cuda_context_ = nullptr;
}

void D457RealSenseDepthSourceOp::compute(InputContext& input, OutputContext& output, ExecutionContext& context) {
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
  if (input_tensor->rank() != 3 || input_tensor->shape().dimension(2) != 1) {
    throw std::runtime_error("Expected input tensor shape (H, W, 1)");
  }

  width_ = input_tensor->shape().dimension(1);
  height_ = input_tensor->shape().dimension(0);

  // Copy depth to host
  std::vector<uint16_t> host_depth(width_ * height_);
  cudaMemcpy(host_depth.data(), input_tensor->pointer(), host_depth.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);

  // Colorize
  std::vector<uint8_t> rgb_data(width_ * height_ * 3);
  colorizer_->set_equalize(true);
  colorizer_->set_depth_units(0.001f); // mm to meters

  colorizer_->colorize(
      host_depth.data(),
      rgb_data.data(),
      width_,
      height_
  );

  // Allocate GPU output tensor
  const size_t num_elements = width_ * height_ * 3;
  nvidia::gxf::Shape shape{static_cast<int32_t>(height_), static_cast<int32_t>(width_), 3};
  const auto type = nvidia::gxf::PrimitiveType::kUnsigned8;
  const auto stride = nvidia::gxf::ComputeTrivialStrides(shape, nvidia::gxf::PrimitiveTypeSize(type));

  auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      fragment()->executor().context(), allocator_->gxf_cid());
  if (!allocator_handle) {
    throw std::runtime_error("Failed to get GXF allocator handle");
  }

  auto maybe_entity_out = nvidia::gxf::CreateTensorMap(context.context(), allocator_handle.value(),
    {{"output", nvidia::gxf::MemoryStorageType::kDevice, shape, type, 0, stride}}, false);
  if (!maybe_entity_out) {
    throw std::runtime_error("Failed to create output entity");
  }

  auto entity_out = std::move(maybe_entity_out.value());
  auto maybe_tensor_out = entity_out.get<nvidia::gxf::Tensor>("output");
  if (!maybe_tensor_out) {
    throw std::runtime_error("Failed to retrieve output tensor");
  }

  auto tensor = maybe_tensor_out.value();

  // Copy colorized RGB to GPU
  cudaMemcpy(tensor->pointer(), rgb_data.data(), num_elements, cudaMemcpyHostToDevice);

  output.emit(entity_out);
}


}  // namespace holoscan::operators
