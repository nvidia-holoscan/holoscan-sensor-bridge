// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realsense_dual_source.hpp"

#include <librealsense2/rs.hpp>
#include <cuda_runtime.h>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/fragment.hpp>
#include <gxf/std/tensor.hpp>
#include <hololink/native/cuda_helper.hpp>
#include <iostream>

namespace holoscan::operators {

void D455RealSenseDualSourceOp::setup(OperatorSpec& spec) {
  spec.output<gxf::Entity>("rgb_output");
  spec.output<gxf::Entity>("depth_output");
  spec.param(allocator_, "allocator", "Allocator", "Memory allocator");
}

void D455RealSenseDualSourceOp::start() {
  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_COLOR, 1920, 1080, RS2_FORMAT_RGB8, 30);
  cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
  pipe_.start(cfg);

  // colorizer_ = std::make_unique<rs2::colorizer>();
  colorizer_ = std::make_unique<rs2::utils::Colorizer>();

  CudaCheck(cuInit(0));
  CudaCheck(cuDeviceGet(&cuda_device_, 0));
  CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
  context_guard_ = std::make_unique<hololink::native::CudaContextScopedPush>(cuda_context_);
}

void D455RealSenseDualSourceOp::stop() {
  pipe_.stop();
  CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
  cuda_context_ = nullptr;
}

nvidia::gxf::Entity create_tensor_entity(const rs2::video_frame& frame,
        holoscan::Fragment* fragment,
        const std::shared_ptr<holoscan::Allocator>& allocator,
        ExecutionContext& context,
        const std::string& tensor_name) {
    const int width = frame.get_width();
    const int height = frame.get_height();
    const size_t num_elements = width * height * 3;

    nvidia::gxf::Shape shape{height, width, 3};
    auto type = nvidia::gxf::PrimitiveType::kUnsigned8;
    auto stride = nvidia::gxf::ComputeTrivialStrides(shape, nvidia::gxf::PrimitiveTypeSize(type));

    auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
    fragment->executor().context(), allocator->gxf_cid());
    if (!allocator_handle) {
    throw std::runtime_error("Failed to get GXF allocator handle");
    }

    auto maybe_entity = nvidia::gxf::CreateTensorMap(context.context(), allocator_handle.value(),
    {{tensor_name.c_str(), nvidia::gxf::MemoryStorageType::kDevice, shape, type, 0, stride}}, false);
    if (!maybe_entity) {
    throw std::runtime_error("Failed to create tensor entity");
    }

    auto entity = std::move(maybe_entity.value());
    auto maybe_tensor = entity.get<nvidia::gxf::Tensor>(tensor_name.c_str());
    if (!maybe_tensor) {
    throw std::runtime_error("Failed to retrieve tensor");
    }

    cudaMemcpy(maybe_tensor.value()->pointer(), frame.get_data(), num_elements, cudaMemcpyHostToDevice);
    return entity;
}


nvidia::gxf::Entity create_tensor_entity(const std::vector<uint8_t>& rgb_data,
    holoscan::Fragment* fragment,
    const std::shared_ptr<holoscan::Allocator>& allocator,
    ExecutionContext& context,
    const std::string& tensor_name,
    int& width,
    int& height) {
    const size_t num_elements = width * height * 3;

    nvidia::gxf::Shape shape{height, width, 3};
    auto type = nvidia::gxf::PrimitiveType::kUnsigned8;
    auto stride = nvidia::gxf::ComputeTrivialStrides(shape, nvidia::gxf::PrimitiveTypeSize(type));

    auto allocator_handle = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
    fragment->executor().context(), allocator->gxf_cid());
    if (!allocator_handle) {
    throw std::runtime_error("Failed to get GXF allocator handle");
    }

    auto maybe_entity = nvidia::gxf::CreateTensorMap(context.context(), allocator_handle.value(),
    {{tensor_name.c_str(), nvidia::gxf::MemoryStorageType::kDevice, shape, type, 0, stride}}, false);
    if (!maybe_entity) {
    throw std::runtime_error("Failed to create tensor entity");
    }

    auto entity = std::move(maybe_entity.value());
    auto maybe_tensor = entity.get<nvidia::gxf::Tensor>(tensor_name.c_str());
    if (!maybe_tensor) {
    throw std::runtime_error("Failed to retrieve tensor");
    }

    cudaMemcpy(maybe_tensor.value()->pointer(), rgb_data.data(), num_elements, cudaMemcpyHostToDevice);
    return entity;
}



void D455RealSenseDualSourceOp::compute(InputContext&, OutputContext& output, ExecutionContext& context) {
    rs2::frameset frames = pipe_.wait_for_frames();
    rs2::video_frame color_frame = frames.get_color_frame();
    rs2::depth_frame depth_frame = frames.get_depth_frame();

    int width = depth_frame.get_width();
    int height = depth_frame.get_height();
    const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(depth_frame.get_data());

    std::vector<uint8_t> rgb_data(width * height * 3);
    colorizer_->set_equalize(true);
    colorizer_->set_depth_units(0.001f); // for Z16 mm to m
    
    colorizer_->colorize(
        depth_data,         // const uint16_t*
        rgb_data.data(),    // uint8_t*
        width,
        height
    );

    // rs2::video_frame depth_colorized_frame = colorizer_->colorize();
  
    auto rgb_entity = create_tensor_entity(color_frame, fragment(), allocator_.get(), context, "rgb_output");
    auto depth_entity = create_tensor_entity(rgb_data, fragment(), allocator_.get(), context, "depth_output", width, height);    
  
    output.emit(rgb_entity, "rgb_output");
    output.emit(depth_entity, "depth_output");
  }  

}  // namespace holoscan::operators
