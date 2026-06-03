/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "sub_frame_combiner.hpp"

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/hololink.hpp>

#include <cuda.h>

namespace hololink::operators {

void SubFrameCombinerOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(allocator_, "allocator", "Allocator",
        "Allocator used to allocate the output image");
    spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal",
        "Device to use for CUDA operations", 0);
    spec.param(out_tensor_name_, "out_tensor_name", "OutputTensorName",
        "Name of the output tensor", std::string(""));
    spec.param(height_, "height", "Height",
        "Height of the output image", 0U);
}

void SubFrameCombinerOp::start()
{
    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, cuda_device_ordinal_.get()));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
}

void SubFrameCombinerOp::stop()
{
    output_entity_ = nvidia::gxf::Entity();

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

void SubFrameCombinerOp::emit_frame(holoscan::OutputContext& op_output)
{
    auto result = holoscan::gxf::Entity(std::move(output_entity_));
    op_output.emit(result, "output");

    expected_frame_number_++;
    received_rows_ = 0;
}

void SubFrameCombinerOp::compute(holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input");
    }
    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    const auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
    if (!maybe_tensor) {
        throw std::runtime_error("Tensor not found in message");
    }

    const auto input_tensor = maybe_tensor.value();
    if (input_tensor->rank() != 3) {
        throw std::runtime_error("Tensor must be two dimensional");
    }
    const auto input_shape = input_tensor->shape();

    auto frame_number = metadata()->get<int64_t>("frame_number");
    if (frame_number != expected_frame_number_) {
        // if the frame number has changed, emit the current frame and start a new one
        if (!output_entity_.is_null()) {
            HOLOSCAN_LOG_WARN(
                "Partial frame detected, sub-frame(s) dropped (expected frame {}, new frame {})",
                expected_frame_number_, frame_number);
            emit_frame(op_output);
        }
        expected_frame_number_ = frame_number;
    }

    // allocate the output entity if it is not already allocated
    if (output_entity_.is_null()) {
        // get handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
        auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
            fragment()->executor().context(), allocator_->gxf_cid());

        // create the output
        nvidia::gxf::Shape shape = { int(height_.get()), int(input_shape.dimension(1)), int(input_shape.dimension(2)) };
        const uint64_t bytes_per_element = nvidia::gxf::PrimitiveTypeSize(input_tensor->element_type());
        auto entity
            = CreateTensorMap(context.context(), allocator.value(),
                { { out_tensor_name_.get(), nvidia::gxf::MemoryStorageType::kDevice, shape,
                    input_tensor->element_type(), bytes_per_element,
                    nvidia::gxf::ComputeTrivialStrides(shape,
                        bytes_per_element) } },
                false);

        if (!entity) {
            throw std::runtime_error("failed to create out_message");
        }
        output_entity_ = entity.value();
    }

    const auto tensor = output_entity_.get<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());
    if (!tensor) {
        throw std::runtime_error(
            fmt::format("failed to get out_tensor with name \"{}\"", out_tensor_name_.get()));
    }

    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    auto cuda_stream = op_input.receive_cuda_stream();

    const auto sub_frame_offset = metadata()->get<int64_t>("sub_frame_offset");

    const uint64_t input_stride = input_tensor->stride(0);
    const uint64_t output_stride = tensor->get()->stride(0);
    if (input_stride != output_stride) {
        throw std::runtime_error("Input and output strides must be the same");
    }

    CudaCheck(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(tensor->get()->pointer() + sub_frame_offset * output_stride),
        reinterpret_cast<CUdeviceptr>(input_tensor->pointer()), input_tensor->size(), cuda_stream));

    received_rows_ += input_shape.dimension(0);
    if (received_rows_ > height_.get()) {
        throw std::runtime_error("Received more rows than the height of the output image");
    }
    // Emit the frame if we have received all the rows
    if (received_rows_ == height_.get()) {
        emit_frame(op_output);
    }
}

} // namespace hololink::operators
