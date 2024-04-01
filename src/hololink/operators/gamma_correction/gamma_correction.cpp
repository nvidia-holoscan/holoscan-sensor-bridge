/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gamma_correction.hpp"

#include <hololink/native/cuda_helper.hpp>
#include <holoscan/holoscan.hpp>

namespace {

const char* source = R"(
extern "C" {

/**
 * Apply gamma correction.
 *
 * @param in [in] pointer to image
 * @param components [in] components per pixel
 * @param width [in] width of the image
 * @param height [in] height of the image
 */
__global__ void applyGammaCorrection(unsigned short *image,
                                     int components,
                                     int width,
                                     int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx_x >= width) || (idx_y >= height))
        return;

    const int index = (idx_y * width + idx_x) * components;
    const float range = (1 << (sizeof(unsigned short) * 8)) - 1;

    // apply gamma correction to each component except alpha
    for (int component = 0; component < min(components, 3); ++component) {
        float value = (float)(image[index + component]);
        value = powf(value / range, 1.f / GAMMA) * range;
        image[index + component] = (unsigned short)(value + 0.5f);
    }
}

})";

} // anonymous namespace

namespace hololink::operators {

void GammaCorrectionOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");

    spec.param(gamma_, "gamma", "Gamma", "Gamma correction value", 2.2f);
    spec.param(
        cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal", "Device to use for CUDA operations", 0);
    cuda_stream_handler_.define_params(spec);
}

void GammaCorrectionOp::start()
{
    CudaCheck(cuInit(0));
    CUdevice device;
    CudaCheck(cuDeviceGet(&cuda_device_, cuda_device_ordinal_.get()));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    int integrated = 0;
    CudaCheck(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, cuda_device_));
    is_integrated_ = (integrated != 0);

    hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset(new hololink::native::CudaFunctionLauncher(
        source, { "applyGammaCorrection" }, { fmt::format("-D GAMMA={}", gamma_.get()) }));
}

void GammaCorrectionOp::stop()
{
    hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);

    cuda_function_launcher_.reset();

    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

void GammaCorrectionOp::compute(holoscan::InputContext& input, holoscan::OutputContext& output, holoscan::ExecutionContext& context)
{
    auto maybe_entity = input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
        throw std::runtime_error("Failed to receive input");
    }

    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    // get the CUDA stream from the input message
    gxf_result_t stream_handler_result = cuda_stream_handler_.from_message(context.context(), entity);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error("Failed to get the CUDA stream from incoming messages");
    }

    const auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
    if (!maybe_tensor) {
        throw std::runtime_error("Tensor not found in message");
    }

    const auto input_tensor = maybe_tensor.value();

    if (input_tensor->storage_type() == nvidia::gxf::MemoryStorageType::kHost) {
        if (!is_integrated_ && !host_memory_warning_) {
            host_memory_warning_ = true;
            HOLOSCAN_LOG_WARN(
                "The input tensor is stored in host memory, this will reduce performance of this "
                "operator. For best performance store the input tensor in device memory.");
        }
    } else if (input_tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
        throw std::runtime_error(
            fmt::format("Unsupported storage type {}", (int)input_tensor->storage_type()));
    }

    if (input_tensor->rank() != 3) {
        throw std::runtime_error("Tensor must be an image");
    }
    if (input_tensor->element_type() != nvidia::gxf::PrimitiveType::kUnsigned16) {
        throw std::runtime_error(fmt::format("Unexpected image data type '{}', expected '{}'", int(input_tensor->element_type()), int(nvidia::gxf::PrimitiveType::kUnsigned16)));
    }

    const uint32_t height = input_tensor->shape().dimension(0);
    const uint32_t width = input_tensor->shape().dimension(1);
    const uint32_t components = input_tensor->shape().dimension(2);

    hololink::native::CudaContextScopedPush cur_cuda_context(cuda_context_);
    const cudaStream_t cuda_stream = cuda_stream_handler_.get_cuda_stream(context.context());

    if (gamma_ != 1.f) {
        cuda_function_launcher_->launch(
            "applyGammaCorrection",
            { width, height, 1 },
            cuda_stream,
            input_tensor->pointer(), components, width, height);
    }

    // pass the CUDA stream to the output message
    auto out_message = nvidia::gxf::Expected<nvidia::gxf::Entity>(entity);
    stream_handler_result
        = cuda_stream_handler_.to_message(out_message);
    if (stream_handler_result != GXF_SUCCESS) {
        throw std::runtime_error("Failed to add the CUDA stream to the outgoing messages");
    }

    // Emit the tensor
    output.emit(entity);
}

} // namespace hololink::operators
