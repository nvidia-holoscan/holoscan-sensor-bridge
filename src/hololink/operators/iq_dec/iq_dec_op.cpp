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
#include "iq_dec_op.hpp"

#include <imgui.h>

#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {

static const char* input_name = "input";
static const char* output_name = "output";

void IQDecoderOp::initialize()
{
    auto frag = fragment();

    // Find if there is an argument for 'allocator'
    auto has_allocator = std::find_if(
        args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "allocator"); });
    // Create the allocator if there is no argument provided.
    if (has_allocator == args().end()) {
        allocator_ = frag->make_resource<holoscan::UnboundedAllocator>("allocator");
        add_arg(allocator_.get());
    }
    Operator::initialize();
}

void IQDecoderOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<std::any>(input_name);
    spec.output<Tensor>(output_name);

    // Register converters for arguments not defined by Holoscan
    register_converter<ImGuiRenderer*>();

    spec.param(scale_,
        "scale",
        "scale",
        "Scale",
        1.0f);

    spec.param(renderer_, "renderer", "ImGuiRenderer", "Pointer to ImGuiRenderer object");
}

void IQDecoderOp::start()
{
    if (renderer_.has_value() && renderer_.get())
        renderer_handle_ = renderer_->add_draw_function(name(), [this] {
            float scale = scale_;
            if (ImGui::InputFloat("IQ decoding scale", &scale, 0.1f))
                if (scale < 0)
                    scale = 0;

            std::unique_lock<std::mutex> lock(mutex_);
            scale_ = scale;
        });
}

void IQDecoderOp::stop()
{
    if (renderer_.has_value() && renderer_.get())
        renderer_->remove_draw_function(renderer_handle_);
}

void IQDecoderOp::compute(holoscan::InputContext& op_input, [[maybe_unused]] holoscan::OutputContext& op_output,
    [[maybe_unused]] holoscan::ExecutionContext& context)
{
    Tensor encoded_tensor;
    auto maybe_any = op_input.receive<std::any>(input_name);
    if (auto tensor_pointer = std::any_cast<Tensor>(&maybe_any.value())) {
        encoded_tensor = *tensor_pointer;
    } else if (auto entity_pointer = std::any_cast<holoscan::gxf::Entity>(&maybe_any.value())) {
        encoded_tensor = entity_pointer->get<holoscan::Tensor>();
    } else {
        HSB_LOG_ERROR("Failed to receive message from port '{}'", input_name);
        throw std::exception();
    }

    HSB_LOG_DEBUG("Received {} bytes", encoded_tensor->nbytes());
    auto number_of_iq_elements = encoded_tensor->nbytes() / sizeof(uint16_t); // Each in-phase/quadrature sample is encoded into 16 bit (each sample is 32 bit)
    if (number_of_iq_elements % 8 != 0) {
        throw std::runtime_error("Invalid encoded tensor size");
    }

    int tensor_size = static_cast<int>(number_of_iq_elements);
    // Try to acquire a buffer from the pool
    auto unique_dl_managed_tensor_context = dl_managed_tensor_context_pool_.acquire();
    // The size of the required buffer changed, release the buffer manually
    if (unique_dl_managed_tensor_context && unique_dl_managed_tensor_context->dl_shape[0] != tensor_size) {
        auto ptr = unique_dl_managed_tensor_context.release();
        decltype(dl_managed_tensor_context_pool_)::Deleter {}(ptr);
    }
    std::shared_ptr<holoscan::DLManagedTensorContext> dl_managed_tensor_context(std::move(unique_dl_managed_tensor_context));
    // If failed to acquire the buffer from the pool, allocate a new one and store it in the pool
    if (!dl_managed_tensor_context) {
        nvidia::gxf::Tensor gxf_tensor;

        auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
            allocator_->gxf_cid());
        if (!gxf_tensor.reshape<float>(
                nvidia::gxf::Shape { tensor_size },
                nvidia::gxf::MemoryStorageType::kDevice,
                allocator.value()))
            throw std::runtime_error("Failed to allocate cuda memory");

        auto maybe_dl_ctx = gxf_tensor.toDLManagedTensorContext();
        if (!maybe_dl_ctx)
            HSB_LOG_ERROR(
                "Failed to get std:shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
        dl_managed_tensor_context = dl_managed_tensor_context_pool_.emplace(std::move(*maybe_dl_ctx.value()));
    }
    auto tensor_iq = std::make_shared<holoscan::Tensor>(dl_managed_tensor_context);

    try {
        // Decode values
        std::unique_lock<std::mutex> lock(mutex_);
        float scale = scale_;
        lock.unlock();
        cuda_iq_decode(reinterpret_cast<float*>(tensor_iq->data()),
            reinterpret_cast<const int16_t*>(encoded_tensor->data()),
            number_of_iq_elements / 2, scale); // To components (IQ) per signal
    } catch (const std::exception& e) {
        HSB_LOG_ERROR("Failed to decode data: {}", e.what());
    }

    // Emit the samples
    op_output.emit(tensor_iq);
}

} // namespace hololink::operators
