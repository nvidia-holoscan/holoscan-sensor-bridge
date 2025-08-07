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
#include "iq_enc_op.hpp"

#include <imgui.h>

#include <hololink/core/logging_internal.hpp>

namespace hololink::operators {

static const char* input_name = "input";
static const char* output_name = "output";

void IQEncoderOp::initialize()
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

void IQEncoderOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input<Tensor>(input_name);
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

void IQEncoderOp::start()
{
    if (renderer_.has_value() && renderer_.get())
        renderer_handle_ = renderer_->add_draw_function(name(), [this] {
            float scale = scale_;
            if (ImGui::InputFloat("IQ encoding scale", &scale, 0.1f))
                if (scale < 0)
                    scale = 0;

            std::unique_lock<std::mutex> lock(mutex_);
            scale_ = scale;
        });
}

void IQEncoderOp::stop()
{
    if (renderer_.has_value() && renderer_.get())
        renderer_->remove_draw_function(renderer_handle_);
}

void IQEncoderOp::compute(holoscan::InputContext& op_input, [[maybe_unused]] holoscan::OutputContext& op_output,
    [[maybe_unused]] holoscan::ExecutionContext& context)
{
    auto maybe_tensor = op_input.receive<Tensor>(input_name);
    if (!maybe_tensor) {
        HSB_LOG_ERROR("Failed to receive message from port '{}'", input_name);
        throw std::exception();
    }

    auto tensor_iq = maybe_tensor.value();
    if (tensor_iq->size() % 8) {
        throw std::runtime_error("Invalid IQ signal");
    }

    // Create CUDA buffer
    int tensor_size = static_cast<int>(tensor_iq->size());
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
        if (!gxf_tensor.reshape<int16_t>(nvidia::gxf::Shape { tensor_size },
                nvidia::gxf::MemoryStorageType::kDevice,
                allocator.value()))
            throw std::runtime_error("Failed to allocate cuda memory");

        auto maybe_dl_ctx = gxf_tensor.toDLManagedTensorContext();
        if (!maybe_dl_ctx)
            HSB_LOG_ERROR(
                "Failed to get std:shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
        dl_managed_tensor_context = dl_managed_tensor_context_pool_.emplace(std::move(*maybe_dl_ctx.value()));
    }
    auto tensor = std::make_shared<holoscan::Tensor>(dl_managed_tensor_context);
    HSB_LOG_DEBUG("Allocated ({})", (void*)tensor->data());

    try {
        // Encode values
        std::unique_lock<std::mutex> lock(mutex_);
        float scale = scale_;
        lock.unlock();
        cuda_iq_encode(reinterpret_cast<int16_t*>(tensor->data()),
            reinterpret_cast<const float*>(tensor_iq->data()),
            tensor_iq->size() / 2, scale); // To components (IQ) per signal
    } catch (const std::exception& e) {
        HSB_LOG_ERROR("Failed to encode data: {}", e.what());
    }

    // Emit the encoded message
    op_output.emit(tensor);
}

} // namespace hololink::operators
