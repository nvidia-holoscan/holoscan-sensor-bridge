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
#include "sig_gen_op.hpp"

#include <hololink/common/cuda_helper.hpp>

#include "sig_gen_op_gui.hpp"

namespace hololink::operators {

void SignalGeneratorOp::SignalExpression::update_expression(const std::string& str)
{
    if (str_ != str || !expr_) {
        str_ = str;
        expr_ = create_signal_expression(parser_, str_);
    }
}

void SignalGeneratorOp::SignalExpression::set_cuda_toolkit_include_path(const std::string& cuda_toolkit_include_path)
{
    parser_.set_cuda_toolkit_include_path(cuda_toolkit_include_path);
}

std::string SignalGeneratorOp::SignalExpression::str() const
{
    return str_;
}

size_t SignalGeneratorOp::SignalExpression::evaluate(float* device_output, size_t count,
    size_t samples_index, const Rational& sampling_interval, size_t stride)
{
    return evaluate_expression(expr_, device_output, count, samples_index, sampling_interval, stride);
}

void SignalGeneratorOp::start()
{
    CudaCheck(cuInit(0));
    CUdevice device;
    CudaCheck(cuDeviceGet(&device, 0));
    CUcontext context;
    CudaCheck(cuDevicePrimaryCtxRetain(&context, device));
    CudaCheck(cuCtxSetCurrent(context));

    if (renderer_.has_value() && renderer_.get())
        gui_ = std::make_unique<GUI>(renderer_.get(),
            name(),

            signal_expression_strs_[0].get(),
            signal_expression_strs_[1].get(),
            samples_count_.get(),
            sampling_interval_.get());

    for (int signal_component_index = 0; signal_component_index < number_of_signal_components; ++signal_component_index)
        signal_expressions_[signal_component_index].set_cuda_toolkit_include_path(cuda_toolkit_include_path_.get());
}

void SignalGeneratorOp::stop()
{
    gui_.reset();
    CUdevice device;
    CudaCheck(cuDeviceGet(&device, 0));
    CudaCheck(cuDevicePrimaryCtxRelease(device));
}

void SignalGeneratorOp::initialize()
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

    if (samples_count_.get() % 4)
        throw std::runtime_error("Samples Count must be a multiple of 4");
}

void SignalGeneratorOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output<Tensor>("output");

    /// Register converters for arguments not defined by Holoscan
    register_converter<hololink::ImGuiRenderer*>();
    register_converter<Rational>();

    // Signal Expression Parameters
    spec.param(allocator_, "allocator", "allocator", "Allocator used to allocate tensor output");
    spec.param(renderer_, "renderer", "ImGuiRenderer", "Pointer to ImGuiRenderer object");
    spec.param(samples_count_,
        "samples_count",
        "Samples Count",
        "The number of samples to generate",
        0u);
    spec.param(sampling_interval_,
        "sampling_interval",
        "Sampling Interval",
        "The interval between sequential samples",
        Rational(1, 128));
    spec.param(signal_expression_strs_[0],
        "in_phase",
        "in_phase",
        "In Phase",
        std::string("0"));
    spec.param(signal_expression_strs_[1],
        "quadrature",
        "quadrature",
        "Quadrature",
        std::string("0"));
    spec.param(cuda_toolkit_include_path_,
        "cuda_toolkit_include_path",
        "cuda_toolkit_include_path",
        "cuda toolkit include path",
        std::string(CUDA_TOOLKIT_INCLUDE_PATH));
}

void SignalGeneratorOp::compute([[maybe_unused]] holoscan::InputContext& /*op_input*/, holoscan::OutputContext& op_output,
    holoscan::ExecutionContext& context)
{
    static SignalGeneratorOp* self = this;
//#define SIG_GEN_OP_MEASURE_TIME
#ifdef SIG_GEN_OP_MEASURE_TIME
    if (self == this) {
        static std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        static size_t counter = 0;
        static std::chrono::microseconds total_duration(0);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_duration += duration;
        start = end;
        ++counter;
        HSB_LOG_INFO("Cycle {} us, {} us (avg)", duration.count(), total_duration.count() / counter);
    }
#endif
    // Update parameters from GUI
    if (gui_ && gui_->is_running()) {
        // Get latest values from the GUI
        auto lock = gui_->lock();
        for (int signal_component_index = 0; signal_component_index < number_of_signal_components; ++signal_component_index) {
            auto signal_component_type = static_cast<SignalComponentType>(signal_component_index);
            auto& gui_signal_expression = gui_->get_signal_component_expression(signal_component_type);
            signal_expression_strs_[signal_component_index].get() = gui_signal_expression;
        }
        sampling_interval_.get() = gui_->get_sampling_interval();
    }

    // Update expressions if needed
    for (int signal_component_index = 0; signal_component_index < number_of_signal_components; ++signal_component_index)
        signal_expressions_[signal_component_index].update_expression(signal_expression_strs_[signal_component_index].get());

    // Create a CUDA buffer
    // All the signal components are interleaved in one buffer
    int tensor_size = static_cast<int>(samples_count_.get() * number_of_signal_components);
    auto unique_dl_managed_tensor_context = dl_managed_tensor_context_pool_.acquire();
    if (unique_dl_managed_tensor_context && unique_dl_managed_tensor_context->dl_shape[0] != tensor_size) { // The size of the required buffer changed
        auto ptr = unique_dl_managed_tensor_context.release();
        decltype(dl_managed_tensor_context_pool_)::Deleter {}(ptr);
    }
    std::shared_ptr<holoscan::DLManagedTensorContext> dl_managed_tensor_context(std::move(unique_dl_managed_tensor_context));
    if (!dl_managed_tensor_context) {
        nvidia::gxf::Tensor gxf_tensor;
        auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
            allocator_->gxf_cid());
        if (!gxf_tensor.reshape<float>(nvidia::gxf::Shape { tensor_size },
                nvidia::gxf::MemoryStorageType::kDevice,
                allocator.value()))
            throw std::runtime_error("Failed to allocate cuda memory");
        auto maybe_dl_ctx = gxf_tensor.toDLManagedTensorContext();
        if (!maybe_dl_ctx)
            HSB_LOG_ERROR("Failed to get std:shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
        dl_managed_tensor_context = dl_managed_tensor_context_pool_.emplace(std::move(*maybe_dl_ctx.value()));
    }
    auto tensor = std::make_shared<holoscan::Tensor>(dl_managed_tensor_context);

    // Generate the samples
    auto sampling_index = index_;
    for (size_t offset = 0; offset < number_of_signal_components; ++offset) {
        // The signal samples are interleaved, a stride is used
        index_ = signal_expressions_[offset].evaluate(
            reinterpret_cast<float*>(tensor->data()) + offset,
            samples_count_.get(),
            sampling_index,
            sampling_interval_.get(),
            number_of_signal_components); // stride
    }

    if (gui_ && gui_->is_running()) {
        gui_->push_samples(tensor);
    }

    // Emit the samples
    op_output.emit(tensor);
}

} // namespace hololink::operators
