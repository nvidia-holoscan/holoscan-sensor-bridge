/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "sig_gen_op.hpp"

#include <hololink/common/cuda_helper.hpp>
#include <hololink/operators/output_pressure.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "sig_gen_op_gui.hpp"

namespace hololink::operators {

namespace {
    constexpr DLDataType kFloat32 { kDLFloat, 32, 1 };
}

SignalGeneratorOp::SignalGeneratorOp(unsigned samples_count,
    Rational sampling_interval,
    std::string in_phase,
    std::string quadrature,
    std::string cuda_toolkit_include_path,
    std::shared_ptr<hololink::ImGuiRenderer> renderer,
    std::chrono::nanoseconds period,
    uint64_t count)
    : renderer_(std::move(renderer))
    , samples_count_(samples_count)
    , sampling_interval_(std::move(sampling_interval))
    , signal_expression_strs_ { std::move(in_phase), std::move(quadrature) }
    // If no CUDA toolkit path is provided, fall back to the compile-time default
    , cuda_toolkit_include_path_(cuda_toolkit_include_path.empty()
              ? std::string(CUDA_TOOLKIT_INCLUDE_PATHS)
              : std::move(cuda_toolkit_include_path))
    , period_(period)
    , count_(count)
{
    if (samples_count_ % 4)
        throw std::runtime_error("Samples Count must be a multiple of 4");
    if (period_.count() <= 0)
        throw std::runtime_error("The generation period must be positive");
}

SignalGeneratorOp::~SignalGeneratorOp() = default;

void SignalGeneratorOp::SignalExpression::update_expression(const std::string& str)
{
    if (str_ != str || !expr_) {
        str_ = str;
        expr_ = create_signal_expression(parser_, str_);
    }
}

void SignalGeneratorOp::SignalExpression::set_cuda_toolkit_include_path(const std::string& cuda_toolkit_include_path)
{
    parser_.set_cuda_toolkit_include_paths(cuda_toolkit_include_path);
}

std::string SignalGeneratorOp::SignalExpression::str() const
{
    return str_;
}

size_t SignalGeneratorOp::SignalExpression::evaluate(float* device_output, size_t count,
    size_t samples_index, const Rational& sampling_interval, size_t stride,
    SignalContext* device_context, cudaStream_t stream)
{
    return evaluate_expression(expr_, device_output, count, samples_index, sampling_interval, stride,
        device_context, stream);
}

void SignalGeneratorOp::setup(holoscan::OperatorSpec& spec)
{
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::TensorOutputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
                .dtype = kFloat32,
                .rank = 1U,
            },
            .bounds = holoscan::tensor_bounds(uint64_t(samples_count_) * number_of_signal_components * sizeof(float)),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
}

holoscan::Contract SignalGeneratorOp::contract() const
{
    holoscan::Contract contract;
    // Late ticks coalesce, so a generation slower than the period runs back to back.
    contract.trigger(holoscan::OnClock { .period = period_ });
    if (count_)
        contract.complete_after(holoscan::ComputeCalls { .limit = count_ });
    return contract;
}

void SignalGeneratorOp::start()
{
    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, 0));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    // CUDA context state is per thread, and the runtime may run compute() on any of its threads:
    // every CUDA-using callback pushes the context.
    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);

    device_context_ = create_signal_context();
    if (renderer_)
        gui_ = std::make_shared<GUI>(renderer_,
            "Signal Generator",
            signal_expression_strs_[0],
            signal_expression_strs_[1],
            samples_count_,
            sampling_interval_);

    for (int i = 0; i < number_of_signal_components; ++i)
        signal_expressions_[i].set_cuda_toolkit_include_path(cuda_toolkit_include_path_);
}

void SignalGeneratorOp::stop()
{
    {
        hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
        gui_.reset();
        // The compiled expressions hold CUDA modules: release them with the context current.
        signal_expressions_ = {};
        destroy_signal_context(device_context_);
        device_context_ = nullptr;
    }
    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

holoscan::expected<void, holoscan::Error> SignalGeneratorOp::compute(holoscan::ExecutionContext& context)
{
    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    const cudaStream_t stream = context.cuda_stream().get();

    // Update parameters from GUI
    if (gui_ && gui_->is_running()) {
        // Get latest values from the GUI
        auto lock = gui_->lock();
        for (int i = 0; i < number_of_signal_components; ++i) {
            auto signal_component_type = static_cast<SignalComponentType>(i);
            auto& gui_signal_expression = gui_->get_signal_component_expression(signal_component_type);
            signal_expression_strs_[i] = gui_signal_expression;
        }
        sampling_interval_ = gui_->get_sampling_interval();
    }

    // Update expressions if needed
    for (int i = 0; i < number_of_signal_components; ++i)
        signal_expressions_[i].update_expression(signal_expression_strs_[i]);

    // All the signal components are interleaved in one buffer
    const std::array<int64_t, 1> shape { int64_t(samples_count_) * number_of_signal_components };
    auto loan = output.allocate_tensor(holoscan::TensorLoanRequest { .shape = shape, .dtype = kFloat32 });
    if (!loan) {
        if (!is_output_pressure(loan.error())) {
            return holoscan::make_unexpected(std::move(loan).error());
        }
        record_dropped_output("SignalGeneratorOp", dropped_, loan.error());
        return {};
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
    }
    float* const samples = static_cast<float*>(writer->data());

    // Generate the samples; the signal components are interleaved, so a stride is used
    size_t next_index = index_;
    for (size_t offset = 0; offset < number_of_signal_components; ++offset) {
        next_index = signal_expressions_[offset].evaluate(
            samples + offset,
            samples_count_,
            index_,
            sampling_interval_,
            number_of_signal_components,
            device_context_,
            stream);
    }

    if (gui_ && gui_->is_running())
        gui_->stage_samples(samples, stream);

    auto committed = std::move(*writer).commit();
    if (!committed) {
        return holoscan::make_unexpected(std::move(committed).error());
    }
    auto emitted = output.emit(std::move(*loan),
        holoscan::EmitOptions {
            .capture_time = context.activation_time(),
            .frame_id = frame_id_,
        });
    if (!emitted) {
        if (!is_output_pressure(emitted.error())) {
            return holoscan::make_unexpected(std::move(emitted).error());
        }
        record_dropped_output("SignalGeneratorOp", dropped_, emitted.error());
        // Unpublished: regenerating these samples next time keeps the emitted signal continuous.
        // But a recipient may already have seen a partially published buffer, so move past that.
        if (!holoscan::publication_error_consumes_authority(emitted.error().code)) {
            return {};
        }
    }
    index_ = next_index;
    ++frame_id_;
    return {};
}

} // namespace hololink::operators
