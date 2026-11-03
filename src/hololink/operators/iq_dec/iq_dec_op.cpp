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
#include "iq_dec_op.hpp"

#include <algorithm>
#include <array>

#include <fmt/format.h>
#include <holoviz/imgui/ui.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include <hololink/core/logging_internal.hpp>
#include <hololink/operators/output_pressure.hpp>

#include "iq_dec_kernels.hpp"

namespace hololink::operators {

namespace {
    constexpr DLDataType kFloat32 { kDLFloat, 32, 1 };
    constexpr uint64_t kEncodedBytesPerSample = 2 * sizeof(int16_t);
    constexpr uint64_t kDecodedBytesPerSample = 2 * sizeof(float);
}

IQDecoderOp::IQDecoderOp(uint64_t max_samples, float scale, std::shared_ptr<hololink::ImGuiRenderer> renderer)
    : max_samples_(max_samples)
    , scale_(scale)
    , renderer_(std::move(renderer))
{
    if (!max_samples_)
        throw std::runtime_error("IQDecoderOp: max_samples must be positive");
}

void IQDecoderOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input(input, "input");
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::TensorOutputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
                .dtype = kFloat32,
                .rank = 1U,
            },
            .bounds = holoscan::tensor_bounds(max_samples_ * kDecodedBytesPerSample),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
}

holoscan::Contract IQDecoderOp::contract() const
{
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach { input });
    return contract;
}

void IQDecoderOp::start()
{
    if (renderer_)
        renderer_handle_ = renderer_->add_draw_function("IQ Decoder", [this] {
            float scale = scale_;
            if (ImGui::InputFloat("IQ decoding scale", &scale, 0.1f))
                scale_ = std::max(scale, 0.0f);
        });
}

void IQDecoderOp::stop()
{
    if (renderer_handle_) {
        renderer_->remove_draw_function(*renderer_handle_);
        renderer_handle_.reset();
    }
}

holoscan::expected<void, holoscan::Error> IQDecoderOp::compute(holoscan::ExecutionContext& context)
{
    auto maybe_sample = input.receive();
    if (!maybe_sample) {
        return holoscan::make_unexpected(std::move(maybe_sample).error());
    }
    const holoscan::Tensor& in_tensor = maybe_sample->data;

    // The input may carry any dtype tag (e.g. the receiver's raw bytes) but physically holds
    // int16 encoded IQ samples, so size it in bytes.
    const uint64_t in_bytes = static_cast<uint64_t>(in_tensor.nbytes());
    if (in_bytes % (8 * sizeof(int16_t))) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument,
            "Invalid encoded tensor: byte size must encode a multiple of 8 int16 elements" });
    }
    const uint64_t samples = in_bytes / kEncodedBytesPerSample;
    if (samples > max_samples_) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument,
            fmt::format("Encoded tensor holds {} IQ samples, more than max_samples={}", samples, max_samples_) });
    }

    const std::array<int64_t, 1> shape { static_cast<int64_t>(samples * 2) };
    auto loan = output.allocate_tensor(holoscan::TensorLoanRequest { .shape = shape, .dtype = kFloat32 });
    if (!loan) {
        if (!is_output_pressure(loan.error())) {
            return holoscan::make_unexpected(std::move(loan).error());
        }
        record_dropped_output("IQDecoderOp", dropped_, loan.error());
        return {};
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
    }

    // count is the number of int16 elements per component (2 components per signal)
    cuda_iq_decode(static_cast<float*>(writer->data()),
        static_cast<const int16_t*>(in_tensor.data()),
        static_cast<size_t>(samples),
        scale_,
        context.cuda_stream().get());

    auto committed = std::move(*writer).commit();
    if (!committed) {
        return holoscan::make_unexpected(std::move(committed).error());
    }
    auto emitted = output.emit(std::move(*loan),
        holoscan::EmitOptions {
            .capture_time = maybe_sample->metadata.capture_time(),
            .frame_id = maybe_sample->metadata.frame_id,
        });
    if (!emitted) {
        if (!is_output_pressure(emitted.error())) {
            return holoscan::make_unexpected(std::move(emitted).error());
        }
        record_dropped_output("IQDecoderOp", dropped_, emitted.error());
    }
    return {};
}

} // namespace hololink::operators
