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
#include "iq_enc_op.hpp"

#include <array>
#include <stdexcept>

#include <holoviz/imgui/ui.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include <hololink/core/logging_internal.hpp>
#include <hololink/operators/output_pressure.hpp>

#include "iq_enc_kernels.hpp"

namespace hololink::operators {

namespace {
    constexpr DLDataType kFloat32 { kDLFloat, 32, 1 };
    constexpr DLDataType kInt16 { kDLInt, 16, 1 };
}

IQEncoderOp::IQEncoderOp(float scale, std::shared_ptr<hololink::ImGuiRenderer> renderer)
    : scale_(scale)
    , renderer_(std::move(renderer))
{
    // The encoder divides by the scale.
    if (!(scale > 0.0f)) {
        throw std::invalid_argument("IQEncoderOp: scale must be positive");
    }
}

void IQEncoderOp::setup(holoscan::OperatorSpec& spec)
{
    spec.input(input, "input")
        .expects_tensor(holoscan::TensorInputSpec {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
                .dtype = kFloat32,
            },
        });
    // float32 in, int16 out: each output slot needs half its producer's bytes.
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(
            holoscan::TensorOutputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = kInt16,
                    .rank = 1U,
                },
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            },
            holoscan::TensorOutputDerivations {
                .max_byte_span = holoscan::input_max_byte_span(input).ceiling_divided_by(holoscan::tensor_byte_span(2U)),
            });
}

holoscan::Contract IQEncoderOp::contract() const
{
    holoscan::Contract contract;
    contract.trigger(holoscan::OnEach { input });
    return contract;
}

void IQEncoderOp::start()
{
    if (renderer_)
        renderer_handle_ = renderer_->add_draw_function("IQ Encoder", [this] {
            float scale = scale_;
            if (ImGui::InputFloat("IQ encoding scale", &scale, 0.1f) && (scale > 0.0f))
                scale_ = scale;
        });
}

void IQEncoderOp::stop()
{
    if (renderer_handle_) {
        renderer_->remove_draw_function(*renderer_handle_);
        renderer_handle_.reset();
    }
}

holoscan::expected<void, holoscan::Error> IQEncoderOp::compute(holoscan::ExecutionContext& context)
{
    auto maybe_sample = input.receive();
    if (!maybe_sample) {
        return holoscan::make_unexpected(std::move(maybe_sample).error());
    }
    const holoscan::Tensor& in_tensor = maybe_sample->data;

    if (in_tensor.size() % 8) {
        return holoscan::make_unexpected(holoscan::Error {
            holoscan::ErrorCode::kInvalidArgument,
            "Invalid IQ signal: element count must be a multiple of 8" });
    }

    const std::array<int64_t, 1> shape { in_tensor.size() };
    auto loan = output.allocate_tensor(holoscan::TensorLoanRequest { .shape = shape, .dtype = kInt16 });
    if (!loan) {
        if (!is_output_pressure(loan.error())) {
            return holoscan::make_unexpected(std::move(loan).error());
        }
        record_dropped_output("IQEncoderOp", dropped_, loan.error());
        return {};
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
        return holoscan::make_unexpected(std::move(writer).error());
    }

    // count is number of IQ pairs (2 components per signal)
    cuda_iq_encode(static_cast<int16_t*>(writer->data()),
        static_cast<const float*>(in_tensor.data()),
        static_cast<size_t>(in_tensor.size()) / 2,
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
        record_dropped_output("IQEncoderOp", dropped_, emitted.error());
    }
    return {};
}

} // namespace hololink::operators
