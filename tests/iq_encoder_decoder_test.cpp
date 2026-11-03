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

#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <hololink/operators/iq_dec/iq_dec_op.hpp>
#include <hololink/operators/iq_enc/iq_enc_op.hpp>

namespace hololink::tests {

using Data = std::vector<float>;

constexpr DLDataType kFloat32 { kDLFloat, 32, 1 };
constexpr size_t kMaxFloats = 64;

// Emits one tensor holding the given floats.
class DataSourceOp : public holoscan::Operator<> {
public:
    explicit DataSourceOp(std::shared_ptr<const Data> data)
        : data_(std::move(data))
    {
    }

    holoscan::Output<holoscan::Tensor> output { "output" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.output(output, "output")
            .produces_tensor(holoscan::TensorOutputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = kFloat32,
                    .rank = 1U,
                },
                .bounds = holoscan::tensor_bounds(kMaxFloats * sizeof(float)),
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            });
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract contract;
        contract.trigger(holoscan::OnClock { .period = std::chrono::milliseconds(1) });
        contract.complete_after(holoscan::ComputeCalls { .limit = 1 });
        return contract;
    }

    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override
    {
        const std::array<int64_t, 1> shape { static_cast<int64_t>(data_->size()) };
        auto loan = output.allocate_tensor(holoscan::TensorLoanRequest { .shape = shape, .dtype = kFloat32 });
        if (!loan)
            return holoscan::make_unexpected(std::move(loan).error());
        auto writer = loan->write(context.cuda_stream());
        if (!writer)
            return holoscan::make_unexpected(std::move(writer).error());
        if (cudaMemcpyAsync(writer->data(), data_->data(), data_->size() * sizeof(float), cudaMemcpyHostToDevice,
                context.cuda_stream().get())
            != cudaSuccess)
            return holoscan::make_unexpected(holoscan::Error { holoscan::ErrorCode::kFailure, "cudaMemcpyAsync failed" });
        auto committed = std::move(*writer).commit();
        if (!committed)
            return holoscan::make_unexpected(std::move(committed).error());
        return output.emit(std::move(*loan), holoscan::EmitOptions { .capture_time = context.activation_time() });
    }

private:
    std::shared_ptr<const Data> data_;
};

// Copies the tensor it receives to the host.
class DataCaptureOp : public holoscan::Operator<> {
public:
    explicit DataCaptureOp(std::shared_ptr<Data> data)
        : data_(std::move(data))
    {
    }

    holoscan::Input<holoscan::Tensor> input { "input" };

    void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract contract;
        contract.trigger(holoscan::OnEach { input });
        return contract;
    }

    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override
    {
        auto sample = input.receive();
        if (!sample)
            return holoscan::make_unexpected(std::move(sample).error());
        const holoscan::Tensor& tensor = sample->data;
        data_->resize(tensor.size());
        return tensor.copy_to_host(data_->data(), data_->size() * sizeof(float), context.cuda_stream());
    }

private:
    std::shared_ptr<Data> data_;
};

// Encodes then decodes data; returns the decoded data and how the run ended.
struct RunResult {
    Data decoded;
    holoscan::RunTermination termination = holoscan::RunTermination::kRunning;
    std::vector<holoscan::Diagnostic> diagnostics;
};

RunResult encode_decode(const Data& data, float scale = 1.0f)
{
    holoscan::Graph graph { "iq_encoder_decoder_test" };
    auto decoded = std::make_shared<Data>();
    auto source = graph.op<DataSourceOp>("source", std::make_shared<const Data>(data));
    auto encoder = graph.op<operators::IQEncoderOp>("encoder", scale);
    auto decoder = graph.op<operators::IQDecoderOp>("decoder", uint64_t { kMaxFloats / 2 }, scale);
    auto capture = graph.op<DataCaptureOp>("capture", decoded);
    graph.add_flow(source->output, encoder->input);
    graph.add_flow(encoder->output, decoder->input);
    graph.add_flow(decoder->output, capture->input);
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    for (const char* operator_path : { "source", "encoder", "decoder" })
        options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = operator_path,
            .output_port = "output",
            .device = holoscan::DeviceId { 0 },
        });
    auto plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
        holoscan::log_diagnostics(plan);
        ADD_FAILURE() << "graph compilation failed";
        return {};
    }

    auto session = holoscan::run_async(plan);
    EXPECT_TRUE(session.wait_for(std::chrono::seconds(30)));
    const auto diagnostics = session.diagnostics();
    return RunResult {
        .decoded = *decoded,
        .termination = session.termination(),
        .diagnostics = { diagnostics.begin(), diagnostics.end() },
    };
}

TEST(IQEncoderDecoderTest, DefaultRange)
{
    Data d { .0, .1, .2, .3, .4, .5, .6, .7 };
    auto result = encode_decode(d);
    EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
    ASSERT_EQ(result.decoded.size(), d.size());
    EXPECT_TRUE(std::equal(result.decoded.begin(), result.decoded.end(), d.begin(),
        [](float lhs, float rhs) { return std::abs(lhs - rhs) <= 1.0f / SHRT_MAX; }));
}

TEST(IQEncoderDecoderTest, ClampedRange)
{
    Data d { 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 };
    auto result = encode_decode(d);
    EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
    ASSERT_EQ(result.decoded.size(), d.size());
    EXPECT_TRUE(std::all_of(result.decoded.begin(), result.decoded.end(), [](float value) { return value == 1.0f; }));
}

TEST(IQEncoderDecoderTest, ScaledRange)
{
    Data d { 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 };
    float scale = 2.0f;
    auto result = encode_decode(d, scale);
    EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
    ASSERT_EQ(result.decoded.size(), d.size());
    EXPECT_TRUE(std::equal(result.decoded.begin(), result.decoded.end(), d.begin(),
        [scale](float lhs, float rhs) { return std::abs(lhs - rhs) <= scale / SHRT_MAX; }));
}

TEST(IQEncoderDecoderTest, InvalidBufferSize)
{
    // The encoder rejects a buffer that isn't a whole number of 4-sample IQ groups, failing the
    // run.
    for (size_t i = 1; i < 8; ++i) {
        auto result = encode_decode(Data(i, 0.0f));
        EXPECT_EQ(result.termination, holoscan::RunTermination::kFailed) << "size " << i;
        EXPECT_TRUE(std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
            [](const holoscan::Diagnostic& diagnostic) {
                return diagnostic.detail && (diagnostic.detail->operator_path.find("encoder") != std::string::npos)
                    && (diagnostic.detail->operation == holoscan::DiagnosticOperation::kCompute)
                    && (diagnostic.detail->error.code == holoscan::ErrorCode::kInvalidArgument);
            }))
            << "size " << i;
    }
}

} // namespace hololink::tests
