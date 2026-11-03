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

#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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
#include <holoscan/time/realtime_clock.hpp>

#include <hololink/operators/sig_gen/sig_gen_op.hpp>

namespace hololink::tests {

using Data = std::vector<float>;

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

// Generates one buffer with SignalGeneratorOp and returns it.
Data generate(unsigned samples_count, const operators::Rational& sampling_interval,
    const std::string& in_phase, const std::string& quadrature)
{
    holoscan::Graph graph { "signal_generator_test" };
    auto data = std::make_shared<Data>();
    auto generator = graph.op<operators::SignalGeneratorOp>("generator",
        samples_count,
        sampling_interval,
        in_phase,
        quadrature,
        std::string {},
        std::shared_ptr<ImGuiRenderer> {},
        std::chrono::nanoseconds { std::chrono::milliseconds(1) },
        uint64_t { 1 });
    auto capture = graph.op<DataCaptureOp>("capture", data);
    graph.add_flow(generator->output, capture->input);
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
        .operator_path = "generator",
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
    EXPECT_TRUE(session.wait_for(std::chrono::seconds(60)));
    EXPECT_EQ(session.termination(), holoscan::RunTermination::kCompleted);
    return *data;
}

TEST(SignalGeneratorTest, SamplesCount)
{
    unsigned samples_count = 4;
    auto data = generate(samples_count, 0, "0", "0");
    EXPECT_EQ(data.size(), samples_count * 2);
}

TEST(SignalGeneratorTest, InvalidSamplesCount)
{
    // graph.op() constructs the operator right away, so the constructor's validation throws there.
    for (unsigned i = 1; i < 4; ++i) {
        holoscan::Graph graph { "signal_generator_test" };
        EXPECT_ANY_THROW(graph.op<operators::SignalGeneratorOp>("generator", i, operators::Rational(0)));
    }
}

TEST(SignalGeneratorTest, SamplingInterval)
{
    unsigned samples_count = 4;
    operators::Rational sampling_interval(1, samples_count);
    auto data = generate(samples_count, sampling_interval, "x", "x + 1 / " + std::to_string(samples_count * 2));
    ASSERT_EQ(data.size(), samples_count * 2);
    for (size_t i = 0; i < data.size(); ++i)
        EXPECT_EQ(data[i], static_cast<float>(i) / data.size());
}

TEST(SignalGeneratorTest, SinCos)
{
    unsigned samples_count = 4;
    operators::Rational sampling_interval(1, samples_count);
    auto data = generate(samples_count, sampling_interval, "sinpi(2*x)", "cospi(2*x)");
    ASSERT_EQ(data.size(), samples_count * 2);
    for (size_t i = 0; i < samples_count; ++i) {
        EXPECT_NEAR(data[2 * i], sin(M_PI * 2 * i / samples_count), 1e-10);
        EXPECT_NEAR(data[2 * i + 1], cos(M_PI * 2 * i / samples_count), 1e-10);
    }
}

} // namespace hololink::tests
