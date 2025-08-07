/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cmath>

#include <hololink/operators/sig_gen/sig_gen_op.hpp>

namespace hololink::tests {

using Tensor = std::shared_ptr<holoscan::Tensor>;
using Data = std::vector<float>;

class DataValidatorOp : public holoscan::Operator {
public:
    using DataValidator = std::function<void(const Data&)>;
    HOLOSCAN_OPERATOR_FORWARD_ARGS(DataValidatorOp)
    DataValidatorOp() = default;

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input<Tensor>(input_name);

        // Register converters for arguments not defined by Holoscan
        register_converter<DataValidator>();

        spec.param(data_validator_, "data_validator", "DataValidator", "Data Validator");
    }

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override
    {
        auto maybe_tensor = op_input.receive<Tensor>(input_name);
        auto tensor = maybe_tensor.value();
        Data data(tensor->size());
        cudaMemcpy(&data.front(), tensor->data(), tensor->size() * sizeof(float), cudaMemcpyDeviceToHost);
        data_validator_.get()(data);
    }

private:
    static inline const char* input_name = "input";
    holoscan::Parameter<DataValidator> data_validator_;
};

class SignalGeneratorTestApp : public holoscan::Application {
public:
    SignalGeneratorTestApp(
        unsigned samples_count,
        const operators::Rational& sampling_interval,
        const std::string& in_phase,
        const std::string& quadrature,
        const DataValidatorOp::DataValidator& data_validator)
        : samples_count_(samples_count)
        , sampling_interval_(sampling_interval)
        , in_phase_(in_phase)
        , quadrature_(quadrature)
        , data_validator_(data_validator)
    {
    }

    void compose() override
    {
        signal_generator_ = make_operator<operators::SignalGeneratorOp>(
            "Signal Generator",
            make_condition<holoscan::CountCondition>(1),
            holoscan::Arg("samples_count", samples_count_),
            holoscan::Arg("sampling_interval", sampling_interval_),
            holoscan::Arg("in_phase", in_phase_),
            holoscan::Arg("quadrature", quadrature_));
        data_validator_op_ = make_operator<DataValidatorOp>(
            "DataValidator",
            holoscan::Arg("data_validator", data_validator_));

        add_flow(signal_generator_, data_validator_op_, { { "output", "input" } });
    }

private:
    unsigned samples_count_;
    operators::Rational sampling_interval_;
    std::string in_phase_;
    std::string quadrature_;
    const DataValidatorOp::DataValidator& data_validator_;

    std::shared_ptr<hololink::operators::SignalGeneratorOp> signal_generator_;
    std::shared_ptr<DataValidatorOp> data_validator_op_;
};

TEST(SignalGeneratorTest, SamplesCount)
{
    unsigned samples_count = 4;
    SignalGeneratorTestApp app(
        samples_count,
        0,
        "0",
        "0",
        DataValidatorOp::DataValidator([samples_count](const Data& data) {
            EXPECT_EQ(data.size(), samples_count * 2);
        }));
    app.run();
}

TEST(IQEncoderDecoderTest, InvalidSamplesCount)
{
    for (size_t i = 1; i < 4; ++i) {
        EXPECT_ANY_THROW({
            SignalGeneratorTestApp app(
                i,
                0,
                "0",
                "0",
                DataValidatorOp::DataValidator([](const Data&) {
                }));
            app.run();
        });
    }
}

TEST(SignalGeneratorTest, SamplingInterval)
{
    unsigned samples_count = 4;
    operators::Rational sampling_interval(1, samples_count);
    SignalGeneratorTestApp app(
        samples_count,
        sampling_interval,
        "x",
        "x + 1 / " + std::to_string(samples_count * 2),
        DataValidatorOp::DataValidator([](const Data& data) {
            for (size_t i = 0; i < data.size(); ++i)
                EXPECT_EQ(data[i], static_cast<float>(i) / data.size());
        }));
    app.run();
}

TEST(SignalGeneratorTest, SinCos)
{
    unsigned samples_count = 4;
    operators::Rational sampling_interval(1, samples_count);
    SignalGeneratorTestApp app(
        samples_count,
        sampling_interval,
        "sinpi(2*x)",
        "cospi(2*x)",
        DataValidatorOp::DataValidator([samples_count](const Data& data) {
            for (size_t i = 0; i < samples_count; ++i) {
                EXPECT_NEAR(data[2 * i], sin(M_PI * 2 * i / samples_count), 1e-10);
                EXPECT_NEAR(data[2 * i + 1], cos(M_PI * 2 * i / samples_count), 1e-10);
            }
        }));
    app.run();
}

} // namespace hololink::tests

#define YAML_CONVERTER(TYPE)                                                                \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
                                                                                            \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

YAML_CONVERTER(hololink::tests::DataValidatorOp::DataValidator);
