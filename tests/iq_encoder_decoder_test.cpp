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

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include <hololink/operators/iq_dec/iq_dec_op.hpp>
#include <hololink/operators/iq_enc/iq_enc_op.hpp>

using namespace hololink;

namespace hololink::tests {

using Tensor = std::shared_ptr<holoscan::Tensor>;
using Data = std::vector<float>;

Tensor create_tensor(void* context, gxf_uid_t allocator_uid, const Data& data)
{
    nvidia::gxf::Tensor gxf_tensor;
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context, allocator_uid);
    if (!gxf_tensor.reshape<float>(
            nvidia::gxf::Shape { static_cast<int>(data.size()) },
            nvidia::gxf::MemoryStorageType::kDevice,
            allocator.value()))
        throw std::runtime_error("Failed to allocate cuda memory");
    auto maybe_dl_ctx = gxf_tensor.toDLManagedTensorContext();
    auto tensor = std::make_shared<holoscan::Tensor>(*maybe_dl_ctx);
    cudaMemcpy(tensor->data(), &data.front(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
    return tensor;
}

class DataGeneratorOp : public holoscan::Operator {
public:
    using DataGenerator = std::function<Data()>;
    HOLOSCAN_OPERATOR_FORWARD_ARGS(DataGeneratorOp)
    DataGeneratorOp() = default;

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.output<Tensor>(output_name);

        // Register converters for arguments not defined by Holoscan
        register_converter<DataGenerator>();

        spec.param(data_generator_, "data_generator", "DataGenerator", "Data Generator");
    }

    void initialize() override
    {
        // Create an allocator for the operator
        allocator_ = fragment()->make_resource<holoscan::UnboundedAllocator>("pool");
        // Add the allocator to the operator so that it is initialized
        add_arg(allocator_);

        // Call the base class initialize function
        Operator::initialize();
    }

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override
    {
        auto data = data_generator_.get()();
        auto tensor = create_tensor(context.context(), allocator_->gxf_cid(), data);
        op_output.emit(tensor);
    }

private:
    static inline const char* output_name = "output";
    std::shared_ptr<holoscan::Allocator> allocator_;
    holoscan::Parameter<DataGenerator> data_generator_;
};

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

class IQEncoderDecoderTestApp : public holoscan::Application {
public:
    IQEncoderDecoderTestApp(
        const DataGeneratorOp::DataGenerator& data_generator,
        const DataValidatorOp::DataValidator& data_validator,
        float scale = 1.0f)
        : data_generator_(data_generator)
        , data_validator_(data_validator)
        , scale_(scale)
    {
    }

    void compose() override
    {
        data_generator_op_ = make_operator<DataGeneratorOp>(
            "DataGenerator",
            make_condition<holoscan::CountCondition>(1),
            holoscan::Arg("data_generator", data_generator_));
        iq_encoder_op_ = make_operator<operators::IQEncoderOp>(
            "IQ Encoder",
            holoscan::Arg("scale", scale_));
        iq_decoder_op_ = make_operator<operators::IQDecoderOp>(
            "IQ Decoder",
            holoscan::Arg("scale", scale_));
        data_validator_op_ = make_operator<DataValidatorOp>(
            "DataValidator",
            holoscan::Arg("data_validator", data_validator_));

        add_flow(data_generator_op_, iq_encoder_op_, { { "output", "input" } });
        add_flow(iq_encoder_op_, iq_decoder_op_, { { "output", "input" } });
        add_flow(iq_decoder_op_, data_validator_op_, { { "output", "input" } });
    }

private:
    const DataGeneratorOp::DataGenerator& data_generator_;
    const DataValidatorOp::DataValidator& data_validator_;
    float scale_ = 1.0f;

    std::shared_ptr<DataGeneratorOp> data_generator_op_;
    std::shared_ptr<hololink::operators::IQEncoderOp> iq_encoder_op_;
    std::shared_ptr<hololink::operators::IQDecoderOp> iq_decoder_op_;
    std::shared_ptr<DataValidatorOp> data_validator_op_;
};

TEST(IQEncoderDecoderTest, DefaultRange)
{
    Data d { .0, .1, .2, .3, .4, .5, .6, .7 };
    IQEncoderDecoderTestApp app(
        DataGeneratorOp::DataGenerator([&d] {
            return d;
        }),
        DataValidatorOp::DataValidator([&d](const Data& data) {
            EXPECT_TRUE(std::equal(data.begin(), data.end(), d.begin(), [](float lhs, float rhs) { return std::abs(lhs - rhs) <= 1.0f / SHRT_MAX; }));
        }));
    app.run();
}

TEST(IQEncoderDecoderTest, ClampedRange)
{
    Data d { 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 };
    IQEncoderDecoderTestApp app(
        DataGeneratorOp::DataGenerator([&d] {
            return d;
        }),
        DataValidatorOp::DataValidator([](const Data& data) {
            EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](float value) { return value == 1.0f; }));
        }));
    app.run();
}

TEST(IQEncoderDecoderTest, ScaledRange)
{
    Data d { 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 };
    float scale = 2.0f;
    IQEncoderDecoderTestApp app(
        DataGeneratorOp::DataGenerator([&d] {
            return d;
        }),
        DataValidatorOp::DataValidator([&d, scale](const Data& data) {
            EXPECT_TRUE(std::equal(data.begin(), data.end(), d.begin(), [scale](float lhs, float rhs) { return std::abs(lhs - rhs) <= scale / SHRT_MAX; }));
        }),
        scale);
    app.run();
}

TEST(IQEncoderDecoderTest, InvalidBufferSize)
{
    for (size_t i = 1; i < 8; ++i) {
        EXPECT_ANY_THROW({
            Data d(i, 0.0f);
            IQEncoderDecoderTestApp app(
                DataGeneratorOp::DataGenerator([&d] {
                    return d;
                }),
                DataValidatorOp::DataValidator([&d](const Data&) {
                }));
            app.run();
        });
    }
}

} // namespace hololink::tests

#define YAML_CONVERTER(TYPE)                                                                \
    template <>                                                                             \
    struct YAML::convert<TYPE> {                                                            \
        static Node encode(TYPE&) { throw std::runtime_error("Unsupported"); }              \
                                                                                            \
        static bool decode(const Node&, TYPE&) { throw std::runtime_error("Unsupported"); } \
    };

YAML_CONVERTER(hololink::tests::DataGeneratorOp::DataGenerator);
YAML_CONVERTER(hololink::tests::DataValidatorOp::DataValidator);
