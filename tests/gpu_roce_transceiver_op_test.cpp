/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Construction test for GpuRoceTransceiverOp. Live start() requires a GPU, a
 * RoCE device and an HSB peer; this test does not call start().
 * make_operator() runs Holoscan setup(), which registers the parameters and
 * the argument setter of hololink_channels: Holoscan's default setter rejects
 * a vector of a custom type such as DataChannel pointers.
 */

#include <gtest/gtest.h>

#include <any>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include <holoscan/core/argument_setter.hpp>
#include <holoscan/holoscan.hpp>

#include <hololink/core/data_channel.hpp>
#include <hololink/operators/gpu_roce_transceiver/gpu_roce_transceiver_op.hpp>

namespace {

using GpuRoceTransceiverOp = hololink::operators::GpuRoceTransceiverOp;
using DataChannels = std::vector<hololink::DataChannel*>;

class GpuRoceTransceiverOpTestApp : public holoscan::Application {
public:
    std::shared_ptr<GpuRoceTransceiverOp> op() const { return op_; }

    void compose() override
    {
        op_ = make_operator<GpuRoceTransceiverOp>(
            "gpu_roce_transceiver",
            holoscan::Arg("frame_context", static_cast<CUcontext>(nullptr)),
            holoscan::Arg("frame_size", std::size_t { 4096 }),
            holoscan::Arg(
                "hololink_channel", static_cast<hololink::DataChannel*>(nullptr)),
            holoscan::Arg("tx_ibv_qp", std::string("2,4")),
            holoscan::Arg("device_start", std::function<void()> { [] {} }),
            holoscan::Arg("device_stop", std::function<void()> { [] {} }));
        add_operator(op_);
    }

private:
    std::shared_ptr<GpuRoceTransceiverOp> op_;
};

} // namespace

TEST(GpuRoceTransceiverOp, HololinkChannelsArgument)
{
    auto app = holoscan::make_application<GpuRoceTransceiverOpTestApp>();
    ASSERT_NO_THROW(app->compose_graph());
    ASSERT_NE(app->op(), nullptr);

    auto& params = app->op()->spec()->params();
    for (const char* key : { "hololink_channel", "hololink_channels", "tx_ibv_qp", "ibv_name",
             "ibv_port", "gpu_id", "forward" }) {
        EXPECT_NE(params.find(key), params.end()) << key;
    }
    auto channels_param = params.find("hololink_channels");
    ASSERT_NE(channels_param, params.end());

    // Only the pointer values are checked, the channels are never dereferenced
    const DataChannels channels {
        reinterpret_cast<hololink::DataChannel*>(0x1000),
        reinterpret_cast<hololink::DataChannel*>(0x2000),
    };
    holoscan::Arg arg("hololink_channels", channels);
    ASSERT_NO_THROW(holoscan::ArgumentSetter::set_param(channels_param->second, arg));

    auto* param = std::any_cast<holoscan::Parameter<DataChannels>*>(channels_param->second.value());
    EXPECT_EQ(param->get(), channels);
}
