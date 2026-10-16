/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Construction smoke test for CpuRoceTransceiverOp. Live start() requires a
 * RoCE device and an HSB peer; this test does not call start() or
 * start_receiver() (hololink_channel is unused until peer_ip()).
 * make_operator() runs Holoscan setup() and registers spec.param() keys.
 * compose_graph() proves composition succeeds and the C++ symbol is
 * reachable. Idle ring accessors and stop_receiver() do not open ibverbs.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <cuda.h>
#include <holoscan/holoscan.hpp>

#include <hololink/core/data_channel.hpp>
#include <hololink/operators/cpu_roce_transceiver/cpu_roce_transceiver_op.hpp>

namespace {

using CpuRoceTransceiverOp = hololink::operators::CpuRoceTransceiverOp;

void expect_cpu_roce_param_keys(CpuRoceTransceiverOp& op)
{
    ASSERT_NE(op.spec(), nullptr);
    const auto& params = op.spec()->params();
    for (const char* key : { "ibv_name", "ibv_port", "tx_ibv_qp", "slots", "forward",
             "gpu_memory", "frame_metadata", "poll_cpu", "rx_cpu", "tx_cpu" }) {
        EXPECT_NE(params.find(key), params.end()) << key;
    }
}

void expect_idle_rings(CpuRoceTransceiverOp& op)
{
    EXPECT_FALSE(op.frames_ready());
    for (const bool is_rx : { true, false }) {
        EXPECT_EQ(op.get_ring_addr(is_rx), nullptr);
        EXPECT_EQ(op.get_ring_stride_sz(is_rx), 0u);
        EXPECT_EQ(op.get_ring_stride_num(is_rx), 0u);
        EXPECT_EQ(op.get_ring_flag_addr(is_rx), nullptr);
    }

    const auto idle = op.get_next_frame(1.0, static_cast<CUstream>(0));
    EXPECT_EQ(std::get<0>(idle), static_cast<CUdeviceptr>(0));
    EXPECT_EQ(std::get<1>(idle), nullptr);

    EXPECT_NO_THROW(op.stop_receiver());
    EXPECT_NO_THROW(op.stop_receiver());

    const auto t0 = std::chrono::steady_clock::now();
    const auto after_stop = op.get_next_frame(1000.0, static_cast<CUstream>(0));
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_LT(elapsed, std::chrono::milliseconds(200));
    EXPECT_EQ(std::get<0>(after_stop), static_cast<CUdeviceptr>(0));
    EXPECT_EQ(std::get<1>(after_stop), nullptr);
}

class CpuRoceTransceiverOpTestApp : public holoscan::Application {
public:
    std::shared_ptr<CpuRoceTransceiverOp> op() const { return op_; }

    void compose() override
    {
        op_ = make_operator<CpuRoceTransceiverOp>(
            "cpu_roce_transceiver",
            holoscan::Arg("frame_context", static_cast<CUcontext>(nullptr)),
            holoscan::Arg("frame_size", std::size_t { 4096 }),
            holoscan::Arg(
                "hololink_channel", static_cast<hololink::DataChannel*>(nullptr)),
            holoscan::Arg("ibv_name", std::string("unused-without-start")),
            holoscan::Arg("ibv_port", std::uint32_t { 1 }),
            holoscan::Arg("tx_ibv_qp", std::uint32_t { 2 }),
            holoscan::Arg("slots", std::uint32_t { 2 }),
            holoscan::Arg("forward", std::uint32_t { 1 }),
            holoscan::Arg("device_start", std::function<void()> { [] {} }),
            holoscan::Arg("device_stop", std::function<void()> { [] {} }));
        add_operator(op_);
    }

private:
    std::shared_ptr<CpuRoceTransceiverOp> op_;
};

} // namespace

TEST(CpuRoceTransceiverOp, ConstructsAndComposes)
{
    auto app = holoscan::make_application<CpuRoceTransceiverOpTestApp>();
    EXPECT_NO_THROW(app->compose_graph());
    ASSERT_NE(app->op(), nullptr);
    EXPECT_EQ(app->op()->name(), "cpu_roce_transceiver");
    expect_cpu_roce_param_keys(*app->op());
    expect_idle_rings(*app->op());
}
