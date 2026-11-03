/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hardware-free tests of CpuRoceTransceiverOp. A live start() needs a RoCE
 * device and an HSB peer, so every run here is one whose start() fails before
 * it reaches either: on a bad parameter or HSB IP version, on a board that no
 * IBV device reaches, or, with a valid configuration, on an IBV device that
 * doesn't exist, which fails before anything is configured on the board. The
 * board is one on the loopback address that is never contacted. Idle ring
 * accessors and stop_receiver() do not open ibverbs.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run_session.hpp>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/metadata.hpp>
#include <hololink/operators/cpu_roce_transceiver/cpu_roce_transceiver_op.hpp>

#include "fake_receiver.hpp"

namespace hololink::operators {

// Reports a CQ monitor failure as CpuRoceTransceiverOp's monitor thread does.
struct CpuRoceTransceiverOpTestAccess {
    static void monitor_failure(CpuRoceTransceiverOp& op, const char* what)
    {
        op.on_monitor_failure(what);
    }
};

} // namespace hololink::operators

namespace {

using hololink::operators::CpuRoceTransceiverOp;
using hololink::operators::CpuRoceTransceiverOpTestAccess;
using hololink::operators::CpuRoceTransceiverParams;
using hololink::tests::fake_receiver::SinkCount;
using hololink::tests::fake_receiver::TensorSinkOp;

// What a start that reaches the NIC fails with, here: "Failed to find IB device" or, on a host
// without ib_uverbs, "Failed to get IB device list".
constexpr const char* kNoIbvDevice = "IB device";

// The channel of a board on the loopback address that is never contacted: BaseReceiverOp only
// binds its data socket through it. hsb_ip_version is left out when empty.
std::shared_ptr<hololink::DataChannel> unconnected_channel(
    std::optional<int64_t> hsb_ip_version = 0x2602)
{
    hololink::Metadata metadata;
    metadata["peer_ip"] = std::string("127.0.0.1");
    metadata["control_port"] = int64_t(8192);
    metadata["serial_number"] = std::string("cpu-roce-transceiver-op-test");
    metadata["sequence_number_checking"] = int64_t(0);
    metadata["fpga_uuid"] = std::string("00000000-0000-0000-0000-000000000000");
    metadata["data_plane"] = int64_t(0);
    metadata["sensor"] = int64_t(0);
    metadata["vp_mask"] = int64_t(1);
    metadata["sif_address"] = int64_t(0x01000000);
    metadata["vp_address"] = int64_t(0x1000);
    metadata["hif_address"] = int64_t(0x02000300);
    if (hsb_ip_version) {
        metadata["hsb_ip_version"] = *hsb_ip_version;
    }
    return std::make_shared<hololink::DataChannel>(metadata);
}

// Forward mode with a host ring on an IBV device that doesn't exist: valid, and never reaches the
// board.
CpuRoceTransceiverParams valid_params()
{
    return CpuRoceTransceiverParams {
        .frame_size = 4096,
        .ibv_name = "no-such-ibv-device",
        .slots = 2,
        .forward = 1,
    };
}

struct RunResult {
    bool compiled = false;
    holoscan::RunTermination termination = holoscan::RunTermination::kFailed;
    std::string diagnostics; // every diagnostic of the run, formatted
};

// Declares the transceiver as "rx" with a sink on its tensor output (as an app's graph has),
// compiles the graph and, when \p run, runs it until it ends by itself or 10 s pass.
RunResult run_transceiver(std::shared_ptr<hololink::DataChannel> channel,
    const CpuRoceTransceiverParams& params, bool run = true)
{
    RunResult result;
    holoscan::Graph graph { "cpu_roce_transceiver_op_test" };
    auto rx = graph.op<CpuRoceTransceiverOp>("rx", std::move(channel));
    rx.set_params(params);
    auto sink = graph.op<TensorSinkOp>("sink", std::make_shared<SinkCount>());
    graph.add_flow(rx->tensor, sink->tensor);
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    holoscan::CompileOptions compile_options;
    compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
        .operator_path = "rx",
        .output_port = "tensor",
        .device = holoscan::DeviceId { 0 },
    });
    auto plan = holoscan::compile(graph, std::move(compile_options));
    if (!plan.ok()) {
        holoscan::log_diagnostics(plan);
        return result;
    }
    result.compiled = true;
    if (!run) {
        return result;
    }
    auto session = holoscan::run_async(plan);
    EXPECT_TRUE(session.wait_for(std::chrono::seconds(10))) << "the run did not end by itself";
    session.request_stop();
    session.wait();
    result.termination = session.termination();
    for (const auto& diagnostic : session.diagnostics()) {
        result.diagnostics += holoscan::format_diagnostic(diagnostic, plan.graph());
        result.diagnostics += '\n';
    }
    return result;
}

// Runs the transceiver, whose start must fail with \p message.
void expect_start_failure(std::shared_ptr<hololink::DataChannel> channel,
    const CpuRoceTransceiverParams& params, const std::string& message)
{
    const RunResult result = run_transceiver(std::move(channel), params);
    ASSERT_TRUE(result.compiled);
    EXPECT_EQ(result.termination, holoscan::RunTermination::kFailed);
    EXPECT_NE(result.diagnostics.find(message), std::string::npos) << result.diagnostics;
}

class CpuRoceTransceiverOpTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
        ASSERT_EQ(cuDeviceGet(&device_, 0), CUDA_SUCCESS);
        ASSERT_EQ(cuDevicePrimaryCtxRetain(&context_, device_), CUDA_SUCCESS);
    }

    void TearDown() override { cuDevicePrimaryCtxRelease(device_); }

    CUdevice device_ = 0;
    CUcontext context_ = nullptr;
};

// One parameter set that start() must reject, and the message it rejects it with.
struct RejectedParams {
    const char* name;
    void (*change)(CpuRoceTransceiverParams&);
    const char* message;
};

} // namespace

TEST_F(CpuRoceTransceiverOpTest, ConstructsAndCompiles)
{
    const RunResult result = run_transceiver(unconnected_channel(), valid_params(), false);
    EXPECT_TRUE(result.compiled);
}

TEST_F(CpuRoceTransceiverOpTest, IdleOperatorHasNoRings)
{
    CpuRoceTransceiverOp op(unconnected_channel());
    EXPECT_FALSE(op.frames_ready());
    EXPECT_FALSE(op.receive_path_failed());
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

// Every parameter check fails the run's start with its own message, before the NIC is opened.
TEST_F(CpuRoceTransceiverOpTest, RejectsInvalidParameters)
{
    const RejectedParams cases[] = {
        { "forward 2", [](CpuRoceTransceiverParams& p) { p.forward = 2; },
            "forward must be 0 (Duplex) or 1 (Forward)" },
        { "gpu_memory in Duplex",
            [](CpuRoceTransceiverParams& p) {
                p.forward = 0;
                p.gpu_memory = true;
            },
            "gpu_memory is currently Forward-only" },
        { "no slots", [](CpuRoceTransceiverParams& p) { p.slots = 0; },
            "requires at least one payload-ring slot" },
        { "ring larger than size_t",
            [](CpuRoceTransceiverParams& p) {
                p.frame_size = std::size_t { 1 } << 52;
                p.slots = 4096;
            },
            "payload ring size overflows size_t" },
        { "stride over 4 GiB", [](CpuRoceTransceiverParams& p) { p.frame_size = std::size_t { 1 } << 32; },
            "stride exceeds ibv_sge length" },
        { "frame_size 0", [](CpuRoceTransceiverParams& p) { p.frame_size = 0; },
            "requires a non-zero frame_size" },
        { "too many slots", [](CpuRoceTransceiverParams& p) { p.slots = 4097; },
            "requires slots <= 4096" },
        { "poll_cpu in Duplex",
            [](CpuRoceTransceiverParams& p) {
                p.forward = 0;
                p.poll_cpu = 0;
            },
            "poll_cpu is not valid in Duplex" },
        { "rx_cpu without tx_cpu",
            [](CpuRoceTransceiverParams& p) {
                p.forward = 0;
                p.rx_cpu = 0;
            },
            "Duplex requires both rx_cpu and tx_cpu, or neither" },
        { "rx_cpu in Forward", [](CpuRoceTransceiverParams& p) { p.rx_cpu = 0; },
            "rx_cpu and tx_cpu are Duplex-only" },
    };
    for (const RejectedParams& rejected : cases) {
        SCOPED_TRACE(rejected.name);
        CpuRoceTransceiverParams params = valid_params();
        params.frame_context = context_;
        rejected.change(params);
        expect_start_failure(unconnected_channel(), params, rejected.message);
    }
}

// The 12-bit immediate-data slot layout needs HSB IP 0x2602 or newer.
TEST_F(CpuRoceTransceiverOpTest, RequiresHsbIp0x2602)
{
    for (const std::optional<int64_t> version : { std::optional<int64_t> {}, std::optional<int64_t> { 0x2601 } }) {
        SCOPED_TRACE(version ? "hsb_ip_version 0x2601" : "no hsb_ip_version");
        expect_start_failure(unconnected_channel(version), valid_params(), "requires hsb_ip_version >= 0x2602");
    }
}

// A valid configuration, Forward and Duplex, passes the operator's checks and the transport's,
// and its start fails only at the NIC.
TEST_F(CpuRoceTransceiverOpTest, ValidConfigurationReachesTheNic)
{
    for (const uint32_t forward : { 1u, 0u }) {
        SCOPED_TRACE(forward ? "Forward" : "Duplex");
        CpuRoceTransceiverParams params = valid_params();
        params.frame_context = context_;
        params.forward = forward;
        expect_start_failure(unconnected_channel(), params, kNoIbvDevice);
    }
}

// Without an ibv_name, the start uses the IBV device on the network interface that reaches the
// board, and fails with the interface's name when there's none: the board's, on the loopback
// address, is "lo".
TEST_F(CpuRoceTransceiverOpTest, PicksTheIbvDeviceThatReachesTheBoard)
{
    CpuRoceTransceiverParams params = valid_params();
    params.frame_context = context_;
    params.ibv_name = "";
    expect_start_failure(unconnected_channel(), params,
        "No InfiniBand device is bound to interface \"lo\" (local IP 127.0.0.1), which reaches the "
        "board at 127.0.0.1");
}

// The GPU ring: an integrated GPU, or a discrete one that can't export its memory as a DMA-BUF,
// gets pinned host memory, which gpu_memory refuses; otherwise the start fails at the NIC. Either
// way, start_receiver() undoes the allocation itself.
TEST_F(CpuRoceTransceiverOpTest, GpuRingFailsAtTheAllocationOrTheNic)
{
    int integrated = 0;
    ASSERT_EQ(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device_), CUDA_SUCCESS);
    CpuRoceTransceiverParams params = valid_params();
    params.frame_context = context_;
    params.gpu_memory = true;
    const RunResult result = run_transceiver(unconnected_channel(), params);
    ASSERT_TRUE(result.compiled);
    EXPECT_EQ(result.termination, holoscan::RunTermination::kFailed);
    const bool fell_back = result.diagnostics.find("fell back to pinned host memory") != std::string::npos;
    if (integrated) {
        EXPECT_TRUE(fell_back) << result.diagnostics;
    } else {
        EXPECT_TRUE(fell_back || result.diagnostics.find(kNoIbvDevice) != std::string::npos)
            << result.diagnostics;
    }
}

// The CQ monitor's failure, as monitor_loop() reports it: outside teardown it marks the receive
// path failed and wakes get_next_frame(), so that compute() fails the run; during teardown it is
// only logged.
TEST_F(CpuRoceTransceiverOpTest, ReportsAMonitorFailureOnlyOutsideTeardown)
{
    {
        CpuRoceTransceiverOp op(unconnected_channel());
        std::chrono::steady_clock::duration waited {};
        std::thread waiter([&] {
            const auto t0 = std::chrono::steady_clock::now();
            (void)op.get_next_frame(5000.0, static_cast<CUstream>(0));
            waited = std::chrono::steady_clock::now() - t0;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CpuRoceTransceiverOpTestAccess::monitor_failure(op, "test failure");
        waiter.join();
        EXPECT_LT(waited, std::chrono::seconds(2));
        EXPECT_TRUE(op.receive_path_failed());
        EXPECT_TRUE(op.frames_ready());
    }
    {
        CpuRoceTransceiverOp op(unconnected_channel());
        op.stop_receiver();
        CpuRoceTransceiverOpTestAccess::monitor_failure(op, "test failure");
        EXPECT_FALSE(op.receive_path_failed());
        EXPECT_FALSE(op.frames_ready());
    }
}
