/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Integration test for RoceReceiverOp against a live Hololink Sensor Bridge board
// with an IMX274-Dual camera.
//
// Discovers the board via Enumerator, initialises the IMX274 camera (left sensor,
// expander_configuration=0) in 1920×1080 RAW10 @60 FPS mode, builds an HSDK5 graph
// with RoceReceiverOp → FrameSinkOp, and verifies that frames arrive.
//
// Requirements:
//   - Connected HSB board (default 192.168.0.2, or HOLOLINK_IP env var)
//   - IMX274-Dual camera attached
//   - RoCE NIC (default: the one that reaches the board, or IBV_NAME env var)
//   - GPU access (--gpus all)
//
// Skips when the board is not reachable.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <capture_metadata_traits.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/sensors/camera/imx274/imx274_mode.hpp>
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>

namespace hololink::tests {

// ---------------------------------------------------------------------------
// Shared state — communicated from operator back to test via params.
// ---------------------------------------------------------------------------

struct RoceFrameResults {
    std::atomic<int> count { 0 };
    std::atomic<bool> done { false };
    std::mutex mutex;
    std::condition_variable cv;
    int expect_frames { 3 };

    // Validation data populated by the sink on each received frame.
    std::vector<std::size_t> frame_byte_sizes; // tensor.nbytes() per frame
    std::vector<int64_t> frame_numbers; // schema frame_number field per frame
    std::vector<int64_t> crc_values; // schema crc field per frame
    std::vector<int64_t> bytes_written; // schema bytes_written field per frame
    std::vector<int64_t> received_s; // schema received_s field per frame
    std::vector<uint8_t> first_bytes; // 64 bytes from image-data region of first frame

    void notify()
    {
        if (count.load() >= expect_frames) {
            done = true;
            cv.notify_all();
        }
    }

    bool wait(std::chrono::seconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, timeout, [this] { return done.load(); });
    }
};

// ---------------------------------------------------------------------------
// Sink operator.
// ---------------------------------------------------------------------------

struct FrameSinkParams {
    std::shared_ptr<RoceFrameResults> results;
};

class FrameSinkOp : public holoscan::Operator<FrameSinkParams> {
public:
    holoscan::Input<holoscan::Tensor> tensor;
    holoscan::Input<hololink::CaptureMetadataT> metadata;

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input(tensor, "tensor")
            .queue_depth(4U)
            .expects_tensor(holoscan::TensorInputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                },
            });
        spec.input(metadata, "metadata").queue_depth(4U);
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        // Upstream stamps both ports with an identical capture_time, so OnAligned pairs them on
        // an exact match (see kMetadataPairingWindow).
        // ReuseLast/silent drops are deliberately not used here -- a frame without
        // its metadata counterpart should fail loudly, not proceed with stale data.
        c.trigger(holoscan::OnAligned {
                      .ports = holoscan::ports(tensor, metadata),
                      .window = hololink::operators::kMetadataPairingWindow,
                  })
            .input(tensor, holoscan::Exact {})
            .input(metadata, holoscan::Exact {});
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override
    {
        // Exact{} aligns by capture_time only; this also confirms frame identity (the tensor's
        // frame_id is its metadata's frame_number), as every downstream operator does.
        auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(tensor, metadata);
        if (!maybe_paired)
            return holoscan::make_unexpected(std::move(maybe_paired).error());
        if (!*maybe_paired)
            return holoscan::make_unexpected(
                holoscan::Error { holoscan::ErrorCode::kInvalidArgument, "frame identity mismatch" });
        auto& paired = **maybe_paired;

        const holoscan::Tensor& t = paired.tensor.data;
        const hololink::CaptureMetadataT& frame = paired.metadata.data;

        auto& r = *params().results;
        {
            std::lock_guard<std::mutex> lock(r.mutex);
            r.frame_byte_sizes.push_back(static_cast<std::size_t>(t.nbytes()));
            r.frame_numbers.push_back(frame.frame_number);
            r.crc_values.push_back(frame.crc);
            r.bytes_written.push_back(frame.bytes_written);
            r.received_s.push_back(frame.received_s);

            // On the first frame, sample 64 bytes from the active-image region.
            // For IMX274 1920x1080 RAW10: image data starts at byte 19376
            // (176 bytes embedded metadata + 2400 bytes/line × 8 optical black lines).
            static constexpr std::size_t kImageDataOffset = 19376;
            if (r.frame_numbers.size() == 1
                && static_cast<std::size_t>(t.nbytes()) > kImageDataOffset + 64) {
                r.first_bytes.resize(64);
                cudaStream_t stream = context.cuda_stream().get();
                cudaMemcpyAsync(
                    r.first_bytes.data(),
                    static_cast<const uint8_t*>(t.data()) + kImageDataOffset,
                    64, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
            }
        }

        r.count++;
        r.notify();
        return {};
    }
};

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

TEST(RoceReceiverOpTest, ReceivesFramesFromBoard)
{
    const char* ip_env = std::getenv("HOLOLINK_IP");
    const std::string board_ip = ip_env ? ip_env : "192.168.0.2";

    Metadata channel_meta;
    try {
        channel_meta = Enumerator::find_channel(
            board_ip, std::make_shared<Timeout>(5.f));
    } catch (const std::exception& e) {
        GTEST_SKIP() << "HSB board not found at " << board_ip << ": " << e.what()
                     << ". Set HOLOLINK_IP env var if using a non-default address.";
    }

    // The verbs device: IBV_NAME, else none, so that the receiver picks the one on the network
    // interface that reaches the board, as the examples do by default.
    std::string ibv_name;
    if (const char* ibv_env = std::getenv("IBV_NAME")) {
        ibv_name = ibv_env;
        std::cout << "Using InfiniBand device " << ibv_name << " for board " << board_ip << "\n";
    } else if (hololink::infiniband_devices().empty()) {
        GTEST_SKIP() << "No InfiniBand device on this host; set IBV_NAME to name one.";
    }

    ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
    CUdevice cu_dev;
    ASSERT_EQ(cuDeviceGet(&cu_dev, 0), CUDA_SUCCESS);
    CUcontext cu_ctx;
    ASSERT_EQ(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev), CUDA_SUCCESS);
    struct CtxGuard {
        CUdevice dev;
        ~CtxGuard() { cuDevicePrimaryCtxRelease(dev); }
    } ctx_guard { cu_dev };
    cuCtxSetCurrent(cu_ctx);

    auto data_channel = std::make_shared<DataChannel>(channel_meta);
    auto hololink_dev = data_channel->hololink();
    hololink_dev->start();
    struct DevGuard {
        std::shared_ptr<hololink::Hololink> dev;
        ~DevGuard() { dev->stop(); }
    } dev_guard { hololink_dev };
    hololink_dev->reset();

    // IMX274-Dual: use left camera (expander_configuration=0), 1920x1080 RAW10 @60 FPS.
    // CSI frame size: start_byte=19376 + 2400 bytes/line * 1080 lines = 2,611,376 bytes.
    constexpr auto kCameraMode = sensors::imx274_mode::IMX274_MODE_1920X1080_60FPS;
    constexpr std::size_t kFrameSize = 2'611'376;

    auto camera = std::make_shared<sensors::NativeImx274Sensor>(
        *data_channel, /*expander_configuration=*/0);
    camera->set_mode(kCameraMode);
    camera->setup_clock();
    camera->configure(kCameraMode);

    auto results = std::make_shared<RoceFrameResults>();
    results->expect_frames = 3;

    holoscan::Graph graph { "roce_receiver_op_test" };

    auto rx = graph.op<hololink::operators::RoceReceiverOp>(
        "rx", data_channel, [camera] { camera->start(); }, [camera] { camera->stop(); });
    rx.set_params(hololink::operators::RoceReceiverParams {
        .frame_context = cu_ctx,
        .frame_size = kFrameSize,
        .trim = true,
        .ibv_name = ibv_name,
        .ibv_port = 1,
        .pages = 4,
        .queue_size = 4,
    });

    auto sink = graph.op<FrameSinkOp>("sink");
    sink.set_params(FrameSinkParams { .results = results });

    graph.add_flow(rx->tensor, sink->tensor,
        holoscan::ConnectionOptions { .queue_depth = 4U });
    graph.add_flow(rx->metadata, sink->metadata,
        holoscan::ConnectionOptions { .queue_depth = 4U });
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
        .operator_path = "rx",
        .output_port = "tensor",
        .device = holoscan::DeviceId { 0 },
    });
    auto plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
        for (const auto& d : plan.diagnostics()) {
            std::cerr << "Diagnostic: token=" << d.token
                      << " required=" << d.required
                      << " provided=" << d.provided << "\n";
        }
    }
    ASSERT_TRUE(plan.ok()) << "Graph compilation failed";

    auto session = holoscan::run_async(plan);

    const bool got_frames = results->wait(std::chrono::seconds(10));

    session.request_stop();
    session.wait();

    ASSERT_TRUE(got_frames)
        << "Timed out after 10 s. Received " << results->count.load() << "/3 frames "
        << "(IMX274-Dual 1920x1080 mode, frame_size=" << kFrameSize << ").";
    EXPECT_NE(session.termination(), holoscan::RunTermination::kFailed);
    for (const auto& diagnostic : session.diagnostics()) {
        ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
    }
    std::lock_guard<std::mutex> lock(results->mutex);

    // --- Frame size ---
    // Every received tensor must be within 5% of the expected CSI frame size.
    for (std::size_t i = 0; i < results->frame_byte_sizes.size(); ++i) {
        const std::size_t n = results->frame_byte_sizes[i];
        EXPECT_GE(n, kFrameSize * 95 / 100)
            << "Frame " << i << ": tensor too small (" << n << " bytes)";
        EXPECT_LE(n, kFrameSize * 105 / 100)
            << "Frame " << i << ": tensor too large (" << n << " bytes)";
    }

    // --- What the HSB wrote, and when the frame arrived (from CaptureMetadataT) ---
    for (std::size_t i = 0; i < results->bytes_written.size(); ++i) {
        EXPECT_GE(results->bytes_written[i], int64_t(kFrameSize * 95 / 100))
            << "Frame " << i << ": bytes_written too small";
        EXPECT_LE(results->bytes_written[i], int64_t(kFrameSize * 105 / 100))
            << "Frame " << i << ": bytes_written too large";
        EXPECT_NE(results->received_s[i], 0) << "Frame " << i << ": received_s is zero";
    }

    // --- Monotonic frame numbers (from CaptureMetadataT) ---
    // The board-side frame counter must strictly increase across received frames.
    for (std::size_t i = 1; i < results->frame_numbers.size(); ++i) {
        EXPECT_GT(results->frame_numbers[i], results->frame_numbers[i - 1])
            << "frame_number did not increase: frame " << i - 1
            << " = " << results->frame_numbers[i - 1]
            << ", frame " << i << " = " << results->frame_numbers[i];
    }

    // --- CRC present (from CaptureMetadataT) ---
    // The HSB always writes a non-zero CRC for real frames.
    for (std::size_t i = 0; i < results->crc_values.size(); ++i) {
        EXPECT_NE(results->crc_values[i], 0)
            << "Frame " << i << ": CRC is zero — metadata may not have been filled";
    }

    // --- Non-zero pixel data ---
    // 64 bytes from the active-image region of the first frame must not all be zero.
    ASSERT_FALSE(results->first_bytes.empty()) << "No first-frame bytes were captured";
    const bool has_nonzero = std::any_of(
        results->first_bytes.begin(), results->first_bytes.end(),
        [](uint8_t b) { return b != 0; });
    EXPECT_TRUE(has_nonzero) << "64 image-data bytes of first frame are all zero — "
                             << "DMA may not have written real pixel data";
}

} // namespace hololink::tests
