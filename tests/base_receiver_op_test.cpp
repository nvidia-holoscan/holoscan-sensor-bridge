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

// Hardware-free tests of the native BaseReceiverOp, the base of LinuxReceiverOp, RoceReceiverOp
// and GpuRoceTransceiverOp, with the cases hololink_module_receiver_op_test runs on
// hololink_module's own copy of it. NativeFakeReceiverOp hands it frames from a device buffer on a
// thread of its own (fake_receiver.hpp). BaseReceiverOp binds its data socket through a
// DataChannel, here one for a board on the loopback address that is never contacted.

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include <cuda.h>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run_session.hpp>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/metadata.hpp>
#include <hololink/operators/base_receiver_op.hpp>
#include <hololink/operators/base_receiver_op.inl>

#include "fake_receiver.hpp"

namespace {

using hololink::tests::fake_receiver::FakeFrames;
using hololink::tests::fake_receiver::FakeSource;

struct NativeFakeReceiverParams {
    CUcontext frame_context = nullptr;
    std::size_t frame_size = 0;
    bool trim = false;
    int64_t frame_limit = 0;
    bool frame_limit_ends_run = false; // as LinuxReceiverParams::frame_limit_ends_run
    bool watch_ingress = false; // as LinuxReceiverParams::watch_ingress
};

// BaseReceiverOp::setup() sizes the tensor pool from frame_size through a declared schema path.
HOLOSCAN_PARAM_SCHEMA(
    NativeFakeReceiverParams,
    holoscan::param_schema("com.nvidia.hololink.tests.NativeFakeReceiverParams", 1).partial(),
    holoscan::field<&NativeFakeReceiverParams::frame_size>("frame_size"));

class NativeFakeReceiverOp
    : public hololink::operators::BaseReceiverOp<NativeFakeReceiverOp, NativeFakeReceiverParams> {
public:
    using Base = hololink::operators::BaseReceiverOp<NativeFakeReceiverOp, NativeFakeReceiverParams>;

    NativeFakeReceiverOp(std::shared_ptr<hololink::DataChannel> channel, std::shared_ptr<FakeSource> source)
        : Base(std::move(channel), hololink::tests::fake_receiver::device_start(source),
            hololink::tests::fake_receiver::device_stop(source))
        , source_(source)
        , frames_(source)
    {
    }

protected:
    void start_receiver() override
    {
        const NativeFakeReceiverParams& p = params();
        frames_.start(p.frame_context, p.frame_size, [this] { frame_ready(); });
    }

    void stop_receiver() override
    {
        frames_.stop();
        source_->receiver_stopped = true;
    }

    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> get_next_frame(
        double timeout_ms, CUstream /*cuda_stream*/) override
    {
        return frames_.next(timeout_ms);
    }

    bool frames_ready() override { return frames_.ready(); }

    bool watch_ingress() const override { return params().watch_ingress; }

    bool receive_path_failed() override { return frames_.receive_path_failed(); }

    void timeout(holoscan::ExecutionContext& context) override
    {
        source_->report_timeout();
        Base::timeout(context);
    }

private:
    std::shared_ptr<FakeSource> source_;
    FakeFrames frames_;
};

} // namespace

template class hololink::operators::BaseReceiverOp<NativeFakeReceiverOp, NativeFakeReceiverParams>;

namespace {

// The inputs of a HolovizOp that renders on one camera's layer: an activation per "left" tensor,
// taking the newest "right" one, or the last one again.
class LeftTriggeredOp : public holoscan::Operator<> {
public:
    holoscan::Input<holoscan::Tensor> left { "left" };
    holoscan::Input<holoscan::Tensor> right { "right" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        const holoscan::TensorInputSpec device_tensor {
            .representation = holoscan::TensorRepresentation {
                .memory_kind = holoscan::MemoryKind::kCudaDevice,
            },
        };
        spec.input(left, "left").expects_tensor(device_tensor);
        spec.input(right, "right").expects_tensor(device_tensor);
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        using namespace std::chrono_literals;
        holoscan::Contract c;
        c.trigger(holoscan::OnEach { left })
            .input(right, holoscan::LatestWithin { .max_age = 100ms })
            .on_missing(right, holoscan::ReuseLast { .max_age = 100ms });
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override
    {
        auto sample = left.receive();
        if (!sample) {
            return holoscan::make_unexpected(std::move(sample).error());
        }
        (void)right.receive();
        return {};
    }
};

} // namespace

namespace hololink::tests {

namespace {

    using namespace hololink::tests::fake_receiver;

    constexpr std::size_t kFrameSize = 4096;

    // The channel of a board on the loopback address that is never contacted: BaseReceiverOp only
    // binds its data socket through it, and the fake receiver configures nothing on the board.
    std::shared_ptr<hololink::DataChannel> unconnected_channel()
    {
        hololink::Metadata metadata;
        metadata["peer_ip"] = std::string("127.0.0.1");
        metadata["control_port"] = int64_t(8192);
        metadata["serial_number"] = std::string("base-receiver-op-test");
        metadata["sequence_number_checking"] = int64_t(0);
        metadata["fpga_uuid"] = std::string("00000000-0000-0000-0000-000000000000");
        metadata["data_plane"] = int64_t(0);
        metadata["sensor"] = int64_t(0);
        metadata["vp_mask"] = int64_t(1);
        metadata["sif_address"] = int64_t(0x01000000);
        metadata["vp_address"] = int64_t(0x1000);
        metadata["hif_address"] = int64_t(0x02000300);
        return std::make_shared<hololink::DataChannel>(metadata);
    }

    class NativeBaseReceiverOp : public ::testing::Test {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
            ASSERT_EQ(cuDeviceGet(&device_, 0), CUDA_SUCCESS);
            ASSERT_EQ(cuDevicePrimaryCtxRetain(&context_, device_), CUDA_SUCCESS);
        }

        void TearDown() override { cuDevicePrimaryCtxRelease(device_); }

        // Runs NativeFakeReceiverOp -> sink; see run_receiver().
        holoscan::RunTermination run(const std::shared_ptr<FakeSource>& source,
            const std::shared_ptr<SinkCount>& sink_count, const RunOptions& options = {},
            bool watch_ingress = false, bool frame_limit_ends_run = false)
        {
            return run_receiver(
                [&](holoscan::Graph& graph, int64_t frame_limit) {
                    auto rx = graph.op<NativeFakeReceiverOp>("rx", unconnected_channel(), source);
                    rx.set_params(NativeFakeReceiverParams {
                        .frame_context = context_,
                        .frame_size = kFrameSize,
                        .frame_limit = frame_limit,
                        .frame_limit_ends_run = frame_limit_ends_run,
                        .watch_ingress = watch_ingress,
                    });
                    return rx;
                },
                source, sink_count, options);
        }

        struct LeftTriggeredRun {
            bool ended = false;
            holoscan::RunTermination termination = holoscan::RunTermination::kFailed;
            int left_delivered = 0;
            int right_delivered = 0;
        };

        // Runs a "left" receiver with a frame_limit of 3 and a "right" one with a frame_limit of 20,
        // sending \p right_frames frames, into a LeftTriggeredOp. The right receiver's frames from
        // its fourth on wait until the left receiver has taken its last frame, so they reach
        // LeftTriggeredOp after the left stream has ended. \p left_ends_run and \p right_ends_run
        // are the receivers' frame_limit_ends_run. Returns whether the run ended within 2 s, its
        // termination, and how many frames each receiver took.
        LeftTriggeredRun run_left_triggered(bool left_ends_run, bool right_ends_run = false, int right_frames = 20)
        {
            auto left_source = std::make_shared<FakeSource>();
            left_source->frames = 3;
            auto right_source = std::make_shared<FakeSource>();
            right_source->frames = right_frames;
            right_source->before_frame = [left_source](int frame) {
                if (frame != 3) {
                    return;
                }
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (left_source->delivered < left_source->frames && std::chrono::steady_clock::now() < deadline) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                // Lets the left receiver's last frame reach LeftTriggeredOp first.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            };

            holoscan::Graph graph { "left_triggered_test" };
            auto left = graph.op<NativeFakeReceiverOp>("left", unconnected_channel(), left_source);
            left.set_params(NativeFakeReceiverParams {
                .frame_context = context_,
                .frame_size = kFrameSize,
                .frame_limit = left_source->frames,
                .frame_limit_ends_run = left_ends_run,
            });
            auto right = graph.op<NativeFakeReceiverOp>("right", unconnected_channel(), right_source);
            right.set_params(NativeFakeReceiverParams {
                .frame_context = context_,
                .frame_size = kFrameSize,
                .frame_limit = 20,
                .frame_limit_ends_run = right_ends_run,
            });
            auto consumer = graph.op<LeftTriggeredOp>("consumer");
            const holoscan::ConnectionOptions edge {
                .queue_policy = holoscan::QueuePolicy::kDropOldest,
                .queue_depth = 1U,
            };
            graph.add_flow(left->tensor, consumer->left, edge);
            graph.add_flow(right->tensor, consumer->right, edge);
            graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

            holoscan::CompileOptions compile_options;
            for (const char* receiver : { "left", "right" }) {
                compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
                    .operator_path = receiver,
                    .output_port = "tensor",
                    .device = holoscan::DeviceId { 0 },
                });
            }
            auto plan = holoscan::compile(graph, std::move(compile_options));
            if (!plan.ok()) {
                holoscan::log_diagnostics(plan);
                ADD_FAILURE() << "graph compilation failed";
                return {};
            }
            auto session = holoscan::run_async(plan);
            LeftTriggeredRun result;
            result.ended = session.wait_for(std::chrono::seconds(2));
            if (!result.ended) {
                session.request_stop();
                session.wait();
            }
            result.termination = session.termination();
            result.left_delivered = left_source->delivered.load();
            result.right_delivered = right_source->delivered.load();
            return result;
        }

        // Like run(), for a run expected to report an error.
        holoscan::RunTermination run_expecting_failure(
            const std::shared_ptr<FakeSource>& source, const std::shared_ptr<SinkCount>& sink_count)
        {
            return run(source, sink_count, { .expect_failure = true });
        }

        CUdevice device_ = 0;
        CUcontext context_ = nullptr;
    };

} // namespace

// Apps may leave the metadata output unconnected; the receiver must keep delivering.
TEST_F(NativeBaseReceiverOp, RunsWithItsMetadataOutputUnconnected)
{
    auto source = std::make_shared<FakeSource>();
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count);
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), source->frames);
    EXPECT_GT(sink_count->received.load(), 0);
}

// Every tensor goes out with its own frame's metadata: the same frame identity, and every
// CaptureMetadataT field from the receiver's report of that frame.
TEST_F(NativeBaseReceiverOp, PairsEveryFrameWithItsMetadata)
{
    auto source = std::make_shared<FakeSource>();
    auto sink_count = std::make_shared<SinkCount>();
    auto paired_log = std::make_shared<PairedLog>();
    const auto termination = run(source, sink_count, { .paired_log = paired_log });
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), source->frames);
    std::lock_guard lock(paired_log->mutex);
    expect_paired_frames(*paired_log, kFrameSize);
}

// A linear chain runs in one partition: the receiver's activations wait for the slow sink, and the
// frames wait in the receiver. Nothing fails.
TEST_F(NativeBaseReceiverOp, SurvivesASlowConsumer)
{
    auto source = std::make_shared<FakeSource>();
    source->frames = 40;
    source->interval = std::chrono::microseconds(1000);
    auto sink_count = std::make_shared<SinkCount>();
    sink_count->delay = std::chrono::milliseconds(20);
    const auto termination = run(source, sink_count, { .edge = { .queue_depth = 1U } });
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), source->frames);
    EXPECT_GT(sink_count->received.load(), 0);
}

// With the slow sink in a partition of its own its one-deep queue fills, and the receiver's emit is
// backpressured: the receiver must drop those frames rather than fail the run.
TEST_F(NativeBaseReceiverOp, DropsFramesForASlowConsumerInAnotherPartition)
{
    auto source = std::make_shared<FakeSource>();
    source->frames = 40;
    source->interval = std::chrono::microseconds(1000);
    auto sink_count = std::make_shared<SinkCount>();
    sink_count->delay = std::chrono::milliseconds(50);
    const auto termination = run(source, sink_count, { .edge = { .queue_depth = 1U }, .sink_partition = true });
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), source->frames);
    EXPECT_GT(sink_count->received.load(), 0);
    EXPECT_LT(sink_count->received.load(), source->frames) << "no frame was dropped";
}

// A consumer that holds on to its frames uses up the receiver's tensor pool: the receiver must drop
// the frames it has no slot for (kResourceExhausted) rather than fail the run, and deliver again
// once the slots come back.
TEST_F(NativeBaseReceiverOp, DropsFramesWhileItsTensorPoolIsExhausted)
{
    constexpr int kRelease = 24; // well past the pool's few slots
    auto source = std::make_shared<FakeSource>();
    source->frames = 32;
    source->interval = std::chrono::microseconds(1000);
    auto sink_count = std::make_shared<SinkCount>();
    exhaust_the_pool_until(source, sink_count, kRelease);
    const auto termination = run(source, sink_count);
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), source->frames);
    EXPECT_LT(sink_count->received.load(), kRelease) << "no frame was dropped while the pool was exhausted";
    // The frames before the pool ran out, and every frame after the release.
    EXPECT_GT(sink_count->received.load(), source->frames - kRelease);
}

// Frames that complete back to back each get their compute(): the re-arm notification compute()
// posts while frames are still queued must be accepted, not coalesced away (notification
// capacity 8, not 1).
TEST_F(NativeBaseReceiverOp, DeliversABurstOfQueuedFrames)
{
    auto source = std::make_shared<FakeSource>();
    source->frames = 6;
    source->interval = std::chrono::microseconds(0);
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .edge = { .queue_depth = 8U } });
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), source->frames);
    EXPECT_EQ(sink_count->received.load(), source->frames);
}

// A start() that fails gets no stop(), so it must tear the receiver down itself: a receiver thread
// left running would abort the process when the operator is destroyed.
TEST_F(NativeBaseReceiverOp, StopsTheReceiverWhenTheDeviceFailsToStart)
{
    auto source = std::make_shared<FakeSource>();
    source->fail_device_start = true;
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run_expecting_failure(source, sink_count);
    EXPECT_EQ(termination, holoscan::RunTermination::kFailed);
    EXPECT_TRUE(source->receiver_stopped);
}

// The receiver is torn down even when stopping the device fails.
TEST_F(NativeBaseReceiverOp, StopsTheReceiverWhenTheDeviceFailsToStop)
{
    auto source = std::make_shared<FakeSource>();
    source->fail_device_stop = true;
    auto sink_count = std::make_shared<SinkCount>();
    (void)run_expecting_failure(source, sink_count);
    EXPECT_TRUE(source->receiver_stopped);
}

// The Linux receiver's "packets_dropped" and the RoCE receiver's "dropped" both land in
// CaptureMetadataT::packets_dropped.
TEST_F(NativeBaseReceiverOp, MapsBothDropCounters)
{
    auto source = std::make_shared<FakeSource>();
    source->frames = 3;
    source->extra_metadata = std::make_shared<hololink::Metadata>();
    (*source->extra_metadata)["packets_dropped"] = int64_t(2);
    (*source->extra_metadata)["dropped"] = int64_t(3);
    auto sink_count = std::make_shared<SinkCount>();
    auto metadata_log = std::make_shared<MetadataLog>();
    const auto termination = run(source, sink_count, { .metadata_log = metadata_log });
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    std::lock_guard lock(metadata_log->mutex);
    ASSERT_FALSE(metadata_log->packets_dropped.empty());
    for (const int64_t dropped : metadata_log->packets_dropped) {
        EXPECT_EQ(dropped, 5);
    }
}

// As HSDK4's CountCondition on the receiver: frame_limit frames, then the run completes.
TEST_F(NativeBaseReceiverOp, CompletesAfterItsFrameLimit)
{
    auto source = std::make_shared<FakeSource>();
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .frame_limit = 3 });
    EXPECT_EQ(termination, holoscan::RunTermination::kCompleted);
    EXPECT_EQ(sink_count->received.load(), 3);
}

// With watch_ingress, a receiver whose frames stop still runs compute() about once a second, and
// each of these ingress timeouts counts toward frame_limit, as HSDK4's timeouts counted toward its
// CountCondition: two frames and a timeout complete a limit of three.
TEST_F(NativeBaseReceiverOp, CountsAnIngressTimeoutTowardItsFrameLimit)
{
    auto source = std::make_shared<FakeSource>();
    source->frames = 2;
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .frame_limit = 3 }, /*watch_ingress=*/true);
    EXPECT_EQ(termination, holoscan::RunTermination::kCompleted);
    EXPECT_EQ(sink_count->received.load(), 2);
}

// HSDK5 ends a run once every operator has completed, and an operator triggered by one input never
// completes while another input of its holds a sample that came after the triggering input's
// stream ended. Here the right receiver's last frames come after the left receiver completed at
// its frame_limit: both receivers take all their frames, and still the run does not end by itself,
// as a stereo player whose HolovizOp renders on the left camera's layer does not. (If HSDK5 changes
// this, frame_limit_ends_run may go.)
TEST_F(NativeBaseReceiverOp, ATriggeredConsumerKeepsAFrameLimitedRunGoing)
{
    const auto run = run_left_triggered(/*left_ends_run=*/false);
    EXPECT_FALSE(run.ended);
    EXPECT_NE(run.termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(run.left_delivered, 3);
    EXPECT_EQ(run.right_delivered, 20);
}

// With frame_limit_ends_run on the left receiver only, its frame_limit ends the whole run, before
// the right receiver reaches its own.
TEST_F(NativeBaseReceiverOp, EndsTheRunAtItsFrameLimitWithFrameLimitEndsRun)
{
    const auto run = run_left_triggered(/*left_ends_run=*/true);
    EXPECT_TRUE(run.ended);
    EXPECT_EQ(run.termination, holoscan::RunTermination::kRequestedCompleted);
    EXPECT_EQ(run.left_delivered, 3);
    EXPECT_LT(run.right_delivered, 20);
}

// With frame_limit_ends_run on both receivers, the run ends once the last of them reaches its
// frame_limit.
TEST_F(NativeBaseReceiverOp, EndsTheRunOnceEveryReceiverReachesItsFrameLimit)
{
    const auto run = run_left_triggered(/*left_ends_run=*/true, /*right_ends_run=*/true);
    EXPECT_TRUE(run.ended);
    EXPECT_EQ(run.termination, holoscan::RunTermination::kRequestedCompleted);
    EXPECT_EQ(run.left_delivered, 3);
    EXPECT_EQ(run.right_delivered, 20);
}

// A receiver that never reaches its frame_limit, as a camera that stops sending, keeps a run with
// frame_limit_ends_run going: the other receiver's frame_limit does not end it.
TEST_F(NativeBaseReceiverOp, KeepsTheRunGoingWhileAReceiverMissesItsFrameLimit)
{
    const auto run = run_left_triggered(/*left_ends_run=*/true, /*right_ends_run=*/true, /*right_frames=*/5);
    EXPECT_FALSE(run.ended);
    EXPECT_NE(run.termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(run.left_delivered, 3);
    EXPECT_EQ(run.right_delivered, 5);
}

// frame_limit_ends_run ends the run on the frame_limit-th frame. Samples still in the pipeline are
// dropped, so the sink may see fewer.
TEST_F(NativeBaseReceiverOp, EndsTheRunAfterFrameLimitFramesWithFrameLimitEndsRun)
{
    auto source = std::make_shared<FakeSource>();
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .frame_limit = 3 }, /*watch_ingress=*/false,
        /*frame_limit_ends_run=*/true);
    EXPECT_EQ(termination, holoscan::RunTermination::kRequestedCompleted);
    EXPECT_EQ(source->delivered.load(), 3);
    EXPECT_LE(sink_count->received.load(), 3);
}

// The call that reaches frame_limit ends the run even when it brings no frame, as complete_after()
// counts it: two frames and an ingress timeout end a limit of three.
TEST_F(NativeBaseReceiverOp, EndsTheRunOnAnIngressTimeoutThatReachesItsFrameLimit)
{
    auto source = std::make_shared<FakeSource>();
    source->frames = 2;
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .frame_limit = 3 }, /*watch_ingress=*/true,
        /*frame_limit_ends_run=*/true);
    EXPECT_EQ(termination, holoscan::RunTermination::kRequestedCompleted);
    EXPECT_EQ(source->delivered.load(), 2);
}

// Frames that land while the receiver starts are posted before the run takes notifications
// (kNotReady); compute() must still run for them, rather than wait for a frame that never comes.
TEST_F(NativeBaseReceiverOp, DeliversFramesQueuedWhileItStarts)
{
    auto source = std::make_shared<FakeSource>();
    source->frames = 4;
    source->frames_during_device_start = true;
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .frame_limit = 4 });
    EXPECT_EQ(termination, holoscan::RunTermination::kCompleted);
    EXPECT_EQ(sink_count->received.load(), 4);
}

// A receive path that fails for good (a RoCE receiver's fatal CQ/QP event) fails the run: no frame
// will come, so a frame limit would never be reached and the run would never end.
TEST_F(NativeBaseReceiverOp, FailsTheRunWhenTheReceivePathFails)
{
    auto source = std::make_shared<FakeSource>();
    source->fail_receive_path_after = 3;
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .frame_limit = 100, .expect_failure = true });
    EXPECT_EQ(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), 3);
}

// A stall of the receiver's lane -- here the sink, which shares its thread, taking 60 ms over one
// frame -- leaves no backlog behind. The frames that land meanwhile replace each other in the
// receiver (queue_size 1, LinuxReceiverOp's and RoceReceiverOp's default), and only the first of
// them posts a notification: one stays outstanding until compute() takes a frame. With one
// notification per frame, the stall left activations with no frame to take, which waited for
// later frames and ran ahead of the sink on their thread for the rest of the run, so that its
// depth-2 queue dropped frames.
TEST_F(NativeBaseReceiverOp, LeavesNoBacklogAfterALaneStall)
{
    auto source = std::make_shared<FakeSource>();
    source->frames = 60;
    source->interval = std::chrono::microseconds(4000);
    source->queue_size = 1;
    auto sink_count = std::make_shared<SinkCount>();
    sink_count->stall_after = 10;
    sink_count->stall = std::chrono::milliseconds(60);
    const auto termination = run(source, sink_count, { .edge = { .queue_depth = 2U } });
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    EXPECT_GT(source->ring_drops.load(), 0) << "no frame landed during the stall";
    EXPECT_EQ(source->delivered.load() + source->ring_drops.load(), source->frames);
    EXPECT_EQ(sink_count->received.load(), source->delivered.load()) << "frames were dropped downstream";
    EXPECT_EQ(source->idle_takes.load(), 0) << "compute() ran with no frame queued";
}

// A sub-frame-like stream (sub_frame_source()): compute() takes one chunk per activation, and the
// trailing chunk, which lands while the activation for the chunk before it runs, gets its
// activation from compute()'s re-arm alone, not from a notification of its own as well. So no
// activation is left with no chunk to take, the receiver never gets ahead of the sink on their
// thread, and every chunk reaches the sink through a depth-2 queue.
TEST_F(NativeBaseReceiverOp, DeliversEveryChunkOfASubFrameStream)
{
    auto source = sub_frame_source(30);
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .edge = { .queue_depth = 2U } });
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), source->frames);
    EXPECT_EQ(sink_count->received.load(), source->frames) << "chunks were dropped downstream";
    EXPECT_EQ(source->idle_takes.load(), 0) << "compute() ran with no chunk queued";
}

// frame_limit counts the chunks of that stream exactly: the run completes after frame_limit
// chunks, every one of them delivered.
TEST_F(NativeBaseReceiverOp, CompletesAfterItsFrameLimitOfASubFrameStream)
{
    constexpr int kFrameLimit = 100;
    auto source = sub_frame_source(30);
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .edge = { .queue_depth = 2U }, .frame_limit = kFrameLimit });
    EXPECT_EQ(termination, holoscan::RunTermination::kCompleted);
    EXPECT_EQ(source->delivered.load(), kFrameLimit);
    EXPECT_EQ(sink_count->received.load(), kFrameLimit) << "chunks were dropped downstream";
    EXPECT_EQ(source->idle_takes.load(), 0) << "compute() ran with no chunk queued";
}

// The ingress watch still reports a stall, without compute() waiting for a frame: frames 0 and 1,
// then nothing for 2.5 s, then frames 2 and 3. Each activation the watch posts during the gap finds
// no frame and reports an ingress timeout at once, and frames 2 and 3 get activations of their own.
// (A compute() that waited for the next frame took frame 2 in the watch's activation instead, so a
// later activation found no frame, and reported a timeout after the gap.)
TEST_F(NativeBaseReceiverOp, ReportsIngressTimeoutsWithoutWaitingForAFrame)
{
    auto source = std::make_shared<FakeSource>();
    source->frames = 4;
    source->interval = std::chrono::microseconds(0);
    source->before_frame = [](int frame) {
        if (frame == 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2500));
        }
    };
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, {}, /*watch_ingress=*/true);
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), source->frames);
    EXPECT_EQ(sink_count->received.load(), source->frames);
    std::lock_guard lock(source->timeouts_mutex);
    ASSERT_FALSE(source->timeouts.empty()) << "no ingress timeout was reported";
    for (const int delivered : source->timeouts) {
        EXPECT_EQ(delivered, 2) << "an ingress timeout was reported outside the gap";
    }
    EXPECT_EQ(source->idle_takes.load(), static_cast<int>(source->timeouts.size()))
        << "compute() ran with no frame queued other than for an ingress timeout";
}

// A receiver that reports a frame ready but yields none -- HsbControllerOp over a RoCE receiver
// whose receive path failed, until its watchdog reports the device lost -- doesn't run compute()
// over and over: an activation that finds no frame re-arms only if none was reported ready when
// it took. Its few empty activations don't use up frame_limit.
TEST_F(NativeBaseReceiverOp, IdlesWhileItsReceiverReportsAFrameItDoesNotYield)
{
    auto source = std::make_shared<FakeSource>();
    source->fail_receive_path_after = 3;
    source->report_receive_path_failure = false;
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count,
        { .frame_limit = 20, .expect_completion = false, .run_at_least = std::chrono::milliseconds(600) });
    EXPECT_NE(termination, holoscan::RunTermination::kCompleted) << "empty activations used up frame_limit";
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), 3);
    EXPECT_EQ(sink_count->received.load(), 3);
    EXPECT_LE(source->empty_takes.load(), 2) << "compute() kept running with no frame to take";
}

} // namespace hololink::tests
