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

// Hardware-free tests of hololink_module's BaseReceiverOp, the base of its Linux, RoCE and
// HsbController receivers. FakeReceiverOp hands it frames from a device buffer on a thread of its
// own, the way a real receiver's receive thread does (fake_receiver.hpp, shared with
// hololink_base_receiver_op_test, which runs the same cases on the native BaseReceiverOp).

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <tuple>

#include <cuda.h>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run_session.hpp>

#include <hololink/core/metadata.hpp>

#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/operators/base_receiver_op.hpp"
#include "hololink/module/operators/base_receiver_op.inl"

#include "fake_receiver.hpp"

namespace {

using hololink::tests::fake_receiver::FakeFrames;
using hololink::tests::fake_receiver::FakeSource;

struct FakeReceiverParams {
    CUcontext frame_context = nullptr;
    std::size_t frame_size = 0;
    int64_t frame_limit = 0;
    bool frame_limit_ends_run = false;
};

// BaseReceiverOp::setup() sizes the tensor pool from frame_size through a declared schema path.
HOLOSCAN_PARAM_SCHEMA(
    FakeReceiverParams,
    holoscan::param_schema("com.nvidia.hololink.tests.FakeReceiverParams", 1).partial(),
    holoscan::field<&FakeReceiverParams::frame_size>("frame_size"));

class FakeReceiverOp
    : public hololink::module::operators::BaseReceiverOp<FakeReceiverOp, FakeReceiverParams> {
public:
    using Base = hololink::module::operators::BaseReceiverOp<FakeReceiverOp, FakeReceiverParams>;

    explicit FakeReceiverOp(std::shared_ptr<FakeSource> source)
        : Base(hololink::module::EnumerationMetadata {}, hololink::tests::fake_receiver::device_start(source),
            hololink::tests::fake_receiver::device_stop(source))
        , source_(source)
        , frames_(source)
    {
    }

protected:
    void start_receiver() override
    {
        const FakeReceiverParams& p = params();
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

    bool receive_path_failed() override { return frames_.receive_path_failed(); }

private:
    std::shared_ptr<FakeSource> source_;
    FakeFrames frames_;
};

} // namespace

template class hololink::module::operators::BaseReceiverOp<FakeReceiverOp, FakeReceiverParams>;

namespace hololink::tests {

namespace {

    using namespace hololink::tests::fake_receiver;

    constexpr std::size_t kFrameSize = 4096;

    class ModuleBaseReceiverOp : public ::testing::Test {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
            ASSERT_EQ(cuDeviceGet(&device_, 0), CUDA_SUCCESS);
            ASSERT_EQ(cuDevicePrimaryCtxRetain(&context_, device_), CUDA_SUCCESS);
        }

        void TearDown() override { cuDevicePrimaryCtxRelease(device_); }

        // Runs FakeReceiverOp -> sink; see run_receiver().
        holoscan::RunTermination run(const std::shared_ptr<FakeSource>& source,
            const std::shared_ptr<SinkCount>& sink_count, const RunOptions& options = {},
            bool frame_limit_ends_run = false)
        {
            return run_receiver(
                [&](holoscan::Graph& graph, int64_t frame_limit) {
                    auto rx = graph.op<FakeReceiverOp>("rx", source);
                    rx.set_params(FakeReceiverParams { .frame_context = context_,
                        .frame_size = kFrameSize,
                        .frame_limit = frame_limit,
                        .frame_limit_ends_run = frame_limit_ends_run });
                    return rx;
                },
                source, sink_count, options);
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
TEST_F(ModuleBaseReceiverOp, RunsWithItsMetadataOutputUnconnected)
{
    auto source = std::make_shared<FakeSource>();
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count);
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    // Every frame reaches the operator; a loaded machine may still make it drop one.
    EXPECT_EQ(source->delivered.load(), source->frames);
    EXPECT_GT(sink_count->received.load(), 0);
}

// Every tensor goes out with its own frame's metadata: the same frame identity, and every
// CaptureMetadataT field from the receiver's report of that frame.
TEST_F(ModuleBaseReceiverOp, PairsEveryFrameWithItsMetadata)
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

// A consumer slower than the camera must not fail the run. Here the receiver and the sink form a
// linear chain, which runs in one partition, on one thread: the receiver's activations wait for the
// sink, so the frames wait in the receiver (a real one overwrites its oldest pages once its ring is
// full).
TEST_F(ModuleBaseReceiverOp, SurvivesASlowConsumer)
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
// backpressured (SDK 5.0.0.7 first waits up to 5 ms for room): the receiver must drop those frames
// rather than fail the run, and keep taking frames from the camera.
TEST_F(ModuleBaseReceiverOp, DropsFramesForASlowConsumerInAnotherPartition)
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
TEST_F(ModuleBaseReceiverOp, DropsFramesWhileItsTensorPoolIsExhausted)
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
// posts while frames are still queued must be accepted, not coalesced away.
TEST_F(ModuleBaseReceiverOp, DeliversABurstOfQueuedFrames)
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
TEST_F(ModuleBaseReceiverOp, StopsTheReceiverWhenTheDeviceFailsToStart)
{
    auto source = std::make_shared<FakeSource>();
    source->fail_device_start = true;
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run_expecting_failure(source, sink_count);
    EXPECT_EQ(termination, holoscan::RunTermination::kFailed);
    EXPECT_TRUE(source->receiver_stopped);
}

// The receiver is torn down even when stopping the device fails.
TEST_F(ModuleBaseReceiverOp, StopsTheReceiverWhenTheDeviceFailsToStop)
{
    auto source = std::make_shared<FakeSource>();
    source->fail_device_stop = true;
    auto sink_count = std::make_shared<SinkCount>();
    (void)run_expecting_failure(source, sink_count);
    EXPECT_TRUE(source->receiver_stopped);
}

// The Linux receiver's "packets_dropped" and the RoCE receivers' "dropped" both land in
// CaptureMetadataT::packets_dropped.
TEST_F(ModuleBaseReceiverOp, MapsBothDropCounters)
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

// As HSDK4's CountCondition on the receiver: frame_limit frames, then the run completes. An
// activation with no frame queued yet waits for the next one, as with a real receiver, so every
// compute() call within the limit delivers a frame.
TEST_F(ModuleBaseReceiverOp, CompletesAfterItsFrameLimit)
{
    auto source = std::make_shared<FakeSource>();
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .frame_limit = 3 });
    EXPECT_EQ(termination, holoscan::RunTermination::kCompleted);
    EXPECT_EQ(sink_count->received.load(), 3);
}

// With frame_limit_ends_run, the frame_limit-th frame ends the whole run (base_receiver_op_test
// shows the run this is for). Samples still in the pipeline are dropped, so the sink may see fewer.
TEST_F(ModuleBaseReceiverOp, EndsTheRunAfterFrameLimitFramesWithFrameLimitEndsRun)
{
    auto source = std::make_shared<FakeSource>();
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .frame_limit = 3 }, /*frame_limit_ends_run=*/true);
    EXPECT_EQ(termination, holoscan::RunTermination::kRequestedCompleted);
    EXPECT_EQ(source->delivered.load(), 3);
    EXPECT_LE(sink_count->received.load(), 3);
}

// Frames that land while the receiver starts are posted before the run takes notifications
// (kNotReady); compute() must still run for them, rather than wait for a frame that never comes.
TEST_F(ModuleBaseReceiverOp, DeliversFramesQueuedWhileItStarts)
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
TEST_F(ModuleBaseReceiverOp, FailsTheRunWhenTheReceivePathFails)
{
    auto source = std::make_shared<FakeSource>();
    source->fail_receive_path_after = 3;
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .frame_limit = 100, .expect_failure = true });
    EXPECT_EQ(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), 3);
}

// A stall of the receiver's lane -- here the sink, which shares its thread, taking 60 ms over one
// frame -- leaves no backlog behind; see NativeBaseReceiverOp.LeavesNoBacklogAfterALaneStall.
TEST_F(ModuleBaseReceiverOp, LeavesNoBacklogAfterALaneStall)
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

// Every chunk of a sub-frame-like stream reaches the sink through a depth-2 queue, with no
// activation left with no chunk to take; see
// NativeBaseReceiverOp.DeliversEveryChunkOfASubFrameStream.
TEST_F(ModuleBaseReceiverOp, DeliversEveryChunkOfASubFrameStream)
{
    auto source = sub_frame_source(30);
    auto sink_count = std::make_shared<SinkCount>();
    const auto termination = run(source, sink_count, { .edge = { .queue_depth = 2U } });
    EXPECT_NE(termination, holoscan::RunTermination::kFailed);
    EXPECT_EQ(source->delivered.load(), source->frames);
    EXPECT_EQ(sink_count->received.load(), source->frames) << "chunks were dropped downstream";
    EXPECT_EQ(source->idle_takes.load(), 0) << "compute() ran with no chunk queued";
}

// frame_limit counts the chunks of that stream exactly.
TEST_F(ModuleBaseReceiverOp, CompletesAfterItsFrameLimitOfASubFrameStream)
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

// A receiver that reports a frame ready but yields none -- HsbControllerOp over a RoCE receiver
// whose receive path failed, until its watchdog reports the device lost -- doesn't run compute()
// over and over: an activation that finds no frame re-arms only if none was reported ready when
// it took. Its few empty activations don't use up frame_limit.
TEST_F(ModuleBaseReceiverOp, IdlesWhileItsReceiverReportsAFrameItDoesNotYield)
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
