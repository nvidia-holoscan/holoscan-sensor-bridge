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

// NewestFrameSlot, the hand-off between FusaCoeCaptureCore's acquire thread and
// FusaCoeCaptureOp::compute(): on its own, and driving an OnNotified source in a Holoscan run the
// way FusaCoeCaptureOp does, with a simulated camera in place of NvFusa. Needs no GPU or camera.
// SlotCaptureOp re-implements the operators' posting and take protocol rather than running
// FusaCoeCaptureCore or either FusaCoeCaptureOp, which need NvFusa: keep it in step with them.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <hololink/core/newest_frame_slot.hpp>

namespace hololink::tests {

namespace {

    using Slot = hololink::core::NewestFrameSlot<int>;

} // namespace

// Only a frame stored in an empty slot asks for a notification; one stored over a frame
// compute() has not taken replaces it, and is counted.
TEST(NewestFrameSlot, NotifiesOnlyWhenTheSlotWasEmpty)
{
    std::array<int, 3> frames { 1, 2, 3 };
    Slot slot;

    auto stored = slot.store(&frames[0]);
    EXPECT_TRUE(stored.notify);
    EXPECT_EQ(stored.replaced, nullptr);

    stored = slot.store(&frames[1]);
    EXPECT_FALSE(stored.notify);
    EXPECT_EQ(stored.replaced, &frames[0]);
    EXPECT_EQ(slot.peek(), &frames[1]);

    EXPECT_EQ(slot.take(), &frames[1]);
    EXPECT_EQ(slot.take(), nullptr);

    stored = slot.store(&frames[2]);
    EXPECT_TRUE(stored.notify);
    EXPECT_EQ(stored.replaced, nullptr);

    EXPECT_EQ(slot.take_replaced(), 1u);
    EXPECT_EQ(slot.take_replaced(), 0u);
    EXPECT_EQ(slot.reset(), &frames[2]);
    EXPECT_EQ(slot.peek(), nullptr);
}

// A camera that delivers frames while the run refuses notifications, then faster than compute()
// runs: every notification finds a frame, the newest, and every frame is either taken or counted
// as replaced.
TEST(NewestFrameSlot, OneNotificationPerFrameComputeCanTake)
{
    std::array<int, 20> frames {};
    Slot slot;
    unsigned pending_notifications = 0;
    unsigned taken = 0;
    const auto deliver = [&](int i) {
        frames[i] = i;
        if (slot.store(&frames[i]).notify) {
            ++pending_notifications;
        }
    };
    const auto compute = [&]() -> int* {
        EXPECT_GT(pending_notifications, 0u);
        --pending_notifications;
        int* frame = slot.take();
        EXPECT_NE(frame, nullptr) << "compute() ran for a frame that was already taken";
        taken += frame != nullptr;
        return frame;
    };

    // Start-up: four frames before the first compute(); it takes the newest.
    for (int i = 0; i < 4; ++i) {
        deliver(i);
    }
    EXPECT_EQ(pending_notifications, 1u);
    EXPECT_EQ(*compute(), 3);
    EXPECT_EQ(slot.take_replaced(), 3u);

    // In step: one compute() per frame.
    for (int i = 4; i < 10; ++i) {
        deliver(i);
        EXPECT_EQ(*compute(), i);
    }
    // Behind: two frames per compute(), and one arriving while compute() runs.
    for (int i = 10; i < 18; i += 2) {
        deliver(i);
        deliver(i + 1);
        EXPECT_EQ(*compute(), i + 1);
    }
    deliver(18);
    int* running = compute();
    deliver(19);
    EXPECT_EQ(*running, 18);
    EXPECT_EQ(*compute(), 19);

    EXPECT_EQ(pending_notifications, 0u);
    EXPECT_EQ(taken + slot.take_replaced() + 3u, frames.size());
}

namespace {

    using namespace std::chrono_literals;

    constexpr auto kCameraPeriod = 10ms;
    // Frames the simulated driver holds while the capture thread is busy, as NvFusa's issued
    // capture requests do; it loses later ones.
    constexpr std::size_t kDriverBuffers = 3;
    // The other operators' kStart, e.g. HolovizOp opening its window: the run takes no
    // notification until it is over.
    constexpr auto kStartUp = 100ms;
    constexpr uint64_t kFrameLimit = 60;
    constexpr uint32_t kQueueDepth = 2;

    struct Observed {
        std::atomic<uint64_t> refused_not_ready { 0 };
        std::atomic<uint64_t> refused_backpressured { 0 };
        std::atomic<uint64_t> startup_replaced { 0 };
        std::atomic<uint64_t> computes { 0 };
        std::atomic<uint64_t> empty_computes { 0 };
        std::atomic<uint64_t> published { 0 };
        std::atomic<uint64_t> consumed { 0 };
        std::atomic<uint64_t> max_queued { 0 };
    };

    // An OnNotified capture source, like FusaCoeCaptureOp: a capture thread fed by a simulated
    // camera stores each frame in a NewestFrameSlot and, when the slot was empty, posts one
    // notification, retrying while the run refuses it; compute() takes the newest frame without
    // waiting and publishes it on two ports, paired by capture time.
    class SlotCaptureOp : public holoscan::Operator<> {
    public:
        explicit SlotCaptureOp(std::shared_ptr<Observed> observed)
            : observed_(std::move(observed))
        {
        }

        holoscan::Output<uint64_t> frame;
        holoscan::Output<uint64_t> metadata;

        void setup(holoscan::OperatorSpec& spec) override
        {
            spec.output(frame, "frame").max_emits_per_compute(1U);
            spec.output(metadata, "metadata").max_emits_per_compute(1U);
            spec.notification_source(source_, "frame-ready").capacity(8).sender_reference_capacity(1);
            spec.lifecycle().stage(holoscan::LifecycleStage::kArm, &SlotCaptureOp::on_arm);
        }

        [[nodiscard]] holoscan::Contract contract() const override
        {
            holoscan::Contract contract;
            contract.trigger(holoscan::OnNotified { .event = source_ });
            contract.complete_after(holoscan::ComputeCalls { .limit = kFrameLimit });
            return contract;
        }

        void start() override
        {
            stop_ = false;
            thread_ = std::thread([this] { capture(); });
            std::this_thread::sleep_for(kStartUp);
        }

        void stop() override
        {
            stop_ = true;
            if (thread_.joinable()) {
                thread_.join();
            }
            sender_ = holoscan::NotificationSender {};
        }

        holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override
        {
            ++observed_->computes;
            uint64_t frame_number = 0;
            {
                const std::lock_guard lock(mutex_);
                if (const uint64_t* taken = slot_.take()) {
                    frame_number = *taken;
                    if (!frame_taken_) {
                        frame_taken_ = true;
                        observed_->startup_replaced = slot_.take_replaced();
                    }
                }
            }
            if (frame_number == 0) {
                ++observed_->empty_computes;
                return {};
            }
            auto now = context.clock().now();
            const holoscan::EmitOptions options {
                .capture_time = now ? *now : context.activation_time(),
                .frame_id = frame_number,
            };
            // A full queue fails the run: the consumer must never fall behind.
            if (auto emitted = frame.emit(frame_number, options); !emitted) {
                return emitted;
            }
            ++observed_->published;
            return metadata.emit(frame_number, options);
        }

    private:
        holoscan::LifecycleStatus on_arm(holoscan::LifecycleContext& context) noexcept
        {
            auto sender = context.notification_sender(source_);
            if (!sender) {
                return holoscan::LifecycleStatus::kFatalFailure;
            }
            sender_ = std::move(*sender);
            return holoscan::LifecycleStatus::kOk;
        }

        // FusaCoeCaptureOp::frame_ready() and FusaCoeCaptureCore::notify_compute().
        void notify_compute()
        {
            while (!stop_) {
                auto posted = sender_.post();
                if (posted) {
                    return;
                }
                const auto code = posted.error().code;
                if (code != holoscan::ErrorCode::kNotReady && code != holoscan::ErrorCode::kBackpressured) {
                    return;
                }
                if (code == holoscan::ErrorCode::kNotReady) {
                    ++observed_->refused_not_ready;
                } else {
                    ++observed_->refused_backpressured;
                }
                std::this_thread::sleep_for(1ms);
            }
        }

        void capture()
        {
            std::deque<uint64_t> completed;
            uint64_t camera_frames = 0;
            auto next_frame = std::chrono::steady_clock::now() + kCameraPeriod;
            while (!stop_) {
                for (const auto now = std::chrono::steady_clock::now(); next_frame <= now; next_frame += kCameraPeriod) {
                    ++camera_frames;
                    if (completed.size() < kDriverBuffers) {
                        completed.push_back(camera_frames);
                    }
                }
                if (completed.empty()) {
                    std::this_thread::sleep_until(next_frame);
                    continue;
                }
                uint64_t* acquired = &frames_[completed.front() % frames_.size()];
                *acquired = completed.front();
                completed.pop_front();
                bool notify = false;
                {
                    const std::lock_guard lock(mutex_);
                    notify = slot_.store(acquired).notify;
                }
                if (notify) {
                    notify_compute();
                }
            }
        }

        std::shared_ptr<Observed> observed_;
        holoscan::NotificationSource source_;
        holoscan::NotificationSender sender_;
        std::thread thread_;
        std::atomic<bool> stop_ { false };
        std::mutex mutex_;
        hololink::core::NewestFrameSlot<uint64_t> slot_;
        bool frame_taken_ = false;
        // Frame buffers, reused round-robin: more than the slot and the driver hold at once.
        std::array<uint64_t, 16> frames_ {};
    };

    // Takes each frame and its metadata, paired on an exact capture time as CsiToBayerOp does, and
    // records how many published frames were queued for it.
    class PairedConsumerOp : public holoscan::Operator<> {
    public:
        explicit PairedConsumerOp(std::shared_ptr<Observed> observed)
            : observed_(std::move(observed))
        {
        }

        holoscan::Input<uint64_t> frame;
        holoscan::Input<uint64_t> metadata;

        void setup(holoscan::OperatorSpec& spec) override
        {
            spec.input(frame, "frame").queue_depth(kQueueDepth);
            spec.input(metadata, "metadata").queue_depth(kQueueDepth);
        }

        [[nodiscard]] holoscan::Contract contract() const override
        {
            holoscan::Contract contract;
            contract.trigger(holoscan::OnAligned { .ports = holoscan::ports(frame, metadata), .window = 0ns })
                .input(frame, holoscan::Exact {})
                .input(metadata, holoscan::Exact {});
            return contract;
        }

        holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override
        {
            auto taken = frame.receive();
            auto paired = metadata.receive();
            if (!taken || !paired) {
                return {};
            }
            const uint64_t queued = observed_->published.load() - observed_->consumed.load();
            uint64_t max_queued = observed_->max_queued.load();
            while (queued > max_queued && !observed_->max_queued.compare_exchange_weak(max_queued, queued)) { }
            ++observed_->consumed;
            return {};
        }

    private:
        std::shared_ptr<Observed> observed_;
    };

} // namespace

// The capture's notifications are refused through start-up while frames keep arriving, and the
// capture shares one thread with its consumer, as a FuSa camera chain does. Each compute() still
// takes a frame, so frame_limit counts frames exactly, and the consumer keeps up: none is dropped
// at queue depth 2. (One notification per stored frame instead leaves notifications without a
// frame after start-up, and a compute() that waits for the next frame on them keeps the consumer
// that many frames behind, and full at this depth.)
TEST(NewestFrameSlot, DrivesAnOnNotifiedSourceWithoutABacklog)
{
    auto observed = std::make_shared<Observed>();
    holoscan::Graph graph { "newest_frame_slot_test" };
    auto capture = graph.op<SlotCaptureOp>("capture", observed);
    auto consumer = graph.op<PairedConsumerOp>("consumer", observed);
    const holoscan::ConnectionOptions edge { .queue_depth = kQueueDepth };
    graph.add_flow(capture->frame, consumer->frame, edge);
    graph.add_flow(capture->metadata, consumer->metadata, edge);
    graph.partition("chain").include_all();
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    auto plan = holoscan::compile(graph);
    if (!plan.ok()) {
        holoscan::log_diagnostics(plan);
        FAIL() << "graph compilation failed";
    }

    auto session = holoscan::run_async(plan);
    if (!session.wait_for(10s)) {
        session.request_stop();
        ADD_FAILURE() << "the frame-limited run did not end by itself";
    }
    session.wait();
    for (const auto& diagnostic : session.diagnostics()) {
        if (diagnostic.severity == holoscan::DiagnosticSeverity::kError) {
            ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
        }
    }

    std::cout << "refused kNotReady " << observed->refused_not_ready.load() << ", kBackpressured "
              << observed->refused_backpressured.load() << "; frames replaced at start-up "
              << observed->startup_replaced.load() << "; most frames queued for the consumer "
              << observed->max_queued.load() << std::endl;
    EXPECT_GT(observed->refused_not_ready.load(), 0u) << "no notification was refused at start-up";
    EXPECT_EQ(observed->computes.load(), kFrameLimit);
    EXPECT_EQ(observed->empty_computes.load(), 0u);
    EXPECT_EQ(observed->published.load(), kFrameLimit);
    EXPECT_EQ(observed->consumed.load(), kFrameLimit);
}

} // namespace hololink::tests
