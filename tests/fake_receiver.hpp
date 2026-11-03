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

// A receiver without a network, shared by the tests of the native and the hololink_module
// BaseReceiverOp:
//
// - FakeFrames is a receiver's receive side. A thread of its own queues frames from one device
//   buffer, as a receiver's receive thread does, and next() waits for a frame up to its timeout,
//   as the real receivers' get_next_frame() does. BaseReceiverOp asks with no timeout, so it never
//   waits; FakeSource counts the calls that find no frame queued, which a BaseReceiverOp that
//   waited would turn into waits.
// - TensorSinkOp, MetadataSinkOp and PairedSinkOp take what the receiver emits.
// - run_receiver() runs a receiver into them.

#ifndef HOLOLINK_TESTS_FAKE_RECEIVER_HPP
#define HOLOLINK_TESTS_FAKE_RECEIVER_HPP

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <capture_metadata_traits.hpp>
#include <hololink/core/metadata.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>

namespace hololink::tests::fake_receiver {

// Every byte of every frame the fake receiver delivers.
constexpr uint8_t kFrameByte = 0x5A;

// What the fake receiver delivers, and what the test observes of it.
struct FakeSource {
    int frames = 10;
    std::chrono::microseconds interval { 2000 };
    std::atomic<int> delivered { 0 }; // frames handed to the operator by get_next_frame()
    std::atomic<bool> done { false }; // all frames queued
    std::shared_ptr<hololink::Metadata> extra_metadata; // stamped onto every frame's metadata
    bool fail_device_start = false; // the device start callback throws
    bool fail_device_stop = false; // the device stop callback throws
    // The device start callback returns only once every frame is queued, so all of them land
    // while the operator starts, before the run takes notifications.
    bool frames_during_device_start = false;
    // After this many frames the receive path fails, as a RoCE receiver's does on a fatal CQ/QP
    // event: no frame comes again. -1: never.
    int fail_receive_path_after = -1;
    // Whether receive_path_failed() then reports it. HsbControllerOp reports no failure of the
    // RoCE receiver behind it (its watchdog reports the device lost instead): frames_ready() then
    // stays true while get_next_frame() yields nothing.
    bool report_receive_path_failure = true;
    std::atomic<bool> receiver_stopped { false }; // stop_receiver() ran
    // Called on the receive thread before frame \p frame is queued.
    std::function<void(int frame)> before_frame;
    // The pause after frame \p frame, instead of interval.
    std::function<std::chrono::microseconds(int frame)> interval_after;
    // The most frames the receiver holds, as a receiver's ring of queue_size descriptors: a frame
    // that lands on a full queue takes the place of the oldest one, which is lost. 0: no limit.
    std::size_t queue_size = 0;
    std::atomic<int> ring_drops { 0 }; // frames lost that way
    std::atomic<int> idle_takes { 0 }; // get_next_frame() calls that found no frame queued
    std::atomic<int> empty_takes { 0 }; // get_next_frame() calls that returned no frame
    // The frames delivered when each ingress timeout was reported (BaseReceiverOp::timeout()).
    std::mutex timeouts_mutex;
    std::vector<int> timeouts;

    void report_timeout()
    {
        std::lock_guard lock(timeouts_mutex);
        timeouts.push_back(delivered.load());
    }
};

// A sub-frame-like stream of \p frames frames of five chunks each: four chunks a millisecond
// apart, the fourth followed at once by a short trailing one (the HSB's last chunk of a frame
// whose CSI data carries a header, which completes right after the one before it), then a 10 ms
// gap before the next frame. FakeSource::frames counts the chunks.
inline std::shared_ptr<FakeSource> sub_frame_source(int frames)
{
    constexpr int kChunksPerFrame = 5;
    auto source = std::make_shared<FakeSource>();
    source->frames = frames * kChunksPerFrame;
    source->interval_after = [](int chunk) {
        switch (chunk % kChunksPerFrame) {
        case kChunksPerFrame - 2:
            return std::chrono::microseconds(0);
        case kChunksPerFrame - 1:
            return std::chrono::microseconds(10000);
        default:
            return std::chrono::microseconds(1000);
        }
    };
    return source;
}

// The receiver's device start and stop callbacks, failing when \p source says so.
inline std::function<void()> device_start(std::shared_ptr<FakeSource> source)
{
    return [source = std::move(source)] {
        if (source->fail_device_start) {
            throw std::runtime_error("device start failed");
        }
        if (source->frames_during_device_start) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!source->done && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    };
}

inline std::function<void()> device_stop(std::shared_ptr<FakeSource> source)
{
    return [source = std::move(source)] {
        if (source->fail_device_stop) {
            throw std::runtime_error("device stop failed");
        }
    };
}

// Frame \p frame's metadata, as a receiver's get_next_frame() reports it. Each field carries a value
// of its own, so a field that lands in the wrong CaptureMetadataT field shows.
inline void stamp_frame_metadata(hololink::Metadata& metadata, int frame, std::size_t frame_size)
{
    metadata["frame_number"] = int64_t(frame);
    metadata["bytes_written"] = int64_t(frame_size);
    metadata["received_frame_number"] = int64_t(1000 + frame);
    metadata["timestamp_s"] = int64_t(1'700'000'000 + frame);
    metadata["timestamp_ns"] = int64_t(100'000 + frame);
    metadata["metadata_s"] = int64_t(1'700'000'100 + frame);
    metadata["metadata_ns"] = int64_t(200'000 + frame);
    metadata["received_s"] = int64_t(1'700'000'200 + frame);
    metadata["received_ns"] = int64_t(300'000 + frame);
    metadata["crc"] = int64_t(0xC0DE00 + frame);
    metadata["psn"] = int64_t(400 + frame);
    metadata["frame_packets_received"] = int64_t(500 + frame);
    metadata["frame_bytes_received"] = int64_t(frame_size);
    metadata["imm_data"] = int64_t(600 + frame);
    metadata["page_number"] = int64_t(frame % 2);
}

// The CaptureMetadataT a BaseReceiverOp makes of stamp_frame_metadata()'s fields.
inline void expect_frame_metadata(const hololink::CaptureMetadataT& metadata, int64_t frame, std::size_t frame_size)
{
    SCOPED_TRACE("frame " + std::to_string(frame));
    EXPECT_EQ(metadata.frame_number, frame);
    EXPECT_EQ(metadata.bytes_written, int64_t(frame_size));
    EXPECT_EQ(metadata.received_frame_number, 1000 + frame);
    EXPECT_EQ(metadata.timestamp_s, 1'700'000'000 + frame);
    EXPECT_EQ(metadata.timestamp_ns, 100'000 + frame);
    EXPECT_EQ(metadata.metadata_s, 1'700'000'100 + frame);
    EXPECT_EQ(metadata.metadata_ns, 200'000 + frame);
    EXPECT_EQ(metadata.received_s, 1'700'000'200 + frame);
    EXPECT_EQ(metadata.received_ns, 300'000 + frame);
    EXPECT_EQ(metadata.crc, 0xC0DE00 + frame);
    EXPECT_EQ(metadata.psn, 400 + frame);
    EXPECT_EQ(metadata.frame_packets_received, 500 + frame);
    EXPECT_EQ(metadata.frame_bytes_received, int64_t(frame_size));
    EXPECT_EQ(metadata.imm_data, 600 + frame);
    EXPECT_EQ(metadata.page_number, frame % 2);
    EXPECT_EQ(metadata.packets_dropped, 0);
}

// A receiver's receive side: queues FakeSource::frames frames from one device buffer filled with
// kFrameByte, calling frame_ready() after each, and hands them out through next().
class FakeFrames {
public:
    explicit FakeFrames(std::shared_ptr<FakeSource> source)
        : source_(std::move(source))
    {
    }
    FakeFrames(const FakeFrames&) = delete;
    FakeFrames& operator=(const FakeFrames&) = delete;
    ~FakeFrames() { stop(); }

    void start(CUcontext context, std::size_t frame_size, std::function<void()> frame_ready)
    {
        check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
        check(cuMemAlloc(&buffer_, frame_size), "cuMemAlloc");
        check(cuMemsetD8(buffer_, kFrameByte, frame_size), "cuMemsetD8");
        frame_size_ = frame_size;
        {
            std::lock_guard lock(mutex_);
            stopping_ = false;
            receive_path_failed_ = false;
            queue_.clear();
        }
        thread_ = std::thread([this, context, frame_ready = std::move(frame_ready)] {
            // Bound to the frame context, as a real receive thread is; before_frame() may release
            // tensors here.
            cuCtxSetCurrent(context);
            for (int frame = 0; frame < source_->frames && !stopping_; ++frame) {
                if (frame == source_->fail_receive_path_after) {
                    // Wakes the consumer, as RoceReceiver does on a fatal event.
                    receive_path_failed_ = true;
                    ready_.notify_all();
                    frame_ready();
                    break;
                }
                if (source_->before_frame) {
                    source_->before_frame(frame);
                }
                {
                    std::lock_guard lock(mutex_);
                    if (source_->queue_size > 0 && queue_.size() >= source_->queue_size) {
                        queue_.pop_front();
                        source_->ring_drops++;
                    }
                    queue_.push_back(frame);
                }
                ready_.notify_all();
                frame_ready();
                const auto pause = source_->interval_after ? source_->interval_after(frame) : source_->interval;
                if (pause.count() > 0) {
                    std::this_thread::sleep_for(pause);
                }
            }
            {
                std::lock_guard lock(mutex_);
                source_->done = true;
            }
            ready_.notify_all();
        });
    }

    void stop()
    {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
        if (buffer_) {
            cuMemFree(buffer_);
            buffer_ = 0;
        }
    }

    // The next frame, waiting up to \p timeout_ms for one as a real receiver does; {0, nullptr}
    // when none came, or none will (every frame was queued and taken, or the receiver stopped).
    std::tuple<CUdeviceptr, std::shared_ptr<hololink::Metadata>> next(double timeout_ms)
    {
        int frame = 0;
        {
            std::unique_lock lock(mutex_);
            if (queue_.empty()) {
                source_->idle_takes++;
            }
            (void)ready_.wait_for(lock, std::chrono::duration<double, std::milli>(timeout_ms),
                [this] { return !queue_.empty() || source_->done || stopping_; });
            if (queue_.empty()) {
                source_->empty_takes++;
                return { 0, nullptr };
            }
            frame = queue_.front();
            queue_.pop_front();
        }
        source_->delivered++;
        auto metadata = std::make_shared<hololink::Metadata>();
        stamp_frame_metadata(*metadata, frame, frame_size_);
        if (source_->extra_metadata) {
            for (const auto& [key, value] : *source_->extra_metadata) {
                (*metadata)[key] = value;
            }
        }
        return { buffer_, metadata };
    }

    // As RoceReceiver::frames_ready(): true also once the receive path has failed, as
    // next() then no longer waits.
    bool ready()
    {
        std::lock_guard lock(mutex_);
        return !queue_.empty() || receive_path_failed_;
    }

    bool receive_path_failed() const { return receive_path_failed_ && source_->report_receive_path_failure; }

private:
    static void check(CUresult result, const char* call)
    {
        if (result != CUDA_SUCCESS) {
            throw std::runtime_error(std::string(call) + " failed");
        }
    }

    std::shared_ptr<FakeSource> source_;
    CUdeviceptr buffer_ = 0;
    std::size_t frame_size_ = 0;
    std::thread thread_;
    std::atomic<bool> stopping_ { false };
    std::atomic<bool> receive_path_failed_ { false };
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<int> queue_;
};

struct SinkCount {
    std::atomic<int> received { 0 };
    std::chrono::milliseconds delay { 0 };
    // The sink takes this much longer, once, over the frame it receives after stall_after frames:
    // a stall of the thread it shares with the receiver, unless it runs in a partition of its own.
    int stall_after = -1;
    std::chrono::milliseconds stall { 0 };
    // While set, the sink keeps every tensor it receives, so the receiver's pool slots behind them
    // stay in use until release_retained().
    std::mutex mutex;
    bool retain = false;
    std::vector<holoscan::Tensor> retained;

    // Takes the time the sink spends over the frame it is receiving.
    void take_time() const
    {
        std::this_thread::sleep_for(received.load() == stall_after ? delay + stall : delay);
    }

    // Lets go of the retained tensors and stops retaining.
    void release_retained()
    {
        std::vector<holoscan::Tensor> released;
        std::lock_guard lock(mutex);
        retain = false;
        released.swap(retained);
    }
};

// Counts tensors alone, optionally taking its time over each one or keeping it.
class TensorSinkOp : public holoscan::Operator<> {
public:
    explicit TensorSinkOp(std::shared_ptr<SinkCount> count)
        : count_(std::move(count))
    {
    }

    holoscan::Input<holoscan::Tensor> tensor { "tensor" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input(tensor, "tensor")
            .expects_tensor(holoscan::TensorInputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                },
            });
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnEach { tensor });
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override
    {
        auto sample = tensor.receive();
        if (!sample) {
            return holoscan::make_unexpected(std::move(sample).error());
        }
        count_->take_time();
        {
            std::lock_guard lock(count_->mutex);
            if (count_->retain) {
                count_->retained.push_back(sample->data); // a copy keeps its claim on the pool slot
            }
        }
        count_->received++;
        return {};
    }

    // Retained tensors must not outlive the run that owns their pool.
    void stop() override { count_->release_retained(); }

private:
    std::shared_ptr<SinkCount> count_;
};

// Records the packets_dropped of every CaptureMetadataT it receives.
struct MetadataLog {
    std::mutex mutex;
    std::vector<int64_t> packets_dropped;
};

class MetadataSinkOp : public holoscan::Operator<> {
public:
    explicit MetadataSinkOp(std::shared_ptr<MetadataLog> log)
        : log_(std::move(log))
    {
    }

    holoscan::Input<hololink::CaptureMetadataT> metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override { spec.input(metadata, "metadata"); }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnEach { metadata });
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override
    {
        auto sample = metadata.receive();
        if (!sample) {
            return holoscan::make_unexpected(std::move(sample).error());
        }
        std::lock_guard lock(log_->mutex);
        log_->packets_dropped.push_back(sample->data.packets_dropped);
        return {};
    }

private:
    std::shared_ptr<MetadataLog> log_;
};

// What PairedSinkOp received: each tensor with the metadata it was paired with.
struct PairedFrame {
    uint64_t frame_id = 0; // the tensor sample's
    hololink::CaptureMetadataT metadata;
    int64_t nbytes = 0;
    bool intact = false; // every byte is kFrameByte
};

struct PairedLog {
    std::mutex mutex;
    std::vector<PairedFrame> frames;
    int mismatches = 0; // pairs dropped for a frame identity mismatch
};

// Pairs each tensor with its metadata, as every downstream operator does, and records both.
class PairedSinkOp : public holoscan::Operator<> {
public:
    PairedSinkOp(std::shared_ptr<PairedLog> log, std::shared_ptr<SinkCount> count)
        : log_(std::move(log))
        , count_(std::move(count))
    {
    }

    holoscan::Input<holoscan::Tensor> tensor { "tensor" };
    holoscan::Input<hololink::CaptureMetadataT> metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input(tensor, "tensor")
            .expects_tensor(holoscan::TensorInputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                },
            });
        spec.input(metadata, "metadata");
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnAligned {
                      .ports = holoscan::ports(tensor, metadata),
                      .window = hololink::operators::kMetadataPairingWindow,
                  })
            .input(tensor, holoscan::Exact {})
            .input(metadata, holoscan::Exact {});
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override
    {
        auto paired = hololink::operators::receive_paired_tensor_metadata(tensor, metadata);
        if (!paired) {
            return holoscan::make_unexpected(std::move(paired).error());
        }
        if (!*paired) {
            std::lock_guard lock(log_->mutex);
            log_->mismatches++;
            return {};
        }
        count_->take_time();
        const auto& pair = **paired;
        PairedFrame frame;
        frame.frame_id = pair.tensor.metadata.frame_id;
        frame.metadata = pair.metadata.data;
        frame.nbytes = pair.tensor.data.nbytes();
        std::vector<uint8_t> bytes(static_cast<std::size_t>(frame.nbytes));
        auto copied = pair.tensor.data.copy_to_host(bytes.data(), bytes.size(), context.cuda_stream());
        if (!copied) {
            return copied;
        }
        frame.intact = std::all_of(bytes.begin(), bytes.end(), [](uint8_t byte) { return byte == kFrameByte; });
        {
            std::lock_guard lock(log_->mutex);
            log_->frames.push_back(std::move(frame));
        }
        count_->received++;
        return {};
    }

private:
    std::shared_ptr<PairedLog> log_;
    std::shared_ptr<SinkCount> count_;
};

struct RunOptions {
    holoscan::ConnectionOptions edge { .queue_depth = 4U };
    // Also takes the receiver's metadata to a MetadataSinkOp logging here; otherwise, without
    // paired_log, the metadata output stays unconnected, as apps may leave it.
    std::shared_ptr<MetadataLog> metadata_log;
    // Takes tensor and metadata to a PairedSinkOp logging here, instead of a TensorSinkOp.
    std::shared_ptr<PairedLog> paired_log;
    // The receiver's frame_limit: the run then completes by itself.
    int64_t frame_limit = 0;
    // Runs the sink in a partition of its own, so its queue fills while it is behind. (The
    // receiver and its sink otherwise form a linear chain, which runs in one partition, on one
    // thread: the receiver's activations then wait for the sink, and the frames wait in the
    // receiver, as a real receiver's pages do until it overwrites the oldest.)
    bool sink_partition = false;
    // The run is meant to report an error.
    bool expect_failure = false;
    // With a frame_limit, the run is meant to complete by itself; otherwise it runs as one without.
    bool expect_completion = true;
    // The run lasts at least this long, even once every frame is queued and the sink is quiet.
    std::chrono::milliseconds run_at_least { 0 };
};

// Runs the receiver \p make_receiver(graph, frame_limit) declares as "rx" into its sink until every
// frame is queued and the sink has gone quiet, or with a frame_limit until the run completes by
// itself; returns the termination.
template <typename MakeReceiver>
holoscan::RunTermination run_receiver(MakeReceiver&& make_receiver, const std::shared_ptr<FakeSource>& source,
    const std::shared_ptr<SinkCount>& sink_count, const RunOptions& options)
{
    holoscan::Graph graph { "fake_receiver_test" };
    auto rx = make_receiver(graph, options.frame_limit);
    if (options.paired_log) {
        auto sink = graph.op<PairedSinkOp>("sink", options.paired_log, sink_count);
        graph.add_flow(rx->tensor, sink->tensor, options.edge);
        graph.add_flow(rx->metadata, sink->metadata, options.edge);
        if (options.sink_partition) {
            graph.partition("sink").add(sink);
        }
    } else {
        auto sink = graph.op<TensorSinkOp>("sink", sink_count);
        graph.add_flow(rx->tensor, sink->tensor, options.edge);
        if (options.sink_partition) {
            graph.partition("sink").add(sink);
        }
    }
    if (options.metadata_log) {
        auto metadata_sink = graph.op<MetadataSinkOp>("metadata_sink", options.metadata_log);
        graph.add_flow(rx->metadata, metadata_sink->metadata, options.edge);
    }
    if (options.sink_partition) {
        graph.auto_partition_remaining();
    }
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
        ADD_FAILURE() << "graph compilation failed";
        return holoscan::RunTermination::kFailed;
    }
    auto session = holoscan::run_async(plan);
    const auto started = std::chrono::steady_clock::now();
    if (options.frame_limit > 0 && options.expect_completion) {
        EXPECT_TRUE(session.wait_for(std::chrono::seconds(20))) << "the run did not complete";
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    int last = -1;
    while (std::chrono::steady_clock::now() < deadline && session.is_running()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const int now = sink_count->received.load();
        if (source->done && now == last && std::chrono::steady_clock::now() - started >= options.run_at_least) {
            break;
        }
        last = now;
    }
    session.request_stop();
    session.wait();
    for (const auto& diagnostic : session.diagnostics()) {
        if (!options.expect_failure && diagnostic.severity == holoscan::DiagnosticSeverity::kError) {
            ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
        }
    }
    return session.termination();
}

// Lets the sink keep every tensor until frame \p release, so the receiver's tensor pool runs out:
// before frame \p release is queued, waits until the receiver has taken every earlier frame, then
// lets the retained tensors go.
inline void exhaust_the_pool_until(
    const std::shared_ptr<FakeSource>& source, const std::shared_ptr<SinkCount>& sink_count, int release)
{
    sink_count->retain = true;
    source->before_frame = [source = source.get(), sink_count, release](int frame) {
        if (frame != release) {
            return;
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (source->delivered.load() < release && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        sink_count->release_retained();
    };
}

// Every paired frame carries its own metadata, as stamp_frame_metadata() stamped it, and its own
// identity, and the frames came in order.
inline void expect_paired_frames(const PairedLog& log, std::size_t frame_size)
{
    EXPECT_EQ(log.mismatches, 0) << "tensors arrived with another frame's metadata";
    ASSERT_FALSE(log.frames.empty());
    int64_t previous = -1;
    for (const PairedFrame& frame : log.frames) {
        EXPECT_GT(frame.metadata.frame_number, previous);
        previous = frame.metadata.frame_number;
        EXPECT_EQ(frame.frame_id, static_cast<uint64_t>(frame.metadata.frame_number));
        EXPECT_EQ(frame.nbytes, int64_t(frame_size));
        EXPECT_TRUE(frame.intact) << "frame " << frame.metadata.frame_number << " has the wrong bytes";
        expect_frame_metadata(frame.metadata, frame.metadata.frame_number, frame_size);
    }
}

} // namespace hololink::tests::fake_receiver

#endif // HOLOLINK_TESTS_FAKE_RECEIVER_HPP
