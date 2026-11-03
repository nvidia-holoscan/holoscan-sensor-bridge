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

// ComputeCrcOp hands each frame's CRC to its CheckCrcOp through a CrcChannel. The checker must
// get the CRC of the very frame it is looking at, even while ComputeCrcOp is already computing
// later frames' CRCs, and frames dropped between the two must not shift the pairing.

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <hololink/core/csi_controller.hpp>
#include <hololink/core/csi_formats.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/compute_crc/compute_crc.hpp>
#include <hololink/operators/compute_crc/crc_channel.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#if defined(HOLOLINK_TEST_MODULE_OPERATORS) && defined(HOLOLINK_MODULE_BUILD_CRC)
#include <hololink/module/csi_converter.hpp>
#include <hololink/module/operators/compute_crc_op.hpp>
#include <hololink/module/operators/csi_to_bayer_op.hpp>
#endif

#include "synthetic_csi_source.hpp"

namespace hololink::tests {

namespace {

    using hololink::operators::CrcChannel;
    using namespace hololink::tests::synthetic;

    // CRC-32/JAMCRC, the HSB's format: the standard reflected CRC-32 without its final inversion.
    uint32_t jamcrc(const uint8_t* data, std::size_t size)
    {
        static const std::array<uint32_t, 256> table = [] {
            std::array<uint32_t, 256> t {};
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int bit = 0; bit < 8; ++bit) {
                    c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                }
                t[i] = c;
            }
            return t;
        }();
        uint32_t crc = 0xFFFFFFFFu;
        for (std::size_t i = 0; i < size; ++i) {
            crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
        }
        return crc;
    }

    // Publishes \p value as frame \p frame_number's CRC, as ComputeCrcOp would.
    void publish(CrcChannel& channel, int64_t frame_number, uint32_t value)
    {
        CrcChannel::Slot* slot = channel.acquire(frame_number);
        ASSERT_EQ(cudaMemcpy(slot->result_dptr, &value, sizeof(value), cudaMemcpyHostToDevice), cudaSuccess);
        channel.publish(slot, nullptr);
    }

} // namespace

TEST(CrcChannel, TakesEachFrameItsOwnCrc)
{
    CrcChannel channel;
    publish(channel, 10, 0xA);
    publish(channel, 11, 0xB);
    publish(channel, 12, 0xC);
    EXPECT_EQ(channel.take(10), std::optional<uint32_t>(0xA));
    EXPECT_EQ(channel.take(11), std::optional<uint32_t>(0xB));
    EXPECT_EQ(channel.take(12), std::optional<uint32_t>(0xC));
    EXPECT_EQ(channel.take(12), std::nullopt);
}

// A frame dropped between the two operators leaves its CRC behind; taking a later frame's CRC
// discards it instead of handing it to the wrong frame.
TEST(CrcChannel, RecyclesFramesThatNeverReachTheChecker)
{
    CrcChannel channel;
    for (int64_t frame = 1; frame <= 5; ++frame) {
        publish(channel, frame, uint32_t(frame));
    }
    EXPECT_EQ(channel.take(4), std::optional<uint32_t>(4));
    for (int64_t frame = 1; frame <= 4; ++frame) {
        EXPECT_EQ(channel.take(frame), std::nullopt) << "frame " << frame;
    }
    EXPECT_EQ(channel.take(5), std::optional<uint32_t>(5));
}

TEST(CrcChannel, EvictsTheOldestBeyondItsDepth)
{
    CrcChannel channel(2);
    publish(channel, 1, 1);
    publish(channel, 2, 2);
    publish(channel, 3, 3);
    EXPECT_EQ(channel.take(1), std::nullopt);
    EXPECT_EQ(channel.take(2), std::optional<uint32_t>(2));
    EXPECT_EQ(channel.take(3), std::optional<uint32_t>(3));
}

namespace {

    using FrameCrcs = std::vector<std::pair<int64_t, uint32_t>>; // {frame_number, computed_crc}

    struct CrcLog {
        std::mutex mutex;
        FrameCrcs crcs;
    };

    // A checker slower than the source: records each frame's CRC and takes its time, so later
    // frames' CRCs are in flight while it validates.
    template <typename Base>
    class RecordingCheckCrcOp : public Base {
    public:
        RecordingCheckCrcOp(std::shared_ptr<CrcChannel> channel, std::shared_ptr<CrcLog> log)
            : Base(std::move(channel))
            , log_(std::move(log))
        {
        }

        using Base::check_crc;
        void check_crc(uint32_t computed_crc, const hololink::CaptureMetadataT& metadata) override
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            std::lock_guard lock(log_->mutex);
            log_->crcs.emplace_back(metadata.frame_number, computed_crc);
        }

    private:
        std::shared_ptr<CrcLog> log_;
    };

    // Takes the images the pipeline produces, and with kForwardMetadata passes their metadata on
    // to a checker after it. That keeps the pipeline a single line: an operator whose outputs go
    // to two different operators forks the graph into partitions that run independently, so a
    // slow checker on its own branch costs frames.
    template <bool kForwardMetadata>
    class ImageSinkOp : public holoscan::Operator<> {
    public:
        holoscan::Input<holoscan::schema::ImageT> image { "image" };
        holoscan::Input<hololink::CaptureMetadataT> metadata { "metadata" };
        holoscan::Output<hololink::CaptureMetadataT> forwarded_metadata { "forwarded_metadata" };

        void setup(holoscan::OperatorSpec& spec) override
        {
            (void)spec.input(image, "image").tensor_field(holoscan::schema::ImageTensorFields::data);
            if constexpr (kForwardMetadata) {
                spec.input(metadata, "metadata");
                spec.output(forwarded_metadata, "forwarded_metadata");
            }
        }

        [[nodiscard]] holoscan::Contract contract() const override
        {
            holoscan::Contract c;
            if constexpr (kForwardMetadata) {
                c.trigger(holoscan::OnAligned {
                              .ports = holoscan::ports(image, metadata),
                              .window = hololink::operators::kMetadataPairingWindow,
                          })
                    .input(image, holoscan::Exact {})
                    .input(metadata, holoscan::Exact {});
            } else {
                c.trigger(holoscan::OnEach { image });
            }
            return c;
        }

        [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override
        {
            if constexpr (kForwardMetadata) {
                auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(image, metadata);
                if (!maybe_paired) {
                    return holoscan::make_unexpected(std::move(maybe_paired).error());
                }
                if (!*maybe_paired) {
                    return holoscan::make_unexpected(
                        holoscan::Error { holoscan::ErrorCode::kInvalidArgument, "frame identity mismatch" });
                }
                auto& paired = **maybe_paired;
                auto emitted = forwarded_metadata.emit(paired.metadata.data,
                    hololink::operators::propagate_emit_options(
                        paired.metadata.metadata, static_cast<uint64_t>(paired.metadata.data.frame_number)));
                if (!emitted) {
                    return holoscan::make_unexpected(std::move(emitted).error());
                }
            } else {
                auto sample = image.receive();
                if (!sample) {
                    return holoscan::make_unexpected(std::move(sample).error());
                }
            }
            return {};
        }
    };

    constexpr Geometry kFrame { .sub_frame_rows = 0 };
    // Bounds every payload ComputeCrcOp sees here (a CSI frame, or its 16-bit image).
    constexpr uint64_t kMaxCrcBytes = 64 * 1024;

    std::shared_ptr<const SourceConfig> distinct_frames(uint32_t frames)
    {
        SourceConfig config { .geometry = kFrame, .frames = frames };
        for (uint32_t frame = 0; frame < frames; ++frame) {
            config.csi_frames.push_back(csi_frame(kFrame, frame));
        }
        return std::make_shared<const SourceConfig>(std::move(config));
    }

    // Compiles and runs \p graph, its \p tensor_producers' "tensor" outputs on device 0, and
    // returns the CRCs the checker recorded in \p log.
    FrameCrcs run(holoscan::Graph& graph, const std::vector<const char*>& tensor_producers, CrcLog& log)
    {
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
        holoscan::CompileOptions options;
        for (const char* producer : tensor_producers) {
            options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
                .operator_path = producer,
                .output_port = "tensor",
                .device = holoscan::DeviceId { 0 },
            });
        }
        auto plan = holoscan::compile(graph, std::move(options));
        if (!plan.ok()) {
            holoscan::log_diagnostics(plan);
            ADD_FAILURE() << "graph compilation failed";
            return {};
        }
        auto session = holoscan::run_async(plan);
        EXPECT_TRUE(session.wait_for(std::chrono::seconds(30))) << "session did not finish";
        for (const auto& diagnostic : session.diagnostics()) {
            ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
        }
        EXPECT_EQ(session.termination(), holoscan::RunTermination::kCompleted);
        std::lock_guard lock(log.mutex);
        return log.crcs;
    }

    struct NativeOperators {
        using CheckCrc = hololink::operators::CheckCrcOp;
        using CsiToBayer = hololink::operators::CsiToBayerOp;

        static auto compute_crc(holoscan::Graph& graph, std::shared_ptr<CrcChannel> channel)
        {
            return graph.op<hololink::operators::ComputeCrcTensorOp>(
                "compute_crc", std::move(channel), 0, kMaxCrcBytes);
        }
        static hololink::operators::CsiToBayerParams csi_to_bayer_params()
        {
            return {
                .geometry = hololink::csi::CsiGeometry {
                    .start_byte = kFrame.start_byte,
                    .bytes_per_line = kFrame.bytes_per_line(),
                    .pixel_width = kFrame.width,
                    .pixel_height = kFrame.height,
                    .pixel_format = hololink::csi::PixelFormat::RAW_10,
                },
                .bayer_format = hololink::csi::BayerFormat::RGGB,
            };
        }
    };

#if defined(HOLOLINK_TEST_MODULE_OPERATORS) && defined(HOLOLINK_MODULE_BUILD_CRC)
    struct ModuleOperators {
        using CheckCrc = hololink::module::operators::CheckCrcOp;
        using CsiToBayer = hololink::module::operators::CsiToBayerOp;

        static auto compute_crc(holoscan::Graph& graph, std::shared_ptr<CrcChannel> channel)
        {
            auto op = graph.op<hololink::module::operators::ComputeCrcOp>("compute_crc", std::move(channel));
            op.set_params(hololink::module::operators::ComputeCrcParams { .frame_size = kMaxCrcBytes });
            return op;
        }
        static hololink::module::operators::CsiToBayerParams csi_to_bayer_params()
        {
            return {
                .geometry = hololink::module::csi::CsiGeometry {
                    .start_byte = kFrame.start_byte,
                    .bytes_per_line = kFrame.bytes_per_line(),
                    .pixel_width = kFrame.width,
                    .pixel_height = kFrame.height,
                    .pixel_format = hololink::module::csi::PixelFormat::RAW_10,
                },
                .bayer_format = hololink::module::csi::BayerFormat::RGGB,
            };
        }
    };
#endif

    // source -> ComputeCrc -> CsiToBayer -> sink -> a slow CheckCrc, the CRCs covering the received
    // CSI frames. With \p kFork the checker takes CsiToBayerOp's metadata on a branch of its own,
    // through a one-deep queue that drops the oldest, in a partition of its own: neither held back
    // by the checker nor run in step with it, CsiToBayerOp outruns it.
    template <typename Operators, bool kFork = false>
    FrameCrcs run_csi_frame_crcs(uint32_t frames)
    {
        auto channel = std::make_shared<CrcChannel>();
        auto log = std::make_shared<CrcLog>();
        holoscan::Graph graph { "crc_pairing_test" };
        auto source = graph.op<ChunkSourceOp>("source", distinct_frames(frames), std::make_shared<SourceStats>());
        auto compute_crc = Operators::compute_crc(graph, channel);
        auto csi = graph.op<typename Operators::CsiToBayer>("csi_to_bayer");
        csi.set_params(Operators::csi_to_bayer_params());
        auto sink = graph.op<ImageSinkOp<!kFork>>("sink");
        auto check_crc = graph.op<RecordingCheckCrcOp<typename Operators::CheckCrc>>("check_crc", channel, log);
        const holoscan::ConnectionOptions edge { .queue_depth = 4U };
        graph.add_flow(source->tensor, compute_crc->input_tensor, edge);
        graph.add_flow(source->metadata, compute_crc->input_metadata, edge);
        graph.add_flow(compute_crc->output_tensor, csi->input_tensor, edge);
        graph.add_flow(compute_crc->output_metadata, csi->input_metadata, edge);
        graph.add_flow(csi->output_tensor, sink->image, edge);
        if constexpr (kFork) {
            graph.add_flow(csi->output_metadata, check_crc->input_metadata,
                holoscan::ConnectionOptions { .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U });
            graph.partition("checker").add(check_crc);
            graph.auto_partition_remaining();
        } else {
            graph.add_flow(csi->output_metadata, sink->metadata, edge);
            graph.add_flow(sink->forwarded_metadata, check_crc->input_metadata, edge);
        }
        return run(graph, { "source", "csi_to_bayer" }, *log);
    }

    void expect_csi_frame_crcs(const FrameCrcs& crcs, uint32_t frames)
    {
        ASSERT_EQ(crcs.size(), frames);
        for (uint32_t frame = 0; frame < frames; ++frame) {
            SCOPED_TRACE("frame " + std::to_string(frame));
            const std::vector<uint8_t> bytes = csi_frame(kFrame, frame);
            EXPECT_EQ(crcs[frame].first, int64_t(frame));
            EXPECT_EQ(crcs[frame].second, jamcrc(bytes.data(), bytes.size()));
        }
    }

    // Some but not all of the \p sent frames reached the checker, in order, each with its own CRC.
    void expect_crcs_of_some_frames(const FrameCrcs& crcs, uint32_t sent)
    {
        ASSERT_FALSE(crcs.empty());
        // At 5 ms a frame against the source's 1 ms the checker can't keep up; if it ever does, the
        // drops this is about didn't happen.
        EXPECT_LT(crcs.size(), sent) << "no frame was dropped before the checker";
        int64_t previous = -1;
        for (const auto& [frame_number, crc] : crcs) {
            SCOPED_TRACE("frame " + std::to_string(frame_number));
            ASSERT_GT(frame_number, previous);
            ASSERT_LT(frame_number, int64_t(sent));
            const std::vector<uint8_t> bytes = csi_frame(kFrame, static_cast<uint32_t>(frame_number));
            EXPECT_EQ(crc, jamcrc(bytes.data(), bytes.size()));
            previous = frame_number;
        }
    }

} // namespace

TEST(CrcPairing, SlowCheckerSeesEachFramesCrc)
{
    expect_csi_frame_crcs(run_csi_frame_crcs<NativeOperators>(12), 12);
}

// On a branch of its own the slow checker falls behind, and its queue drops frames whose CRCs
// ComputeCrcOp already computed; the frames that do reach the checker still get their own CRCs.
TEST(CrcPairing, FramesDroppedBeforeTheCheckerLeaveThePairingIntact)
{
    expect_crcs_of_some_frames(run_csi_frame_crcs<NativeOperators, true>(24), 24);
}

// The CRC of an image covers its pixels: source -> CsiToBayer -> ComputeCrc -> ImageProcessor ->
// sink -> a slow CheckCrc.
TEST(CrcPairing, ImageCrcsAfterCsiToBayer)
{
    auto channel = std::make_shared<CrcChannel>();
    auto log = std::make_shared<CrcLog>();
    holoscan::Graph graph { "crc_pairing_test" };
    auto source = graph.op<ChunkSourceOp>("source", distinct_frames(6), std::make_shared<SourceStats>());
    auto csi = graph.op<hololink::operators::CsiToBayerOp>("csi_to_bayer");
    csi.set_params(NativeOperators::csi_to_bayer_params());
    auto compute_crc = graph.op<hololink::operators::ComputeCrcImageOp>("compute_crc", channel, 0, kMaxCrcBytes);
    auto ip = graph.op<hololink::operators::ImageProcessorOp>("image_processor");
    ip.set_params(hololink::operators::ImageProcessorParams {
        .pixel_format = static_cast<int>(hololink::csi::PixelFormat::RAW_10),
        .bayer_format = static_cast<int>(hololink::csi::BayerFormat::RGGB),
        .optical_black = 0,
        .cuda_device_ordinal = 0,
    });
    auto sink = graph.op<ImageSinkOp<true>>("sink");
    auto check_crc = graph.op<RecordingCheckCrcOp<hololink::operators::CheckCrcOp>>("check_crc", channel, log);
    const holoscan::ConnectionOptions edge { .queue_depth = 4U };
    graph.add_flow(source->tensor, csi->input_tensor, edge);
    graph.add_flow(source->metadata, csi->input_metadata, edge);
    graph.add_flow(csi->output_tensor, compute_crc->input_tensor, edge);
    graph.add_flow(csi->output_metadata, compute_crc->input_metadata, edge);
    graph.add_flow(compute_crc->output_tensor, ip->input_tensor, edge);
    graph.add_flow(compute_crc->output_metadata, ip->input_metadata, edge);
    graph.add_flow(ip->output_tensor, sink->image, edge);
    graph.add_flow(ip->output_metadata, sink->metadata, edge);
    graph.add_flow(sink->forwarded_metadata, check_crc->input_metadata, edge);
    const FrameCrcs crcs = run(graph, { "source", "csi_to_bayer", "image_processor" }, *log);

    ASSERT_EQ(crcs.size(), 6U);
    for (uint32_t frame = 0; frame < 6; ++frame) {
        SCOPED_TRACE("frame " + std::to_string(frame));
        const std::vector<uint16_t> pixels = expected_bayer(kFrame, frame);
        EXPECT_EQ(crcs[frame].first, int64_t(frame));
        EXPECT_EQ(crcs[frame].second,
            jamcrc(reinterpret_cast<const uint8_t*>(pixels.data()), pixels.size() * sizeof(uint16_t)));
    }
}

#if defined(HOLOLINK_TEST_MODULE_OPERATORS) && defined(HOLOLINK_MODULE_BUILD_CRC)
TEST(CrcPairing, ModuleSlowCheckerSeesEachFramesCrc)
{
    expect_csi_frame_crcs(run_csi_frame_crcs<ModuleOperators>(12), 12);
}

// See FramesDroppedBeforeTheCheckerLeaveThePairingIntact.
TEST(CrcPairing, ModuleFramesDroppedBeforeTheCheckerLeaveThePairingIntact)
{
    expect_crcs_of_some_frames(run_csi_frame_crcs<ModuleOperators, true>(24), 24);
}
#endif

} // namespace hololink::tests
