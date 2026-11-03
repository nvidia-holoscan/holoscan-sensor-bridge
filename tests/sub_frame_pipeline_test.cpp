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

// Hardware-free test of the sub-frame pipeline:
//   ChunkSourceOp -> CsiToBayerOp (sub-frame) -> [ImageProcessorOp ->] SubFrameCombinerOp ->
//     FrameCaptureOp
//
// ChunkSourceOp (synthetic_csi_source.hpp) stands in for a receiver in sub-frame mode: it
// delivers synthetic CSI-2 RAW10 frames (sensor header included) the way the HSB does, as
// consecutive chunks of bytes_per_line * sub_frame_rows bytes, each its own receiver frame with
// its own frame_number.
// The assembled frames are compared pixel for pixel with what CsiToBayerOp's unpacking makes of
// the source data. Chunks can be dropped, delivered short or delivered with a hole to check that
// a band missing data is withheld, and its frame discarded, rather than emitted with stale rows.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

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
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <capture_metadata_traits.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/csi_formats.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/packed_format_converter/packed_format_converter.hpp>
#include <hololink/operators/sub_frame_combiner/sub_frame_combiner.hpp>
#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
#include <hololink/module/csi_converter.hpp>
#include <hololink/module/operators/csi_to_bayer_op.hpp>
#include <hololink/module/operators/image_processor_op.hpp>
#include <hololink/module/operators/packed_format_converter_op.hpp>
#endif

#include "synthetic_csi_source.hpp"

namespace hololink::tests {

namespace {

    using namespace hololink::tests::synthetic;

    struct RunResult {
        std::vector<CapturedFrame> frames;
        holoscan::RunTermination termination = holoscan::RunTermination::kRunning;
        uint32_t source_dropped = 0;
        // The operators' error messages, when the run reported diagnostics.
        std::vector<std::string> errors;
        // The frame numbers the slow metadata branch received (PipelineOptions::slow_metadata_branch).
        std::vector<int64_t> branch_frames;
    };

    enum class Pipeline {
        kBands, // CsiToBayerOp -> capture: the bands (or full frames) themselves
        kFrames, // CsiToBayerOp -> SubFrameCombinerOp -> capture
        kFramesThroughImageProcessor, // ... -> ImageProcessorOp -> SubFrameCombinerOp -> capture
        kThroughImageProcessor, // CsiToBayerOp -> ImageProcessorOp -> capture
    };

    struct PipelineOptions {
        Pipeline pipeline = Pipeline::kFrames;
        std::optional<hololink::csi::BayerFormat> bayer_format = hololink::csi::BayerFormat::RGGB;
        // CsiToBayerParams/SubFrameCombinerParams::drop_incomplete
        bool drop_incomplete = false;
        // Whether the source delivers the CSI header (receivers) or strips it (FusaCoeCaptureOp).
        bool header_included = true;
        // Run hololink_module's CsiToBayerOp and ImageProcessorOp instead of the native ones.
        bool module_operators = false;
        // How long the capturing sink takes over each image (a consumer slower than the source).
        std::chrono::milliseconds sink_delay { 0 };
        // The run is meant to fail: its diagnostics go to RunResult::errors instead of failing
        // the test.
        bool expect_failure = false;
        // Runs the capturing sink in a partition of its own, fed through blocking one-deep queues.
        // (A linear chain runs in one partition, on one thread, where no queue ever fills.)
        bool sink_partition = false;
        // Also passes CsiToBayerOp's metadata to a metadata-only consumer that takes this long
        // over each sample, as a metadata printer next to the converter does, in a partition of
        // its own and through a one-deep queue that drops the oldest, as such consumers must be
        // connected (see emit_tensor_and_metadata()).
        std::optional<std::chrono::milliseconds> slow_metadata_branch;
        // The capturing sink holds on to every image it receives, and so to the pool slot behind
        // it, until the source reaches this chunk; then it lets them all go and stops holding on.
        std::optional<uint64_t> release_at_chunk;
    };

    // What the slow metadata branch received.
    struct BranchLog {
        std::mutex mutex;
        std::vector<int64_t> frame_numbers;
        std::chrono::milliseconds delay { 0 };
    };

    // Takes metadata alone, as a metadata printer does, and takes its time over each sample.
    class MetadataBranchOp : public holoscan::Operator<> {
    public:
        explicit MetadataBranchOp(std::shared_ptr<BranchLog> log)
            : log_(std::move(log))
        {
        }

        holoscan::Input<hololink::CaptureMetadataT> metadata { "metadata" };

        void setup(holoscan::OperatorSpec& spec) override { spec.input(metadata, "metadata"); }

        [[nodiscard]] holoscan::Contract contract() const override
        {
            holoscan::Contract contract;
            contract.trigger(holoscan::OnEach { metadata });
            return contract;
        }

        holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override
        {
            auto sample = metadata.receive();
            if (!sample) {
                return holoscan::make_unexpected(std::move(sample).error());
            }
            std::this_thread::sleep_for(log_->delay);
            std::lock_guard lock(log_->mutex);
            log_->frame_numbers.push_back(sample->data.frame_number);
            return {};
        }

    private:
        std::shared_ptr<BranchLog> log_;
    };

    // The converters under test: the native ones or hololink_module's.
    struct NativeOperators {
        using CsiToBayer = hololink::operators::CsiToBayerOp;
        using ImageProcessor = hololink::operators::ImageProcessorOp;

        static hololink::operators::CsiToBayerParams csi_to_bayer_params(
            const Geometry& g, const PipelineOptions& options)
        {
            return {
                .cuda_device_ordinal = 0,
                .sub_frame_rows = g.sub_frame_rows,
                .geometry = hololink::csi::CsiGeometry {
                    .start_byte = g.start_byte,
                    .bytes_per_line = g.bytes_per_line(),
                    .pixel_width = g.width,
                    .pixel_height = g.height,
                    .pixel_format = hololink::csi::PixelFormat::RAW_10,
                    .trailing_bytes = g.trailing_bytes,
                },
                .bayer_format = options.bayer_format,
                .header_included = options.header_included,
                .drop_incomplete = options.drop_incomplete,
            };
        }
        static hololink::operators::ImageProcessorParams image_processor_params()
        {
            return {
                .pixel_format = static_cast<int>(hololink::csi::PixelFormat::RAW_10),
                .bayer_format = static_cast<int>(hololink::csi::BayerFormat::RGGB),
                .optical_black = 0,
                .cuda_device_ordinal = 0,
            };
        }
    };

#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
    struct ModuleOperators {
        using CsiToBayer = hololink::module::operators::CsiToBayerOp;
        using ImageProcessor = hololink::module::operators::ImageProcessorOp;

        static hololink::module::operators::CsiToBayerParams csi_to_bayer_params(
            const Geometry& g, const PipelineOptions& options)
        {
            return {
                .cuda_device_ordinal = 0,
                .sub_frame_rows = g.sub_frame_rows,
                .geometry = hololink::module::csi::CsiGeometry {
                    .start_byte = g.start_byte,
                    .bytes_per_line = g.bytes_per_line(),
                    .pixel_width = g.width,
                    .pixel_height = g.height,
                    .pixel_format = hololink::module::csi::PixelFormat::RAW_10,
                    .trailing_bytes = g.trailing_bytes,
                },
                .bayer_format = options.bayer_format ? std::optional(static_cast<hololink::module::csi::BayerFormat>(*options.bayer_format)) : std::nullopt,
                .header_included = options.header_included,
                .drop_incomplete = options.drop_incomplete,
            };
        }
        static hololink::module::operators::ImageProcessorParams image_processor_params()
        {
            return {
                .pixel_format = static_cast<int>(hololink::module::csi::PixelFormat::RAW_10),
                .bayer_format = static_cast<int>(hololink::module::csi::BayerFormat::RGGB),
                .optical_black = 0,
                .cuda_device_ordinal = 0,
            };
        }
    };
#endif

    template <typename Operators>
    RunResult run_pipeline_with(SourceConfig config, const PipelineOptions& options)
    {
        for (uint32_t frame = 0; frame < config.frames; ++frame) {
            std::vector<uint8_t> bytes = csi_frame(config.geometry, frame);
            if (!options.header_included) {
                bytes.erase(bytes.begin(), bytes.begin() + config.geometry.start_byte);
            }
            config.csi_frames.push_back(std::move(bytes));
        }
        const Pipeline pipeline = options.pipeline;
        const Geometry g = config.geometry;
        auto stats = std::make_shared<SourceStats>();
        auto captures = std::make_shared<Captures>();
        captures->delay = options.sink_delay;
        if (options.release_at_chunk) {
            captures->retain = true;
            config.before_chunk = [captures, release = *options.release_at_chunk](uint64_t chunk) {
                if (chunk == release) {
                    captures->release_retained();
                }
            };
        }
        auto shared_config = std::make_shared<const SourceConfig>(std::move(config));
        auto branch_log = std::make_shared<BranchLog>();

        holoscan::Graph graph { "sub_frame_pipeline_test" };
        auto source = graph.op<ChunkSourceOp>("source", shared_config, stats);
        auto csi = graph.op<typename Operators::CsiToBayer>("csi_to_bayer");
        csi.set_params(Operators::csi_to_bayer_params(g, options));
        auto sink = graph.op<FrameCaptureOp>("sink", captures);

        // The chain runs in step with the source, so shallow queues suffice (and keep the 64 MiB
        // output pool slots of CsiToBayerOp and ImageProcessorOp few).
        const holoscan::ConnectionOptions chunk_edge { .queue_depth = 4U };
        const holoscan::ConnectionOptions frame_edge { .queue_depth = 4U };
        // A consumer in a partition of its own gets a blocking one-deep queue, full as soon as it
        // falls behind.
        const holoscan::ConnectionOptions one_deep_edge { .queue_depth = 1U };
        const holoscan::ConnectionOptions& sink_edge = options.sink_partition ? one_deep_edge : chunk_edge;
        graph.add_flow(source->tensor, csi->input_tensor, chunk_edge);
        graph.add_flow(source->metadata, csi->input_metadata, chunk_edge);
        std::vector<std::string> tensor_producers { "source", "csi_to_bayer" };
        const auto add_image_processor = [&]() {
            auto ip = graph.op<typename Operators::ImageProcessor>("image_processor");
            ip.set_params(Operators::image_processor_params());
            graph.add_flow(csi->output_tensor, ip->input_tensor, chunk_edge);
            graph.add_flow(csi->output_metadata, ip->input_metadata, chunk_edge);
            tensor_producers.emplace_back("image_processor");
            return ip;
        };
        if (pipeline == Pipeline::kBands) {
            graph.add_flow(csi->output_tensor, sink->image, sink_edge);
            graph.add_flow(csi->output_metadata, sink->metadata, sink_edge);
        } else if (pipeline == Pipeline::kThroughImageProcessor) {
            auto ip = add_image_processor();
            graph.add_flow(ip->output_tensor, sink->image, sink_edge);
            graph.add_flow(ip->output_metadata, sink->metadata, sink_edge);
        } else {
            auto combiner = graph.op<hololink::operators::SubFrameCombinerOp>("sub_frame_combiner");
            combiner.set_params(hololink::operators::SubFrameCombinerParams {
                .cuda_device_ordinal = 0,
                .height = g.height,
                .drop_incomplete = options.drop_incomplete,
            });
            tensor_producers.emplace_back("sub_frame_combiner");
            if (pipeline == Pipeline::kFramesThroughImageProcessor) {
                auto ip = add_image_processor();
                graph.add_flow(ip->output_tensor, combiner->input_tensor, chunk_edge);
                graph.add_flow(ip->output_metadata, combiner->input_metadata, chunk_edge);
            } else {
                graph.add_flow(csi->output_tensor, combiner->input_tensor, chunk_edge);
                graph.add_flow(csi->output_metadata, combiner->input_metadata, chunk_edge);
            }
            const holoscan::ConnectionOptions& combined_edge = options.sink_partition ? one_deep_edge : frame_edge;
            graph.add_flow(combiner->output_tensor, sink->image, combined_edge);
            graph.add_flow(combiner->output_metadata, sink->metadata, combined_edge);
        }
        if (options.slow_metadata_branch) {
            branch_log->delay = *options.slow_metadata_branch;
            auto branch = graph.op<MetadataBranchOp>("metadata_branch", branch_log);
            graph.add_flow(csi->output_metadata, branch->metadata,
                holoscan::ConnectionOptions { .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U });
            graph.partition("metadata_branch").add(branch);
        }
        if (options.sink_partition) {
            graph.partition("sink").add(sink);
        }
        if (options.sink_partition || options.slow_metadata_branch) {
            graph.auto_partition_remaining();
        }
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        holoscan::CompileOptions compile_options;
        for (const std::string& producer : tensor_producers) {
            compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
                .operator_path = producer,
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
        EXPECT_TRUE(session.wait_for(std::chrono::seconds(30))) << "session did not finish";
        RunResult result;
        for (const auto& diagnostic : session.diagnostics()) {
            if (options.expect_failure) {
                result.errors.push_back(diagnostic.detail ? diagnostic.detail->error.message : diagnostic.token);
            } else if (session.termination() != holoscan::RunTermination::kCompleted) {
                ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, holoscan::ExecutionGraphView {});
            }
        }

        result.termination = session.termination();
        result.source_dropped = stats->dropped.load();
        {
            std::lock_guard lock(branch_log->mutex);
            result.branch_frames = branch_log->frame_numbers;
        }
        std::lock_guard lock(captures->mutex);
        result.frames = std::move(captures->frames);
        return result;
    }

    RunResult run_pipeline(SourceConfig config, const PipelineOptions& options = {})
    {
#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
        if (options.module_operators) {
            return run_pipeline_with<ModuleOperators>(std::move(config), options);
        }
#endif
        return run_pipeline_with<NativeOperators>(std::move(config), options);
    }

    // Every captured frame is a whole sensor frame matching its source pixel for pixel, and the
    // frames that came out are exactly \p expected_frames, in order. Frame \p first_frame is the
    // source's first.
    void expect_exact_frames(const RunResult& result, const Geometry& g,
        const std::vector<int64_t>& expected_frames, int64_t first_frame = 0)
    {
        EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
        EXPECT_EQ(result.source_dropped, 0U) << "the source dropped chunks of its own";
        std::vector<int64_t> frame_numbers;
        for (const CapturedFrame& frame : result.frames) {
            frame_numbers.push_back(frame.frame_number);
        }
        ASSERT_EQ(frame_numbers, expected_frames);
        for (const CapturedFrame& frame : result.frames) {
            SCOPED_TRACE("frame " + std::to_string(frame.frame_number));
            EXPECT_EQ(frame.sub_frame_offset, 0);
            EXPECT_EQ(frame.sub_frame_rows, 0U);
            EXPECT_EQ(frame.image_width, static_cast<int32_t>(g.width));
            EXPECT_EQ(frame.image_height, static_cast<int32_t>(g.height));
            EXPECT_EQ(frame.encoding, holoscan::schema::ImageEncoding_BAYER_RGGB);
            EXPECT_EQ(frame.shape, (std::vector<int64_t> { g.height, g.width, 1 }));
            EXPECT_EQ(frame.dtype.code, kDLUInt);
            EXPECT_EQ(frame.dtype.bits, 16);
            const std::vector<uint16_t> expected = expected_bayer(g, static_cast<uint32_t>(frame.frame_number - first_frame));
            ASSERT_EQ(frame.bytes.size(), expected.size() * sizeof(uint16_t));
            std::vector<uint16_t> actual(expected.size());
            std::copy_n(frame.bytes.data(), frame.bytes.size(), reinterpret_cast<uint8_t*>(actual.data()));
            for (uint32_t row = 0; row < g.height; ++row) {
                const auto first = std::size_t(row) * g.width;
                // Whole frames (sub_frame_rows 0) have no bands to name.
                ASSERT_TRUE(std::equal(actual.begin() + first, actual.begin() + first + g.width,
                    expected.begin() + first))
                    << "row " << row << " differs"
                    << (g.sub_frame_rows ? " (band " + std::to_string(row / g.sub_frame_rows) + ")" : "");
            }
        }
    }

    // Every band CsiToBayerOp emitted holds exactly its rows of its frame, and the bands that came
    // out are exactly \p expected_bands ({frame, band index}), in order: a band whose data was
    // partly lost upstream must be withheld, not emitted with stale bytes.
    void expect_exact_bands(const RunResult& result, const Geometry& g,
        const std::vector<std::pair<int64_t, int64_t>>& expected_bands)
    {
        EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
        EXPECT_EQ(result.source_dropped, 0U) << "the source dropped chunks of its own";
        std::vector<std::pair<int64_t, int64_t>> bands;
        for (const CapturedFrame& band : result.frames) {
            bands.emplace_back(band.frame_number, band.sub_frame_offset / g.sub_frame_rows);
        }
        ASSERT_EQ(bands, expected_bands);
        for (const CapturedFrame& band : result.frames) {
            SCOPED_TRACE("frame " + std::to_string(band.frame_number) + " offset "
                + std::to_string(band.sub_frame_offset));
            EXPECT_EQ(band.sub_frame_rows, g.sub_frame_rows);
            ASSERT_EQ(band.sub_frame_offset % g.sub_frame_rows, 0);
            ASSERT_LE(band.sub_frame_offset + g.sub_frame_rows, g.height);
            ASSERT_EQ(band.shape, (std::vector<int64_t> { g.sub_frame_rows, g.width, 1 }));
            const std::vector<uint16_t> expected = expected_bayer(g, static_cast<uint32_t>(band.frame_number));
            const auto first = static_cast<std::size_t>(band.sub_frame_offset) * g.width;
            const std::size_t count = std::size_t(g.sub_frame_rows) * g.width;
            ASSERT_EQ(band.bytes.size(), count * sizeof(uint16_t));
            std::vector<uint16_t> actual(count);
            std::copy_n(band.bytes.data(), band.bytes.size(), reinterpret_cast<uint8_t*>(actual.data()));
            EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected.begin() + first));
        }
    }

    // All bands of \p frames frames, except those listed in \p missing.
    std::vector<std::pair<int64_t, int64_t>> all_bands_except(
        const Geometry& g, uint32_t frames, const std::set<std::pair<int64_t, int64_t>>& missing = {})
    {
        std::vector<std::pair<int64_t, int64_t>> bands;
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int64_t band = 0; band < g.height / g.sub_frame_rows; ++band) {
                if (!missing.count({ frame, band })) {
                    bands.emplace_back(frame, band);
                }
            }
        }
        return bands;
    }

    // The header (944 bytes) is shorter than a chunk (16 rows, 1536 bytes): every band straddles
    // two chunks.
    constexpr Geometry kShortHeader {};
    // The header is longer than a chunk (8 rows, 768 bytes): the first chunk is all header.
    constexpr Geometry kLongHeader { .sub_frame_rows = 8 };
    // No header: chunks and bands coincide.
    constexpr Geometry kNoHeader { .sub_frame_rows = 8, .start_byte = 0 };
    // Trailing bytes that add up to a whole band after the image's last row.
    constexpr Geometry kTrailingBand { .trailing_bytes = 96 * 16 };
    // Whole frames, as a receiver delivers them outside sub-frame mode.
    constexpr Geometry kFullFrame { .sub_frame_rows = 0 };

} // namespace

// --- Bands (CsiToBayerOp alone) --------------------------------------------------------------
// With the short header, chunk k of a frame holds the tail of band k - 1 and the head of band k
// (chunk 0: the header, then the head of band 0; the last chunk: the tail of the last band).

TEST(SubFramePipelineTest, CutsExactBandsWithShortHeader)
{
    expect_exact_bands(run_pipeline({ .geometry = kShortHeader }, { .pipeline = Pipeline::kBands }), kShortHeader,
        all_bands_except(kShortHeader, 3));
}

TEST(SubFramePipelineTest, CutsExactBandsWithLongHeader)
{
    expect_exact_bands(run_pipeline({ .geometry = kLongHeader }, { .pipeline = Pipeline::kBands }), kLongHeader,
        all_bands_except(kLongHeader, 3));
}

TEST(SubFramePipelineTest, CutsExactBandsWithoutHeader)
{
    expect_exact_bands(run_pipeline({ .geometry = kNoHeader }, { .pipeline = Pipeline::kBands }), kNoHeader,
        all_bands_except(kNoHeader, 3));
}

// Chunks are numbered across frames: chunk n + k is chunk k of frame 1. With drop_incomplete, a band
// part of whose data was lost upstream is withheld; by default it is passed on marked degraded
// (see the "Lost data" tests below).

TEST(SubFramePipelineTest, WithholdsBandMissingItsFirstChunk)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_exact_bands(run_pipeline({ .geometry = kShortHeader, .dropped_chunks = { n } }, { .pipeline = Pipeline::kBands, .drop_incomplete = true }),
        kShortHeader, all_bands_except(kShortHeader, 3, { { 1, 0 } }));
}

TEST(SubFramePipelineTest, WithholdsBandsMissingAMiddleChunk)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_exact_bands(run_pipeline({ .geometry = kShortHeader, .dropped_chunks = { n + 1 } }, { .pipeline = Pipeline::kBands, .drop_incomplete = true }),
        kShortHeader, all_bands_except(kShortHeader, 3, { { 1, 0 }, { 1, 1 } }));
}

TEST(SubFramePipelineTest, WithholdsBandMissingItsLastChunk)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_exact_bands(run_pipeline({ .geometry = kShortHeader, .dropped_chunks = { 2 * n - 1 } }, { .pipeline = Pipeline::kBands, .drop_incomplete = true }),
        kShortHeader, all_bands_except(kShortHeader, 3, { { 1, 2 } }));
}

TEST(SubFramePipelineTest, WithholdsBandsWithATruncatedChunk)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_exact_bands(run_pipeline({ .geometry = kShortHeader, .truncated_chunks = { n + 1 } }, { .pipeline = Pipeline::kBands, .drop_incomplete = true }),
        kShortHeader, all_bands_except(kShortHeader, 3, { { 1, 0 }, { 1, 1 } }));
}

TEST(SubFramePipelineTest, WithholdsBandsWithAHoledChunk)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_exact_bands(run_pipeline({ .geometry = kShortHeader, .holed_chunks = { n + 1 } }, { .pipeline = Pipeline::kBands, .drop_incomplete = true }),
        kShortHeader, all_bands_except(kShortHeader, 3, { { 1, 0 }, { 1, 1 } }));
}

TEST(SubFramePipelineTest, WithholdsOnlyTheHoledBandWithoutHeader)
{
    // Without a header a chunk is exactly one band, so a hole spoils that band alone.
    const uint64_t n = kNoHeader.chunks_per_frame();
    expect_exact_bands(run_pipeline({ .geometry = kNoHeader, .holed_chunks = { n + 1 } }, { .pipeline = Pipeline::kBands, .drop_incomplete = true }),
        kNoHeader, all_bands_except(kNoHeader, 3, { { 1, 1 } }));
}

TEST(SubFramePipelineTest, KeepsBandsWhenOnlyAHeaderChunkIsLost)
{
    // With the long header, the first chunk of a frame holds nothing but header.
    const uint64_t n = kLongHeader.chunks_per_frame();
    expect_exact_bands(run_pipeline({ .geometry = kLongHeader, .dropped_chunks = { n } }, { .pipeline = Pipeline::kBands }),
        kLongHeader, all_bands_except(kLongHeader, 3));
}

TEST(SubFramePipelineTest, SkipsABandOfTrailingBytes)
{
    // The trailing bytes fill a band past the image's last row: no image in it, so no band.
    expect_exact_bands(run_pipeline({ .geometry = kTrailingBand }, { .pipeline = Pipeline::kBands }), kTrailingBand,
        all_bands_except(kTrailingBand, 3));
}

// --- Frames (CsiToBayerOp -> SubFrameCombinerOp) ----------------------------------------------

TEST(SubFramePipelineTest, AssemblesFramesWithShortHeader)
{
    expect_exact_frames(run_pipeline({ .geometry = kShortHeader }), kShortHeader, { 0, 1, 2 });
}

TEST(SubFramePipelineTest, AssemblesFramesWithLongHeader)
{
    expect_exact_frames(run_pipeline({ .geometry = kLongHeader }), kLongHeader, { 0, 1, 2 });
}

TEST(SubFramePipelineTest, AssemblesFramesWithoutHeader)
{
    expect_exact_frames(run_pipeline({ .geometry = kNoHeader }), kNoHeader, { 0, 1, 2 });
}

TEST(SubFramePipelineTest, AssemblesFramesWithTrailingBytes)
{
    expect_exact_frames(run_pipeline({ .geometry = kTrailingBand }), kTrailingBand, { 0, 1, 2 });
}

TEST(SubFramePipelineTest, AssemblesFramesThroughImageProcessor)
{
    // The stream starts at frame 5, as a live one starts at whatever frame the sensor is on. The
    // first frame has no previous frame to take white-balance gains from, so it keeps unit gains
    // and (optical_black being 0) comes through unchanged. Later frames take their gains from the
    // frame before, so only their shape is checked.
    const uint64_t n = kShortHeader.chunks_per_frame();
    RunResult result = run_pipeline({ .geometry = kShortHeader, .first_chunk = 5 * n },
        { .pipeline = Pipeline::kFramesThroughImageProcessor });
    EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
    ASSERT_EQ(result.frames.size(), 3U);
    for (const CapturedFrame& frame : result.frames) {
        EXPECT_EQ(frame.shape, (std::vector<int64_t> { kShortHeader.height, kShortHeader.width, 1 }));
        EXPECT_EQ(frame.encoding, holoscan::schema::ImageEncoding_BAYER_RGGB);
        EXPECT_EQ(frame.sub_frame_offset, 0);
    }
    result.frames.resize(1);
    expect_exact_frames(result, kShortHeader, { 5 }, /*first_frame=*/5);
}

// With drop_incomplete, a frame with a band missing must be dropped, never shown with stale rows,
// and the frames around it must be unaffected.

TEST(SubFramePipelineTest, DiscardsFrameMissingItsFirstChunk)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_exact_frames(run_pipeline({ .geometry = kShortHeader, .dropped_chunks = { n } }, { .drop_incomplete = true }),
        kShortHeader, { 0, 2 });
}

TEST(SubFramePipelineTest, DiscardsFrameMissingAMiddleChunk)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_exact_frames(run_pipeline({ .geometry = kShortHeader, .dropped_chunks = { n + 1 } }, { .drop_incomplete = true }),
        kShortHeader, { 0, 2 });
}

TEST(SubFramePipelineTest, DiscardsFrameMissingItsLastChunk)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_exact_frames(run_pipeline({ .geometry = kShortHeader, .dropped_chunks = { 2 * n - 1 } }, { .drop_incomplete = true }),
        kShortHeader, { 0, 2 });
}

TEST(SubFramePipelineTest, DiscardsFrameWithATruncatedChunk)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_exact_frames(run_pipeline({ .geometry = kShortHeader, .truncated_chunks = { n + 1 } }, { .drop_incomplete = true }),
        kShortHeader, { 0, 2 });
}

TEST(SubFramePipelineTest, DiscardsFrameWithAHoledChunk)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_exact_frames(run_pipeline({ .geometry = kShortHeader, .holed_chunks = { n + 1 } }, { .drop_incomplete = true }),
        kShortHeader, { 0, 2 });
}

TEST(SubFramePipelineTest, KeepsFrameWhenOnlyAHeaderChunkIsLost)
{
    const uint64_t n = kLongHeader.chunks_per_frame();
    expect_exact_frames(run_pipeline({ .geometry = kLongHeader, .dropped_chunks = { n } }),
        kLongHeader, { 0, 1, 2 });
}

// --- Lost data, passed on (the default) ---------------------------------------------------------
// As in HSDK4, a band or frame part of whose data was lost upstream is still passed on; HSDK5 marks
// it kDegraded. Its lost rows hold stale content, so only the clean ones are compared exactly.

namespace {

    // The bands that came out are exactly \p expected ({frame, band index, degraded}), in order;
    // every band not marked degraded matches its source exactly.
    void expect_bands(const RunResult& result, const Geometry& g,
        const std::vector<std::tuple<int64_t, int64_t, bool>>& expected)
    {
        EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
        EXPECT_EQ(result.source_dropped, 0U);
        std::vector<std::tuple<int64_t, int64_t, bool>> bands;
        RunResult clean { .termination = result.termination };
        for (const CapturedFrame& band : result.frames) {
            bands.emplace_back(band.frame_number, band.sub_frame_offset / g.sub_frame_rows, band.degraded);
            if (!band.degraded) {
                clean.frames.push_back(band);
            }
        }
        ASSERT_EQ(bands, expected);
        std::vector<std::pair<int64_t, int64_t>> clean_bands;
        for (const auto& [frame, index, degraded] : expected) {
            if (!degraded) {
                clean_bands.emplace_back(frame, index);
            }
        }
        expect_exact_bands(clean, g, clean_bands);
    }

    // The frames that came out are exactly \p expected ({frame, degraded}), in order; every frame
    // not marked degraded matches its source exactly.
    void expect_frames(const RunResult& result, const Geometry& g,
        const std::vector<std::pair<int64_t, bool>>& expected)
    {
        EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
        std::vector<std::pair<int64_t, bool>> frames;
        RunResult clean { .termination = result.termination };
        std::vector<int64_t> clean_numbers;
        for (const CapturedFrame& frame : result.frames) {
            frames.emplace_back(frame.frame_number, frame.degraded);
            if (!frame.degraded) {
                clean.frames.push_back(frame);
                clean_numbers.push_back(frame.frame_number);
            }
        }
        ASSERT_EQ(frames, expected);
        expect_exact_frames(clean, g, clean_numbers);
    }

} // namespace

TEST(SubFramePipelineTest, PassesTheBandsOfAHoledChunkMarkedDegraded)
{
    // The hole spoils the band the chunk completes and the one it starts (the data itself is
    // intact here, so even those match).
    const uint64_t n = kShortHeader.chunks_per_frame();
    const RunResult result = run_pipeline(
        { .geometry = kShortHeader, .holed_chunks = { n + 1 } }, { .pipeline = Pipeline::kBands });
    expect_bands(result, kShortHeader,
        { { 0, 0, false }, { 0, 1, false }, { 0, 2, false }, { 1, 0, true }, { 1, 1, true },
            { 1, 2, false }, { 2, 0, false }, { 2, 1, false }, { 2, 2, false } });
}

TEST(SubFramePipelineTest, PassesTheBandAfterALostChunkMarkedDegraded)
{
    // The lost chunk held the tail of band 0, which never completes, and the head of band 1, which
    // is passed on with stale rows.
    const uint64_t n = kShortHeader.chunks_per_frame();
    const RunResult result = run_pipeline(
        { .geometry = kShortHeader, .dropped_chunks = { n + 1 } }, { .pipeline = Pipeline::kBands });
    expect_bands(result, kShortHeader,
        { { 0, 0, false }, { 0, 1, false }, { 0, 2, false }, { 1, 1, true }, { 1, 2, false },
            { 2, 0, false }, { 2, 1, false }, { 2, 2, false } });
}

TEST(SubFramePipelineTest, EmitsAFrameWithAHoledChunkMarkedDegraded)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_frames(run_pipeline({ .geometry = kShortHeader, .holed_chunks = { n + 1 } }), kShortHeader,
        { { 0, false }, { 1, true }, { 2, false } });
}

TEST(SubFramePipelineTest, EmitsAFrameMissingAMiddleChunkMarkedDegraded)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_frames(run_pipeline({ .geometry = kShortHeader, .dropped_chunks = { n + 1 } }), kShortHeader,
        { { 0, false }, { 1, true }, { 2, false } });
}

TEST(SubFramePipelineTest, EmitsAFrameMissingItsLastChunkMarkedDegraded)
{
    // Frame 1's last band never completes; the frame goes out when frame 2 starts, as in HSDK4.
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_frames(run_pipeline({ .geometry = kShortHeader, .dropped_chunks = { 2 * n - 1 } }), kShortHeader,
        { { 0, false }, { 1, true }, { 2, false } });
}

// --- Full frames (CsiToBayerOp outside sub-frame mode) ------------------------------------------

TEST(SubFramePipelineTest, ConvertsFullFramesWithTheHeader)
{
    expect_exact_frames(run_pipeline({ .geometry = kFullFrame }, { .pipeline = Pipeline::kBands }),
        kFullFrame, { 0, 1, 2 });
}

// FusaCoeCaptureOp strips the header before CsiToBayerOp sees the frame.
TEST(SubFramePipelineTest, ConvertsHeaderStrippedFullFrames)
{
    expect_exact_frames(run_pipeline({ .geometry = kFullFrame },
                            { .pipeline = Pipeline::kBands, .header_included = false }),
        kFullFrame, { 0, 1, 2 });
}

// A frame the HSB wrote only half of: its rows that arrived are converted, the rest are zero.
TEST(SubFramePipelineTest, PassesARuntFullFrameMarkedDegraded)
{
    const Geometry& g = kFullFrame;
    const RunResult result = run_pipeline({ .geometry = g, .truncated_chunks = { 1 } }, { .pipeline = Pipeline::kBands });
    expect_frames(result, g, { { 0, false }, { 1, true }, { 2, false } });
    ASSERT_EQ(result.frames.size(), 3U);
    const CapturedFrame& runt = result.frames[1];
    const std::size_t written = g.csi_length() / 2;
    const uint32_t valid_rows = static_cast<uint32_t>((written - g.start_byte - g.width * 5 / 4) / g.bytes_per_line() + 1);
    std::vector<uint16_t> expected = expected_bayer(g, 1);
    std::fill(expected.begin() + std::size_t(valid_rows) * g.width, expected.end(), uint16_t { 0 });
    std::vector<uint16_t> actual(expected.size());
    ASSERT_EQ(runt.bytes.size(), actual.size() * sizeof(uint16_t));
    std::copy_n(runt.bytes.data(), runt.bytes.size(), reinterpret_cast<uint8_t*>(actual.data()));
    EXPECT_EQ(actual, expected);
}

TEST(SubFramePipelineTest, DropsARuntFullFrameWhenAsked)
{
    expect_exact_frames(run_pipeline({ .geometry = kFullFrame, .truncated_chunks = { 1 } },
                            { .pipeline = Pipeline::kBands, .drop_incomplete = true }),
        kFullFrame, { 0, 2 });
}

TEST(SubFramePipelineTest, PassesAHoledFullFrameMarkedDegraded)
{
    expect_frames(run_pipeline({ .geometry = kFullFrame, .holed_chunks = { 1 } }, { .pipeline = Pipeline::kBands }),
        kFullFrame, { { 0, false }, { 1, true }, { 2, false } });
}

// --- A consumer slower than the source ------------------------------------------------------------
// A linear chain runs in one partition, on one thread: a slow sink holds the source back, so the
// chain slows down, no queue fills and nothing is lost. With the sink in a partition of its own,
// its one-deep queue fills instead: CsiToBayerOp's emit is then backpressured (SDK 5.0.0.7 first
// waits up to 5 ms for room), and the converter must drop that frame rather than fail the run.
// Frames are also dropped when CsiToBayerOp's output pool runs out (kResourceExhausted), here
// because the sink holds on to its frames. Either way, every frame that arrives must be intact and
// carry its own metadata. (The receivers' cases are in hololink_module_receiver_op_test and
// hololink_base_receiver_op_test.)

namespace {

    // Frames sent to a consumer that takes 50 ms over each: about 2-5 of them reach it, the rest
    // are dropped (immediately on SDK 5.0.0.5, after 5 ms on 5.0.0.7).
    constexpr uint32_t kSlowConsumerFrames = 24;
    constexpr std::chrono::milliseconds kSlowConsumerDelay { 50 };
    // Frames sent while the sink holds on to every image until the source reaches
    // kReleaseFrame: CsiToBayerOp's output pool has a few slots (its queue's depth, plus its own
    // loan and one being retired), far fewer than kReleaseFrame.
    constexpr uint32_t kPoolFrames = 32;
    constexpr uint64_t kReleaseFrame = 24;

    std::vector<int64_t> frame_numbers_of(const RunResult& result)
    {
        std::vector<int64_t> frame_numbers;
        for (const CapturedFrame& frame : result.frames) {
            frame_numbers.push_back(frame.frame_number);
        }
        return frame_numbers;
    }

    // Some but not all of the \p sent whole frames came out, in order: frames were dropped, and each
    // one that came out matches its source pixel for pixel and carries its own frame's metadata.
    void expect_some_intact_frames(const RunResult& result, const Geometry& g, uint32_t sent)
    {
        const std::vector<int64_t> frame_numbers = frame_numbers_of(result);
        ASSERT_FALSE(frame_numbers.empty()) << "no frame came through";
        EXPECT_LT(frame_numbers.size(), sent) << "no frame was dropped";
        EXPECT_TRUE(std::adjacent_find(frame_numbers.begin(), frame_numbers.end(), std::greater_equal<int64_t>())
            == frame_numbers.end())
            << "frames out of order or repeated";
        EXPECT_LT(frame_numbers.back(), int64_t(sent));
        // Also requires the run to complete and the source to have dropped nothing itself.
        expect_exact_frames(result, g, frame_numbers);
    }

    // The sink held on to every image until the source reached frame \p release: the frames before
    // it that found the pool empty were dropped, and once the slots came back every later frame came
    // through.
    void expect_frames_around_an_exhausted_pool(
        const RunResult& result, const Geometry& g, uint32_t sent, uint64_t release)
    {
        expect_some_intact_frames(result, g, sent);
        const std::vector<int64_t> frame_numbers = frame_numbers_of(result);
        const auto first_after = std::find_if(frame_numbers.begin(), frame_numbers.end(),
            [&](int64_t frame_number) { return frame_number >= int64_t(release); });
        const auto before = std::distance(frame_numbers.begin(), first_after);
        EXPECT_GT(before, 0) << "no frame came through before the pool ran out";
        EXPECT_LT(before, int64_t(release)) << "no frame was dropped while the pool was exhausted";
        std::vector<int64_t> expected_after(sent - release);
        std::iota(expected_after.begin(), expected_after.end(), int64_t(release));
        EXPECT_EQ(std::vector<int64_t>(first_after, frame_numbers.end()), expected_after)
            << "frames were lost after the pool's slots came back";
    }

    // The slow metadata branch fell behind: its queue dropped the oldest metadata, so it saw some
    // frames, in order, but not all, while the converter never waited for it. The run completed,
    // and every frame the sink paired is intact. (A blocking queue there would make the converter
    // drop the metadata of frames it already published, and SDK 5.0.0.5 then fails the run when
    // the stream ends.)
    void expect_a_lagging_metadata_branch(const RunResult& result, const Geometry& g, uint32_t sent)
    {
        EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
        ASSERT_FALSE(result.branch_frames.empty()) << "no metadata reached the branch";
        EXPECT_LT(result.branch_frames.size(), sent) << "the branch never fell behind";
        EXPECT_TRUE(std::adjacent_find(result.branch_frames.begin(), result.branch_frames.end(),
                        std::greater_equal<int64_t>())
            == result.branch_frames.end())
            << "branch frames out of order or repeated";
        const std::vector<int64_t> frame_numbers = frame_numbers_of(result);
        ASSERT_FALSE(frame_numbers.empty()) << "no frame reached the sink";
        expect_exact_frames(result, g, frame_numbers);
    }

} // namespace

TEST(SubFramePipelineTest, CompletesWithASlowConsumer)
{
    expect_exact_bands(run_pipeline({ .geometry = kShortHeader, .frames = 6 },
                           { .pipeline = Pipeline::kBands, .sink_delay = std::chrono::milliseconds(20) }),
        kShortHeader, all_bands_except(kShortHeader, 6));
}

TEST(SubFramePipelineTest, DropsFramesForASlowConsumerInAnotherPartition)
{
    expect_some_intact_frames(run_pipeline({ .geometry = kFullFrame, .frames = kSlowConsumerFrames },
                                  { .pipeline = Pipeline::kBands, .sink_delay = kSlowConsumerDelay, .sink_partition = true }),
        kFullFrame, kSlowConsumerFrames);
}

TEST(SubFramePipelineTest, KeepsPairingWhenASlowMetadataBranchFallsBehind)
{
    expect_a_lagging_metadata_branch(run_pipeline({ .geometry = kFullFrame, .frames = kSlowConsumerFrames },
                                         { .pipeline = Pipeline::kBands, .slow_metadata_branch = kSlowConsumerDelay }),
        kFullFrame, kSlowConsumerFrames);
}

TEST(SubFramePipelineTest, DropsFramesWhileItsOutputPoolIsExhausted)
{
    expect_frames_around_an_exhausted_pool(run_pipeline({ .geometry = kFullFrame, .frames = kPoolFrames },
                                               { .pipeline = Pipeline::kBands, .release_at_chunk = kReleaseFrame }),
        kFullFrame, kPoolFrames, kReleaseFrame);
}

// --- Frame identity ---------------------------------------------------------------------------

namespace {

    // A tensor arriving with another frame's metadata is dropped rather than converted under the
    // wrong frame's identity, and the run goes on: the other frames come through.
    void expect_a_mismatched_frame_to_be_dropped(bool module_operators)
    {
        const RunResult result = run_pipeline({ .geometry = kFullFrame, .misnumbered_chunks = { 1 } },
            { .pipeline = Pipeline::kBands, .module_operators = module_operators });
        EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
        std::vector<int64_t> frame_numbers;
        for (const CapturedFrame& frame : result.frames) {
            frame_numbers.push_back(frame.frame_number);
        }
        EXPECT_EQ(frame_numbers, (std::vector<int64_t> { 0, 2 }));
    }

} // namespace

#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
// --- hololink_module's CsiToBayerOp and ImageProcessorOp ------------------------------------------
// Same pipelines, same expectations as the native converters.

TEST(ModuleConverterTest, CutsExactBandsWithShortHeader)
{
    expect_exact_bands(run_pipeline({ .geometry = kShortHeader },
                           { .pipeline = Pipeline::kBands, .module_operators = true }),
        kShortHeader, all_bands_except(kShortHeader, 3));
}

TEST(ModuleConverterTest, CutsExactBandsWithLongHeader)
{
    expect_exact_bands(run_pipeline({ .geometry = kLongHeader },
                           { .pipeline = Pipeline::kBands, .module_operators = true }),
        kLongHeader, all_bands_except(kLongHeader, 3));
}

TEST(ModuleConverterTest, SkipsABandOfTrailingBytes)
{
    expect_exact_bands(run_pipeline({ .geometry = kTrailingBand },
                           { .pipeline = Pipeline::kBands, .module_operators = true }),
        kTrailingBand, all_bands_except(kTrailingBand, 3));
}

TEST(ModuleConverterTest, AssemblesFramesWithShortHeader)
{
    expect_exact_frames(run_pipeline({ .geometry = kShortHeader }, { .module_operators = true }), kShortHeader,
        { 0, 1, 2 });
}

TEST(ModuleConverterTest, AssemblesFramesThroughImageProcessor)
{
    // See SubFramePipelineTest.AssemblesFramesThroughImageProcessor: the first frame keeps unit
    // gains, so it comes through unchanged.
    const uint64_t n = kShortHeader.chunks_per_frame();
    RunResult result = run_pipeline({ .geometry = kShortHeader, .first_chunk = 5 * n },
        { .pipeline = Pipeline::kFramesThroughImageProcessor, .module_operators = true });
    EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
    ASSERT_EQ(result.frames.size(), 3U);
    result.frames.resize(1);
    expect_exact_frames(result, kShortHeader, { 5 }, /*first_frame=*/5);
}

TEST(ModuleConverterTest, PassesTheBandsOfAHoledChunkMarkedDegraded)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    const RunResult result = run_pipeline({ .geometry = kShortHeader, .holed_chunks = { n + 1 } },
        { .pipeline = Pipeline::kBands, .module_operators = true });
    expect_bands(result, kShortHeader,
        { { 0, 0, false }, { 0, 1, false }, { 0, 2, false }, { 1, 0, true }, { 1, 1, true },
            { 1, 2, false }, { 2, 0, false }, { 2, 1, false }, { 2, 2, false } });
}

TEST(ModuleConverterTest, WithholdsBandsWithAHoledChunk)
{
    const uint64_t n = kShortHeader.chunks_per_frame();
    expect_exact_bands(run_pipeline({ .geometry = kShortHeader, .holed_chunks = { n + 1 } },
                           { .pipeline = Pipeline::kBands, .drop_incomplete = true, .module_operators = true }),
        kShortHeader, all_bands_except(kShortHeader, 3, { { 1, 0 }, { 1, 1 } }));
}

TEST(ModuleConverterTest, ConvertsFullFramesWithAndWithoutTheHeader)
{
    expect_exact_frames(run_pipeline({ .geometry = kFullFrame },
                            { .pipeline = Pipeline::kBands, .module_operators = true }),
        kFullFrame, { 0, 1, 2 });
    expect_exact_frames(run_pipeline({ .geometry = kFullFrame },
                            { .pipeline = Pipeline::kBands, .header_included = false, .module_operators = true }),
        kFullFrame, { 0, 1, 2 });
}

TEST(ModuleConverterTest, PassesARuntFullFrameMarkedDegraded)
{
    expect_frames(run_pipeline({ .geometry = kFullFrame, .truncated_chunks = { 1 } },
                      { .pipeline = Pipeline::kBands, .module_operators = true }),
        kFullFrame, { { 0, false }, { 1, true }, { 2, false } });
}

TEST(ModuleConverterTest, CompletesWithASlowConsumer)
{
    expect_exact_bands(run_pipeline({ .geometry = kShortHeader, .frames = 6 },
                           { .pipeline = Pipeline::kBands,
                               .module_operators = true,
                               .sink_delay = std::chrono::milliseconds(20) }),
        kShortHeader, all_bands_except(kShortHeader, 6));
}

TEST(ModuleConverterTest, DropsFramesForASlowConsumerInAnotherPartition)
{
    expect_some_intact_frames(run_pipeline({ .geometry = kFullFrame, .frames = kSlowConsumerFrames },
                                  { .pipeline = Pipeline::kBands,
                                      .module_operators = true,
                                      .sink_delay = kSlowConsumerDelay,
                                      .sink_partition = true }),
        kFullFrame, kSlowConsumerFrames);
}

TEST(ModuleConverterTest, KeepsPairingWhenASlowMetadataBranchFallsBehind)
{
    expect_a_lagging_metadata_branch(run_pipeline({ .geometry = kFullFrame, .frames = kSlowConsumerFrames },
                                         { .pipeline = Pipeline::kBands,
                                             .module_operators = true,
                                             .slow_metadata_branch = kSlowConsumerDelay }),
        kFullFrame, kSlowConsumerFrames);
}

TEST(ModuleConverterTest, DropsFramesWhileItsOutputPoolIsExhausted)
{
    expect_frames_around_an_exhausted_pool(run_pipeline({ .geometry = kFullFrame, .frames = kPoolFrames },
                                               { .pipeline = Pipeline::kBands,
                                                   .module_operators = true,
                                                   .release_at_chunk = kReleaseFrame }),
        kFullFrame, kPoolFrames, kReleaseFrame);
}

TEST(ModuleConverterTest, DropsATensorPairedWithAnotherFramesMetadata)
{
    expect_a_mismatched_frame_to_be_dropped(true);
}
#endif

TEST(SubFramePipelineTest, DropsATensorPairedWithAnotherFramesMetadata)
{
    expect_a_mismatched_frame_to_be_dropped(false);
}

// --- Sample flags ----------------------------------------------------------------------------

// A flag the receiver set on a frame (here kDegraded) reaches the consumer through every operator,
// and frames without it stay unflagged.

namespace {

    void expect_the_degraded_flag_to_propagate(bool module_operators)
    {
        const RunResult result = run_pipeline(
            { .geometry = kFullFrame, .degraded_chunks = { 1 } },
            { .pipeline = Pipeline::kThroughImageProcessor, .module_operators = module_operators });
        EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
        std::vector<int64_t> frame_numbers;
        std::vector<bool> degraded;
        for (const CapturedFrame& frame : result.frames) {
            frame_numbers.push_back(frame.frame_number);
            degraded.push_back(frame.degraded);
        }
        EXPECT_EQ(frame_numbers, (std::vector<int64_t> { 0, 1, 2 }));
        EXPECT_EQ(degraded, (std::vector<bool> { false, true, false }));
    }

} // namespace

TEST(SubFramePipelineTest, PropagatesTheDegradedFlag)
{
    expect_the_degraded_flag_to_propagate(false);
}

#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
TEST(ModuleConverterTest, PropagatesTheDegradedFlag)
{
    expect_the_degraded_flag_to_propagate(true);
}
#endif

// --- PackedFormatConverterOp -----------------------------------------------------------------
// FusaCoeCaptureOp delivers the HSB packetizer's line format with the CSI header already
// stripped: each line packs its pixels (RAW_10: 3 per 4-byte word, RAW_12: 2 per 3 bytes) and is
// padded to 64 bytes. The converter must turn it into exactly what its kernels define, configured
// only through its Params (graph.op<>() can't reach a live instance to configure).

namespace {

    uint32_t packed_line_bytes(hololink::csi::PixelFormat format, uint32_t width)
    {
        switch (format) {
        case hololink::csi::PixelFormat::RAW_8:
            return width;
        case hololink::csi::PixelFormat::RAW_10:
            return (width + 2) / 3 * 4;
        default:
            return (width + 1) / 2 * 3;
        }
    }

    // What the converter's kernels make of a packed frame: [height, width] uint16.
    std::vector<uint16_t> expected_unpacked(const std::vector<uint8_t>& frame,
        hololink::csi::PixelFormat format, uint32_t width, uint32_t height, uint32_t bytes_per_line)
    {
        std::vector<uint16_t> out(std::size_t(width) * height);
        for (uint32_t row = 0; row < height; ++row) {
            const uint8_t* in = frame.data() + std::size_t(row) * bytes_per_line;
            uint16_t* o = out.data() + std::size_t(row) * width;
            for (uint32_t col = 0; col < width; ++col) {
                switch (format) {
                case hololink::csi::PixelFormat::RAW_8:
                    o[col] = uint16_t(in[col] << 8);
                    break;
                case hololink::csi::PixelFormat::RAW_10: {
                    const uint8_t* b = in + col / 3 * 4;
                    const uint32_t word = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16
                        | uint32_t(b[3]) << 24;
                    const uint32_t shifted = col % 3 == 0 ? word << 6 : col % 3 == 1 ? word >> 4
                                                                                     : word >> 14;
                    o[col] = uint16_t(shifted & 0xFFC0);
                    break;
                }
                default: {
                    const uint8_t* b = in + col / 2 * 3;
                    o[col] = col % 2 == 0 ? uint16_t(((b[1] & 0x0F) << 12 | b[0] << 4) & 0xFFF0)
                                          : uint16_t((b[2] << 8 | (b[1] & 0xF0)) & 0xFFF0);
                    break;
                }
                }
            }
        }
        return out;
    }

    struct PackedRun {
        RunResult result;
        std::vector<std::vector<uint8_t>> frames;
        uint32_t bytes_per_line = 0;
    };

    // The converter under test: the native one or hololink_module's, which has its own copies of
    // the kernels.
    struct NativePackedFormatConverter {
        using Op = hololink::operators::PackedFormatConverterOp;

        static hololink::operators::PackedFormatConverterParams params(
            std::optional<hololink::csi::BayerFormat> bayer_format,
            const std::optional<hololink::csi::CsiGeometry>& geometry)
        {
            hololink::operators::PackedFormatConverterParams params {
                .cuda_device_ordinal = 0,
                .bayer_format = bayer_format,
            };
            if (geometry) {
                params.geometry = *geometry;
            }
            return params;
        }
    };

#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
    struct ModulePackedFormatConverter {
        using Op = hololink::module::operators::PackedFormatConverterOp;

        // hololink_module's CSI enums carry the native ones' values.
        static hololink::module::operators::PackedFormatConverterParams params(
            std::optional<hololink::csi::BayerFormat> bayer_format,
            const std::optional<hololink::csi::CsiGeometry>& geometry)
        {
            hololink::module::operators::PackedFormatConverterParams params {
                .cuda_device_ordinal = 0,
                .bayer_format = bayer_format
                    ? std::optional(static_cast<hololink::module::csi::BayerFormat>(*bayer_format))
                    : std::nullopt,
            };
            if (geometry) {
                params.geometry = hololink::module::csi::CsiGeometry {
                    .start_byte = geometry->start_byte,
                    .bytes_per_line = geometry->bytes_per_line,
                    .pixel_width = geometry->pixel_width,
                    .pixel_height = geometry->pixel_height,
                    .pixel_format = static_cast<hololink::module::csi::PixelFormat>(geometry->pixel_format),
                    .trailing_bytes = geometry->trailing_bytes,
                };
            }
            return params;
        }
    };
#endif

    template <typename Converter = NativePackedFormatConverter>
    PackedRun run_packed_format_converter(hololink::csi::PixelFormat format, uint32_t width,
        uint32_t height, std::optional<hololink::csi::BayerFormat> bayer_format, bool configure = true,
        bool drop_last_row = false)
    {
        PackedRun run;
        run.bytes_per_line = (packed_line_bytes(format, width) + 63) & ~63u;
        SourceConfig config { .geometry = kFullFrame };
        for (uint32_t frame = 0; frame < config.frames; ++frame) {
            std::vector<uint8_t> bytes(std::size_t(run.bytes_per_line) * height);
            for (uint32_t row = 0; row < height; ++row) {
                for (uint32_t byte = 0; byte < run.bytes_per_line; ++byte) {
                    bytes[std::size_t(row) * run.bytes_per_line + byte] = pattern(frame, row, byte);
                }
            }
            if (drop_last_row) {
                bytes.resize(bytes.size() - run.bytes_per_line);
            }
            config.csi_frames.push_back(bytes);
        }
        run.frames = config.csi_frames;
        auto shared_config = std::make_shared<const SourceConfig>(std::move(config));
        auto stats = std::make_shared<SourceStats>();
        auto captures = std::make_shared<Captures>();

        holoscan::Graph graph { "packed_format_converter_test" };
        auto source = graph.op<ChunkSourceOp>("source", shared_config, stats);
        auto converter = graph.op<typename Converter::Op>("converter");
        std::optional<hololink::csi::CsiGeometry> geometry;
        if (configure) {
            geometry = hololink::csi::CsiGeometry {
                // Ignored: FusaCoeCaptureOp strips the header and the trailing bytes.
                .start_byte = 4096,
                .bytes_per_line = run.bytes_per_line,
                .pixel_width = width,
                .pixel_height = height,
                .pixel_format = format,
                .trailing_bytes = 128,
            };
        }
        converter.set_params(Converter::params(bayer_format, geometry));
        auto sink = graph.op<FrameCaptureOp>("sink", captures);
        const holoscan::ConnectionOptions edge { .queue_depth = 4U };
        graph.add_flow(source->tensor, converter->input_tensor, edge);
        graph.add_flow(source->metadata, converter->input_metadata, edge);
        graph.add_flow(converter->output_tensor, sink->image, edge);
        graph.add_flow(converter->output_metadata, sink->metadata, edge);
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        holoscan::CompileOptions options;
        for (const char* producer : { "source", "converter" }) {
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
            return run;
        }
        auto session = holoscan::run_async(plan);
        EXPECT_TRUE(session.wait_for(std::chrono::seconds(30))) << "session did not finish";
        run.result.termination = session.termination();
        run.result.source_dropped = stats->dropped.load();
        std::lock_guard lock(captures->mutex);
        run.result.frames = std::move(captures->frames);
        return run;
    }

    void expect_unpacked_frames(const PackedRun& run, hololink::csi::PixelFormat format,
        uint32_t width, uint32_t height, holoscan::schema::ImageEncoding encoding)
    {
        EXPECT_EQ(run.result.termination, holoscan::RunTermination::kCompleted);
        EXPECT_EQ(run.result.source_dropped, 0U);
        ASSERT_EQ(run.result.frames.size(), run.frames.size());
        for (std::size_t i = 0; i < run.result.frames.size(); ++i) {
            const CapturedFrame& frame = run.result.frames[i];
            SCOPED_TRACE("frame " + std::to_string(frame.frame_number));
            EXPECT_EQ(frame.frame_number, static_cast<int64_t>(i));
            EXPECT_EQ(frame.encoding, encoding);
            EXPECT_EQ(frame.shape, (std::vector<int64_t> { height, width, 1 }));
            EXPECT_EQ(frame.dtype.bits, 16);
            const std::vector<uint16_t> expected = expected_unpacked(run.frames[i], format, width,
                height, run.bytes_per_line);
            ASSERT_EQ(frame.bytes.size(), expected.size() * sizeof(uint16_t));
            std::vector<uint16_t> actual(expected.size());
            std::copy_n(frame.bytes.data(), frame.bytes.size(), reinterpret_cast<uint8_t*>(actual.data()));
            EXPECT_EQ(actual, expected);
        }
    }

} // namespace

TEST(PackedFormatConverterTest, ConvertsPackedRaw8)
{
    const auto run = run_packed_format_converter(
        hololink::csi::PixelFormat::RAW_8, 72, 8, hololink::csi::BayerFormat::RGGB);
    expect_unpacked_frames(run, hololink::csi::PixelFormat::RAW_8, 72, 8,
        holoscan::schema::ImageEncoding_BAYER_RGGB);
}

// 70 pixels: 23 whole words plus one holding a single pixel.
TEST(PackedFormatConverterTest, ConvertsPackedRaw10WithAPartialGroup)
{
    const auto run = run_packed_format_converter(
        hololink::csi::PixelFormat::RAW_10, 70, 8, hololink::csi::BayerFormat::GBRG);
    expect_unpacked_frames(run, hololink::csi::PixelFormat::RAW_10, 70, 8,
        holoscan::schema::ImageEncoding_BAYER_GBRG);
}

// 71 pixels: the last 3-byte group holds a single pixel.
TEST(PackedFormatConverterTest, ConvertsPackedRaw12WithAPartialGroup)
{
    const auto run = run_packed_format_converter(
        hololink::csi::PixelFormat::RAW_12, 71, 8, hololink::csi::BayerFormat::BGGR);
    expect_unpacked_frames(run, hololink::csi::PixelFormat::RAW_12, 71, 8,
        holoscan::schema::ImageEncoding_BAYER_BGGR);
}

// Without a Bayer format the CFA phase is left to the consumer.
TEST(PackedFormatConverterTest, StampsACustomEncodingWithoutABayerFormat)
{
    const auto run = run_packed_format_converter(hololink::csi::PixelFormat::RAW_10, 72, 8, std::nullopt);
    expect_unpacked_frames(run, hololink::csi::PixelFormat::RAW_10, 72, 8,
        holoscan::schema::ImageEncoding_CUSTOM);
}

TEST(PackedFormatConverterTest, RequiresTheGeometry)
{
    const auto run = run_packed_format_converter(hololink::csi::PixelFormat::RAW_10, 72, 8,
        hololink::csi::BayerFormat::RGGB, /*configure=*/false);
    EXPECT_EQ(run.result.termination, holoscan::RunTermination::kFailed);
    EXPECT_TRUE(run.result.frames.empty());
}

// A frame shorter than its geometry fails the run rather than being read past its end.
TEST(PackedFormatConverterTest, RejectsAFrameShorterThanItsGeometry)
{
    const auto run = run_packed_format_converter(hololink::csi::PixelFormat::RAW_10, 72, 8,
        hololink::csi::BayerFormat::RGGB, /*configure=*/true, /*drop_last_row=*/true);
    EXPECT_EQ(run.result.termination, holoscan::RunTermination::kFailed);
    EXPECT_TRUE(run.result.frames.empty());
}

#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
// hololink_module's PackedFormatConverterOp, the default (packetized) path of the module FuSa
// players: same frames, same expectations.

TEST(ModulePackedFormatConverterTest, ConvertsPackedRaw8)
{
    const auto run = run_packed_format_converter<ModulePackedFormatConverter>(
        hololink::csi::PixelFormat::RAW_8, 72, 8, hololink::csi::BayerFormat::RGGB);
    expect_unpacked_frames(run, hololink::csi::PixelFormat::RAW_8, 72, 8,
        holoscan::schema::ImageEncoding_BAYER_RGGB);
}

TEST(ModulePackedFormatConverterTest, ConvertsPackedRaw10WithAPartialGroup)
{
    const auto run = run_packed_format_converter<ModulePackedFormatConverter>(
        hololink::csi::PixelFormat::RAW_10, 70, 8, hololink::csi::BayerFormat::GBRG);
    expect_unpacked_frames(run, hololink::csi::PixelFormat::RAW_10, 70, 8,
        holoscan::schema::ImageEncoding_BAYER_GBRG);
}

TEST(ModulePackedFormatConverterTest, ConvertsPackedRaw12WithAPartialGroup)
{
    const auto run = run_packed_format_converter<ModulePackedFormatConverter>(
        hololink::csi::PixelFormat::RAW_12, 71, 8, hololink::csi::BayerFormat::BGGR);
    expect_unpacked_frames(run, hololink::csi::PixelFormat::RAW_12, 71, 8,
        holoscan::schema::ImageEncoding_BAYER_BGGR);
}

TEST(ModulePackedFormatConverterTest, StampsACustomEncodingWithoutABayerFormat)
{
    const auto run = run_packed_format_converter<ModulePackedFormatConverter>(
        hololink::csi::PixelFormat::RAW_10, 72, 8, std::nullopt);
    expect_unpacked_frames(run, hololink::csi::PixelFormat::RAW_10, 72, 8,
        holoscan::schema::ImageEncoding_CUSTOM);
}

TEST(ModulePackedFormatConverterTest, RequiresTheGeometry)
{
    const auto run = run_packed_format_converter<ModulePackedFormatConverter>(
        hololink::csi::PixelFormat::RAW_10, 72, 8, hololink::csi::BayerFormat::RGGB, /*configure=*/false);
    EXPECT_EQ(run.result.termination, holoscan::RunTermination::kFailed);
    EXPECT_TRUE(run.result.frames.empty());
}

TEST(ModulePackedFormatConverterTest, RejectsAFrameShorterThanItsGeometry)
{
    const auto run = run_packed_format_converter<ModulePackedFormatConverter>(
        hololink::csi::PixelFormat::RAW_10, 72, 8, hololink::csi::BayerFormat::RGGB,
        /*configure=*/true, /*drop_last_row=*/true);
    EXPECT_EQ(run.result.termination, holoscan::RunTermination::kFailed);
    EXPECT_TRUE(run.result.frames.empty());
}
#endif

// --- Image descriptors -----------------------------------------------------------------------

// Without a Bayer format, CsiToBayerOp leaves the CFA phase to the consumer (a demosaic then uses
// its own grid position) instead of guessing one.

namespace {

    void expect_a_custom_encoding_without_a_bayer_format(bool module_operators)
    {
        const RunResult result = run_pipeline({ .geometry = kFullFrame },
            { .pipeline = Pipeline::kBands, .bayer_format = std::nullopt, .module_operators = module_operators });
        EXPECT_EQ(result.termination, holoscan::RunTermination::kCompleted);
        ASSERT_EQ(result.frames.size(), 3U);
        for (const CapturedFrame& frame : result.frames) {
            EXPECT_EQ(frame.encoding, holoscan::schema::ImageEncoding_CUSTOM);
        }
    }

} // namespace

TEST(ImageDescriptorTest, CsiToBayerStampsACustomEncodingWithoutABayerFormat)
{
    expect_a_custom_encoding_without_a_bayer_format(false);
}

#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
TEST(ModuleImageDescriptorTest, CsiToBayerStampsACustomEncodingWithoutABayerFormat)
{
    expect_a_custom_encoding_without_a_bayer_format(true);
}
#endif

namespace {

    constexpr uint32_t kPaddedWidth = 64;
    constexpr uint32_t kPaddedHeight = 16;
    constexpr uint32_t kRowPitch = kPaddedWidth * 2 + 64; // bytes; 64 bytes of padding per row

    // All four samples of a 2x2 Bayer block share one value, so every channel has the same
    // histogram, ImageProcessorOp's white-balance gains are exactly 1 and (with optical_black 0)
    // the image comes through unchanged.
    uint16_t block_value(uint32_t frame, uint32_t row, uint32_t col)
    {
        return static_cast<uint16_t>(((frame * 131u + (row / 2) * 17u + (col / 2) * 5u + 1u) * 64u) & 0xFFC0u);
    }

    // Delivers whole 16-bit Bayer images with padded rows (byte strides [kRowPitch, 2, 2]) that a
    // plane layout describes, as a pitched producer would, plus their CaptureMetadataT. The
    // encoding is CUSTOM, the only one-plane encoding the schema lets declare its own layout.
    class PaddedImageSourceOp : public holoscan::Operator<> {
    public:
        holoscan::Output<holoscan::schema::ImageT> image { "tensor" };
        holoscan::Output<hololink::CaptureMetadataT> metadata { "metadata" };

        void setup(holoscan::OperatorSpec& spec) override
        {
            spec.output(image, "tensor")
                .max_emits_per_compute(1U)
                .tensor_field(holoscan::schema::ImageTensorFields::data)
                .produces_tensor(holoscan::TensorOutputSpec {
                    .representation = holoscan::TensorRepresentation {
                        .memory_kind = holoscan::MemoryKind::kCudaDevice,
                        .dtype = DLDataType { kDLUInt, 16, 1 },
                    },
                    .bounds = holoscan::TensorBounds { .max_rank = 3U, .max_byte_span = kMaxChunkBytes },
                    .storage = holoscan::TensorOutputStorage::kRuntimePool,
                });
            spec.output(metadata, "metadata").max_emits_per_compute(1U);
        }

        [[nodiscard]] holoscan::Contract contract() const override
        {
            holoscan::Contract contract;
            contract.trigger(holoscan::OnClock { .period = std::chrono::milliseconds(1) });
            contract.complete_after(holoscan::ComputeCalls { .limit = 2 });
            return contract;
        }

        static holoscan::unexpected<holoscan::Error> failed(const char* step, const holoscan::Error& error)
        {
            return holoscan::make_unexpected(holoscan::Error { error.code,
                std::string(step) + ": " + holoscan::runtime_error_message(error) });
        }

        holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override
        {
            const uint32_t frame = next_frame_++;
            std::vector<uint8_t> bytes(std::size_t(kRowPitch) * kPaddedHeight, 0xEE);
            for (uint32_t row = 0; row < kPaddedHeight; ++row) {
                auto* pixels = reinterpret_cast<uint16_t*>(bytes.data() + std::size_t(row) * kRowPitch);
                for (uint32_t col = 0; col < kPaddedWidth; ++col) {
                    pixels[col] = block_value(frame, row, col);
                }
            }
            const std::array<int64_t, 3> shape { kPaddedHeight, kPaddedWidth, 1 };
            const std::array<int64_t, 3> strides { kRowPitch, 2, 2 };
            auto loan = image.allocate_tensor(holoscan::TensorLoanRequest {
                .shape = shape, .dtype = DLDataType { kDLUInt, 16, 1 }, .byte_strides = holoscan::TensorByteStridesSpan { strides } });
            if (!loan) {
                return failed("allocate_tensor", loan.error());
            }
            auto writer = loan->write(context.cuda_stream());
            if (!writer) {
                return failed("write", writer.error());
            }
            // The tensor spans the last row's pixels, not its padding.
            const std::size_t span = std::size_t(kRowPitch) * (kPaddedHeight - 1) + kPaddedWidth * 2;
            if (cudaMemcpyAsync(writer->data(), bytes.data(), span, cudaMemcpyHostToDevice,
                    context.cuda_stream().get())
                != cudaSuccess) {
                return holoscan::make_unexpected(
                    holoscan::Error { holoscan::ErrorCode::kFailure, "cudaMemcpyAsync failed" });
            }
            auto committed = std::move(*writer).commit();
            if (!committed) {
                return failed("commit", committed.error());
            }
            holoscan::schema::ImageT descriptor;
            descriptor.height = kPaddedHeight;
            descriptor.width = kPaddedWidth;
            // A single-plane encoding must not carry plane layouts; CUSTOM may describe its own.
            descriptor.encoding = holoscan::schema::ImageEncoding_CUSTOM;
            descriptor.plane_layouts.emplace_back(0, kRowPitch, span);
            const holoscan::EmitOptions options { .capture_time = context.activation_time(), .frame_id = frame };
            auto emitted = image.emit_tensor(std::move(*loan), descriptor, options);
            if (!emitted) {
                return failed("emit_tensor", emitted.error());
            }
            hololink::CaptureMetadataT frame_metadata;
            frame_metadata.frame_number = frame;
            return metadata.emit(std::move(frame_metadata), options);
        }

    private:
        uint32_t next_frame_ = 0;
    };

    // ImageProcessorOp reads images whose rows are padded, writes them densely, and drops the
    // input's plane layouts, which would describe the wrong memory for its output (emit_tensor()
    // rejects a layout that reaches past the tensor).
    template <typename Operators>
    void expect_row_padded_images_to_be_read()
    {
        auto captures = std::make_shared<Captures>();
        holoscan::Graph graph { "padded_image_test" };
        auto source = graph.op<PaddedImageSourceOp>("source");
        auto ip = graph.op<typename Operators::ImageProcessor>("image_processor");
        ip.set_params(Operators::image_processor_params());
        auto sink = graph.op<FrameCaptureOp>("sink", captures);
        const holoscan::ConnectionOptions edge { .queue_depth = 4U };
        graph.add_flow(source->image, ip->input_tensor, edge);
        graph.add_flow(source->metadata, ip->input_metadata, edge);
        graph.add_flow(ip->output_tensor, sink->image, edge);
        graph.add_flow(ip->output_metadata, sink->metadata, edge);
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
        holoscan::CompileOptions options;
        for (const char* producer : { "source", "image_processor" }) {
            options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
                .operator_path = producer,
                .output_port = "tensor",
                .device = holoscan::DeviceId { 0 },
            });
        }
        auto plan = holoscan::compile(graph, std::move(options));
        if (!plan.ok()) {
            holoscan::log_diagnostics(plan);
            FAIL() << "graph compilation failed";
        }
        auto session = holoscan::run_async(plan);
        ASSERT_TRUE(session.wait_for(std::chrono::seconds(30)));
        for (const auto& diagnostic : session.diagnostics()) {
            ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
        }
        EXPECT_EQ(session.termination(), holoscan::RunTermination::kCompleted);
        std::lock_guard lock(captures->mutex);
        ASSERT_EQ(captures->frames.size(), 2U);
        for (const CapturedFrame& frame : captures->frames) {
            SCOPED_TRACE("frame " + std::to_string(frame.frame_number));
            EXPECT_EQ(frame.plane_layouts, 0U);
            EXPECT_EQ(frame.shape, (std::vector<int64_t> { kPaddedHeight, kPaddedWidth, 1 }));
            ASSERT_EQ(frame.bytes.size(), std::size_t(kPaddedWidth) * kPaddedHeight * 2);
            std::vector<uint16_t> actual(std::size_t(kPaddedWidth) * kPaddedHeight);
            std::copy_n(frame.bytes.data(), frame.bytes.size(), reinterpret_cast<uint8_t*>(actual.data()));
            std::vector<uint16_t> expected(actual.size());
            for (uint32_t row = 0; row < kPaddedHeight; ++row) {
                for (uint32_t col = 0; col < kPaddedWidth; ++col) {
                    expected[std::size_t(row) * kPaddedWidth + col]
                        = block_value(static_cast<uint32_t>(frame.frame_number), row, col);
                }
            }
            EXPECT_EQ(actual, expected);
        }
    }

} // namespace

TEST(ImageDescriptorTest, ImageProcessorReadsRowPaddedImages)
{
    expect_row_padded_images_to_be_read<NativeOperators>();
}

#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
TEST(ModuleImageDescriptorTest, ImageProcessorReadsRowPaddedImages)
{
    expect_row_padded_images_to_be_read<ModuleOperators>();
}
#endif

} // namespace hololink::tests
