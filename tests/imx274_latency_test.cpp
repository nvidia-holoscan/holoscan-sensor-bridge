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

// The latency tools' shared code (examples/imx274_latency.hpp): the report's settled-frame rule
// and its Python-compatible number format, and DisplayHandoffOp, which records each frame as it
// is handed to the display and completes the run at the frame limit. Needs a GPU.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>

#include "imx274_latency.hpp"
#include "synthetic_csi_source.hpp"

namespace hololink::tests {

namespace {

    namespace latency = hololink::examples::imx274_latency;
    using namespace hololink::tests::synthetic;

    std::vector<latency::FrameTimes> frames(std::size_t count)
    {
        std::vector<latency::FrameTimes> times(count);
        for (std::size_t i = 0; i < count; ++i) {
            const double start = 1000.0 + static_cast<double>(i) / 60.0;
            times[i] = latency::FrameTimes {
                .frame_number = static_cast<int64_t>(i),
                .frame_start_s = start,
                .frame_end_s = start + 0.016,
                .received_s = start + 0.0165,
                .operator_s = start + 0.017,
                .complete_s = start + 0.019,
            };
        }
        return times;
    }

} // namespace

// The report prints numbers the way the HSDK4 Python tool did (Python's repr of a float).
TEST(LatencyReport, FormatsNumbersLikePython)
{
    using latency::detail::python_float;
    EXPECT_EQ(python_float(1.0), "1.0");
    EXPECT_EQ(python_float(0.0), "0.0");
    EXPECT_EQ(python_float(0.0161), "0.0161");
    EXPECT_EQ(python_float(-0.0002), "-0.0002");
    EXPECT_EQ(python_float(1e-05), "1e-05");
    EXPECT_EQ(latency::detail::round4(0.016349), 0.0163);
}

// The report leaves out the first and last five frames, and needs 100 settled frames.
TEST(LatencyReport, NeedsAHundredSettledFrames)
{
    EXPECT_EQ(latency::report(frames(110)), EXIT_SUCCESS);
    EXPECT_EQ(latency::report(frames(109)), EXIT_FAILURE);
    EXPECT_EQ(latency::report(frames(10)), EXIT_FAILURE);
    EXPECT_EQ(latency::report({}), EXIT_FAILURE);
}

// DisplayHandoffOp pairs each image with its frame's metadata, records the frame once the GPU has
// finished the image, and completes the run when it has recorded frame_limit frames.
TEST(DisplayHandoff, RecordsPairedFramesAndCompletesAtItsLimit)
{
    constexpr Geometry kFrame { .sub_frame_rows = 0 };
    constexpr uint32_t kFrames = 8;
    constexpr int64_t kLimit = 5;

    SourceConfig config { .geometry = kFrame, .frames = kFrames };
    for (uint32_t frame = 0; frame < kFrames; ++frame) {
        config.csi_frames.push_back(csi_frame(kFrame, frame));
    }

    holoscan::Graph graph { "display_handoff_test" };
    auto source = graph.op<ChunkSourceOp>(
        "source", std::make_shared<const SourceConfig>(std::move(config)), std::make_shared<SourceStats>());
    auto csi = graph.op<hololink::operators::CsiToBayerOp>("csi_to_bayer");
    csi.set_params(hololink::operators::CsiToBayerParams {
        .geometry = hololink::csi::CsiGeometry {
            .start_byte = kFrame.start_byte,
            .bytes_per_line = kFrame.bytes_per_line(),
            .pixel_width = kFrame.width,
            .pixel_height = kFrame.height,
            .pixel_format = hololink::csi::PixelFormat::RAW_10,
        },
        .bayer_format = hololink::csi::BayerFormat::RGGB,
    });
    auto ip = graph.op<hololink::operators::ImageProcessorOp>("image_processor");
    ip.set_params(hololink::operators::ImageProcessorParams {
        .pixel_format = int(hololink::csi::PixelFormat::RAW_10),
        .bayer_format = int(hololink::csi::BayerFormat::RGGB),
        .optical_black = 50,
        .cuda_device_ordinal = 0,
    });
    auto recorder = std::make_shared<latency::LatencyRecorder>();
    auto handoff = graph.op<latency::DisplayHandoffOp>("display_handoff");
    handoff.set_params(latency::DisplayHandoffParams { .recorder = recorder, .frame_limit = kLimit });

    const holoscan::ConnectionOptions edge = hololink::examples::pipeline_edge();
    graph.add_flow(source->tensor, csi->input_tensor, edge);
    graph.add_flow(source->metadata, csi->input_metadata, edge);
    graph.add_flow(csi->output_tensor, ip->input_tensor, edge);
    graph.add_flow(csi->output_metadata, ip->input_metadata, edge);
    graph.add_flow(ip->output_tensor, handoff->image, edge);
    graph.add_flow(ip->output_metadata, handoff->metadata, edge);
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    for (const char* producer : { "source", "csi_to_bayer", "image_processor" }) {
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
    ASSERT_TRUE(session.wait_for(std::chrono::seconds(30))) << "session did not finish";
    for (const auto& diagnostic : session.diagnostics()) {
        ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
    }
    EXPECT_EQ(session.termination(), holoscan::RunTermination::kRequestedCompleted);

    std::vector<int64_t> frame_numbers;
    for (const latency::FrameTimes& times : recorder->frames()) {
        frame_numbers.push_back(times.frame_number);
        EXPECT_GT(times.complete_s, 0.0) << "frame " << times.frame_number;
    }
    EXPECT_EQ(frame_numbers, (std::vector<int64_t> { 0, 1, 2, 3, 4 }));
}

} // namespace hololink::tests
