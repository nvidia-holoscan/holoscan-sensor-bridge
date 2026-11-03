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

// Runs a converter chain the way the example players do, under Holoscan's managed shutdown
// signals:
//   ChunkSourceOp -> CsiToBayerOp -> ImageProcessorOp -> FrameCaptureOp
// with the native operators ("native") or the hololink_module ones ("module"). Both converters
// compile their kernels with NVRTC in start(); the run must still complete, because a changed
// SIGINT/SIGTERM disposition fails a managed run with RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED.
//
// No gtest: the signal domain has to be created before any other thread exists, and gtest_main
// can't promise that.

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <hololink/core/csi_controller.hpp>
#include <hololink/core/csi_formats.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
#include <hololink/module/csi_converter.hpp>
#include <hololink/module/operators/csi_to_bayer_op.hpp>
#include <hololink/module/operators/image_processor_op.hpp>
#endif

#include "synthetic_csi_source.hpp"

namespace {

using namespace hololink::tests::synthetic;

constexpr Geometry kGeometry { .sub_frame_rows = 0 };

struct NativeOperators {
    using CsiToBayer = hololink::operators::CsiToBayerOp;
    using ImageProcessor = hololink::operators::ImageProcessorOp;

    static hololink::operators::CsiToBayerParams csi_to_bayer_params(const Geometry& g)
    {
        return {
            .cuda_device_ordinal = 0,
            .geometry = hololink::csi::CsiGeometry {
                .start_byte = g.start_byte,
                .bytes_per_line = g.bytes_per_line(),
                .pixel_width = g.width,
                .pixel_height = g.height,
                .pixel_format = hololink::csi::PixelFormat::RAW_10,
                .trailing_bytes = g.trailing_bytes,
            },
            .bayer_format = hololink::csi::BayerFormat::RGGB,
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

    static hololink::module::operators::CsiToBayerParams csi_to_bayer_params(const Geometry& g)
    {
        return {
            .cuda_device_ordinal = 0,
            .geometry = hololink::module::csi::CsiGeometry {
                .start_byte = g.start_byte,
                .bytes_per_line = g.bytes_per_line(),
                .pixel_width = g.width,
                .pixel_height = g.height,
                .pixel_format = hololink::module::csi::PixelFormat::RAW_10,
                .trailing_bytes = g.trailing_bytes,
            },
            .bayer_format = hololink::module::csi::BayerFormat::RGGB,
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
int run_chain(holoscan::ManagedShutdownSignals& signals)
{
    SourceConfig config { .geometry = kGeometry, .frames = 3 };
    for (uint32_t frame = 0; frame < config.frames; ++frame) {
        config.csi_frames.push_back(csi_frame(kGeometry, frame));
    }
    auto source_config = std::make_shared<const SourceConfig>(std::move(config));
    auto captures = std::make_shared<Captures>();

    holoscan::Graph graph { "managed_shutdown_signals_test" };
    auto source = graph.op<ChunkSourceOp>("source", source_config, std::make_shared<SourceStats>());
    auto csi = graph.op<typename Operators::CsiToBayer>("csi_to_bayer");
    csi.set_params(Operators::csi_to_bayer_params(kGeometry));
    auto ip = graph.op<typename Operators::ImageProcessor>("image_processor");
    ip.set_params(Operators::image_processor_params());
    auto sink = graph.op<FrameCaptureOp>("sink", captures);
    const holoscan::ConnectionOptions edge { .queue_depth = 4U };
    graph.add_flow(source->tensor, csi->input_tensor, edge);
    graph.add_flow(source->metadata, csi->input_metadata, edge);
    graph.add_flow(csi->output_tensor, ip->input_tensor, edge);
    graph.add_flow(csi->output_metadata, ip->input_metadata, edge);
    graph.add_flow(ip->output_tensor, sink->image, edge);
    graph.add_flow(ip->output_metadata, sink->metadata, edge);
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
        std::cerr << "graph compilation failed" << std::endl;
        return 1;
    }

    auto session = holoscan::run_async(plan, signals);
    if (!session.wait_for(std::chrono::seconds(30))) {
        std::cerr << "the run did not finish" << std::endl;
        session.request_stop();
        session.wait();
        return 1;
    }
    for (const auto& diagnostic : session.diagnostics()) {
        std::cerr << "Session diagnostic: " << holoscan::format_diagnostic(diagnostic, plan.graph())
                  << std::endl;
    }
    if (session.termination() != holoscan::RunTermination::kCompleted) {
        std::cerr << "the run did not complete" << std::endl;
        return 1;
    }
    std::lock_guard lock(captures->mutex);
    if (captures->frames.size() != source_config->frames) {
        std::cerr << "captured " << captures->frames.size() << " of " << source_config->frames
                  << " frames" << std::endl;
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    auto created = holoscan::ManagedShutdownSignals::create();
    if (!created) {
        std::cerr << holoscan::runtime_error_message(created.error()) << std::endl;
        return 1;
    }
    auto signals = std::move(created).value();

    const std::string_view chain = argc > 1 ? argv[1] : "native";
    int result = 1;
    if (chain == "native") {
        result = run_chain<NativeOperators>(signals);
#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
    } else if (chain == "module") {
        result = run_chain<ModuleOperators>(signals);
#endif
    } else {
        std::cerr << "unknown chain " << chain << std::endl;
    }

    if (auto closed = signals.close(); !closed) {
        std::cerr << holoscan::runtime_error_message(closed.error()) << std::endl;
        return 1;
    }
    std::cout << chain << ": " << (result == 0 ? "PASSED" : "FAILED") << std::endl;
    return result;
}
