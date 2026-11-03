/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// IMX274 live viewer over the Linux socket receiver:
//   LinuxReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp

#include <getopt.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/vision/bayer_demosaic_op.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/linux_receiver/linux_receiver_op.hpp>
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

struct PlayerOptions {
    int32_t camera_mode = 0;
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    std::string hololink_ip = hololink::env_hololink_ip(0, "192.168.0.2");
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
    int32_t expander_configuration = 0;
    int32_t pattern = 0;
    bool pattern_set = false;
};

int run_player(const PlayerOptions& options, holoscan::ManagedShutdownSignals& signals)
{
    // set the Holoscan log level
    holoscan::set_log_level(options.log_level);

    std::cout << "Initializing." << std::endl;

    // Get a handle to the GPU
    CudaCheck(cuInit(0));
    const int cu_device_ordinal = 0;
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    cuCtxSetCurrent(cu_context);

    // Get a handle to the data source
    hololink::Metadata channel_metadata = hololink::Enumerator::find_channel(options.hololink_ip);
    auto hololink_channel = std::make_shared<hololink::DataChannel>(channel_metadata);

    // Get a handle to the camera
    auto camera = std::make_shared<hololink::sensors::NativeImx274Sensor>(
        *hololink_channel, options.expander_configuration);
    const auto camera_mode = static_cast<hololink::sensors::imx274_mode::Mode>(options.camera_mode);
    camera->set_mode(camera_mode);
    const auto width = static_cast<uint32_t>(camera->get_width());
    const auto height = static_cast<uint32_t>(camera->get_height());
    const auto bayer_format = camera->get_bayer_format();
    const auto pixel_format = camera->get_pixel_format();

    // Build the graph.
    holoscan::Graph graph { "linux_imx274_player" };

    hololink::operators::CsiToBayerParams csi_params {
        .cuda_device_ordinal = cu_device_ordinal,
        .bayer_format = bayer_format,
    };
    camera->configure_converter(csi_params.geometry);
    const auto& geometry = csi_params.geometry;
    const size_t frame_size = geometry.start_byte
        + static_cast<size_t>(geometry.bytes_per_line) * geometry.pixel_height + geometry.trailing_bytes;
    auto csi = graph.op<hololink::operators::CsiToBayerOp>("csi_to_bayer");
    csi.set_params(csi_params);

    auto receiver = graph.op<hololink::operators::LinuxReceiverOp>(
        "receiver", hololink_channel, [camera] { camera->start(); }, [camera] { camera->stop(); });
    receiver.set_params(hololink::operators::LinuxReceiverParams {
        .frame_context = cu_context,
        .frame_size = frame_size,
        .frame_limit = options.frame_limit,
        .frame_limit_ends_run = true,
    });

    auto ip = graph.op<hololink::operators::ImageProcessorOp>("image_processor");
    ip.set_params(hololink::operators::ImageProcessorParams {
        .pixel_format = int(pixel_format),
        .bayer_format = int(bayer_format),
        .optical_black = 50, // Optical black value for imx274 is 50
        .cuda_device_ordinal = cu_device_ordinal,
        .pixel_width = geometry.pixel_width,
        .pixel_height = geometry.pixel_height,
    });

    auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>("demosaic");
    demosaic.set_params(holoscan::vision::BayerDemosaicOpParams {
        .bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(bayer_format),
        .generate_alpha = true,
        .alpha_value = 65535U,
    });

    holoscan::CompileOptions compile_options;
    holoscan::visualization::LayerStack layers;
    layers.add_image_layer("video", hx::rgba16_layer(width, height));
    auto visualizer = hx::add_holoviz(graph, "holoviz", std::move(layers),
        hx::DisplayConfig { .fullscreen = options.fullscreen, .headless = options.headless }, compile_options,
        cu_device_ordinal);

    const holoscan::ConnectionOptions edge = hx::pipeline_edge();
    graph.add_flow(receiver->tensor, csi->input_tensor, edge);
    graph.add_flow(receiver->metadata, csi->input_metadata, edge);
    graph.add_flow(csi->output_tensor, ip->input_tensor, edge);
    graph.add_flow(csi->output_metadata, ip->input_metadata, edge);
    graph.add_flow(ip->output_tensor, demosaic->input, edge);
    graph.add_flow(demosaic->output, visualizer->image_layer_input(0), hx::display_edge());

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    for (const auto& [operator_path, output_port] :
        { std::pair<std::string_view, std::string_view> { "receiver", "tensor" }, { "csi_to_bayer", "tensor" },
            { "image_processor", "tensor" }, { "demosaic", "output" } }) {
        compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = operator_path,
            .output_port = output_port,
            .device = holoscan::DeviceId { cu_device_ordinal },
        });
    }
    const auto plan = holoscan::compile(graph, std::move(compile_options));
    if (!plan.ok()) {
        hx::print_compile_failure(plan);
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return EXIT_FAILURE;
    }

    // Run it.
    std::shared_ptr<hololink::Hololink> hololink = hololink_channel->hololink();
    hololink->start();
    hololink->reset();
    camera->setup_clock();
    camera->configure(camera_mode);
    camera->set_digital_gain_reg(0x4);
    if (options.pattern_set) {
        camera->test_pattern(static_cast<uint8_t>(options.pattern));
    }
    std::cout << "Calling run" << std::endl;
    auto session = holoscan::run_async(plan, signals);
    session.wait();
    const bool failed = hx::report_session(session, plan);
    hololink->stop();

    CudaCheck(cuDevicePrimaryCtxRelease(cu_device));

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv)
{
    PlayerOptions options;

    // Parse args
    const struct option long_options[] = { { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 }, { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 }, { "frame-limit", required_argument, nullptr, 0 },
        { "configuration", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "expander-configuration", required_argument, nullptr, 0 },
        { "pattern", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 }, { 0, 0, nullptr, 0 } };
    while (true) {
        int option_index = 0;
        const int c = getopt_long(argc, argv, "h", long_options, &option_index);

        if (c == -1) {
            break;
        }

        const std::string argument(optarg ? optarg : "");
        if (c == 0) {
            const struct option* cur_option = &long_options[option_index];
            if (cur_option->name == std::string("camera-mode")) {
                options.camera_mode = std::stoi(argument);
            } else if (cur_option->name == std::string("headless")) {
                options.headless = true;
            } else if (cur_option->name == std::string("fullscreen")) {
                options.fullscreen = true;
            } else if (cur_option->name == std::string("frame-limit")) {
                options.frame_limit = std::stoll(argument);
            } else if (cur_option->name == std::string("configuration")) {
                if (!hx::check_configuration(argument)) {
                    return EXIT_FAILURE;
                }
            } else if (cur_option->name == std::string("hololink")) {
                options.hololink_ip = argument;
            } else if (cur_option->name == std::string("log-level")) {
                const auto level = hx::parse_log_level(argument);
                if (!level) {
                    throw std::runtime_error(fmt::format("Unhandled log level \"{}\"", argument));
                }
                options.log_level = *level;
            } else if (cur_option->name == std::string("expander-configuration")) {
                options.expander_configuration = std::stoul(argument);
            } else if (cur_option->name == std::string("pattern")) {
                options.pattern = std::stoi(argument);
                options.pattern_set = true;
            } else {
                throw std::runtime_error(fmt::format("Unhandled option \"{}\"", cur_option->name));
            }
        } else {
            switch (c) {
            case 'h':
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                          << "Options:" << std::endl
                          << "  -h, --help     display this information" << std::endl
                          << "  --hololink     IP address of Hololink board (default `" << options.hololink_ip
                          << "`)" << std::endl
                          << std::endl;
                return EXIT_SUCCESS;

            default:
                throw std::runtime_error("Unhandled option ");
            }
        }
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the camera and the board
    // are stopped on the way out.
    return hx::run_with_managed_signals(
        [&](holoscan::ManagedShutdownSignals& signals) { return run_player(options, signals); });
}
