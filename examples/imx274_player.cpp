/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Single IMX274 live viewer over a single RoCE network:
//   RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp

#include <getopt.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
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
#include <hololink/core/logging.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

struct PlayerOptions {
    hololink::sensors::imx274_mode::Mode camera_mode = hololink::sensors::imx274_mode::IMX274_MODE_1920X1080_60FPS;
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    std::string hololink_ip;
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
    std::string ibv_name; // empty: the receiver uses the device on the board's network interface
    uint32_t ibv_port = 1;
    int32_t expander_configuration = 0;
    std::optional<int32_t> pattern;
};

int run_player(const PlayerOptions& options, holoscan::ManagedShutdownSignals& signals)
{
    // set the Holoscan log level
    holoscan::set_log_level(options.log_level);

    std::cout << "Initializing." << std::endl;

    // Get a handle to the GPU
    const int cuda_device_ordinal = 0;
    CudaCheck(cuInit(0));
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    cuCtxSetCurrent(cu_context);

    // Get a handle to the data source
    auto channel_metadata = hololink::Enumerator::find_channel(options.hololink_ip);
    auto hololink_channel = std::make_shared<hololink::DataChannel>(channel_metadata);

    // Get a handle to the camera
    auto camera = std::make_shared<hololink::sensors::NativeImx274Sensor>(
        *hololink_channel, options.expander_configuration);
    camera->set_mode(options.camera_mode);

    // Build the graph.
    holoscan::Graph graph { "imx274_player" };

    hololink::csi::CsiGeometry geometry;
    camera->configure_converter(geometry);
    const size_t frame_size = geometry.start_byte + geometry.bytes_per_line * geometry.pixel_height
        + geometry.trailing_bytes;

    holoscan::CompileOptions compile_options;
    holoscan::visualization::LayerStack layers;
    layers.add_image_layer("video",
        hx::rgba16_layer(static_cast<uint32_t>(camera->get_width()), static_cast<uint32_t>(camera->get_height())));

    auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers),
        hx::DisplayConfig { .fullscreen = options.fullscreen, .headless = options.headless }, compile_options,
        cuda_device_ordinal);

    auto receiver = graph.op<hololink::operators::RoceReceiverOp>(
        "receiver",
        hololink_channel,
        [camera] { camera->start(); },
        [camera] { camera->stop(); });
    receiver.set_params(hololink::operators::RoceReceiverParams {
        .frame_context = cu_context,
        .frame_size = frame_size,
        .ibv_name = options.ibv_name,
        .ibv_port = options.ibv_port,
        .frame_limit = options.frame_limit,
        .frame_limit_ends_run = true,
    });

    const auto bayer_format = camera->get_bayer_format();
    const auto pixel_format = camera->get_pixel_format();

    auto csi = graph.op<hololink::operators::CsiToBayerOp>("csi_to_bayer");
    csi.set_params(hololink::operators::CsiToBayerParams {
        .cuda_device_ordinal = cuda_device_ordinal,
        .geometry = geometry,
        .bayer_format = bayer_format,
    });
    auto ip = graph.op<hololink::operators::ImageProcessorOp>("image_processor");
    ip.set_params(hololink::operators::ImageProcessorParams {
        .pixel_format = int(pixel_format),
        .bayer_format = int(bayer_format),
        .optical_black = 50, // Optical black value for imx274 is 50
        .cuda_device_ordinal = cuda_device_ordinal,
        .pixel_width = geometry.pixel_width,
        .pixel_height = geometry.pixel_height,
    });

    const auto bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(bayer_format);
    auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>("bayer_demosaic");
    demosaic.set_params(holoscan::vision::BayerDemosaicOpParams {
        .bayer_grid_pos = bayer_grid_pos,
        .generate_alpha = true,
        .alpha_value = 65535U,
    });

    graph.add_flow(receiver->tensor, csi->input_tensor, hx::pipeline_edge());
    graph.add_flow(receiver->metadata, csi->input_metadata, hx::pipeline_edge());
    graph.add_flow(csi->output_tensor, ip->input_tensor, hx::pipeline_edge());
    graph.add_flow(csi->output_metadata, ip->input_metadata, hx::pipeline_edge());
    graph.add_flow(ip->output_tensor, demosaic->input, hx::pipeline_edge());
    graph.add_flow(demosaic->output, viz->image_layer_input(0), hx::display_edge());

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    for (const auto& operator_path : { std::string_view("receiver"), std::string_view("csi_to_bayer"), std::string_view("image_processor") }) {
        compile_options.deployment.bind_tensor_output_device(
            holoscan::TensorOutputDevicePlacement {
                .operator_path = operator_path,
                .output_port = "tensor",
                .device = holoscan::DeviceId { cuda_device_ordinal },
            });
    }
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement {
            .operator_path = "bayer_demosaic",
            .output_port = "output",
            .device = holoscan::DeviceId { cuda_device_ordinal },
        });

    const auto plan = holoscan::compile(graph, std::move(compile_options));
    if (!plan.ok()) {
        hx::print_compile_failure(plan);
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return -1;
    }

    // Run it.
    auto hololink_dev = hololink_channel->hololink();
    hololink_dev->start();
    hololink_dev->reset();
    camera->setup_clock();
    camera->configure(options.camera_mode);
    camera->set_digital_gain_reg(4);
    if (options.pattern) {
        camera->test_pattern(static_cast<uint8_t>(*options.pattern));
    }
    std::cout << "Calling run" << std::endl;
    auto session = holoscan::run_async(plan, signals);
    session.wait();
    const bool failed = hx::report_session(session, plan);

    hololink_dev->stop();
    CudaCheck(cuDevicePrimaryCtxRelease(cu_device));

    return failed ? -1 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    hololink::logging::hsb_log_level = hololink::logging::HSB_LOG_LEVEL_INFO;

    const std::string default_hololink_ip = hololink::env_hololink_ip(0, "192.168.0.2");
    PlayerOptions options { .hololink_ip = default_hololink_ip };

    // Parse args
    const struct option long_options[] = { { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 }, { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 }, { "frame-limit", required_argument, nullptr, 0 },
        { "configuration", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "ibv-name", required_argument, nullptr, 0 },
        { "ibv-port", required_argument, nullptr, 0 },
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
                options.camera_mode = static_cast<hololink::sensors::imx274_mode::Mode>(std::stoi(argument));
            } else if (cur_option->name == std::string("headless")) {
                options.headless = true;
            } else if (cur_option->name == std::string("fullscreen")) {
                options.fullscreen = true;
            } else if (cur_option->name == std::string("frame-limit")) {
                options.frame_limit = std::stoll(argument);
            } else if (cur_option->name == std::string("configuration")) {
                if (!hx::check_configuration(argument)) {
                    return -1;
                }
            } else if (cur_option->name == std::string("hololink")) {
                options.hololink_ip = argument;
            } else if (cur_option->name == std::string("log-level")) {
                const auto level = hx::parse_log_level(argument);
                if (!level) {
                    throw std::runtime_error(fmt::format("Unhandled log level \"{}\"", argument));
                }
                options.log_level = *level;
            } else if (cur_option->name == std::string("ibv-name")) {
                options.ibv_name = argument;
            } else if (cur_option->name == std::string("ibv-port")) {
                options.ibv_port = std::stoul(argument);
            } else if (cur_option->name == std::string("expander-configuration")) {
                options.expander_configuration = std::stoul(argument);
            } else if (cur_option->name == std::string("pattern")) {
                options.pattern = std::stoi(argument);
            } else {
                throw std::runtime_error(fmt::format("Unhandled option \"{}\"", cur_option->name));
            }
        } else {
            switch (c) {
            case 'h':
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                          << "Options:" << std::endl
                          << "  -h, --help     display this information" << std::endl
                          << "  --hololink     IP address of Hololink board (default `"
                          << default_hololink_ip << "`)" << std::endl
                          << std::endl;
                return EXIT_SUCCESS;

            default:
                throw std::runtime_error("Unhandled option ");
            }
        }
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the camera and the board
    // are stopped on the way out.
    return hx::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
        try {
            return run_player(options, signals);
        } catch (std::exception& e) {
            std::cout << e.what() << std::endl;
            return -1;
        }
    });
}
