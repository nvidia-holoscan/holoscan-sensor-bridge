/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Single-camera VB1940 player using the hololink_module API, ported to Holoscan SDK 5.0's
 * Graph / Operator<Params> shape. See examples/module_imx274_player.cpp for the identical
 * chain against Imx274Cam (this file's structural sibling: same operators, CLI mechanics,
 * and --frame-limit handling, only the sensor driver and optical-black constant differ).
 *
 *   RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
 */

#include <getopt.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

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

#include "hololink/module/cuda_unique.hpp"
#include "hololink/module/tools.hpp"

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/csi_to_bayer_op.hpp"
#include "hololink/module/operators/image_processor_op.hpp"
#include "hololink/module/operators/log_level.hpp"
#include "hololink/module/operators/roce_receiver_op.hpp"

#include "hololink/module/sensors/vb1940/vb1940_cam.hpp"

#include "player_support.hpp"

namespace hx = hololink::examples;

int main(int argc, char** argv)
{
    const std::string default_hololink_ip = hololink::module::env_hololink_ip(0, "192.168.0.2");
    auto camera_mode = hololink::module::sensors::vb1940::Vb1940_Mode::VB1940_MODE_2560X1984_30FPS;
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    std::string hololink_ip = default_hololink_ip;
    std::string module_dir;
    std::chrono::seconds discovery_timeout(30);
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
    bool log_level_given = false;

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 },
        { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "module-dir", required_argument, nullptr, 0 },
        { "discovery-timeout", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { 0, 0, nullptr, 0 },
    };

    while (true) {
        int option_index = 0;
        const int c = getopt_long(argc, argv, "h", long_options, &option_index);
        if (c == -1) {
            break;
        }
        const std::string argument(optarg ? optarg : "");
        if (c == 0) {
            const std::string name = long_options[option_index].name;
            if (name == "camera-mode") {
                camera_mode = static_cast<hololink::module::sensors::vb1940::Vb1940_Mode>(
                    std::stoi(argument));
            } else if (name == "headless") {
                headless = true;
            } else if (name == "fullscreen") {
                fullscreen = true;
            } else if (name == "frame-limit") {
                frame_limit = std::stoll(argument);
            } else if (name == "hololink") {
                hololink_ip = argument;
            } else if (name == "module-dir") {
                module_dir = argument;
            } else if (name == "discovery-timeout") {
                discovery_timeout = std::chrono::seconds(std::stoll(argument));
            } else if (name == "log-level") {
                log_level_given = true;
                if (argument == "trace" || argument == "TRACE") {
                    log_level = holoscan::LogLevel::TRACE;
                } else if (argument == "debug" || argument == "DEBUG") {
                    log_level = holoscan::LogLevel::DEBUG;
                } else if (argument == "info" || argument == "INFO") {
                    log_level = holoscan::LogLevel::INFO;
                } else if (argument == "warn" || argument == "WARN") {
                    log_level = holoscan::LogLevel::WARN;
                } else if (argument == "error" || argument == "ERROR") {
                    log_level = holoscan::LogLevel::ERROR;
                } else if (argument == "critical" || argument == "CRITICAL") {
                    log_level = holoscan::LogLevel::CRITICAL;
                } else if (argument == "off" || argument == "OFF") {
                    log_level = holoscan::LogLevel::OFF;
                } else {
                    throw std::runtime_error("Unhandled log level \"" + argument + "\"");
                }
            } else {
                throw std::runtime_error("Unhandled option \"" + name + "\"");
            }
        } else if (c == 'h') {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --hololink <ip>            IP of the Leopard VB1940 board (default "
                      << default_hololink_ip << ")\n"
                      << "  --module-dir <path>        Directory containing hololink_<UUID>.so\n"
                      << "  --camera-mode <int>        VB1940 mode (0=2560x1984 30fps default, "
                         "1=1920x1080 30fps, 2=2560x1984 30fps 8-bit, 3=2560x1984 60fps)\n"
                      << "  --frame-limit <int>        Exit after this many frames\n"
                      << "  --headless                 Run holoviz without a display\n"
                      << "  --fullscreen               Run holoviz fullscreen\n"
                      << "  --discovery-timeout <s>    Seconds to wait for the bootp announcement (default 30)\n"
                      << "  --log-level <level>        Holoscan log level\n";
            return EXIT_SUCCESS;
        } else {
            throw std::runtime_error("Unhandled option");
        }
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the camera and the board
    // are stopped on the way out. CUDA and the Adapter start threads, so the signals come first.
    return hx::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
        if (log_level_given) {
            // Without --log-level, the module's logger keeps the level HOLOLINK_LOG_LEVEL chose.
            hololink::module::operators::set_log_level(log_level);
        } else {
            holoscan::set_log_level(log_level);
        }
        std::cout << "Initializing." << std::endl;

        HOLOLINK_MODULE_CUDA_CHECK(cuInit(0));
        const int cu_device_ordinal = 0;
        CUdevice cu_device;
        HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGet(&cu_device, cu_device_ordinal));
        CUcontext cu_context;
        HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
        cuCtxSetCurrent(cu_context);

        auto& adapter = hololink::module::Adapter::get_adapter();
        if (!module_dir.empty()) {
            adapter.set_module_directory(std::filesystem::path(module_dir));
        }

        hololink::module::EnumerationMetadata metadata
            = adapter.wait_for_channel(hololink_ip, discovery_timeout);

        // Get the driver object.
        auto camera = std::make_shared<hololink::module::sensors::vb1940::Vb1940Cam>(metadata);
        // Connect to the HSB unit.
        auto hololink = hololink::module::HololinkInterfaceV1::get_service(metadata);
        if (hololink->start() != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("HololinkInterface::start failed");
        }
        // Drive the unit into a known state.
        if (hololink->reset() != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("HololinkInterface::reset failed");
        }
        // Configure the camera (walks the secure-boot FSM the first time after power-up, then
        // applies the per-mode register table).
        camera->configure(camera_mode);

        // Build the graph.
        holoscan::Graph graph { "module_vb1940_player" };

        hololink::module::csi::CsiGeometry geometry;
        camera->configure_converter(
            hololink::module::operators::CsiToBayerOp::make_geometry_adapter(geometry));
        const size_t frame_size = geometry.start_byte
            + geometry.bytes_per_line * geometry.pixel_height + geometry.trailing_bytes;

        const auto bayer_format = camera->bayer_format();
        const auto pixel_format = camera->pixel_format();

        holoscan::CompileOptions compile_options;
        holoscan::visualization::LayerStack layers;
        layers.add_image_layer("video", hx::rgba16_layer(camera->width(), camera->height()));

        auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers),
            hx::DisplayConfig { .fullscreen = fullscreen, .headless = headless }, compile_options,
            cu_device_ordinal);

        auto receiver = graph.op<hololink::module::operators::RoceReceiverOp>(
            "receiver",
            metadata,
            [camera] { camera->start(); },
            [camera] { camera->stop(); });
        receiver.set_params(hololink::module::operators::RoceReceiverParams {
            .frame_context = cu_context,
            .frame_size = frame_size,
            .frame_limit = frame_limit,
            .frame_limit_ends_run = true,
        });

        auto csi = graph.op<hololink::module::operators::CsiToBayerOp>("csi_to_bayer");
        csi.set_params(hololink::module::operators::CsiToBayerParams {
            .cuda_device_ordinal = cu_device_ordinal,
            .geometry = geometry,
            .bayer_format = bayer_format,
        });

        auto ip = graph.op<hololink::module::operators::ImageProcessorOp>("image_processor");
        ip.set_params(hololink::module::operators::ImageProcessorParams {
            .pixel_format = static_cast<int>(pixel_format),
            .bayer_format = static_cast<int>(bayer_format),
            // Optical black value for VB1940 is 8 (RAW10).
            .optical_black = 8,
            .cuda_device_ordinal = cu_device_ordinal,
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
                    .device = holoscan::DeviceId { cu_device_ordinal },
                });
        }
        compile_options.deployment.bind_tensor_output_device(
            holoscan::TensorOutputDevicePlacement {
                .operator_path = "bayer_demosaic",
                .output_port = "output",
                .device = holoscan::DeviceId { cu_device_ordinal },
            });

        const auto plan = holoscan::compile(graph, std::move(compile_options));
        if (!plan.ok()) {
            hx::print_compile_failure(plan);
            hololink->stop();
            HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
            return EXIT_FAILURE;
        }

        std::cout << "Calling run" << std::endl;
        auto session = holoscan::run_async(plan, signals);
        session.wait();
        const bool failed = hx::report_session(session, plan);

        // Stop streaming data and close up sockets.
        hololink->stop();
        HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
        return failed ? EXIT_FAILURE : EXIT_SUCCESS;
    });
}
