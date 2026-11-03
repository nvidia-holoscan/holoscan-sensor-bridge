/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Single-channel IMX274 player driving the hololink_module V1 surface through the SOFTWARE
 * (Linux) receiver, ported to Holoscan SDK 5.0's Graph / Operator<Params> shape. Identical to
 * examples/module_imx274_player.cpp except the receiver operator is
 * hololink::module::operators::LinuxReceiverOp instead of RoceReceiverOp -- so this example
 * runs on hosts with no infiniband device (HOLOLINK_BUILD_ROCE=OFF), reassembling HSB's
 * RoCEv2 UDP stream in user space.
 *
 * Discovery is bootp-driven (`Adapter::start_bootp_listener`); the example calls
 * `Adapter::wait_for_channel(peer_ip, timeout)` until the board with `--hololink <ip>`
 * announces itself, then builds the pipeline:
 *   LinuxReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
 *
 * BayerDemosaicOp demosaics ImageProcessorOp's 16-bit Bayer image into RGBA16 for display.
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
#include "hololink/module/operators/linux_receiver_op.hpp"
#include "hololink/module/operators/log_level.hpp"

#include "hololink/module/sensors/imx274/imx274_cam.hpp"
#include "hololink/module/sensors/imx274/imx274_mode.hpp"

#include "player_support.hpp"

namespace hx = hololink::examples;

int main(int argc, char** argv)
{
    const std::string default_hololink_ip = hololink::module::env_hololink_ip(0, "192.168.0.2");
    auto camera_mode = hololink::module::sensors::imx274::Imx274_Mode::IMX274_MODE_1920X1080_60FPS;
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    std::string hololink_ip = default_hololink_ip;
    std::string module_dir;
    uint32_t expander_configuration = 0;
    bool expander_configuration_set = false;
    uint8_t pattern = 0;
    bool pattern_set = false;
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
        { "expander-configuration", required_argument, nullptr, 0 },
        { "pattern", required_argument, nullptr, 0 },
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
                camera_mode = static_cast<hololink::module::sensors::imx274::Imx274_Mode>(
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
            } else if (name == "expander-configuration") {
                expander_configuration = std::stoul(argument);
                expander_configuration_set = true;
            } else if (name == "pattern") {
                pattern = std::stoul(argument);
                pattern_set = true;
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
                      << "  --hololink <ip>                  IP of the HSB-Lite board (default "
                      << default_hololink_ip << ")\n"
                      << "  --module-dir <path>              Directory containing hololink_<UUID>.so\n"
                      << "  --camera-mode <int>              IMX274 mode (1=1080p60 default, 0=4K60 RAW10, 2=4K60 RAW12)\n"
                      << "  --expander-configuration <0|1>   LI I2C expander output (which camera behind the bus)\n"
                      << "  --pattern <int>                  Enable the IMX274 test pattern with this value\n"
                      << "  --frame-limit <int>              Exit after this many frames\n"
                      << "  --headless                       Run holoviz without a display\n"
                      << "  --fullscreen                     Run holoviz fullscreen\n"
                      << "  --discovery-timeout <s>          Seconds to wait for the bootp announcement (default 30)\n"
                      << "  --log-level <level>              Holoscan log level\n";
            return EXIT_SUCCESS;
        } else {
            throw std::runtime_error("Unhandled option");
        }
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the camera and the board
    // are stopped on the way out.
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

        // If users requested a specific I2C expander configuration, use that.
        if (expander_configuration_set) {
            hololink::module::sensors::imx274::Imx274Cam::use_expander_configuration(
                metadata, expander_configuration);
        }
        // Get the driver object
        auto camera = std::make_shared<hololink::module::sensors::imx274::Imx274Cam>(metadata);
        // Connect to the HSB unit
        auto hololink = hololink::module::HololinkInterfaceV1::get_service(metadata);
        if (hololink->start() != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("HololinkInterface::start failed");
        }
        // Drive the unit into a known state
        if (hololink->reset() != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("HololinkInterface::reset failed");
        }
        // Configure the camera
        camera->configure(camera_mode);
        camera->set_digital_gain_reg(4);
        if (pattern_set) {
            camera->test_pattern_enable(pattern);
        }
        // Re-assert the mode so configure_converter() below reads freshly-updated
        // width_/height_/pixel_format_ state (mirrors the original HSDK4 compose()'s own
        // set_mode() call, made right before configure_converter() there too).
        camera->set_mode(camera_mode);

        // Build the graph.
        holoscan::Graph graph { "module_linux_imx274_player" };

        hololink::module::csi::CsiGeometry geometry;
        camera->configure_converter(
            hololink::module::operators::CsiToBayerOp::make_geometry_adapter(geometry));
        const size_t frame_size = geometry.start_byte
            + geometry.bytes_per_line * geometry.pixel_height + geometry.trailing_bytes;

        const auto bayer_format = camera->bayer_format();
        const auto pixel_format = camera->pixel_format();

        holoscan::visualization::LayerStack layers;
        layers.add_image_layer("video", hx::rgba16_layer(camera->width(), camera->height()));

        holoscan::CompileOptions compile_options;
        auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers),
            hx::DisplayConfig { .fullscreen = fullscreen, .headless = headless }, compile_options,
            cu_device_ordinal);

        auto receiver = graph.op<hololink::module::operators::LinuxReceiverOp>(
            "receiver",
            metadata,
            [camera] { camera->start(); },
            [camera] { camera->stop(); });
        receiver.set_params(hololink::module::operators::LinuxReceiverParams {
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
            // Optical black value for IMX274 is 50.
            .optical_black = 50,
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

        // Stop streaming data and close up sockets
        hololink->stop();
        HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
        return failed ? EXIT_FAILURE : EXIT_SUCCESS;
    });
}
