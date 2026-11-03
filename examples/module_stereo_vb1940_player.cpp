/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Stereo VB1940-AIO player using the hololink_module API, ported to Holoscan SDK 5.0's
 * Graph / Operator<Params> shape.
 *
 * Topology: one Leopard VB1940-AIO board hosts two cameras on a single HSB (sensor 0 = left,
 * sensor 1 = right). Both data planes share the board's HololinkInterface (lifecycle: start /
 * reset / stop runs once per board, not once per channel); each channel gets its own
 * Vb1940Cam pinned to a distinct I2C bus (Leopard wires camera 0 to bus 1, camera 1 to bus
 * 2). The Holoscan pipeline runs two parallel
 * RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp
 * legs into a single HolovizOp that shows the pair side-by-side (one window, two tiled
 * ViewTransform layers -- see examples/module_fusa_coe_imx274_player.cpp for the same
 * N-layer pattern).
 *
 * Defaults match examples/single_network_stereo_vb1940_player.cpp:
 * VB1940_MODE_2560X1984_30FPS (mode 0), one bootp-discovered IP for the HSB, 2-up window
 * split. Mirrors the structure of examples/module_vb1940_player.cpp and
 * examples/module_quad_imx274_player.cpp.
 *
 * Synchronized-shutter mode (PTP-PPS-driven VSYNC) is enabled by default -- the canonical
 * stereo configuration. VsyncStartOp (examples/vsync_start_op.hpp) fires vsync->start() once
 * every other operator's start() (and therefore every camera's bring-up) has completed, so the
 * first VSYNC pulse lands on both cameras simultaneously, and then completes, as the HSDK4
 * original did with a CountCondition(1).
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
#include <vector>

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
#include "hololink/module/ptp_pps_output.hpp"
#include "hololink/module/vsync.hpp"

#include "hololink/module/sensors/vb1940/vb1940_cam.hpp"

#include "player_support.hpp"
#include "vsync_start_op.hpp"

namespace hx = hololink::examples;

// Everything the application needs to drive one stereo channel end-to-end: the per-sensor
// enumeration metadata (one per call to Adapter::use_sensor) and the Vb1940Cam handle. Both
// channels in a stereo pair hold the same HololinkInterface (the locator returns one instance
// per board); the application brings the shared board up once and configures each camera
// independently.
struct Channel {
    hololink::module::EnumerationMetadata metadata;
    std::shared_ptr<hololink::module::sensors::vb1940::Vb1940Cam> camera;
    std::string tensor_name;
};

int main(int argc, char** argv)
{
    const std::string default_hololink_ip = hololink::module::env_hololink_ip(0, "192.168.0.2");
    auto camera_mode = hololink::module::sensors::vb1940::Vb1940_Mode::VB1940_MODE_2560X1984_30FPS;
    bool headless = false;
    int64_t frame_limit = 0;
    std::string hololink_ip = default_hololink_ip;
    std::string module_dir;
    uint32_t window_height = 2160 / 4;
    uint32_t window_width = 3840 / 3;
    std::string window_title;
    std::chrono::seconds discovery_timeout(30);
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
    bool log_level_given = false;
    // Synchronized-shutter mode (PTP-PPS-driven VSYNC) is enabled by default -- that is the
    // canonical stereo configuration. --no-sync leaves both cameras in free-run mode and
    // skips the ptp_pps_output enable; --sync-frequency overrides the default 30 Hz pulse
    // rate (the legacy stereo player's default).
    bool disable_sync = false;
    unsigned sync_frequency_hz = 30;

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 },
        { "headless", no_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "module-dir", required_argument, nullptr, 0 },
        { "discovery-timeout", required_argument, nullptr, 0 },
        { "window-height", required_argument, nullptr, 0 },
        { "window-width", required_argument, nullptr, 0 },
        { "title", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { "no-sync", no_argument, nullptr, 0 },
        { "sync-frequency", required_argument, nullptr, 0 },
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
            } else if (name == "frame-limit") {
                frame_limit = std::stoll(argument);
            } else if (name == "hololink") {
                hololink_ip = argument;
            } else if (name == "module-dir") {
                module_dir = argument;
            } else if (name == "discovery-timeout") {
                discovery_timeout = std::chrono::seconds(std::stoll(argument));
            } else if (name == "window-height") {
                window_height = static_cast<uint32_t>(std::stoul(argument));
            } else if (name == "window-width") {
                window_width = static_cast<uint32_t>(std::stoul(argument));
            } else if (name == "title") {
                window_title = argument;
            } else if (name == "no-sync") {
                disable_sync = true;
            } else if (name == "sync-frequency") {
                sync_frequency_hz = static_cast<unsigned>(std::stoul(argument));
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
                      << "  --hololink <ip>            IP of the Leopard VB1940-AIO board (default "
                      << default_hololink_ip << ")\n"
                      << "  --module-dir <path>        Directory containing hololink_<UUID>.so\n"
                      << "  --camera-mode <int>        VB1940 mode (0=2560x1984 30fps default, "
                         "1=1920x1080 30fps, 2=2560x1984 30fps 8-bit, 3=2560x1984 60fps)\n"
                      << "  --frame-limit <int>        Exit after this many frames per channel\n"
                      << "  --headless                 Run holoviz without a display\n"
                      << "  --window-height <int>      Display window height (default 540)\n"
                      << "  --window-width <int>       Display window width (default 1280)\n"
                      << "  --title <text>             Window title (default 'Holoviz - <ip>')\n"
                      << "  --discovery-timeout <s>    Seconds to wait for the bootp announcement (default 30)\n"
                      << "  --log-level <level>        Holoscan log level\n"
                      << "  --no-sync                  Run cameras in free-run; skip PTP-PPS VSYNC (default: sync on)\n"
                      << "  --sync-frequency <hz>      VSYNC pulse rate when sync is on (default 30; valid: 1, 10, 30, 60, 90, 120)\n";
            return EXIT_SUCCESS;
        } else {
            throw std::runtime_error("Unhandled option");
        }
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the cameras and the board
    // are stopped on the way out. CUDA and the Adapter start threads, so the signals come first.
    return hx::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
        if (log_level_given) {
            // Without --log-level, the module's logger keeps the level HOLOLINK_LOG_LEVEL chose.
            hololink::module::operators::set_log_level(log_level);
        } else {
            holoscan::set_log_level(log_level);
        }
        if (window_title.empty()) {
            window_title = "Holoviz - " + hololink_ip;
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

        // One bootp announcement covers both cameras on a VB1940-AIO HSB; Adapter::use_sensor
        // re-points a per-camera clone at the right sensor (and the supplement's
        // update_metadata stamps the per-sensor address fields).
        const hololink::module::EnumerationMetadata base_metadata
            = adapter.wait_for_channel(hololink_ip, discovery_timeout);

        // Both channels resolve to the same HololinkInterface instance (same serial). Bring the
        // shared board up once.
        auto hololink = hololink::module::HololinkInterfaceV1::get_service(base_metadata);
        if (hololink->start() != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("HololinkInterface::start failed");
        }
        if (hololink->reset() != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("HololinkInterface::reset failed");
        }

        std::shared_ptr<hololink::module::VsyncInterfaceV1> vsync = hololink->null_vsync();
        if (!disable_sync) {
            auto ptp_pps = hololink->ptp_pps_output();
            if (ptp_pps->enable(sync_frequency_hz) != HOLOLINK_MODULE_OK) {
                throw std::runtime_error("PtpPpsOutput::enable failed");
            }
            vsync = ptp_pps;
        }

        std::vector<Channel> channels;
        channels.reserve(2);
        {
            Channel left;
            left.metadata = base_metadata;
            adapter.use_sensor(left.metadata, /*sensor_number=*/0);
            left.camera = std::make_shared<hololink::module::sensors::vb1940::Vb1940Cam>(
                left.metadata, vsync);
            left.tensor_name = "left";
            channels.push_back(std::move(left));
        }
        {
            Channel right;
            right.metadata = base_metadata;
            adapter.use_sensor(right.metadata, /*sensor_number=*/1);
            right.camera = std::make_shared<hololink::module::sensors::vb1940::Vb1940Cam>(
                right.metadata, vsync);
            right.tensor_name = "right";
            channels.push_back(std::move(right));
        }

        // Vb1940Cam::configure is idempotent across the stereo pair -- shared board-level
        // bring-up runs once even though we call it for both cameras.
        for (Channel& ch : channels) {
            ch.camera->configure(camera_mode);
        }

        // Build the graph.
        holoscan::Graph graph { "module_stereo_vb1940_player" };
        holoscan::CompileOptions compile_options;

        // Both stereo cameras share width / height / formats (same mode).
        const auto bayer_format = channels[0].camera->bayer_format();
        const auto pixel_format = channels[0].camera->pixel_format();

        // Pass 1: declare one image layer per channel, side by side -- the LayerStack must be
        // complete before HolovizOp is constructed.
        holoscan::visualization::LayerStack layers;
        for (std::size_t i = 0; i < channels.size(); ++i) {
            layers.add_image_layer("video_" + channels[i].tensor_name,
                hx::rgba16_layer(channels[i].camera->width(), channels[i].camera->height(),
                    holoscan::visualization::ViewTransform {
                        // Side-by-side: left occupies x=[0, 0.5], right x=[0.5, 1].
                        .offset_x = 0.5f * static_cast<float>(i),
                        .offset_y = 0.f,
                        .scale_x = 0.5f,
                        .scale_y = 1.f,
                    }));
        }

        // Renders once per left image (layer 0); the right layer shows its newest image, which at a
        // low --sync-frequency can be the previous pulse's.
        auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers),
            hx::DisplayConfig {
                .title = window_title,
                .width = window_width,
                .height = window_height,
                .headless = headless,
                .max_input_age = hx::max_input_age_for_rate(disable_sync ? 0U : sync_frequency_hz),
            },
            compile_options, cu_device_ordinal);

        std::vector<std::string_view> tensor_output_operators;

        for (std::size_t i = 0; i < channels.size(); ++i) {
            Channel& ch = channels[i];

            hololink::module::csi::CsiGeometry geometry;
            ch.camera->configure_converter(
                hololink::module::operators::CsiToBayerOp::make_geometry_adapter(geometry));
            const size_t frame_size = geometry.start_byte
                + geometry.bytes_per_line * geometry.pixel_height + geometry.trailing_bytes;

            auto cam = ch.camera;
            auto receiver = graph.op<hololink::module::operators::RoceReceiverOp>(
                "receiver_" + ch.tensor_name,
                ch.metadata,
                [cam] { cam->start(); },
                [cam] { cam->stop(); });
            receiver.set_params(hololink::module::operators::RoceReceiverParams {
                .frame_context = cu_context,
                .frame_size = frame_size,
                // Stops after --frame-limit frames, as HSDK4's CountCondition on each receiver did.
                .frame_limit = frame_limit,
                .frame_limit_ends_run = true,
            });

            auto csi = graph.op<hololink::module::operators::CsiToBayerOp>(
                "csi_to_bayer_" + ch.tensor_name);
            csi.set_params(hololink::module::operators::CsiToBayerParams {
                .cuda_device_ordinal = cu_device_ordinal,
                .geometry = geometry,
                .bayer_format = bayer_format,
            });

            auto ip = graph.op<hololink::module::operators::ImageProcessorOp>(
                "image_processor_" + ch.tensor_name);
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
            auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>(
                "bayer_demosaic_" + ch.tensor_name);
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
            graph.add_flow(demosaic->output, viz->image_layer_input(i), hx::display_edge());

            tensor_output_operators.push_back(receiver.name());
            tensor_output_operators.push_back(csi.name());
            tensor_output_operators.push_back(ip.name());
        }

        // One-shot operator that fires vsync->start() after every operator's start() (and
        // therefore every camera's bring-up) has completed. Not wired into any add_flow call.
        auto vsync_starter = graph.op<VsyncStartOp>("vsync_start");
        vsync_starter.set_params(VsyncStartParams { .vsync = vsync });

        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        for (const auto& operator_path : tensor_output_operators) {
            compile_options.deployment.bind_tensor_output_device(
                holoscan::TensorOutputDevicePlacement {
                    .operator_path = operator_path,
                    .output_port = "tensor",
                    .device = holoscan::DeviceId { cu_device_ordinal },
                });
        }
        for (const auto& ch : channels) {
            compile_options.deployment.bind_tensor_output_device(
                holoscan::TensorOutputDevicePlacement {
                    .operator_path = "bayer_demosaic_" + ch.tensor_name,
                    .output_port = "output",
                    .device = holoscan::DeviceId { cu_device_ordinal },
                });
        }

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

        hololink->stop();
        HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
        return failed ? EXIT_FAILURE : EXIT_SUCCESS;
    });
}
