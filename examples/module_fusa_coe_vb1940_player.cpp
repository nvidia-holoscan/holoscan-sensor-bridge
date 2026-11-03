/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * VB1940 CoE capture player using the hololink_module API, ported to Holoscan SDK 5.0's
 * Graph / Operator<Params> shape.
 *
 * Module port of examples/fusa_coe_vb1940_player.cpp, following the same HSDK5 chain as
 * examples/module_fusa_coe_imx274_player.cpp (its HSB-Lite sibling -- see that file for the
 * N-layer ViewTransform pattern and the packetized-vs-unpacketized branch this mirrors
 * exactly): FusaCoeCaptureOp resolves CoeDataChannelInterfaceV1 + FrameMetadataInterfaceV1
 * from enumeration metadata; Vb1940Cam drives the sensor over the module V1 surface. Supports
 * mono (sensor 0 or 1) or stereo (sensor -1) -- native's fusa_coe_vb1940_player.cpp only
 * supports --sensor 0|1, so the stereo branch here (like the pre-port shell's own) is this
 * app's own already-generalized N-channel loop, not a copy of a native stereo precedent.
 *
 *   FusaCoeCaptureOp -> [PackedFormatConverterOp | CsiToBayerOp] -> ImageProcessorOp ->
 *     BayerDemosaicOp -> HolovizOp
 */

#include <getopt.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
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

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/mipi_dphy.hpp"
#include "hololink/module/networking.hpp"
#include "hololink/module/operators/csi_to_bayer_op.hpp"
#include "hololink/module/operators/fusa_coe_capture_op.hpp"
#include "hololink/module/operators/image_processor_op.hpp"
#include "hololink/module/operators/log_level.hpp"
#include "hololink/module/operators/packed_format_converter_op.hpp"
#include "hololink/module/tools.hpp"

#include "hololink/module/sensors/vb1940/vb1940_cam.hpp"

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

hololink::module::MacAddress parse_mac_address(const std::string& mac_str)
{
    hololink::module::MacAddress mac;
    std::istringstream iss(mac_str);
    std::string byte_str;
    int i = 0;

    while (std::getline(iss, byte_str, ':')) {
        if (i >= 6) {
            throw std::invalid_argument("Too many bytes in MAC address");
        }

        if (byte_str.size() != 2) {
            throw std::invalid_argument("Each byte must be exactly two hex digits");
        }

        uint16_t byte;
        std::istringstream hex_stream(byte_str);
        hex_stream >> std::hex >> byte;
        if (hex_stream.fail() || byte > 0xFF) {
            throw std::invalid_argument("Invalid hex in MAC address");
        }
        hex_stream >> std::ws;
        if (hex_stream.peek() != EOF) {
            throw std::invalid_argument("Invalid hex in MAC address");
        }

        mac[i++] = static_cast<uint8_t>(byte);
    }

    if (i != 6) {
        throw std::invalid_argument("MAC address must have exactly 6 bytes");
    }

    return mac;
}

} // namespace

static void print_usage(const char* argv0, const std::string& default_hololink_ip)
{
    std::cout << "Usage: " << argv0 << " [options]\n"
              << "  --hololink <ip>            IP of the Leopard VB1940 board (default "
              << default_hololink_ip << ")\n"
              << "  --module-dir <path>        Directory containing hololink_<UUID>.so\n"
              << "  --camera-mode <int>        VB1940 mode (0=2560x1984 30fps default, "
                 "1=1920x1080 30fps, 2=2560x1984 30fps 8-bit, 3=2560x1984 60fps)\n"
              << "  --sensor <int>             Sensor to use (0 or 1, or -1 for stereo)\n"
              << "  --timeout <ms>             CoE capture timeout in milliseconds "
                 "(default 1500)\n"
              << "  --frame-limit <int>        Exit after this many frames\n"
              << "  --headless                 Run holoviz without a display\n"
              << "  --fullscreen               Run holoviz fullscreen\n"
              << "  --no-packetizer            Capture unpacketized CSI-2 data instead of "
                 "HSB-packed data\n"
              << "  --discovery-timeout <s>    Seconds to wait for bootp (default 30)\n"
              << "  --log-level <level>        Holoscan log level\n";
}

int main(int argc, char** argv)
{
    const std::string default_hololink_ip = hololink::module::env_hololink_ip(0, "192.168.0.2");
    auto camera_mode = hololink::module::sensors::vb1940::Vb1940_Mode::VB1940_MODE_2560X1984_30FPS;
    bool headless = false;
    bool fullscreen = false;
    bool no_packetizer = false;
    int64_t frame_limit = 0;
    int32_t sensor = -1;
    uint32_t timeout = 1500;
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
        { "no-packetizer", no_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "sensor", required_argument, nullptr, 0 },
        { "timeout", required_argument, nullptr, 0 },
        { "module-dir", required_argument, nullptr, 0 },
        { "discovery-timeout", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { 0, 0, nullptr, 0 },
    };

    try {
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
                } else if (name == "no-packetizer") {
                    no_packetizer = true;
                } else if (name == "frame-limit") {
                    frame_limit = std::stoll(argument);
                } else if (name == "hololink") {
                    hololink_ip = argument;
                } else if (name == "sensor") {
                    sensor = std::stoi(argument);
                } else if (name == "timeout") {
                    timeout = static_cast<uint32_t>(std::stoll(argument));
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
                print_usage(argv[0], default_hololink_ip);
                return EXIT_SUCCESS;
            } else {
                throw std::runtime_error("Unhandled option");
            }
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0], default_hololink_ip);
        return EXIT_FAILURE;
    } catch (const std::out_of_range& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0], default_hololink_ip);
        return EXIT_FAILURE;
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0], default_hololink_ip);
        return EXIT_FAILURE;
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

        const int cu_device_ordinal = 0;
        std::shared_ptr<hololink::module::HololinkInterfaceV1> hololink;
        bool hololink_started = false;

        try {
            HOLOLINK_MODULE_CUDA_CHECK(cuInit(0));
            CUdevice cu_device;
            HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGet(&cu_device, cu_device_ordinal));
            CUcontext cu_context;
            HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
            cuCtxSetCurrent(cu_context);

            auto& adapter = hololink::module::Adapter::get_adapter();
            if (!module_dir.empty()) {
                adapter.set_module_directory(std::filesystem::path(module_dir));
            }

            const hololink::module::EnumerationMetadata base_metadata
                = adapter.wait_for_channel(hololink_ip, discovery_timeout);

            hololink = hololink::module::HololinkInterfaceV1::get_service(base_metadata);
            if (hololink->start() != HOLOLINK_MODULE_OK) {
                throw std::runtime_error("HololinkInterface::start failed");
            }
            hololink_started = true;
            if (hololink->reset() != HOLOLINK_MODULE_OK) {
                throw std::runtime_error("HololinkInterface::reset failed");
            }

            if ((sensor == -1) || (sensor == 0) || (sensor == 1)) {
                constexpr unsigned MIPI_LANE_COUNT = 4;
                const unsigned line_rate_mbps
                    = (camera_mode == hololink::module::sensors::vb1940::Vb1940_Mode::VB1940_MODE_2560X1984_60FPS)
                    ? 1140u
                    : 570u;
                auto dphy = hololink::module::MipiDphyInterfaceV1::get_service(base_metadata);
                // Currently, each SIF corresponds with the physical MIPI interface of the same index.
                const std::vector<unsigned> physical_mipi_interfaces = (sensor == -1)
                    ? std::vector<unsigned> { 0u, 1u }
                    : std::vector<unsigned> { static_cast<unsigned>(sensor) };
                for (unsigned physical_mipi_interface : physical_mipi_interfaces) {
                    const auto status = dphy->program(
                        physical_mipi_interface, MIPI_LANE_COUNT, line_rate_mbps);
                    if (status != HOLOLINK_MODULE_OK) {
                        throw std::runtime_error(fmt::format(
                            "Could not program MIPI interface {}: status {}",
                            physical_mipi_interface, status));
                    }
                }
            }

            struct Channel {
                int sensor_id = 0;
                hololink::module::EnumerationMetadata metadata;
                std::shared_ptr<hololink::module::sensors::vb1940::Vb1940Cam> camera;
            };
            auto make_channel = [&adapter, &base_metadata](int sensor_id) {
                Channel ch;
                ch.sensor_id = sensor_id;
                ch.metadata = base_metadata;
                adapter.use_sensor(ch.metadata, sensor_id);
                ch.camera = std::make_shared<hololink::module::sensors::vb1940::Vb1940Cam>(
                    ch.metadata);
                return ch;
            };

            std::vector<Channel> channels;
            if (sensor == -1) {
                channels.push_back(make_channel(0));
                channels.push_back(make_channel(1));
            } else if (sensor == 0 || sensor == 1) {
                channels.push_back(make_channel(sensor));
            } else {
                HOLOSCAN_LOG_ERROR("sensor value must be 0 or 1 (given {})", sensor);
                hololink->stop();
                return EXIT_FAILURE;
            }

            const auto& base_channel_metadata = channels[0].metadata;
            const std::string interface = base_channel_metadata.get<std::string>("interface");
            const std::string mac_str = base_channel_metadata.get<std::string>("mac_id");
            const auto mac_bytes_arr = parse_mac_address(mac_str);
            const std::vector<uint8_t> mac_bytes(mac_bytes_arr.begin(), mac_bytes_arr.end());
            HOLOSCAN_LOG_INFO("Using interface={}, mac={}", interface, mac_str);

            for (auto& ch : channels) {
                ch.camera->configure(camera_mode);
            }

            // Build the graph.
            holoscan::Graph graph { "module_fusa_coe_vb1940_player" };
            holoscan::CompileOptions compile_options;

            // Pass 1: declare one image layer per camera, side by side -- the LayerStack must be
            // complete before HolovizOp is constructed.
            holoscan::visualization::LayerStack layers;
            const float view_width = 1.0f / static_cast<float>(channels.size());
            for (std::size_t i = 0; i < channels.size(); ++i) {
                layers.add_image_layer(fmt::format("video_{}", i),
                    hx::rgba16_layer(channels[i].camera->width(), channels[i].camera->height(),
                        holoscan::visualization::ViewTransform {
                            .offset_x = view_width * static_cast<float>(i),
                            .offset_y = 0.f,
                            .scale_x = view_width,
                            .scale_y = 1.f,
                        }));
            }

            const uint32_t window_width = 800U * static_cast<uint32_t>(channels.size());
            const uint32_t window_height = 620U;
            // Renders once per image of the first camera (layer 0); with both sensors, the other layer
            // shows its newest image.
            auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers),
                hx::DisplayConfig {
                    .width = window_width,
                    .height = window_height,
                    .fullscreen = fullscreen,
                    .headless = headless,
                },
                compile_options, cu_device_ordinal);

            std::vector<std::string_view> tensor_output_operators;

            // Pass 2: build each camera's capture/convert/process/demosaic chain and wire it into viz.
            for (std::size_t i = 0; i < channels.size(); ++i) {
                // Named after the sensor, as in HSDK4: --sensor 1 runs capture_1, and its stall
                // messages say so.
                const std::string name = std::to_string(channels[i].sensor_id);
                auto camera = channels[i].camera;

                const auto pixel_format = camera->pixel_format();
                const auto bayer_format = camera->bayer_format();

                // The packetizer only has programs for RAW_10 and RAW_12, so an 8-bit mode is
                // always captured unpacketized.
                const bool packetizer_enabled = !no_packetizer
                    && pixel_format != hololink::module::csi::PixelFormat::RAW_8;

                // Resolve geometry ONCE using FusaCoeCaptureOp's HSB-alignment-aware adapter, and
                // reuse that same geometry for CsiToBayerParams/PackedFormatConverterParams below.
                hololink::module::csi::CsiGeometry geometry;
                camera->configure_converter(
                    hololink::module::operators::FusaCoeCaptureOp::make_geometry_adapter(
                        geometry, packetizer_enabled));

                auto capture = graph.op<hololink::module::operators::FusaCoeCaptureOp>(
                    fmt::format("capture_{}", name));
                capture.set_params(hololink::module::operators::FusaCoeCaptureParams {
                    .metadata = channels[i].metadata,
                    .device_start = [camera] { camera->start(); },
                    .device_stop = [camera] { camera->stop(); },
                    .interface = interface,
                    .mac_addr = mac_bytes,
                    .timeout = timeout,
                    // A camera that sends nothing fails a --frame-limit run instead of keeping it
                    // going.
                    .stall_timeout_ms = hx::frame_limit_stall_timeout_ms(frame_limit, timeout),
                    .pixel_format = pixel_format,
                    .bayer_format = bayer_format,
                    .packetizer_enabled = packetizer_enabled,
                    .geometry = geometry,
                    // Stops after --frame-limit frames, as HSDK4's CountCondition on each capture
                    // operator did.
                    .frame_limit = frame_limit,
                    .frame_limit_ends_run = true,
                    // Names this camera in the capture's stall messages.
                    .name = std::string(capture.name()),
                });

                auto image_processor = graph.op<hololink::module::operators::ImageProcessorOp>(
                    fmt::format("image_processor_{}", name));
                image_processor.set_params(hololink::module::operators::ImageProcessorParams {
                    .pixel_format = static_cast<int>(pixel_format),
                    .bayer_format = static_cast<int>(bayer_format),
                    // Optical black value for VB1940 over the CoE path is 0 -- a pre-existing
                    // per-app quirk vs. 8 on the plain RoCE VB1940 app; preserve exactly, not a
                    // bug to fix.
                    .optical_black = 0,
                    .cuda_device_ordinal = cu_device_ordinal,
                    .pixel_width = geometry.pixel_width,
                    .pixel_height = geometry.pixel_height,
                });

                const auto bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(bayer_format);
                auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>(
                    fmt::format("bayer_demosaic_{}", name));
                demosaic.set_params(holoscan::vision::BayerDemosaicOpParams {
                    .bayer_grid_pos = bayer_grid_pos,
                    .generate_alpha = true,
                    .alpha_value = 65535U,
                });

                if (packetizer_enabled) {
                    auto converter = graph.op<hololink::module::operators::PackedFormatConverterOp>(
                        fmt::format("packed_format_converter_{}", name));
                    converter.set_params(hololink::module::operators::PackedFormatConverterParams {
                        .cuda_device_ordinal = cu_device_ordinal,
                        .bayer_format = bayer_format,
                        .geometry = geometry,
                    });
                    graph.add_flow(capture->tensor, converter->input_tensor, hx::pipeline_edge());
                    graph.add_flow(capture->metadata, converter->input_metadata, hx::pipeline_edge());
                    graph.add_flow(converter->output_tensor, image_processor->input_tensor, hx::pipeline_edge());
                    graph.add_flow(converter->output_metadata, image_processor->input_metadata, hx::pipeline_edge());
                    tensor_output_operators.push_back(converter.name());
                } else {
                    auto converter = graph.op<hololink::module::operators::CsiToBayerOp>(
                        fmt::format("csi_to_bayer_{}", name));
                    converter.set_params(hololink::module::operators::CsiToBayerParams {
                        .cuda_device_ordinal = cu_device_ordinal,
                        .geometry = geometry,
                        .bayer_format = bayer_format,
                        // Fed by FusaCoeCaptureOp, which already strips the CSI header before
                        // emitting.
                        .header_included = false,
                    });
                    graph.add_flow(capture->tensor, converter->input_tensor, hx::pipeline_edge());
                    graph.add_flow(capture->metadata, converter->input_metadata, hx::pipeline_edge());
                    graph.add_flow(converter->output_tensor, image_processor->input_tensor, hx::pipeline_edge());
                    graph.add_flow(converter->output_metadata, image_processor->input_metadata, hx::pipeline_edge());
                    tensor_output_operators.push_back(converter.name());
                }

                graph.add_flow(image_processor->output_tensor, demosaic->input, hx::pipeline_edge());
                graph.add_flow(demosaic->output, viz->image_layer_input(i), hx::display_edge());

                tensor_output_operators.push_back(capture.name());
                tensor_output_operators.push_back(image_processor.name());
            }

            graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

            for (const auto& operator_path : tensor_output_operators) {
                compile_options.deployment.bind_tensor_output_device(
                    holoscan::TensorOutputDevicePlacement {
                        .operator_path = operator_path,
                        .output_port = "tensor",
                        .device = holoscan::DeviceId { cu_device_ordinal },
                    });
            }
            for (std::size_t i = 0; i < channels.size(); ++i) {
                compile_options.deployment.bind_tensor_output_device(
                    holoscan::TensorOutputDevicePlacement {
                        .operator_path = fmt::format("bayer_demosaic_{}", channels[i].sensor_id),
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

            auto session = holoscan::run_async(plan, signals);
            session.wait();
            const bool failed = hx::report_session(session, plan);

            hololink->stop();
            hololink_started = false;
            HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
            return failed ? EXIT_FAILURE : EXIT_SUCCESS;
        } catch (const std::exception& e) {
            if (hololink && hololink_started) {
                try {
                    hololink->stop();
                } catch (const std::exception& stop_error) {
                    HOLOSCAN_LOG_ERROR(
                        "HololinkInterface::stop failed after application error: {}",
                        stop_error.what());
                }
            }
            HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
            return EXIT_FAILURE;
        }
    });
}
