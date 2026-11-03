/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * 4-channel IMX274 player driving the hololink_module V1 surface, ported to Holoscan SDK
 * 5.0's Graph / Operator<Params> shape.
 *
 * Topology: 4 independent HSB-Lite data planes (each its own IMX274 camera) flowing into a
 * single Holoscan pipeline:
 *   RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
 * with all four legs tiled into one window as a 2x2 grid (one HolovizOp, four ViewTransform
 * layers -- generalizes the N-layer pattern examples/module_fusa_coe_imx274_player.cpp/
 * module_stereo_vb1940_player.cpp already use for N=2 up to N=4; kMaxLayerPorts=8 has
 * headroom). Each camera negotiates its own CsiGeometry/make_geometry_adapter() and
 * HololinkInterface independently -- channels sharing a physical board (none do by default,
 * but nothing assumes they can't) dedup their start()/reset() by HololinkInterface pointer.
 *
 * Default IPs: 192.168.0.200 / .201 / .202 / .203 (devices 0..3). The example pins the
 * cameras to IMX274_MODE_1920X1080_60FPS (mode 1, 1080p RAW10) so the four-channel aggregate
 * fits in the data-plane bandwidth budget; the 4K modes over four channels exceed what the
 * link can sustain.
 */

#include <getopt.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
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
#include "hololink/module/operators/roce_receiver_op.hpp"

#include "hololink/module/sensors/imx274/imx274_cam.hpp"
#include "hololink/module/sensors/imx274/imx274_mode.hpp"

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

// Wait until every expected peer IP has been enumerated by the bootp listener. Returns the
// metadata in the same order. Each Adapter::wait_for_channel call is bounded by the
// remaining time against the cumulative deadline, so the total time to wait for all peers is
// bounded by `timeout` rather than 4 x `timeout`.
std::vector<hololink::module::EnumerationMetadata>
enumerate(hololink::module::Adapter& adapter,
    const std::vector<std::string>& expected_peer_ips,
    std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::vector<hololink::module::EnumerationMetadata> found;
    found.reserve(expected_peer_ips.size());
    for (const auto& ip : expected_peer_ips) {
        const auto now = std::chrono::steady_clock::now();
        const auto remaining = (now >= deadline)
            ? std::chrono::milliseconds(0)
            : std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
        found.push_back(adapter.wait_for_channel(ip, remaining));
    }
    return found;
}

} // namespace

// Everything the application needs to drive one data plane end-to-end: the enumeration
// metadata that identifies the supplement module, the per-board HololinkInterface
// (lifecycle: start / reset / stop), and the IMX274 driver instance for this data plane. Two
// channels that happen to share a board hold the same HololinkInterface shared_ptr (the
// service locator returns one instance per (module, instance_id) pair); the application
// treats them independently regardless.
struct Channel {
    hololink::module::EnumerationMetadata metadata;
    std::shared_ptr<hololink::module::HololinkInterfaceV1> hololink;
    std::shared_ptr<hololink::module::sensors::imx274::Imx274Cam> camera;
};

int main(int argc, char** argv)
{
    // Defaults match the Python sibling at examples/module_imx274_player.py.
    auto camera_mode = hololink::module::sensors::imx274::Imx274_Mode::IMX274_MODE_1920X1080_60FPS;
    bool headless = false;
    int64_t frame_limit = 0;
    std::vector<std::string> device_ips {
        hololink::module::env_hololink_ip(0, "192.168.0.200"),
        hololink::module::env_hololink_ip(1, "192.168.0.201"),
        hololink::module::env_hololink_ip(2, "192.168.0.202"),
        hololink::module::env_hololink_ip(3, "192.168.0.203"),
    };
    std::string module_dir;
    uint32_t window_height = 2160 / 4;
    uint32_t window_width = 3840 / 3;
    std::string window_title("hololink_module \xe2\x80\x94 4-channel IMX274");
    std::chrono::seconds discovery_timeout(30);

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "headless", no_argument, nullptr, 0 },
        { "camera-mode", required_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "device-0", required_argument, nullptr, 0 },
        { "device-1", required_argument, nullptr, 0 },
        { "device-2", required_argument, nullptr, 0 },
        { "device-3", required_argument, nullptr, 0 },
        { "module-dir", required_argument, nullptr, 0 },
        { "discovery-timeout", required_argument, nullptr, 0 },
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
            if (name == "headless") {
                headless = true;
            } else if (name == "camera-mode") {
                camera_mode = static_cast<hololink::module::sensors::imx274::Imx274_Mode>(
                    std::stoi(argument));
            } else if (name == "frame-limit") {
                frame_limit = std::stoll(argument);
            } else if (name == "device-0") {
                device_ips[0] = argument;
            } else if (name == "device-1") {
                device_ips[1] = argument;
            } else if (name == "device-2") {
                device_ips[2] = argument;
            } else if (name == "device-3") {
                device_ips[3] = argument;
            } else if (name == "module-dir") {
                module_dir = argument;
            } else if (name == "discovery-timeout") {
                discovery_timeout = std::chrono::seconds(std::stoll(argument));
            }
        } else if (c == 'h') {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --device-0 <ip>          Peer IP of device 0 (default 192.168.0.200)\n"
                      << "  --device-1 <ip>          Peer IP of device 1 (default 192.168.0.201)\n"
                      << "  --device-2 <ip>          Peer IP of device 2 (default 192.168.0.202)\n"
                      << "  --device-3 <ip>          Peer IP of device 3 (default 192.168.0.203)\n"
                      << "  --module-dir <path>      Directory containing hololink_<UUID>.so\n"
                      << "  --camera-mode <int>      IMX274 mode (1=1080p60 default, 0=4K60 RAW10, 2=4K60 RAW12)\n"
                      << "  --frame-limit <int>      Exit after this many frames per channel\n"
                      << "  --headless               Run holoviz without a display\n"
                      << "  --discovery-timeout <s>  Seconds to wait for bootp announcements (default 30)\n";
            return EXIT_SUCCESS;
        }
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the cameras and the boards
    // are stopped on the way out.
    return hx::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
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

        const std::vector<hololink::module::EnumerationMetadata> metadatas
            = enumerate(adapter, device_ips, discovery_timeout);

        // Build a channel per device.
        std::vector<Channel> channels;
        channels.reserve(metadatas.size());
        for (const hololink::module::EnumerationMetadata& md : metadatas) {
            Channel ch;
            ch.metadata = md;
            // Get the driver object
            ch.camera = std::make_shared<hololink::module::sensors::imx274::Imx274Cam>(md);
            // Connect to the HSB unit
            ch.hololink = hololink::module::HololinkInterfaceV1::get_service(md);
            channels.push_back(std::move(ch));
        }

        // Bring each unit up; channels sharing a board share a HololinkInterface, so
        // start/reset run once per board.
        std::vector<std::shared_ptr<hololink::module::HololinkInterfaceV1>> started;
        std::set<hololink::module::HololinkInterfaceV1*> brought_up;
        for (Channel& ch : channels) {
            if (brought_up.insert(ch.hololink.get()).second) {
                if (ch.hololink->start() != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error("HololinkInterface::start failed");
                }
                started.push_back(ch.hololink);
                // Drive the unit into a known state
                if (ch.hololink->reset() != HOLOLINK_MODULE_OK) {
                    throw std::runtime_error("HololinkInterface::reset failed");
                }
            }
            // Configure the camera
            ch.camera->configure(camera_mode);
            ch.camera->set_digital_gain_reg(0x4);
            // Re-assert the mode so configure_converter() below reads freshly-updated
            // width_/height_/pixel_format_ state (mirrors the original HSDK4 compose()'s own
            // set_mode() call, made right before configure_converter() there too).
            ch.camera->set_mode(camera_mode);
        }

        // Build the graph.
        holoscan::Graph graph { "module_quad_imx274_player" };

        // All four cameras share width / height / formats (same mode).
        const auto bayer_format = channels[0].camera->bayer_format();
        const auto pixel_format = channels[0].camera->pixel_format();

        // Pass 1: declare one image layer per channel, 2x2 grid (cam0 top-left, cam1 top-right,
        // cam2 bottom-left, cam3 bottom-right) -- the LayerStack must be complete before
        // HolovizOp is constructed.
        holoscan::visualization::LayerStack layers;
        for (std::size_t i = 0; i < channels.size(); ++i) {
            layers.add_image_layer(fmt::format("video_{}", i),
                hx::rgba16_layer(channels[i].camera->width(), channels[i].camera->height(),
                    holoscan::visualization::ViewTransform {
                        .offset_x = 0.5f * static_cast<float>(i % 2),
                        .offset_y = 0.5f * static_cast<float>(i / 2),
                        .scale_x = 0.5f,
                        .scale_y = 0.5f,
                    }));
        }

        holoscan::CompileOptions compile_options;
        // The window renders on each of camera 0's frames; the other layers show their newest one.
        auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers),
            hx::DisplayConfig {
                .title = window_title,
                .width = window_width,
                .height = window_height,
                .headless = headless,
            },
            compile_options, cu_device_ordinal);

        std::vector<std::string_view> tensor_output_operators;

        // Pass 2: build each camera's receive/convert/process/demosaic chain and wire it into viz.
        for (std::size_t i = 0; i < channels.size(); ++i) {
            const std::string name = std::to_string(i);
            Channel& ch = channels[i];

            hololink::module::csi::CsiGeometry geometry;
            ch.camera->configure_converter(
                hololink::module::operators::CsiToBayerOp::make_geometry_adapter(geometry));
            const size_t frame_size = geometry.start_byte
                + geometry.bytes_per_line * geometry.pixel_height + geometry.trailing_bytes;

            auto cam = ch.camera;
            auto receiver = graph.op<hololink::module::operators::RoceReceiverOp>(
                "receiver_" + name,
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

            auto csi = graph.op<hololink::module::operators::CsiToBayerOp>("csi_to_bayer_" + name);
            csi.set_params(hololink::module::operators::CsiToBayerParams {
                .cuda_device_ordinal = cu_device_ordinal,
                .geometry = geometry,
                .bayer_format = bayer_format,
            });

            auto ip = graph.op<hololink::module::operators::ImageProcessorOp>(
                "image_processor_" + name);
            ip.set_params(hololink::module::operators::ImageProcessorParams {
                .pixel_format = static_cast<int>(pixel_format),
                .bayer_format = static_cast<int>(bayer_format),
                // IMX274 optical black.
                .optical_black = 50,
                .cuda_device_ordinal = cu_device_ordinal,
                .pixel_width = geometry.pixel_width,
                .pixel_height = geometry.pixel_height,
            });

            const auto bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(bayer_format);
            auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>("bayer_demosaic_" + name);
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
                    .operator_path = fmt::format("bayer_demosaic_{}", i),
                    .output_port = "output",
                    .device = holoscan::DeviceId { cu_device_ordinal },
                });
        }

        const auto plan = holoscan::compile(graph, std::move(compile_options));
        if (!plan.ok()) {
            hx::print_compile_failure(plan);
            for (auto& hololink : started) {
                hololink->stop();
            }
            HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
            return EXIT_FAILURE;
        }

        std::cout << "Calling run" << std::endl;
        auto session = holoscan::run_async(plan, signals);
        session.wait();
        const bool failed = hx::report_session(session, plan);

        // Stop streaming data and close up sockets
        for (auto& hololink : started) {
            hololink->stop();
        }
        HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));

        return failed ? EXIT_FAILURE : EXIT_SUCCESS;
    });
}
