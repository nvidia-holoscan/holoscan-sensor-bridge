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

// Single VB1940 live viewer over a single RoCE network:
//   RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

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
#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/sensors/camera/camera_sensor.hpp>
#include <hololink/sensors/camera/vb1940/native_vb1940_sensor.hpp>

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

int run_player(hololink::args::VariablesMap& variables_map, uint32_t use_sensor,
    holoscan::ManagedShutdownSignals& signals)
{
    // Empty: the receiver uses the device on the board's network interface.
    const std::string ibv_name = variables_map["ibv-name"].as<std::string>();

    const int cuda_device_ordinal = 0;
    CudaCheck(cuInit(0));
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    cuCtxSetCurrent(cu_context);

    const std::string hololink_ip = variables_map["hololink"].as<std::string>();
    HOLOSCAN_LOG_INFO("Finding Hololink channel for IP: {}", hololink_ip);
    auto channel_metadata = hololink::Enumerator::find_channel(hololink_ip);
    hololink::DataChannel::use_sensor(channel_metadata, use_sensor);
    auto hololink_channel = std::make_shared<hololink::DataChannel>(channel_metadata);

    auto hololink_dev = hololink_channel->hololink();
    hololink_dev->start();
    hololink_dev->reset();

    auto camera = std::make_shared<hololink::sensors::NativeVb1940Sensor>(*hololink_channel);

    const auto camera_mode = static_cast<hololink::sensors::vb1940_mode::Mode>(
        variables_map["camera-mode"].as<int>());

    hololink_dev->write_uint32(0x8, 0x0); // Keep the sensor RESET at low
    camera->setup_clock();
    // Release the sensor RESET to high
    hololink_dev->write_uint32(0x8, use_sensor == 0 ? 0x1 : 0x2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    camera->get_register_32(0x0000); // DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
    camera->get_register_32(0x0734); // EXT_CLOCK(25MHz = 0x017d7840)
    camera->configure(camera_mode);

    // Build the graph.
    holoscan::Graph graph { "vb1940_player" };

    hololink::csi::CsiGeometry geometry;
    camera->configure_converter(geometry);
    const size_t frame_size = geometry.start_byte + geometry.bytes_per_line * geometry.pixel_height
        + geometry.trailing_bytes;

    holoscan::CompileOptions compile_options;
    holoscan::visualization::LayerStack layers;
    layers.add_image_layer("video",
        hx::rgba16_layer(static_cast<uint32_t>(camera->get_width()), static_cast<uint32_t>(camera->get_height())));

    auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers),
        hx::DisplayConfig {
            .fullscreen = variables_map["fullscreen"].as<bool>(),
            .headless = variables_map["headless"].as<bool>(),
        },
        compile_options, cuda_device_ordinal);

    const uint32_t ibv_port = variables_map["ibv-port"].as<uint32_t>();

    auto receiver = graph.op<hololink::operators::RoceReceiverOp>(
        "receiver",
        hololink_channel,
        [camera] { camera->start(); },
        [camera] { camera->stop(); });
    receiver.set_params(hololink::operators::RoceReceiverParams {
        .frame_context = cu_context,
        .frame_size = frame_size,
        .ibv_name = ibv_name,
        .ibv_port = ibv_port,
        .frame_limit = variables_map["frame-limit"].as<int>(),
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
        .optical_black = 8, // Optical black value for vb1940 is 8 for RAW10
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
        hololink_dev->stop();
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return -1;
    }

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
    using namespace hololink::args;
    OptionsDescription options_description("Allowed options");

    // clang-format off
    options_description.add_options()
        ("camera-mode", value<int>()->default_value(static_cast<int>(hololink::sensors::vb1940_mode::VB1940_MODE_2560X1984_30FPS)), "VB1940 mode (default: 0)")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("fullscreen", bool_switch()->default_value(false), "Run in fullscreen mode")
        ("frame-limit", value<int>()->default_value(0), "Exit after receiving this many frames")
        ("configuration", value<std::string>(), "Configuration file")
        ("hololink", value<std::string>()->default_value(hololink::env_hololink_ip(0, "192.168.0.2")), "IP address of Hololink board")
        ("ibv-name", value<std::string>()->default_value(""), "IBV device to use (default: the device on the board's network interface)")
        ("ibv-port", value<uint32_t>()->default_value(1), "Port number of IBV device, with --ibv-name (default: 1)")
        ("use-sensor", value<uint32_t>()->default_value(0), "Use the specific sensor (0 or 1, default=0)")
        ;
    // clang-format on

    hololink::logging::hsb_log_level = hololink::logging::HSB_LOG_LEVEL_INFO;

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);

    const uint32_t use_sensor = variables_map["use-sensor"].as<uint32_t>();
    if (use_sensor >= 2) {
        HOLOSCAN_LOG_ERROR("Invalid value use_sensor={}; must be 0 or 1.", use_sensor);
        return -1;
    }
    if (variables_map.count("configuration")
        && !hx::check_configuration(variables_map["configuration"].as<std::string>())) {
        return -1;
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the camera and the board
    // are stopped on the way out.
    return hx::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
        try {
            return run_player(variables_map, use_sensor, signals);
        } catch (const std::exception& e) {
            HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
            return -1;
        }
    });
}
