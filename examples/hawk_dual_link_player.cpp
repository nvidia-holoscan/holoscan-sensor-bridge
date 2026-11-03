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

// Two Hawk modules (one per GMSL link) live viewer over RoCE, one HolovizOp window per link:
//   RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
//
// hololink_module's Hawk/Ar0234/Max9295d/Max96716a/TauroTechDa326InterfaceV1 sensor driver is
// unchanged; only the receiver (hololink::module::operators::RoceReceiverOp, itself ported to
// HSDK5's Operator<Params> shape) and the downstream converters (native, already-ported
// hololink::operators::CsiToBayerOp/ImageProcessorOp) changed.

#include <getopt.h>

#include <cctype>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
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

#include "hawk.hpp"
#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/roce_receiver_op.hpp"
#include "hololink/module/taurotech_da326/taurotech_da326.hpp"
#include "max96716a.hpp"
#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

using hololink::module::sensors::ar0234::Ar0234;
using hololink::module::sensors::hawk::Hawk;
using hololink::module::sensors::max96716a::Max96716a;

// Geometry-capture adapter: see hawk_single_link_player.cpp for the rationale (module
// Hawk::configure_converter() only accepts the native shared_ptr<hololink::csi::CsiConverter>
// form). Pure geometry math, duplicated per app file.
class GeometryCaptureConverter : public hololink::csi::CsiConverter {
public:
    explicit GeometryCaptureConverter(hololink::csi::CsiGeometry& out)
        : out_(out)
    {
    }

    uint32_t receiver_start_byte() override { return 0; }

    uint32_t received_line_bytes(uint32_t line_bytes) override
    {
        return (line_bytes + 7u) & ~7u;
    }

    uint32_t transmitted_line_bytes(
        hololink::csi::PixelFormat pixel_format, uint32_t pixel_width) override
    {
        switch (pixel_format) {
        case hololink::csi::PixelFormat::RAW_8:
            return pixel_width;
        case hololink::csi::PixelFormat::RAW_10:
            return pixel_width * 5 / 4;
        case hololink::csi::PixelFormat::RAW_12:
            return pixel_width * 3 / 2;
        default:
            return pixel_width;
        }
    }

    void configure(uint32_t start_byte, uint32_t received_bytes_per_line, uint32_t pixel_width,
        uint32_t pixel_height, hololink::csi::PixelFormat pixel_format,
        uint32_t trailing_bytes = 0) override
    {
        out_ = { start_byte, received_bytes_per_line, pixel_width, pixel_height, pixel_format,
            trailing_bytes };
    }

private:
    hololink::csi::CsiGeometry& out_;
};

int run_player(int argc, char** argv, holoscan::ManagedShutdownSignals& signals)
{
    int32_t camera_mode_int = hololink::module::sensors::ar0234::ar0234_mode::AR0234_MODE_1920X1200_60FPS;
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    std::string hololink_ip = hololink::env_hololink_ip(0, "192.168.0.2");
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
    int32_t window_width = 1920;
    int32_t window_height = 1080;
    std::string title = "AR0234";
    std::string sensor_arg = "left";
    uint16_t exposure = 0x02DC;
    int32_t pattern = 0;
    bool pattern_set = false;
    bool skip_setup = false;
    bool skip_reset = false;

    const struct option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 },
        { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 },
        { "frame-limit", required_argument, nullptr, 0 },
        { "configuration", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 },
        { "window-height", required_argument, nullptr, 0 },
        { "window-width", required_argument, nullptr, 0 },
        { "title", required_argument, nullptr, 0 },
        { "sensor", required_argument, nullptr, 0 },
        { "exposure", required_argument, nullptr, 0 },
        { "pattern", required_argument, nullptr, 0 },
        { "skip-setup", no_argument, nullptr, 0 },
        { "skip-reset", no_argument, nullptr, 0 },
        { 0, 0, nullptr, 0 }
    };

    while (true) {
        int option_index = 0;
        const int c = getopt_long(argc, argv, "h", long_options, &option_index);
        if (c == -1)
            break;
        const std::string argument(optarg ? optarg : "");
        if (c == 0) {
            const std::string name = long_options[option_index].name;
            if (name == "camera-mode")
                camera_mode_int = std::stoi(argument);
            else if (name == "headless")
                headless = true;
            else if (name == "fullscreen")
                fullscreen = true;
            else if (name == "frame-limit")
                frame_limit = std::stoll(argument);
            else if (name == "configuration") {
                if (!hx::check_configuration(argument))
                    return EXIT_FAILURE;
            } else if (name == "hololink")
                hololink_ip = argument;
            else if (name == "window-height")
                window_height = std::stoi(argument);
            else if (name == "window-width")
                window_width = std::stoi(argument);
            else if (name == "title")
                title = argument;
            else if (name == "sensor")
                sensor_arg = argument;
            else if (name == "exposure")
                exposure = static_cast<uint16_t>(std::stoul(argument, nullptr, 0));
            else if (name == "pattern") {
                pattern = std::stoi(argument, nullptr, 0);
                pattern_set = true;
            } else if (name == "skip-setup")
                skip_setup = true;
            else if (name == "skip-reset")
                skip_reset = true;
            else if (name == "log-level") {
                const auto level = hx::parse_log_level(argument);
                if (!level)
                    throw std::runtime_error("Unhandled log level: " + argument);
                log_level = *level;
            }
        } else if (c == 'h') {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                      << "Stream from two Hawks (one per GMSL link)." << std::endl;
            return EXIT_SUCCESS;
        }
    }

    holoscan::set_log_level(log_level);

    CUdevice cu_device;
    std::shared_ptr<hololink::module::HololinkInterfaceV1> holo_iface;

    try {
        // CUDA
        CudaCheck(cuInit(0));
        int cu_device_ordinal = 0;
        CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
        CUcontext cu_context;
        CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
        cuCtxSetCurrent(cu_context);

        // Both SIFs always — one per link.
        const std::vector<int64_t> sif_list = { 0, 1 };

        hololink::module::EnumerationMetadata adapter_metadata
            = hololink::module::Adapter::get_adapter().wait_for_channel(hololink_ip,
                std::chrono::seconds(30));

        auto& adapter = hololink::module::Adapter::get_adapter();
        std::vector<hololink::module::EnumerationMetadata> per_channel_metadatas;
        for (auto sif : sif_list) {
            hololink::module::EnumerationMetadata sif_meta(adapter_metadata);
            adapter.use_sensor(sif_meta, sif);
            per_channel_metadatas.push_back(std::move(sif_meta));
        }

        holo_iface = hololink::module::HololinkInterfaceV1::get_service(adapter_metadata);
        holo_iface->start();
        auto board = hololink::module::taurotech_da326::TauroTechDa326InterfaceV1::get_service(adapter_metadata);
        // sensor_holo provides I2C access for sensor/deserializer setup;
        // sensors will be migrated to HololinkInterfaceV1::get_i2c() separately.
        auto sensor_holo = board->hololink();
        auto deserializer = std::make_shared<Max96716a>(sensor_holo);
        auto hawk_a = std::make_shared<Hawk>(sensor_holo, skip_setup);
        auto hawk_b = std::make_shared<Hawk>(sensor_holo, skip_setup);
        std::vector<std::shared_ptr<Hawk>> hawk_modules = { hawk_a, hawk_b };
        auto camera_mode = static_cast<hololink::module::sensors::ar0234::ar0234_mode::Mode>(camera_mode_int);
        for (auto& hawk : hawk_modules) {
            hawk->set_mode(camera_mode);
        }

        if (!skip_setup) {
            if (!skip_reset) {
                holo_iface->reset();
            }

            board->release_reset();
            board->power_cycle();
            board->check_power();
            board->setup_clock();

            // Deser sanity check
            uint8_t dev_id = deserializer->get_register(Max96716a::DEV_ID_REG);
            if (dev_id != Max96716a::DEV_ID) {
                throw std::runtime_error(fmt::format(
                    "Deserializer mismatch: expected 0x{:02X}, got 0x{:02X}",
                    Max96716a::DEV_ID, dev_id));
            }

            // Both serializers default to 0x40. Remap Link-B serializer to 0x41.
            deserializer->enable_link_exclusive(Max96716a::GmslLink::LINK_B);
            hawk_b->set_serializer_i2c_address(hawk_b->get_serializer_i2c_address() + 1);
            deserializer->enable_both_links();

            // Serializer sanity check
            for (auto& hawk : hawk_modules) {
                uint8_t ser_id = hawk->serializer().get_register(
                    hololink::module::sensors::max9295d::Max9295d::DEV_ID_REG);
                if (ser_id != hololink::module::sensors::max9295d::Max9295d::DEV_ID) {
                    throw std::runtime_error(fmt::format(
                        "Serializer mismatch: expected 0x{:02X}, got 0x{:02X}",
                        hololink::module::sensors::max9295d::Max9295d::DEV_ID, ser_id));
                }
            }

            // Remap hawk_b sensors from default 0x10/0x18 to 0x11/0x1A.
            hawk_b->remap_sensor_addresses(0x11, 0x1A);

            // Sensor sanity check
            for (auto& hawk : hawk_modules) {
                for (auto& sensor : hawk->sensors()) {
                    uint16_t s_id = sensor->get_register(Ar0234::DEV_ID_REG);
                    if (s_id != Ar0234::DEV_ID) {
                        throw std::runtime_error(fmt::format(
                            "Sensor mismatch at i2c=0x{:02X}: expected 0x{:04X}, got 0x{:04X}",
                            sensor->get_i2c_address(), Ar0234::DEV_ID, s_id));
                    }
                }
            }

            // Deserializer settings
            deserializer->configure_video_pipe();
            uint8_t left, right;
            if (sensor_arg == "left") {
                left = deserializer->stream_id_to_pipe_mapping(
                    Max96716a::GmslLink::LINK_A, 0, Max96716a::VideoPipe::PIPE_Y);
                right = deserializer->stream_id_to_pipe_mapping(
                    Max96716a::GmslLink::LINK_B, 0, Max96716a::VideoPipe::PIPE_Z);
            } else if (sensor_arg == "right") {
                left = deserializer->stream_id_to_pipe_mapping(
                    Max96716a::GmslLink::LINK_A, 2, Max96716a::VideoPipe::PIPE_Y);
                right = deserializer->stream_id_to_pipe_mapping(
                    Max96716a::GmslLink::LINK_B, 2, Max96716a::VideoPipe::PIPE_Z);
            } else {
                throw std::runtime_error("Non-existent config " + sensor_arg);
            }
            deserializer->set_register(Max96716a::VIDEO_PIPE_SEL,
                static_cast<uint8_t>(left | right));

            // Sensor programming
            for (auto& hawk : hawk_modules) {
                hawk->configure(camera_mode);
                if (pattern_set) {
                    hawk->test_pattern(static_cast<uint16_t>(pattern));
                }
                hawk->set_exposure_reg(exposure);
            }
        }

        // A first then B, label by sensor side.
        std::string sensor_label = sensor_arg;
        if (!sensor_label.empty()) {
            sensor_label[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(sensor_label[0])));
        }
        std::vector<std::string> window_titles = {
            sensor_label + " A",
            sensor_label + " B",
        };

        int32_t per_h = window_height / 2;
        int32_t per_w = window_width / 2;

        // Build the graph.
        holoscan::Graph graph { "hawk_dual_link_player" };
        holoscan::CompileOptions compile_options;

        std::vector<std::string> tensor_output_operator_paths;
        std::vector<std::string> image_output_operator_paths;

        for (size_t idx = 0; idx < sif_list.size(); ++idx) {
            auto hawk = hawk_modules[idx];
            const std::string suffix = std::to_string(idx);

            hololink::csi::CsiGeometry geometry;
            hawk->configure_converter(std::make_shared<GeometryCaptureConverter>(geometry));
            const size_t frame_size = geometry.start_byte
                + geometry.bytes_per_line * geometry.pixel_height + geometry.trailing_bytes;
            const auto bayer_format = hawk->bayer_format();
            const auto pixel_format = hawk->pixel_format();
            const auto bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(bayer_format);

            holoscan::visualization::LayerStack layers;
            // The sensor's image; HolovizOp scales it into the window.
            layers.add_image_layer("video", hx::rgba16_layer(geometry.pixel_width, geometry.pixel_height));

            auto viz = hx::add_holoviz(graph, "holoviz_" + suffix, std::move(layers),
                hx::DisplayConfig {
                    .title = window_titles[idx],
                    .width = static_cast<uint32_t>(per_w),
                    .height = static_cast<uint32_t>(per_h),
                    .fullscreen = fullscreen,
                    .headless = headless,
                },
                compile_options, cu_device_ordinal);

            auto receiver = graph.op<hololink::module::operators::RoceReceiverOp>(
                "receiver_" + suffix,
                per_channel_metadatas[idx],
                [hawk] { hawk->start(); },
                [hawk] { hawk->stop(); });
            receiver.set_params(hololink::module::operators::RoceReceiverParams {
                .frame_context = cu_context,
                .frame_size = frame_size,
                // Stop after --frame-limit frames, as HSDK4's CountCondition on the receiver did.
                .frame_limit = frame_limit,
                .frame_limit_ends_run = true,
            });

            auto csi = graph.op<hololink::operators::CsiToBayerOp>("csi_to_bayer_" + suffix);
            csi.set_params(hololink::operators::CsiToBayerParams {
                .cuda_device_ordinal = cu_device_ordinal,
                .geometry = geometry,
                .bayer_format = bayer_format,
            });

            auto ip = graph.op<hololink::operators::ImageProcessorOp>("image_processor_" + suffix);
            ip.set_params(hololink::operators::ImageProcessorParams {
                .pixel_format = int(pixel_format),
                .bayer_format = int(bayer_format),
                .optical_black = 50,
                .cuda_device_ordinal = cu_device_ordinal,
                .pixel_width = geometry.pixel_width,
                .pixel_height = geometry.pixel_height,
            });

            auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>("bayer_demosaic_" + suffix);
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

            tensor_output_operator_paths.push_back("receiver_" + suffix);
            tensor_output_operator_paths.push_back("csi_to_bayer_" + suffix);
            tensor_output_operator_paths.push_back("image_processor_" + suffix);
            image_output_operator_paths.push_back("bayer_demosaic_" + suffix);
        }

        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        for (const auto& operator_path : tensor_output_operator_paths) {
            compile_options.deployment.bind_tensor_output_device(
                holoscan::TensorOutputDevicePlacement {
                    .operator_path = operator_path,
                    .output_port = "tensor",
                    .device = holoscan::DeviceId { cu_device_ordinal },
                });
        }
        for (const auto& operator_path : image_output_operator_paths) {
            compile_options.deployment.bind_tensor_output_device(
                holoscan::TensorOutputDevicePlacement {
                    .operator_path = operator_path,
                    .output_port = "output",
                    .device = holoscan::DeviceId { cu_device_ordinal },
                });
        }

        const auto plan = holoscan::compile(graph, std::move(compile_options));
        if (!plan.ok()) {
            hx::print_compile_failure(plan);
            holo_iface->stop();
            CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
            return EXIT_FAILURE;
        }

        auto session = holoscan::run_async(plan, signals);
        session.wait();
        const bool failed = hx::report_session(session, plan);

        holo_iface->stop();
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));

        return failed ? EXIT_FAILURE : EXIT_SUCCESS;
    } catch (const std::exception& e) {
        if (holo_iface) {
            holo_iface->stop();
        }
        HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
        return EXIT_FAILURE;
    }
}

} // namespace

int main(int argc, char** argv)
{
    // With managed signals, Ctrl+C stops the session cooperatively, so both Hawks and the board
    // are stopped on the way out.
    return hx::run_with_managed_signals(
        [&](holoscan::ManagedShutdownSignals& signals) { return run_player(argc, argv, signals); });
}
