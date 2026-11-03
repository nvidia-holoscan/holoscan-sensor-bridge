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

// Hawk module (one GMSL link, --sensor left|right|both) live viewer over FuSa CoE:
//   FusaCoeCaptureOp -> [PackedFormatConverterOp | CsiToBayerOp] -> ImageProcessorOp ->
//     BayerDemosaicOp -> HolovizOp
//
// The unpack stage depends on the sensor mode's pixel format: RAW_8 modes are always captured
// unpacketized (the HSB packetizer only has programs for RAW_10/RAW_12), so those use
// CsiToBayerOp; RAW_10/RAW_12 modes default to packetized capture (override with
// --no-packetizer), which uses PackedFormatConverterOp instead.
//
// hololink_module's Hawk/Ar0234/Max9295d/Max96716a/TauroTechDa326InterfaceV1 sensor driver is
// unchanged; only the receiver (hololink::module::operators::FusaCoeCaptureOp, itself ported to
// HSDK5's Operator<Params> shape) and the downstream converters (native, already-ported
// hololink::operators::{CsiToBayerOp,PackedFormatConverterOp,ImageProcessorOp}) changed.
// Geometry resolution reuses native FusaCoeCaptureOp::make_geometry_adapter() (a static, HSB-
// packetizer-alignment-aware helper -- see its declaration comment), since module Hawk's
// configure_converter() speaks the same native hololink::csi::CsiConverter interface; the
// resulting hololink::csi::CsiGeometry is then copied into the module-tree
// hololink::module::csi::CsiGeometry my ported FusaCoeCaptureOp's Params expects.

#include <chrono>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
#include "hololink/module/mipi_dphy.hpp"
#include "hololink/module/operators/fusa_coe_capture_op.hpp"
#include "hololink/module/taurotech_da326/taurotech_da326.hpp"
#include "max96716a.hpp"
#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/fusa_coe_capture/fusa_coe_capture.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/packed_format_converter/packed_format_converter.hpp>

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

using hololink::module::sensors::ar0234::Ar0234;
using hololink::module::sensors::hawk::Hawk;
using hololink::module::sensors::max96716a::Max96716a;

std::vector<uint8_t> parse_mac_address(const std::string& mac_str)
{
    std::vector<uint8_t> mac;
    std::istringstream iss(mac_str);
    std::string byte_str;
    while (std::getline(iss, byte_str, ':')) {
        if (mac.size() >= 6) {
            throw std::invalid_argument("Too many bytes in MAC address");
        }
        if (byte_str.size() != 2) {
            throw std::invalid_argument("Each MAC byte must be exactly two hex digits");
        }
        uint16_t byte;
        std::istringstream hex_stream(byte_str);
        hex_stream >> std::hex >> byte;
        if (hex_stream.fail()) {
            throw std::invalid_argument("Invalid hex in MAC address");
        }
        mac.push_back(static_cast<uint8_t>(byte));
    }
    if (mac.size() != 6) {
        throw std::invalid_argument("MAC address must have exactly 6 bytes");
    }
    return mac;
}

std::vector<int> sensor_arg_to_sif(const std::string& sensor)
{
    if (sensor == "left")
        return { 0 };
    if (sensor == "right")
        return { 1 };
    if (sensor == "both")
        return { 0, 1 };
    throw std::runtime_error("Invalid --sensor value: " + sensor);
}

// Runs the player on the options main() parsed.
int run(hololink::args::VariablesMap& variables_map, holoscan::ManagedShutdownSignals& signals)
{
    const auto hololink_ip = variables_map["hololink"].as<std::string>();
    const auto sensor_arg = variables_map["sensor"].as<std::string>();
    const auto channel_arg = variables_map["channel"].as<std::string>();
    const auto interface_arg = variables_map["interface"].as<std::string>();
    const auto title = variables_map["title"].as<std::string>();
    const auto headless = variables_map["headless"].as<bool>();
    const auto fullscreen = variables_map["fullscreen"].as<bool>();
    const auto skip_reset = variables_map["skip-reset"].as<bool>();
    const auto no_packetizer = variables_map["no-packetizer"].as<bool>();
    const int64_t frame_limit = variables_map["frame-limit"].as<int>();
    const int32_t window_width = variables_map["width"].as<int>();
    const int32_t window_height = variables_map["height"].as<int>();
    const uint32_t timeout_ms = static_cast<uint32_t>(variables_map["timeout"].as<int>());
    const uint16_t exposure = static_cast<uint16_t>(variables_map["exposure"].as<int>());
    const int32_t pattern = variables_map["pattern"].as<int>();
    const auto camera_mode_int = variables_map["camera-mode"].as<int>();

    std::shared_ptr<hololink::module::HololinkInterfaceV1> holo_iface;

    try {
        auto sif_list = sensor_arg_to_sif(sensor_arg);
        const std::vector<std::string> sif_name = { "left", "right" };

        auto& adapter = hololink::module::Adapter::get_adapter();
        hololink::module::EnumerationMetadata adapter_metadata
            = adapter.wait_for_channel(hololink_ip, std::chrono::seconds(30));

        std::vector<hololink::module::EnumerationMetadata> sif_metadatas;
        for (int sif : sif_list) {
            hololink::module::EnumerationMetadata sif_meta(adapter_metadata);
            adapter.use_sensor(sif_meta, sif);
            sif_metadatas.push_back(std::move(sif_meta));
        }

        holo_iface = hololink::module::HololinkInterfaceV1::get_service(adapter_metadata);
        holo_iface->start();
        auto board = hololink::module::taurotech_da326::TauroTechDa326InterfaceV1::get_service(adapter_metadata);
        auto legacy_holo = board->hololink();
        auto deserializer = std::make_shared<Max96716a>(legacy_holo);
        auto hawk = std::make_shared<Hawk>(legacy_holo);
        auto camera_mode = static_cast<hololink::module::sensors::ar0234::ar0234_mode::Mode>(camera_mode_int);

        std::vector<std::shared_ptr<Ar0234>> cameras;
        auto hawk_sensors = hawk->sensors();
        for (int sif : sif_list) {
            cameras.push_back(hawk_sensors[sif]);
        }

        if (!skip_reset) {
            holo_iface->reset();
        }
        board->release_reset();
        board->power_cycle();
        board->check_power();
        board->setup_clock();

        uint8_t dev_id = deserializer->get_register(Max96716a::DEV_ID_REG);
        if (dev_id != Max96716a::DEV_ID) {
            throw std::runtime_error(fmt::format(
                "Deserializer mismatch: expected 0x{:02X}, got 0x{:02X}",
                Max96716a::DEV_ID, dev_id));
        }

        auto link = (channel_arg == "B") ? Max96716a::GmslLink::LINK_B
                                         : Max96716a::GmslLink::LINK_A;
        deserializer->enable_link_exclusive(link);

        uint8_t ser_id = hawk->serializer().get_register(hololink::module::sensors::max9295d::Max9295d::DEV_ID_REG);
        if (ser_id != hololink::module::sensors::max9295d::Max9295d::DEV_ID) {
            throw std::runtime_error(fmt::format(
                "Serializer mismatch: expected 0x{:02X}, got 0x{:02X}",
                hololink::module::sensors::max9295d::Max9295d::DEV_ID, ser_id));
        }

        for (auto& s : hawk->sensors()) {
            uint16_t s_id = s->get_register(Ar0234::DEV_ID_REG);
            if (s_id != Ar0234::DEV_ID) {
                throw std::runtime_error(fmt::format(
                    "Sensor mismatch at i2c=0x{:02X}: expected 0x{:04X}, got 0x{:04X}",
                    s->get_i2c_address(), Ar0234::DEV_ID, s_id));
            }
        }

        deserializer->configure_video_pipe();
        uint8_t left = deserializer->stream_id_to_pipe_mapping(
            link, 0, Max96716a::VideoPipe::PIPE_Y);
        uint8_t right = deserializer->stream_id_to_pipe_mapping(
            link, 2, Max96716a::VideoPipe::PIPE_Z);
        deserializer->set_register(Max96716a::VIDEO_PIPE_SEL,
            static_cast<uint8_t>(left | right));

        constexpr unsigned MIPI_LANE_COUNT = 4;
        constexpr unsigned MIPI_LINE_RATE_MBPS = 450;
        auto dphy = hololink::module::MipiDphyInterfaceV1::get_service(adapter_metadata);
        // Currently, each SIF corresponds with the physical MIPI interface of the same index.
        for (int physical_mipi_interface : sif_list) {
            const auto status = dphy->program(static_cast<unsigned>(physical_mipi_interface),
                MIPI_LANE_COUNT, MIPI_LINE_RATE_MBPS);
            if (status != HOLOLINK_MODULE_OK) {
                throw std::runtime_error(fmt::format(
                    "Could not program MIPI interface {}: status {}",
                    physical_mipi_interface, status));
            }
        }

        hawk->configure(camera_mode);
        if (pattern) {
            hawk->test_pattern(static_cast<uint16_t>(pattern));
        }
        hawk->set_exposure_reg(exposure);

        // Same physical NIC interface/MAC for every sensor on this board.
        const std::string interface = interface_arg.empty()
            ? sif_metadatas[0].get<std::string>("interface", "")
            : interface_arg;
        const auto mac_str = sif_metadatas[0].get<std::string>("mac_id", "");
        const auto mac_bytes = parse_mac_address(mac_str);
        HOLOSCAN_LOG_INFO("Using interface={} mac={}", interface, mac_str);

        int32_t per_w = window_width;
        int32_t per_h = window_height;
        if (sif_list.size() > 1) {
            per_w /= 2;
            per_h /= 2;
        }

        // Build the graph.
        holoscan::Graph graph { "fusa_coe_hawk_single_link_player" };
        holoscan::CompileOptions compile_options;

        std::vector<std::string_view> tensor_output_operator_paths;
        std::vector<std::string> image_output_operator_paths;

        for (size_t i = 0; i < cameras.size(); ++i) {
            auto camera = cameras[i];
            const std::string name = std::to_string(i);
            const std::string window_title = title + " - " + sif_name[sif_list[i]];

            const auto pixel_format = camera->get_pixel_format();
            const auto bayer_format = camera->get_bayer_format();

            // The packetizer only has programs for RAW_10 and RAW_12, so an 8-bit mode is
            // always captured unpacketized.
            const bool packetizer_enabled = !no_packetizer
                && pixel_format != hololink::csi::PixelFormat::RAW_8;

            // Resolve geometry using native FusaCoeCaptureOp's HSB-alignment-aware adapter
            // (needs no live instance -- see its declaration comment), then copy the result
            // into the module-tree geometry struct my ported FusaCoeCaptureOp's Params expects.
            hololink::csi::CsiGeometry native_geometry;
            camera->configure_converter(hololink::operators::FusaCoeCaptureOp::make_geometry_adapter(
                native_geometry, packetizer_enabled));
            hololink::module::csi::CsiGeometry geometry {
                .start_byte = native_geometry.start_byte,
                .bytes_per_line = native_geometry.bytes_per_line,
                .pixel_width = native_geometry.pixel_width,
                .pixel_height = native_geometry.pixel_height,
                .pixel_format = static_cast<hololink::module::csi::PixelFormat>(
                    static_cast<int>(native_geometry.pixel_format)),
                .trailing_bytes = native_geometry.trailing_bytes,
            };

            holoscan::visualization::LayerStack layers;
            // The sensor's image; HolovizOp scales it into the window.
            layers.add_image_layer("video", hx::rgba16_layer(native_geometry.pixel_width, native_geometry.pixel_height));

            auto viz = hx::add_holoviz(graph, "holoviz_" + name, std::move(layers),
                hx::DisplayConfig {
                    .title = window_title,
                    .width = static_cast<uint32_t>(per_w),
                    .height = static_cast<uint32_t>(per_h),
                    .fullscreen = fullscreen,
                    .headless = headless,
                },
                compile_options, /*cuda_device_ordinal=*/0);

            auto capture = graph.op<hololink::module::operators::FusaCoeCaptureOp>(
                fmt::format("capture_{}", name));
            capture.set_params(hololink::module::operators::FusaCoeCaptureParams {
                .metadata = sif_metadatas[i],
                .device_start = [hawk] { hawk->start(); },
                .device_stop = [hawk] { hawk->stop(); },
                .interface = interface,
                .mac_addr = mac_bytes,
                .timeout = timeout_ms,
                .pixel_format = static_cast<hololink::module::csi::PixelFormat>(
                    static_cast<int>(pixel_format)),
                .bayer_format = static_cast<hololink::module::csi::BayerFormat>(
                    static_cast<int>(bayer_format)),
                .packetizer_enabled = packetizer_enabled,
                .geometry = geometry,
                // Stop after --frame-limit frames, as HSDK4's CountCondition on the receiver did.
                .frame_limit = frame_limit,
                .frame_limit_ends_run = true,
                // Names this camera in the capture's stall messages.
                .name = std::string(capture.name()),
            });

            auto image_processor = graph.op<hololink::operators::ImageProcessorOp>(
                fmt::format("image_processor_{}", name));
            image_processor.set_params(hololink::operators::ImageProcessorParams {
                .pixel_format = int(pixel_format),
                .bayer_format = int(bayer_format),
                .optical_black = 50,
                .cuda_device_ordinal = 0,
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
                auto converter = graph.op<hololink::operators::PackedFormatConverterOp>(
                    fmt::format("packed_format_converter_{}", name));
                converter.set_params(hololink::operators::PackedFormatConverterParams {
                    .cuda_device_ordinal = 0,
                    .bayer_format = bayer_format,
                    .geometry = native_geometry,
                });
                graph.add_flow(capture->tensor, converter->input_tensor, hx::pipeline_edge());
                graph.add_flow(capture->metadata, converter->input_metadata, hx::pipeline_edge());
                graph.add_flow(converter->output_tensor, image_processor->input_tensor, hx::pipeline_edge());
                graph.add_flow(converter->output_metadata, image_processor->input_metadata, hx::pipeline_edge());
                tensor_output_operator_paths.push_back(converter.name());
            } else {
                auto converter = graph.op<hololink::operators::CsiToBayerOp>(
                    fmt::format("csi_to_bayer_{}", name));
                converter.set_params(hololink::operators::CsiToBayerParams {
                    .cuda_device_ordinal = 0,
                    .geometry = native_geometry,
                    .bayer_format = bayer_format,
                    // Fed by FusaCoeCaptureOp, which already strips the CSI header before emitting.
                    .header_included = false,
                });
                graph.add_flow(capture->tensor, converter->input_tensor, hx::pipeline_edge());
                graph.add_flow(capture->metadata, converter->input_metadata, hx::pipeline_edge());
                graph.add_flow(converter->output_tensor, image_processor->input_tensor, hx::pipeline_edge());
                graph.add_flow(converter->output_metadata, image_processor->input_metadata, hx::pipeline_edge());
                tensor_output_operator_paths.push_back(converter.name());
            }

            graph.add_flow(image_processor->output_tensor, demosaic->input, hx::pipeline_edge());
            graph.add_flow(demosaic->output, viz->image_layer_input(0), hx::display_edge());

            tensor_output_operator_paths.push_back(capture.name());
            tensor_output_operator_paths.push_back(image_processor.name());
            image_output_operator_paths.push_back(fmt::format("bayer_demosaic_{}", name));
        }

        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        for (const auto& operator_path : tensor_output_operator_paths) {
            compile_options.deployment.bind_tensor_output_device(
                holoscan::TensorOutputDevicePlacement {
                    .operator_path = operator_path,
                    .output_port = "tensor",
                    .device = holoscan::DeviceId { 0 },
                });
        }
        for (const auto& operator_path : image_output_operator_paths) {
            compile_options.deployment.bind_tensor_output_device(
                holoscan::TensorOutputDevicePlacement {
                    .operator_path = operator_path,
                    .output_port = "output",
                    .device = holoscan::DeviceId { 0 },
                });
        }

        const auto plan = holoscan::compile(graph, std::move(compile_options));
        if (!plan.ok()) {
            hx::print_compile_failure(plan);
            holo_iface->stop();
            return -1;
        }

        auto session = holoscan::run_async(plan, signals);
        session.wait();
        const bool failed = hx::report_session(session, plan);

        holo_iface->stop();
        return failed ? -1 : 0;
    } catch (const std::exception& e) {
        if (holo_iface) {
            holo_iface->stop();
        }
        HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
        return -1;
    }
}

} // namespace

int main(int argc, char** argv)
{
    using namespace hololink::args;
    OptionsDescription options_description("Allowed options");

    // clang-format off
    options_description.add_options()
        ("camera-mode", value<int>()->default_value(
            static_cast<int>(hololink::module::sensors::ar0234::ar0234_mode::AR0234_MODE_1920X1200_60FPS)),
            "AR0234 mode")
        ("channel", value<std::string>()->default_value("A"), "GMSL link to enable: A or B")
        ("exposure", value<int>()->default_value(0x02DC), "AR0234 exposure register value")
        ("frame-limit", value<int>()->default_value(0), "Exit after this many frames")
        ("fullscreen", bool_switch()->default_value(false), "Run in fullscreen mode")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("height", value<int>()->default_value(1080), "Window height")
        ("hololink", value<std::string>()->default_value(hololink::env_hololink_ip(0, "192.168.0.2")), "IP address of Hololink board")
        ("interface", value<std::string>()->default_value(""), "Ethernet interface (empty = auto)")
        ("no-packetizer", bool_switch()->default_value(false), "Capture unpacketized CSI-2 data instead of HSB-packed data")
        ("pattern", value<int>()->default_value(0), "Sensor test pattern (0 = off)")
        ("sensor", value<std::string>()->default_value("left"),
            "Which sensor(s) to stream: 'left', 'right', or 'both' (opens one window per sensor)")
        ("skip-reset", bool_switch()->default_value(false), "Skip the Hololink reset step")
        ("timeout", value<int>()->default_value(1500), "Capture request timeout in milliseconds")
        ("title", value<std::string>()->default_value("AR0234 (FuSa CoE)"), "Window title prefix")
        ("width", value<int>()->default_value(1920), "Window width")
        ;
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);

    // With managed signals, Ctrl+C stops the session cooperatively, so the cameras and the board
    // are stopped on the way out.
    return hx::run_with_managed_signals(
        [&](holoscan::ManagedShutdownSignals& signals) { return run(variables_map, signals); });
}
