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

// FuSa CoE live viewer — one or two IMX274 sensors via FusaCoeCaptureOp, displayed live with
// Holoviz:
//   FusaCoeCaptureOp -> [PackedFormatConverterOp | CsiToBayerOp] -> ImageProcessorOp ->
//     BayerDemosaicOp -> HolovizOp
//
// The unpack stage depends on the sensor mode's pixel format: RAW_8 modes are always captured
// unpacketized (the HSB packetizer only has programs for RAW_10/RAW_12), so those use
// CsiToBayerOp; RAW_10/RAW_12 modes default to packetized capture (override with
// --no-packetizer), which uses PackedFormatConverterOp instead.
//
// ImageProcessorOp delivers a [H, W, 1] uint16 Bayer mosaic (single-channel white-balanced);
// its least_significant_bit=16-N justification left-aligns the sensor's N-bit signal within the
// 16-bit word. BayerDemosaicOp converts that mosaic to a full-color RGBA16 image, and HolovizOp
// displays both sensors side by side.
//
// Usage:
//   fusa_coe_imx274_player [OPTIONS]
//
// Options:
//   --hololink IP       IP address of Hololink board (default: 192.168.0.2)
//   --sensor N          Sensor index: 0 or 1, or -1 (default) for both sensors side by side
//   --camera-mode N     IMX274 mode (default: 1 = 1920x1080@60fps); the 4k modes (0, 2)
//                       require a single sensor, since two 4k streams overrun the network
//   --timeout N         Per-frame capture timeout in milliseconds (default: 1500)
//   --frame-limit N     Exit after this many frames from each sensor (default: 0 = no limit)
//   --metadata          Print frame metadata for each sensor
//   --no-packetizer     Capture unpacketized CSI-2 data instead of HSB-packed data
//   --headless          Run without opening a window (still renders, presents nowhere)
//   --fullscreen        Run in fullscreen mode

#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/vision/bayer_demosaic_op.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/fusa_coe_capture/fusa_coe_capture.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/packed_format_converter/packed_format_converter.hpp>
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

struct MetadataPrinterParams {
    std::string label;
};

class MetadataPrinterOp : public holoscan::Operator<MetadataPrinterParams> {
public:
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input(input_metadata, "metadata").queue_depth(4U);
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnEach { input_metadata });
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext&) override
    {
        auto sample = input_metadata.receive();
        if (!sample)
            return holoscan::make_unexpected(std::move(sample).error());

        const hololink::CaptureMetadataT& m = sample->data;
        HOLOSCAN_LOG_INFO("{}: frame={}, ts={}.{:09}, meta_ts={}.{:09}, crc=0x{:08x}",
            params().label,
            m.frame_number,
            m.timestamp_s,
            m.timestamp_ns,
            m.metadata_s,
            m.metadata_ns,
            m.crc);
        return {};
    }
};

hololink::core::MacAddress parse_mac_address(const std::string& mac_str)
{
    hololink::core::MacAddress mac {};
    std::istringstream iss(mac_str);
    std::string byte_str;
    int i = 0;
    while (std::getline(iss, byte_str, ':')) {
        if (i >= 6) {
            throw std::invalid_argument("Too many bytes in MAC address");
        }
        if (byte_str.size() != 2) {
            throw std::invalid_argument("Each MAC byte must be exactly two hex digits");
        }
        uint16_t byte = 0;
        std::istringstream hex_stream(byte_str);
        hex_stream >> std::hex >> byte;
        if (hex_stream.fail()) {
            throw std::invalid_argument("Invalid hex in MAC address");
        }
        mac[i++] = static_cast<uint8_t>(byte);
    }
    if (i != 6) {
        throw std::invalid_argument("MAC address must have exactly 6 bytes");
    }
    return mac;
}

int run(
    const std::string& hololink_ip,
    int sensor,
    hololink::sensors::imx274_mode::Mode camera_mode,
    uint32_t timeout_ms,
    int frame_limit,
    bool headless,
    bool fullscreen,
    bool print_metadata,
    bool no_packetizer,
    holoscan::ManagedShutdownSignals& signals)
{
    const int cuda_device_ordinal = 0;

    CudaCheck(cuInit(0));
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    cuCtxSetCurrent(cu_context);

    auto channel_metadata_raw = hololink::Enumerator::find_channel(hololink_ip);

    // sensor == -1 selects both sensors, side by side; sensor == 0 or 1 selects just one.
    std::vector<int> sensor_ids;
    if (sensor == -1) {
        sensor_ids = { 0, 1 };
    } else {
        sensor_ids = { sensor };
    }

    std::vector<hololink::DataChannel> hololink_channels;
    for (int sensor_id : sensor_ids) {
        auto channel_metadata = hololink::Metadata(channel_metadata_raw);
        hololink::DataChannel::use_sensor(channel_metadata, sensor_id);
        hololink_channels.push_back(hololink::DataChannel(channel_metadata));
    }

    // Same physical NIC interface/MAC for every sensor on this board.
    const std::string interface = hololink_channels[0].enumeration_metadata().get<std::string>("interface").value_or("");
    const auto mac_str = hololink_channels[0].enumeration_metadata().get<std::string>("mac_id").value_or("");
    const auto mac_bytes_arr = parse_mac_address(mac_str);
    const std::vector<uint8_t> mac_bytes(mac_bytes_arr.begin(), mac_bytes_arr.end());
    HOLOSCAN_LOG_INFO("Using interface={}, mac={}", interface, mac_str);

    auto hololink_dev = hololink_channels[0].hololink();
    hololink_dev->start();
    hololink_dev->reset();

    std::vector<std::shared_ptr<hololink::sensors::NativeImx274Sensor>> cameras;
    for (std::size_t i = 0; i < hololink_channels.size(); ++i) {
        // The second arg selects the I2C expander output (OUTPUT_1 for sensor 0, OUTPUT_2 for
        // sensor 1) and must match the data channel's sensor id, not the loop index --
        // otherwise --sensor 1 drives sensor 0's I2C path.
        auto camera = std::make_shared<hololink::sensors::NativeImx274Sensor>(
            hololink_channels[i], sensor_ids[i]);
        if (i == 0) {
            camera->setup_clock();
        }
        camera->configure(camera_mode);
        camera->set_digital_gain_reg(4);
        cameras.push_back(camera);
    }

    // Build the graph.
    holoscan::Graph graph { "fusa_coe_imx274_player" };
    holoscan::CompileOptions compile_options;

    // Pass 1: declare one image layer per camera, sized to that camera's own frame -- the
    // LayerStack must be complete before HolovizOp is constructed.
    holoscan::visualization::LayerStack layers;
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        layers.add_image_layer(fmt::format("video_{}", i),
            hx::rgba16_layer(static_cast<uint32_t>(cameras[i]->get_width()),
                static_cast<uint32_t>(cameras[i]->get_height()), hx::side_by_side_view(i, cameras.size())));
    }

    const uint32_t window_width = 960U * static_cast<uint32_t>(cameras.size());
    const uint32_t window_height = 540U;
    // HolovizOp renders once per frame of the first sensor (layer 0); with both sensors, the
    // second one's layer shows its newest frame.
    auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers),
        hx::DisplayConfig {
            .width = window_width,
            .height = window_height,
            .fullscreen = fullscreen,
            .headless = headless,
        },
        compile_options, cuda_device_ordinal);

    std::vector<std::string_view> tensor_output_operators;

    // Pass 2: build each camera's capture/convert/process/demosaic chain and wire it into viz.
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        const std::string name = std::to_string(i);
        auto camera = cameras[i];

        const auto pixel_format = camera->get_pixel_format();
        const auto bayer_format = camera->get_bayer_format();

        // The packetizer only has programs for RAW_10 and RAW_12, so an 8-bit mode is always
        // captured unpacketized.
        const bool packetizer_enabled = !no_packetizer
            && pixel_format != hololink::csi::PixelFormat::RAW_8;

        // Resolve geometry ONCE using FusaCoeCaptureOp's HSB-alignment-aware adapter, and reuse
        // that same geometry for CsiToBayerParams below when the unpacketized path is taken.
        // FUSA's 64-byte HSB line alignment differs from the generic adapter CameraSensor
        // otherwise uses (8-byte alignment).
        hololink::csi::CsiGeometry geometry;
        camera->configure_converter(hololink::operators::FusaCoeCaptureOp::make_geometry_adapter(
            geometry, packetizer_enabled));

        auto capture = graph.op<hololink::operators::FusaCoeCaptureOp>(
            fmt::format("capture_{}", name));
        capture.set_params(hololink::operators::FusaCoeCaptureParams {
            .hololink_channel = &hololink_channels[i],
            .device_start = [camera] { camera->start(); },
            .device_stop = [camera] { camera->stop(); },
            .interface = interface,
            .mac_addr = mac_bytes,
            .timeout = timeout_ms,
            // A camera that sends nothing fails a --frame-limit run instead of keeping it going.
            .stall_timeout_ms = hx::frame_limit_stall_timeout_ms(frame_limit, timeout_ms),
            .pixel_format = pixel_format,
            .bayer_format = bayer_format,
            .packetizer_enabled = packetizer_enabled,
            .geometry = geometry,
            // Stop after --frame-limit frames, as HSDK4's CountCondition on the receiver did.
            .frame_limit = frame_limit,
            .frame_limit_ends_run = true,
            // Names this camera in the capture's stall messages.
            .name = std::string(capture.name()),
        });

        if (print_metadata) {
            auto printer = graph.op<MetadataPrinterOp>(fmt::format("metadata_printer_{}", name));
            printer.set_params(MetadataPrinterParams { .label = fmt::format("sensor_{}", name) });
            // A printer that falls behind skips metadata rather than making the capture drop
            // the metadata of frames it has already published.
            graph.add_flow(capture->metadata, printer->input_metadata,
                holoscan::ConnectionOptions {
                    .queue_policy = holoscan::QueuePolicy::kDropOldest,
                    .queue_depth = 4U,
                });
        }

        auto image_processor = graph.op<hololink::operators::ImageProcessorOp>(
            fmt::format("image_processor_{}", name));
        image_processor.set_params(hololink::operators::ImageProcessorParams {
            .pixel_format = int(pixel_format),
            .bayer_format = int(bayer_format),
            .optical_black = 0,
            .cuda_device_ordinal = cuda_device_ordinal,
            .pixel_width = geometry.pixel_width,
            .pixel_height = geometry.pixel_height,
        });

        const auto bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(bayer_format);
        auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>(fmt::format("bayer_demosaic_{}", name));
        demosaic.set_params(holoscan::vision::BayerDemosaicOpParams {
            .bayer_grid_pos = bayer_grid_pos,
            // HolovizOp's RGBA16 layer needs an alpha channel: make it fully opaque.
            .generate_alpha = true,
            .alpha_value = 65535U,
        });

        if (packetizer_enabled) {
            auto converter = graph.op<hololink::operators::PackedFormatConverterOp>(
                fmt::format("packed_format_converter_{}", name));
            converter.set_params(hololink::operators::PackedFormatConverterParams {
                .cuda_device_ordinal = cuda_device_ordinal,
                .bayer_format = bayer_format,
                .geometry = geometry,
            });
            graph.add_flow(capture->tensor, converter->input_tensor, hx::pipeline_edge());
            graph.add_flow(capture->metadata, converter->input_metadata, hx::pipeline_edge());
            graph.add_flow(converter->output_tensor, image_processor->input_tensor, hx::pipeline_edge());
            graph.add_flow(converter->output_metadata, image_processor->input_metadata, hx::pipeline_edge());
            tensor_output_operators.push_back(converter.name());
        } else {
            auto converter = graph.op<hololink::operators::CsiToBayerOp>(
                fmt::format("csi_to_bayer_{}", name));
            converter.set_params(hololink::operators::CsiToBayerParams {
                .cuda_device_ordinal = cuda_device_ordinal,
                .geometry = geometry,
                .bayer_format = bayer_format,
                // Fed by FusaCoeCaptureOp, which already strips the CSI header before emitting.
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

    // Every capture/converter/image_processor emits its tensor on a port named "tensor";
    // bayer_demosaic emits its image on "output".
    for (const auto& operator_path : tensor_output_operators) {
        compile_options.deployment.bind_tensor_output_device(
            holoscan::TensorOutputDevicePlacement {
                .operator_path = operator_path,
                .output_port = "tensor",
                .device = holoscan::DeviceId { cuda_device_ordinal },
            });
    }
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        compile_options.deployment.bind_tensor_output_device(
            holoscan::TensorOutputDevicePlacement {
                .operator_path = fmt::format("bayer_demosaic_{}", i),
                .output_port = "output",
                .device = holoscan::DeviceId { cuda_device_ordinal },
            });
    }

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

    hololink::logging::hsb_log_level = hololink::logging::HSB_LOG_LEVEL_INFO;

    // clang-format off
    options_description.add_options()
        ("camera-mode", value<int>()->default_value(
            static_cast<int>(hololink::sensors::imx274_mode::IMX274_MODE_1920X1080_60FPS)),
            "IMX274 mode (default: 1; 0 and 2 are 3840x2160 and require --sensor 0 or 1)")
        ("frame-limit", value<int>()->default_value(0), "Exit after receiving this many frames")
        ("fullscreen", bool_switch()->default_value(false), "Run in fullscreen mode")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("hololink", value<std::string>()->default_value(hololink::env_hololink_ip(0, "192.168.0.2")), "IP address of Hololink board")
        ("sensor", value<int>()->default_value(-1), "Sensor to use (0 or 1, or -1 (default) for stereo mode)")
        ("timeout", value<int>()->default_value(1500), "Capture request timeout, in milliseconds")
        ("metadata", bool_switch()->default_value(false), "Print frame metadata")
        ("no-packetizer", bool_switch()->default_value(false), "Capture unpacketized CSI-2 data instead of HSB-packed data")
        ;
    // clang-format on

    auto vm = Parser().parse_command_line(argc, argv, options_description);

    const int sensor = vm["sensor"].as<int>();
    if (sensor != -1 && sensor != 0 && sensor != 1) {
        HOLOSCAN_LOG_ERROR("sensor value must be 0 or 1 (given {})", sensor);
        return -1;
    }

    // Two 4k streams exceed the network bandwidth and take the interface down.
    const int camera_mode_value = vm["camera-mode"].as<int>();
    if (sensor == -1
        && (camera_mode_value == hololink::sensors::imx274_mode::IMX274_MODE_3840X2160_60FPS
            || camera_mode_value == hololink::sensors::imx274_mode::IMX274_MODE_3840X2160_30FPS_12BITS)) {
        HOLOSCAN_LOG_ERROR("--camera-mode {} (3840x2160) requires a single camera; "
                           "select one with --sensor 0 or --sensor 1",
            camera_mode_value);
        return -1;
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the cameras and the board
    // are stopped on the way out.
    return hx::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
        try {
            const auto camera_mode = static_cast<hololink::sensors::imx274_mode::Mode>(vm["camera-mode"].as<int>());

            return run(
                vm["hololink"].as<std::string>(),
                sensor,
                camera_mode,
                static_cast<uint32_t>(vm["timeout"].as<int>()),
                vm["frame-limit"].as<int>(),
                vm["headless"].as<bool>(),
                vm["fullscreen"].as<bool>(),
                vm["metadata"].as<bool>(),
                vm["no-packetizer"].as<bool>(),
                signals);
        } catch (const std::exception& e) {
            HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
            return -1;
        }
    });
}
