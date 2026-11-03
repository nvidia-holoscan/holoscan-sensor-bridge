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

// FuSa CoE live viewer for one or both VB1940 sensors, side by side in one window:
//   FusaCoeCaptureOp -> PackedFormatConverterOp (HSB-packed capture) or CsiToBayerOp
//     (--no-packetizer) -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
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
#include <hololink/core/networking.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/fusa_coe_capture/fusa_coe_capture.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/packed_format_converter/packed_format_converter.hpp>
#include <hololink/sensors/camera/vb1940/native_vb1940_sensor.hpp>

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

// Terminates a pipeline when --no-visualizer is given: the frames still need somewhere to go,
// and counting them also shows the capture path actually delivered.
class FrameSinkOp : public holoscan::Operator<> {
public:
    explicit FrameSinkOp(std::string label)
        : label_(std::move(label))
    {
    }

    holoscan::Input<holoscan::schema::ImageT> input { "input" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        (void)spec.input(input, "input").queue_depth(4U).tensor_field(holoscan::schema::ImageTensorFields::data);
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnEach { input });
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override
    {
        auto sample = input.receive();
        if (!sample) {
            return holoscan::make_unexpected(std::move(sample).error());
        }
        if (++frames_ % 30 == 0) {
            HOLOSCAN_LOG_INFO("{}: received {} frames", label_, frames_);
        }
        return {};
    }

private:
    const std::string label_;
    uint64_t frames_ = 0;
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
            throw std::invalid_argument("Each byte must be exactly two hex digits");
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

struct PlayerOptions {
    std::string hololink_ip;
    std::vector<int> sensors; // 0, 1, or both
    hololink::sensors::vb1940_mode::Mode camera_mode;
    uint32_t timeout_ms;
    int frame_limit;
    int bayer_grid_pos; // -1: the sensor's own
    bool headless;
    bool fullscreen;
    bool no_packetizer;
    bool no_visualizer;
};

int run(const PlayerOptions& options, holoscan::ManagedShutdownSignals& signals)
{
    const int cuda_device_ordinal = 0;
    CudaCheck(cuInit(0));
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    cuCtxSetCurrent(cu_context);

    // Get a handle to the Hololink device, with a data channel per sensor.
    auto channel_metadata = hololink::Enumerator::find_channel(options.hololink_ip);
    std::vector<hololink::DataChannel> hololink_channels;
    for (int sensor : options.sensors) {
        auto sensor_metadata = hololink::Metadata(channel_metadata);
        hololink::DataChannel::use_sensor(sensor_metadata, sensor);
        hololink_channels.push_back(hololink::DataChannel(sensor_metadata));
    }

    // Every sensor arrives on the same NIC.
    const auto enumeration_metadata = hololink_channels[0].enumeration_metadata();
    const std::string interface = enumeration_metadata.get<std::string>("interface").value_or("");
    const std::string mac_str = enumeration_metadata.get<std::string>("mac_id").value_or("");
    const auto mac_address = parse_mac_address(mac_str);
    const std::vector<uint8_t> mac_bytes(mac_address.begin(), mac_address.end());
    HOLOSCAN_LOG_INFO("Using interface={}, mac={}", interface, mac_str);

    // Start the HSB.
    std::shared_ptr<hololink::Hololink> hololink = hololink_channels[0].hololink();
    hololink->start();
    hololink->reset();

    // Get handles to and configure the camera(s).
    std::vector<std::shared_ptr<hololink::sensors::NativeVb1940Sensor>> cameras;
    for (std::size_t i = 0; i < hololink_channels.size(); ++i) {
        auto camera = std::make_shared<hololink::sensors::NativeVb1940Sensor>(hololink_channels[i]);
        if (i == 0) {
            camera->setup_clock();
        }
        camera->configure(options.camera_mode);
        cameras.push_back(camera);
    }

    holoscan::Graph graph { "fusa_coe_vb1940_player" };
    holoscan::CompileOptions compile_options;

    // Holoviz renders the images side by side. It is optional: rendering needs a working Vulkan
    // driver, which --headless does not avoid, so --no-visualizer terminates each pipeline at
    // the demosaic instead -- useful for exercising the capture path on a host with no usable
    // GPU driver.
    std::optional<holoscan::OperatorHandle<holoscan::visualization::HolovizOp>> visualizer;
    if (!options.no_visualizer) {
        holoscan::visualization::LayerStack layers;
        for (std::size_t i = 0; i < cameras.size(); ++i) {
            layers.add_image_layer(std::to_string(i),
                hx::rgba16_layer(static_cast<uint32_t>(cameras[i]->get_width()),
                    static_cast<uint32_t>(cameras[i]->get_height()), hx::side_by_side_view(i, cameras.size())));
        }
        visualizer = hx::add_holoviz(graph, "holoviz", std::move(layers),
            hx::DisplayConfig {
                .width = 800U * static_cast<uint32_t>(cameras.size()),
                .height = 620U,
                .fullscreen = options.fullscreen,
                .headless = options.headless,
            },
            compile_options, cuda_device_ordinal);
    }

    std::vector<std::pair<std::string, std::string>> tensor_outputs;

    // Create the capture/conversion pipelines for each sensor.
    const holoscan::ConnectionOptions edge = hx::pipeline_edge();
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        const std::string name = std::to_string(i);
        auto camera = cameras[i];
        const auto pixel_format = camera->get_pixel_format();
        const auto bayer_format = camera->get_bayer_format();

        // --bayer-grid-pos overrides the color filter phase every software stage interprets
        // the mosaic with: ImageProcessorOp's white balance and BayerDemosaicOp's interpolation
        // must agree on which samples are R/G/B. FusaCoeCaptureParams.bayer_format stays the
        // camera's: it programs the capture hardware.
        const auto bayer_grid_pos = options.bayer_grid_pos >= 0
            ? static_cast<hololink::csi::BayerFormat>(options.bayer_grid_pos)
            : bayer_format;

        // The packetizer only has programs for RAW_10 and RAW_12, so an 8-bit mode is always
        // captured unpacketized.
        const bool packetizer_enabled = !options.no_packetizer && pixel_format != hololink::csi::PixelFormat::RAW_8;

        // FuSa's HSB line alignment differs from the generic adapter's, so the capture geometry
        // comes from FusaCoeCaptureOp's adapter and is shared with the converter.
        hololink::csi::CsiGeometry geometry;
        camera->configure_converter(
            hololink::operators::FusaCoeCaptureOp::make_geometry_adapter(geometry, packetizer_enabled));

        // Capture from Fusa.
        auto capture = graph.op<hololink::operators::FusaCoeCaptureOp>("fusa_coe_capture_" + name);
        capture.set_params(hololink::operators::FusaCoeCaptureParams {
            .hololink_channel = &hololink_channels[i],
            .device_start = [camera] { camera->start(); },
            .device_stop = [camera] { camera->stop(); },
            .interface = interface,
            .mac_addr = mac_bytes,
            .timeout = options.timeout_ms,
            // A camera that sends nothing fails a --frame-limit run instead of keeping it going.
            .stall_timeout_ms = hx::frame_limit_stall_timeout_ms(options.frame_limit, options.timeout_ms),
            .pixel_format = pixel_format,
            .bayer_format = bayer_format,
            .packetizer_enabled = packetizer_enabled,
            .geometry = geometry,
            .frame_limit = options.frame_limit,
            .frame_limit_ends_run = true,
            // Names this camera in the capture's stall messages.
            .name = std::string(capture.name()),
        });
        tensor_outputs.emplace_back(capture.name(), "tensor");

        // Perform basic ISP operations.
        auto image_processor = graph.op<hololink::operators::ImageProcessorOp>("image_processor_" + name);
        image_processor.set_params(hololink::operators::ImageProcessorParams {
            .pixel_format = int(pixel_format),
            .bayer_format = int(bayer_grid_pos),
            .optical_black = 0,
            .cuda_device_ordinal = cuda_device_ordinal,
            .pixel_width = geometry.pixel_width,
            .pixel_height = geometry.pixel_height,
        });
        tensor_outputs.emplace_back(image_processor.name(), "tensor");

        // Convert the captured RAW to 16-bit Bayer. Which converter is correct depends on the
        // packetizer: enabled, the buffer holds HSB-packed data; disabled, it is still CSI-2
        // exactly as the sensor sent it, without its header.
        if (packetizer_enabled) {
            auto converter = graph.op<hololink::operators::PackedFormatConverterOp>("packed_format_converter_" + name);
            converter.set_params(hololink::operators::PackedFormatConverterParams {
                .cuda_device_ordinal = cuda_device_ordinal,
                .bayer_format = bayer_grid_pos,
                .geometry = geometry,
            });
            graph.add_flow(capture->tensor, converter->input_tensor, edge);
            graph.add_flow(capture->metadata, converter->input_metadata, edge);
            graph.add_flow(converter->output_tensor, image_processor->input_tensor, edge);
            graph.add_flow(converter->output_metadata, image_processor->input_metadata, edge);
            tensor_outputs.emplace_back(converter.name(), "tensor");
        } else {
            auto converter = graph.op<hololink::operators::CsiToBayerOp>("csi_to_bayer_" + name);
            converter.set_params(hololink::operators::CsiToBayerParams {
                .cuda_device_ordinal = cuda_device_ordinal,
                .geometry = geometry,
                .bayer_format = bayer_grid_pos,
                .header_included = false,
            });
            graph.add_flow(capture->tensor, converter->input_tensor, edge);
            graph.add_flow(capture->metadata, converter->input_metadata, edge);
            graph.add_flow(converter->output_tensor, image_processor->input_tensor, edge);
            graph.add_flow(converter->output_metadata, image_processor->input_metadata, edge);
            tensor_outputs.emplace_back(converter.name(), "tensor");
        }

        // Bayer demosaic to RGBA16.
        auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>("bayer_demosaic_" + name);
        demosaic.set_params(holoscan::vision::BayerDemosaicOpParams {
            .bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(bayer_grid_pos),
            .generate_alpha = true,
            .alpha_value = 65535U,
        });
        tensor_outputs.emplace_back(demosaic.name(), "output");

        graph.add_flow(image_processor->output_tensor, demosaic->input, edge);
        if (visualizer) {
            graph.add_flow(demosaic->output, (*visualizer)->image_layer_input(i), hx::display_edge());
        } else {
            auto sink = graph.op<FrameSinkOp>("sink_" + name, "sink_" + name);
            graph.add_flow(demosaic->output, sink->input, edge);
        }
    }

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    for (const auto& [operator_path, output_port] : tensor_outputs) {
        compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = operator_path,
            .output_port = output_port,
            .device = holoscan::DeviceId { cuda_device_ordinal },
        });
    }
    const auto plan = holoscan::compile(graph, std::move(compile_options));
    if (!plan.ok()) {
        hx::print_compile_failure(plan);
        hololink->stop();
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return -1;
    }

    auto session = holoscan::run_async(plan, signals);
    session.wait();
    const bool failed = hx::report_session(session, plan);

    hololink->stop();
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
        ("camera-mode", value<int>()->default_value(static_cast<int>(hololink::sensors::vb1940_mode::VB1940_MODE_2560X1984_30FPS)), "VB1940 mode (default: 0)")
        ("frame-limit", value<int>()->default_value(0), "Exit after receiving this many frames")
        ("fullscreen", bool_switch()->default_value(false), "Run in fullscreen mode")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("hololink", value<std::string>()->default_value(hololink::env_hololink_ip(0, "192.168.0.2")), "IP address of Hololink board")
        ("no-packetizer", bool_switch()->default_value(false), "Capture unpacketized CSI-2 data instead of HSB-packed data")
        ("no-visualizer", bool_switch()->default_value(false), "Skip Holoviz rendering; run the capture pipeline only")
        ("sensor", value<int>()->default_value(-1), "Sensor to use (0 or 1, or -1 (default) for stereo mode)")
        ("timeout", value<int>()->default_value(1500), "Capture request timeout, in milliseconds")
        ("bayer-grid-pos", value<int>()->default_value(-1), "Override the demosaic CFA registration: 0=BGGR, 1=RGGB, 2=GBRG, 3=GRBG (default: -1, use the sensor-reported value)")
        ;
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);

    PlayerOptions options {
        .hololink_ip = variables_map["hololink"].as<std::string>(),
        .camera_mode = static_cast<hololink::sensors::vb1940_mode::Mode>(variables_map["camera-mode"].as<int>()),
        .timeout_ms = static_cast<uint32_t>(variables_map["timeout"].as<int>()),
        .frame_limit = variables_map["frame-limit"].as<int>(),
        .bayer_grid_pos = variables_map["bayer-grid-pos"].as<int>(),
        .headless = variables_map["headless"].as<bool>(),
        .fullscreen = variables_map["fullscreen"].as<bool>(),
        .no_packetizer = variables_map["no-packetizer"].as<bool>(),
        .no_visualizer = variables_map["no-visualizer"].as<bool>(),
    };
    const int sensor = variables_map["sensor"].as<int>();
    if (sensor == -1) {
        options.sensors = { 0, 1 };
    } else if (sensor == 0 || sensor == 1) {
        options.sensors = { sensor };
    } else {
        HOLOSCAN_LOG_ERROR("sensor value must be 0 or 1 (given {})", sensor);
        return -1;
    }
    if (options.bayer_grid_pos < -1 || options.bayer_grid_pos > 3) {
        HOLOSCAN_LOG_ERROR("bayer-grid-pos must be 0, 1, 2, 3 or -1 (given {})", options.bayer_grid_pos);
        return -1;
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the cameras and the board
    // are stopped on the way out.
    return hx::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
        try {
            return run(options, signals);
        } catch (const std::exception& e) {
            HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
            return -1;
        }
    });
}
