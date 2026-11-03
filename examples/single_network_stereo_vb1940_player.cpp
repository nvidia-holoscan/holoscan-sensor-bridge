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

// Stereo VB1940 live viewer, single RoCE network — two VB1940 sensors via RoceReceiverOp,
// displayed side by side with Holoviz:
//   RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
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

#include <capture_metadata_traits.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/sensors/camera/camera_sensor.hpp>
#include <hololink/sensors/camera/vb1940/native_vb1940_sensor.hpp>

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

// Prints timestamps from the frame metadata sample, then passes both the tensor and metadata
// samples through unchanged.
struct TimestampPrinterParams {
    std::string camera_name { "unknown" };
};

class TimestampPrinterOp : public holoscan::Operator<TimestampPrinterParams> {
public:
    holoscan::Input<holoscan::Tensor> input_tensor { "in_tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "in_metadata" };
    holoscan::Output<holoscan::Tensor> output_tensor { "tensor" };
    holoscan::Output<hololink::CaptureMetadataT> output_metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input(input_tensor, "in_tensor")
            .expects_tensor(holoscan::TensorInputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                },
            })
            .queue_depth(4U);
        spec.input(input_metadata, "in_metadata").queue_depth(4U);
        spec.output(output_tensor, "tensor");
        spec.output(output_metadata, "metadata");
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnAligned {
                      .ports = holoscan::ports(input_tensor, input_metadata),
                      .window = hololink::operators::kMetadataPairingWindow,
                  })
            .input(input_tensor, holoscan::Exact {})
            .input(input_metadata, holoscan::Exact {});
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext&) override
    {
        auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(input_tensor, input_metadata);
        if (!maybe_paired)
            return holoscan::make_unexpected(std::move(maybe_paired).error());
        if (!*maybe_paired) {
            hx::count_mismatched_frame(mismatched_frames_, "TimestampPrinterOp");
            return {};
        }
        auto& paired = **maybe_paired;

        ++frame_count_;
        const auto& p = params();
        int64_t timestamp_s = paired.metadata.data.timestamp_s;
        int64_t timestamp_ns = paired.metadata.data.timestamp_ns;
        double total_seconds = timestamp_s + timestamp_ns / 1e9;
        HOLOSCAN_LOG_INFO("[{}] Frame {}: timestamp={:.9f}", p.camera_name, frame_count_, total_seconds);

        // Re-emitting each received Sample directly (not building fresh EmitOptions) already
        // preserves its original capture_time/frame_id -- no re-minting needed here.
        if (auto result = output_tensor.emit(paired.tensor); !result) {
            return holoscan::make_unexpected(std::move(result).error());
        }
        if (auto result = output_metadata.emit(paired.metadata); !result) {
            return holoscan::make_unexpected(std::move(result).error());
        }
        return {};
    }

private:
    hololink::operators::DropCounter mismatched_frames_;
    int frame_count_ = 0;
};

int run_player(hololink::args::VariablesMap& variables_map, holoscan::ManagedShutdownSignals& signals)
{
    // Empty: the receivers use the device on the board's network interface.
    const std::string ibv_name = variables_map["ibv-name"].as<std::string>();

    const int cuda_device_ordinal = 0;
    CudaCheck(cuInit(0));
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    cuCtxSetCurrent(cu_context);

    const std::string hololink_ip = variables_map["hololink"].as<std::string>();
    auto channel_metadata_raw = hololink::Enumerator::find_channel(hololink_ip);

    std::vector<std::shared_ptr<hololink::DataChannel>> hololink_channels;
    for (int sensor_id : { 0, 1 }) {
        auto channel_metadata = hololink::Metadata(channel_metadata_raw);
        hololink::DataChannel::use_sensor(channel_metadata, sensor_id);
        hololink_channels.push_back(std::make_shared<hololink::DataChannel>(channel_metadata));
    }

    auto hololink_dev = hololink_channels[0]->hololink();

    // Before the board starts, so that an unsupported --frequency fails before anything is reset.
    auto vsync = hololink::Synchronizer::null_synchronizer();
    const bool trigger_mode = variables_map["trigger"].as<bool>();
    if (trigger_mode) {
        const auto frequency = static_cast<unsigned>(variables_map["frequency"].as<int>());
        vsync = hololink_dev->ptp_pps_output(frequency);
    }

    hololink_dev->start();
    HOLOSCAN_LOG_INFO("Before hololink.reset()");
    hololink_dev->reset();

    std::vector<std::shared_ptr<hololink::sensors::NativeVb1940Sensor>> cameras;
    for (auto& channel : hololink_channels) {
        cameras.push_back(std::make_shared<hololink::sensors::NativeVb1940Sensor>(*channel, vsync));
    }

    const auto camera_mode = static_cast<hololink::sensors::vb1940_mode::Mode>(
        variables_map["camera-mode"].as<int>());

    // Set RESET low for all sensors
    hololink_dev->write_uint32(0x8, 0x0);
    cameras[0]->setup_clock(); // this also sets camera_right's clock
    // Release RESET for all sensors -- 0x3 for stereo mode
    hololink_dev->write_uint32(0x8, 0x3);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    for (size_t i = 0; i < cameras.size(); ++i) {
        cameras[i]->get_register_32(0x0000); // DEVICE_MODEL_ID:"S940"(ASCII code:0x53393430)
        cameras[i]->get_register_32(0x0734); // EXT_CLOCK(25MHz = 0x017d7840)

        cameras[i]->configure(camera_mode);
        cameras[i]->set_analog_gain_reg(variables_map["gain"].as<int>());
        cameras[i]->set_exposure_reg(variables_map["exp"].as<int>());
    }

    try {
        auto cal_eeprom = cameras[0]->get_rgb_calibration_data();
        HOLOSCAN_LOG_INFO("Camera calibration data retrieved successfully: \n{}", cal_eeprom.to_string());
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_WARN("Failed to read camera calibration data: {}", e.what());
    }

    // Build the graph.
    holoscan::Graph graph { "single_network_stereo_vb1940_player" };

    const std::vector<std::string> tensor_names = { "left", "right" };
    std::vector<hololink::csi::CsiGeometry> geometries(cameras.size());
    std::vector<size_t> frame_sizes(cameras.size());
    for (size_t i = 0; i < cameras.size(); ++i) {
        cameras[i]->configure_converter(geometries[i]);
        frame_sizes[i] = geometries[i].start_byte
            + geometries[i].bytes_per_line * geometries[i].pixel_height
            + geometries[i].trailing_bytes;
    }
    if (frame_sizes[0] != frame_sizes[1]) {
        throw std::runtime_error("Both cameras must have the same frame size");
    }

    holoscan::CompileOptions compile_options;

    // Pass 1: declare one image layer per camera, sized to that camera's own frame -- the
    // LayerStack must be complete before HolovizOp is constructed.
    holoscan::visualization::LayerStack layers;
    for (size_t i = 0; i < cameras.size(); ++i) {
        layers.add_image_layer(fmt::format("video_{}", tensor_names[i]),
            hx::rgba16_layer(static_cast<uint32_t>(cameras[i]->get_width()),
                static_cast<uint32_t>(cameras[i]->get_height()),
                holoscan::visualization::ViewTransform {
                    .offset_x = 0.5f * static_cast<float>(i),
                    .offset_y = 0.f,
                    .scale_x = 0.5f,
                    .scale_y = 1.f,
                }));
    }

    std::string window_title = "Holoviz - " + hololink_ip;
    if (variables_map.count("title")) {
        window_title = variables_map["title"].as<std::string>();
    }

    // HolovizOp renders once per left image (layer 0); the right layer shows its newest image,
    // which at a low --frequency can be the previous trigger's.
    auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers),
        hx::DisplayConfig {
            .title = window_title,
            .width = variables_map["window-width"].as<uint32_t>(),
            .height = variables_map["window-height"].as<uint32_t>(),
            .headless = variables_map["headless"].as<bool>(),
            .max_input_age = hx::max_input_age_for_rate(
                trigger_mode ? static_cast<unsigned>(variables_map["frequency"].as<int>()) : 0U),
        },
        compile_options, cuda_device_ordinal);

    const bool print_timestamps = variables_map["print-time"].as<bool>();
    const uint32_t ibv_port = variables_map["ibv-port"].as<uint32_t>();

    std::vector<std::string_view> tensor_output_operators;

    for (size_t i = 0; i < cameras.size(); ++i) {
        const std::string& name = tensor_names[i];
        auto camera = cameras[i];

        auto receiver = graph.op<hololink::operators::RoceReceiverOp>(
            fmt::format("receiver_{}", name),
            hololink_channels[i],
            [camera] { camera->start(); },
            [camera] { camera->stop(); });
        receiver.set_params(hololink::operators::RoceReceiverParams {
            .frame_context = cu_context,
            .frame_size = frame_sizes[i],
            .ibv_name = ibv_name,
            .ibv_port = ibv_port,
            // Stop after --frame-limit frames, as HSDK4's CountCondition on each receiver did.
            .frame_limit = variables_map["frame-limit"].as<int>(),
            .frame_limit_ends_run = true,
        });

        auto csi = graph.op<hololink::operators::CsiToBayerOp>(fmt::format("csi_to_bayer_{}", name));
        csi.set_params(hololink::operators::CsiToBayerParams {
            .cuda_device_ordinal = cuda_device_ordinal,
            .geometry = geometries[i],
            .bayer_format = camera->get_bayer_format(),
        });

        auto ip = graph.op<hololink::operators::ImageProcessorOp>(fmt::format("image_processor_{}", name));
        ip.set_params(hololink::operators::ImageProcessorParams {
            .pixel_format = int(camera->get_pixel_format()),
            .bayer_format = int(camera->get_bayer_format()),
            .optical_black = 8, // Optical black value for vb1940 is 8 for RAW10
            .cuda_device_ordinal = cuda_device_ordinal,
            .pixel_width = geometries[i].pixel_width,
            .pixel_height = geometries[i].pixel_height,
        });

        const auto bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(camera->get_bayer_format());
        auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>(fmt::format("bayer_demosaic_{}", name));
        demosaic.set_params(holoscan::vision::BayerDemosaicOpParams {
            .bayer_grid_pos = bayer_grid_pos,
            .generate_alpha = true,
            .alpha_value = 65535U,
        });

        if (print_timestamps) {
            auto printer = graph.op<TimestampPrinterOp>(fmt::format("timestamp_printer_{}", name));
            printer.set_params(TimestampPrinterParams { .camera_name = name });
            graph.add_flow(receiver->tensor, printer->input_tensor, hx::pipeline_edge());
            graph.add_flow(receiver->metadata, printer->input_metadata, hx::pipeline_edge());
            graph.add_flow(printer->output_tensor, csi->input_tensor, hx::pipeline_edge());
            graph.add_flow(printer->output_metadata, csi->input_metadata, hx::pipeline_edge());
        } else {
            graph.add_flow(receiver->tensor, csi->input_tensor, hx::pipeline_edge());
            graph.add_flow(receiver->metadata, csi->input_metadata, hx::pipeline_edge());
        }
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
                .device = holoscan::DeviceId { cuda_device_ordinal },
            });
    }
    for (const auto& name : tensor_names) {
        compile_options.deployment.bind_tensor_output_device(
            holoscan::TensorOutputDevicePlacement {
                .operator_path = fmt::format("bayer_demosaic_{}", name),
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
    OptionsDescription options_description("Stereo VB1940 Player Options");

    // clang-format off
    options_description.add_options()
        ("camera-mode", value<int>()->default_value(static_cast<int>(hololink::sensors::vb1940_mode::VB1940_MODE_2560X1984_30FPS)), "VB1940 mode (default: 0)")
        ("trigger", bool_switch()->default_value(false), "Run in trigger mode for camera synchronization")
        ("exp", value<int>()->default_value(256), "Set EXPOSURE duration in lines, RANGE(4 to 65535). Default line value is 29.70usec")
        ("gain", value<int>()->default_value(0), "Set Analog Gain, RANGE(0 to 12). Default is 0. Equation is (16/(16-gain))")
        ("frequency", value<int>()->default_value(30), "VSYNC frequency in Hz (10, 30, 60, 90, 120). Default is 30Hz")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("frame-limit", value<int>()->default_value(0), "Exit after receiving this many frames")
        ("configuration", value<std::string>(), "Configuration file")
        ("hololink", value<std::string>()->default_value(hololink::env_hololink_ip(0, "192.168.0.2")), "IP address of Hololink board")
        ("ibv-name", value<std::string>()->default_value(""), "IBV device to use (default: the device on the board's network interface)")
        ("ibv-port", value<uint32_t>()->default_value(1), "Port number of IBV device, with --ibv-name (default: 1)")
        ("window-height", value<uint32_t>()->default_value(2160 / 4), "Set the height of the displayed window")
        ("window-width", value<uint32_t>()->default_value(3840 / 3), "Set the width of the displayed window")
        ("title", value<std::string>(), "Set the window title")
        ("print-time", bool_switch()->default_value(false), "Print timestamp information for received frames")
        ;
    // clang-format on

    hololink::logging::hsb_log_level = hololink::logging::HSB_LOG_LEVEL_INFO;

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);
    if (variables_map.count("configuration")
        && !hx::check_configuration(variables_map["configuration"].as<std::string>())) {
        return -1;
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the cameras and the board
    // are stopped on the way out.
    return hx::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
        try {
            return run_player(variables_map, signals);
        } catch (const std::exception& e) {
            HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
            return -1;
        }
    });
}
