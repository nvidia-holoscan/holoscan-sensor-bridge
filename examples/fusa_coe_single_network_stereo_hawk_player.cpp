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

// Hawk module stereo (both AR0234 sensors of one GMSL link, FSYNC-synced) live viewer over FuSa
// CoE, one HolovizOp window per sensor:
//   FusaCoeCaptureOp -> SkewProbeOp -> [PackedFormatConverterOp | CsiToBayerOp] ->
//     ImageProcessorOp -> BayerDemosaicOp -> HolovizOp
//
// SkewProbeOp is a pass-through that records each frame's FPGA-receive timestamp (from the
// receiver's CaptureMetadataT) to measure left/right stereo skew; see SkewState below.
//
// hololink_module's Hawk/Ar0234/Max9295d/Max96716a/TauroTechDa326InterfaceV1 sensor driver is
// unchanged; only the receiver (hololink::module::operators::FusaCoeCaptureOp, itself ported to
// HSDK5's Operator<Params> shape) and the downstream converters (native, already-ported
// hololink::operators::{CsiToBayerOp,PackedFormatConverterOp,ImageProcessorOp}) changed. See
// fusa_coe_hawk_single_link_player.cpp for the geometry-bridging rationale.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

#include "hawk.hpp"
#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/fusa_coe_capture_op.hpp"
#include "hololink/module/ptp_pps_output.hpp"
#include "hololink/module/taurotech_da326/taurotech_da326.hpp"
#include "max96716a.hpp"
#include <capture_metadata_traits.hpp>
#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/csi_controller.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
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

// ---------------------------------------------------------------------------
// Stereo skew measurement: pairs left/right frames by nearest FPGA-receive timestamp.
// ---------------------------------------------------------------------------

/// Pairs left/right frames by nearest FPGA-receive timestamp; reports stereo skew.
class SkewState {
public:
    static constexpr int64_t HALF_FRAME_NS = 8'333'333LL; // half of 1/60s

    void record(const std::string& label, int64_t ts_ns)
    {
        std::lock_guard lock(mutex_);
        auto& own = (label == "left") ? left_ : right_;
        auto& other = (label == "left") ? right_ : left_;
        if (!other.empty()) {
            size_t best_idx = 0;
            int64_t best_diff = std::llabs(other[0] - ts_ns);
            for (size_t i = 1; i < other.size(); ++i) {
                int64_t d = std::llabs(other[i] - ts_ns);
                if (d < best_diff) {
                    best_diff = d;
                    best_idx = i;
                }
            }
            if (best_diff <= HALF_FRAME_NS) {
                int64_t other_ts = other[best_idx];
                other.erase(other.begin() + best_idx);
                if (label == "right") {
                    deltas_ns_.push_back(ts_ns - other_ts);
                } else {
                    deltas_ns_.push_back(other_ts - ts_ns);
                }
                int64_t cutoff = std::min(ts_ns, other_ts);
                left_.erase(std::remove_if(left_.begin(), left_.end(),
                                [cutoff](int64_t t) { return t < cutoff; }),
                    left_.end());
                right_.erase(std::remove_if(right_.begin(), right_.end(),
                                 [cutoff](int64_t t) { return t < cutoff; }),
                    right_.end());
                return;
            }
        }
        own.push_back(ts_ns);
        if (own.size() > 16) {
            own.erase(own.begin());
        }
    }

    std::string summary() const
    {
        std::lock_guard lock(mutex_);
        const size_t n = deltas_ns_.size();
        if (n == 0)
            return "no paired frames";
        const int64_t sum = std::accumulate(deltas_ns_.begin(), deltas_ns_.end(), int64_t { 0 });
        const double avg_us = std::abs(static_cast<double>(sum) / n) / 1000.0;
        return fmt::format("frames={}, average={:.2f} us", n, avg_us);
    }

private:
    // Both legs' probes record from their own threads.
    mutable std::mutex mutex_;
    std::vector<int64_t> left_;
    std::vector<int64_t> right_;
    std::vector<int64_t> deltas_ns_;
};

// Pass-through operator that records each frame's FPGA-receive timestamp (from the typed
// CaptureMetadataT metadata port) and re-emits both tensor and metadata unchanged.
struct SkewProbeOpParams {
    std::string label;
    SkewState* state = nullptr;
};

class SkewProbeOp : public holoscan::Operator<SkewProbeOpParams> {
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
            hx::count_mismatched_frame(mismatched_frames_, "SkewProbeOp");
            return {};
        }
        auto& paired = **maybe_paired;

        const auto& p = params();
        if (p.state != nullptr) {
            const int64_t ts_ns = paired.metadata.data.timestamp_s * 1'000'000'000LL
                + paired.metadata.data.timestamp_ns;
            p.state->record(p.label, ts_ns);
        }

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
};

// Globals so a std::terminate handler can still emit the skew summary when an exception
// escapes a worker thread (e.g. FusaCoeCaptureOp shutdown error).
SkewState* g_skew_state = nullptr;
std::terminate_handler g_prev_terminate = nullptr;

[[noreturn]] void skew_terminate_handler()
{
    if (g_skew_state) {
        std::cout << "Stereo skew: " << g_skew_state->summary() << std::endl;
        g_skew_state = nullptr;
    }
    if (g_prev_terminate) {
        g_prev_terminate();
    }
    std::abort();
}

// Runs the player on the options main() parsed.
int run(hololink::args::VariablesMap& variables_map, holoscan::ManagedShutdownSignals& signals)
{
    const auto hololink_ip = variables_map["hololink"].as<std::string>();
    const auto channel_arg = variables_map["channel"].as<std::string>();
    const auto interface_arg = variables_map["interface"].as<std::string>();
    const auto title = variables_map["title"].as<std::string>();
    const auto headless = variables_map["headless"].as<bool>();
    const auto skip_reset = variables_map["skip-reset"].as<bool>();
    const auto no_packetizer = variables_map["no-packetizer"].as<bool>();
    const auto disable_sync = variables_map["disable-sync"].as<bool>();
    const int64_t frame_limit = variables_map["frame-limit"].as<int>();
    const int32_t width = variables_map["width"].as<int>();
    const int32_t height = variables_map["height"].as<int>();
    const uint32_t timeout_ms = static_cast<uint32_t>(variables_map["timeout"].as<int>());
    const uint16_t exposure = static_cast<uint16_t>(variables_map["exposure"].as<int>());
    const int32_t pattern = variables_map["pattern"].as<int>();
    const auto camera_mode_int = variables_map["camera-mode"].as<int>();

    std::shared_ptr<hololink::module::HololinkInterfaceV1> holo_iface;
    SkewState skew_state;

    try {
        // Both SIFs always — stereo.
        std::vector<int> sif_list = { 0, 1 };

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
        for (auto& s : hawk->sensors()) {
            cameras.push_back(s);
        }

        // Bring-up starts here: from now on the stereo skew summary is printed on the way out,
        // also by the terminate handler. As in HSDK4, a failure before this point prints none.
        g_skew_state = &skew_state;
        g_prev_terminate = std::set_terminate(skew_terminate_handler);

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

        // Sensor programming — fsync sync-sink mode unless --disable-sync.
        hawk->configure(camera_mode, /*fsync=*/!disable_sync);
        if (pattern) {
            hawk->test_pattern(static_cast<uint16_t>(pattern));
        }
        hawk->set_exposure_reg(exposure);

        if (!disable_sync) {
            hawk->serializer().set_register(0x02D6, 0x10);
            constexpr unsigned frequency_hz = 60;
            holo_iface->ptp_pps_output()->enable(frequency_hz);
            holo_iface->ptp_pps_output()->start();
            holo_iface->or_uint32(0x70000014u, 1u << 4u);
            holo_iface->and_uint32(0x0000002Cu, ~(1u << 8u));
            deserializer->route_pin_to_gmsl_gpio(link, /*pin=*/8, /*tx_id=*/0x01);
            hawk->serializer().route_gmsl_gpio_to_pin(/*pin=*/9, /*rx_id=*/0x01);
            hawk->serializer().route_gmsl_gpio_to_pin(/*pin=*/10, /*rx_id=*/0x01);
        } else {
            HOLOSCAN_LOG_INFO("Sync disabled (--disable-sync); sensors free-run.");
        }

        // Same physical NIC interface/MAC for every sensor on this board.
        const std::string interface = interface_arg.empty()
            ? sif_metadatas[0].get<std::string>("interface", "")
            : interface_arg;
        const auto mac_str = sif_metadatas[0].get<std::string>("mac_id", "");
        const auto mac_bytes = parse_mac_address(mac_str);
        HOLOSCAN_LOG_INFO("Using interface={} mac={}", interface, mac_str);

        // Build the graph.
        holoscan::Graph graph { "fusa_coe_single_network_stereo_hawk_player" };
        holoscan::CompileOptions compile_options;

        const std::vector<std::string> sif_name = { "left", "right" };
        const int32_t per_w = width / 2;
        const int32_t per_h = height / 2;
        std::vector<std::string_view> tensor_output_operator_paths;
        std::vector<std::string> image_output_operator_paths;

        for (size_t i = 0; i < cameras.size(); ++i) {
            auto camera = cameras[i];
            const std::string name = std::to_string(i);
            const std::string label = sif_name[i];
            const std::string window_title = title + " - " + label;

            const auto pixel_format = camera->get_pixel_format();
            const auto bayer_format = camera->get_bayer_format();

            const bool packetizer_enabled = !no_packetizer
                && pixel_format != hololink::csi::PixelFormat::RAW_8;

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
                    .headless = headless,
                },
                compile_options, /*cuda_device_ordinal=*/0);

            auto capture = graph.op<hololink::module::operators::FusaCoeCaptureOp>(
                fmt::format("fusa_coe_capture_{}", name));
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

            auto skew_probe = graph.op<SkewProbeOp>("skew_probe_" + label);
            skew_probe.set_params(SkewProbeOpParams { .label = label, .state = &skew_state });

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

            graph.add_flow(capture->tensor, skew_probe->input_tensor, hx::pipeline_edge());
            graph.add_flow(capture->metadata, skew_probe->input_metadata, hx::pipeline_edge());

            if (packetizer_enabled) {
                auto converter = graph.op<hololink::operators::PackedFormatConverterOp>(
                    fmt::format("packed_format_converter_{}", name));
                converter.set_params(hololink::operators::PackedFormatConverterParams {
                    .cuda_device_ordinal = 0,
                    .bayer_format = bayer_format,
                    .geometry = native_geometry,
                });
                graph.add_flow(skew_probe->output_tensor, converter->input_tensor, hx::pipeline_edge());
                graph.add_flow(skew_probe->output_metadata, converter->input_metadata, hx::pipeline_edge());
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
                    .header_included = false,
                });
                graph.add_flow(skew_probe->output_tensor, converter->input_tensor, hx::pipeline_edge());
                graph.add_flow(skew_probe->output_metadata, converter->input_metadata, hx::pipeline_edge());
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
            std::cout << "Stereo skew: " << skew_state.summary() << std::endl;
            g_skew_state = nullptr;
            return -1;
        }

        auto session = holoscan::run_async(plan, signals);
        session.wait();
        const bool failed = hx::report_session(session, plan);

        holo_iface->stop();
        std::cout << "Stereo skew: " << skew_state.summary() << std::endl;
        g_skew_state = nullptr; // suppress duplicate from terminate handler
        return failed ? -1 : 0;
    } catch (const std::exception& e) {
        if (holo_iface) {
            holo_iface->stop();
        }
        if (g_skew_state) {
            std::cout << "Stereo skew: " << skew_state.summary() << std::endl;
            g_skew_state = nullptr;
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
        ("disable-sync", bool_switch()->default_value(false), "Skip VSYNC/FSYNC setup; sensors free-run")
        ("exposure", value<int>()->default_value(0x02DC), "AR0234 exposure register value")
        ("frame-limit", value<int>()->default_value(0), "Exit after this many frames")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("height", value<int>()->default_value(1080), "Total window height (split across left/right)")
        ("hololink", value<std::string>()->default_value(hololink::env_hololink_ip(0, "192.168.0.2")), "IP address of Hololink board")
        ("interface", value<std::string>()->default_value(""), "Ethernet interface (empty = auto)")
        ("no-packetizer", bool_switch()->default_value(false), "Capture unpacketized CSI-2 data instead of HSB-packed data")
        ("pattern", value<int>()->default_value(0), "Sensor test pattern (0 = off)")
        ("skip-reset", bool_switch()->default_value(false), "Skip the Hololink reset step")
        ("timeout", value<int>()->default_value(1500), "Capture request timeout in milliseconds")
        ("title", value<std::string>()->default_value("Hawk stereo (FuSa CoE)"), "Window title")
        ("width", value<int>()->default_value(1920), "Total window width (split across left/right)")
        ;
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);

    // With managed signals, Ctrl+C stops the session cooperatively, so the cameras and the board
    // are stopped and the stereo skew summary printed on the way out.
    return hx::run_with_managed_signals(
        [&](holoscan::ManagedShutdownSignals& signals) { return run(variables_map, signals); });
}
