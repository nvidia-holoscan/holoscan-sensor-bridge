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

// FuSa CoE frame saver — capture N frames per leg from a VB1940 sensor (mono or
// stereo) via FusaCoeCaptureOp and write them to disk using the HSDK5
// holoscan::Graph API. No HolovizOp/BayerDemosaicOp dependency — this is a
// headless capture-correctness smoke test, not a viewer.
//
// Each frame produces two files, under a per-sensor subdirectory:
//   sensor<N>/frame_<seq>.raw — raw bytes from the device tensor (packed CSI-2 or HSB-packed Bayer)
//   sensor<N>/frame_<seq>.txt — metadata sidecar (frame number, hardware timestamps, CRC, geometry)
//
// Usage:
//   fusa_coe_frame_saver [OPTIONS]
//
// Options:
//   --hololink IP      IP address of Hololink board (default: 192.168.0.2)
//   --sensor N         Sensor index: 0, 1, or -1 for both legs (stereo) (default: -1)
//   --camera-mode N    VB1940 mode (default: 0 = 2560x1984@30fps)
//   --frames N         Number of frames to save per leg then stop (default: 5)
//   --output-dir DIR   Output directory; created if absent (default: /tmp/fusa_frames)
//   --timeout N        Per-frame capture timeout in milliseconds (default: 1500)
//   --timeout-s S      Wall-clock seconds to wait for all frames (default: 60)
//   --no-packetizer    Capture raw CSI-2 instead of HSB-packed data

#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/dlpack.hpp>
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
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/packetizer_program.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/fusa_coe_capture/fusa_coe_capture.hpp>
#include <hololink/sensors/camera/vb1940/native_vb1940_sensor.hpp>

#define HOLOLINK_PLAYER_SUPPORT_NO_DISPLAY
#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

// ---------------------------------------------------------------------------
// Save progress
// ---------------------------------------------------------------------------

// How many frames one leg's saver has written, for the report after the run.
struct SaveStatus {
    std::atomic<std::uint32_t> frames_saved { 0 };
};

static std::string pad8(std::uint64_t n)
{
    auto s = std::to_string(n);
    if (s.size() < 8) {
        s.insert(0, 8 - s.size(), '0');
    }
    return s;
}

// ---------------------------------------------------------------------------
// FrameSaverOp
// ---------------------------------------------------------------------------

struct FrameSaverParams {
    std::string output_dir { "/tmp/fusa_frames" };
    // --frames, shared by every leg's saver: the last one to save that many completes the run.
    std::shared_ptr<hx::FrameLimitGroup> frame_limit;
    std::shared_ptr<SaveStatus> status;
    // Frame geometry, resolved from the sensor before the graph runs.
    std::uint32_t pixel_width { 0 };
    std::uint32_t pixel_height { 0 };
    std::string pixel_format; // "RAW_8", "RAW_10", "RAW_12"
    std::string bayer_format; // "RGGB", "BGGR", "GBRG", "GRBG"
    bool packetizer_enabled { false };
};

class FrameSaverOp final : public holoscan::Operator<FrameSaverParams> {
public:
    holoscan::Input<holoscan::Tensor> input_tensor { "tensor" };
    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input(input_tensor, "tensor").queue_depth(4U);
        spec.input(input_metadata, "metadata").queue_depth(4U);
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        // Upstream stamps both ports with an identical capture_time, so OnAligned pairs them on
        // an exact match (see kMetadataPairingWindow).
        holoscan::Contract c;
        c.trigger(holoscan::OnAligned {
                      .ports = holoscan::ports(input_tensor, input_metadata),
                      .window = hololink::operators::kMetadataPairingWindow,
                  })
            .input(input_tensor, holoscan::Exact {})
            .input(input_metadata, holoscan::Exact {});
        return c;
    }

    void start() override
    {
        const auto& p = params();
        std::filesystem::create_directories(p.output_dir);
        HOLOSCAN_LOG_INFO("FrameSaverOp: saving up to {} frames to {}",
            p.frame_limit ? p.frame_limit->limit() : 0, p.output_dir);
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override
    {
        const auto& p = params();

        auto tensor_sample = input_tensor.receive();
        if (!tensor_sample) {
            return holoscan::make_unexpected(std::move(tensor_sample).error());
        }
        auto metadata_sample = input_metadata.receive();
        if (!metadata_sample) {
            return holoscan::make_unexpected(std::move(metadata_sample).error());
        }

        // Drain without writing once this leg has saved enough frames.
        if (leg_done_) {
            return {};
        }

        const hololink::CaptureMetadataT& frame = metadata_sample->data;
        const holoscan::Tensor& tensor = tensor_sample->data;
        const std::size_t nbytes = tensor.nbytes();

        // Copy from device to a host staging buffer.
        std::vector<std::uint8_t> host_buf(nbytes);
        if (tensor.device().device_type == kDLCUDA) {
            cudaError_t err = cudaMemcpyAsync(
                host_buf.data(), tensor.data(), nbytes,
                cudaMemcpyDeviceToHost,
                context.cuda_stream().get());
            if (err != cudaSuccess) {
                return holoscan::make_unexpected(holoscan::Error {
                    holoscan::ErrorCode::kFailure,
                    "FrameSaverOp: cudaMemcpyAsync failed" });
            }
            err = cudaStreamSynchronize(context.cuda_stream().get());
            if (err != cudaSuccess) {
                return holoscan::make_unexpected(holoscan::Error {
                    holoscan::ErrorCode::kFailure,
                    "FrameSaverOp: cudaStreamSynchronize failed" });
            }
        } else {
            std::memcpy(host_buf.data(), tensor.data(), nbytes);
        }

        const std::uint64_t seq = static_cast<std::uint64_t>(frame.frame_number);
        const std::string base = p.output_dir + "/frame_" + pad8(seq);

        // Write raw bytes.
        {
            std::ofstream f(base + ".raw", std::ios::binary);
            if (!f) {
                return holoscan::make_unexpected(holoscan::Error {
                    holoscan::ErrorCode::kFailure,
                    "FrameSaverOp: cannot open " + base + ".raw" });
            }
            f.write(reinterpret_cast<const char*>(host_buf.data()),
                static_cast<std::streamsize>(nbytes));
            if (!f) {
                return holoscan::make_unexpected(holoscan::Error {
                    holoscan::ErrorCode::kFailure,
                    "FrameSaverOp: write failed for " + base + ".raw" });
            }
        }

        // Write metadata sidecar.
        {
            std::ofstream f(base + ".txt");
            if (!f) {
                return holoscan::make_unexpected(holoscan::Error {
                    holoscan::ErrorCode::kFailure,
                    "FrameSaverOp: cannot open " + base + ".txt" });
            }
            f << "frame_number=" << frame.frame_number << "\n"
              << "timestamp_s=" << frame.timestamp_s << "\n"
              << "timestamp_ns=" << frame.timestamp_ns << "\n"
              << "metadata_s=" << frame.metadata_s << "\n"
              << "metadata_ns=" << frame.metadata_ns << "\n"
              << "crc=0x" << std::hex << frame.crc << std::dec << "\n"
              << "bytes=" << nbytes << "\n"
              << "pixel_width=" << p.pixel_width << "\n"
              << "pixel_height=" << p.pixel_height << "\n"
              << "bytes_per_line=" << (p.pixel_height > 0 ? nbytes / p.pixel_height : 0) << "\n"
              << "pixel_format=" << p.pixel_format << "\n"
              << "bayer_format=" << p.bayer_format << "\n"
              << "packetizer_enabled=" << (p.packetizer_enabled ? "1" : "0") << "\n";
            if (!f) {
                return holoscan::make_unexpected(holoscan::Error {
                    holoscan::ErrorCode::kFailure,
                    "FrameSaverOp: write failed for " + base + ".txt" });
            }
        }

        ++frames_saved_;
        HOLOSCAN_LOG_INFO("FrameSaverOp: saved {}.raw ({} bytes, {}x{} {})",
            base, nbytes, p.pixel_width, p.pixel_height, p.pixel_format);

        if (p.status) {
            p.status->frames_saved.store(frames_saved_);
        }
        if (p.frame_limit && p.frame_limit->limit() > 0 && frames_saved_ >= p.frame_limit->limit()) {
            leg_done_ = true;
            if (p.frame_limit->leg_done()) {
                context.request_completion();
            }
        }

        return {};
    }

private:
    std::uint32_t frames_saved_ { 0 };
    bool leg_done_ { false };
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static hololink::core::MacAddress parse_mac_address(const std::string& mac_str)
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

static std::string pixel_format_name(hololink::csi::PixelFormat fmt)
{
    switch (fmt) {
    case hololink::csi::PixelFormat::RAW_8:
        return "RAW_8";
    case hololink::csi::PixelFormat::RAW_10:
        return "RAW_10";
    case hololink::csi::PixelFormat::RAW_12:
        return "RAW_12";
    default:
        return "UNKNOWN";
    }
}

static std::string bayer_format_name(hololink::csi::BayerFormat fmt)
{
    switch (fmt) {
    case hololink::csi::BayerFormat::RGGB:
        return "RGGB";
    case hololink::csi::BayerFormat::BGGR:
        return "BGGR";
    case hololink::csi::BayerFormat::GBRG:
        return "GBRG";
    case hololink::csi::BayerFormat::GRBG:
        return "GRBG";
    default:
        return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

// Declares one leg's capture+saver pair.
void declare_leg(holoscan::Graph& graph, int id,
    hololink::operators::FusaCoeCaptureParams capture_params,
    FrameSaverParams saver_params)
{
    auto capture = graph.op<hololink::operators::FusaCoeCaptureOp>("camera" + std::to_string(id));
    // Names this camera in the capture's stall messages.
    capture_params.name = std::string(capture.name());
    capture.set_params(std::move(capture_params));

    auto saver = graph.op<FrameSaverOp>("saver" + std::to_string(id));
    saver.set_params(std::move(saver_params));

    graph.add_flow(capture->tensor, saver->input_tensor,
        holoscan::ConnectionOptions { .queue_depth = 4U });
    graph.add_flow(capture->metadata, saver->input_metadata,
        holoscan::ConnectionOptions { .queue_depth = 4U });
}

// Per-leg state. Kept alive for the whole run() call (through session.wait()):
// FusaCoeCaptureParams.hololink_channel is a raw pointer into hololink_channel,
// and device_start/device_stop close over camera.
struct Leg {
    int sensor_id { 0 };
    hololink::DataChannel hololink_channel;
    std::shared_ptr<hololink::sensors::NativeVb1940Sensor> camera;
    std::shared_ptr<SaveStatus> status { std::make_shared<SaveStatus>() };
    std::string interface;
    std::vector<uint8_t> mac_addr;
    hololink::csi::PixelFormat pixel_format {};
    hololink::csi::BayerFormat bayer_format {};
    std::uint32_t width { 0 };
    std::uint32_t height { 0 };
    bool packetizer_enabled { false };

    explicit Leg(hololink::Metadata metadata)
        : hololink_channel(metadata)
    {
    }
};

int run(
    const std::string& hololink_ip,
    int sensor_id,
    hololink::sensors::vb1940_mode::Mode camera_mode,
    std::uint32_t frames,
    const std::string& output_dir,
    std::uint32_t timeout_ms,
    double timeout_s,
    bool no_packetizer,
    holoscan::ManagedShutdownSignals& signals)
{
    // sensor_id == -1 means stereo: one leg per physical sensor.
    const std::vector<int> sensor_ids = (sensor_id < 0)
        ? std::vector<int> { 0, 1 }
        : std::vector<int> { sensor_id };

    auto base_metadata_raw = hololink::Enumerator::find_channel(hololink_ip);
    auto base_metadata = hololink::Metadata(base_metadata_raw);

    std::vector<std::unique_ptr<Leg>> legs;
    std::shared_ptr<hololink::Hololink> hololink_dev;
    holoscan::Graph graph { "fusa_coe_frame_saver" };

    // Pass 1: construct each leg's DataChannel + NativeVb1940Sensor only. No
    // per-sensor hardware access yet (that's pass 2, below) -- just the one-time,
    // shared hololink_dev->start()/reset().
    for (int id : sensor_ids) {
        auto metadata = base_metadata;
        hololink::DataChannel::use_sensor(metadata, id);

        auto leg = std::make_unique<Leg>(metadata);
        leg->sensor_id = id;
        leg->interface = metadata.get<std::string>("interface").value_or("");
        auto mac_str = metadata.get<std::string>("mac_id").value_or("");
        auto mac_bytes_arr = parse_mac_address(mac_str);
        leg->mac_addr.assign(mac_bytes_arr.begin(), mac_bytes_arr.end());

        HOLOSCAN_LOG_INFO("[sensor {}] interface={}, mac={}", id, leg->interface, mac_str);

        if (!hololink_dev) {
            hololink_dev = leg->hololink_channel.hololink();
            hololink_dev->start();
            hololink_dev->reset();
        }

        leg->camera = std::make_shared<hololink::sensors::NativeVb1940Sensor>(
            leg->hololink_channel);
        legs.push_back(std::move(leg));
    }

    // FIX (issue #8): mirrors examples/single_network_stereo_vb1940_player.cpp's proven
    // stereo bring-up sequence exactly. Hololink::setup_clock() is unconditional -- it
    // re-writes the *shared*, non-sensor-specific board register 0x8 (through an
    // intermediate 0x30 before settling on 0x03) on every call. Calling it once per leg
    // (as this file used to) glitches whichever sensor was already brought up by an
    // earlier call, silently knocking it out of the SW_STBY state start() assumes it's
    // still in. HSDK4's Vb1940Cam follows the same principle for the same shared register
    // (0x8: 0x0 -> mask, committed exactly once, atomically, after every sensor's presence
    // is known). Hold every requested sensor in reset, program the clock generator once
    // (it configures every sensor's clock regardless of which camera object makes the
    // call), then release reset for every requested sensor atomically.
    {
        uint32_t reset_mask = 0;
        for (int id : sensor_ids) {
            reset_mask |= (1U << id);
        }
        hololink_dev->write_uint32(0x8, 0x0);
        legs.front()->camera->setup_clock();
        hololink_dev->write_uint32(0x8, reset_mask);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Every leg saves --frames frames; the run completes once the last one has.
    auto frame_limit = std::make_shared<hx::FrameLimitGroup>(frames, static_cast<uint32_t>(legs.size()));

    // Pass 2: per-leg configuration and Holoscan operator declaration. Every
    // requested sensor is now out of reset and stable, so configure()'s secure-boot/
    // FSM walk -- genuinely per-sensor local-I2C traffic, not shared board state --
    // is safe to run independently for each leg.
    for (auto& leg : legs) {
        const int id = leg->sensor_id;
        const auto& interface = leg->interface;
        const auto& mac_bytes = leg->mac_addr;

        leg->camera->configure(camera_mode);

        leg->pixel_format = leg->camera->get_pixel_format();
        leg->bayer_format = leg->camera->get_bayer_format();
        leg->width = leg->camera->get_width();
        leg->height = leg->camera->get_height();

        // The packetizer only has programs for RAW_10 and RAW_12.
        leg->packetizer_enabled = !no_packetizer
            && leg->pixel_format != hololink::csi::PixelFormat::RAW_8;

        // No live FusaCoeCaptureOp instance exists before the graph compiles (see
        // holoscan::Graph::op()), so the sensor negotiates geometry with a freestanding
        // adapter instead of the operator itself — mirrors linux_imx274_player.cpp's
        // camera->configure_converter(csi_params.geometry) for CsiToBayerOp, but using
        // FUSA's own HSB-packetizer alignment rules (see make_geometry_adapter()).
        hololink::csi::CsiGeometry geometry;
        leg->camera->configure_converter(hololink::operators::FusaCoeCaptureOp::make_geometry_adapter(
            geometry, leg->packetizer_enabled));

        const std::string leg_output_dir = output_dir + "/sensor" + std::to_string(id);

        auto capture_params = hololink::operators::FusaCoeCaptureParams {
            .hololink_channel = &leg->hololink_channel,
            .device_start = [camera = leg->camera]() { camera->start(); },
            .device_stop = [camera = leg->camera]() { camera->stop(); },
            .interface = interface,
            .mac_addr = mac_bytes,
            .timeout = timeout_ms,
            .pixel_format = leg->pixel_format,
            .bayer_format = leg->bayer_format,
            .packetizer_enabled = leg->packetizer_enabled,
            .geometry = geometry,
        };
        auto saver_params = FrameSaverParams {
            .output_dir = leg_output_dir,
            .frame_limit = frame_limit,
            .status = leg->status,
            .pixel_width = leg->width,
            .pixel_height = leg->height,
            .pixel_format = pixel_format_name(leg->pixel_format),
            .bayer_format = bayer_format_name(leg->bayer_format),
            .packetizer_enabled = leg->packetizer_enabled,
        };

        declare_leg(graph, id, capture_params, saver_params);

        std::cout << "Capturing " << frames << " frames"
                  << " (sensor=" << id
                  << " mode=" << static_cast<int>(camera_mode)
                  << " " << leg->width << "x" << leg->height
                  << " " << pixel_format_name(leg->pixel_format)
                  << " packetizer=" << (leg->packetizer_enabled ? "on" : "off") << ")"
                  << " -> " << leg_output_dir << "/\n";
    }

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    // Each camera<N> output is a Tensor-bearing schema port; compile rejects an
    // unresolved device (IMPLICIT_DEVICE_ZERO_REJECTED) rather than assuming one.
    holoscan::CompileOptions compile_options;
    for (const auto& leg : legs) {
        const std::string operator_path = "camera" + std::to_string(leg->sensor_id);
        compile_options.deployment.bind_tensor_output_device(
            holoscan::TensorOutputDevicePlacement {
                .operator_path = operator_path,
                .output_port = "tensor",
                .device = holoscan::DeviceId { 0 },
            });
    }

    const auto plan = holoscan::compile(graph, compile_options);
    if (!plan.ok()) {
        hx::print_compile_failure(plan);
        hololink_dev->stop();
        return 1;
    }

    auto session = holoscan::run_async(plan, signals);

    // One deadline for every leg.
    const bool ended = session.wait_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(timeout_s)));
    if (!ended) {
        session.request_stop();
    }
    session.wait();
    const bool failed = hx::report_session(session, plan);

    bool timed_out = false;
    if (!ended) {
        for (const auto& leg : legs) {
            const std::uint32_t saved = leg->status->frames_saved.load();
            if (saved < frames) {
                timed_out = true;
                std::cerr << "[sensor " << leg->sensor_id << "] timed out after " << timeout_s
                          << "s — only " << saved << "/" << frames
                          << " frames saved. Try a larger --timeout-s.\n";
            }
        }
    }

    hololink_dev->stop();

    for (auto& leg : legs) {
        std::cout << "[sensor " << leg->sensor_id << "] saved "
                  << leg->status->frames_saved.load()
                  << " frames to " << output_dir << "/sensor" << leg->sensor_id << "/\n";
    }
    return (failed || timed_out) ? 1 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    using namespace hololink::args;

    hololink::logging::hsb_log_level = hololink::logging::HSB_LOG_LEVEL_INFO;

    OptionsDescription options_description("Allowed options");
    // clang-format off
    options_description.add_options()
        ("hololink", value<std::string>()->default_value(hololink::env_hololink_ip(0, "192.168.0.2")),
            "IP address of Hololink board")
        ("sensor", value<int>()->default_value(-1),
            "Sensor index (0 or 1), or -1 for both legs (stereo)")
        ("camera-mode", value<int>()->default_value(
            static_cast<int>(hololink::sensors::vb1940_mode::VB1940_MODE_2560X1984_30FPS)),
            "VB1940 mode")
        ("frames", value<uint32_t>()->default_value(5),
            "Number of frames to save then stop")
        ("output-dir", value<std::string>()->default_value("/tmp/fusa_frames"),
            "Output directory (created if absent)")
        ("timeout", value<uint32_t>()->default_value(1500),
            "Per-frame capture timeout in milliseconds")
        ("timeout-s", value<double>()->default_value(60.0),
            "Seconds to wait for all frames before giving up")
        ("no-packetizer", bool_switch()->default_value(false),
            "Capture raw CSI-2 instead of HSB-packed data")
        ;
    // clang-format on

    auto vm = Parser().parse_command_line(argc, argv, options_description);

    // With managed signals, Ctrl+C stops the session cooperatively, so the cameras and the board
    // are stopped, and the saved frames reported, on the way out.
    return hx::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
        try {
            const int sensor_id = vm["sensor"].as<int>();
            if (sensor_id < -1 || sensor_id > 1) {
                throw std::invalid_argument("--sensor must be 0, 1, or -1 (both)");
            }
            const uint32_t frames = vm["frames"].as<uint32_t>();
            if (frames == 0) {
                throw std::invalid_argument("--frames must be greater than 0");
            }

            return run(
                vm["hololink"].as<std::string>(),
                sensor_id,
                static_cast<hololink::sensors::vb1940_mode::Mode>(vm["camera-mode"].as<int>()),
                frames,
                vm["output-dir"].as<std::string>(),
                vm["timeout"].as<uint32_t>(),
                vm["timeout-s"].as<double>(),
                vm["no-packetizer"].as<bool>(),
                signals);
        } catch (const std::exception& e) {
            HOLOSCAN_LOG_ERROR("fusa_coe_frame_saver: {}", e.what());
            return 1;
        }
    });
}
