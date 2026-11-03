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

/**
 * @brief Receive generic test frames over Fusa CoE from a remote TestSensor emulator.
 *
 * Start the emulator on another host first, for example:
 *   serve_coe_test_data <emulator_ip>
 *
 * Then run this application to configure the TestSensor over I2C and print received
 * frame bytes to stdout.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
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

#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/on_scope_exit.hpp>
#include <hololink/core/serializer.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/fusa_coe_capture/fusa_coe_capture.hpp>

#define HOLOLINK_PLAYER_SUPPORT_NO_DISPLAY
#include "player_support.hpp"

namespace hx = hololink::examples;

using hololink::core::OnScopeExit;

// These constants must match the definitions in hololink::emulation::sensors::TestSensor
static constexpr uint8_t TEST_I2C_ADDRESS = 0x5A;
static constexpr uint16_t FRAME_SIZE_REG = 0x0010u;
static constexpr uint16_t PATTERN_MODE_REG = 0x0014u;
static constexpr uint16_t CONSTANT_BYTE_REG = 0x0015u;
static constexpr uint16_t FRAME_RATE_REG = 0x0020u;
static constexpr uint16_t STREAMING_REG = 0x0030u;

enum class TestPatternMode : uint8_t {
    CONSTANT = 0,
    INCREMENTING = 1,
};

namespace {

hololink::core::MacAddress parse_mac_address(const std::string& mac_str)
{
    hololink::core::MacAddress mac;
    std::istringstream iss(mac_str);
    std::string byte_str;
    int i = 0;

    while (std::getline(iss, byte_str, ':')) {
        if (i >= 6)
            throw std::invalid_argument("Too many bytes in MAC address");

        if (byte_str.size() != 2)
            throw std::invalid_argument("Each byte must be exactly two hex digits");

        uint16_t byte;
        std::istringstream hex_stream(byte_str);
        hex_stream >> std::hex >> byte;
        if (hex_stream.fail())
            throw std::invalid_argument("Invalid hex in MAC address");

        mac[i++] = static_cast<uint8_t>(byte);
    }

    if (i != 6)
        throw std::invalid_argument("MAC address must have exactly 6 bytes");

    return mac;
}

class TestSensorController {
public:
    explicit TestSensorController(std::shared_ptr<hololink::Hololink::I2c> i2c)
        : i2c_(std::move(i2c))
    {
    }

    void write_register_u8(uint16_t reg, uint8_t value)
    {
        std::vector<uint8_t> write_bytes(3);
        hololink::core::Serializer serializer(write_bytes.data(), write_bytes.size());
        serializer.append_uint16_be(reg);
        serializer.append_uint8(value);
        i2c_->i2c_transaction(TEST_I2C_ADDRESS, write_bytes, 0);
    }

    void write_register_u32_le(uint16_t reg, uint32_t value)
    {
        std::vector<uint8_t> write_bytes(6);
        hololink::core::Serializer serializer(write_bytes.data(), write_bytes.size());
        serializer.append_uint16_be(reg);
        serializer.append_uint8(static_cast<uint8_t>(value));
        serializer.append_uint8(static_cast<uint8_t>(value >> 8));
        serializer.append_uint8(static_cast<uint8_t>(value >> 16));
        serializer.append_uint8(static_cast<uint8_t>(value >> 24));
        i2c_->i2c_transaction(TEST_I2C_ADDRESS, write_bytes, 0);
    }

    void configure(uint32_t frame_size_bytes, uint32_t frame_rate_hz, uint8_t pattern_mode, uint8_t constant_byte)
    {
        write_register_u32_le(FRAME_SIZE_REG, frame_size_bytes);
        write_register_u8(PATTERN_MODE_REG, pattern_mode);
        write_register_u8(CONSTANT_BYTE_REG, constant_byte);
        write_register_u32_le(FRAME_RATE_REG, frame_rate_hz);
    }

    void start_streaming() { write_register_u8(STREAMING_REG, 1); }
    void stop_streaming() { write_register_u8(STREAMING_REG, 0); }

private:
    std::shared_ptr<hololink::Hololink::I2c> i2c_;
};

// ---------------------------------------------------------------------------
// Print progress: FrameDataPrinterOp completes the run itself once it has printed
// --frame-limit frames; run() reads how many it printed when --timeout-s expires first.
// ---------------------------------------------------------------------------
struct PrintStatus {
    std::atomic<std::uint32_t> frames_printed { 0 };
};

// ---------------------------------------------------------------------------
// FrameDataPrinterOp
// ---------------------------------------------------------------------------

struct FrameDataPrinterParams {
    // Complete the run once this many frames are printed; 0 or less: no limit.
    int max_frames { 0 };
    std::uint32_t print_max_bytes { 64 };
    std::shared_ptr<PrintStatus> status;
};

class FrameDataPrinterOp final : public holoscan::Operator<FrameDataPrinterParams> {
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

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext& context) override
    {
        const auto& p = params();

        auto tensor_sample = input_tensor.receive();
        if (!tensor_sample)
            return holoscan::make_unexpected(std::move(tensor_sample).error());
        auto metadata_sample = input_metadata.receive();
        if (!metadata_sample)
            return holoscan::make_unexpected(std::move(metadata_sample).error());

        // Drain without printing once we have enough frames.
        if (p.max_frames > 0 && frames_printed_ >= static_cast<std::uint32_t>(p.max_frames)) {
            return {};
        }

        const hololink::CaptureMetadataT& frame = metadata_sample->data;
        const holoscan::Tensor& tensor = tensor_sample->data;
        const std::size_t frame_size = tensor.nbytes();
        const std::size_t print_count = p.print_max_bytes == 0
            ? frame_size
            : std::min<std::size_t>(frame_size, p.print_max_bytes);

        std::vector<std::uint8_t> host_data(print_count);
        if (tensor.device().device_type == kDLCUDA) {
            cudaError_t err = cudaMemcpyAsync(
                host_data.data(), tensor.data(), print_count,
                cudaMemcpyDeviceToHost,
                context.cuda_stream().get());
            if (err != cudaSuccess) {
                return holoscan::make_unexpected(holoscan::Error {
                    holoscan::ErrorCode::kFailure,
                    "FrameDataPrinterOp: cudaMemcpyAsync failed" });
            }
            err = cudaStreamSynchronize(context.cuda_stream().get());
            if (err != cudaSuccess) {
                return holoscan::make_unexpected(holoscan::Error {
                    holoscan::ErrorCode::kFailure,
                    "FrameDataPrinterOp: cudaStreamSynchronize failed" });
            }
        } else {
            std::memcpy(host_data.data(), tensor.data(), print_count);
        }

        ++frames_printed_;

        std::cout << "Frame " << frames_printed_;
        if (frame.frame_number >= 0) {
            std::cout << " (metadata frame_number=" << frame.frame_number << ")";
        }
        std::cout << " bytes[" << print_count << "/" << frame_size << "]:";
        std::cout << std::hex << std::setfill('0');
        for (std::size_t i = 0; i < print_count; ++i) {
            if (i % 16 == 0) {
                std::cout << '\n'
                          << "    " << std::setw(4) << i << ": ";
            }
            std::cout << std::setw(2) << static_cast<unsigned>(host_data[i]) << ' ';
        }
        std::cout << std::dec << std::setfill(' ') << std::endl;

        if (p.status) {
            p.status->frames_printed.store(frames_printed_);
        }
        if (p.max_frames > 0 && frames_printed_ == static_cast<std::uint32_t>(p.max_frames)) {
            context.request_completion();
        }

        return {};
    }

private:
    std::uint32_t frames_printed_ { 0 };
};

int run(
    const std::string& hololink_ip,
    uint32_t sensor,
    uint32_t frame_size,
    uint32_t frame_rate,
    uint8_t pattern_mode,
    uint8_t constant_byte,
    uint32_t print_max_bytes,
    uint32_t timeout_ms,
    int frame_limit,
    double timeout_s,
    holoscan::ManagedShutdownSignals& signals)
{
    auto channel_metadata = hololink::Enumerator::find_channel(hololink_ip);
    hololink::Metadata metadata(channel_metadata);
    hololink::DataChannel::use_sensor(metadata, sensor);
    hololink::DataChannel hololink_channel(metadata);

    auto interface = metadata.get<std::string>("interface").value_or("");
    auto mac_str = metadata.get<std::string>("mac_id").value_or("");
    auto mac_bytes_arr = parse_mac_address(mac_str);
    std::vector<uint8_t> mac_bytes(mac_bytes_arr.begin(), mac_bytes_arr.end());
    HOLOSCAN_LOG_INFO("Using interface={}, mac={}", interface, mac_str);

    std::shared_ptr<hololink::Hololink> hololink_dev = hololink_channel.hololink();
    hololink_dev->start();
    // Hololink instances are cached in a process-global registry (see
    // Hololink::from_enumeration_metadata()), so a thrown exception unwinding this
    // local shared_ptr does not drop the object's refcount to zero -- the
    // destructor's own stop() safety net does not fire. Guard it explicitly.
    OnScopeExit stop_hololink([&hololink_dev] { hololink_dev->stop(); });
    hololink_dev->reset();

    auto i2c = hololink_dev->get_i2c(hololink::CAM_I2C_BUS + sensor);
    auto test_sensor = std::make_shared<TestSensorController>(i2c);
    test_sensor->configure(frame_size, frame_rate, pattern_mode, constant_byte);

    HOLOSCAN_LOG_INFO(
        "Configured TestSensor: frame_size={}, frame_rate={} Hz, pattern_mode={}, constant_byte=0x{:02x}",
        frame_size, frame_rate, static_cast<unsigned>(pattern_mode), constant_byte);
    HOLOSCAN_LOG_INFO("Ensure serve_coe_test_data is running on the emulator host");

    holoscan::Graph graph { "fusa_coe_test_sensor_receiver" };

    // The TestSensor emits an opaque byte blob, not a real CSI image -- frame_size_bytes
    // routes FusaCoeCaptureOp::start() to core_.configure_frame_size(), the purpose-built
    // no-CFA/no-packetizer capture path (NvSciColor_R8), same as the HSDK4-era
    // configure_frame_size() call this replaces.
    auto capture = graph.op<hololink::operators::FusaCoeCaptureOp>("capture");
    capture.set_params(hololink::operators::FusaCoeCaptureParams {
        .hololink_channel = &hololink_channel,
        .device_start = [test_sensor] { test_sensor->start_streaming(); },
        .device_stop = [test_sensor] { test_sensor->stop_streaming(); },
        .interface = interface,
        .mac_addr = mac_bytes,
        .timeout = timeout_ms,
        // A sensor that sends nothing fails a --frame-limit run instead of keeping it going.
        .stall_timeout_ms = hx::frame_limit_stall_timeout_ms(frame_limit, timeout_ms),
        .frame_size_bytes = frame_size,
        // Names this camera in the capture's stall messages.
        .name = std::string(capture.name()),
    });

    auto status = std::make_shared<PrintStatus>();
    auto printer = graph.op<FrameDataPrinterOp>("printer");
    printer.set_params(FrameDataPrinterParams {
        .max_frames = frame_limit,
        .print_max_bytes = print_max_bytes,
        .status = status,
    });

    graph.add_flow(capture->tensor, printer->input_tensor, hx::pipeline_edge());
    graph.add_flow(capture->metadata, printer->input_metadata, hx::pipeline_edge());

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions compile_options;
    compile_options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement {
            .operator_path = "capture",
            .output_port = "tensor",
            .device = holoscan::DeviceId { 0 },
        });

    const auto plan = holoscan::compile(graph, compile_options);
    if (!plan.ok()) {
        hx::print_compile_failure(plan);
        return -1;
    }

    auto session = holoscan::run_async(plan, signals);

    // The printer completes the run after --frame-limit frames (0: run until Ctrl+C);
    // --timeout-s, when given, bounds the wait for them.
    bool ended = true;
    if (frame_limit > 0 && timeout_s > 0.0) {
        ended = session.wait_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(timeout_s)));
        if (!ended) {
            session.request_stop();
        }
    }
    session.wait();
    const bool failed = hx::report_session(session, plan);
    if (failed) {
        return -1;
    }

    const auto printed = status->frames_printed.load();
    if (!ended && printed < static_cast<std::uint32_t>(frame_limit)) {
        std::cerr << "Timed out after " << timeout_s << "s -- only "
                  << printed << "/" << frame_limit
                  << " frames printed.\n";
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    using namespace hololink::args;

    hololink::logging::hsb_log_level = hololink::logging::HSB_LOG_LEVEL_INFO;

    OptionsDescription options_description("fusa_coe_test_sensor_receiver options");
    // clang-format off
    options_description.add_options()
        ("hololink", value<std::string>()->default_value(hololink::env_hololink_ip(0, "192.168.0.2")), "IP address of Hololink board")
        ("sensor", value<uint32_t>()->default_value(0), "Sensor index (I2C bus = CAM_I2C_BUS + sensor)")
        ("frame-size", value<uint32_t>()->default_value(256), "Test frame payload size in bytes")
        ("frame-rate", value<uint32_t>()->default_value(30), "TestSensor frame rate in Hz")
        ("pattern-mode", value<uint32_t>()->default_value(static_cast<uint32_t>(TestPatternMode::INCREMENTING)),
            "TestSensor pattern mode (0=constant, 1=incrementing)")
        ("constant-byte", value<uint32_t>()->default_value(0xA5), "Constant pattern byte when pattern-mode=0")
        ("frame-limit", value<int>()->default_value(10), "Exit after receiving this many frames (0=infinite, default=10)")
        ("print-max-bytes", value<uint32_t>()->default_value(64), "Max bytes to print per frame (0=all)")
        ("timeout", value<uint32_t>()->default_value(1500), "Capture timeout in milliseconds")
        ("timeout-s", value<double>()->default_value(0.0), "Seconds to wait for --frame-limit frames before giving up (0=no limit)")
        ;
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);

    // With managed signals, Ctrl+C stops the session cooperatively, so the TestSensor and the
    // board are stopped on the way out.
    return hx::run_with_managed_signals([&](holoscan::ManagedShutdownSignals& signals) {
        try {
            const auto frame_size = variables_map["frame-size"].as<uint32_t>();
            const auto pattern_mode = static_cast<uint8_t>(variables_map["pattern-mode"].as<uint32_t>());

            if (frame_size == 0) {
                throw std::invalid_argument("frame-size must be greater than 0");
            }
            if (pattern_mode != static_cast<uint8_t>(TestPatternMode::CONSTANT)
                && pattern_mode != static_cast<uint8_t>(TestPatternMode::INCREMENTING)) {
                throw std::invalid_argument("pattern-mode must be 0 (constant) or 1 (incrementing)");
            }

            return run(
                variables_map["hololink"].as<std::string>(),
                variables_map["sensor"].as<uint32_t>(),
                frame_size,
                variables_map["frame-rate"].as<uint32_t>(),
                pattern_mode,
                static_cast<uint8_t>(variables_map["constant-byte"].as<uint32_t>()),
                variables_map["print-max-bytes"].as<uint32_t>(),
                variables_map["timeout"].as<uint32_t>(),
                variables_map["frame-limit"].as<int>(),
                variables_map["timeout-s"].as<double>(),
                signals);
        } catch (const std::exception& e) {
            HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
            return -1;
        }
    });
}
