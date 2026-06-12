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
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <holoscan/holoscan.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/holoargs.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/core/serializer.hpp>
#include <hololink/operators/fusa_coe_capture/fusa_coe_capture.hpp>

// These constants must match the definitions in hololink::emulation::sensors::TestSensor
static constexpr uint8_t TEST_I2C_ADDRESS = 0x5A;
static constexpr uint16_t DEVICE_ID_REG = 0x0000u;
static constexpr uint16_t FRAME_SIZE_REG = 0x0010u;
static constexpr uint16_t PATTERN_MODE_REG = 0x0014u;
static constexpr uint16_t CONSTANT_BYTE_REG = 0x0015u;
static constexpr uint16_t FRAME_RATE_REG = 0x0020u;
static constexpr uint16_t STREAMING_REG = 0x0030u;
static constexpr uint16_t STATUS_REG = 0x0031u;

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
    TestSensorController(std::shared_ptr<hololink::Hololink::I2c> i2c)
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

    void start_streaming()
    {
        write_register_u8(STREAMING_REG, 1);
    }

    void stop_streaming()
    {
        write_register_u8(STREAMING_REG, 0);
    }

private:
    std::shared_ptr<hololink::Hololink::I2c> i2c_;
};

class FrameDataPrinterOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(FrameDataPrinterOp)

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input<holoscan::gxf::Entity>("input");
        spec.param(frame_size_bytes_, "frame_size_bytes", "FrameSizeBytes",
            "Number of payload bytes to print per frame", 256u);
        spec.param(print_max_bytes_, "print_max_bytes", "PrintMaxBytes",
            "Maximum number of bytes to print per frame (0 = all)", 64u);
    }

    void start() override
    {
        CudaCheck(cuInit(0));
        CudaCheck(cuDeviceGet(&cuda_device_, 0));
        CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    }

    void stop() override
    {
        if (cuda_context_) {
            CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
            cuda_context_ = nullptr;
        }
    }

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
        holoscan::ExecutionContext& context) override
    {
        (void)op_output;
        (void)context;

        hololink::common::CudaContextScopedPush scoped_cuda_context(cuda_context_);

        auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
        if (!maybe_entity) {
            HOLOSCAN_LOG_WARN("Failed to receive entity");
            return;
        }

        auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());
        const auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
        if (!maybe_tensor) {
            throw std::runtime_error("Tensor not found in message");
        }

        const auto tensor = maybe_tensor.value();
        if (tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
            throw std::runtime_error("Expected device memory tensor from FusaCoeCaptureOp");
        }

        const uint32_t frame_size = tensor->size();
        const uint32_t print_count = print_max_bytes_.get() == 0
            ? frame_size
            : std::min(frame_size, print_max_bytes_.get());

        std::vector<uint8_t> host_data(print_count);
        CudaCheck(cuMemcpyDtoH(host_data.data(),
            reinterpret_cast<CUdeviceptr>(tensor->pointer()),
            print_count));

        frame_count_++;
        int64_t frame_number = metadata()->get<int64_t>("frame_number");

        std::cout << "Frame " << frame_count_;
        if (frame_number >= 0) {
            std::cout << " (metadata frame_number=" << frame_number << ")";
        }
        std::cout << " bytes[" << print_count << "/" << frame_size << "]:";

        std::cout << std::hex << std::setfill('0');
        for (uint32_t i = 0; i < print_count; ++i) {
            if (i % 16 == 0) {
                std::cout << '\n'
                          << "    " << std::setw(4) << i << ": ";
            }
            std::cout << std::setw(2) << static_cast<unsigned>(host_data[i]) << ' ';
        }
        std::cout << std::dec << std::setfill(' ') << std::endl;
    }

private:
    holoscan::Parameter<uint32_t> frame_size_bytes_;
    holoscan::Parameter<uint32_t> print_max_bytes_;
    CUdevice cuda_device_ = 0;
    CUcontext cuda_context_ = nullptr;
    uint64_t frame_count_ = 0;
};

class Application : public holoscan::Application {
public:
    Application(
        const std::string& interface,
        const hololink::core::MacAddress& mac,
        hololink::DataChannel& hololink_channel,
        std::shared_ptr<TestSensorController> test_sensor,
        uint32_t frame_size_bytes,
        uint32_t print_max_bytes,
        uint32_t timeout_ms,
        int frame_limit)
        : interface_(interface)
        , mac_(mac)
        , hololink_channel_(hololink_channel)
        , test_sensor_(std::move(test_sensor))
        , frame_size_bytes_(frame_size_bytes)
        , print_max_bytes_(print_max_bytes)
        , timeout_ms_(timeout_ms)
        , frame_limit_(frame_limit)
    {
        enable_metadata(true);
    }

    void compose() override
    {
        using namespace holoscan;

        std::shared_ptr<Condition> condition;
        if (frame_limit_ > 0) {
            condition = make_condition<CountCondition>("count", frame_limit_);
        } else {
            condition = make_condition<BooleanCondition>("ok", true);
        }

        auto fusa_coe_capture = make_operator<hololink::operators::FusaCoeCaptureOp>(
            "fusa_coe_capture",
            Arg("interface", interface_),
            Arg("mac_addr", std::vector<uint8_t>(mac_.begin(), mac_.end())),
            Arg("hololink_channel", &hololink_channel_),
            Arg("out_tensor_name", std::string("frame")),
            Arg("timeout", timeout_ms_),
            Arg("device_start", std::function<void()>([this] { test_sensor_->start_streaming(); })),
            Arg("device_stop", std::function<void()>([this] { test_sensor_->stop_streaming(); })),
            condition);
        fusa_coe_capture->configure_frame_size(frame_size_bytes_);

        auto frame_printer = make_operator<FrameDataPrinterOp>(
            "frame_printer",
            Arg("frame_size_bytes", frame_size_bytes_),
            Arg("print_max_bytes", print_max_bytes_),
            condition);

        add_flow(fusa_coe_capture, frame_printer, { { "output", "input" } });
    }

private:
    std::string interface_;
    hololink::core::MacAddress mac_;
    hololink::DataChannel& hololink_channel_;
    std::shared_ptr<TestSensorController> test_sensor_;
    uint32_t frame_size_bytes_;
    uint32_t print_max_bytes_;
    uint32_t timeout_ms_;
    int frame_limit_;
};

} // namespace

int main(int argc, char** argv)
{
    using namespace hololink::args;

    hololink::logging::hsb_log_level = hololink::logging::HSB_LOG_LEVEL_INFO;

    OptionsDescription options_description("fusa_coe_test_sensor_receiver options");
    // clang-format off
    options_description.add_options()
        ("hololink", value<std::string>()->default_value("192.168.0.2"), "IP address of Hololink board")
        ("sensor", value<uint32_t>()->default_value(0), "Sensor index (I2C bus = CAM_I2C_BUS + sensor)")
        ("frame-size", value<uint32_t>()->default_value(256), "Test frame payload size in bytes")
        ("frame-rate", value<uint32_t>()->default_value(30), "TestSensor frame rate in Hz")
        ("pattern-mode", value<uint32_t>()->default_value(static_cast<uint32_t>(TestPatternMode::INCREMENTING)),
            "TestSensor pattern mode (0=constant, 1=incrementing)")
        ("constant-byte", value<uint32_t>()->default_value(0xA5), "Constant pattern byte when pattern-mode=0")
        ("frame-limit", value<int>()->default_value(10), "Exit after receiving this many frames (0=infinite, default=10)")
        ("print-max-bytes", value<uint32_t>()->default_value(64), "Max bytes to print per frame (0=all)")
        ("timeout", value<uint32_t>()->default_value(1500), "Capture timeout in milliseconds")
        ;
    // clang-format on

    try {
        auto variables_map = Parser().parse_command_line(argc, argv, options_description);
        const auto hololink_ip = variables_map["hololink"].as<std::string>();
        const auto sensor = variables_map["sensor"].as<uint32_t>();
        const auto frame_size = variables_map["frame-size"].as<uint32_t>();
        const auto frame_rate = variables_map["frame-rate"].as<uint32_t>();
        const auto pattern_mode = static_cast<uint8_t>(variables_map["pattern-mode"].as<uint32_t>());
        const auto constant_byte = static_cast<uint8_t>(variables_map["constant-byte"].as<uint32_t>());
        const auto frame_limit = variables_map["frame-limit"].as<int>();
        const auto print_max_bytes = variables_map["print-max-bytes"].as<uint32_t>();
        const auto timeout_ms = variables_map["timeout"].as<uint32_t>();

        if (frame_size == 0) {
            throw std::invalid_argument("frame-size must be greater than 0");
        }
        if (pattern_mode != static_cast<uint8_t>(TestPatternMode::CONSTANT)
            && pattern_mode != static_cast<uint8_t>(TestPatternMode::INCREMENTING)) {
            throw std::invalid_argument("pattern-mode must be 0 (constant) or 1 (incrementing)");
        }

        auto channel_metadata = hololink::Enumerator::find_channel(hololink_ip);
        hololink::Metadata metadata(channel_metadata);
        hololink::DataChannel::use_sensor(metadata, sensor);
        hololink::DataChannel hololink_channel(metadata);

        auto interface = metadata.get<std::string>("interface").value_or("");
        auto mac_str = metadata.get<std::string>("mac_id").value_or("");
        auto mac_bytes = parse_mac_address(mac_str);
        HOLOSCAN_LOG_INFO("Using interface={}, mac={}", interface, mac_str);

        std::shared_ptr<hololink::Hololink> hololink = hololink_channel.hololink();
        hololink->start();
        hololink->reset();

        auto i2c = hololink->get_i2c(hololink::CAM_I2C_BUS + sensor);
        auto test_sensor = std::make_shared<TestSensorController>(i2c);
        test_sensor->configure(frame_size, frame_rate, pattern_mode, constant_byte);

        HOLOSCAN_LOG_INFO(
            "Configured TestSensor: frame_size={}, frame_rate={} Hz, pattern_mode={}, constant_byte=0x{:02x}",
            frame_size, frame_rate, static_cast<unsigned>(pattern_mode), constant_byte);
        HOLOSCAN_LOG_INFO("Ensure serve_coe_test_data is running on the emulator host");

        auto app = std::make_unique<Application>(
            interface,
            mac_bytes,
            hololink_channel,
            test_sensor,
            frame_size,
            print_max_bytes,
            timeout_ms,
            frame_limit);
        app->run();

        hololink->stop();
        return 0;
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
        return -1;
    }
}
