/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "native_imx274_sensor.hpp"

#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

#include "hololink/sensors/li_i2c_expander.hpp"
#include "imx274_mode.hpp"
#include "renesas_bajoran_lite_ts1.hpp"

#include "hololink/core/deserializer.hpp"
#include "hololink/core/logging_internal.hpp"
#include "hololink/core/serializer.hpp"
#include "hololink/core/timeout.hpp"

namespace hololink::sensors {

NativeImx274Sensor::NativeImx274Sensor(DataChannel& data_channel, int expander_configuration,
    uint32_t i2c_bus)
    : hololink_(data_channel.hololink())
{
    sensor_id_ = DRIVER_NAME;
    i2c_ = hololink_->get_i2c(i2c_bus);
    i2c_expander_ = std::make_shared<LII2CExpander>(*hololink_, i2c_bus);

    // Configure I2C expander
    if (expander_configuration == 1) {
        i2c_expander_configuration_ = I2CExpanderOutputEN::OUTPUT_2;
    } else {
        i2c_expander_configuration_ = I2CExpanderOutputEN::OUTPUT_1;
    }

    initialize_supported_modes();
}

NativeImx274Sensor::~NativeImx274Sensor() { }

void NativeImx274Sensor::configure(CameraMode mode)
{
    // Make sure this is a version we know about.
    uint32_t version = get_version();
    HSB_LOG_INFO("version={}", version);
    assert(version == VERSION);

    configure_camera(mode);
}

void NativeImx274Sensor::set_mode(CameraMode mode)
{
    if (mode >= 0 && mode < imx274_mode::IMX274_MODE_COUNT) {
        mode_ = mode;
        // Set the width, height, pixel format, and bayer format
        auto frame_format = mode_frame_formats_[mode_.value()];
        width_ = frame_format->width();
        height_ = frame_format->height();
        pixel_format_ = frame_format->pixel_format();
        bayer_format_ = hololink::csi::BayerFormat::RGGB;
    } else {
        HSB_LOG_ERROR("Incorrect mode for IMX274");
    }
}

void NativeImx274Sensor::setup_clock()
{
    // Set the clock driver.
    hololink_->setup_clock(hololink::renesas::DEVICE_CONFIGURATION);
}

uint32_t NativeImx274Sensor::get_version() const
{
    // TODO: get the version or the name of the sensor from the sensor
    return VERSION;
}

uint8_t NativeImx274Sensor::get_register(uint16_t reg)
{
    HSB_LOG_DEBUG("get_register(register={}(0x{:X}))", reg, reg);
    i2c_expander_->configure(i2c_expander_configuration_);

    std::vector<uint8_t> write_bytes(2);
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint16_be(static_cast<uint16_t>(reg));

    const uint32_t read_byte_count = 1;

    // Read one byte
    auto reply = i2c_->i2c_transaction(I2C_ADDRESS, write_bytes, read_byte_count);

    // Deserialize the reply
    core::Deserializer deserializer(reply.data(), reply.size());
    uint8_t value;
    if (!deserializer.next_uint8(value)) {
        throw std::runtime_error("Failed to read register value");
    }

    HSB_LOG_DEBUG("get_register(register={}(0x{:X}))={}(0x{:X})", reg, reg, value, value);

    return value;
}

void NativeImx274Sensor::set_register(uint16_t reg, uint8_t value,
    const std::shared_ptr<Timeout>& timeout)
{
    HSB_LOG_DEBUG("set_register(register={}(0x{:X}), value={}(0x{:X}))", reg, reg, value, value);

    i2c_expander_->configure(i2c_expander_configuration_);

    std::vector<uint8_t> write_bytes(3);
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint16_be(static_cast<uint16_t>(reg));
    serializer.append_uint8(value);

    const uint32_t read_byte_count = 0;
    i2c_->i2c_transaction(I2C_ADDRESS, write_bytes, read_byte_count, timeout);
}

void NativeImx274Sensor::start()
{
    apply_register_settings(imx274_mode::IMX274_START_SEQUENCE);
}

void NativeImx274Sensor::stop()
{
    apply_register_settings(imx274_mode::IMX274_STOP_SEQUENCE);
}

void NativeImx274Sensor::configure_converter(std::shared_ptr<hololink::csi::CsiConverter> converter)
{
    // Get starting byte position
    uint32_t start_byte = converter->receiver_start_byte();

    // Calculate transmitted and received line bytes
    uint32_t transmitted_line_bytes = converter->transmitted_line_bytes(pixel_format_, width_);
    uint32_t received_line_bytes = converter->received_line_bytes(transmitted_line_bytes);

    // Account for 175 bytes of metadata preceding image data
    start_byte += converter->received_line_bytes(175);

    // Account for optical black lines based on pixel format
    if (pixel_format_ == hololink::csi::PixelFormat::RAW_10) {
        // 8 lines of optical black before real image data
        start_byte += received_line_bytes * 8;
    } else if (pixel_format_ == hololink::csi::PixelFormat::RAW_12) {
        // 16 lines of optical black before real image data
        start_byte += received_line_bytes * 16;
    } else {
        throw std::runtime_error("Incorrect pixel format for IMX274");
    }

    // Configure the converter
    converter->configure(
        start_byte,
        received_line_bytes,
        width_,
        height_,
        pixel_format_);
}

void NativeImx274Sensor::initialize_supported_modes()
{
    // Initialize supported modes
    supported_modes_ = {
        imx274_mode::IMX274_MODE_3840X2160_60FPS,
        imx274_mode::IMX274_MODE_1920X1080_60FPS,
        imx274_mode::IMX274_MODE_3840X2160_60FPS_12BITS
    };

    // Initialize frame formats
    mode_frame_formats_[imx274_mode::IMX274_MODE_3840X2160_60FPS] = std::make_shared<Imx274FrameFormat>(imx274_mode::IMX274_MODE_3840X2160_60FPS,
        "IMX274_MODE_3840X2160_60FPS",
        3840,
        2160,
        60,
        csi::PixelFormat::RAW_10);
    mode_frame_formats_[imx274_mode::IMX274_MODE_1920X1080_60FPS] = std::make_shared<Imx274FrameFormat>(imx274_mode::IMX274_MODE_1920X1080_60FPS,
        "IMX274_MODE_1920X1080_60FPS",
        1920,
        1080,
        60,
        csi::PixelFormat::RAW_10);
    mode_frame_formats_[imx274_mode::IMX274_MODE_3840X2160_60FPS_12BITS] = std::make_shared<Imx274FrameFormat>(imx274_mode::IMX274_MODE_3840X2160_60FPS_12BITS,
        "IMX274_MODE_3840X2160_60FPS_12BITS",
        3840,
        2160,
        60,
        csi::PixelFormat::RAW_12);
}

void NativeImx274Sensor::set_exposure_reg(int32_t value)
{
    if (value < 0x0C) {
        HSB_LOG_WARN(
            "Exposure value ({}) is below the minimum limit of 0x0C for IMX274. Setting to 0x0C.",
            value);
        value = 0x0C;
    } else if (value > 0xFFFF) {
        HSB_LOG_WARN(
            "Exposure value ({}) is above the maximum limit of 0xFFFF for IMX274. Setting to "
            "0xFFFF.",
            value);
        value = 0xFFFF;
    }

    uint8_t lsb = (value >> 8) & 0xFF;
    uint8_t msb = value & 0xFF;
    set_register(imx274_mode::REG_EXP_MSB, msb);
    set_register(imx274_mode::REG_EXP_LSB, lsb);
    std::this_thread::sleep_for(std::chrono::milliseconds(imx274_mode::IMX274_WAIT_MS));
}

void NativeImx274Sensor::set_analog_gain_reg(int32_t value)
{
    if (value < 0x00) {
        HSB_LOG_WARN("Analog gain value ({}) is below the minimum limit of 0x00 for IMX274. Setting to 0x00.", value);
        value = 0x00;
    } else if (value > 0xFFFF) {
        HSB_LOG_WARN("Analog gain value ({}) is above the maximum limit of 0xFFFF for IMX274. Setting to 0xFFFF.", value);
        value = 0xFFFF;
    }

    uint8_t lsb = (value >> 8) & 0xFF; // high bits go to LSB register
    uint8_t msb = value & 0xFF; // low bits go to MSB register
    set_register(imx274_mode::REG_AG_LSB, lsb);
    set_register(imx274_mode::REG_AG_MSB, msb);
    std::this_thread::sleep_for(std::chrono::milliseconds(imx274_mode::IMX274_WAIT_MS));
}

void NativeImx274Sensor::set_digital_gain_reg(int32_t value)
{
    // IMX274 can only have 0(1), 1(2), 2(4), 3(8), 4(16), 5(32), 6(64) value only.
    uint8_t reg_value = 0x00;
    if (value >= 0x40) {
        reg_value = 0x06;
    } else if (value >= 0x20) {
        reg_value = 0x05;
    } else if (value >= 0x10) {
        reg_value = 0x04;
    } else if (value >= 0x08) {
        reg_value = 0x03;
    } else if (value >= 0x04) {
        reg_value = 0x02;
    } else if (value >= 0x02) {
        reg_value = 0x01;
    }

    set_register(imx274_mode::REG_DG, reg_value);
    std::this_thread::sleep_for(std::chrono::milliseconds(imx274_mode::IMX274_WAIT_MS));
}

void NativeImx274Sensor::configure_camera(CameraMode mode)
{
    set_mode(mode);

    switch (mode_.value()) {
    case imx274_mode::IMX274_MODE_3840X2160_60FPS:
        apply_register_settings(imx274_mode::IMX274_MODE_3840X2160_60FPS_SEQUENCE);
        break;
    case imx274_mode::IMX274_MODE_1920X1080_60FPS:
        apply_register_settings(imx274_mode::IMX274_MODE_1920X1080_60FPS_SEQUENCE);
        break;
    case imx274_mode::IMX274_MODE_3840X2160_60FPS_12BITS:
        apply_register_settings(imx274_mode::IMX274_MODE_3840X2160_60FPS_12BITS_SEQUENCE);
        break;
    default:
        HSB_LOG_ERROR("Unsupported camera mode: {}", mode_.value());
        break;
    }
}

template <typename ContainerT>
void NativeImx274Sensor::apply_register_settings(const ContainerT& settings)
{
    static_assert(std::is_same_v<typename ContainerT::value_type,
                      std::pair<uint16_t, uint8_t>>,
        "Container must hold std::pair<uint16_t, uint8_t>");

    for (const auto& reg_val : settings) {
        if (reg_val.first == imx274_mode::IMX274_TABLE_WAIT_MS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(reg_val.second));
        } else {
            set_register(reg_val.first, reg_val.second);
        }
    }
}

void NativeImx274Sensor::test_pattern(uint8_t pattern)
{
    if (pattern == 0) {
        // Disable test pattern
        set_register(imx274_mode::REG_TEST_ENABLE, 0x00);
        set_register(imx274_mode::REG_TEST_SETUP1, 0x00);
        set_register(imx274_mode::REG_TEST_SETUP2, 0x00);
        set_register(imx274_mode::REG_TEST_SETUP3, 0x00);
    } else {
        // Enable test pattern
        set_register(imx274_mode::REG_TEST_ENABLE, 0x11);
        set_register(imx274_mode::REG_TEST_SETUP4, 0x01);
        set_register(imx274_mode::REG_TEST_SETUP1, 0x01);
        set_register(imx274_mode::REG_TEST_SETUP2, 0x01);
        set_register(imx274_mode::REG_TEST_SETUP3, 0x11);
        set_register(imx274_mode::REG_TEST_PATTERN, pattern);
    }
}

} // namespace hololink::sensors
