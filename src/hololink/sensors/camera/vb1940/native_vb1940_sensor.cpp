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

#include "native_vb1940_sensor.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <thread>
#include <vector>

#include "hololink/sensors/li_i2c_expander.hpp"
#include "renesas_bajoran_lite_ts2.hpp"
#include "vb1940_mode.hpp"

#include "hololink/core/deserializer.hpp"
#include "hololink/core/hololink.hpp"
#include "hololink/core/logging_internal.hpp"
#include "hololink/core/serializer.hpp"
#include "hololink/core/timeout.hpp"

namespace hololink::sensors {

namespace {
    // register map
    constexpr uint16_t DEVICE_REVISION_REG = 0x0004;
    constexpr uint16_t SYSTEM_UP_REG = 0x0514;
    constexpr uint16_t BOOT_REG = 0x0515;
    constexpr uint16_t SW_STBY_REG = 0x0516;
    constexpr uint16_t STREAMING_REG = 0x0517;
    constexpr uint16_t SYSTEM_FSM_STATE_REG = 0x0044;
    constexpr uint16_t BOOT_FSM_REG = 0x0200;

    enum class SystemFsmState : uint8_t {
        HW_STBY = 0x0,
        SYSTEM_UP = 0x1,
        BOOT = 0x2,
        SW_STBY = 0x3,
        STREAMING = 0x4,
        STALL = 0x5,
        HALT = 0x6
    };

    enum class BootFsmState : uint8_t {
        HW_STBY = 0x00,
        COLD_BOOT = 0x01,
        CLOCK_INIT = 0x02,
        NVM_DWLD = 0x10,
        NVM_UNPACK = 0x11,
        SYSTEM_BOOT = 0x12,
        NONCE_GRNERATION = 0x20,
        EPH_KEYS_GENERATION = 0x21,
        WAIT_CERTIFICATE = 0x22,
        CERTIFICATE_PARSING = 0x23,
        CERIFICATE_VERIF_ROOT = 0x24,
        CERTIFICATE_VERIF_USER = 0x25,
        CERTIFICATE_CHECK_FIELDS = 0x26,
        ECDH = 0x30,
        ECDH_SS_GEN = 0x31,
        ECDH_MASTER_KEY_GEN = 0x32,
        ECDH_SESSION_KEY_GEN = 0x33,
        AUTHENTICATION = 0x40,
        AUTHENTICATION_MSG_CREATE = 0x41,
        AUTHENTICATION_MSG_SIGN = 0x42,
        PROVISIONING = 0x50,
        PROVISIONING_UID = 0x51,
        PROVISIONING_EC_PRIV_KEY = 0x52,
        PROVISIONING_EC_PUB_KEY = 0x53,
        CTM_PROVISIONING_CID = 0x54,
        CTM_PROVISIONING_OEM_ROOT_KEY = 0x54,
        PAIRING = 0x55,
        FWP_SETUP = 0x60,
        VTP_SETUP = 0x61,
        FA_RETURN = 0x70,
        BOOT_WAITING_CMD = 0x80,
        BOOT_COMPLETED = 0xBC
    };

    // Helper function to convert bytes (big-endian) to double
    double bytes_to_double(const std::vector<uint8_t>& bytes, size_t offset)
    {
        if (offset + 8 > bytes.size()) {
            throw std::runtime_error("Insufficient bytes for double conversion");
        }

        uint64_t raw_value = 0;
        std::memcpy(&raw_value, &bytes[offset], 8);

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        raw_value = __builtin_bswap64(raw_value);
#endif

        double result;
        std::memcpy(&result, &raw_value, sizeof(double));
        return result;
    }
}

NativeVb1940Sensor::NativeVb1940Sensor(DataChannel& data_channel, std::shared_ptr<Synchronizer> synchronizer)
    : hololink_(data_channel.hololink())
    , i2c_(hololink_->get_i2c(data_channel.enumeration_metadata().get<int64_t>("i2c_bus").value()))
    , synchronizer_(synchronizer)
{
    sensor_id_ = DRIVER_NAME;
    initialize_supported_modes();
}

NativeVb1940Sensor::~NativeVb1940Sensor() = default;

void NativeVb1940Sensor::configure(CameraMode mode)
{
    // Make sure this is a version we know about.
    uint32_t version = get_version();
    HSB_LOG_INFO("version={}", version);
    assert(version == VERSION);

    // Get device id
    HSB_LOG_DEBUG("##1.get device id");
    get_device_id();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    uint8_t status = get_register(SYSTEM_FSM_STATE_REG);
    HSB_LOG_DEBUG("system fsm state:0x{:X}", status);

    if (status == static_cast<uint8_t>(SystemFsmState::SW_STBY)) {
        // Configure the camera based on the mode
        HSB_LOG_DEBUG("##sensor configuration");
        configure_camera(mode);
    } else {
        // Status check (step 2 from Python)
        HSB_LOG_DEBUG("##2.status check");
        status_check();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // Start sensor
        HSB_LOG_DEBUG("##3.start sensor");
        set_register_8(SYSTEM_UP_REG, 0x01);

        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            uint8_t ret = get_register(SYSTEM_FSM_STATE_REG);
            HSB_LOG_DEBUG("system fsm state:0x{:X}", ret);
            if (ret == static_cast<uint8_t>(SystemFsmState::BOOT)) {
                break;
            }
        }

        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            uint8_t ret = get_register(BOOT_FSM_REG);
            HSB_LOG_DEBUG("boot fsm state:0x{:X}", ret);
            if (ret == static_cast<uint8_t>(BootFsmState::WAIT_CERTIFICATE)) {
                break;
            }
        }

        // Secure boot
        HSB_LOG_DEBUG("##4.secure boot");
        do_secure_boot();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        uint8_t ret = get_register(SYSTEM_FSM_STATE_REG);
        HSB_LOG_DEBUG("system fsm state:0x{:X}", ret);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ret = get_register(SYSTEM_FSM_STATE_REG);
        HSB_LOG_DEBUG("system fsm state:0x{:X}", ret);

        // Update VT PATCH to RAM
        HSB_LOG_DEBUG("##5.VT_PATCH");
        write_vt_patch();

        // Configure the camera based on the mode
        HSB_LOG_DEBUG("##6.sensor configuration");
        configure_camera(mode);
    }
}

void NativeVb1940Sensor::set_mode(CameraMode mode)
{
    if (mode >= 0 && mode < vb1940_mode::VB1940_MODE_COUNT) {
        mode_ = mode;
        // Set the width, height, pixel format, and bayer format
        auto frame_format = mode_frame_formats_[mode_.value()];
        width_ = frame_format->width();
        height_ = frame_format->height();
        pixel_format_ = frame_format->pixel_format();
        bayer_format_ = hololink::csi::BayerFormat::GBRG;
    } else {
        HSB_LOG_ERROR("Incorrect mode for VB1940");
    }
}

void NativeVb1940Sensor::setup_clock()
{
    // Set the clock driver for TS2 (using renesas_bajoran_lite_ts2 like Python)
    hololink_->setup_clock(hololink::renesas_ts2::DEVICE_CONFIGURATION);
}

uint32_t NativeVb1940Sensor::get_version() const
{
    // TODO: get the version or the name of the sensor from the sensor
    return VERSION;
}

uint32_t NativeVb1940Sensor::get_device_id()
{
    uint32_t id = get_register_32(DEVICE_REVISION_REG);
    HSB_LOG_DEBUG("Device ID:0x{:X}", id);
    return id;
}

void NativeVb1940Sensor::status_check()
{
    uint32_t count = 0;
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        uint8_t ret = get_register(SYSTEM_FSM_STATE_REG);
        count++;
        if (count >= 30) {
            throw std::runtime_error("Incorrect system fsm state:0x" + std::to_string(ret) + ". Please reconnect camera");
        }
        HSB_LOG_DEBUG("system fsm state:0x{:X}", ret);
        if (ret == static_cast<uint8_t>(SystemFsmState::SYSTEM_UP)) {
            break;
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    set_register_8(SYSTEM_UP_REG, 0x01);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        uint8_t ret = get_register(SYSTEM_FSM_STATE_REG);
        HSB_LOG_DEBUG("system fsm state:0x{:X}", ret);
        if (ret == static_cast<uint8_t>(SystemFsmState::BOOT)) {
            break;
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        uint8_t ret = get_register(BOOT_FSM_REG);
        HSB_LOG_DEBUG("boot fsm state:0x{:X}", ret);
        if (ret == static_cast<uint8_t>(BootFsmState::WAIT_CERTIFICATE)) {
            break;
        }
    }
}

void NativeVb1940Sensor::do_secure_boot()
{
    // Load certificate
    HSB_LOG_DEBUG("##secure boot-load certificate");
    write_certificate();
    set_register_8(BOOT_REG, 0x01);
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        uint8_t ret = get_register(BOOT_FSM_REG);
        HSB_LOG_DEBUG("boot fsm state:0x{:X}", ret);
        if (ret == static_cast<uint8_t>(BootFsmState::BOOT_WAITING_CMD)) {
            break;
        }
    }
    // Load FWP
    HSB_LOG_DEBUG("##secure boot-load FWP");
    write_fw();
    set_register_8(BOOT_REG, 0x02);
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        uint8_t ret = get_register(BOOT_FSM_REG);
        HSB_LOG_DEBUG("boot fsm state:0x{:X}", ret);
        if (ret == static_cast<uint8_t>(BootFsmState::FWP_SETUP)) {
            break;
        }
    }
    // End boot
    HSB_LOG_DEBUG("##secure boot-end boot");
    set_register_8(BOOT_REG, 0x10);
}

void NativeVb1940Sensor::write_certificate()
{
    write_data_in_pages(vb1940_mode::VB1940_CERTIFICATE_START_ADDR, vb1940_mode::VB1940_CERTIFICATE);
}

void NativeVb1940Sensor::write_fw()
{
    write_data_in_pages(vb1940_mode::VB1940_FWP_START_ADDR, vb1940_mode::VB1940_FW);
}

void NativeVb1940Sensor::write_vt_patch()
{
    HSB_LOG_DEBUG("VT_PATCH: LEDC_RAM");
    write_data_in_pages(vb1940_mode::LDEC_RAM_CONTENT_START_ADDR, vb1940_mode::LDEC_RAM_CONTENT);

    HSB_LOG_DEBUG("VT_PATCH: RD_RAM_SEQ_1");
    write_data_in_pages(vb1940_mode::RD_RAM_SEQ_1_CONTENT_START_ADDR, vb1940_mode::RD_RAM_SEQ_1_CONTENT);

    HSB_LOG_DEBUG("VT_PATCH: GT_RAM_PAT");
    write_data_in_pages(vb1940_mode::GT_RAM_PAT_CONTENT_START_ADDR, vb1940_mode::GT_RAM_PAT_CONTENT);

    HSB_LOG_DEBUG("VT_PATCH: GT_RAM_SQE_1");
    write_data_in_pages(vb1940_mode::GT_RAM_SEQ_1_CONTENT_START_ADDR, vb1940_mode::GT_RAM_SEQ_1_CONTENT);

    HSB_LOG_DEBUG("VT_PATCH: GT_RAM_SQE_2");
    write_data_in_pages(vb1940_mode::GT_RAM_SEQ_2_CONTENT_START_ADDR, vb1940_mode::GT_RAM_SEQ_2_CONTENT);

    HSB_LOG_DEBUG("VT_PATCH: GT_RAM_SQE_3");
    write_data_in_pages(vb1940_mode::GT_RAM_SEQ_3_CONTENT_START_ADDR, vb1940_mode::GT_RAM_SEQ_3_CONTENT);

    HSB_LOG_DEBUG("VT_PATCH: GT_RAM_SQE_4");
    write_data_in_pages(vb1940_mode::GT_RAM_SEQ_4_CONTENT_START_ADDR, vb1940_mode::GT_RAM_SEQ_4_CONTENT);

    HSB_LOG_DEBUG("VT_PATCH: RD_RAM_PAT");
    write_data_in_pages(vb1940_mode::RD_RAM_PAT_CONTENT_START_ADDR, vb1940_mode::RD_RAM_PAT_CONTENT);
}

uint8_t NativeVb1940Sensor::get_register(uint16_t reg)
{
    HSB_LOG_DEBUG("get_register(register={}(0x{:X}))", reg, reg);

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

uint32_t NativeVb1940Sensor::get_register_32(uint16_t reg)
{
    HSB_LOG_DEBUG("get_register_32(register={}(0x{:X}))", reg, reg);

    std::vector<uint8_t> write_bytes(2);
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint16_be(static_cast<uint16_t>(reg)); // Use big-endian like Python

    const uint32_t read_byte_count = 4;

    // Read four bytes
    auto reply = i2c_->i2c_transaction(I2C_ADDRESS, write_bytes, read_byte_count);

    // Deserialize the reply
    core::Deserializer deserializer(reply.data(), reply.size());
    uint32_t value;
    if (!deserializer.next_uint32_be(value)) // Use big-endian like Python
    {
        throw std::runtime_error("Failed to read register value");
    }

    HSB_LOG_DEBUG("get_register_32(register={}(0x{:X}))={}(0x{:X})", reg, reg, value, value);

    return value;
}

void NativeVb1940Sensor::set_register_8(uint16_t reg, uint8_t value,
    const std::shared_ptr<Timeout>& timeout)
{
    HSB_LOG_DEBUG("set_register_8(register={}(0x{:X}), value={}(0x{:X}))", reg, reg, value, value);

    std::vector<uint8_t> write_bytes(3);
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint16_be(static_cast<uint16_t>(reg)); // Use big-endian like Python
    serializer.append_uint8(value);

    const uint32_t read_byte_count = 0;
    i2c_->i2c_transaction(I2C_ADDRESS, write_bytes, read_byte_count, timeout);
}

void NativeVb1940Sensor::set_register_16(uint16_t reg, uint16_t value,
    const std::shared_ptr<Timeout>& timeout)
{
    HSB_LOG_DEBUG("set_register_16(register={}(0x{:X}), value={}(0x{:X}))", reg, reg, value, value);

    std::vector<uint8_t> write_bytes(4);
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint16_be(static_cast<uint16_t>(reg)); // Use big-endian like Python
    serializer.append_uint16_be(value); // Use big-endian like Python

    const uint32_t read_byte_count = 0;
    i2c_->i2c_transaction(I2C_ADDRESS, write_bytes, read_byte_count, timeout);
}

void NativeVb1940Sensor::set_register_32(uint16_t reg, uint32_t value,
    const std::shared_ptr<Timeout>& timeout)
{
    HSB_LOG_DEBUG("set_register_32(register={}(0x{:X}), value={}(0x{:X}))", reg, reg, value, value);

    std::vector<uint8_t> write_bytes(6);
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint16_be(static_cast<uint16_t>(reg)); // Use big-endian like Python
    serializer.append_uint32_be(value); // Use big-endian like Python

    const uint32_t read_byte_count = 0;
    i2c_->i2c_transaction(I2C_ADDRESS, write_bytes, read_byte_count, timeout);
}

void NativeVb1940Sensor::set_register_buffer(uint16_t reg, const std::vector<uint8_t>& data_buffer,
    const std::shared_ptr<Timeout>& timeout)
{
    HSB_LOG_DEBUG("set_register_buffer(register={}(0x{:X}), data size={})", reg, reg, data_buffer.size());

    // Serialize the register (16-bit) and data buffer
    std::vector<uint8_t> write_bytes(2 + data_buffer.size());
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint16_be(static_cast<uint16_t>(reg)); // Use big-endian like Python
    serializer.append_buffer(const_cast<uint8_t*>(data_buffer.data()), data_buffer.size());

    const uint32_t read_byte_count = 0;
    i2c_->i2c_transaction(I2C_ADDRESS, write_bytes, read_byte_count, timeout);
}

void NativeVb1940Sensor::start()
{
    synchronizer_->attach(shared_from_this());

    apply_register_settings(vb1940_mode::VB1940_START_SEQUENCE);

    // Wait for streaming state
    uint32_t count = 0;
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(vb1940_mode::VB1940_WAIT_MS_START));
        uint8_t ret = get_register(SW_STBY_REG);
        HSB_LOG_DEBUG("SW_STBY state:0x{:X}", ret);
        count++;
        if (count == 30) {
            break;
        }
        if (ret == 0x00 || ret == 0x01) {
            break;
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    count = 0;
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(vb1940_mode::VB1940_WAIT_MS_START));
        uint8_t ret = get_register(SYSTEM_FSM_STATE_REG);
        HSB_LOG_DEBUG("system fsm state:0x{:X}", ret);
        count++;
        if (count == 30) {
            break;
        }
        if (ret == static_cast<uint8_t>(SystemFsmState::STREAMING)) {
            break;
        }
    }
}

void NativeVb1940Sensor::stop()
{
    apply_register_settings(vb1940_mode::VB1940_STOP_SEQUENCE);

    // Wait for stop
    uint32_t count = 0;
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(vb1940_mode::VB1940_WAIT_MS_START));
        uint8_t ret = get_register(STREAMING_REG);
        HSB_LOG_DEBUG("STREAMING state:0x{:X}", ret);
        count++;
        if (count == 30) {
            break;
        }
        if (ret == 0x00 || ret == 0x01) {
            break;
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    count = 0;
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(vb1940_mode::VB1940_WAIT_MS_START));
        uint8_t ret = get_register(SYSTEM_FSM_STATE_REG);
        HSB_LOG_DEBUG("system fsm state:0x{:X}", ret);
        count++;
        if (count == 30) {
            break;
        }
        if (ret == static_cast<uint8_t>(SystemFsmState::SW_STBY)) {
            break;
        }
    }

    synchronizer_->detach(shared_from_this());
}

void NativeVb1940Sensor::configure_converter(std::shared_ptr<hololink::csi::CsiConverter> converter)
{
    // Get starting byte position
    uint32_t start_byte = converter->receiver_start_byte();

    // Calculate transmitted and received line bytes
    uint32_t transmitted_line_bytes = converter->transmitted_line_bytes(pixel_format_, width_);
    uint32_t received_line_bytes = converter->received_line_bytes(transmitted_line_bytes);

    // VB1940 sensor has 1 line of status before the real image data starts
    start_byte += received_line_bytes;

    // VB1940 sensor has 2 lines of status after the real image data is complete
    uint32_t trailing_bytes = received_line_bytes * 2;

    // Configure the converter
    converter->configure(
        start_byte,
        received_line_bytes,
        width_,
        height_,
        pixel_format_,
        trailing_bytes);
}

void NativeVb1940Sensor::initialize_supported_modes()
{
    // Initialize supported modes
    supported_modes_ = {
        vb1940_mode::VB1940_MODE_2560X1984_30FPS,
        vb1940_mode::VB1940_MODE_1920X1080_30FPS,
        vb1940_mode::VB1940_MODE_2560X1984_30FPS_8BIT
    };

    // Initialize frame formats
    mode_frame_formats_[vb1940_mode::VB1940_MODE_2560X1984_30FPS] = std::make_shared<Vb1940FrameFormat>(vb1940_mode::VB1940_MODE_2560X1984_30FPS,
        "VB1940_MODE_2560X1984_30FPS",
        2560,
        1984,
        30,
        csi::PixelFormat::RAW_10);

    mode_frame_formats_[vb1940_mode::VB1940_MODE_1920X1080_30FPS] = std::make_shared<Vb1940FrameFormat>(vb1940_mode::VB1940_MODE_1920X1080_30FPS,
        "VB1940_MODE_1920X1080_30FPS",
        1920,
        1080,
        30,
        csi::PixelFormat::RAW_10);

    mode_frame_formats_[vb1940_mode::VB1940_MODE_2560X1984_30FPS_8BIT] = std::make_shared<Vb1940FrameFormat>(vb1940_mode::VB1940_MODE_2560X1984_30FPS_8BIT,
        "VB1940_MODE_2560X1984_30FPS_8BIT",
        2560,
        1984,
        30,
        csi::PixelFormat::RAW_8);
}

void NativeVb1940Sensor::set_exposure_reg(int32_t value)
{
    // The minimum integration time is 30us(4lines).
    // value: integration time in lines, in little endian.
    if (value < 0x0004) {
        HSB_LOG_WARN("Exposure value ({}) is below the minimum limit of 0x0004 for VB1940. Setting to 0x0004.", value);
        value = 0x0004;
    } else if (value > 0xFFFF) {
        HSB_LOG_WARN("Exposure value ({}) is above the maximum limit of 0xFFFF for VB1940. Setting to 0xFFFF.", value);
        value = 0xFFFF;
    }

    // If set_register_16 is used to set exposure, change the value passed in into big endian
    set_register_8(vb1940_mode::REG_EXP, static_cast<uint8_t>(value & 0xFF));
    set_register_8(vb1940_mode::REG_EXP + 1, static_cast<uint8_t>((value >> 8) & 0xFF));
    std::this_thread::sleep_for(std::chrono::milliseconds(vb1940_mode::VB1940_WAIT_MS));
}

void NativeVb1940Sensor::set_analog_gain_reg(int32_t value)
{
    if (value < 0x00) {
        HSB_LOG_WARN("Analog gain value ({}) is below the minimum limit of 0x00 for VB1940. Setting to 0x00.", value);
        value = 0x00;
    } else if (value > 0x18) {
        HSB_LOG_WARN("Analog gain value ({}) is above the maximum limit of 0x18 for VB1940. Setting to 0x18.", value);
        value = 0x18;
    }

    set_register_8(vb1940_mode::REG_AG, static_cast<uint8_t>(value));
    std::this_thread::sleep_for(std::chrono::milliseconds(vb1940_mode::VB1940_WAIT_MS));
}

void NativeVb1940Sensor::configure_camera(CameraMode mode)
{
    set_mode(mode);

    switch (mode_.value()) {
    case vb1940_mode::VB1940_MODE_2560X1984_30FPS:
        apply_register_settings(vb1940_mode::VB1940_MODE_2560X1984_30FPS_SEQUENCE);
        break;
    case vb1940_mode::VB1940_MODE_1920X1080_30FPS:
        apply_register_settings(vb1940_mode::VB1940_MODE_1920X1080_30FPS_SEQUENCE);
        break;
    case vb1940_mode::VB1940_MODE_2560X1984_30FPS_8BIT:
        apply_register_settings(vb1940_mode::VB1940_MODE_2560X1984_30FPS_8BIT_SEQUENCE);
        break;
    default:
        HSB_LOG_ERROR("Unsupported camera mode: {}", mode_.value());
        break;
    }
}

template <typename ContainerT>
void NativeVb1940Sensor::apply_register_settings(const ContainerT& settings)
{
    static_assert(std::is_same_v<typename ContainerT::value_type,
                      std::pair<uint16_t, uint8_t>>,
        "Container must hold std::pair<uint16_t, uint8_t>");

    for (const auto& reg_val : settings) {
        if (reg_val.first == vb1940_mode::VB1940_TABLE_WAIT_MS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(reg_val.second));
        } else {
            uint8_t value = reg_val.second;
            // Handle trigger mode: modify register 0xAC6 when trigger mode is enabled
            if (reg_val.first == 0xAC6 && synchronizer_->is_enabled()) {
                value = 0x01;
            }
            set_register_8(reg_val.first, value);
        }
    }
}

void NativeVb1940Sensor::write_data_in_pages(uint32_t start_addr, const std::vector<uint8_t>& data)
{
    if (data.empty())
        return;

    uint32_t data_size = data.size();
    uint32_t page_size = vb1940_mode::VB1940_PAGE_SIZE;
    uint32_t page_count = (data_size + page_size - 1) / page_size;

    for (uint32_t id = 0; id < page_count; ++id) {
        uint32_t offset = id * page_size;
        uint16_t reg = static_cast<uint16_t>(start_addr + offset);
        uint32_t remaining_size = std::min(data_size - offset, page_size);

        std::vector<uint8_t> data_buffer(data.begin() + offset, data.begin() + offset + remaining_size);
        if (!data_buffer.empty()) {
            set_register_buffer(reg, data_buffer);
        }
    }
}

uint8_t NativeVb1940Sensor::get_eeprom_register(uint16_t reg)
{
    HSB_LOG_DEBUG("get_eeprom_register(register={}(0x{:X}))", reg, reg);

    std::vector<uint8_t> write_bytes(2);
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint16_be(static_cast<uint16_t>(reg));

    const uint32_t read_byte_count = 1;

    // Read one byte
    auto reply = i2c_->i2c_transaction(EEPROM_I2C_ADDRESS, write_bytes, read_byte_count);

    // Deserialize the reply
    core::Deserializer deserializer(reply.data(), reply.size());
    uint8_t value;
    if (!deserializer.next_uint8(value)) {
        throw std::runtime_error("Failed to read register value");
    }

    HSB_LOG_DEBUG("get_eeprom_register(register={}(0x{:X}),value={}(0x{:X}))", reg, reg, value, value);

    return value;
}

std::vector<uint8_t> NativeVb1940Sensor::get_eeprom_page(uint32_t page_num, uint32_t page_offset, uint32_t data_len)
{
    if (page_num >= EEPROM_MAX_PAGE_NUM) {
        throw std::runtime_error("Page number should be in range from 0 to 255");
    }
    if (page_offset >= EEPROM_PAGE_SIZE || data_len > EEPROM_PAGE_SIZE - page_offset) {
        throw std::out_of_range("EEPROM read exceeds page boundary");
    }
    uint16_t register_addr = static_cast<uint16_t>((page_num << 6) + page_offset);
    std::vector<uint8_t> write_bytes(2);
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint16_be(register_addr);
    const uint32_t read_byte_count = data_len;
    auto reply = i2c_->i2c_transaction(EEPROM_I2C_ADDRESS, write_bytes, read_byte_count);
    core::Deserializer deserializer(reply.data(), reply.size());
    std::vector<uint8_t> data(read_byte_count);
    deserializer.next_buffer(data);

    if (data.size() != read_byte_count) {
        throw std::runtime_error("Read " + std::to_string(data.size()) + " != " + std::to_string(read_byte_count));
    }

    HSB_LOG_DEBUG("get_eeprom_page(register={}(0x{:X}),buffer_size={})", register_addr, register_addr, data.size());

    return data;
}

void NativeVb1940Sensor::set_eeprom_register(uint16_t reg, uint8_t value, const std::shared_ptr<Timeout>& timeout)
{
    HSB_LOG_DEBUG("set_eeprom_register(register={}(0x{:X}), value={}(0x{:X}))", reg, reg, value, value);

    std::vector<uint8_t> write_bytes(3);
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint16_be(static_cast<uint16_t>(reg));
    serializer.append_uint8(value);

    const uint32_t read_byte_count = 0;
    i2c_->i2c_transaction(EEPROM_I2C_ADDRESS, write_bytes, read_byte_count, timeout);
}

void NativeVb1940Sensor::set_eeprom_page(uint32_t page_num, uint32_t page_offset, const std::vector<uint8_t>& data_buffer, const std::shared_ptr<Timeout>& timeout)
{
    if (page_num >= EEPROM_MAX_PAGE_NUM) {
        throw std::runtime_error("Page number should be in range from 0 to 255");
    }
    if (page_offset + data_buffer.size() > EEPROM_PAGE_SIZE) {
        throw std::runtime_error("page_offset(" + std::to_string(page_offset) + ") + data_len(" + std::to_string(data_buffer.size()) + ") should not be greater than 64");
    }

    uint16_t register_addr = static_cast<uint16_t>((page_num << 6) + page_offset);
    HSB_LOG_DEBUG("set_eeprom_page(page_num={}(0x{:X}), page_offset={}(0x{:X}), data_len={})",
        page_num, page_num, page_offset, page_offset, data_buffer.size());

    std::vector<uint8_t> write_bytes(2 + data_buffer.size());
    core::Serializer serializer(write_bytes.data(), write_bytes.size());
    serializer.append_uint16_be(register_addr);
    serializer.append_buffer(const_cast<uint8_t*>(data_buffer.data()), data_buffer.size());

    const uint32_t read_byte_count = 0;
    i2c_->i2c_transaction(EEPROM_I2C_ADDRESS, write_bytes, read_byte_count, timeout);
}

CalibrationData NativeVb1940Sensor::parse_calibration_data(const std::vector<double>& data)
{
    CalibrationData calib_data;

    size_t left_index = static_cast<size_t>(CalibrationData::CameraIndex::kLeft);
    size_t right_index = static_cast<size_t>(CalibrationData::CameraIndex::kRight);

    calib_data.intrinsic_parameters[left_index][0] = data[0]; // fx
    calib_data.intrinsic_parameters[left_index][1] = data[1]; // fy
    calib_data.intrinsic_parameters[left_index][2] = data[2]; // cx
    calib_data.intrinsic_parameters[left_index][3] = data[3]; // cy

    for (int i = 0; i < 8; ++i) {
        calib_data.distortion_parameters[left_index][i] = data[4 + i];
    }

    calib_data.intrinsic_parameters[right_index][0] = data[12]; // fx
    calib_data.intrinsic_parameters[right_index][1] = data[13]; // fy
    calib_data.intrinsic_parameters[right_index][2] = data[14]; // cx
    calib_data.intrinsic_parameters[right_index][3] = data[15]; // cy

    for (int i = 0; i < 8; ++i) {
        calib_data.distortion_parameters[right_index][i] = data[16 + i];
    }

    calib_data.R = std::array<double, 3>({ data[24], data[25], data[26] });
    calib_data.T = std::array<double, 3>({ data[27], data[28], data[29] });
    return calib_data;
}

CalibrationData NativeVb1940Sensor::read_calibration_from_eeprom(bool rgb)
{
    // Read the calibration data from EEPROM.
    // Following is EEPROM layout of every part of
    // calibration data. Each part occupies 256 bytes.
    // The prefixs 'L' and 'R' denote left and right,
    // respectively.
    // |-8----8-----8-----8-----8-----8-----8-----8-|
    // L_fx  L_fy  L_cx  L_cy  L_k1  L_k2  L_p1  L_p2
    // L_k3  L_k4  L_k5  L_k6  R_fx  R_fy  R_cx  R_cy
    // R_k1  R_k2  R_p1  R_p2  R_k3  R_k4  R_k5  R_k6
    // Rx    Ry    Rz    Tx    Ty    Tz    sn

    CalibrationData calib_data;
    uint32_t calib_pages = CALIB_SIZE / EEPROM_PAGE_SIZE;
    const uint32_t step = 8;

    try {
        // Get raw data from EEPROM pages
        std::vector<uint8_t> calib_data_raw;
        if (rgb) {
            // RGB calibration data: pages 0-3
            HSB_LOG_DEBUG("Reading RGB calibration data from EEPROM pages 0-3");
            for (uint32_t page = 0; page < calib_pages; ++page) {
                auto page_data = get_eeprom_page(page);
                calib_data_raw.insert(calib_data_raw.end(), page_data.begin(), page_data.end());
            }
        } else {
            // IR calibration data: pages 4-7
            HSB_LOG_DEBUG("Reading IR calibration data from EEPROM pages 4-7");
            for (uint32_t page = calib_pages; page < calib_pages * 2; ++page) {
                auto page_data = get_eeprom_page(page);
                calib_data_raw.insert(calib_data_raw.end(), page_data.begin(), page_data.end());
            }
        }

        // Convert bytes to doubles
        std::vector<double> calib_data_parsed;
        for (uint32_t i = 0; i < 30; ++i) { // 30 double values instead of 32
            try {
                double parsed_data = bytes_to_double(calib_data_raw, i * step);
                calib_data_parsed.push_back(parsed_data);
            } catch (const std::exception& e) {
                HSB_LOG_ERROR("Error in parsing {} calibration data: {}", rgb ? "RGB" : "IR", e.what());
                throw;
            }
        }

        // Parse calibration data
        calib_data = parse_calibration_data(calib_data_parsed);

    } catch (const std::exception& e) {
        HSB_LOG_ERROR("Failed to read {} calibration data: {}", rgb ? "RGB" : "IR", e.what());
        throw;
    }

    return calib_data;
}

CalibrationData NativeVb1940Sensor::get_rgb_calibration_data()
{
    return read_calibration_from_eeprom(true);
}

CalibrationData NativeVb1940Sensor::get_ir_calibration_data()
{
    return read_calibration_from_eeprom(false);
}

std::string CalibrationData::to_string() const
{
    std::stringstream ss;

    // Print intrinsic parameters
    ss << "Intrinsics (fx, fy, cx, cy):\n";
    for (size_t i = 0; i < intrinsic_parameters.size(); ++i) {
        const auto& intrinsic = intrinsic_parameters[i];
        ss << "[" << i << "]: " << std::fixed << std::setprecision(6) << intrinsic[0]
           << ", " << intrinsic[1] << ", " << intrinsic[2] << ", " << intrinsic[3] << "\n";
    }

    // Print distortion parameters
    ss << "Distortion (k1, k2, p1, p2, k3, k4, k5, k6):\n";
    for (size_t i = 0; i < distortion_parameters.size(); ++i) {
        const auto& distortion = distortion_parameters[i];
        ss << "[" << i << "]: " << std::fixed << std::setprecision(6) << distortion[0]
           << ", " << distortion[1] << ", " << distortion[2] << ", " << distortion[3]
           << ", " << distortion[4] << ", " << distortion[5] << ", " << distortion[6] << ", " << distortion[7] << "\n";
    }

    // Print rotation parameters
    ss << "Rotation (Rx, Ry, Rz): " << std::fixed << std::setprecision(6)
       << R[0] << ", " << R[1] << ", " << R[2] << "\n";

    // Print translation parameters
    ss << "Translation (Tx, Ty, Tz): " << std::fixed << std::setprecision(6)
       << T[0] << ", " << T[1] << ", " << T[2];

    return ss.str();
}

} // namespace hololink::sensors
