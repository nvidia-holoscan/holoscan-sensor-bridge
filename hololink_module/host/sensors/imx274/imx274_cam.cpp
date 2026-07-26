/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/sensors/imx274/imx274_cam.hpp"

#include <chrono>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace hololink::module::sensors::imx274 {

// expander_configuration source for the metadata-based constructor:
// prefer an explicit "expander_configuration" entry when the
// supplement stamps one, else fall back to "data_plane" (the HSB-Lite
// carrier conflates the two — the data_plane index doubles as the
// I2C-expander output selector).
static uint32_t expander_configuration_from(const EnumerationMetadata& metadata)
{
    if (metadata.contains("expander_configuration")) {
        return static_cast<uint32_t>(
            metadata.get<int64_t>("expander_configuration"));
    }
    return static_cast<uint32_t>(metadata.get<int64_t>("data_plane"));
}

static uint8_t expander_mask_for_configuration(uint32_t configuration)
{
    switch (configuration) {
    case 0:
        return static_cast<uint8_t>(LIExpanderOutputEN::OUTPUT_1);
    case 1:
        return static_cast<uint8_t>(LIExpanderOutputEN::OUTPUT_2);
    case 2:
        return static_cast<uint8_t>(LIExpanderOutputEN::OUTPUT_3);
    case 3:
        return static_cast<uint8_t>(LIExpanderOutputEN::OUTPUT_4);
    default:
        throw std::runtime_error(
            "While selecting LI I2C expander output for Imx274Cam: "
            "expander_configuration must be in [0, 3]");
    }
}

Imx274Cam::Imx274Cam(const EnumerationMetadata& metadata, uint32_t i2c_bus)
    : Imx274Cam(
        HololinkInterfaceV1::get_service(metadata),
        OscillatorInterfaceV1::get_service(metadata),
        i2c_bus,
        expander_configuration_from(metadata))
{
}

void Imx274Cam::use_expander_configuration(EnumerationMetadata& metadata,
    uint32_t expander_configuration)
{
    metadata["expander_configuration"] = static_cast<int64_t>(expander_configuration);
}

Imx274Cam::Imx274Cam(std::shared_ptr<HololinkInterfaceV1> hololink,
    std::shared_ptr<OscillatorInterfaceV1> oscillator,
    uint32_t i2c_bus, uint32_t expander_configuration)
    : hololink_(std::move(hololink))
    , oscillator_(std::move(oscillator))
    , i2c_bus_(i2c_bus)
{
    if (!hololink_) {
        throw std::runtime_error(
            "While constructing Imx274Cam: hololink handle is null");
    }
    if (!oscillator_) {
        throw std::runtime_error(
            "While constructing Imx274Cam: oscillator handle is null");
    }
    i2c_ = hololink_->get_i2c<>(i2c_bus_, CAM_I2C_ADDRESS);
    auto expander_i2c = hololink_->get_i2c<>(
        i2c_bus_, LII2CExpander::I2C_EXPANDER_ADDRESS);
    i2c_expander_ = std::make_shared<LII2CExpander>(std::move(expander_i2c));
    i2c_expander_configuration_ = expander_mask_for_configuration(expander_configuration);
}

void Imx274Cam::setup_clock(const std::vector<std::vector<uint8_t>>& clock_profile)
{
    if (clock_profile.empty()) {
        throw std::runtime_error(
            "While calling Imx274Cam::setup_clock: clock_profile is empty "
            "(typically obtained from the HSB-Lite supplement)");
    }
    // The module delegates board-level clock setup through the
    // HsbLiteInterfaceV1 supplement, not through the camera driver
    // directly. setup_clock here is kept for parity with the Python
    // surface so application code can read the clock profile through
    // a single object — but the actual write goes through the
    // supplement. Callers that have a HsbLiteInterface in hand should
    // call it directly; this method is a convenience.
    throw std::runtime_error(
        "While calling Imx274Cam::setup_clock: drive the HSB-Lite supplement's "
        "HsbLiteInterface::setup_clock from the application; the IMX274 driver "
        "doesn't reach the on-board clock chip directly");
}

void Imx274Cam::configure(Imx274_Mode mode)
{
    if (get_version() != VERSION) {
        throw std::runtime_error(
            "While configuring Imx274Cam: driver VERSION mismatch");
    }
    // IMX274 needs a 25 MHz reference clock on the HSB-Lite carrier.
    // The per-data-plane oscillator the supplement publishes either
    // programs that rate (returns true) or rejects the request
    // (returns false) — bail out before touching the sensor if the
    // oscillator can't deliver.
    if (!oscillator_->enable(25'000'000)) {
        throw std::runtime_error(
            "While configuring Imx274Cam: oscillator does not support the "
            "IMX274's 25 MHz reference clock");
    }
    configure_camera(mode);
}

void Imx274Cam::start()
{
    for (const auto& [reg, val] : IMX274_START_SEQUENCE) {
        if (reg == IMX274_TABLE_WAIT_MS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(val));
        } else {
            set_register(reg, val);
        }
    }
}

void Imx274Cam::stop()
{
    for (const auto& [reg, val] : IMX274_STOP_SEQUENCE) {
        if (reg == IMX274_TABLE_WAIT_MS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(val));
        } else {
            set_register(reg, val);
        }
    }
    // Let the egress buffer drain.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

uint8_t Imx274Cam::get_register(uint16_t reg)
{
    std::vector<uint8_t> write_bytes {
        static_cast<uint8_t>((reg >> 8) & 0xFF),
        static_cast<uint8_t>(reg & 0xFF),
    };
    std::vector<uint8_t> read_bytes(1);

    std::unique_ptr<I2cLockV1> i2c_lock_handle;
    if (hololink_->i2c_lock(i2c_lock_handle) != HOLOLINK_MODULE_OK || !i2c_lock_handle) {
        throw std::runtime_error(
            "While reading Imx274 register: failed to acquire I2C lock");
    }
    std::lock_guard<I2cLockV1> guard(*i2c_lock_handle);
    i2c_expander_->configure(i2c_expander_configuration_);
    const hololink_module_status_t s = i2c_->i2c_transaction(
        CAM_I2C_ADDRESS, write_bytes, read_bytes);
    if (s != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(
            "While reading Imx274 register: i2c_transaction failed");
    }
    return read_bytes.empty() ? 0 : read_bytes[0];
}

void Imx274Cam::set_register(uint16_t reg, uint8_t value)
{
    std::vector<uint8_t> write_bytes {
        static_cast<uint8_t>((reg >> 8) & 0xFF),
        static_cast<uint8_t>(reg & 0xFF),
        value,
    };
    std::vector<uint8_t> read_bytes;

    std::unique_ptr<I2cLockV1> i2c_lock_handle;
    if (hololink_->i2c_lock(i2c_lock_handle) != HOLOLINK_MODULE_OK || !i2c_lock_handle) {
        throw std::runtime_error(
            "While writing Imx274 register: failed to acquire I2C lock");
    }
    std::lock_guard<I2cLockV1> guard(*i2c_lock_handle);
    i2c_expander_->configure(i2c_expander_configuration_);
    const hololink_module_status_t s = i2c_->i2c_transaction(
        CAM_I2C_ADDRESS, write_bytes, read_bytes);
    if (s != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(
            "While writing Imx274 register: i2c_transaction failed");
    }
}

template <typename TableT>
static void apply_register_table(Imx274Cam& cam, const TableT& table)
{
    for (const auto& [reg, val] : table) {
        if (reg == IMX274_TABLE_WAIT_MS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(val));
        } else {
            cam.set_register(reg, val);
        }
    }
}

void Imx274Cam::configure_camera(Imx274_Mode mode)
{
    set_mode(mode);
    switch (mode) {
    case Imx274_Mode::IMX274_MODE_3840X2160_60FPS:
        apply_register_table(*this, IMX274_MODE_3840X2160_60FPS_SEQUENCE);
        break;
    case Imx274_Mode::IMX274_MODE_1920X1080_60FPS:
        apply_register_table(*this, IMX274_MODE_1920X1080_60FPS_SEQUENCE);
        break;
    case Imx274_Mode::IMX274_MODE_3840X2160_60FPS_12BITS:
        apply_register_table(*this, IMX274_MODE_3840X2160_60FPS_12BITS_SEQUENCE);
        break;
    default:
        throw std::runtime_error(
            "While configuring Imx274Cam: requested mode is not supported");
    }
}

void Imx274Cam::set_mode(Imx274_Mode mode)
{
    const size_t mode_index = static_cast<size_t>(mode);
    if (mode_index >= IMX_FRAME_FORMAT.size()) {
        throw std::runtime_error(
            "While setting Imx274Cam mode: requested mode is out of range");
    }
    const FrameFormat& fmt = IMX_FRAME_FORMAT[mode_index];
    mode_ = mode;
    width_ = fmt.width;
    height_ = fmt.height;
    pixel_format_ = fmt.pixel_format;
}

void Imx274Cam::set_exposure_reg(uint16_t value)
{
    if (value < 0x000C) {
        value = 0x000C;
    }
    set_register(REG_EXP_LSB, static_cast<uint8_t>((value >> 8) & 0xFF));
    set_register(REG_EXP_MSB, static_cast<uint8_t>(value & 0xFF));
    std::this_thread::sleep_for(std::chrono::milliseconds(IMX274_WAIT_MS));
}

void Imx274Cam::set_digital_gain_reg(uint16_t value)
{
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
    set_register(REG_DG, reg_value);
    std::this_thread::sleep_for(std::chrono::milliseconds(IMX274_WAIT_MS));
}

void Imx274Cam::set_analog_gain_reg(uint16_t value)
{
    set_register(REG_AG_LSB, static_cast<uint8_t>((value >> 8) & 0xFF));
    set_register(REG_AG_MSB, static_cast<uint8_t>(value & 0xFF));
    std::this_thread::sleep_for(std::chrono::milliseconds(IMX274_WAIT_MS));
}

void Imx274Cam::configure_converter(std::shared_ptr<CsiConverterV1> converter)
{
    if (!converter) {
        throw std::runtime_error("Invalid converter passed to Imx274Cam::configure_converter.");
    }

    uint32_t start_byte = converter->receiver_start_byte();
    const uint32_t transmitted_line_bytes
        = converter->transmitted_line_bytes(pixel_format_, width_);
    const uint32_t received_line_bytes
        = converter->received_line_bytes(transmitted_line_bytes);

    // 175 bytes of metadata precede image data.
    start_byte += converter->received_line_bytes(175);
    if (pixel_format_ == PixelFormat::RAW_10) {
        // 8 lines of optical black before real image data.
        start_byte += received_line_bytes * 8;
    } else if (pixel_format_ == PixelFormat::RAW_12) {
        // 16 lines of optical black before real image data.
        start_byte += received_line_bytes * 16;
    } else {
        throw std::runtime_error(
            "While configuring the CSI converter for Imx274Cam: pixel format "
            "is not supported");
    }
    converter->configure(start_byte, received_line_bytes, width_, height_, pixel_format_);
}

void Imx274Cam::test_pattern_disable()
{
    set_register(0x303C, 0);
    set_register(0x377F, 0);
    set_register(0x3781, 0);
    set_register(0x370B, 0);
}

void Imx274Cam::test_pattern_enable(uint8_t pattern)
{
    set_register(0x303C, 0x11);
    set_register(0x370E, 0x01);
    set_register(0x377F, 0x01);
    set_register(0x3781, 0x01);
    set_register(0x370B, 0x11);
    set_register(0x303D, pattern);
}

} // namespace hololink::module::sensors::imx274
