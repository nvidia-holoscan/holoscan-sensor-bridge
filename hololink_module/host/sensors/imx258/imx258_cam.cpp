/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Module-native IMX258 camera driver. Ports the Imx258Camera class of the
 * reference bring-up camera setup (camera_hdmi_setup.py: I2C sequence, model-ID check,
 * --verify-registers readback) onto the module V1 I2C surface, following
 * hololink_module/host/sensors/imx274/imx274_cam.cpp minus the expander
 * and oscillator. */

#include "hololink/module/sensors/imx258/imx258_cam.hpp"

#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include "hololink/module/logging.hpp"

namespace hololink::module::sensors::imx258 {

Imx258Cam::Imx258Cam(const EnumerationMetadata& metadata, uint32_t i2c_bus)
    : Imx258Cam(HololinkInterfaceV1::get_service(metadata), i2c_bus)
{
}

Imx258Cam::Imx258Cam(std::shared_ptr<HololinkInterfaceV1> hololink, uint32_t i2c_bus)
    : hololink_(std::move(hololink))
    , i2c_bus_(i2c_bus)
{
    if (!hololink_) {
        throw std::runtime_error(
            "While constructing Imx258Cam: hololink handle is null");
    }
    i2c_ = hololink_->get_i2c<>(i2c_bus_, IMX258_I2C_ADDRESS);
}

uint16_t Imx258Cam::model_id()
{
    const uint16_t hi = get_register(REG_MODEL_ID_HI);
    const uint16_t lo = get_register(REG_MODEL_ID_LO);
    return static_cast<uint16_t>((hi << 8) | lo);
}

// Deviation from the ported reference (camera_hdmi_setup.py
// --verify-registers): the reference only WARNED on readback mismatches and
// carried on. Here any mismatch throws once the table has been walked -- a
// sensor that does not hold its configuration would otherwise stream garbage
// that only shows up as a corrupted image downstream. Every mismatch is still
// logged first so the failing registers are visible.
void Imx258Cam::configure(Imx258_Mode mode, bool verify_registers)
{
    if (get_version() != VERSION) {
        throw std::runtime_error(
            "While configuring Imx258Cam: driver VERSION mismatch");
    }
    set_mode(mode);

    // Prove the sensor is actually there before writing 80-odd registers
    // at it. A write that does not NAK only tells us the controller was
    // happy; reading the model ID back tells us something answered.
    const uint16_t id = model_id();
    HSB_LOG_INFO("IMX258 model ID = {:#06x} (expect {:#06x})", id, MODEL_ID);
    if (id != MODEL_ID) {
        throw std::runtime_error(fmt::format(
            "While configuring Imx258Cam: sensor model ID {:#06x} != {:#06x}; "
            "nothing is answering at I2C address {:#04x} on bus {}",
            id, MODEL_ID, IMX258_I2C_ADDRESS, i2c_bus_));
    }

    const std::pair<uint16_t, uint8_t>* table = nullptr;
    size_t table_size = 0;
    const char* lane_rate = nullptr;
    switch (mode) {
    case Imx258_Mode::IMX258_MODE_1920X1080_60FPS_1194MBPS:
        table = IMX258_MODE_1920X1080_60FPS_1194MBPS_SEQUENCE.data();
        table_size = IMX258_MODE_1920X1080_60FPS_1194MBPS_SEQUENCE.size();
        lane_rate = "1194.75 Mb/s/lane";
        break;
    default:
        throw std::runtime_error(
            "While configuring Imx258Cam: requested mode is not supported");
    }

    unsigned mismatches = 0;
    for (size_t i = 0; i < table_size; i++) {
        const auto& [reg, value] = table[i];
        set_register(reg, value);
        // REG_MODE_SELECT latches stream state rather than reading back
        // as written.
        if (verify_registers && reg != REG_MODE_SELECT) {
            const uint8_t readback = get_register(reg);
            if (readback != value) {
                HSB_LOG_WARN("IMX258 reg {:#06x}: wrote {:#04x}, read {:#04x}",
                    reg, value, readback);
                mismatches++;
            }
        }
    }
    if (mismatches) {
        throw std::runtime_error(fmt::format(
            "While configuring Imx258Cam: {} register(s) did not read back "
            "as written",
            mismatches));
    }
    HSB_LOG_INFO("IMX258 configured: RAW10 {}x{}, 4 lanes at {}, ~{} Hz",
        width_, height_, lane_rate, framerate_);
}

void Imx258Cam::start()
{
    set_register(REG_MODE_SELECT, 0x01);
}

void Imx258Cam::stop()
{
    set_register(REG_MODE_SELECT, 0x00);
}

Imx258Cam::StreamingStatus Imx258Cam::streaming_status(std::chrono::milliseconds sample_interval)
{
    StreamingStatus status;
    status.mode_select = get_register(REG_MODE_SELECT);
    status.frame_count_before = get_register(REG_FRAME_COUNT);
    std::this_thread::sleep_for(sample_interval);
    status.frame_count_after = get_register(REG_FRAME_COUNT);
    status.advancing = status.frame_count_after != status.frame_count_before;
    return status;
}

uint8_t Imx258Cam::get_register(uint16_t reg)
{
    std::vector<uint8_t> write_bytes {
        static_cast<uint8_t>((reg >> 8) & 0xFF),
        static_cast<uint8_t>(reg & 0xFF),
    };
    std::vector<uint8_t> read_bytes(1);

    std::unique_ptr<I2cLockV1> i2c_lock_handle;
    if (hololink_->i2c_lock(i2c_lock_handle) != HOLOLINK_MODULE_OK || !i2c_lock_handle) {
        throw std::runtime_error(
            "While reading Imx258 register: failed to acquire I2C lock");
    }
    std::lock_guard<I2cLockV1> guard(*i2c_lock_handle);
    const hololink_module_status_t s = i2c_->i2c_transaction(
        IMX258_I2C_ADDRESS, write_bytes, read_bytes);
    if (s != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(fmt::format(
            "While reading Imx258 register {:#06x}: i2c_transaction failed", reg));
    }
    return read_bytes.empty() ? 0 : read_bytes[0];
}

void Imx258Cam::set_register(uint16_t reg, uint8_t value)
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
            "While writing Imx258 register: failed to acquire I2C lock");
    }
    std::lock_guard<I2cLockV1> guard(*i2c_lock_handle);
    const hololink_module_status_t s = i2c_->i2c_transaction(
        IMX258_I2C_ADDRESS, write_bytes, read_bytes);
    if (s != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(fmt::format(
            "While writing Imx258 register {:#06x}: i2c_transaction failed", reg));
    }
}

void Imx258Cam::set_mode(Imx258_Mode mode)
{
    const size_t mode_index = static_cast<size_t>(mode);
    if (mode_index >= IMX258_FRAME_FORMAT.size()) {
        throw std::runtime_error(
            "While setting Imx258Cam mode: requested mode is out of range");
    }
    const FrameFormat& fmt = IMX258_FRAME_FORMAT[mode_index];
    mode_ = mode;
    width_ = fmt.width;
    height_ = fmt.height;
    framerate_ = fmt.framerate;
    pixel_format_ = fmt.pixel_format;
}

void Imx258Cam::set_exposure_reg(uint16_t value)
{
    set_register(REG_COARSE_INTEG_HI, static_cast<uint8_t>((value >> 8) & 0xFF));
    set_register(REG_COARSE_INTEG_LO, static_cast<uint8_t>(value & 0xFF));
}

void Imx258Cam::set_analog_gain_reg(uint16_t value)
{
    set_register(REG_ANA_GAIN_HI, static_cast<uint8_t>((value >> 8) & 0xFF));
    set_register(REG_ANA_GAIN_LO, static_cast<uint8_t>(value & 0xFF));
}

void Imx258Cam::configure_converter(std::shared_ptr<CsiConverterV1> converter)
{
    if (!converter) {
        throw std::runtime_error("Invalid converter passed to Imx258Cam::configure_converter.");
    }
    if (pixel_format_ != PixelFormat::RAW_10) {
        throw std::runtime_error(
            "While configuring the CSI converter for Imx258Cam: pixel format "
            "is not supported");
    }
    configure_converter(*converter, width_, height_);
}

void Imx258Cam::configure_converter(CsiConverterV1& converter,
    uint32_t width, uint32_t height)
{
    // The sensor emits a measured non-image prefix ahead of pixel row 0
    // (see RAW10_START_BYTE); no metadata/optical-black line arithmetic
    // as on the IMX274.
    const uint32_t start_byte = converter.receiver_start_byte() + RAW10_START_BYTE;
    const uint32_t transmitted_line_bytes
        = converter.transmitted_line_bytes(PixelFormat::RAW_10, width);
    const uint32_t received_line_bytes
        = converter.received_line_bytes(transmitted_line_bytes);
    converter.configure(start_byte, received_line_bytes, width, height, PixelFormat::RAW_10);
}

} // namespace hololink::module::sensors::imx258
