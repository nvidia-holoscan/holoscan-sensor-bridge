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

#include "ar0234.hpp"

#include <cassert>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include <hololink/core/logging_internal.hpp>

namespace hololink::module::sensors::ar0234 {

Ar0234::Ar0234(std::shared_ptr<HololinkInterfaceV1> hololink,
    uint32_t i2c_address,
    uint32_t i2c_bus,
    bool skip_i2c)
    : i2c_address_(i2c_address)
    , i2c_bus_(i2c_bus)
    , skip_i2c_(skip_i2c)
{
    if (!hololink) {
        throw std::runtime_error(
            "While constructing Ar0234: hololink handle is null");
    }
    i2c_ = hololink->get_i2c<>(i2c_bus_, i2c_address_);
}

uint32_t Ar0234::get_i2c_address() const
{
    return i2c_address_;
}

void Ar0234::set_i2c_address(uint32_t i2c_address)
{
    i2c_address_ = i2c_address;
}

uint16_t Ar0234::get_register(uint16_t reg)
{
    if (skip_i2c_) {
        HSB_LOG_DEBUG("Skip i2c");
        return 0;
    }

    HSB_LOG_DEBUG("get_register(register={}(0x{:X}))", reg, reg);

    std::vector<uint8_t> write_bytes = { uint8_t(reg >> 8), uint8_t(reg & 0xFF) };
    std::vector<uint8_t> read_bytes(2);
    const auto s = i2c_->i2c_transaction(i2c_address_, write_bytes, read_bytes);
    if (s != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(fmt::format(
            "While reading AR0234 register: i2c_addr=0x{:02X}, register=0x{:04X}, status={}",
            i2c_address_, reg, s));
    }

    uint16_t value = uint16_t((uint16_t(read_bytes[0]) << 8) | read_bytes[1]);

    HSB_LOG_DEBUG(
        "get_register(i2c_addr=0x{:02X}, register=0x{:04X}) = 0x{:04X}",
        i2c_address_, reg, value);
    return value;
}

void Ar0234::set_register(uint16_t reg, uint16_t value)
{
    if (skip_i2c_) {
        HSB_LOG_DEBUG("Skip i2c");
        return;
    }

    HSB_LOG_DEBUG(
        "set_register(i2c_addr=0x{:02X}, register=0x{:04X}, value=0x{:04X})",
        i2c_address_, reg, value);

    std::vector<uint8_t> write_bytes = {
        uint8_t(reg >> 8), uint8_t(reg & 0xFF),
        uint8_t(value >> 8), uint8_t(value & 0xFF)
    };
    std::vector<uint8_t> read_bytes;
    const auto s = i2c_->i2c_transaction(i2c_address_, write_bytes, read_bytes);
    if (s != HOLOLINK_MODULE_OK) {
        throw std::runtime_error(fmt::format(
            "While writing AR0234 register: i2c_addr=0x{:02X}, register=0x{:04X}, status={}",
            i2c_address_, reg, s));
    }
}

void Ar0234::set_mode(ar0234_mode::Mode mode)
{
    switch (mode) {
    case ar0234_mode::AR0234_MODE_1920X1200_60FPS: {
        const auto& fmt = ar0234_mode::AR0234_1920X1200_60FPS_FORMAT;
        width_ = fmt.width;
        height_ = fmt.height;
        pixel_format_ = fmt.pixel_format;
        break;
    }
    default:
        throw std::runtime_error(fmt::format("Unsupported AR0234 mode: {}", static_cast<int>(mode)));
    }
    mode_ = mode;
}

void Ar0234::configure_camera(ar0234_mode::Mode mode, bool fsync)
{
    set_mode(mode);
    fsync_ = fsync;

    switch (mode) {
    case ar0234_mode::AR0234_MODE_1920X1200_60FPS:
        apply_register_table(ar0234_mode::AR0234_MODE_1920X1200_60FPS_SEQUENCE);
        break;
    default:
        throw std::runtime_error(fmt::format("No register table for AR0234 mode {}", static_cast<int>(mode)));
    }
}

void Ar0234::start()
{
    if (fsync_) {
        apply_register_table(ar0234_mode::AR0234_START_FSYNC_SEQUENCE);
    } else {
        apply_register_table(ar0234_mode::AR0234_START_SEQUENCE);
    }
}

void Ar0234::stop()
{
    apply_register_table(ar0234_mode::AR0234_STOP_SEQUENCE);
}

void Ar0234::configure_converter(std::shared_ptr<hololink::csi::CsiConverter> converter)
{
    if (width_ == 0 || height_ == 0) {
        throw std::runtime_error(
            "Ar0234::configure_converter: dimensions are zero — call "
            "configure_camera() before configure_converter().");
    }
    HSB_LOG_DEBUG("configure_converter:width={},height={},bpp={}", width_, height_,
        static_cast<int>(pixel_format_));
    assert(pixel_format_ == hololink::csi::PixelFormat::RAW_10);

    uint32_t start_byte = converter->receiver_start_byte();
    uint32_t transmitted_line_bytes = converter->transmitted_line_bytes(pixel_format_, width_);
    uint32_t received_line_bytes = converter->received_line_bytes(transmitted_line_bytes);

    converter->configure(start_byte, received_line_bytes, width_, height_, pixel_format_);
}

void Ar0234::set_exposure_reg(uint16_t value)
{
    set_register(ar0234_mode::REG_EXP, value);
    std::this_thread::sleep_for(std::chrono::milliseconds(ar0234_mode::AR0234_WAIT_MS));
}

void Ar0234::test_pattern(uint16_t pattern)
{
    set_register(ar0234_mode::REG_TP, pattern);
}

template <typename ContainerT>
void Ar0234::apply_register_table(const ContainerT& table)
{
    static_assert(std::is_same_v<typename ContainerT::value_type,
                      std::pair<uint16_t, uint16_t>>,
        "Container must hold std::pair<uint16_t, uint16_t>");

    for (const auto& reg_val : table) {
        if (reg_val.first == ar0234_mode::AR0234_TABLE_WAIT_MS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(reg_val.second));
        } else {
            set_register(reg_val.first, reg_val.second);
        }
    }
}

} // namespace hololink::module::sensors::ar0234
