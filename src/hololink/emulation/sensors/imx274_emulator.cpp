/**
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
 *
 * See README.md for detailed information.
 */

#include "imx274_emulator.hpp"
#include "../hsb_emulator.hpp"
#include "../i2c_interface.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>

#define IMX274_I2C_ADDRESS 0x1A

// Mode select / streaming control. 0x00 = streaming, non-zero = standby.
// IMX274_START_SEQUENCE writes 0x00; IMX274_STOP_SEQUENCE writes 0x01;
// the mode-config sequences write 0x12 first (still standby).
#define REG_MODE_SELECT 0x3000u
#define REG_MDSEL1 0x3004u
// A/C conversion bit depth. 0x07 == 12 bits, 0x05 == 10 bits
#define REG_AD_BIT_DEPTH 0x306Bu
#define REG_SVR 0x300Cu
#define REG_VCUT_MODE 0x30E2u

namespace hololink::emulation::sensors {

namespace {
    bool is_write(size_t write_size)
    {
        return write_size > sizeof(uint16_t);
    }
} // namespace

IMX274Emulator::IMX274Emulator()
    : I2CPeripheral()
{
    reset();
}

IMX274Emulator::~IMX274Emulator() { }

void IMX274Emulator::reset()
{
    memset(memory_, 0, sizeof(memory_));
    // Default to non-streaming after reset (matches IMX274_STOP_SEQUENCE value).
    memory_[REG_MODE_SELECT] = 0x01;
}

void IMX274Emulator::attach_to_i2c(I2CController& i2c_controller, uint8_t bus_address)
{
    i2c_controller.attach_i2c_peripheral(bus_address, IMX274_I2C_ADDRESS, this);
}

I2CStatus IMX274Emulator::i2c_transaction(uint16_t peripheral_address, const uint8_t* write_bytes, uint16_t write_size, uint8_t* read_bytes, uint16_t read_size)
{
    if (write_size < 2) {
        return I2CStatus::I2C_STATUS_INVALID_REGISTER_ADDRESS;
    }

    uint16_t reg_addr = write_bytes[1] | (write_bytes[0] << 8);

    if (peripheral_address != IMX274_I2C_ADDRESS) {
        return I2CStatus::I2C_STATUS_SUCCESS; // do nothing for non-camera addresses
    }

    // for writes, copy the write bytes to the memory map
    if (is_write(write_size)) {
        if (reg_addr + write_size > IMX274_MEMORY_SIZE) {
            return I2CStatus::I2C_STATUS_WRITE_FAILED;
        }
        memcpy(&memory_[reg_addr], write_bytes + 2, write_size - 2);
    }
    // for reads, copy the memory map to the read bytes
    else {
        if (read_size > IMX274_MEMORY_SIZE - reg_addr) {
            return I2CStatus::I2C_STATUS_READ_FAILED;
        }
        memcpy(read_bytes, &memory_[reg_addr], read_size);
    }
    return I2CStatus::I2C_STATUS_SUCCESS;
}

bool IMX274Emulator::is_streaming() const
{
    return memory_[REG_MODE_SELECT] == 0x00;
}

uint16_t IMX274Emulator::get_pixel_width() const
{
    // really we should be looking at the horizontal cropping registers 3037-303B, but we don't access them yet
    uint16_t width = 3840u;
    if (memory_[REG_MDSEL1] == 0x02u) {
        width /= 2;
    } else if (memory_[REG_MDSEL1] >= 0x03u) {
        width /= 3;
    }
    return width;
}

uint16_t IMX274Emulator::get_pixel_height() const
{
    uint16_t height = 2160u;
    if (memory_[REG_VCUT_MODE]) {
        height /= memory_[REG_VCUT_MODE];
    }
    return height;
}

uint8_t IMX274Emulator::get_pixel_bits() const
{
    uint8_t pb = memory_[REG_AD_BIT_DEPTH] == 0x07u ? 12u : 10u;
    return pb;
}

uint16_t IMX274Emulator::get_bytes_per_line() const
{
    uint32_t line_width_bits = get_pixel_width() * get_pixel_bits();
    assert(line_width_bits <= 8u * UINT16_MAX - 56);
    // convert to bytes and pad to 8 bytes
    uint16_t line_width_bytes = (line_width_bits + 63) / 64 * 8;
    return line_width_bytes;
}

uint32_t IMX274Emulator::get_image_start_byte() const
{
    uint32_t bpl = get_bytes_per_line();
    if (bpl == 0) {
        return 0;
    }
    uint32_t metadata = 175u;
    uint32_t optical_black_lines = 8u;
    if (get_pixel_bits() == 12) {
        optical_black_lines *= 2;
        metadata += 35u;
    }
    metadata = (metadata + 7u) & (~0x07u); // pad to 8 bytes
    return metadata + optical_black_lines * bpl;
}

uint32_t IMX274Emulator::get_csi_length() const
{
    return get_image_start_byte() + get_pixel_height() * get_bytes_per_line();
}

} // namespace hololink::emulation::sensors
