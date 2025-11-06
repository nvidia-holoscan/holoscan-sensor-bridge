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

#include <cassert>
#include <cstdint>
#include <set>
#include <unordered_map>

#include "../hsb_emulator.hpp"
#include "../i2c_interface.hpp"
#include "vb1940_emulator.hpp"

#define CAM_I2C_ADDRESS 0x10
#define EEPROM_I2C_ADDRESS 0x51
#define VCL_EN_I2C_ADDRESS_1 0x70
#define VCL_EN_I2C_ADDRESS_2 0x71
#define VCL_PWM_I2C_ADDRESS 0x21

// these are read only registers
// 4 byte reads
#define DEVICE_ID_REG 0x0000
#define DEVICE_REVISION_REG 0x0004
// 1 byte reads
#define SYSTEM_FSM_STATE_REG 0x0044
#define BOOT_FSM_REG 0x0200

// these are read/write registers
// 1 byte read/writes
#define SYSTEM_UP_REG 0x0514
#define BOOT_REG 0x0515
#define SW_STBY_REG 0x0516
#define STREAMING_REG 0x0517

#define VB1940_WIDTH_LO 0x91A
#define VB1940_WIDTH_HI 0x91B
#define VB1940_HEIGHT_LO 0x91C
#define VB1940_HEIGHT_HI 0x91D
#define VB1940_DATA_TYPE 0x91E
#define VB1940_MASTER_MODE 0xAC6
#define VB1940_GPIO0_FSYNC_IN 0xAD4
#define VB1940_GPIO2_STROBE 0xAD6
#define VB1940_GPIO3_STROBE 0xAD7
#define VB1940_CONTEXT_SWITCH_SEQUENCE_VECTOR_0 0xADC
#define VB1940_CONTEXT_SWITCH_SEQUENCE_VECTOR_1 0xAE0
#define VB1940_CONTEXT_SWITCH_LOOP_ELEMENT 0xAE4
#define VB1940_LINE_LENGTH_LO 0x934
#define VB1940_LINE_LENGTH_HI 0x935
#define VB1940_ORIENTATION 0x937
#define VB1940_CONFIG6_GS_OP_RGB_PWLOFF_SINGLE_EXP_VTSS1_DESCSS1_CFA_BAYER_RAW10_31 0xB88
#define VB1940_FRAME_LENGTH 0xB8E
#define VB1940_FRAME_LENGTH_HI 0xB8F
#define VB1940_VC0 0xB8C

#define VB1940_BAYER_RAW10 0x2B
#define VB1940_BAYER_RAW8 0x2A

namespace hololink::emulation::sensors {

enum SystemFsmState : uint8_t {
    SYSTEM_HW_STBY = 0x0,
    SYSTEM_UP = 0x1,
    BOOT = 0x2,
    SW_STBY = 0x3,
    STREAMING = 0x4,
    STALL = 0x5,
    HALT = 0x6
};
enum BootFsmState : uint8_t {
    BOOT_HW_STBY = 0x00,
    COLD_BOOT = 0x01,
    CLOCK_INIT = 0x02,
    NVM_DWLD = 0x10,
    NVM_UNPACK = 0x11,
    SYSTEM_BOOT = 0x12,
    NONCE_GRNERATION = 0x20,
    EPH_KEYS_GENERATION = 0x21,
    WAIT_CERTIFICATE = 0x22,
    CERTIFICATE_PARSING = 0x23,
    CERTIFICATE_VERIF_ROOT = 0x24,
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
    PAIRING = 0x55,
    FWP_SETUP = 0x60,
    VTP_SETUP = 0x61,
    FA_RETURN = 0x70,
    BOOT_WAITING_CMD = 0x80,
    BOOT_COMPLETED = 0xBC
};

Vb1940Emulator::Vb1940Emulator()
    : I2CPeripheral()
{
    i2c_addresses_ = {
        CAM_I2C_ADDRESS,
        EEPROM_I2C_ADDRESS,
        VCL_EN_I2C_ADDRESS_1,
        VCL_EN_I2C_ADDRESS_2,
        VCL_PWM_I2C_ADDRESS
    };
    reset();
    build_state_machine();
    // set device id
    memory_[DEVICE_ID_REG + 0] = 0x30;
    memory_[DEVICE_ID_REG + 1] = 0x34;
    memory_[DEVICE_ID_REG + 2] = 0x39;
    memory_[DEVICE_ID_REG + 3] = 0x53;
}

Vb1940Emulator::~Vb1940Emulator() { }

void Vb1940Emulator::reset()
{
    memory_[SYSTEM_FSM_STATE_REG] = SystemFsmState::SYSTEM_UP;
    memory_[BOOT_FSM_REG] = BootFsmState::COLD_BOOT;
    memory_[SW_STBY_REG] = 0;
    memory_[STREAMING_REG] = 0;
    memory_[SYSTEM_UP_REG] = 0;
    memory_[BOOT_REG] = 0;
}

void Vb1940Emulator::attach_to_i2c(I2CController& i2c_controller, uint8_t bus_address)
{
    for (const auto& address : i2c_addresses_) {
        i2c_controller.attach_i2c_peripheral(bus_address, address, this);
    }
}

void Vb1940Emulator::build_state_machine()
{
    callback_map_[SYSTEM_UP_REG] = &Vb1940Emulator::system_up_reg;
    callback_map_[BOOT_REG] = &Vb1940Emulator::boot_reg;
    callback_map_[SW_STBY_REG] = &Vb1940Emulator::sw_stby_reg;
    callback_map_[STREAMING_REG] = &Vb1940Emulator::streaming_reg;
}

I2CStatus Vb1940Emulator::i2c_transaction(uint8_t peripheral_address, const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes)
{

    if (write_bytes.size() < 2) {
        return I2CStatus::I2C_STATUS_INVALID_REGISTER_ADDRESS;
    }

    uint16_t reg_addr = write_bytes[1] | (write_bytes[0] << 8);

    if (peripheral_address != CAM_I2C_ADDRESS) {
        return I2CStatus::I2C_STATUS_SUCCESS; // do nothing for non-camera addresses
    }

    auto it = callback_map_.find(reg_addr);
    if (it != callback_map_.end()) {
        I2CStatus status = (this->*it->second)(write_bytes, read_bytes);
        return status;
    }

    // for writes, copy the write bytes to the memory map
    if (is_write(write_bytes)) {
        if (reg_addr + write_bytes.size() - 2 > VB1940_MEMORY_SIZE) {
            return I2CStatus::I2C_STATUS_WRITE_FAILED;
        }
        std::copy(write_bytes.begin() + 2, write_bytes.end(), &memory_[reg_addr]);
    }
    // for reads, copy the memory map to the read bytes
    else {
        if (read_bytes.size() > VB1940_MEMORY_SIZE - reg_addr) {
            return I2CStatus::I2C_STATUS_READ_FAILED;
        }
        std::copy(&memory_[reg_addr], &memory_[reg_addr + read_bytes.size()], read_bytes.begin());
    }
    return I2CStatus::I2C_STATUS_SUCCESS;
}

I2CStatus Vb1940Emulator::system_up_reg(const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes)
{
    if (is_write(write_bytes)) {
        if (write_bytes[2] == 0x01) {
            reset();
            memory_[SYSTEM_FSM_STATE_REG] = SystemFsmState::BOOT;
            // we will not actually check the certificate here, but a true emulator might want to
            memory_[BOOT_FSM_REG] = BootFsmState::WAIT_CERTIFICATE;
        }
    }
    return I2CStatus::I2C_STATUS_SUCCESS;
}

I2CStatus Vb1940Emulator::boot_reg(const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes)
{
    if (is_write(write_bytes)) {
        switch (write_bytes[2]) {
        case 0x01:
            memory_[BOOT_FSM_REG] = BootFsmState::BOOT_WAITING_CMD;
            // drivers will write fw_updates after receiving this
            break;
        case 0x02:
            memory_[BOOT_FSM_REG] = BootFsmState::FWP_SETUP;
            break;
        case 0x10:
            memory_[BOOT_FSM_REG] = BootFsmState::BOOT_COMPLETED;
            // uddf driver applies vt_patch after secure_boot is completed
            // eventually, the system must be put in SW_STBY in order to start()
            memory_[SYSTEM_FSM_STATE_REG] = SystemFsmState::SW_STBY;
            break;
        }
    }
    return I2CStatus::I2C_STATUS_SUCCESS;
}

I2CStatus Vb1940Emulator::sw_stby_reg(const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes)
{
    if (is_write(write_bytes)) {
        if (write_bytes[2] == 0x01) {
            memory_[SYSTEM_FSM_STATE_REG] = SystemFsmState::STREAMING;
        }
    } else if (read_bytes.size() > 0) {
        read_bytes[0] = memory_[SW_STBY_REG];
    }
    return I2CStatus::I2C_STATUS_SUCCESS;
}

I2CStatus Vb1940Emulator::streaming_reg(const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes)
{
    if (is_write(write_bytes)) {
        if (write_bytes[2] == 0x01) {
            memory_[SYSTEM_FSM_STATE_REG] = SystemFsmState::SW_STBY;
        }
    } else if (read_bytes.size() > 0) {
        read_bytes[0] = memory_[STREAMING_REG];
    }
    return I2CStatus::I2C_STATUS_SUCCESS;
}

bool Vb1940Emulator::is_write(const std::vector<uint8_t>& write_bytes)
{
    return write_bytes.size() > sizeof(uint16_t);
}

bool Vb1940Emulator::is_streaming() const
{
    return memory_[SYSTEM_FSM_STATE_REG] == SystemFsmState::STREAMING;
}

// returns 0 if unset or the number of bits per pixel
uint16_t Vb1940Emulator::get_pixel_width() const
{
    return memory_[VB1940_WIDTH_LO] | (memory_[VB1940_WIDTH_HI] << 8);
}

// returns 0 if unset or the number of bits per pixel
uint16_t Vb1940Emulator::get_pixel_height() const
{
    return memory_[VB1940_HEIGHT_LO] | (memory_[VB1940_HEIGHT_HI] << 8);
}

// returns 0 if unset or the number of bits per pixel
uint8_t Vb1940Emulator::get_pixel_bits() const
{
    if (memory_[VB1940_DATA_TYPE] == VB1940_BAYER_RAW10) {
        return 10;
    }
    if (memory_[VB1940_DATA_TYPE] == VB1940_BAYER_RAW8) {
        return 8;
    }
    return 0;
}

uint16_t Vb1940Emulator::get_bytes_per_line() const
{
    uint32_t line_width_bits = get_pixel_width() * get_pixel_bits();
    assert(line_width_bits <= 8u * UINT16_MAX - 56);
    // convert to bytes and pad to 8 bytes
    uint16_t line_width_bytes = (line_width_bits + 63) / 64 * 8;

    return line_width_bytes;
}

uint16_t Vb1940Emulator::get_image_start_byte() const
{
    return get_bytes_per_line();
}

uint32_t Vb1940Emulator::get_csi_length() const
{
    uint16_t line_width_bytes = get_bytes_per_line();
    // vb1940 csi is line_width in bytes * (3 + pixel_height) with 1 line leading and 2 lines trailing
    uint32_t trailing_bytes = line_width_bytes * 2;
    uint32_t leading_bytes = get_image_start_byte();
    return leading_bytes + get_pixel_height() * line_width_bytes + trailing_bytes;
}

} // namespace hololink::emulation::sensors
