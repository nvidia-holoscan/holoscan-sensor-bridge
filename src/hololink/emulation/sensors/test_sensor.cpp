/**
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

#include "test_sensor.hpp"

#include <cstring>

namespace hololink::emulation::sensors {

namespace {

    constexpr uint32_t DEFAULT_FRAME_SIZE = 4096u;
    constexpr uint32_t DEFAULT_FRAME_RATE_HZ = 30u;

    bool is_write(uint16_t write_size)
    {
        return write_size > sizeof(uint16_t);
    }

    uint32_t read_u32_le(const uint8_t* memory, uint16_t reg_addr)
    {
        return static_cast<uint32_t>(memory[reg_addr]) | (static_cast<uint32_t>(memory[reg_addr + 1]) << 8)
            | (static_cast<uint32_t>(memory[reg_addr + 2]) << 16) | (static_cast<uint32_t>(memory[reg_addr + 3]) << 24);
    }

    void write_u32_le(uint8_t* memory, uint16_t reg_addr, uint32_t value)
    {
        memory[reg_addr] = static_cast<uint8_t>(value);
        memory[reg_addr + 1] = static_cast<uint8_t>(value >> 8);
        memory[reg_addr + 2] = static_cast<uint8_t>(value >> 16);
        memory[reg_addr + 3] = static_cast<uint8_t>(value >> 24);
    }

} // namespace

TestSensor::TestSensor()
    : I2CPeripheral()
{
    reset();
}

TestSensor::~TestSensor() { }

void TestSensor::reset()
{
    memset(memory_, 0, sizeof(memory_));
    memory_[DEVICE_ID_REG + 0] = 'H';
    memory_[DEVICE_ID_REG + 1] = 'L';
    memory_[DEVICE_ID_REG + 2] = 'T';
    memory_[DEVICE_ID_REG + 3] = 'K';
    write_u32_le(memory_, FRAME_SIZE_REG, DEFAULT_FRAME_SIZE);
    memory_[PATTERN_MODE_REG] = static_cast<uint8_t>(TestPatternMode::INCREMENTING);
    memory_[CONSTANT_BYTE_REG] = 0xA5;
    write_u32_le(memory_, FRAME_RATE_REG, DEFAULT_FRAME_RATE_HZ);
    memory_[STREAMING_REG] = 0;
}

void TestSensor::attach_to_i2c(I2CController& i2c_controller, uint8_t bus_address)
{
    i2c_controller.attach_i2c_peripheral(bus_address, TEST_I2C_ADDRESS, this);
}

I2CStatus TestSensor::i2c_transaction(uint16_t peripheral_address, const uint8_t* write_bytes, uint16_t write_size, uint8_t* read_bytes, uint16_t read_size)
{
    if (write_size < 2) {
        return I2CStatus::I2C_STATUS_INVALID_REGISTER_ADDRESS;
    }

    if (peripheral_address != TEST_I2C_ADDRESS) {
        return I2CStatus::I2C_STATUS_INVALID_PERIPHERAL_ADDRESS;
    }

    uint16_t reg_addr = write_bytes[1] | (write_bytes[0] << 8);

    if (reg_addr == STREAMING_REG) {
        return streaming_reg(write_bytes, write_size, read_bytes, read_size);
    }

    if (is_write(write_size)) {
        uint16_t data_size = write_size - 2;
        if (reg_addr + data_size > TEST_SENSOR_MEMORY_SIZE) {
            return I2CStatus::I2C_STATUS_WRITE_FAILED;
        }
        if (reg_addr == FRAME_SIZE_REG) {
            if (data_size != sizeof(uint32_t)) {
                return I2CStatus::I2C_STATUS_WRITE_FAILED;
            }
            const uint32_t frame_size = static_cast<uint32_t>(write_bytes[2])
                | (static_cast<uint32_t>(write_bytes[3]) << 8)
                | (static_cast<uint32_t>(write_bytes[4]) << 16)
                | (static_cast<uint32_t>(write_bytes[5]) << 24);
            if (frame_size == 0) {
                return I2CStatus::I2C_STATUS_WRITE_FAILED;
            }
        }
        memcpy(&memory_[reg_addr], write_bytes + 2, data_size);
    } else {
        if (reg_addr + read_size > TEST_SENSOR_MEMORY_SIZE) {
            return I2CStatus::I2C_STATUS_READ_FAILED;
        }
        memcpy(read_bytes, &memory_[reg_addr], read_size);
    }
    return I2CStatus::I2C_STATUS_SUCCESS;
}

I2CStatus TestSensor::streaming_reg(const uint8_t* write_bytes, uint16_t write_size, uint8_t* read_bytes, uint16_t read_size)
{
    if (is_write(write_size)) {
        memory_[STREAMING_REG] = write_bytes[2] ? 1 : 0;
        memory_[STATUS_REG] = memory_[STREAMING_REG];
    } else if (read_size > 0) {
        if (read_size > TEST_SENSOR_MEMORY_SIZE - STREAMING_REG) {
            return I2CStatus::I2C_STATUS_READ_FAILED;
        }
        memcpy(read_bytes, &memory_[STREAMING_REG], read_size);
    }
    return I2CStatus::I2C_STATUS_SUCCESS;
}

bool TestSensor::is_streaming() const
{
    return memory_[STREAMING_REG] != 0;
}

uint32_t TestSensor::get_frame_size() const
{
    return read_u32_le(memory_, FRAME_SIZE_REG);
}

uint32_t TestSensor::get_frame_rate_hz() const
{
    uint32_t hz = read_u32_le(memory_, FRAME_RATE_REG);
    return hz > 0 ? hz : DEFAULT_FRAME_RATE_HZ;
}

TestPatternMode TestSensor::get_pattern_mode() const
{
    return memory_[PATTERN_MODE_REG] == static_cast<uint8_t>(TestPatternMode::CONSTANT) ? TestPatternMode::CONSTANT : TestPatternMode::INCREMENTING;
}

uint8_t TestSensor::get_constant_byte() const
{
    return memory_[CONSTANT_BYTE_REG];
}

void TestSensor::fill_frame(uint8_t* data, uint64_t frame_index) const
{
    const uint32_t size = get_frame_size();
    if (get_pattern_mode() == TestPatternMode::CONSTANT) {
        memset(data, get_constant_byte(), size);
        return;
    }
    for (uint32_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>((frame_index + i) & 0xFF);
    }
}

} // namespace hololink::emulation::sensors
