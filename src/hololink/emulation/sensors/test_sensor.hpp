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

#ifndef HOLOLINK_EMULATION_TEST_SENSOR_HPP
#define HOLOLINK_EMULATION_TEST_SENSOR_HPP

#include <cstdint>

#include "../i2c_interface.hpp"

#define TEST_SENSOR_MEMORY_SIZE 256u

namespace hololink::emulation::sensors {

/** I2C peripheral address for the test sensor. */
constexpr uint8_t TEST_I2C_ADDRESS = 0x5A;

/**
 * @brief Pattern mode for generated frame payload.
 */
enum class TestPatternMode : uint8_t {
    CONSTANT = 0,
    INCREMENTING = 1,
};

/**
 * @brief I2C-controlled test sensor for validation and development.
 *
 * Register map (16-bit big-endian address, same transaction format as Vb1940Emulator):
 *
 * | Address | Access | Description |
 * |---------|--------|-------------|
 * | 0x0000  | R      | Device ID bytes: 'H','L','T','K' |
 * | 0x0010  | R/W    | Frame data size in bytes (32-bit, little-endian) |
 * | 0x0014  | R/W    | Pattern mode: 0=constant, 1=incrementing |
 * | 0x0015  | R/W    | Constant pattern byte (used when pattern mode is constant) |
 * | 0x0020  | R/W    | Frame rate in Hz (32-bit, little-endian) |
 * | 0x0030  | R/W    | Streaming control: write 1=start, 0=stop; read returns state |
 * | 0x0031  | R      | Status: bit0 = streaming active |
 */
class TestSensor : public I2CPeripheral {
public:
    TestSensor();
    ~TestSensor();

    void reset();

    void attach_to_i2c(I2CController& i2c_controller, uint8_t bus_address) override;

    I2CStatus i2c_transaction(uint16_t peripheral_address, const uint8_t* write_bytes, uint16_t write_size, uint8_t* read_bytes, uint16_t read_size) override;

    bool is_streaming() const;
    uint32_t get_frame_size() const;
    uint32_t get_frame_rate_hz() const;
    TestPatternMode get_pattern_mode() const;
    uint8_t get_constant_byte() const;

    /**
     * @brief Fill a frame buffer according to the current pattern configuration.
     * @param data Destination buffer (must be at least get_frame_size() bytes).
     * @param frame_index Frame sequence number used by the incrementing pattern.
     */
    void fill_frame(uint8_t* data, uint64_t frame_index) const;

private:
    static constexpr uint16_t DEVICE_ID_REG = 0x0000u;
    static constexpr uint16_t FRAME_SIZE_REG = 0x0010u;
    static constexpr uint16_t PATTERN_MODE_REG = 0x0014u;
    static constexpr uint16_t CONSTANT_BYTE_REG = 0x0015u;
    static constexpr uint16_t FRAME_RATE_REG = 0x0020u;
    static constexpr uint16_t STREAMING_REG = 0x0030u;
    static constexpr uint16_t STATUS_REG = 0x0031u;

    I2CStatus streaming_reg(const uint8_t* write_bytes, uint16_t write_size, uint8_t* read_bytes, uint16_t read_size);

    uint8_t memory_[TEST_SENSOR_MEMORY_SIZE] = { 0 };
};

} // namespace hololink::emulation::sensors

#endif // HOLOLINK_EMULATION_TEST_SENSOR_HPP
