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

#ifndef VB1940_EMULATOR_HPP
#define VB1940_EMULATOR_HPP

#include <cstdint>
#include <set>
#include <unordered_map>
#include <vector>

#include "../data_plane.hpp"
#include "../i2c_interface.hpp"

#define VB1940_MEMORY_SIZE 65536u

namespace hololink::emulation::sensors {

class Vb1940Emulator : public I2CPeripheral {
public:
    Vb1940Emulator();
    ~Vb1940Emulator();

    /**
     * @brief Reset the Vb1940Emulator internal state machines to their initial states
     */
    void reset();

    /**
     * python:
     *
     * `def attach_to_i2c(self: hemu.sensors.Vb1940Emulator, i2c_controller: hemu.I2CController, bus_address: int)`
     *
     * @brief Attach the Vb1940Emulator to an I2C controller at a specified bus
     * @param i2c_controller The I2C controller to attach to. Retrieved from HSBEmulator with get_i2c(controller_address)
     * @param bus_address The bus address of the I2C controller
     */
    void attach_to_i2c(I2CController& i2c_controller, uint8_t bus_address) override;

    /**
     * python:
     *
     * `def i2c_transaction(self: hemu.sensors.Vb1940Emulator, peripheral_address: int, write_bytes: List[int], read_bytes: List[int]) -> hemu.I2CStatus`
     *
     * @brief Receive and act on I2C transaction from HSBEmulator's I2C controller(s)
     * @param peripheral_address The address of the peripheral to transaction with
     * @param write_bytes The bytes to write to the peripheral
     * @param read_bytes The bytes to read from the peripheral
     * @return The status of the I2C transaction
     */
    I2CStatus i2c_transaction(uint8_t peripheral_address, const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes) override;

    /**
     * @brief Check if the Host application has set the sensor to streaming mode. It is expected that the host application does not send any commands that set the sensor to streaming mode if the sensor is not yet configured.
     * @return True if the Vb1940Emulator is streaming, false otherwise
     */
    bool is_streaming() const;

    /**
     * @brief Get the width of the configured image in pixels.
     * @return The width of the image in pixels or 0 if the sensor has not been configured; see is_streaming().
     */
    uint16_t get_pixel_width() const;

    /**
     * @brief Get the height of the configured image in pixels.
     * @return The height of the image in pixels or 0 if the sensor has not been configured; see is_streaming().
     */
    uint16_t get_pixel_height() const;

    /**
     * @brief Get the number of bytes per line of the sensor CSI image frame - not the packetized CSI image frame.
     * @return The number of bytes per line of the CSI image frame or 0 if the sensor has not been configured; see is_streaming().
     */
    uint16_t get_bytes_per_line() const;

    /**
     * @brief Get the start byte offset for the CSI image frame
     * @return The byte offset of the CSI image frame where the image starts or 0 if the sensor has not been configured; see is_streaming().
     */
    uint16_t get_image_start_byte() const;

    /**
     * @brief Get the length of the CSI data in bytes for the CSI image frame - not the packetized CSI image frame.
     * @return The length of the CSI data in bytes or 0 if the sensor has not been configured; see is_streaming().
     */
    uint32_t get_csi_length() const;

    /**
     * @brief Get the number of bits per pixel.
     * @return The number of bits per pixel or 0 if the sensor has not been configured; see is_streaming().
     */
    uint8_t get_pixel_bits() const;

private:
    /**
     * @brief Build the internal state machine for the Vb1940Emulator
     */
    void build_state_machine();

    /**
     * @brief state machine callback for writes to the system up register
     */
    I2CStatus system_up_reg(const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes);

    /**
     * @brief state machine callback for writes to the boot register
     */
    I2CStatus boot_reg(const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes);

    /**
     * @brief state machine callback for writes to the software standby register
     */
    I2CStatus sw_stby_reg(const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes);

    /**
     * @brief state machine callback for writes to the streaming register
     */
    I2CStatus streaming_reg(const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes);

    /**
     * @brief Check if the write bytes from an i2c transaction are a register write operation
     * @param write_bytes The bytes to check
     * @return True if the write bytes are a write operation, false otherwise
     */
    bool is_write(const std::vector<uint8_t>& write_bytes);

    /**
     * @brief The set of I2C peripheral addresses associated with a Vb1940 sensor
     */
    std::set<uint8_t> i2c_addresses_;

    /**
     * @brief The map of register addresses to their corresponding state machine callback
     */
    typedef I2CStatus (Vb1940Emulator::*i2c_callback_type)(const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes);
    std::unordered_map<uint16_t, i2c_callback_type> callback_map_;

    /**
     * @brief The memory map of the Vb1940Emulator
     */
    uint8_t memory_[VB1940_MEMORY_SIZE] = { 0 };
};

} // namespace hololink::emulation::sensors

#endif // VB1940_EMULATOR_HPP
