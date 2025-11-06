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

#ifndef EMULATION_I2C_INTERFACE_HPP
#define EMULATION_I2C_INTERFACE_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <set>

#include "hsb_emulator.hpp"

namespace hololink::emulation {

// only FATAL_ERROR will stop program execution. All others are warnings for logging purposes
enum I2CStatus {
    I2C_STATUS_SUCCESS = 0,
    I2C_STATUS_BUSY,
    I2C_STATUS_WRITE_FAILED,
    I2C_STATUS_READ_FAILED,
    I2C_STATUS_MESSAGE_NOT_UNDERSTOOD,
    I2C_STATUS_TIMEOUT,
    I2C_STATUS_NACK,
    I2C_STATUS_BUFFER_SIZE_SMALL, // peripheral expects more data than provided or larger buffer for return
    I2C_STATUS_BUFFER_SIZE_LARGE, // peripheral expected less data than provided or smaller buffer for return
    I2C_STATUS_INVALID_PERIPHERAL_ADDRESS,
    I2C_STATUS_INVALID_REGISTER_ADDRESS,
    I2C_STATUS_FATAL_ERROR,
};

/**
 * @brief Abstract base class for all I2C-based emulated sensors giving them access to I2C transactions originating from the HSB host
 * @note configuration of the "bus" is done by HSBEmulator calling get_i2c_config(). Derived class can assume this only happens at most once before any calls to start() or reset() without an intervening call to stop()
 */
class I2CPeripheral {
public:
    I2CPeripheral() { }
    virtual ~I2CPeripheral() = default;

    /**
     * @brief start the peripheral. This will be called when I2CController itself starts and should not be done by the client code
     */
    virtual void start() { }

    /**
     * python:
     *
     * `def attach_to_i2c(self: hemu.I2CPeripheral, i2c_controller: hemu.I2CController, bus_address: int)`
     *
     * @brief attach the peripheral to the I2C controller. This should be called by client code before start() is called
     *
     * @param i2c_controller The I2C controller to attach to. Retrieved from HSBEmulator with get_i2c(controller_address)
     * @param bus_address The bus address of the peripheral.
     */
    virtual void attach_to_i2c(I2CController& i2c_controller, uint8_t bus_address) = 0;

    /**
     * @brief stop the peripheral. This will be called when the I2CController is stopped and should not be done by the client code
     *
     * @note Peripheral should release all resources in case a subsequent start() is called.
     */
    virtual void stop() { }

    /**
     * python:
     *
     * `def i2c_transaction(self: hemu.I2CPeripheral, peripheral_address: int, write_bytes: List[int], read_bytes: List[int]) -> hemu.I2CStatus`
     *
     * @brief perform an I2C transaction. This will be called by the I2CController when a transaction is requested.
     *
     * @param peripheral_address The peripheral address to communicate with.
     * @param write_bytes The bytes to write to the peripheral.
     * @param read_bytes The bytes to read from the peripheral. This shall be filled with 0s to the requested read size and replaced by peripheral. This is to ensure peripheral can get the read count from read_bytes.size() and enough valid data is returned for no-ops.
     * @return I2CStatus The status of the transaction.
     *
     * @note read request size is the size of the read_bytes vector as initialized by the caller
     */
    virtual I2CStatus i2c_transaction(uint8_t peripheral_address, const std::vector<uint8_t>& write_bytes, std::vector<uint8_t>& read_bytes) = 0;
};

}

#endif