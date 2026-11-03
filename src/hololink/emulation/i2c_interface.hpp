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

#include <cstdint>

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
 * @brief Abstract base class for all I2C-based emulated sensors giving them access to I2C transactions originating from the HSB host.
 * @note configuration of the "bus" is done by HSBEmulator calling get_i2c_config(). Derived class can assume this only happens at most once before any calls to start() or reset() without an intervening call to stop()
 */
class I2CPeripheral {
public:
    I2CPeripheral() { }
    virtual ~I2CPeripheral() = default;

    /**
     * @brief start the peripheral. This will be called when I2CController itself starts and should not be done by the client code.
     */
    virtual void start() { }

    /**
     * python:
     *
     * `def attach_to_i2c(self: hemu.I2CPeripheral, i2c_controller: hemu.I2CController, bus_address: int)`
     *
     * @brief attach the peripheral to the I2C controller. This should be called by client code before start() is called.
     *
     * @param i2c_controller The I2C controller to attach to. Retrieved from HSBEmulator with get_i2c(controller_address).
     * @param bus_address The bus address of the peripheral.
     */
    virtual void attach_to_i2c(I2CController& i2c_controller, uint8_t bus_address) = 0;

    /**
     * @brief stop the peripheral. This will be called when the I2CController is stopped and should not be done by the client code.
     *
     * @note Peripheral should release all resources in case a subsequent start() is called.
     */
    virtual void stop() { }

    /**
     * python:
     *
     * `def i2c_transaction(self: hemu.I2CPeripheral, peripheral_address: int, write_bytes: bytes, write_size: int, read_bytes: bytearray, read_size: int) -> hemu.I2CStatus`
     *
     * @brief perform an I2C transaction. This will be called by the I2CController when a transaction is requested.
     *
     * @param peripheral_address The peripheral address to communicate with.
     * @param write_bytes The bytes to write to the peripheral.
     * @param write_size The size of the write bytes.
     * @param read_bytes Buffer the peripheral fills with the bytes read. The caller provides @p read_size bytes
     *        of storage; the peripheral overwrites them with its reply.
     * @param read_size The number of bytes requested by the host.
     * @return I2CStatus The status of the transaction.
     *
     * @note @p write_bytes and @p read_bytes may alias (the I2C controller hands the same data buffer for both),
     *       so consume the write bytes before producing the read bytes.
     *
     * @note Python override: `write_bytes` arrives as an immutable `bytes` copy of length `write_size`;
     *       `read_bytes` is a `bytearray` of length `read_size` to fill, e.g.
     *       `read_bytes[0:2] = bytes([0x12, 0x34])`; it is copied into the controller's buffer when the
     *       override returns, so it must keep its length. An exception from the override fails the
     *       transaction (I2C_STATUS_FATAL_ERROR) and is reported through sys.unraisablehook.
     *       Calling `i2c_transaction` *on* a peripheral from Python takes
     *       `(peripheral_address, write_bytes, read_bytes=None, write_size=None, read_size=None)` where
     *       `write_bytes` is any byte buffer and `read_bytes` a writable byte buffer (e.g. `bytearray(n)`).
     */
    virtual I2CStatus i2c_transaction(uint16_t peripheral_address, const uint8_t* write_bytes, uint16_t write_size, uint8_t* read_bytes, uint16_t read_size) = 0;
};

}

#endif