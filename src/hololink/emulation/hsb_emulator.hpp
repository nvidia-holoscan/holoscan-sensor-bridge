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

#ifndef EMULATION_HSB_EMULATOR_H
#define EMULATION_HSB_EMULATOR_H

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "address_memory.hpp"
#include "hsb_config.hpp"

namespace hololink::emulation {

enum ECB_CMD_CODE {
    WR_BYTE = 0x01,
    WR_WORD = 0x02,
    WR_DWORD = 0x04,
    WR_BLOCK = 0x09,
    RMW_BYTE = 0x0A,
    RD_BYTE = 0x11,
    RD_WORD = 0x12,
    RD_DWORD = 0x14,
    RD_BLOCK = 0x19,
    GET_INFO = 0x20,
};

enum ECB_RESPONSE_CODE {
    ECB_SUCCESS = 0x00,
    ECB_ADDRESS_ERROR = 0x03,
    ECB_COMMAND_ERROR = 0x04,
    ECB_FLAG_ERROR = 0x06,
    ECB_SEQUENCE_ERROR = 0x0B
};

template <typename T>
using UniqueDel = std::unique_ptr<T, std::function<void(T*)>>;

// forward declarations
class DataPlane;
class I2CPeripheral;
struct I2CControllerCtxt;
struct ControlMessage;
class HSBEmulator;
struct HSBEmulatorCtxt;

/**
 * @brief callback function type for control plane callback
 * @param ctxt The context to pass to the callback.
 * @param addr_val The address and value to pass to the callback. may be a C style array
 * @param count The number of address and value pairs passed to the callback.
 * @return > 0 for the number of address and successfully processed value pairs. <= 0 for an error. Note 0 is a valid number of consumed pairs but will be treated as an error.
 */
using ControlPlaneCallback_f = std::function<int(void* ctxt, struct AddressValuePair* addr_val, int count)>;

/**
 * @brief class that manages I2C transaction events from the host
 * runs a separate thread for execution but only one event may ever be executed at a time for each controller
 * @note This should not be instantiated directly by user code. Use HSBEmulator::get_i2c() instead. The default and primary I2CController is at address hololink::I2C_CTRL.
 */
class I2CController {
public:
    friend class HSBEmulator;

    ~I2CController();

    /**
     * python:
     *
     * `def attach_i2c_peripheral(self: hemu.I2CController, bus_address: int, peripheral_address: int, peripheral: hemu.I2CPeripheral)`
     *
     * @brief attach an I2C peripheral as a callback on the specified (bus address, peripheral address) pair.
     *
     * @param bus_address The bus address of the peripheral. For multiplexing or bus expanded addressing, this is the bus address of the peripheral.
     * @param peripheral_address The peripheral address of the peripheral.
     * @param peripheral The peripheral to attach. This is a pointer to the peripheral object. The caller is responsible for managing the lifetime of the peripheral object.
     */
    void attach_i2c_peripheral(uint32_t bus_address, uint16_t peripheral_address, I2CPeripheral* peripheral);

private:
    /**
     * @brief construct an I2C controller on HSB Emulator at the specified address.
     *
     * @param hsb_emulator The HSB emulator to attach to.
     * @param controller_address The address of the I2C controller.
     */
    explicit I2CController(HSBEmulator& hsb_emulator, uint32_t controller_address);

    /**
     * @brief start the I2C controller.
     */
    void start();

    /**
     * @brief execute an I2C transaction.
     *
     * @param value The command register (bits 0-15) and peripheral address register (bits 16-22).
     * @note This method is called by the HSBEmulator to schedule an I2C transaction. private as it should only be called by the HSBEmulator.
     */
    void execute(uint32_t value);

    /**
     * @brief stop the I2C controller.
     */
    void stop();

    /**
     * @brief check if the I2C controller is running.
     * @return true if the I2C controller is running, false otherwise.
     */
    bool is_running();

    /**
     * @brief internal method that performs the actual i2c_transaction() call to the target peripheral
     */
    void i2c_execute();

    /**
     * @brief run the I2C controller thread.
     */
    void run();

    UniqueDel<struct I2CControllerCtxt> ctxt_ = { nullptr };
};

/**
 * @brief The `HSBEmulator` class represents the interface that a host application has to an HSB and acts as the emulation device's controller.
 * It manages the `DataPlane` objects and the `I2CController` and all communication with the internal memory model; see `AddressMemory` for more details.
 */
class HSBEmulator {
public:
    friend class DataPlane;
    friend class I2CController;

    /**
     *
     * python:
     *
     * `hemu.HSBEmulator(config: hemu.HSBConfiguration = hemu.HSB_EMULATOR_CONFIG)`
     *
     * @brief Construct a new HSBEmulator object with the specified configuration.
     * @param config The configuration of the emulator. Two fully populated options
     * are provided in hsb_config.hpp: HSB_EMULATOR_CONFIG or HSB_LEOPARD_EAGLE_CONFIG.
     * @note HSB_EMULATOR_CONFIG is roughly equivalent to a Lattice board.
     * @note HSB_LEOPARD_EAGLE_CONFIG is roughly equivalent to a Leopard Eagle board.
     */
    HSBEmulator(const HSBConfiguration& config);

    /**
     * @brief Construct a new HSBEmulator object. Defaults to HSB_EMULATOR_CONFIG, which is roughly equivalent to a Lattice board.
     */
    HSBEmulator();
    ~HSBEmulator();

    /**
     * @brief Start the emulator. This will start the BootP broadcasting via the DataPlane objects as well as the control thread to listen for control messages from the host.
     *
     * @note It is safe to call this function multiple times and after a subsequent call to stop()
     */
    void start();

    /**
     * @brief Stop the emulator. This will shut down the control message thread, all BootP broadcasts in DataPlane objects. Data transmission is still possible until HSBEmulator object is destroyed
     *
     * @note It is safe to call this function multiple times
     */
    void stop();

    /**
     * @brief Check if the emulator is running.
     * @return true if the emulator is running, false otherwise.
     */
    bool is_running();

    /**
     * python:
     *
     * `def write(self: hemu.HSBEmulator, address: int, value: int)`
     *
     * @brief Write a value to a register.
     * @param address The address of the register to write.
     * @param value The value to write.
     * @return 0 on success, 1 on address error.
     */
    int write(uint32_t address, uint32_t value);

    /**
     * python:
     *
     * `def read(self: hemu.HSBEmulator, address: int) -> int`
     *
     * @brief Read a value from a register.
     * @param address The address of the register to read.
     * @param value The value read will be stored here.
     * @return 0 on success, 1 on address error.
     */
    int read(uint32_t address, uint32_t& value);

    /**
     * @brief Set a read callback for a range of addresses. See ControlPlaneCallback_f for more details.
     * @param start_address The start address of the range.
     * @param end_address The end address of the range (exclusive).
     * @param callback The callback to set.
     * @param ctxt The context to pass to the callback.
     * @return 0 on success, else error code.
     * @note This method is not yet implemented for linux targets
     */
    int register_read_callback(uint32_t start_address, uint32_t end_address, ControlPlaneCallback_f callback, void* ctxt = nullptr);

    /**
     * @brief Register a write callback for a range of addresses. See ControlPlaneCallback_f for more details.
     * @param start_address The start address of the range.
     * @param end_address The end address of the range (exclusive).
     * @param callback The callback to set.
     * @param ctxt The context to pass to the callback.
     * @return 0 on success, else error code.
     * @note This method is not yet implemented for linux targets
     */
    int register_write_callback(uint32_t start_address, uint32_t end_address, ControlPlaneCallback_f callback, void* ctxt = nullptr);

    /**
     * @brief method to explicitly handle pending control messages to the HSBEmulator client. This method is required for MCU targets, but optional for linux targets
     */
    int handle_msgs();

    /**
     * python:
     *
     * `def get_i2c(self: hemu.HSBEmulator, controller_address: int = hololink.I2C_CTRL) -> hemu.I2CController`
     *
     * @brief Get a reference to the I2C controller.
     * @return A reference to the I2C controller.
     *
     */
    I2CController& get_i2c(uint32_t controller_address = hololink::I2C_CTRL);

    const HSBConfiguration& get_config() { return configuration_; }

private:
    // utilities for use by friend DataPlane
    /**
     * @brief Add a data plane to the emulator for start()/stop()/is_running() management
     * @param data_plane The data plane to add.
     * @return 0 on success, 1 on failure.
     */
    int add_data_plane(DataPlane& data_plane);

    /**
     * @return The register map of the emulated fpga
     */
    UniqueDel<AddressMemory>& get_memory() { return registers_; }

    UniqueDel<AddressMemory> registers_ = { nullptr }; // multiple devices may share this object using references
    UniqueDel<HSBEmulatorCtxt> ctxt_ = { nullptr };

    HSBConfiguration configuration_;
    I2CController i2c_controller_;
};

}

#endif