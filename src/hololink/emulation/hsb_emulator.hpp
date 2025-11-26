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
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hsb_config.hpp"
#include "mem_register.hpp"
#include "net.hpp"

namespace hololink::emulation {

// forward declarations
struct ControlMessage;
class DataPlane;
class APBEventHandler;
class I2CPeripheral;
class RenesasI2CPeripheral;
class HSBEmulator;

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
    void attach_i2c_peripheral(uint32_t bus_address, uint8_t peripheral_address, I2CPeripheral* peripheral);

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

    std::mutex i2c_mutex_;
    std::condition_variable i2c_cv_;
    // outer index is bus address, inner index is peripheral address. A map of a map so that buses and peripherals get null/default initialized as they are accessed.
    std::unordered_map<uint32_t, std::unordered_map<uint8_t, I2CPeripheral*>> i2c_bus_map_;
    std::atomic<bool> running_ { false };
    uint16_t peripheral_address_ { 0 };
    uint16_t cmd_ { 0 };
    std::thread i2c_thread_;
    std::shared_ptr<MemRegister> registers_;
    uint32_t controller_address_;
    uint32_t status_address_;
    uint32_t bus_en_address_;
    uint32_t num_bytes_address_;
    uint32_t clk_cnt_address_;
    uint32_t data_buffer_address_;
};

/**
 * @brief The `HSBEmulator` class represents the interface that a host application has to an HSB and acts as the emulation device's controller.
 * It manages the `DataPlane` objects and the `I2CController` and all communication with the internal memory model; see `MemRegister` for more details.
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
     */
    void write(uint32_t address, uint32_t value);

    /**
     * python:
     *
     * `def read(self: hemu.HSBEmulator, address: int) -> int`
     *
     * @brief Read a value from a register.
     * @param address The address of the register to read.
     * @return The value read.
     */
    uint32_t read(uint32_t address);

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

private:
    // utilities for use by friend DataPlane
    /**
     * @brief Add a data plane to the emulator for start()/stop()/is_running() management
     * @param data_plane The data plane to add.
     */
    void add_data_plane(DataPlane& data_plane);

    /**
     * @return A constant reference to the configuration of the emulator.
     *      const reference because the DataPlane should not change this copy of the configuration
     */
    const HSBConfiguration& get_config() const { return configuration_; }

    /**
     * @return The register map of the emulated fpga
     */
    std::shared_ptr<MemRegister> get_memory() const { return registers_; }

    /**
     * @brief method for separate thread to listen for control messages from the host.
     */
    void control_listen();

    /**
     * @brief method to handle control messages targeting the SPI bus.
     * @param address The address of the register to read or write.
     * @param value Pointer to the value that is to be written or nullptr
     * @return if value is nullptr, returns the value read at the address otherwise writes value to address and returns 0 on success, >0 on failure.
     */
    void handle_spi_control_write(uint32_t address, uint32_t value);

    /**
     * @brief method to detect HSB Host application polling conditions
     * @return true if the polling condition is met, false otherwise.
     *
     * @note This method is used as a workaround to handle HSB Host applications listening for APB sequence I2C transactions to finish
     */
    bool detect_poll(uint32_t address);

    /**
     * @brief method to handle HSB Host application polling conditions when detected.
     *
     * @note This method is used as a workaround to handle HSB Host applications listening for APB sequence I2C transactions to finish
     */
    void handle_poll();

    /**
     * @brief method to handle read control messages from the HSB Host application.
     * @param message The control message to handle.
     * @param control_socket The socket to use to send the reply message to the HSB Host application.
     * @param host_addr The address of the HSB Host application.
     * @param host_addr_len The length of the HSB Host application address.
     */
    void handle_read_message(ControlMessage& message, int control_socket, struct sockaddr_in* host_addr, socklen_t host_addr_len);

    /**
     * @brief method to handle write control messages from the HSB Host application.
     * @param message The control message to handle.
     * @param control_socket The socket to use to send the reply message to the HSB Host application.
     * @param host_addr The address of the HSB Host application.
     * @param host_addr_len The length of the HSB Host application address.
     */
    void handle_write_message(ControlMessage& message, int control_socket, struct sockaddr_in* host_addr, socklen_t host_addr_len);

    /**
     * @brief method to handle reply messages to the HSB Host application.
     * @param message The control message to handle.
     * @param message_buffer The buffer to use to send the reply message to the HSB Host application.
     * @param message_length The length of the reply message.
     * @param control_socket The socket to use to send the reply message to the HSB Host application.
     * @param host_addr The address of the HSB Host application.
     * @param host_addr_len The length of the HSB Host application address.
     */
    void handle_reply_message(ControlMessage& message, uint8_t* message_buffer, size_t message_length, int control_socket, struct sockaddr_in* host_addr, socklen_t host_addr_len);

    HSBConfiguration configuration_;
    std::shared_ptr<MemRegister> registers_; // multiple devices may share this object
    std::unique_ptr<APBEventHandler> apb_event_handler_; // owned by HSBEmulator and never shared
    std::unique_ptr<I2CController> i2c_controller_; // owned by HSBEmulator and never shared
    std::unique_ptr<RenesasI2CPeripheral> renesas_i2c_; // owned by HSBEmulator and never shared
    std::thread control_thread_;

    std::vector<DataPlane*> data_plane_list_;
    /* for workaround to handle polling conditions */
    uint32_t last_read_address_ { 0 };
    unsigned short poll_count_ { 0 };
    /* end workaround */
    std::atomic<bool> running_ { false };
};

}

#endif