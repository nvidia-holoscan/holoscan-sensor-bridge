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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "hsb_config.hpp"
#include "mem_register.hpp"

// HSBEmulator specific constants
#define HSB_EMULATOR_DATE 20250608
#define HSB_EMULATOR_TAG 0xE0
#define HSB_EMULATOR_TAG_LENGTH 0x04
#define HSB_EMULATOR_VENDOR_ID \
    {                          \
        'N', 'V', 'D', 'A'     \
    }
#define HSB_EMULATOR_DEFAULT_ENUM_VERSION 2
#define HSB_EMULATOR_BOARD_ID hololink::HOLOLINK_LITE_BOARD_ID
#define HSB_EMULATOR_UUID                                                                              \
    {                                                                                                  \
        0x88, 0x9b, 0x7c, 0xe3, 0x65, 0xa5, 0x42, 0x47, 0x8b, 0x05, 0x4f, 0xf1, 0x90, 0x4c, 0x33, 0x59 \
    }
#define HSB_EMULATOR_SERIAL_NUM \
    {                           \
        3, 1, 4, 1, 5, 9, 3     \
    }
#define HSB_EMULATOR_HSB_IP_VERSION 0x2506
#define HSB_EMULATOR_FPGA_CRC 0x5AA5

namespace hololink::emulation {

// forward declarations
struct ControlMessage;
class DataPlane;

/**
 * @brief HSBEmulator is primary running of the emulated HSB. The interface is intentionally simple/minimal to provide ONLY basic functionality. Dynamic interaction on the emulator side is not supported and requires a lot more work on making it thread-safe.
 */
class HSBEmulator {
public:
    HSBEmulator();
    ~HSBEmulator();
    friend class DataPlane;

    /**
     * @brief Start the emulator. This will start the BootP broadcasting via the DataPlane objects as well as the control thread to listen for control messages from the host.
     *
     * @note It is safe to call this function multiple times and after a subsequent call to stop() (though not thoroughly tested)
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
     * @return The register map of the emulatoed fpga
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
     * @brief method to handle control messages targeting the I2C bus.
     * @param address The address of the register to read or write.
     * @param value Pointer to the value that is to be written or nullptr
     * @return if value is nullptr, returns the value read at the address otherwise writes value to address and returns 0 on success, >0 on failure.
     */
    void handle_i2c_control_write(uint32_t address, uint32_t value);

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
     * @brief method to handle invalid control messages from the HSB Host application.
     * @param message The control message to handle.
     * @param control_socket The socket to use to send the reply message to the HSB Host application.
     * @param host_addr The address of the HSB Host application.
     * @param host_addr_len The length of the HSB Host application address.
     */
    void handle_invalid_message(ControlMessage& message, int control_socket, struct sockaddr_in* host_addr, socklen_t host_addr_len);

    HSBConfiguration configuration_ {
        .tag = HSB_EMULATOR_TAG,
        .tag_length = HSB_EMULATOR_TAG_LENGTH,
        .vendor_id = HSB_EMULATOR_VENDOR_ID,
        .enum_version = HSB_EMULATOR_DEFAULT_ENUM_VERSION,
        .board_id = HSB_EMULATOR_BOARD_ID,
        .uuid = HSB_EMULATOR_UUID,
        .serial_num = HSB_EMULATOR_SERIAL_NUM,
        .hsb_ip_version = HSB_EMULATOR_HSB_IP_VERSION,
        .fpga_crc = HSB_EMULATOR_FPGA_CRC,
    };
    std::shared_ptr<MemRegister> registers_;
    std::thread control_thread_;

    std::vector<DataPlane*> data_plane_list_;
    /* for workaround to handle polling conditions */
    uint32_t last_read_address_ { 0 };
    unsigned char poll_count_ { 0 };
    /* end workaround */
    std::atomic<bool> running_ { false };
};

}

#endif