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

#ifndef HSB_CONFIG_HPP
#define HSB_CONFIG_HPP

#include <cstdint>
#include <map>

#include "hololink/core/data_channel.hpp"
#include "hololink/core/hololink.hpp"
#include "hololink/core/serializer.hpp"

namespace hololink::emulation {

// control plane constants
#define CONTROL_UDP_PORT 8192
#define CONTROL_INTERVAL_MSEC 1000

// SPI interfaces
// SPI control flags
#define SPI_START 0b0000'0000'0000'0001
#define SPI_STATUS (hololink::SPI_CTRL + 0x80)
// SPI status flags
#define SPI_IDLE 0
#define SPI_BUSY 0b0000'0000'0000'0001
#define SPI_FSM_ERR 0b0000'0000'0000'0010
#define SPI_DONE 0b0000'0000'0001'0000
// SPI_CFG
#define SPI_CFG_CPOL 0b0000'0000'0001'0000
#define SPI_CFG_CPHA 0b0000'0000'0010'0000

// I2C status flags
#define I2C_IDLE 0
#define I2C_STATUS (hololink::I2C_CTRL + 0x80)

// HSBConfiguration constants. Note currently in hololink::core, but might be useful/have to change if we have a unified HSBConfiguration class across HSB types and emulator.
// size of the uuid member in HSBConfiguration. must >= UUID_SIZE
#define BOARD_VERSION_SIZE 20
// size of the UUID itself within the BOARD_VERSION_SIZE allocation
#define UUID_SIZE 16
#define BOARD_SERIAL_NUM_SIZE 7
#define VENDOR_ID_SIZE 4

/**
 * @brief DataPlaneID is used to identify the data plane on the HSB.
 * @note Currently, only 2 data planes are supported. Controlled in DataPlane() constructor.
 */
enum DataPlaneID : uint8_t {
    DATA_PLANE_0 = 0,
    DATA_PLANE_1 = 1,
};

struct HSBConfiguration {
    uint8_t tag; // 0xE0
    uint8_t tag_length; // 0x04
    uint8_t vendor_id[VENDOR_ID_SIZE]; // "NVDA"
    DataPlaneID data_plane; // Eth Port
    uint8_t enum_version;
    uint16_t board_id; // 2: HSB 10G, 3: HSB 100G // unused in enum_version = 2
    uint8_t uuid[BOARD_VERSION_SIZE];
    uint8_t serial_num[BOARD_SERIAL_NUM_SIZE];
    uint16_t hsb_ip_version;
    uint16_t fpga_crc;
};

// returns 0 on failure or the number of bytes written on success
// Note that on failure, serializer and buffer contents are in indeterminate state.
static inline size_t serialize_hsb_configuration(hololink::core::Serializer& serializer, HSBConfiguration& configuration)
{
    size_t start = serializer.length();
    return serializer.append_uint8(configuration.tag)
            && serializer.append_uint8(configuration.tag_length)
            && serializer.append_buffer(configuration.vendor_id, sizeof(configuration.vendor_id))
            && serializer.append_uint8(configuration.data_plane)
            && serializer.append_uint8(configuration.enum_version)
            && serializer.append_uint16_le(configuration.board_id)
            && serializer.append_buffer(configuration.uuid, sizeof(configuration.uuid))
            && serializer.append_buffer(configuration.serial_num, sizeof(configuration.serial_num))
            && serializer.append_uint16_le(configuration.hsb_ip_version)
            && serializer.append_uint16_le(configuration.fpga_crc)
        ? serializer.length() - start
        : 0;
}

/**
 * @brief SensorID is used to identify the sensor configuration on the HSB.
 * @note Currently, only 2 sensors are supported. Controlled in Sensor() constructor.
 */
enum SensorID {
    SENSOR_0 = 0,
    SENSOR_1 = 1,
};

// straight copied from hololink/core/data_channel.cpp (before the changes for eagle.
// Cannot make those changes here without updating uuid of HSBEmulator and registering
// it in holoscan::core)
struct DataPlaneConfiguration {
    uint32_t hif_address;
};
extern std::map<DataPlaneID, DataPlaneConfiguration> data_plane_map;
struct SensorConfiguration {
    uint32_t sensor_interface;
    uint32_t vp_mask;
    hololink::Hololink::Event frame_end_event;
    uint32_t sif_address;
    uint32_t vp_address;
};
extern std::map<SensorID, SensorConfiguration> sensor_map;
extern std::map<uint32_t, uint32_t> address_map;

} // namespace hololink::emulation

#endif // HSB_CONFIG_HPP