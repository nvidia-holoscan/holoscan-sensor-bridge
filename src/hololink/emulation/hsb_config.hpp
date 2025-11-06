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

#include "hololink/core/hololink.hpp"

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
/*
#define I2C_STATUS (hololink::I2C_CTRL + hololink::I2C_REG_STATUS)
#define I2C_BUS_END (hololink::I2C_CTRL + hololink::I2C_REG_BUS_EN)
#define I2C_NUM_BYTES (hololink::I2C_CTRL + hololink::I2C_REG_NUM_BYTES)
#define I2C_CLK_CNT (hololink::I2C_CTRL + hololink::I2C_REG_CLK_CNT)
#define I2C_DATA_BUFFER (hololink::I2C_CTRL + hololink::I2C_REG_DATA_BUFFER)
*/

// HSBConfiguration constants. Note currently in hololink::core, but might be useful/have to change if we have a unified HSBConfiguration class across HSB types and emulator.
// size of the uuid member in HSBConfiguration. must >= UUID_SIZE
#define BOARD_VERSION_SIZE 20
#define UUID_STR_LEN 37
#define BOARD_SERIAL_NUM_SIZE 7
#define VENDOR_ID_SIZE 4

// HSBConfiguration constants
#define MAX_SENSORS 32 // 5 bits of indices
#define MAX_DATA_PLANES 256 // 8 bits of indices
#define MAX_SIFS_PER_SENSOR 32 // 5 bits of indices
#define MAX_SIFS 32 // 5 bits of indices
#define HSB_DEFAULT_SENSOR_COUNT 2
#define HSB_DEFAULT_DATA_PLANE_COUNT 2
#define HSB_DEFAULT_SIFS_PER_SENSOR 2

// HSBEmulator specific constants
#define HSB_EMULATOR_DATE 20250608
#define HSB_EMULATOR_TAG 0xE0
#define HSB_EMULATOR_TAG_LENGTH 0x04
#define HSB_EMULATOR_VENDOR_ID \
    {                          \
        'N', 'V', 'D', 'A'     \
    }
#define HSB_EMULATOR_ENUM_VERSION 2
#define HSB_EMULATOR_BOARD_ID hololink::HOLOLINK_LITE_BOARD_ID
#define HSB_EMULATOR_UUID                      \
    {                                          \
        0x16, 0xaf, 0x13, 0x9b,                \
            0x9d, 0x14,                        \
            0x48, 0x2f,                        \
            0x98, 0x05,                        \
            0x3f, 0xeb, 0x72, 0xae, 0x0c, 0xec \
    }
#define HSB_EMULATOR_SERIAL_NUM \
    {                           \
        3, 1, 4, 1, 5, 9, 3     \
    }
#define HSB_EMULATOR_HSB_IP_VERSION 0x2508
#define HSB_EMULATOR_FPGA_CRC 0x5AA5

// names should be stable, but ordering and field sizes may change.
struct HSBConfiguration {
    // BootP configuration parameters
    uint8_t tag; // 0xE0
    uint8_t tag_length; // 0x04
    uint8_t vendor_id[VENDOR_ID_SIZE]; // "NVDA"
    uint8_t data_plane; // Eth Port, set to 0. DataPlane fills this in
    uint8_t enum_version;
    uint8_t board_id_lo; // 0x00
    uint8_t board_id_hi; // 0x00
    uint8_t uuid[BOARD_VERSION_SIZE];
    uint8_t serial_num[BOARD_SERIAL_NUM_SIZE];
    uint16_t hsb_ip_version;
    uint16_t fpga_crc;
    // configuration parameters not part of BootP
    uint8_t sensor_count; // must be able to mask vp in 32 bits so sensor_count < 32; 5 bits
    uint8_t data_plane_count; // must be < 256 or 1 byte otherwise hif_address runs into I2C address space
    uint8_t sifs_per_sensor; // sifs_per_sensor * sensor_count must be <= 32 otherwise vp_mask doesn't fit into 32 bits.
                             // must be <= 64 otherwise vp_address runs into APB_RAM address space
};

extern const HSBConfiguration HSB_EMULATOR_CONFIG;

extern const HSBConfiguration HSB_LEOPARD_EAGLE_CONFIG;

int hsb_config_set_uuid(HSBConfiguration& config, const char* uuid_str);

} // namespace hololink::emulation

#endif // HSB_CONFIG_HPP