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

#include <stdint.h>

// This is a workaround to allow backward compatibility with the existing examples which require namespace to hololink.
#ifdef __cplusplus
namespace hololink {
// I2C interfaces
constexpr uint32_t BL_I2C_BUS = 0;
constexpr uint32_t CAM_I2C_BUS = 1;
// I2C status flags
constexpr uint32_t I2C_CTRL = 0x03000200;
}
#else
// I2C interfaces
#define BL_I2C_BUS 0u
#define CAM_I2C_BUS 1u
// I2C address
#define I2C_CTRL 0x03000200u
#endif

#define REGISTER_SIZE 4u

// control plane constants
#define CONTROL_UDP_PORT 8192u
#define CONTROL_INTERVAL_MSEC 1000u
#define MIN_VALID_CONTROL_LENGTH 10u

#define SOFT_RESET_REG_CTRL 0x08u
#define RESET_REG_CTRL 0x04u

// SPI interfaces
#define SPI_CTRL 0x03000000u
// SPI address offsets
#define SPI_REG_CONTROL 0x00u
#define SPI_REG_BUS_EN 0x04u
#define SPI_REG_NUM_BYTES 0x08u
#define SPI_REG_SPI_MODE 0x0Cu
#define SPI_REG_NUM_CMD_BYTES 0x10u
#define SPI_REG_STATUS 0x80u
#define SPI_REG_DATA_BUFFER 0x100u
// SPI control flags
#define SPI_START 0x01u
// SPI status flags
#define SPI_IDLE 0x00u
#define SPI_BUSY 0x01u
#define SPI_FSM_ERR 0x02u
#define SPI_SPI_ERR 0x04u
#define SPI_DONE 0x10u
// SPI_CFG
#define SPI_CFG_CPOL 0x10u
#define SPI_CFG_CPHA 0x20u

// I2C interfaces
// see above for I2C_CTRL
// I2C address offsets
#define I2C_REG_CONTROL 0x00u
#define I2C_REG_BUS_EN 0x04u
#define I2C_REG_NUM_BYTES 0x08u
#define I2C_REG_CLK_CNT 0x0Cu
#define I2C_REG_STATUS 0x80u
#define I2C_REG_DATA_BUFFER 0x100u
// I2C control flags
#define I2C_START 0x01u
#define I2C_10B_ADDRESS 0x02u
// i2c status flags
#define I2C_IDLE 0x00u
#define I2C_BUSY 0x01u
#define I2C_FSM_ERR 0x02u
#define I2C_I2C_ERR 0x04u
#define I2C_I2C_NAK 0x08u
#define I2C_DONE 0x10u

#define HSB_IP_VERSION 0x80u
#define FPGA_DATE 0x84u
#define FPGA_PTP_CTRL 0x104u
#define FPGA_PTP_DELAY_ASYMMETRY 0x10Cu
#define FPGA_PTP_CTRL_DPLL_CFG1 0x110u
#define FPGA_PTP_CTRL_DPLL_CFG2 0x114u
#define FPGA_PTP_CTRL_DELAY_AVG_FACTOR 0x118u
#define FPGA_PTP_SYNC_TS_0 0x180u
#define FPGA_PTP_SYNC_STAT 0x188u
#define FPGA_PTP_OFM 0x18Cu

// Async event messaging
#define CTRL_EVENT 0x00000200u
#define CTRL_EVT_RISING (CTRL_EVENT + 0x04u)
#define CTRL_EVT_FALLING (CTRL_EVENT + 0x08u)
#define CTRL_EVT_CLEAR (CTRL_EVENT + 0x0Cu)
#define CTRL_EVT_HOST_MAC_ADDR_LO (CTRL_EVENT + 0x10u)
#define CTRL_EVT_HOST_MAC_ADDR_HI (CTRL_EVENT + 0x14u)
#define CTRL_EVT_HOST_IP_ADDR (CTRL_EVENT + 0x18u)
#define CTRL_EVT_HOST_UDP_PORT (CTRL_EVENT + 0x1Cu)
#define CTRL_EVT_FPGA_UDP_PORT (CTRL_EVENT + 0x20u)
#define CTRL_EVT_APB_INTERRUPT_EN (CTRL_EVENT + 0x24u)
#define CTRL_EVT_APB_TIMEOUT (CTRL_EVENT + 0x28u)
#define CTRL_EVT_SW_EVENT (CTRL_EVENT + 0x2Cu)
#define CTRL_EVT_STAT (CTRL_EVENT + 0x80u)

// APB event constants
#define APB_RAM 0x2000u
#define APB_SIF_0_FRAME_END_START_ADDRESS (APB_RAM + 0x100u)
#define APB_SIF_1_FRAME_END_START_ADDRESS (APB_RAM + 0x200u)
#define APB_SW_EVENT_START_ADDRESS (APB_RAM + 0x800u)
#define APB_RAM_DATA_SIZE 0x1000u

typedef enum {
    EVENT_I2C_BUSY = 0,
    EVENT_SW_EVENT = 2,
    EVENT_SIF_0_FRAME_END = 16,
    EVENT_SIF_1_FRAME_END = 17,
    EVENT_GPIO0 = 18,
    EVENT_GPIO1 = 19,
    EVENT_SIF_0_FRAME_START = 20,
    EVENT_SIF_1_FRAME_START = 21,
} Event;

// request packet flag bits
#define REQUEST_FLAGS_ACK_REQUEST 0x01u
#define REQUEST_FLAGS_SEQUENCE_CHECK 0x02u

// HSBConfiguration constants. Note currently in hololink::core, but might be useful/have to change if we have a unified HSBConfiguration class across HSB types and emulator.
// size of the uuid member in HSBConfiguration. must >= UUID_SIZE
#define BOARD_VERSION_SIZE 20u
#define UUID_STR_LEN 37u
#define BOARD_SERIAL_NUM_SIZE 7u
#define VENDOR_ID_SIZE 4u

// HSBConfiguration constants
#define MAX_SENSORS 32u // 5 bits of indices
#ifndef MAX_DATA_PLANES
#define MAX_DATA_PLANES 256u // 8 bits of indices
#endif
#define MAX_SIFS_PER_SENSOR 32u // 5 bits of indices
#define MAX_SIFS 32u // 5 bits of indices
#define HSB_DEFAULT_SENSOR_COUNT 1u
#define HSB_DEFAULT_DATA_PLANE_COUNT 1u
#define HSB_DEFAULT_SIFS_PER_SENSOR 1u

// HSBEmulator specific constants
#define HSB_EMULATOR_DATE 20260219u
#define HSB_EMULATOR_TAG 0xE0u
#define HSB_EMULATOR_TAG_LENGTH 0x04u
#define HSB_EMULATOR_VENDOR_ID \
    {                          \
        'N', 'V', 'D', 'A'     \
    }
#define HSB_EMULATOR_VENDOR_ID_SIZE 4u
#define HSB_EMULATOR_ENUM_VERSION 2u
#define HSB_EMULATOR_BOARD_ID 2u
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
#define HSB_EMULATOR_HSB_IP_VERSION 0x2602u
#define HSB_EMULATOR_FPGA_CRC 0x5AA5u
#define HSB_DEFAULT_TIMEOUT_MSEC 100u

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

#define SENTINEL_ADDRESS 0xFFFFFFFFu
#define SENTINEL_VALUE 0xFFFFFFFFu

struct AddressValuePair {
    uint32_t address;
    uint32_t value;
};

// Declarations must be in hololink::emulation to match the definitions in hsb_config.cpp.
#ifdef __cplusplus
namespace hololink::emulation {

extern const struct HSBConfiguration HSB_EMULATOR_CONFIG;

extern const struct HSBConfiguration HSB_LEOPARD_EAGLE_CONFIG;

int hsb_config_set_uuid(struct HSBConfiguration* config, const char* uuid_str);
const char* validate_configuration(const struct HSBConfiguration* config);

}
#endif

#endif // HSB_CONFIG_HPP