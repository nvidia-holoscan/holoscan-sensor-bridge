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

#ifndef HOLOLINK_MODULE_PCIE_PCIE_REGS_HPP
#define HOLOLINK_MODULE_PCIE_PCIE_REGS_HPP

#include <cstdint>

/**
 * @file
 * @brief BAR0 register map of the Hololink PCIe endpoint (map version 0x2606).
 *
 * Ported from hololink_pcie_bringup/lib/hololink_pcie/regs.py, which is generated from the
 * FPGA register CSVs by scripts/generate_registers.py. Only the blocks hsb_pcie touches are
 * carried across; regenerate from the CSVs rather than hand-editing when the map moves.
 *
 * Do not confuse this with hololink/emulation/hsb_config.hpp, which is the *host-facing* HSB
 * address map carried in ECB control messages. The two are unrelated address spaces; the
 * hsb_pcie module's AddressPolicy translates between them.
 *
 * Installed with the hololink_module headers for the hsb_pcie module and the module-native
 * PCIe receiver; the namespace is hololink::pcie.
 */

namespace hololink::pcie {

inline constexpr uint32_t REG_MAP_VERSION = 0x2606u;

// The driver only exposes BAR0, and only aligned 32-bit accesses within it.
inline constexpr uint32_t BAR0 = 0u;
// Named REG_SIZE, not REGISTER_SIZE: hololink/emulation/hsb_config.hpp defines the latter
// as an object-like macro, which would rewrite any constexpr of that name here.
inline constexpr uint32_t REG_SIZE = 4u;

// ---------------------------------------------------------------------------
// design_global -- identity and GPIO
// ---------------------------------------------------------------------------
inline constexpr uint32_t DESIGN_GLOBAL_BASE = 0x00000000u;
inline constexpr uint32_t DESIGN_GLOBAL_END = 0x000000ffu;

inline constexpr uint32_t GLB_SCRATCH = DESIGN_GLOBAL_BASE + 0x00u;
inline constexpr uint32_t HLNK_CTRL_0 = DESIGN_GLOBAL_BASE + 0x04u;
inline constexpr uint32_t HLNK_CTRL_1 = DESIGN_GLOBAL_BASE + 0x08u;
inline constexpr uint32_t HSB_VER = DESIGN_GLOBAL_BASE + 0x80u;
inline constexpr uint32_t HSB_DATE = DESIGN_GLOBAL_BASE + 0x84u;
inline constexpr uint32_t HSB_STAT = DESIGN_GLOBAL_BASE + 0x88u;
// HSB_STAT doubles as the identity magic register on current bring-up bitfiles.
inline constexpr uint32_t HSB_MAGIC = HSB_STAT;
inline constexpr uint32_t HSB_MAGIC_EXPECT = 0x484c5043u; // ASCII "HLPC"

// 128-bit board UUID. Per hololink_pcie_bringup/examples/igx_imx258_hdmi/README.md this is
// programmed once per board; identity.py only ever reads it back.
inline constexpr uint32_t HSB_UUID_0 = DESIGN_GLOBAL_BASE + 0xacu;
inline constexpr uint32_t HSB_UUID_1 = DESIGN_GLOBAL_BASE + 0xb0u;
inline constexpr uint32_t HSB_UUID_2 = DESIGN_GLOBAL_BASE + 0xb4u;
inline constexpr uint32_t HSB_UUID_3 = DESIGN_GLOBAL_BASE + 0xb8u;

inline constexpr uint32_t HLNK_CTRL_0_SYS_RST_MASK = 0x00000008u;
// HLNK_CTRL_1 SW_SEN_RST[1:0]: one bit per sensor, active low, reset value 0. On the
// CertusPro-NX Versa bit 0 is the IMX258's hardware reset on J37, and the camera NAKs on
// I2C until it is released. The Ethernet SDK's trigger_reset() leaves them at 0x3;
// PcieHololink::start() releases them too.
inline constexpr uint32_t HLNK_CTRL_1_SW_SEN_RST_MASK = 0x00000003u;

// ---------------------------------------------------------------------------
// pcie_dp -- per-sensor data plane destination configuration
// ---------------------------------------------------------------------------
inline constexpr uint32_t PCIE_DP_BASE = 0x00001000u;
inline constexpr uint32_t PCIE_DP_STRIDE = 0x00000040u;

inline constexpr uint32_t PCIE_DP_BUF_BASE_HI = 0x00u; // fixed upper 32 bits of the IOVA
inline constexpr uint32_t PCIE_DP_REQUESTER_ID = 0x04u;
inline constexpr uint32_t PCIE_DP_BUF_INC = 0x08u; // slot stride, in 128-byte pages
inline constexpr uint32_t PCIE_DP_BUF_START = 0x0cu; // lower 32 bits of the IOVA
inline constexpr uint32_t PCIE_DP_BUF_LEN = 0x10u;
inline constexpr uint32_t PCIE_DP_METADATA_CTRL = 0x20u;
inline constexpr uint32_t PCIE_DP_BUF_PTR_RANGE = 0x28u;
inline constexpr uint32_t PCIE_DP_PSN = 0x3cu;

inline constexpr uint32_t PCIE_DP_MD_DISABLE_MASK = 0x00010000u;
inline constexpr uint32_t PCIE_DP_MSIX_DISABLE_MASK = 0x00020000u;

// ---------------------------------------------------------------------------
// MSI-X table and pending-bit array
// ---------------------------------------------------------------------------
inline constexpr uint32_t MSIX_TABLE_BASE = 0x00008000u;
inline constexpr uint32_t MSIX_TABLE_END = 0x000083ffu;
inline constexpr uint32_t MSIX_TABLE_STRIDE = 0x00000010u;
inline constexpr uint32_t MSIX_TABLE_ENTRIES = 64u;

inline constexpr uint32_t MSIX_MESSAGE_ADDR_LO = 0x00u;
inline constexpr uint32_t MSIX_MESSAGE_ADDR_HI = 0x04u;
inline constexpr uint32_t MSIX_MESSAGE_DATA = 0x08u;
inline constexpr uint32_t MSIX_VECTOR_CONTROL = 0x0cu;
inline constexpr uint32_t MSIX_VECTOR_MASK = 0x00000001u;

inline constexpr uint32_t MSIX_PBA_BASE = 0x00009000u;
inline constexpr uint32_t MSIX_PBA_LO = 0x00u;
inline constexpr uint32_t MSIX_PBA_HI = 0x04u;

// ---------------------------------------------------------------------------
// Sensor RX packetizer
// ---------------------------------------------------------------------------
inline constexpr uint32_t PACK_CTRL_BASE = 0x00010000u;
inline constexpr uint32_t PACK_CTRL_STRIDE = 0x00000200u;
inline constexpr uint32_t PACK_SIF_TVLD_STAT = 0x00000080u;
inline constexpr uint32_t PACK_PSN_STAT = 0x00000084u;

// ---------------------------------------------------------------------------
// dp_pkt -- data-plane packetizer arm/disarm
// ---------------------------------------------------------------------------
inline constexpr uint32_t DP_PKT_BASE = 0x00030300u;
inline constexpr uint32_t DP_PKT_SCRATCH = 0x00u;
inline constexpr uint32_t DP_PKT_LEN = 0x04u;
inline constexpr uint32_t DP_PKT_VIP_MASK = 0x0cu; // one bit per sensor; 0 disarms everything
inline constexpr uint32_t DP_PKT_CRC_XOR = 0x10u;
inline constexpr uint32_t DP_PKT_STAT = 0x80u;

// ---------------------------------------------------------------------------
// SPI and I2C controllers
//
// Both blocks use byte-identical register offsets and status bits to their HSB counterparts
// in hololink/emulation/hsb_config.hpp (CONTROL 0x00, BUS_EN 0x04, NUM_BYTES 0x08,
// CLK_CNT 0x0c, STATUS 0x80, DATA_BUFFER 0x100; BUSY/FSM_ERR/ERR/NACK/DONE =
// 0x01/0x02/0x04/0x08/0x10), so forwarding is a pure base-address rebase. The PCIe block
// adds TIMEOUT and NUM_TRANSACTIONS, which have no HSB equivalent.
// ---------------------------------------------------------------------------
inline constexpr uint32_t SPI_CTRL_BASE = 0x00040000u;
inline constexpr uint32_t SPI_CTRL_END = 0x000401ffu;

inline constexpr uint32_t I2C_CTRL_BASE = 0x00040200u;
inline constexpr uint32_t I2C_CTRL_END = 0x000403ffu;

inline constexpr uint32_t I2C_CONTROL = 0x000u;
inline constexpr uint32_t I2C_BUS_EN = 0x004u;
inline constexpr uint32_t I2C_NUM_BYTES = 0x008u;
inline constexpr uint32_t I2C_CLK_CNT = 0x00cu;
inline constexpr uint32_t I2C_TIMEOUT = 0x010u; // no HSB equivalent
inline constexpr uint32_t I2C_NUM_TRANSACTIONS = 0x014u; // no HSB equivalent
inline constexpr uint32_t I2C_STATUS = 0x080u;
inline constexpr uint32_t I2C_DATA_BUFFER = 0x100u;

inline constexpr uint32_t I2C_CONTROL_START_MASK = 0x00000001u;
inline constexpr uint32_t I2C_STATUS_BUSY_MASK = 0x00000001u;
inline constexpr uint32_t I2C_STATUS_FSM_ERR_MASK = 0x00000002u;
inline constexpr uint32_t I2C_STATUS_I2C_ERR_MASK = 0x00000004u;
inline constexpr uint32_t I2C_STATUS_I2C_NACK_MASK = 0x00000008u;
inline constexpr uint32_t I2C_STATUS_DONE_MASK = 0x00000010u;

// i2c.sv: SCL is approximately apb_clk / (2 * clk_cnt). The CertusPro-NX Versa drives its
// I2C block from a 60 MHz APB clock.
inline constexpr uint32_t DEFAULT_APB_CLK_HZ = 60000000u;
inline constexpr uint32_t DEFAULT_SCL_HZ = 400000u;
inline constexpr uint32_t DEFAULT_SCL_TIMEOUT = 5000000u;

// ---------------------------------------------------------------------------
// BAR0 extent. usr_module ends the map, so its end + 1 is the window size to map.
// ---------------------------------------------------------------------------
inline constexpr uint32_t USR_MODULE_BASE = 0x00080000u;
inline constexpr uint32_t USR_MODULE_END = 0x000fffffu;
inline constexpr uint32_t BAR0_SIZE = USR_MODULE_END + 1u;

/// Byte offset of MSI-X table entry `vector`.
inline constexpr uint32_t msix_table_entry(uint32_t vector)
{
    return MSIX_TABLE_BASE + vector * MSIX_TABLE_STRIDE;
}

/// Byte offset of word `word` of sensor `sensor`'s data-plane configuration block.
inline constexpr uint32_t pcie_dp_reg(uint32_t sensor, uint32_t reg)
{
    return PCIE_DP_BASE + sensor * PCIE_DP_STRIDE + reg;
}

/// clk_cnt divider producing approximately `scl_hz`.
inline constexpr uint32_t clk_cnt_for(uint32_t scl_hz, uint32_t apb_clk_hz = DEFAULT_APB_CLK_HZ)
{
    return scl_hz == 0 ? 1u : (apb_clk_hz + scl_hz) / (2u * scl_hz);
}

} // namespace hololink::pcie

#endif /* HOLOLINK_MODULE_PCIE_PCIE_REGS_HPP */
