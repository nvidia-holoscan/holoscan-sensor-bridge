// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//==============================================================================
// Package Include File for Hololink Simple Testbench
// Contains parameter definitions and interface signal widths
//==============================================================================

`ifndef FPGA_SENSOR_TOP_PKG_SVH
`define FPGA_SENSOR_TOP_PKG_SVH

//==============================================================================
// System Parameters - Based on HOLOLINK_def.svh and sensor configurations
//==============================================================================

// Sensor Interface Parameters
`ifndef DATAPATH_WIDTH
  `define DATAPATH_WIDTH  512                 // Sensor interface data width
`endif
  `define DATAKEEP_WIDTH  (`DATAPATH_WIDTH/8) // Sensor interface data keep width
  `define DATAUSER_WIDTH  1                   // Sensor interface data user width
`ifndef SENSOR_IF_INST
  `define SENSOR_IF_INST  2                   // Sensor interface instantiation number
`endif 

// Host Interface Parameters
`ifndef HOST_WIDTH
  `define HOST_WIDTH      512                 // Host interface data width
`endif
  `define HOSTKEEP_WIDTH  (`HOST_WIDTH/8)     // Host interface data keep width
  `define HOSTUSER_WIDTH  1                   // Host interface data user width
`ifndef HOST_IF_INST
  `define HOST_IF_INST    2                   // Host interface instantiation number
`endif 

// Peripheral Control Parameters
`ifndef SPI_INST
  `define SPI_INST  2   // SPI interface instantiation number
`endif
`ifndef  I2C_INST
  `define I2C_INST  2   // I2C interface instantiation number
`endif
  `define GPIO_OUT  64  // MISC GPIO OUT
  `define GPIO_IN   128 // MISC GPIO IN
  `define GPIO_INST 64  // Total GPIO instances

// Register Interface Parameters
  `define REG_INST 8

// User Event Width
`define HLNK_USER_EVT_WIDTH 16

// Clock Frequencies
`define CLK_FREQ  156250000

// ECB Command Codes
`define ECB_CMD_WR_DWORD      8'h04
`define ECB_CMD_WR_DWORD_RESP 8'h84
`define ECB_CMD_RD_DWORD      8'h14
`define ECB_CMD_RD_DWORD_RESP 8'h94

// Ethernet UDP Port for ECB
`define ETH_UDP_CMD_PORT      16'h4321

// Default MAC and IP addresses for testing
`define DEFAULT_HOST_MAC      48'h123456789ABC
`define DEFAULT_FPGA_MAC      48'hCAFEC0FFEE00
`define DEFAULT_HOST_IP       32'hC0A80101  // 192.168.1.1
`define DEFAULT_FPGA_IP       32'hC0A80164  // 192.168.1.100

//==============================================================================
// Common Register Address Definitions (Examples)
//==============================================================================

// System Control Registers
`define REG_SYS_CTRL          32'h00000000
`define REG_SYS_STATUS        32'h00000004
`define REG_SYS_VERSION       32'h00000008

// Data Generator Registers  
`define REG_DATA_GEN_ENA      32'h00020000
`define REG_DATA_GEN_CTRL     32'h00020004

// RoCE Configuration Registers (Base addresses)
`define REG_ROCE_BASE         32'h00010000
`define REG_ROCE_DEST_QP      32'h00010000
`define REG_ROCE_RKEY         32'h00010004
`define REG_ROCE_BUF_MASK     32'h00010008
`define REG_ROCE_HOST_MAC_LO  32'h00010010
`define REG_ROCE_HOST_MAC_HI  32'h00010014
`define REG_ROCE_HOST_IP      32'h00010018
`define REG_ROCE_HOST_PORT    32'h0001001C

`endif // FPGA_SENSOR_TOP_PKG_SVH
