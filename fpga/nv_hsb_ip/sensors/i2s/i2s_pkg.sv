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

`ifndef i2s_pkg
`define i2s_pkg

package i2s_pkg;

//------------------------------------------------------------------------------
// I2S Configuration Constants
//------------------------------------------------------------------------------

  // Bit Depth Selections
  localparam I2S_BD_8B   = 2'b00;   // 8 bits
  localparam I2S_BD_16B  = 2'b01;   // 16 bits
  localparam I2S_BD_24B  = 2'b10;   // 24 bits
  localparam I2S_BD_32B  = 2'b11;   // 32 bits

  // Data Format Selections
  localparam I2S_FMT_I2S = 2'b00;   // Standard I2S
  localparam I2S_FMT_LJ  = 2'b01;   // Left Justified
  localparam I2S_FMT_RJ  = 2'b10;   // Right Justified
  localparam I2S_FMT_PCM = 2'b11;   // PCM/DSP Mode

  // Channel Mode
  localparam I2S_CH_STEREO = 1'b0;  // Stereo mode
  localparam I2S_CH_MONO   = 1'b1;  // Mono mode

  // Clock Source Selection
  localparam I2S_CLK_INT = 1'b0;    // Internal PLL
  localparam I2S_CLK_EXT = 1'b1;    // External clock

  // Register Map Constants
  localparam I2S_N_CTRL_REGS = 9;   // Number of control registers
  localparam I2S_N_STAT_REGS = 4;   // Number of status registers


//------------------------------------------------------------------------------
// I2S Register Reset Values
//------------------------------------------------------------------------------
  localparam [I2S_N_CTRL_REGS*32-1:0] I2S_CTRL_RST_VAL = {
    32'h00000000,// CTRL8: Reserved
    32'h00000000,// CTRL7: LRCLK polarity                                   
    32'h00000000,// CTRL6: Interrupts disabled
    32'h00000000,// CTRL5: FIFO control
    32'h00000004,// CTRL4: Stereo, internal clock, loopback mode
    32'h00000002,// CTRL3: 24-bit I2S format
    32'h00000003,// CTRL2: BCLK divider
    32'h00000002,// CTRL1: MCLK divider
    32'h00000000 // CTRL0: TX/RX enabled                             
  };
endpackage

`endif 
