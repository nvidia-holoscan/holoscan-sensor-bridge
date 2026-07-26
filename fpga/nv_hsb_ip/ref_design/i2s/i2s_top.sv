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

//------------------------------------------------------------------------------
// I2S Peripheral Top Level Module
// 
// This module provides a complete I2S audio interface with:
// - Variable sample rates (8K to 192K Hz)
// - Variable sample widths (8, 16, 24, 32 bits)
// - AXI-Stream host interface for both TX and RX
// - APB configuration interface
// - Support for I2S, Left/Right Justified, and PCM formats
//------------------------------------------------------------------------------

module i2s_top
  import apb_pkg::*;
  import i2s_pkg::*;
#(
  parameter AXI_DWIDTH       = 32,           // Host AXI-Stream data width
  parameter TX_FIFO_DEPTH    = 1024,         // TX buffer depth
  parameter RX_FIFO_DEPTH    = 1024,         // RX buffer depth
  parameter APB_ADDR_WIDTH   = 12,           // APB address space width
  localparam AXI_KWIDTH      = AXI_DWIDTH/8, // AXI-Stream keep width
  localparam MAX_SAMPLE_BITS = 32            // I2S max sample width (fixed; 8/16/24/32 at runtime)
)(
  //----------------------------------------------------------------------------
  // Clock and Reset
  //----------------------------------------------------------------------------
  input                     i_apb_clk,        // APB clock domain
  input                     i_apb_rst,        // APB reset (active high, synchronous to i_apb_clk)
  input                     i_ref_clk,        // Reference clock for I2S clk gen / serdes
  input                     i_ref_rst,        // Reset (active high, synchronous to i_ref_clk)
  input                     i_axis_clk,       // AXI-Stream clock domain
  input                     i_axis_rst,       // Reset (active high, synchronous to i_axis_clk)
  
  //----------------------------------------------------------------------------
  // APB Configuration Interface
  //----------------------------------------------------------------------------
  input  apb_m2s            i_apb_m2s,        // APB interface 
  output apb_s2m            o_apb_s2m,        // APB interface 
  
  //----------------------------------------------------------------------------
  // Host AXI-Stream TX Interface (Host to I2S)
  //----------------------------------------------------------------------------
  input                     i_tx_axis_tvalid, // TX data valid
  input  [AXI_DWIDTH-1:0]   i_tx_axis_tdata,  // TX data
  input  [AXI_KWIDTH-1:0]   i_tx_axis_tkeep,  // TX data keep
  input                     i_tx_axis_tlast,  // TX packet boundary
  input                     i_tx_axis_tuser,  // TX user signal
  output                    o_tx_axis_tready, // TX ready
  
  //----------------------------------------------------------------------------
  // Host AXI-Stream RX Interface (I2S to Host)
  //----------------------------------------------------------------------------
  output                    o_rx_axis_tvalid, // RX data valid
  output [AXI_DWIDTH-1:0]   o_rx_axis_tdata,  // RX data
  output [AXI_KWIDTH-1:0]   o_rx_axis_tkeep,  // RX data keep
  output                    o_rx_axis_tlast,  // RX packet boundary
  output                    o_rx_axis_tuser,  // RX user signal
  input                     i_rx_axis_tready, // RX ready
  
  //----------------------------------------------------------------------------
  // I2S Physical Interface
  //----------------------------------------------------------------------------
  output                    o_i2s_bclk,       // I2S bit clock output
  output                    o_i2s_lrclk,      // I2S L/R clock (word select)
  output                    o_i2s_sdata_tx,   // I2S serial data output
  input                     i_i2s_sdata_rx,   // I2S serial data input
  output                    o_i2s_mclk_out,   // I2S mclk output
  
  //----------------------------------------------------------------------------
  // Status and Interrupts
  //----------------------------------------------------------------------------
  output                    o_tx_underrun,    // TX FIFO underrun
  output                    o_rx_overrun,     // RX FIFO overrun
  output                    o_pll_locked,     // Clock PLL locked status
  output                    o_irq             // Interrupt request
);

//------------------------------------------------------------------------------
// Internal Signals
//------------------------------------------------------------------------------

  // Configuration signals from APB registers
  logic [31:0]              ctrl_regs [I2S_N_CTRL_REGS];
  logic [31:0]              stat_regs [I2S_N_STAT_REGS];
  
  // Clock generation signals
  logic                       i2s_mclk;
  logic                       i2s_bclk;
  logic                       i2s_lrclk;
  logic                       clk_locked;
  logic                       bclk_en;
  logic                       bclk_posedge;
  logic                       bclk_negedge;
  logic                       lrclk_en;
  
  // TX path internal signals
  logic                       tx_sample_valid;
  logic                       tx_sample_last;
  logic                       tx_data_avail;
  logic [MAX_SAMPLE_BITS-1:0] tx_sample_l;
  logic [MAX_SAMPLE_BITS-1:0] tx_sample_r;
  logic                       tx_sample_ready;
  
  // RX path internal signals
  logic                       rx_i2s_valid;
  logic [MAX_SAMPLE_BITS-1:0] rx_i2s_data;
  logic                       rx_i2s_ready;
  
  // Configuration decode
  logic [31:0]                mclk_div;
  logic [31:0]                bclk_div;
  logic [1:0]                 bit_depth_sel;
  logic [1:0]                 data_format_sel;
  logic                       channel_mode;
  logic                       tx_enable;
  logic                       rx_enable;
  logic                       loopback_mode;
  logic                       lrclk_polarity;
  logic [15:0]                rx_pkt_size;
  logic                       slot_mode;
  logic                       sdata_rx_internal;  // Internal RX data signal


//------------------------------------------------------------------------------
// Configuration Decode
//------------------------------------------------------------------------------

  // Extract configuration from control registers
  assign tx_enable        = ctrl_regs[0][0];
  assign rx_enable        = ctrl_regs[0][1];
  assign mclk_div         = ctrl_regs[1];
  assign bclk_div         = ctrl_regs[2];
  assign bit_depth_sel    = ctrl_regs[3][1:0];
  assign data_format_sel  = ctrl_regs[3][3:2];
  assign channel_mode     = ctrl_regs[4][0];
  assign loopback_mode    = ctrl_regs[4][2];
  assign rx_pkt_size      = ctrl_regs[5][15:0];
  assign lrclk_polarity   = ctrl_regs[7][0];
  assign slot_mode        = ctrl_regs[8][0];

  // Status register assignments
  assign stat_regs[0] = {31'b0, clk_locked};
  assign stat_regs[1] = 32'b0; // FIFO status (TBD)
  assign stat_regs[2] = {30'b0, o_rx_overrun, o_tx_underrun};
  assign stat_regs[3] = 32'b0; // Debug status

//------------------------------------------------------------------------------
// Module Instantiations
//------------------------------------------------------------------------------

  // APB Register Interface
  s_apb_reg #(
    .N_CTRL               ( I2S_N_CTRL_REGS     ),
    .N_STAT               ( I2S_N_STAT_REGS     ),
    .W_OFST               ( APB_ADDR_WIDTH      ),
    .RST_VAL              ( I2S_CTRL_RST_VAL    )
  ) u_apb_reg (
    .i_aclk               ( i_apb_clk           ),
    .i_arst               ( i_apb_rst           ),
    .i_apb_m2s            ( i_apb_m2s           ),
    .o_apb_s2m            ( o_apb_s2m           ),
    .i_pclk               ( i_ref_clk           ),
    .i_prst               ( i_ref_rst             ),
    .o_ctrl               ( ctrl_regs           ),
    .i_stat               ( stat_regs           )
  );

  // Clock Generation
  i2s_clk_gen u_clk_gen (
    .i_ref_clk            ( i_ref_clk           ),
    .i_rst                ( i_ref_rst             ),
    .i_mclk_div           ( mclk_div            ),
    .i_bclk_div           ( bclk_div            ),
    .i_bit_depth_sel      ( bit_depth_sel       ),
    .i_slot_mode          ( slot_mode           ),
    .i_enable             ( tx_enable | rx_enable ),
    .o_mclk               ( i2s_mclk            ),
    .o_bclk               ( i2s_bclk            ),
    .o_lrclk              ( i2s_lrclk           ),
    .o_locked             ( clk_locked          ),
    .o_bclk_en            ( bclk_en             ),
    .o_lrclk_en           ( lrclk_en            ),
    .o_bclk_posedge       ( bclk_posedge        ),
    .o_bclk_negedge       ( bclk_negedge        )
  );

  // TX AXI-Stream Buffer (Clock Domain Crossing with mode awareness)
  i2s_axis_buffer_tx #(
    .AXI_DWIDTH           ( AXI_DWIDTH          ),
    .FIFO_DEPTH           ( TX_FIFO_DEPTH       )
  ) u_axis_buffer_tx (
    .i_axi_clk            ( i_axis_clk          ),
    .i_axi_rst            ( i_axis_rst          ),
    .i_axis_tvalid        ( i_tx_axis_tvalid    ),
    .i_axis_tdata         ( i_tx_axis_tdata     ),
    .i_axis_tkeep         ( i_tx_axis_tkeep     ),
    .i_axis_tlast         ( i_tx_axis_tlast     ),
    .o_axis_tready        ( o_tx_axis_tready    ),
    .i_channel_mode       ( channel_mode        ),
    .i_ref_clk            ( i_ref_clk           ),
    .i_bclk_negedge       ( bclk_negedge        ),
    .i_i2s_rst            ( i_ref_rst             ),
    .o_i2s_valid          ( tx_sample_valid     ),
    .o_i2s_last           ( tx_sample_last      ),
    .o_i2s_data_avail     ( tx_data_avail       ),
    .o_i2s_sample_l       ( tx_sample_l         ),
    .o_i2s_sample_r       ( tx_sample_r         ),
    .i_i2s_ready          ( tx_sample_ready     ),
    .o_underrun           ( o_tx_underrun       )
  );


  // Internal loopback logic
  assign sdata_rx_internal = loopback_mode ? o_i2s_sdata_tx : i_i2s_sdata_rx;

  // I2S Serializer/Deserializer
  i2s_serdes u_serdes (
    .i_bclk               ( i2s_bclk            ),
    .i_lrclk              ( i2s_lrclk           ),
    .i_ref_clk            ( i_ref_clk           ),
    .i_rst                ( i_ref_rst             ),
    .i_bclk_en            ( bclk_en             ),
    .i_lrclk_en           ( lrclk_en            ),
    .i_bit_depth          ( bit_depth_sel       ),
    .i_data_format        ( data_format_sel     ),
    .i_channel_mode       ( channel_mode        ),
    .i_slot_mode          ( slot_mode           ),
    .i_lrclk_polarity     ( lrclk_polarity      ),
    .i_tx_enable          ( tx_enable           ),
    .i_rx_enable          ( rx_enable           ),
    .i_tx_valid           ( tx_sample_valid     ),
    .i_tx_ready           ( tx_data_avail       ),
    .i_tx_last            ( tx_sample_last      ),
    .i_tx_data_l          ( tx_sample_l         ),
    .i_tx_data_r          ( tx_sample_r         ),
    .o_tx_ready           ( tx_sample_ready     ),
    .o_sdata_tx           ( o_i2s_sdata_tx      ),
    .i_sdata_rx           ( sdata_rx_internal   ), // Changed from i_i2s_sdata_rx
    .o_rx_valid           ( rx_i2s_valid        ),
    .o_rx_data            ( rx_i2s_data         ),
    .i_rx_ready           ( rx_i2s_ready        )
  );

  // RX AXI-Stream Buffer (Clock Domain Crossing)
  i2s_axis_buffer_rx #(
    .AXI_DWIDTH           ( AXI_DWIDTH          ),
    .FIFO_DEPTH           ( RX_FIFO_DEPTH       )
  ) u_axis_buffer_rx (
    .i_ref_clk            ( i_ref_clk           ),
    .i_bclk_posedge       ( bclk_posedge        ),
    .i_i2s_rst            ( i_ref_rst              ),
    .i_i2s_valid          ( rx_i2s_valid        ),
    .i_i2s_data           ( rx_i2s_data         ),
    .o_i2s_ready          ( rx_i2s_ready        ),
    .i_axi_clk            ( i_axis_clk          ),
    .i_axi_rst            ( i_axis_rst          ),
    .o_axis_tvalid        ( o_rx_axis_tvalid    ),
    .o_axis_tdata         ( o_rx_axis_tdata     ),
    .o_axis_tkeep         ( o_rx_axis_tkeep     ),
    .o_axis_tlast         ( o_rx_axis_tlast     ),
    .i_axis_tready        ( i_rx_axis_tready    ),
    .i_pkt_size           ( rx_pkt_size         ),
    .o_fifo_full          (                     ),
    .o_fifo_aempty        (                     ),
    .o_overrun            ( o_rx_overrun        )
  );
  assign o_rx_axis_tuser = 1'b0;
//------------------------------------------------------------------------------
// Output Assignments
//------------------------------------------------------------------------------

  assign o_i2s_bclk     = i2s_bclk;
  assign o_i2s_lrclk    = i2s_lrclk;
  assign o_i2s_mclk_out = i2s_mclk;
  assign o_pll_locked   = clk_locked;
  assign o_irq          = o_tx_underrun | o_rx_overrun; // Simple interrupt logic

endmodule 
