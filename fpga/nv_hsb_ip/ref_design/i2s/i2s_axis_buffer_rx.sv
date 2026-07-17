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
// I2S RX AXI-Stream Buffer Module
//
// Thin wrapper around axis_buffer providing clock domain crossing and
// buffering for the I2S receive data path. Converts from a simple valid/ready
// interface in the I2S clock domain to AXI-Stream in the host clock domain.
//
// axis_buffer handles: gearbox (sample width -> AXI width), dual-clock FIFO,
// and output skid buffer. This wrapper adds I2S-specific concerns:
// - BCLK-gated input handshake
// - Packet formation (tlast every PKT_SIZE AXI beats)
// - Overrun detection
//------------------------------------------------------------------------------

module i2s_axis_buffer_rx #(
  parameter AXI_DWIDTH    = 64,
  parameter FIFO_DEPTH    = 1024,
  parameter AFULL_THRESH  = FIFO_DEPTH * 3/4,
  parameter AEMPTY_THRESH = FIFO_DEPTH / 4,
  localparam AXI_KWIDTH      = AXI_DWIDTH/8,
  localparam MAX_SAMPLE_BITS = 32,
  localparam PKT_CNT_W      = 16
)(
  //----------------------------------------------------------------------------
  // I2S Clock Domain (I2S side)
  //----------------------------------------------------------------------------
  input                         i_ref_clk,
  input                         i_i2s_rst,
  input                         i_bclk_posedge,

  input                         i_i2s_valid,
  input  [MAX_SAMPLE_BITS-1:0]  i_i2s_data,
  output                        o_i2s_ready,

  //----------------------------------------------------------------------------
  // AXI-Stream Clock Domain (Host side)
  //----------------------------------------------------------------------------
  input                         i_axi_clk,
  input                         i_axi_rst,

  output                        o_axis_tvalid,
  output [AXI_DWIDTH-1:0]       o_axis_tdata,
  output [AXI_KWIDTH-1:0]       o_axis_tkeep,
  output                        o_axis_tlast,
  input                         i_axis_tready,

  //----------------------------------------------------------------------------
  // Configuration
  //----------------------------------------------------------------------------
  input  [PKT_CNT_W-1:0]       i_pkt_size,

  //----------------------------------------------------------------------------
  // Status and Control
  //----------------------------------------------------------------------------
  output                        o_fifo_full,
  output                        o_fifo_empty,
  output                        o_fifo_afull,
  output                        o_fifo_aempty,
  output                        o_overrun
);

//------------------------------------------------------------------------------
// I2S input → AXI-Stream (BCLK-gated)
//------------------------------------------------------------------------------

  localparam I2S_KWIDTH = MAX_SAMPLE_BITS / 8;

  logic buf_rx_tvalid;
  assign buf_rx_tvalid = i_i2s_valid && i_bclk_posedge;

//------------------------------------------------------------------------------
// axis_buffer: gearbox + dual-clock FIFO + skid buffer
//------------------------------------------------------------------------------

  logic                    buf_tx_tvalid;
  logic [AXI_DWIDTH-1:0]  buf_tx_tdata;
  logic [AXI_KWIDTH-1:0]  buf_tx_tkeep;
  logic                    buf_tx_tready;
  logic                    buf_fifo_full;
  logic                    buf_fifo_afull;

  axis_buffer #(
    .IN_DWIDTH          ( MAX_SAMPLE_BITS     ),
    .OUT_DWIDTH         ( AXI_DWIDTH          ),
    .BUF_DEPTH          ( FIFO_DEPTH          ),
    .DUAL_CLOCK         ( 1                   ),
    .WAIT2SEND          ( 0                   ),
    .LATENCY_CNT        ( 0                   ),
    .OUTPUT_SKID        ( 1                   ),
    .ALMOST_FULL_DEPTH  ( AFULL_THRESH        ),
    .ALMOST_EMPTY_DEPTH ( AEMPTY_THRESH       ),
    .W_USER             ( 1                   )
  ) u_axis_buffer (
    .in_clk             ( i_ref_clk           ),
    .in_rst             ( i_i2s_rst           ),
    .out_clk            ( i_axi_clk           ),
    .out_rst            ( i_axi_rst           ),

    .i_axis_rx_tvalid   ( buf_rx_tvalid       ),
    .i_axis_rx_tdata    ( i_i2s_data          ),
    .i_axis_rx_tlast    ( 1'b0                ),
    .i_axis_rx_tuser    ( 1'b0                ),
    .i_axis_rx_tkeep    ( {I2S_KWIDTH{1'b1}}  ),
    .o_axis_rx_tready   ( o_i2s_ready         ),

    .o_fifo_aempty      ( o_fifo_aempty       ),
    .o_fifo_afull       ( buf_fifo_afull      ),
    .o_fifo_empty       ( o_fifo_empty        ),
    .o_fifo_full        ( buf_fifo_full       ),

    .o_axis_tx_tvalid   ( buf_tx_tvalid       ),
    .o_axis_tx_tdata    ( buf_tx_tdata        ),
    .o_axis_tx_tlast    (                     ),
    .o_axis_tx_tuser    (                     ),
    .o_axis_tx_tkeep    ( buf_tx_tkeep        ),
    .i_axis_tx_tready   ( buf_tx_tready       )
  );

  assign o_fifo_full  = buf_fifo_full;
  assign o_fifo_afull = buf_fifo_afull;

//------------------------------------------------------------------------------
// Packet Formation (tlast every PKT_SIZE AXI beats)
//------------------------------------------------------------------------------

  logic [PKT_CNT_W-1:0] sample_count;
  logic                  pkt_last;

  always_ff @(posedge i_axi_clk) begin
    if (i_axi_rst) begin
      sample_count <= '0;
    end else if (buf_tx_tvalid && buf_tx_tready) begin
      if (pkt_last)
        sample_count <= '0;
      else
        sample_count <= sample_count + 1'b1;
    end
  end

  // i_pkt_size == 0 disables packetization (tlast never asserts)
  assign pkt_last = (i_pkt_size != '0) && (sample_count >= (i_pkt_size - PKT_CNT_W'(1)));

//------------------------------------------------------------------------------
// AXI-Stream Output
//------------------------------------------------------------------------------

  assign o_axis_tvalid = buf_tx_tvalid;
  assign o_axis_tdata  = buf_tx_tdata;
  assign o_axis_tkeep  = (pkt_last && buf_tx_tvalid) ? buf_tx_tkeep : {AXI_KWIDTH{1'b1}};
  assign o_axis_tlast  = pkt_last && buf_tx_tvalid;
  assign buf_tx_tready = i_axis_tready;

//------------------------------------------------------------------------------
// Overrun Detection
//------------------------------------------------------------------------------

  logic overrun_flag;

  always_ff @(posedge i_ref_clk) begin
    if (i_i2s_rst) begin
      overrun_flag <= 1'b0;
    end else begin
      if (i_i2s_valid && buf_fifo_full)
        overrun_flag <= 1'b1;
      if (!buf_fifo_afull)
        overrun_flag <= 1'b0;
    end
  end

  assign o_overrun    = overrun_flag;

endmodule
