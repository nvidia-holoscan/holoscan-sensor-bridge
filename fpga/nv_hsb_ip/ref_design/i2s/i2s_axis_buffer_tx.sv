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
// I2S TX AXI-Stream Buffer Module
//
// Thin wrapper around axis_buffer providing clock domain crossing and
// buffering for the I2S transmit data path. Converts from AXI-Stream in the
// host clock domain to a simple valid/ready interface in the I2S clock domain.
//
// The host always sends a flat stream of 32-bit samples:
//   Stereo: L0, R0, L1, R1, ... (interleaved by the host)
//   Mono:   S0, S1, S2, ...     (single-channel, no padding)
//
// axis_buffer gearboxes from AXI width to 32-bit samples. Stereo pairing
// logic latches consecutive L/R samples before presenting them to the serdes.
// In mono mode each sample is presented directly as L, with R=0.
//------------------------------------------------------------------------------

module i2s_axis_buffer_tx #(
  parameter AXI_DWIDTH    = 64,
  parameter FIFO_DEPTH    = 1024,
  parameter AFULL_THRESH  = FIFO_DEPTH * 3/4,
  parameter AEMPTY_THRESH = FIFO_DEPTH / 4,
  localparam AXI_KWIDTH      = AXI_DWIDTH/8,
  localparam MAX_SAMPLE_BITS = 32
)(
  //----------------------------------------------------------------------------
  // AXI-Stream Clock Domain (Host side)
  //----------------------------------------------------------------------------
  input                          i_axi_clk,
  input                          i_axi_rst,

  input                          i_axis_tvalid,
  input  [AXI_DWIDTH-1:0]       i_axis_tdata,
  input  [AXI_KWIDTH-1:0]       i_axis_tkeep,
  input                          i_axis_tlast,
  output                         o_axis_tready,

  //----------------------------------------------------------------------------
  // Configuration
  //----------------------------------------------------------------------------
  input                          i_channel_mode,   // 0=stereo, 1=mono

  //----------------------------------------------------------------------------
  // I2S Clock Domain (I2S side)
  //----------------------------------------------------------------------------
  input                          i_ref_clk,
  input                          i_i2s_rst,
  input                          i_bclk_negedge,

  output                         o_i2s_valid,
  output                         o_i2s_last,
  output                         o_i2s_data_avail,
  output [MAX_SAMPLE_BITS-1:0]   o_i2s_sample_l,
  output [MAX_SAMPLE_BITS-1:0]   o_i2s_sample_r,
  input                          i_i2s_ready,

  //----------------------------------------------------------------------------
  // Status
  //----------------------------------------------------------------------------
  output                         o_underrun
);

//------------------------------------------------------------------------------
// tkeep normalization: non-last beats must have tkeep all-ones
//------------------------------------------------------------------------------

  logic [AXI_KWIDTH-1:0] axis_tkeep_norm;
  assign axis_tkeep_norm = i_axis_tlast ? i_axis_tkeep : {AXI_KWIDTH{1'b1}};

//------------------------------------------------------------------------------
// axis_buffer: gearbox + dual-clock FIFO + skid buffer
// Output is 32-bit individual samples (not stereo pairs).
//------------------------------------------------------------------------------

  logic                          buf_tx_tvalid;
  logic [MAX_SAMPLE_BITS-1:0]    buf_tx_tdata;
  logic                          buf_tx_tlast;
  logic                          buf_tx_tready;
  logic                          buf_fifo_empty;

  axis_buffer #(
    .IN_DWIDTH          ( AXI_DWIDTH          ),
    .OUT_DWIDTH         ( MAX_SAMPLE_BITS     ),
    .BUF_DEPTH          ( FIFO_DEPTH          ),
    .DUAL_CLOCK         ( 1                   ),
    .WAIT2SEND          ( 0                   ),
    .LATENCY_CNT        ( 0                   ),
    .OUTPUT_SKID        ( 1                   ),
    .ALMOST_FULL_DEPTH  ( AFULL_THRESH        ),
    .ALMOST_EMPTY_DEPTH ( AEMPTY_THRESH       ),
    .W_USER             ( 1                   )
  ) u_axis_buffer (
    .in_clk             ( i_axi_clk           ),
    .in_rst             ( i_axi_rst           ),
    .out_clk            ( i_ref_clk           ),
    .out_rst            ( i_i2s_rst           ),

    .i_axis_rx_tvalid   ( i_axis_tvalid       ),
    .i_axis_rx_tdata    ( i_axis_tdata        ),
    .i_axis_rx_tlast    ( i_axis_tlast        ),
    .i_axis_rx_tuser    ( 1'b0                ),
    .i_axis_rx_tkeep    ( axis_tkeep_norm     ),
    .o_axis_rx_tready   ( o_axis_tready       ),

    .o_fifo_aempty      (                     ),
    .o_fifo_afull       (                     ),
    .o_fifo_empty       ( buf_fifo_empty      ),
    .o_fifo_full        (                     ),

    .o_axis_tx_tvalid   ( buf_tx_tvalid       ),
    .o_axis_tx_tdata    ( buf_tx_tdata        ),
    .o_axis_tx_tlast    ( buf_tx_tlast        ),
    .o_axis_tx_tuser    (                     ),
    .o_axis_tx_tkeep    (                     ),
    .i_axis_tx_tready   ( buf_tx_tready       )
  );

//------------------------------------------------------------------------------
// Stereo pairing / mono pass-through
//
// Stereo: latch first pop as L, second pop as R, then present pair.
// Mono:   each pop is a complete sample (L), R=0.
//------------------------------------------------------------------------------

  logic                          pair_phase;
  logic [MAX_SAMPLE_BITS-1:0]    latched_l;

  // Pop control: only advance the FIFO when the serdes has consumed the
  // current sample. In stereo, the L-phase pop is unconditional (latching
  // the L word while waiting for R); the R-phase pop waits for i_i2s_ready.
  assign buf_tx_tready = buf_tx_tvalid && i_bclk_negedge &&
                         (i_channel_mode ? i_i2s_ready
                                         : (!pair_phase || i_i2s_ready));

  logic [MAX_SAMPLE_BITS-1:0] i2s_sample_l;
  logic [MAX_SAMPLE_BITS-1:0] i2s_sample_r;
  logic                       i2s_valid;
  logic                       i2s_last;
  logic                       i2s_data_avail;

  always_ff @(posedge i_ref_clk) begin
    if (i_i2s_rst) begin
      i2s_sample_l   <= '0;
      i2s_sample_r   <= '0;
      i2s_valid      <= 1'b0;
      i2s_last       <= 1'b0;
      i2s_data_avail <= 1'b0;
      pair_phase     <= 1'b0;
      latched_l      <= '0;
    end else if (i_bclk_negedge) begin
      if (i_channel_mode) begin
        // Mono: present current buffer head directly; FIFO pops on i_i2s_ready
        i2s_sample_l   <= buf_tx_tdata;
        i2s_sample_r   <= '0;
        i2s_valid      <= buf_tx_tvalid;
        i2s_last       <= buf_tx_tlast && buf_tx_tvalid && buf_fifo_empty;
        i2s_data_avail <= buf_tx_tvalid;
      end else begin
        // Stereo: pair consecutive L, R words
        i2s_valid <= 1'b0;
        if (buf_tx_tvalid) begin
          if (!pair_phase) begin
            latched_l  <= buf_tx_tdata;
            pair_phase <= 1'b1;
          end else if (i_i2s_ready) begin
            i2s_sample_l <= latched_l;
            i2s_sample_r <= buf_tx_tdata;
            i2s_valid    <= 1'b1;
            i2s_last     <= buf_tx_tlast && buf_fifo_empty;
            pair_phase   <= 1'b0;
          end
        end
        i2s_data_avail <= buf_tx_tvalid && pair_phase;
      end
    end
  end

  assign o_i2s_valid      = i2s_valid;
  assign o_i2s_last       = i2s_last;
  assign o_i2s_data_avail = i2s_data_avail;
  assign o_i2s_sample_l   = i2s_sample_l;
  assign o_i2s_sample_r   = i2s_sample_r;

//------------------------------------------------------------------------------
// Underrun Detection
//------------------------------------------------------------------------------

  logic underrun_flag;

  always_ff @(posedge i_ref_clk) begin
    if (i_i2s_rst) begin
      underrun_flag <= 1'b0;
    end else begin
      if (i_i2s_ready && !buf_tx_tvalid)
        underrun_flag <= 1'b1;
      if (buf_tx_tvalid)
        underrun_flag <= 1'b0;
    end
  end

  assign o_underrun = underrun_flag;

endmodule
