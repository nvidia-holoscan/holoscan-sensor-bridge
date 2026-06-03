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
// This module provides clock domain crossing and buffering for I2S transmit
// data path. It converts from AXI-Stream interface in the host clock domain
// to simple valid/ready interface in the I2S clock domain.
//
// Features:
// - Dual-clock FIFO for clock domain crossing
// - Configurable FIFO depth
// - Almost-full/empty thresholds for flow control
// - Underrun detection and handling
//------------------------------------------------------------------------------

module i2s_axis_buffer_tx #(
  parameter AXI_DWIDTH  = 64,              // AXI-Stream data width
  parameter I2S_DWIDTH  = 32,              // I2S data width
  parameter FIFO_DEPTH  = 1024,            // FIFO depth (must be power of 2)
  parameter AFULL_THRESH = FIFO_DEPTH * 3/4, // Almost full threshold
  parameter AEMPTY_THRESH = FIFO_DEPTH / 4,  // Almost empty threshold
  localparam AXI_KWIDTH = AXI_DWIDTH/8,    // AXI-Stream keep width
  localparam ADDR_WIDTH = $clog2(FIFO_DEPTH) // FIFO address width
)(
  //----------------------------------------------------------------------------
  // AXI-Stream Clock Domain (Host side)
  //----------------------------------------------------------------------------
  input                    i_axi_clk,      // AXI-Stream clock
  input                    i_axi_rst,      // AXI-Stream reset (active high)
  
  // Configuration
  input                    i_stereo_mode,  // 1=stereo, 0=mono
  
  // AXI-Stream Input
  input                    i_axis_tvalid,  // AXI data valid
  input  [AXI_DWIDTH-1:0]  i_axis_tdata,   // AXI data
  input  [AXI_KWIDTH-1:0]  i_axis_tkeep,   // AXI data keep
  input                    i_axis_tlast,   // AXI packet boundary
  output                   o_axis_tready,  // AXI ready
  
  //----------------------------------------------------------------------------
  // I2S Clock Domain (I2S side)
  //----------------------------------------------------------------------------
  input                    i_ref_clk,      // Reference clock
  input                    i_i2s_rst,      // I2S reset (active high)
  input                    i_bclk_posedge, // BCLK positive edge
  input                    i_bclk_negedge, // BCLK negative edge
  
  // I2S Output Interface (doubled width for stereo L+R samples)
  output                   o_i2s_valid,    // I2S data valid
  output [I2S_DWIDTH-1:0]   o_i2s_sample_l, // I2S data left
  output [I2S_DWIDTH-1:0]   o_i2s_sample_r, // I2S data right
  input                    i_i2s_ready,    // I2S ready
  
  //----------------------------------------------------------------------------
  // Status and Control
  //----------------------------------------------------------------------------
  output                   o_fifo_full,    // FIFO full flag
  output                   o_fifo_empty,   // FIFO empty flag
  output                   o_fifo_afull,   // FIFO almost full
  output                   o_fifo_aempty,  // FIFO almost empty
  output [ADDR_WIDTH:0]    o_fifo_count,   // FIFO fill count
  output                   o_underrun      // Underrun error flag
);

//------------------------------------------------------------------------------
// Internal Signals
//------------------------------------------------------------------------------

  // FIFO interface signals (doubled width for L+R samples)
  logic                    fifo_wr_en;
  logic [I2S_DWIDTH*2-1:0] fifo_wr_data;
  logic                    fifo_full;
  logic                    fifo_afull;
  
  logic                    fifo_rd_en;
  logic [I2S_DWIDTH*2-1:0] fifo_rd_data;
  logic                    fifo_empty;
  logic                    fifo_aempty;
  logic                    fifo_rd_valid;
  
  logic [ADDR_WIDTH:0]     fifo_wr_count;
  logic [ADDR_WIDTH:0]     fifo_rd_count;
  
  // Width conversion signals
  logic                    conv_valid;
  logic [I2S_DWIDTH*2-1:0] conv_data;
  logic                    conv_ready;
  
  // Underrun detection
  logic                    underrun_flag;
  logic                    i2s_ready_sync;

//------------------------------------------------------------------------------
// Width Conversion (AXI-Stream to I2S samples)
//------------------------------------------------------------------------------

generate
  if (AXI_DWIDTH == I2S_DWIDTH*2) begin: gen_no_conv
    // Direct connection when AXI width matches doubled I2S width
    assign conv_valid = i_axis_tvalid;
    // Handle mono/stereo mode
    assign conv_data  = i_stereo_mode ? i_axis_tdata : 
                       {i_axis_tdata[I2S_DWIDTH-1:0], i_axis_tdata[I2S_DWIDTH-1:0]}; // Duplicate mono sample
    assign o_axis_tready = conv_ready;
    
  end else if (AXI_DWIDTH > I2S_DWIDTH*2) begin: gen_downsize
    // Downsize from wider AXI - extract samples based on mode
    localparam RATIO_STEREO = AXI_DWIDTH / (I2S_DWIDTH*2);
    localparam RATIO_MONO = AXI_DWIDTH / I2S_DWIDTH;
    localparam CNT_WIDTH = $clog2(RATIO_MONO); // Use larger ratio for counter
    
    logic [CNT_WIDTH-1:0]  sample_cnt;
    logic [AXI_DWIDTH-1:0] data_reg;
    logic                  data_valid;
    logic [3:0]            effective_ratio;
    logic [5:0]            shift_amount;
    
    assign effective_ratio = i_stereo_mode ? RATIO_STEREO : RATIO_MONO;
    assign shift_amount = i_stereo_mode ? (I2S_DWIDTH*2) : I2S_DWIDTH;
    
    always_ff @(posedge i_axi_clk) begin
      if (i_axi_rst) begin
        sample_cnt <= '0;
        data_reg   <= '0;
        data_valid <= 1'b0;
      end else begin
        if (i_axis_tvalid && o_axis_tready) begin
          data_reg   <= i_axis_tdata;
          data_valid <= 1'b1;
          sample_cnt <= '0;
        end else if (conv_valid && conv_ready) begin
          data_reg   <= data_reg >> shift_amount;
          sample_cnt <= sample_cnt + 1'b1;
          if (sample_cnt == (effective_ratio - 1)) begin
            data_valid <= 1'b0;
          end
        end
      end
    end
    
    assign conv_valid = data_valid;
    // Handle mono/stereo mode in output
    assign conv_data = i_stereo_mode ? data_reg[I2S_DWIDTH*2-1:0] :
                      {data_reg[I2S_DWIDTH-1:0], data_reg[I2S_DWIDTH-1:0]}; // Duplicate mono
    assign o_axis_tready = !data_valid || (conv_ready && (sample_cnt == (effective_ratio - 1)));
    
  end else begin: gen_upsize
    // Upsize from narrower AXI - accumulate based on mode
    localparam RATIO_STEREO = (I2S_DWIDTH*2) / AXI_DWIDTH;
    localparam RATIO_MONO = I2S_DWIDTH / AXI_DWIDTH;
    localparam CNT_WIDTH = $clog2(RATIO_STEREO); // Use larger ratio for counter
    
    logic [CNT_WIDTH-1:0]  sample_cnt;
    logic [I2S_DWIDTH*2-1:0] data_reg;
    logic                  data_valid;
    logic [3:0]            effective_ratio;
    
    assign effective_ratio = i_stereo_mode ? RATIO_STEREO : RATIO_MONO;
    
    always_ff @(posedge i_axi_clk) begin
      if (i_axi_rst) begin
        sample_cnt <= '0;
        data_reg   <= '0;
        data_valid <= 1'b0;
      end else begin
        if (i_axis_tvalid && o_axis_tready) begin
          data_reg <= {data_reg[I2S_DWIDTH*2-AXI_DWIDTH-1:0], i_axis_tdata};
          sample_cnt <= sample_cnt + 1'b1;
          if (sample_cnt == (effective_ratio - 1)) begin
            data_valid <= 1'b1;
            sample_cnt <= '0;
          end
        end else if (conv_valid && conv_ready) begin
          data_valid <= 1'b0;
        end
      end
    end
    
    assign conv_valid = data_valid;
    // Handle mono/stereo mode in output
    assign conv_data = i_stereo_mode ? data_reg :
                      {data_reg[I2S_DWIDTH-1:0], data_reg[I2S_DWIDTH-1:0]}; // Duplicate mono
    assign o_axis_tready = !data_valid;
    
  end
endgenerate

//------------------------------------------------------------------------------
// Dual-Clock FIFO Instance
//------------------------------------------------------------------------------

  dc_fifo_stub #(
    .DATA_WIDTH   ( I2S_DWIDTH*2   ),
    .FIFO_DEPTH   ( FIFO_DEPTH     ),
    .ALMOST_FULL  ( AFULL_THRESH   ),
    .ALMOST_EMPTY ( AEMPTY_THRESH  ),
    .MEM_STYLE    ( "BLOCK"        )
  ) u_dc_fifo (
    .wrrst        ( i_axi_rst      ),
    .rdrst        ( i_i2s_rst      ),
    // Write side (AXI clock domain)
    .wrclk        ( i_axi_clk      ),
    .wren         ( fifo_wr_en     ),
    .wrdin        ( fifo_wr_data   ),
    .wrcount      ( fifo_wr_count  ),
    .full         ( fifo_full      ),
    .afull        ( fifo_afull     ),
    .over         (                ), // Unconnected
    
    // Read side (Reference clock domain with BCLK enable)
    .rdclk        ( i_ref_clk      ),
    .rden         ( fifo_rd_en     ),
    .rddout       ( fifo_rd_data   ),
    .rdval        ( fifo_rd_valid  ),
    .rdcount      ( fifo_rd_count  ),
    .empty        ( fifo_empty     ),
    .aempty       ( fifo_aempty    ),
    .under        (                )  // Unconnected
  );

//------------------------------------------------------------------------------
// FIFO Write Interface
//------------------------------------------------------------------------------

  assign fifo_wr_en   = conv_valid && conv_ready;
  assign fifo_wr_data = conv_data;
  assign conv_ready   = !fifo_full;

//------------------------------------------------------------------------------
// FIFO Read Interface
//------------------------------------------------------------------------------

  logic [I2S_DWIDTH-1:0]   i2s_sample_l;    
  logic [I2S_DWIDTH-1:0]   i2s_sample_r;
  logic i2s_valid;

  always_ff @(posedge i_ref_clk) begin
    if (i_i2s_rst) begin
      i2s_sample_l <= 'b0;
      i2s_sample_r <= 'b0;
      i2s_valid    <= 'b0;
    end
    else if (i_bclk_negedge) begin
      i2s_sample_r <= fifo_rd_data[I2S_DWIDTH-1:0];
      i2s_sample_l <= fifo_rd_data[I2S_DWIDTH*2-1:I2S_DWIDTH];
      i2s_valid    <= fifo_rd_en;
    end
  end

  assign fifo_rd_en   = !fifo_empty && i_i2s_ready && i_bclk_negedge;
  // assign o_i2s_valid  = fifo_rd_valid;
  // assign o_i2s_sample_r = fifo_rd_data[I2S_DWIDTH-1:0];
  // assign o_i2s_sample_l = fifo_rd_data[I2S_DWIDTH*2-1:I2S_DWIDTH];
  assign o_i2s_valid    = i2s_valid;
  assign o_i2s_sample_r = i2s_sample_r;
  assign o_i2s_sample_l = i2s_sample_l;

//------------------------------------------------------------------------------
// Underrun Detection
//------------------------------------------------------------------------------

  // Synchronize I2S ready to AXI clock domain for underrun detection
  data_sync #(
    .DATA_WIDTH  ( 1           ),
    .SYNC_DEPTH  ( 2           )
  ) u_ready_sync (
    .clk         ( i_axi_clk   ),
    .rst_n       ( !i_axi_rst  ),
    .sync_in     ( i_i2s_ready ),
    .sync_out    ( i2s_ready_sync )
  );

  // Detect underrun condition
  always_ff @(posedge i_ref_clk) begin
    if (i_i2s_rst) begin
      underrun_flag <= 1'b0;
    end else begin
      if (i_i2s_ready && fifo_empty) begin
        underrun_flag <= 1'b1;
      end
      // Clear underrun flag when FIFO has data again
      if (!fifo_empty) begin
        underrun_flag <= 1'b0;
      end
    end
  end

//------------------------------------------------------------------------------
// Status Output Assignments
//------------------------------------------------------------------------------

  assign o_fifo_full   = fifo_full;
  assign o_fifo_empty  = fifo_empty;
  assign o_fifo_afull  = fifo_afull;
  assign o_fifo_aempty = fifo_aempty;
  assign o_fifo_count  = fifo_rd_count; // Use read side count for I2S domain
  assign o_underrun    = underrun_flag;

endmodule 
