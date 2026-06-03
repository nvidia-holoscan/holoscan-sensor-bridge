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
// This module provides clock domain crossing and buffering for I2S receive
// data path. It converts from simple valid/ready interface in the I2S clock 
// domain to AXI-Stream interface in the host clock domain.
//
// Features:
// - Dual-clock FIFO for clock domain crossing
// - Configurable FIFO depth
// - Almost-full/empty thresholds for flow control
// - Overrun detection and handling
// - Packet formation for AXI-Stream output
//------------------------------------------------------------------------------

module i2s_axis_buffer_rx #(
  parameter I2S_DWIDTH  = 32,              // I2S data width
  parameter AXI_DWIDTH  = 64,              // AXI-Stream data width
  parameter FIFO_DEPTH  = 1024,            // FIFO depth (must be power of 2)
  parameter AFULL_THRESH = FIFO_DEPTH * 3/4, // Almost full threshold
  parameter AEMPTY_THRESH = FIFO_DEPTH / 4,  // Almost empty threshold
  parameter PKT_SIZE    = 64,              // Samples per AXI-Stream packet
  localparam AXI_KWIDTH = AXI_DWIDTH/8,    // AXI-Stream keep width
  localparam ADDR_WIDTH = $clog2(FIFO_DEPTH) // FIFO address width
)(
  //----------------------------------------------------------------------------
  // I2S Clock Domain (I2S side)
  //----------------------------------------------------------------------------
  input                    i_ref_clk,      // Reference clock
  input                    i_i2s_rst,      // I2S reset (active high)
  input                    i_bclk_posedge, // BCLK positive edge
  input                    i_bclk_negedge, // BCLK negative edge
  
  // I2S Input Interface
  input                    i_i2s_valid,    // I2S data valid
  input  [I2S_DWIDTH-1:0]  i_i2s_data,     // I2S data
  output                   o_i2s_ready,    // I2S ready
  
  //----------------------------------------------------------------------------
  // AXI-Stream Clock Domain (Host side)
  //----------------------------------------------------------------------------
  input                    i_axi_clk,      // AXI-Stream clock
  input                    i_axi_rst,      // AXI-Stream reset (active high)
  
  // AXI-Stream Output
  output                   o_axis_tvalid,  // AXI data valid
  output [AXI_DWIDTH-1:0]  o_axis_tdata,   // AXI data
  output [AXI_KWIDTH-1:0]  o_axis_tkeep,   // AXI data keep
  output                   o_axis_tlast,   // AXI packet boundary
  input                    i_axis_tready,  // AXI ready
  
  //----------------------------------------------------------------------------
  // Status and Control
  //----------------------------------------------------------------------------
  output                   o_fifo_full,    // FIFO full flag
  output                   o_fifo_empty,   // FIFO empty flag
  output                   o_fifo_afull,   // FIFO almost full
  output                   o_fifo_aempty,  // FIFO almost empty
  output [ADDR_WIDTH:0]    o_fifo_count,   // FIFO fill count
  output                   o_overrun       // Overrun error flag
);

//------------------------------------------------------------------------------
// Internal Signals
//------------------------------------------------------------------------------

  // FIFO interface signals
  logic                    fifo_wr_en;
  logic [I2S_DWIDTH-1:0]   fifo_wr_data;
  logic                    fifo_full;
  logic                    fifo_afull;
  
  logic                    fifo_rd_en;
  logic [I2S_DWIDTH-1:0]   fifo_rd_data;
  logic                    fifo_empty;
  logic                    fifo_aempty;
  logic                    fifo_rd_valid;
  
  logic [ADDR_WIDTH:0]     fifo_wr_count;
  logic [ADDR_WIDTH:0]     fifo_rd_count;
  
  // Width conversion signals
  logic                    conv_valid;
  logic [AXI_DWIDTH-1:0]   conv_data;
  logic [AXI_KWIDTH-1:0]   conv_keep;
  logic                    conv_ready;
  
  // Packet formation signals
  logic [$clog2(PKT_SIZE):0] sample_count;
  logic                    pkt_last;
  
  // Overrun detection
  logic                    overrun_flag;

//------------------------------------------------------------------------------
// FIFO Write Interface (I2S side)
//------------------------------------------------------------------------------

  assign fifo_wr_en   = i_i2s_valid && o_i2s_ready && i_bclk_posedge;
  assign fifo_wr_data = i_i2s_data;
  assign o_i2s_ready  = !fifo_full;

//------------------------------------------------------------------------------
// Dual-Clock FIFO Instance
//------------------------------------------------------------------------------

  dc_fifo_stub #(
    .DATA_WIDTH   ( I2S_DWIDTH     ),
    .FIFO_DEPTH   ( FIFO_DEPTH     ),
    .ALMOST_FULL  ( AFULL_THRESH   ),
    .ALMOST_EMPTY ( AEMPTY_THRESH  ),
    .MEM_STYLE    ( "BLOCK"        )
  ) u_dc_fifo (
    // Asynchronous reset
    .wrrst        ( i_i2s_rst      ),
    .rdrst        ( i_axi_rst      ),
    // Write side (Reference clock domain with BCLK enable)
    .wrclk        ( i_ref_clk      ),
    .wren         ( fifo_wr_en     ),
    .wrdin        ( fifo_wr_data   ),
    .wrcount      ( fifo_wr_count  ),
    .full         ( fifo_full      ),
    .afull        ( fifo_afull     ),
    .over         (                ), // Unconnected
    
    // Read side (AXI clock domain)
    .rdclk        ( i_axi_clk      ),
    .rden         ( fifo_rd_en     ),
    .rddout       ( fifo_rd_data   ),
    .rdval        ( fifo_rd_valid  ),
    .rdcount      ( fifo_rd_count  ),
    .empty        ( fifo_empty     ),
    .aempty       ( fifo_aempty    ),
    .under        (                )  // Unconnected
  );

//------------------------------------------------------------------------------
// Width Conversion (I2S samples to AXI-Stream)
//------------------------------------------------------------------------------

generate
  if (I2S_DWIDTH == AXI_DWIDTH) begin: gen_no_conv
    // No conversion needed - direct connection
    assign fifo_rd_en   = !fifo_empty && conv_ready;
    assign conv_valid   = fifo_rd_valid;
    assign conv_data    = fifo_rd_data;
    assign conv_keep    = {AXI_KWIDTH{1'b1}};
    
  end else if (I2S_DWIDTH < AXI_DWIDTH) begin: gen_upsize
    // Upsize from narrower I2S to wider AXI
    localparam RATIO = AXI_DWIDTH / I2S_DWIDTH;
    localparam CNT_WIDTH = $clog2(RATIO);
    
    logic [CNT_WIDTH-1:0]  sample_cnt;
    logic [AXI_DWIDTH-1:0] data_reg;
    logic [AXI_KWIDTH-1:0] keep_reg;
    logic                  data_valid;
    
    always_ff @(posedge i_axi_clk) begin
      if (i_axi_rst) begin
        sample_cnt <= '0;
        data_reg   <= '0;
        keep_reg   <= '0;
        data_valid <= 1'b0;
      end else begin
        if (fifo_rd_valid && fifo_rd_en) begin
          data_reg   <= {data_reg[AXI_DWIDTH-I2S_DWIDTH-1:0], fifo_rd_data};
          keep_reg   <= {keep_reg[AXI_KWIDTH-I2S_DWIDTH/8-1:0], {(I2S_DWIDTH/8){1'b1}}};
          sample_cnt <= sample_cnt + 1'b1;
          if (sample_cnt == (RATIO - 1)) begin
            data_valid <= 1'b1;
            sample_cnt <= '0;
          end
        end else if (conv_valid && conv_ready) begin
          data_valid <= 1'b0;
          keep_reg   <= '0;
        end
      end
    end
    
    assign fifo_rd_en = !fifo_empty && (!data_valid || conv_ready);
    assign conv_valid = data_valid;
    assign conv_data  = data_reg;
    assign conv_keep  = keep_reg;
    
  end else begin: gen_downsize
    // Downsize from wider I2S to narrower AXI
    localparam RATIO = I2S_DWIDTH / AXI_DWIDTH;
    localparam CNT_WIDTH = $clog2(RATIO);
    
    logic [CNT_WIDTH-1:0]  sample_cnt;
    logic [I2S_DWIDTH-1:0] data_reg;
    logic                  data_valid;
    logic                  data_available;
    
    always_ff @(posedge i_axi_clk) begin
      if (i_axi_rst) begin
        sample_cnt     <= '0;
        data_reg       <= '0;
        data_valid     <= 1'b0;
        data_available <= 1'b0;
      end else begin
        if (fifo_rd_valid && fifo_rd_en) begin
          data_reg       <= fifo_rd_data;
          data_available <= 1'b1;
          sample_cnt     <= '0;
        end else if (conv_valid && conv_ready) begin
          data_reg   <= data_reg >> AXI_DWIDTH;
          sample_cnt <= sample_cnt + 1'b1;
          if (sample_cnt == (RATIO - 1)) begin
            data_available <= 1'b0;
          end
        end
        
        data_valid <= data_available;
      end
    end
    
    assign fifo_rd_en = !fifo_empty && !data_available;
    assign conv_valid = data_valid;
    assign conv_data  = data_reg[AXI_DWIDTH-1:0];
    assign conv_keep  = {AXI_KWIDTH{1'b1}};
    
  end
endgenerate

//------------------------------------------------------------------------------
// Packet Formation
//------------------------------------------------------------------------------

  always_ff @(posedge i_axi_clk) begin
    if (i_axi_rst) begin
      sample_count <= '0;
      pkt_last     <= 1'b0;
    end else begin
      if (conv_valid && conv_ready) begin
        if (sample_count >= (PKT_SIZE - 1)) begin
          sample_count <= '0;
          pkt_last     <= 1'b1;
        end else begin
          sample_count <= sample_count + 1'b1;
          pkt_last     <= 1'b0;
        end
      end else begin
        pkt_last <= 1'b0;
      end
    end
  end

//------------------------------------------------------------------------------
// AXI-Stream Output
//------------------------------------------------------------------------------

  assign o_axis_tvalid = conv_valid;
  assign o_axis_tdata  = conv_data;
  assign o_axis_tkeep  = conv_keep;
  assign o_axis_tlast  = pkt_last;
  assign conv_ready    = i_axis_tready;

//------------------------------------------------------------------------------
// Overrun Detection
//------------------------------------------------------------------------------

  // Detect overrun condition
  always_ff @(posedge i_ref_clk) begin
    if (i_i2s_rst) begin
      overrun_flag <= 1'b0;
    end else begin
      if (i_i2s_valid && fifo_full) begin
        overrun_flag <= 1'b1;
      end
      // Clear overrun flag when FIFO has space again
      if (!fifo_afull) begin
        overrun_flag <= 1'b0;
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
  assign o_fifo_count  = fifo_wr_count; // Use write side count for I2S domain
  assign o_overrun     = overrun_flag;

endmodule 
