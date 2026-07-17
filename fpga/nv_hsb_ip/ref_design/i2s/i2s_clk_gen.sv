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
// I2S Clock Generation Module
//
// This module generates the required I2S clocks (MCLK, BCLK, LRCLK) from a
// reference clock based on the configured sample rate and bit depth.
// 
// Clock relationships:
// - MCLK = 256 * LRCLK (typical for I2S)
// - BCLK = 2 * bit_depth * LRCLK (bit clock)
// - LRCLK = sample_rate (left/right clock, word select)
//------------------------------------------------------------------------------

module i2s_clk_gen
  import i2s_pkg::*;
(
  //----------------------------------------------------------------------------
  // Clock and Reset
  //----------------------------------------------------------------------------
  input                    i_ref_clk,       // Reference clock input
  input                    i_rst,           // Reset (active high)
  
  //----------------------------------------------------------------------------
  // Configuration
  //----------------------------------------------------------------------------
  input  [31:0]            i_mclk_div,      // MCLK divider
  input  [31:0]            i_bclk_div,      // BCLK divider
  input  [1:0]             i_bit_depth_sel, // Bit depth selection
  input                    i_slot_mode,     // 0=fixed 32-BCLK slots, 1=word-size slots
  input                    i_enable,        // Clock generation enable
  
  //----------------------------------------------------------------------------
  // Generated Clock Outputs
  //----------------------------------------------------------------------------
  output                   o_mclk,          // mclk output
  output                   o_bclk,          // Bit clock output
  output                   o_lrclk,         // Left/Right clock output
  output                   o_locked,        // CLK locked indicator

  output                   o_bclk_en,       // BCLK enable
  output                   o_bclk_posedge,  // BCLK positive edge
  output                   o_bclk_negedge,  // BCLK negative edge
  output                   o_lrclk_en       // LRCLK enable
);

//------------------------------------------------------------------------------
// Clock Generation Parameters
//------------------------------------------------------------------------------

  localparam [5:0] SLOT_FIXED = 6'd32;

  logic [5:0] target_bit_depth;
  logic [5:0] slot_width;

  always_comb begin
    case (i_bit_depth_sel)
      I2S_BD_8B:   target_bit_depth = 6'd8;
      I2S_BD_16B:  target_bit_depth = 6'd16;
      I2S_BD_24B:  target_bit_depth = 6'd24;
      I2S_BD_32B:  target_bit_depth = 6'd32;
      default:     target_bit_depth = 6'd16;
    endcase
  end

  assign slot_width = (i_slot_mode == I2S_SLOT_WORD) ? target_bit_depth : SLOT_FIXED;

  // Clock dividers for internal generation
  logic [15:0]             mclk_div;
  logic [7:0]              bclk_div;
  logic [15:0]             lrclk_div;
  
  // Internal clock signals
  logic                    int_mclk;
  logic                    int_bclk;
  logic                    int_lrclk;
  logic                    mclk_en;
  logic                    bclk_en;
  logic                    lrclk_en;
  
  // Clock selection outputs
  logic                    sel_mclk;
  logic                    sel_bclk;

//------------------------------------------------------------------------------
// Clock Divider Calculation
//------------------------------------------------------------------------------
  always_comb begin
    mclk_div = i_mclk_div;
    case (i_bclk_div)
      8'd1: bclk_div = 8'd2;
      8'd2: bclk_div = 8'd4;
      8'd3: bclk_div = 8'd8;
      8'd4: bclk_div = 8'd16;
      8'd5: bclk_div = 8'd32;
      8'd6: bclk_div = 8'd64;
      8'd7: bclk_div = 8'd128;
      8'd8: bclk_div = 8'd256;
      default: bclk_div = 8'd1;
    endcase
    case (i_bclk_div)
      8'd1: lrclk_div = 16'(slot_width) << 1;
      8'd2: lrclk_div = 16'(slot_width) << 2;
      8'd3: lrclk_div = 16'(slot_width) << 3;
      8'd4: lrclk_div = 16'(slot_width) << 4;
      8'd5: lrclk_div = 16'(slot_width) << 5;
      8'd6: lrclk_div = 16'(slot_width) << 6;
      8'd7: lrclk_div = 16'(slot_width) << 7;
      8'd8: lrclk_div = 16'(slot_width) << 8;
      default: lrclk_div = 16'(slot_width);
    endcase
  end

//------------------------------------------------------------------------------
// Internal Clock Generation
//------------------------------------------------------------------------------

  // MCLK Generation (divide from reference clock)
  logic [15:0]             mclk_counter;
  logic                    mclk_int;

  always_ff @(posedge i_ref_clk) begin
    if (i_rst || !i_enable) begin
      mclk_counter <= 16'b0;
      mclk_int     <= 1'b0;
      mclk_en      <= 1'b0;
    end else begin
      if (mclk_counter >= ((mclk_div >> 1)-1)) begin
        mclk_counter <= 16'b0;
        mclk_int     <= ~mclk_int;
        if (mclk_counter == ((mclk_div >> 1)-1) && !mclk_en) begin
          mclk_en <= 1'b1;
        end else begin
          mclk_en <= 1'b0;
        end
      end else begin
        mclk_counter <= mclk_counter + 1'b1;
        mclk_en      <= 1'b0;
      end
    end
  end
    
  assign int_mclk = (mclk_div == 16'd1) ? i_ref_clk : mclk_int;

  // BCLK Generation (divide from MCLK)
  logic [7:0]              bclk_counter;
  logic                    bclk_int;

  always_ff @(posedge i_ref_clk) begin

    if (i_rst || !i_enable) begin
      bclk_counter <= 8'b0;
      bclk_int     <= 1'b0;
      bclk_en      <= 1'b0;
    end else if (mclk_en) begin
      if (bclk_counter >= ((bclk_div >> 1)-1)) begin
        bclk_counter <= 8'b0;
        bclk_int     <= ~bclk_int;
        if (bclk_counter == ((bclk_div >> 1)-1) && !bclk_en) begin
          bclk_en <= 1'b1;
        end else begin
          bclk_en <= 1'b0;
        end
      end else begin
        bclk_counter <= bclk_counter + 1'b1;
        bclk_en      <= 1'b0;
      end
    end
    else begin
      bclk_en <= 1'b0;
    end
  end


  assign int_bclk = (bclk_div == 8'd1) ? int_mclk : bclk_int;


  // LRCLK Generation (divide from BCLK)
  logic [15:0]             lrclk_counter;
  logic                    lrclk_int;

  always_ff @(posedge i_ref_clk) begin
    if (i_rst || !i_enable) begin
      lrclk_counter <= '0;
      lrclk_int     <= 1'b0;
      lrclk_en      <= 1'b0;
    end else if (mclk_en) begin
      if (lrclk_counter >= ((lrclk_div - 1))) begin
        lrclk_counter <= '0;
        lrclk_int     <= ~lrclk_int;
        if (lrclk_counter == (lrclk_div - 1) && !lrclk_en) begin
          lrclk_en <= 1'b1;
        end else begin
          lrclk_en <= 1'b0;
        end
      end else begin
        lrclk_counter <= lrclk_counter + 1'b1;
        lrclk_en      <= 1'b0;
      end
    end
    else begin
      lrclk_en <= 1'b0;
    end 
  end

  assign int_lrclk = (lrclk_div == 16'd1) ? int_bclk : lrclk_int;

//------------------------------------------------------------------------------
// Clock Buffering for Better Distribution
//------------------------------------------------------------------------------

  // Buffered clock signals for improved distribution
  logic                    int_mclk_buffered;
  logic                    int_bclk_buffered;
  logic                    int_lrclk_buffered;
  
  // Clock buffers to reduce skew and improve fanout
  // These help ensure consistent timing across all clock domains
  always_ff @(posedge i_ref_clk) begin
    if (i_rst) begin
      int_mclk_buffered  <= 1'b0;
      int_bclk_buffered  <= 1'b0;
      int_lrclk_buffered <= 1'b0;
    end else begin
      int_mclk_buffered  <= int_mclk;
      int_bclk_buffered  <= int_bclk;
      int_lrclk_buffered <= int_lrclk;
    end
  end

//------------------------------------------------------------------------------
// Clock Source Selection
//------------------------------------------------------------------------------

  assign sel_mclk = int_mclk_buffered;
  assign sel_bclk = int_bclk_buffered;

//------------------------------------------------------------------------------
// Output Assignments
//------------------------------------------------------------------------------

  assign o_mclk   = sel_mclk & i_enable;
  assign o_bclk   = sel_bclk & i_enable;
  assign o_lrclk  = int_lrclk_buffered & i_enable;
  assign o_bclk_posedge = !bclk_int && bclk_en ;
  assign o_bclk_negedge =  bclk_int && bclk_en;
  assign o_locked = 1'b1;
  assign o_bclk_en = bclk_en;
  assign o_lrclk_en = lrclk_en;

//------------------------------------------------------------------------------
// Synthesis Directives
//------------------------------------------------------------------------------

  // Keep critical clocks from being optimized away
  (* keep = "true" *) logic keep_mclk = int_mclk;
  (* keep = "true" *) logic keep_bclk = int_bclk;
  (* keep = "true" *) logic keep_lrclk = int_lrclk;
  
  // Keep buffered clocks from being optimized away
  (* keep = "true" *) logic keep_buffered_mclk  = int_mclk_buffered;
  (* keep = "true" *) logic keep_buffered_bclk  = int_bclk_buffered;
  (* keep = "true" *) logic keep_buffered_lrclk = int_lrclk_buffered;

endmodule 
