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
  input  [1:0]             i_bit_depth_sel,    // Bit depth selection
  input                    i_clock_source_sel, // Clock source (0=internal, 1=external)
  input                    i_enable,            // Clock generation enable
  
  //----------------------------------------------------------------------------
  // External Clock Inputs 
  //----------------------------------------------------------------------------
  input                    i_ext_mclk,      // External mclk
  input                    i_ext_bclk,      // External bit clock
  
  //----------------------------------------------------------------------------
  // Generated Clock Outputs
  //----------------------------------------------------------------------------
  output                   o_mclk,          // mclk output
  output                   o_bclk,          // Bit clock output
  output                   o_lrclk,         // Left/Right clock output
  output                   o_locked,        // CLK locked indicator

  output                   o_mclk_en,       // MCLK enable
  output                   o_bclk_en,       // BCLK enable
  output                   o_bclk_posedge,  // BCLK positive edge
  output                   o_bclk_negedge,  // BCLK negative edge
  output                   o_lrclk_en,      // LRCLK enable
  output                   o_lrclk_negedge, // LRCLK negative edge
  output                   o_lrclk_posedge  // LRCLK positive edge
);

//------------------------------------------------------------------------------
// Clock Generation Parameters
//------------------------------------------------------------------------------

  // Calculate required frequencies based on configuration
  logic [5:0]              target_bit_depth;
  
  // Clock dividers for internal generation
  logic [15:0]             mclk_div;
  logic [7:0]              bclk_div;
  logic [7:0]              lrclk_div;
  
  // Internal clock signals
  logic                    int_mclk;
  logic                    int_bclk;
  logic                    int_lrclk;
  logic                    clk_locked;
  logic                    mclk_en;
  logic                    bclk_en;
  logic                    lrclk_en;
  
  // Clock selection outputs
  logic                    sel_mclk;
  logic                    sel_bclk;
  logic                    sel_lrclk;

//------------------------------------------------------------------------------
// Configuration Decode
//------------------------------------------------------------------------------


  // Decode bit depth from selection
  always_comb begin
    case (i_bit_depth_sel)
      I2S_BD_8B:   target_bit_depth = 6'd8;
      I2S_BD_16B:  target_bit_depth = 6'd16;
      I2S_BD_24B:  target_bit_depth = 6'd24;
      I2S_BD_32B:  target_bit_depth = 6'd32;
      default:     target_bit_depth = 6'd16;        // Default to 16-bit
    endcase
  end


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
      8'd1: lrclk_div = target_bit_depth << 1;
      8'd2: lrclk_div = target_bit_depth << 2;
      8'd3: lrclk_div = target_bit_depth << 3;
      8'd4: lrclk_div = target_bit_depth << 4;
      8'd5: lrclk_div = target_bit_depth << 5;
      8'd6: lrclk_div = target_bit_depth << 6;
      8'd7: lrclk_div = target_bit_depth << 7;
      8'd8: lrclk_div = target_bit_depth << 8;
      default: lrclk_div = target_bit_depth;
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
  logic clk_locked_r;

  always_ff @(posedge i_ref_clk) begin

    if (i_rst) begin
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
  logic [7:0]              lrclk_counter;
  logic                    lrclk_int;
  logic                    lrclk_ext;

  always_ff @(posedge i_ref_clk) begin
    if (i_rst) begin
      lrclk_counter <= 8'b0;
      lrclk_int     <= 1'b0;
      lrclk_en      <= 1'b0;
    end else if (mclk_en) begin
      if (lrclk_counter >= ((lrclk_div - 1))) begin
        lrclk_counter <= 8'b0;
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

  assign o_lrclk_negedge = lrclk_int && bclk_en && !lrclk_en && (lrclk_counter == (lrclk_div - 1));
  assign o_lrclk_posedge = !lrclk_int && bclk_en && !lrclk_en && (lrclk_counter == (lrclk_div - 1));


  assign int_lrclk = (lrclk_div == 8'd1) ? int_bclk : lrclk_int;

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

  // Select between internal and external clocks
  always_comb begin
    if (i_clock_source_sel == I2S_CLK_EXT) begin
      // External clock mode 
      sel_mclk = i_ext_mclk;
      sel_bclk = i_ext_bclk;
      // sel_lrclk = i_ext_lrclk;
    end else begin
      // Internal clock mode 
      // sel_mclk  = clk_locked ? int_mclk_buffered : 1'b0;
      sel_mclk = int_mclk_buffered;
      sel_bclk = int_bclk_buffered;
    end
  end

//------------------------------------------------------------------------------
// Output Assignments
//------------------------------------------------------------------------------

  assign o_mclk   = sel_mclk & i_enable;
  assign o_bclk   = sel_bclk & i_enable;
  assign o_lrclk  = int_lrclk_buffered & i_enable;
  assign o_bclk_posedge = !bclk_int && bclk_en ;
  assign o_bclk_negedge =  bclk_int && bclk_en;
  // assign o_locked = clk_locked;
  assign o_locked = 1'b1;
  assign o_mclk_en = mclk_en;
  assign o_bclk_en = bclk_en;
  assign o_lrclk_en = lrclk_en;

//------------------------------------------------------------------------------
// Synthesis Directives and Assertions
//------------------------------------------------------------------------------

  // Ensure clock dividers are reasonable
 //  `ifdef ASSERT_ON
 //    always_comb begin
 //      assert (!i_enable || mclk_div > 0) else $error("MCLK divider cannot be zero");
 //      assert (!i_enable || bclk_div > 0) else $error("BCLK divider cannot be zero");
 //      assert (!i_enable || lrclk_div > 0) else $error("LRCLK divider cannot be zero");
 //    end
 //  `endif

  // Keep critical clocks from being optimized away
  (* keep = "true" *) logic keep_mclk = int_mclk;
  (* keep = "true" *) logic keep_bclk = int_bclk;
  (* keep = "true" *) logic keep_lrclk = int_lrclk;
  
  // Keep buffered clocks from being optimized away
  (* keep = "true" *) logic keep_buffered_mclk  = int_mclk_buffered;
  (* keep = "true" *) logic keep_buffered_bclk  = int_bclk_buffered;
  (* keep = "true" *) logic keep_buffered_lrclk = int_lrclk_buffered;

endmodule 
