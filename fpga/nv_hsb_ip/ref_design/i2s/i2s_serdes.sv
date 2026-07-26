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
// I2S Serializer/Deserializer Module
//
// This module handles the actual I2S serial protocol for both transmit and
// receive operations. It supports multiple I2S formats and bit depths.
//
// All formats use MSB-first bit ordering. bit_count=0 aligns with the
// LRCLK transition (raw edges used for TX sync). Format selection
// determines where the data window sits within each slot:
//
//   I2S: [1, active_bits+1)  -- explicit 1-BCLK offset after LRCLK
//   LJM: [0, active_bits)    -- data starts at LRCLK edge
//   RJM: [slot_width-active_bits, slot_width)  -- right-aligned in slot
//
// When the slot is an exact fit (active_bits >= slot_width) in I2S
// mode, the LSB overflows into bit_count=0 of the next channel per the
// Philips I2S spec. TX saves the overflow bit and drives it at count=0;
// RX captures it from rx_sdata_reg and inserts it at bit[0] of the
// completed channel word.
//
// Features:
// - Variable bit depths (8, 16, 24, 32 bits)
// - Stereo and mono support
// - Configurable data formats
// - Simultaneous TX and RX operation
//------------------------------------------------------------------------------
 
module i2s_serdes
  import i2s_pkg::*;
#(
  parameter MAX_BITS = 32               // Maximum supported bit depth
)(
  //----------------------------------------------------------------------------
  // Clock and Reset
  //----------------------------------------------------------------------------
  input                    i_ref_clk,   // Reference clock
  input                    i_bclk,      // I2S bit clock
  input                    i_lrclk,     // I2S left/right clock (word select)
  input                    i_rst,       // Reset (active high)
  input                    i_bclk_en,
  input                    i_lrclk_en,
  
  //----------------------------------------------------------------------------
  // Configuration
  //----------------------------------------------------------------------------
  input  [1:0]             i_bit_depth,     // Bit depth selection
  input  [1:0]             i_data_format,   // Data format selection
  input                    i_channel_mode,  // Channel mode (0=stereo, 1=mono)
  input                    i_slot_mode,     // 0=fixed 32-BCLK slots, 1=word-size slots
  input                    i_lrclk_polarity,// LRCLK polarity
  input                    i_tx_enable,     // TX enable
  input                    i_rx_enable,     // RX enable
  
  //----------------------------------------------------------------------------
  // TX Path (parallel to serial)
  //----------------------------------------------------------------------------
  input                    i_tx_valid,      // TX data valid
  input                    i_tx_ready,      // TX ready (buffer has data)
  input                    i_tx_last,       // Last sample in current transfer (from tlast)
  input  [MAX_BITS-1:0]    i_tx_data_l,     // Left channel TX data
  input  [MAX_BITS-1:0]    i_tx_data_r,     // Right channel TX data
  output                   o_tx_ready,      // TX ready for next data
  output    logic          o_sdata_tx,      // Serial data output
  
  //----------------------------------------------------------------------------
  // RX Path (serial to parallel)
  //----------------------------------------------------------------------------
  input                    i_sdata_rx,      // Serial data input
  output                   o_rx_valid,      // RX data valid
  output [MAX_BITS-1:0]    o_rx_data,       // RX data (stereo packed: {R,L})
  input                    i_rx_ready       // RX ready to accept data
);
 
//------------------------------------------------------------------------------
// Configuration Decode
//------------------------------------------------------------------------------
 
  logic [5:0]              active_bits;

  always_comb begin
    case (i_bit_depth)
      I2S_BD_8B:   active_bits = 6'd8;
      I2S_BD_16B:  active_bits = 6'd16;
      I2S_BD_24B:  active_bits = 6'd24;
      I2S_BD_32B:  active_bits = 6'd32;
      default:     active_bits = 6'd16;
    endcase
  end

  logic [5:0] slot_width;
  assign slot_width = (i_slot_mode == I2S_SLOT_WORD) ? active_bits : MAX_BITS[5:0];

  logic i2s_exact_fit;
  assign i2s_exact_fit = (active_bits >= slot_width) &&
                         (i_data_format == I2S_FMT_I2S);

  logic [5:0] data_start;
  logic [5:0] data_end;

  // TX_WAIT_LRCLK uses raw (un-pipelined) LRCLK edges so bit_count=0
  // aligns with the LRCLK transition itself. The I2S 1-BCLK offset is
  // provided explicitly via data_start=1 in the window logic below.

  always_comb begin
    case (i_data_format)
      I2S_FMT_I2S: begin
        data_start = 6'd1;
        data_end   = (active_bits < slot_width) ? active_bits + 6'd1 : slot_width;
      end
      I2S_FMT_LJ: begin
        data_start = 6'd0;
        data_end   = active_bits;
      end
      I2S_FMT_RJ: begin
        data_start = slot_width - active_bits;
        data_end   = slot_width;
      end
      default: begin
        data_start = 6'd0;
        data_end   = active_bits;
      end
    endcase
  end
 
//------------------------------------------------------------------------------
// Internal Signals
//------------------------------------------------------------------------------
 
  // Clock edge detection
  logic                    bclk_posedge;
  logic                    bclk_negedge;
  logic                    lrclk_posedge;
  logic                    lrclk_posedge_r;
  logic                    lrclk_negedge;
  logic                    lrclk_negedge_r;
  
  // TX signals
  logic [MAX_BITS-1:0]     tx_shift_reg_l;
  logic [MAX_BITS-1:0]     tx_shift_reg_r;
  logic [5:0]              tx_bit_count;
  logic                    tx_left_ch;
  logic                    tx_overflow_bit;
  logic                    tx_overflow_valid;

  // RX signals
  logic [MAX_BITS-1:0]     rx_shift_reg_l;
  logic [MAX_BITS-1:0]     rx_shift_reg_r;
  logic [5:0]              rx_bit_count;
  logic                    rx_left_ch;

  // Data window active flags
  logic tx_in_window;
  logic rx_in_window;
  assign tx_in_window = (tx_bit_count >= data_start) && (tx_bit_count < data_end);
  assign rx_in_window = (rx_bit_count >= data_start) && (rx_bit_count < data_end);
 
//------------------------------------------------------------------------------
// Clock Edge Detection
//------------------------------------------------------------------------------
 
  always_ff @(posedge i_ref_clk) begin
    if (i_rst) begin
      lrclk_posedge_r <= 1'b0;
      lrclk_negedge_r <= 1'b0;
    end
    else if (bclk_negedge) begin
      lrclk_posedge_r <= lrclk_posedge;
      lrclk_negedge_r <= lrclk_negedge;
    end
  end
  
  assign bclk_posedge  = i_bclk_en && !i_bclk;
  assign bclk_negedge  = i_bclk_en && i_bclk;
  assign lrclk_posedge = i_lrclk_en && !i_lrclk;
  assign lrclk_negedge = i_lrclk_en && i_lrclk;
 
//------------------------------------------------------------------------------
// TX State Machine
//------------------------------------------------------------------------------

 
  typedef enum logic [2:0] {
    TX_IDLE,
    TX_LOAD_DATA,
    TX_WAIT_LRCLK,
    TX_LEFT_CH,
    TX_RIGHT_CH
  } tx_state_t;
  
  tx_state_t tx_state, tx_state_next, tx_state_prev;
  
  always_ff @(posedge i_ref_clk) begin
    if (i_rst || !i_tx_enable) begin
      tx_state <= TX_IDLE;
      tx_state_prev <= TX_IDLE;
    end else if (bclk_negedge) begin
      tx_state <= tx_state_next;
      tx_state_prev <= tx_state;
    end
  end
  

  // Latch i_tx_last when a sample is loaded so it's stable during serialization
  logic tx_last_r;
  // One-cycle delayed: FSM exit from TX_RIGHT_CH must not see tx_last_r same cycle it is set at preload
  logic tx_last_r_d;

  always_ff @(posedge i_ref_clk) begin
    if (i_rst || !i_tx_enable) begin
      tx_last_r   <= 1'b0;
      tx_last_r_d <= 1'b0;
    end else if (bclk_negedge) begin
      tx_last_r_d <= tx_last_r;
      if ((tx_state == TX_LOAD_DATA && i_tx_valid) ||
          (tx_state == TX_RIGHT_CH && tx_bit_count >= (slot_width - 1) && !tx_last_r))
        tx_last_r <= i_tx_last;
    end
  end

  always_comb begin
    tx_state_next = tx_state;
    
    case (tx_state)
      TX_IDLE: begin
        if (i_tx_enable && i_tx_ready) begin
          tx_state_next = TX_LOAD_DATA;
        end
      end
      
      TX_LOAD_DATA: begin
        if (i_tx_valid) begin
          tx_state_next = TX_WAIT_LRCLK;
        end
      end
      
      TX_WAIT_LRCLK: begin
        if (i_lrclk_polarity) begin
          if (lrclk_posedge) begin
            tx_state_next = TX_LEFT_CH;
          end
        end
        else begin
          if (lrclk_negedge) begin
            tx_state_next = TX_LEFT_CH;
          end
        end
      end
      
      TX_LEFT_CH: begin
        if (tx_bit_count >= (slot_width - 1))
          tx_state_next = TX_RIGHT_CH;
      end
      
      TX_RIGHT_CH: begin
        if (tx_bit_count >= (slot_width - 1)) begin
          if (tx_last_r_d) begin
            tx_state_next = TX_LOAD_DATA;
          end else begin
            tx_state_next = TX_LEFT_CH;
          end
        end
      end
    endcase
  end
 
//------------------------------------------------------------------------------
// TX Data Path
//------------------------------------------------------------------------------
 
  always_ff @(posedge i_ref_clk) begin
    if (i_rst || !i_tx_enable) begin
      tx_shift_reg_l  <= '0;
      tx_shift_reg_r  <= '0;
      tx_bit_count    <= '0;
      tx_left_ch      <= 1'b1;
      tx_overflow_bit   <= 1'b0;
      tx_overflow_valid <= 1'b0;
    end else if (bclk_negedge) begin
      case (tx_state)
        TX_IDLE: begin
          tx_bit_count <= '0;
          tx_left_ch   <= 1'b1;
        end
        
        TX_LOAD_DATA: begin
          if (i_tx_valid) begin
            tx_shift_reg_l <= i_tx_data_l << (MAX_BITS - active_bits);
            tx_shift_reg_r <= i_tx_data_r << (MAX_BITS - active_bits);
          end
        end
        
        TX_WAIT_LRCLK: begin
          if (i_lrclk_polarity) begin
            if (lrclk_posedge) begin
              tx_bit_count <= '0;
              tx_left_ch   <= 1'b1;
            end
          end else begin
            if (lrclk_negedge) begin
              tx_bit_count <= '0;
              tx_left_ch   <= 1'b1;
            end
          end
        end
        
        TX_LEFT_CH: begin
          if (tx_bit_count < (slot_width - 1)) begin
            if (tx_in_window)
              tx_shift_reg_l <= tx_shift_reg_l << 1;
            tx_bit_count <= tx_bit_count + 1'b1;
          end else begin
            if (i2s_exact_fit) begin
              tx_overflow_bit   <= tx_shift_reg_l[MAX_BITS-2];
              tx_overflow_valid <= 1'b1;
            end
            tx_bit_count <= '0;
            tx_left_ch   <= 1'b0;
          end
        end
        
        TX_RIGHT_CH: begin
          if (tx_bit_count < (slot_width - 1)) begin
            if (tx_in_window)
              tx_shift_reg_r <= tx_shift_reg_r << 1;
            tx_bit_count <= tx_bit_count + 1'b1;
          end else begin
            if (i2s_exact_fit) begin
              tx_overflow_bit   <= tx_shift_reg_r[MAX_BITS-2];
              tx_overflow_valid <= 1'b1;
            end
            tx_bit_count <= '0;
            tx_left_ch   <= 1'b1;
            tx_shift_reg_l <= i_tx_data_l << (MAX_BITS - active_bits);
            tx_shift_reg_r <= i_tx_data_r << (MAX_BITS - active_bits);
          end
        end
      endcase
    end
  end
 
//------------------------------------------------------------------------------
// TX Output Assignment
//------------------------------------------------------------------------------
 
  always_comb begin
    if (!i_tx_enable || (tx_state == TX_IDLE) || (tx_state == TX_LOAD_DATA) || (tx_state == TX_WAIT_LRCLK)) begin
      o_sdata_tx = 1'b0;
    end else if (tx_bit_count == 0 && i2s_exact_fit && tx_overflow_valid) begin
      o_sdata_tx = tx_overflow_bit;
    end else if (i_channel_mode == I2S_CH_MONO && !tx_left_ch) begin
      o_sdata_tx = 1'b0;
    end else if (!tx_in_window) begin
      o_sdata_tx = 1'b0;
    end else if (tx_left_ch) begin
      o_sdata_tx = tx_shift_reg_l[MAX_BITS-1];
    end else begin
      o_sdata_tx = tx_shift_reg_r[MAX_BITS-1];
    end
  end

  // Pop the TX buffer at LEFT_CH->RIGHT_CH (not RIGHT_CH->LEFT_CH) so the
  // buffer has the entire RIGHT_CH duration to advance the FIFO and latch the
  // new sample before the serdes preloads at the RIGHT_CH->LEFT_CH boundary.
  assign o_tx_ready = (tx_state == TX_LOAD_DATA && tx_state_prev != TX_IDLE) || (tx_state == TX_LEFT_CH && tx_state_next == TX_RIGHT_CH);
 
//------------------------------------------------------------------------------
// RX State Machine
//------------------------------------------------------------------------------
 
  typedef enum logic [1:0] {
    RX_IDLE,
    RX_WAIT_LRCLK,
    RX_LEFT_CH,
    RX_RIGHT_CH
  } rx_state_t;
  
  rx_state_t rx_state, rx_state_next, rx_state_d1;

  always_ff @(posedge i_ref_clk) begin
    if (i_rst || !i_rx_enable) begin
      rx_state <= RX_IDLE;
    end
    else if (bclk_posedge) begin
      rx_state <= rx_state_next;
    end
  end

  // Delayed FSM state: detect first cycle after channel-complete transitions for parallel export
  always_ff @(posedge i_ref_clk) begin
    if (i_rst || !i_rx_enable) begin
      rx_state_d1 <= RX_IDLE;
    end else if (bclk_posedge) begin
      rx_state_d1 <= rx_state;
    end
  end
  
  always_comb begin
    rx_state_next = rx_state;
    
    case (rx_state)
      RX_IDLE: begin
        if (i_rx_enable) begin
          rx_state_next = RX_WAIT_LRCLK;
        end
      end
      
      RX_WAIT_LRCLK: begin
        if (i_lrclk_polarity) begin
          if (lrclk_posedge_r) begin
            rx_state_next = RX_LEFT_CH;
          end
        end else begin
          if (lrclk_negedge_r) begin
            rx_state_next = RX_LEFT_CH;
          end
        end
      end
      
      RX_LEFT_CH: begin
        if (rx_bit_count >= (slot_width - 1))
          rx_state_next = RX_RIGHT_CH;
      end
      
      RX_RIGHT_CH: begin
        if (rx_bit_count >= (slot_width - 1))
          rx_state_next = RX_LEFT_CH;
      end
    endcase
  end
 
//------------------------------------------------------------------------------
// RX Data Path
//------------------------------------------------------------------------------
 
  logic rx_sdata_reg;
  always_ff @(posedge i_ref_clk) begin
    if (i_rst || !i_rx_enable) begin
      rx_sdata_reg <= 1'b0;
    end else if (bclk_posedge) begin
      rx_sdata_reg <= i_sdata_rx;
    end
  end
 
  always_ff @(posedge i_ref_clk) begin
    if (i_rst || !i_rx_enable) begin
      rx_shift_reg_l <= '0;
      rx_shift_reg_r <= '0;
      rx_bit_count   <= '0;
      rx_left_ch     <= 1'b1;
    end 
    else if (bclk_posedge) begin
      case (rx_state)
        RX_IDLE: begin
          rx_bit_count <= '0;
          rx_left_ch   <= 1'b1;
        end
        
        RX_WAIT_LRCLK: begin
          if (i_lrclk_polarity) begin
            if (lrclk_posedge_r) begin
              rx_bit_count <= '0;
              rx_left_ch   <= 1'b1;
            end
          end else begin
            if (lrclk_negedge_r) begin
              rx_bit_count <= '0;
              rx_left_ch   <= 1'b1;
            end
          end
        end
        
        RX_LEFT_CH: begin
          if (rx_in_window)
            rx_shift_reg_l <= {rx_shift_reg_l[MAX_BITS-2:0], rx_sdata_reg};
          if (rx_bit_count >= (slot_width - 1)) begin
            rx_bit_count <= '0;
            rx_left_ch   <= 1'b0;
          end else begin
            rx_bit_count <= rx_bit_count + 1'b1;
          end
        end
        
        RX_RIGHT_CH: begin
          if (rx_in_window)
            rx_shift_reg_r <= {rx_shift_reg_r[MAX_BITS-2:0], rx_sdata_reg};
          if (rx_bit_count >= (slot_width - 1)) begin
            rx_bit_count <= '0;
            rx_left_ch   <= 1'b1;
          end else begin
            rx_bit_count <= rx_bit_count + 1'b1;
          end
        end
      endcase
    end
  end
 
  logic [MAX_BITS-1:0]     rx_data_mask;
  logic [MAX_BITS-1:0]     rx_data;
  logic                    rx_data_valid;
  always_comb begin
    case (i_bit_depth)
      I2S_BD_8B:   rx_data_mask = 8'hFF;
      I2S_BD_16B:  rx_data_mask = 16'hFFFF;
      I2S_BD_24B:  rx_data_mask = 24'hFFFFFF;
      I2S_BD_32B:  rx_data_mask = 32'hFFFFFFFF;
      default:     rx_data_mask = 16'hFFFF;
    endcase
  end
  always_ff @(posedge i_ref_clk) begin
    if (i_rst || !i_rx_enable) begin
      rx_data       <= '0;
      rx_data_valid <= 1'b0;
    end else if (bclk_posedge) begin
      rx_data_valid <= 1'b0;

      if (rx_state == RX_RIGHT_CH && rx_state_d1 == RX_LEFT_CH) begin
        if (i2s_exact_fit)
          rx_data <= ((rx_shift_reg_l << 1) | {{(MAX_BITS-1){1'b0}}, rx_sdata_reg}) & rx_data_mask;
        else
          rx_data <= rx_shift_reg_l & rx_data_mask;
        rx_data_valid <= 1'b1;
      end else if (rx_state == RX_LEFT_CH && rx_state_d1 == RX_RIGHT_CH) begin
        if (i_channel_mode != I2S_CH_MONO) begin
          if (i2s_exact_fit)
            rx_data <= ((rx_shift_reg_r << 1) | {{(MAX_BITS-1){1'b0}}, rx_sdata_reg}) & rx_data_mask;
          else
            rx_data <= rx_shift_reg_r & rx_data_mask;
          rx_data_valid <= 1'b1;
        end
      end
    end
  end
 
//------------------------------------------------------------------------------
// RX Output Assignment
//------------------------------------------------------------------------------
 
  assign o_rx_valid = rx_data_valid;
  assign o_rx_data  = rx_data;
 
endmodule 
