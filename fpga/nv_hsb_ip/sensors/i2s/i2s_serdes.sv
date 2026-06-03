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
// Supported formats:
// - Standard I2S (MSB justified with 1 BCLK delay)
// - Left Justified (MSB justified with no delay)
// - Right Justified (LSB justified)
// - PCM/DSP Mode (MSB justified with no delay, single clock wide WS)
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
  input                    i_mclk_en,
  input                    i_bclk_en,
  input                    i_lrclk_en,
  input                    i_lrclk_negedge,
  input                    i_lrclk_posedge,
  
  //----------------------------------------------------------------------------
  // Configuration
  //----------------------------------------------------------------------------
  input  [1:0]             i_bit_depth,     // Bit depth selection
  input  [1:0]             i_data_format,   // Data format selection
  input                    i_channel_mode,  // Channel mode (0=stereo, 1=mono)
  input                    i_lrclk_polarity,// LRCLK polarity
  input                    i_tx_enable,     // TX enable
  input                    i_rx_enable,     // RX enable
  
  //----------------------------------------------------------------------------
  // TX Path (parallel to serial)
  //----------------------------------------------------------------------------
  input                    i_tx_valid,      // TX data valid
  input                    i_tx_ready,      // TX ready
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
 
  logic [5:0]              active_bits;     // Active bit depth
  logic                    is_i2s_format;   // Standard I2S format
  logic                    is_lj_format;    // Left justified format
  logic                    is_rj_format;    // Right justified format
  logic                    is_pcm_format;   // PCM/DSP format
  
  // Decode bit depth
  always_comb begin
    case (i_bit_depth)
      I2S_BD_8B:   active_bits = 6'd8;
      I2S_BD_16B:  active_bits = 6'd16;
      I2S_BD_24B:  active_bits = 6'd24;
      I2S_BD_32B:  active_bits = 6'd32;
      default:     active_bits = 6'd16;
    endcase
  end
  
  // Decode data format
  assign is_i2s_format = (i_data_format == I2S_FMT_I2S);
  assign is_lj_format  = (i_data_format == I2S_FMT_LJ);
  assign is_rj_format  = (i_data_format == I2S_FMT_RJ);
  assign is_pcm_format = (i_data_format == I2S_FMT_PCM);
 
//------------------------------------------------------------------------------
// Internal Signals
//------------------------------------------------------------------------------
 
  // Clock edge detection
  logic                    bclk_posedge;
  logic                    bclk_negedge;
  logic                    lrclk_posedge;
  logic                    lrclk_posedge_r;
  logic                    lrclk_posedge_r1;
  logic                    lrclk_posedge_r2;
  logic                    lrclk_negedge;
  logic                    lrclk_negedge_r;
  logic                    lrclk_negedge_r1;
  logic                    lrclk_negedge_r2;
  
  // TX signals
  logic [MAX_BITS-1:0]     tx_shift_reg_l;  // Left channel shift register
  logic [MAX_BITS-1:0]     tx_shift_reg_r;  // Right channel shift register
  logic [5:0]              tx_bit_count;    // TX bit counter
  logic                    tx_left_ch;      // Currently transmitting left channel
  logic                    tx_data_req;     // Request new TX data
  logic                    tx_active;       // TX operation active
  
  // RX signals
  logic [MAX_BITS-1:0]     rx_shift_reg_l;  // Left channel shift register
  logic [MAX_BITS-1:0]     rx_shift_reg_r;  // Right channel shift register
  logic [5:0]              rx_bit_count;    // RX bit counter
  logic                    rx_left_ch;      // Currently receiving left channel
  logic                    rx_data_valid;   // RX data valid
  logic                    rx_active;       // RX operation active
 
//------------------------------------------------------------------------------
// Clock Edge Detection
//------------------------------------------------------------------------------
 
  always_ff @(posedge i_ref_clk) begin
    if (i_rst) begin
      lrclk_posedge_r <= 1'b0;
      lrclk_negedge_r <= 1'b0;
      lrclk_posedge_r1 <= 1'b0;
      lrclk_negedge_r1 <= 1'b0;
      lrclk_posedge_r2 <= 1'b0;
      lrclk_negedge_r2 <= 1'b0;
    end
    else if (bclk_negedge) begin
      lrclk_posedge_r <= lrclk_posedge;
      lrclk_negedge_r <= lrclk_negedge;
    end
    else if (bclk_posedge) begin
      lrclk_posedge_r1 <= lrclk_posedge_r;
      lrclk_negedge_r1 <= lrclk_negedge_r;
      lrclk_posedge_r2 <= lrclk_posedge_r1;
      lrclk_negedge_r2 <= lrclk_negedge_r1;
    end
  end
  
  assign bclk_posedge  = i_bclk_en && !i_bclk;
  assign bclk_negedge  = i_bclk_en && i_bclk;
  // assign lrclk_posedge = i_lrclk_posedge;
  // assign lrclk_negedge = i_lrclk_negedge;
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
  
  logic last_sample;

  always_ff@(posedge i_ref_clk) begin
    if (i_rst || !i_tx_enable) begin
      last_sample <= 1'b0;
    end
    else if (bclk_negedge) begin
      case (tx_state)
        TX_IDLE: begin
    last_sample <= 1'b0;
  end
        TX_LOAD_DATA: begin
    last_sample <= 1'b0;
  end
  TX_LEFT_CH: begin
    last_sample <= last_sample;
  end
  TX_RIGHT_CH: begin
    if (!i_tx_ready && (tx_bit_count == 'b0)) begin
      last_sample <= ~last_sample;
    end
    else begin
      last_sample <= last_sample;
    end
  end
      endcase
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
          if (lrclk_posedge_r || (is_pcm_format && lrclk_negedge_r)) begin
            tx_state_next = TX_LEFT_CH;
          end
        end
        else begin
          if (lrclk_negedge_r || (is_pcm_format && lrclk_posedge_r)) begin
            tx_state_next = TX_LEFT_CH;
          end
        end
      end
      
      TX_LEFT_CH: begin
        if (tx_bit_count >= (active_bits - 1)) begin
          if (i_channel_mode == I2S_CH_MONO) begin
            tx_state_next = TX_LOAD_DATA;
          end else begin
            tx_state_next = TX_RIGHT_CH;
          end
        end
      end
      
      TX_RIGHT_CH: begin
        if (tx_bit_count >= (active_bits - 1)) begin
          if (i_tx_ready || last_sample) begin
            tx_state_next = TX_LEFT_CH;
          end
          else begin
            tx_state_next = TX_LOAD_DATA;
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
      tx_shift_reg_l <= '0;
      tx_shift_reg_r <= '0;
      tx_bit_count   <= '0;
      tx_left_ch     <= 1'b1;
    end else if (bclk_negedge) begin
      case (tx_state)
        TX_IDLE: begin
          tx_bit_count <= '0;
          tx_left_ch   <= 1'b1;
        end
        
        TX_LOAD_DATA: begin
          if (i_tx_valid) begin
            if (is_rj_format) begin
              // Right justified - align data to LSB, pad MSB with zeros
              tx_shift_reg_l <= i_tx_data_l;
              tx_shift_reg_r <= i_tx_data_r;
            end else begin
              // Left justified formats - align data to MSB
              tx_shift_reg_l <= i_tx_data_l << (MAX_BITS - active_bits);
              tx_shift_reg_r <= i_tx_data_r << (MAX_BITS - active_bits);
            end
          end
        end
        
        TX_WAIT_LRCLK: begin
          if (lrclk_negedge || (is_pcm_format && lrclk_posedge)) begin
            tx_bit_count <= '0;
            tx_left_ch   <= 1'b1;
          end
        end
        
        TX_LEFT_CH: begin
          if (tx_bit_count < (active_bits - 1)) begin
            if (is_rj_format) begin
              // Right justified - shift right, output LSB first
              tx_shift_reg_l <= tx_shift_reg_l >> 1;
            end else begin
              // Left justified formats - shift left, output MSB first
              tx_shift_reg_l <= tx_shift_reg_l << 1;
            end
            tx_bit_count <= tx_bit_count + 1'b1;
          end else begin
            tx_bit_count <= '0;
            tx_left_ch   <= 1'b0;
          end
        end
        
        TX_RIGHT_CH: begin
          if (tx_bit_count < (active_bits - 1)) begin
            if (is_rj_format) begin
              // Right justified - shift right, output LSB first
              tx_shift_reg_r <= tx_shift_reg_r >> 1;
            end else begin
              // Left justified formats - shift left, output MSB first
              tx_shift_reg_r <= tx_shift_reg_r << 1;
            end
            tx_bit_count <= tx_bit_count + 1'b1;
          end else begin
            tx_bit_count <= '0;
            tx_left_ch   <= 1'b1;
            if (is_rj_format) begin
              // Right justified - align data to LSB, pad MSB with zeros
              tx_shift_reg_l <= i_tx_data_l;
              tx_shift_reg_r <= i_tx_data_r;
            end else begin
              // Left justified formats - align data to MSB
              tx_shift_reg_l <= i_tx_data_l << (MAX_BITS - active_bits);
              tx_shift_reg_r <= i_tx_data_r << (MAX_BITS - active_bits);
            end
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
    end else if (tx_left_ch) begin
      if (is_rj_format) begin
        o_sdata_tx = tx_shift_reg_l[0];  // LSB for right justified (shifts right)
      end else begin
        o_sdata_tx = tx_shift_reg_l[MAX_BITS-1];  // MSB for left justified (shifts left)
      end
    end else begin
      if (is_rj_format) begin
        o_sdata_tx = tx_shift_reg_r[0];  // LSB for right justified (shifts right)
      end else begin
        o_sdata_tx = tx_shift_reg_r[MAX_BITS-1];  // MSB for left justified (shifts left)
      end
    end
  end
  
  // assign o_tx_ready = (tx_state == TX_IDLE) || (tx_state == TX_LOAD_DATA);
  // assign o_tx_ready = (tx_state == TX_LOAD_DATA && tx_state_prev != TX_IDLE);
  assign o_tx_ready = (tx_state == TX_LOAD_DATA && tx_state_prev != TX_IDLE) || (tx_state == TX_RIGHT_CH && tx_state_next == TX_LEFT_CH);
 
//------------------------------------------------------------------------------
// RX State Machine
//------------------------------------------------------------------------------
 
  typedef enum logic [2:0] {
    RX_IDLE,
    RX_WAIT_LRCLK,
    RX_LEFT_CH,
    RX_RIGHT_CH,
    RX_OUTPUT_DATA
  } rx_state_t;
  
  rx_state_t rx_state, rx_state_next;
  
  always_ff @(posedge i_ref_clk) begin
    if (i_rst || !i_rx_enable) begin
      rx_state <= RX_IDLE;
    end 
    else if (bclk_posedge) begin
      rx_state <= rx_state_next;
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
        if (lrclk_negedge_r1 || (is_pcm_format && lrclk_posedge_r1)) begin
          rx_state_next = RX_LEFT_CH;
        end
      end
      
      RX_LEFT_CH: begin
        if (rx_bit_count >= (active_bits - 1)) begin
          if (i_channel_mode == I2S_CH_MONO) begin
            rx_state_next = RX_OUTPUT_DATA;
          end else begin
            rx_state_next = RX_RIGHT_CH;
          end
        end
      end
      
      RX_RIGHT_CH: begin
        if (rx_bit_count >= (active_bits - 1)) begin
          // rx_state_next = RX_OUTPUT_DATA;
          rx_state_next = RX_LEFT_CH;
        end
      end
      
      RX_OUTPUT_DATA: begin
        if (i_rx_ready) begin
          if (lrclk_negedge || (is_pcm_format && lrclk_posedge)) begin
            rx_state_next = RX_LEFT_CH;
          end else begin
            rx_state_next = RX_WAIT_LRCLK;
          end
        end
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
          if (lrclk_negedge_r1 || (is_pcm_format && lrclk_posedge_r1)) begin
            rx_bit_count <= '0;
            rx_left_ch   <= 1'b1;
          end
        end
        
        RX_LEFT_CH: begin
          if (rx_bit_count < active_bits ) begin
            if (is_rj_format) begin
              rx_shift_reg_l <= {rx_sdata_reg, rx_shift_reg_l[MAX_BITS-1:1]};
            end else begin
              rx_shift_reg_l <= {rx_shift_reg_l[MAX_BITS-2:0], rx_sdata_reg};
            end
            if (rx_bit_count == active_bits - 1) begin
              rx_bit_count <= '0;
            end
            else begin
              rx_bit_count <= rx_bit_count + 1'b1;
            end
          end else begin
            rx_bit_count <= '0;
            rx_left_ch   <= 1'b0;
          end
        end
        
        RX_RIGHT_CH: begin
          if (rx_bit_count < active_bits) begin
            if (is_rj_format) begin
              rx_shift_reg_r <= {rx_sdata_reg, rx_shift_reg_r[MAX_BITS-1:1]};
            end else begin
              rx_shift_reg_r <= {rx_shift_reg_r[MAX_BITS-2:0], rx_sdata_reg};
            end
            if (rx_bit_count == active_bits - 1) begin
              rx_bit_count <= '0;
            end
            else begin
              rx_bit_count <= rx_bit_count + 1'b1;
            end
          end else begin
            rx_bit_count <= '0;
            rx_left_ch   <= 1'b1;
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
      rx_data <= '0;
      rx_data_valid <= 1'b0;
    end 
    else if (bclk_posedge) begin
      if (lrclk_negedge_r1) begin
        rx_data_valid <= 1'b1;
      end
      else if (lrclk_posedge_r1) begin
        rx_data_valid <= 1'b1;
      end
      else begin
        rx_data_valid <= 1'b0;
      end

      if (lrclk_negedge_r2) begin
        rx_data <= rx_shift_reg_r & rx_data_mask;
      end
      else if (lrclk_posedge_r2) begin
        rx_data <= rx_shift_reg_l & rx_data_mask;
      end
    end
  end
 
//------------------------------------------------------------------------------
// RX Output Assignment
//------------------------------------------------------------------------------
 
  assign o_rx_valid = rx_data_valid;
  assign o_rx_data  = rx_data;
 
endmodule 
