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

module uart #(
    parameter OVERSAMPLE = 8,  // Changed from 16 to 8 for better baud rate accuracy
    parameter DATA_WIDTH = 8
)(
  // System
  input                   clk,
  input                   rst_n,

  // Configuration
  input  [15:0]           baud_div,        // Baud rate divisor for 16x oversampling
  input                   tx_en,           // TX enable
  input                   rx_en,           // RX enable
  input                   loopback_en,     // Internal loopback enable
  input  [3:0]            data_width,      // Variable data width (5-8 bits)
  input  [1:0]            parity_mode,     // Parity mode: 00=NONE, 01=ODD, 10=EVEN, 11=reserved
  input [1:0]             stop_bits_mode,  // Stop bits: 00=1 bit, 01=1.5 bits, 10=2 bits, 11=reserved

  // TX Interface
  input  [DATA_WIDTH-1:0] tx_data,
  input                   tx_valid,
  output                  tx_ready,

  // RX Interface
  output [DATA_WIDTH-1:0] rx_data,
  output                  rx_valid,
  input                   rx_ready,
  output                  rx_parity_error,
  output                  rx_frame_error,

  // UART Interface
  output                  uart_tx,
  input                   uart_rx,
  input                   rts_n,

  // glitch filtering
  input  [3:0]            glitch_window_width,
  input  [3:0]            glitch_start_bit_majority,
  input  [3:0]            glitch_data_bit_majority,
  input                   glitch_filter_en,

  // Glitch detection status
  output                  glitch_error
);

// 2-cycle flopped version of rts_n for timing closure
logic rts_n_q, rts_n_q2;

// Baud rate generator
logic [15:0] baud_counter;
logic        baud_tick;

always_ff @(posedge clk) begin
  if (!rst_n) begin
    baud_counter <= 0;
    baud_tick    <= 0;
  end
  else begin
    if (baud_counter >= baud_div) begin
      baud_counter <= 0;
      baud_tick    <= 1;
    end
    else begin
      baud_counter <= baud_counter + 1;
      baud_tick    <= 0;
    end
  end
end

// TX State Machine
typedef enum logic [2:0] {
  TX_IDLE,
  TX_START,
  TX_DATA,
  TX_PARITY,
  TX_STOP
} tx_state_t;

tx_state_t tx_state;
logic [3:0]            tx_oversample_cnt;
logic [2:0]            tx_bit_cnt;
logic [DATA_WIDTH-1:0] tx_shift_reg;
logic                  tx_reg;
logic                  tx_parity_bit;
logic [4:0]            tx_stop_bit_cycles; // Maximum cycles needed for 2 stop bits

// Parity calculation function (for TX - data is LSB-aligned)
function automatic logic calc_parity(input [7:0] data, input [2:0] width, input [1:0] mode);
    logic parity_even;
    case (width)
        4'd5: parity_even = ^data[4:0];
        4'd6: parity_even = ^data[5:0];
        4'd7: parity_even = ^data[6:0];
        5'd8: parity_even = ^data[7:0];
        default: parity_even = ^data[7:0];
    endcase
    
    case (mode)
        2'b00: return 1'b0;           // No parity (won't be used)
        2'b01: return ~parity_even;   // Odd parity
        2'b10: return parity_even;    // Even parity
        2'b11: return parity_even;    // Reserved (default to even)
    endcase
endfunction

// RX Parity calculation function (for RX - data is MSB-aligned in shift register)
function automatic logic calc_rx_parity(input [7:0] data, input [2:0] width, input [1:0] mode);
    logic parity_even;
    case (width)
        4'd5: parity_even = ^data[7:3];
        4'd6: parity_even = ^data[7:2];
        4'd7: parity_even = ^data[7:1];
        4'd8: parity_even = ^data[7:0];
        default: parity_even = ^data[7:0];
    endcase
    
    case (mode)
        2'b00: return 1'b0;           // No parity (won't be used)
        2'b01: return ~parity_even;   // Odd parity
        2'b10: return parity_even;    // Even parity
        2'b11: return parity_even;    // Reserved (default to even)
    endcase
endfunction

// Stop bit duration calculation function
function automatic [4:0] calc_stop_bit_cycles(input [1:0] stop_mode);
    case (stop_mode)
        2'b00: return OVERSAMPLE;           // 1 stop bit
        2'b01: return OVERSAMPLE + (OVERSAMPLE >> 1); // 1.5 stop bits
        2'b10: return 2 * OVERSAMPLE;       // 2 stop bits
        2'b11: return OVERSAMPLE;           // Reserved (default to 1 stop bit)
    endcase
endfunction
    
always_ff @(posedge clk) begin
  if (!rst_n) begin
    tx_state           <= TX_IDLE;
    tx_oversample_cnt  <= 0;
    tx_bit_cnt         <= 0;
    tx_shift_reg       <= 0;
    tx_reg             <= 1; // Idle high
    tx_parity_bit      <= '0;
    tx_stop_bit_cycles <= '0;
  end
  else if (tx_en) begin
    case (tx_state)
      TX_IDLE: begin
        tx_reg <= 1;
        if (tx_valid) begin
          // Mask input data based on data width
          case (data_width)
              4'd5: tx_shift_reg <= {3'b000, tx_data[4:0]};
              4'd6: tx_shift_reg <= {2'b00, tx_data[5:0]};
              4'd7: tx_shift_reg <= {1'b0, tx_data[6:0]};
              4'd8: tx_shift_reg <= tx_data[7:0];
              default: tx_shift_reg <= tx_data[7:0];
          endcase
          // Calculate parity bit for transmission
          tx_parity_bit <= calc_parity(tx_data, data_width, parity_mode);
          // Calculate stop bit cycles for this transmission
          tx_stop_bit_cycles <= calc_stop_bit_cycles(stop_bits_mode);
          tx_state          <= TX_START;
          tx_oversample_cnt <= 0;
        end
      end

      TX_START: begin
        if (baud_tick) begin
          tx_reg <= 0; // Start bit
          if (tx_oversample_cnt >= OVERSAMPLE - 1) begin
            tx_oversample_cnt <= 0;
            tx_state          <= TX_DATA;
            tx_bit_cnt        <= 0;
          end
          else begin
            tx_oversample_cnt <= tx_oversample_cnt + 1;
          end
        end
      end

      TX_DATA: begin
        if (baud_tick) begin
          tx_reg <= tx_shift_reg[0];
          if (tx_oversample_cnt >= OVERSAMPLE - 1) begin
            tx_oversample_cnt <= 0;
            tx_shift_reg      <= tx_shift_reg >> 1;
            if (tx_bit_cnt >= data_width - 1) begin
              // Check if parity is enabled
              if (parity_mode != 2'b00) begin
                tx_state <= TX_PARITY;
              end
              else begin
                tx_state <= TX_STOP;
              end
            end
            else begin
              tx_bit_cnt <= tx_bit_cnt + 1;
            end
          end
          else begin
            tx_oversample_cnt <= tx_oversample_cnt + 1;
          end
        end
      end

      TX_PARITY: begin
          if (baud_tick) begin
              tx_reg <= tx_parity_bit;
              if (tx_oversample_cnt >= OVERSAMPLE - 1) begin
                  tx_oversample_cnt <= 0;
                  tx_state <= TX_STOP;
              end else begin
                  tx_oversample_cnt <= tx_oversample_cnt + 1;
              end
          end
      end
                
      TX_STOP: begin
        if (baud_tick) begin
          tx_reg <= 1; // Stop bit
          if (tx_oversample_cnt >= tx_stop_bit_cycles  - 1) begin
            tx_oversample_cnt <= 0;
            tx_state          <= TX_IDLE;
          end
          else begin
            tx_oversample_cnt <= tx_oversample_cnt + 1;
          end
        end
      end
    endcase
  end
end

assign tx_ready = (tx_state == TX_IDLE) && tx_en;
assign uart_tx  = tx_reg;

// RX State Machine
typedef enum logic [2:0] {
  RX_IDLE,
  RX_START,
  RX_DATA,
  RX_PARITY,
  RX_STOP
} rx_state_t;

rx_state_t rx_state;
logic [3:0]            rx_oversample_cnt;
logic [2:0]            rx_bit_cnt;
logic [DATA_WIDTH-1:0] rx_shift_reg;
logic [1:0]            rx_sync;
logic                  rx_falling_edge;
logic                  rx_glitch_error;  // Indicates glitch detected (majority vote failed)
logic                  rx_sampled_bit;   // Bit sampled at center of period (for non-glitch mode)
logic                  rx_parity_received;
logic                  rx_parity_error_reg;
logic                  rx_frame_error_reg;
logic                  rx_discard_frame;   // Set on bad start/mid-frame glitch; frame not presented at end
logic [4:0]            rx_stop_bit_cycles; // Maximum cycles needed for 2 stop bits

// RX baud generator: lock onto received start edge so RX state machine stays in sync with frame
// Timing: Start is detected 2 cycles after the wire edge (rx_sync is 2 flops). We reset counter
// then, so first rx_baud_tick is at wire_edge + 2 + baud_div. Sampling is thus ~2 cycles late
// per bit; for baud_div >> 2 this is a small fraction of a bit and acceptable.
logic [15:0] rx_baud_counter;
logic        rx_baud_tick;

// Combined start condition: new frame from IDLE or back-to-back (falling edge at end of RX_STOP)
logic rx_start_lock;
assign rx_start_lock = (rx_state == RX_IDLE && rx_falling_edge && rts_n_q2 == 1'b0) ||
  (rx_state == RX_STOP && rx_oversample_cnt >= rx_stop_bit_cycles - 1 && rx_baud_tick && rx_falling_edge);

always_ff @(posedge clk) begin
  if (!rst_n) begin
    rx_baud_counter <= 0;
    rx_baud_tick    <= 0;
  end
  else if (!rx_en) begin
    rx_baud_counter <= 0;
    rx_baud_tick    <= 0;
  end
  else if (rx_start_lock) begin
    // Align to start: new frame from IDLE or back-to-back (falling edge at end of RX_STOP)
    rx_baud_counter <= 16'd2;             // Compensate for 2-flop rx_sync latency
    rx_baud_tick    <= 0;
  end
  else if (rx_state != RX_IDLE) begin
    if (rx_baud_counter >= baud_div) begin
      rx_baud_counter <= 0;
      rx_baud_tick    <= 1;
    end
    else begin
      rx_baud_counter <= rx_baud_counter + 1;
      rx_baud_tick    <= 0;
    end
  end
  else begin
    rx_baud_counter <= 0;
    rx_baud_tick    <= 0;
  end
end

//==========================================================================
// Configurable Glitch Filtering Logic (M-out-of-N Majority Voting)
//==========================================================================
// Window is centered around OVERSAMPLE/2
// glitch_window_width defines span from center:
//   center = OVERSAMPLE/2
//   window = [center - glitch_window_width : center + glitch_window_width]
// Example: OVERSAMPLE=8, glitch_window_width=2
//   center = 4, window = [2:6] = 5 samples
//
// Majority voting: M-out-of-N
//   N = total samples in window = (2 * glitch_window_width) + 1
//   M = glitch_start_bit_majority or glitch_data_bit_majority
//==========================================================================

logic [3:0]            glitch_window_start, glitch_window_end;
logic [3:0]            glitch_center;
logic [OVERSAMPLE-1:0] rx_sample_buffer;  // Stores samples during window
logic [3:0]            rx_sample_count;   // Number of samples collected
logic [3:0]            rx_ones_count;     // Count of '1' samples

// Calculate window boundaries
assign glitch_center = OVERSAMPLE >> 1;
assign glitch_window_start = (glitch_center > glitch_window_width) ?
                              (glitch_center - glitch_window_width) : 4'd0;
assign glitch_window_end = (glitch_center + glitch_window_width < OVERSAMPLE) ?
                            (glitch_center + glitch_window_width) : (OVERSAMPLE - 1);

// Sampling logic: Use center of bit period for better timing margins
logic rx_sample_point;
logic rx_bit_valid;  // Unified signal: bit is ready and valid
logic rx_bit_value;  // Unified signal: the sampled/voted bit value
assign rx_sample_point = (rx_oversample_cnt == (OVERSAMPLE >> 1)) || // center of start bit, data bit or 1 stop bit
                         ((stop_bits_mode == 2'b01) && rx_oversample_cnt == OVERSAMPLE + (OVERSAMPLE>>2)) || // center of 1.5 stop bit
                         (rx_oversample_cnt == (OVERSAMPLE + OVERSAMPLE >> 1)); // center of 2 stop bit
// Unified bit sampling/voting logic
always_comb begin
  if (rx_oversample_cnt >= OVERSAMPLE - 1) begin
    if (glitch_filter_en && rx_sample_count > 0) begin
      // Glitch filter enabled: Use majority voting
      if (rx_state == RX_START) begin
        // Start bits: use configurable threshold
        if (rx_ones_count >= glitch_start_bit_majority) begin
          rx_bit_value    = 1'b1;
          rx_bit_valid    = 1'b1;
          rx_glitch_error = 1'b1;
        end
        else if (rx_ones_count <= (rx_sample_count - glitch_start_bit_majority)) begin
          rx_bit_value    = 1'b0;
          rx_bit_valid    = 1'b1;
          rx_glitch_error = 1'b0;
        end
        else begin
          // Ambiguous - no clear majority
          rx_bit_value    = 1'b0;
          rx_bit_valid    = 1'b0;
          rx_glitch_error = 1'b1;
        end
      end
      else begin
        // Data/Stop bits: use configurable threshold
        if (rx_ones_count >= glitch_data_bit_majority) begin
          rx_bit_value    = 1'b1;
          rx_bit_valid    = 1'b1;
          rx_glitch_error = 1'b0;
        end
        else if (rx_ones_count <= (rx_sample_count - glitch_data_bit_majority)) begin
          rx_bit_value    = 1'b0;
          rx_bit_valid    = 1'b1;
          rx_glitch_error = 1'b0;
        end
        else begin
          // Ambiguous - no clear majority
          rx_bit_value    = 1'b0;
          rx_bit_valid    = 1'b0;
          rx_glitch_error = 1'b1;
        end
      end
    end
    else begin
      // Glitch filter disabled: Use center-sampled bit
      rx_bit_value    = rx_sampled_bit;
      rx_bit_valid    = 1'b1;
      rx_glitch_error = 1'b0;
    end
  end
  else begin
    rx_bit_value    = 1'b1;
    rx_bit_valid    = 1'b0;
    rx_glitch_error = 1'b0;
  end
end

// Sample collection and voting logic
// Note: rx_ones_count is incremented as samples are collected (not counted at end)
always_ff @(posedge clk) begin
  if (!rst_n) begin
  rx_sample_buffer <= '0;
  rx_sample_count  <= 0;
  rx_ones_count    <= 0;
  end
  else if (rx_en && rx_baud_tick) begin
    case (rx_state)
      RX_IDLE: begin
        // Reset for next reception
        rx_sample_buffer <= '0;
        rx_sample_count  <= 0;
        rx_ones_count    <= 0;
      end

      RX_START, RX_DATA, RX_PARITY, RX_STOP: begin
        // Collect samples within the configured window
        if (glitch_filter_en &&
          ((rx_oversample_cnt >= glitch_window_start && rx_oversample_cnt <= glitch_window_end))) begin
          // Store sample in buffer and increment counters
          rx_sample_buffer[rx_sample_count] <= rx_sync[1];
          rx_sample_count                   <= rx_sample_count + 1;
          // Increment ones counter if sample is '1'
          if (rx_sync[1]) begin
            rx_ones_count <= rx_ones_count + 1;
          end
        end

        // At end of bit period, reset for next bit (or next frame)
        if (rx_oversample_cnt == OVERSAMPLE - 1 ||
            rx_oversample_cnt == 2*OVERSAMPLE - 1 ||
            (stop_bits_mode == 2'b01 && rx_oversample_cnt == OVERSAMPLE + (OVERSAMPLE >> 1) - 1)) begin
          rx_sample_buffer <= '0;
          rx_sample_count  <= 0;
          rx_ones_count    <= 0;
        end
      end

      default: begin
        rx_sample_buffer <= '0;
        rx_sample_count  <= 0;
        rx_ones_count    <= 0;
      end
    endcase
  end
end


// Synchronize RX input - use loopback or external RX
logic uart_rx_input;
assign uart_rx_input = loopback_en ? tx_reg : uart_rx;

always_ff @(posedge clk) begin
  if (!rst_n) begin
    rx_sync  <= 2'b11;
    rts_n_q  <= 1'b0;  // Default to ready (active low)
    rts_n_q2 <= 1'b0;  // Default to ready (active low)
  end
  else begin
    rx_sync  <= {rx_sync[0], uart_rx_input};
    rts_n_q  <= rts_n;
    rts_n_q2 <= rts_n_q;
  end
end

assign rx_falling_edge = (rx_sync == 2'b10);

// RX State Machine - Unified logic using rx_bit_value and rx_bit_valid
always_ff @(posedge clk) begin
  if (!rst_n) begin
    rx_state          <= RX_IDLE;
    rx_oversample_cnt <= 0;
    rx_bit_cnt        <= 0;
    rx_shift_reg      <= 0;
    rx_parity_error_reg <= 0;
    rx_frame_error_reg <= 0;
    rx_discard_frame  <= 0;
    rx_sampled_bit    <= 1'b1;
    rx_stop_bit_cycles <= '0;
  end
  else if (rx_en) begin
    case (rx_state)
      RX_IDLE: begin
        if (rx_falling_edge && rts_n_q2 == 1'b0) begin
          rx_state          <= RX_START;
          rx_oversample_cnt <= 0;
          rx_discard_frame  <= 0;  // New frame: clear discard from any previous run
        end
      end

      RX_START: begin
        if (rx_baud_tick) begin
          // Always sample at center for non-glitch-filter mode
          if (rx_sample_point) begin
            rx_sampled_bit <= rx_sync[1];
          end
          
          if (rx_oversample_cnt >= OVERSAMPLE - 1) begin
            // Check start bit (expect '0')
            if (rx_bit_valid && !rx_bit_value) begin
              rx_oversample_cnt <= 0;
              rx_state          <= RX_DATA;
              rx_bit_cnt        <= 0;
            end
            else begin
              // False start or glitch: continue frame to stay bit-aligned, discard at end
              rx_discard_frame  <= 1'b1;
              rx_oversample_cnt <= 0;
              rx_state          <= RX_DATA;
              rx_bit_cnt        <= 0;
            end
          end
          else begin
            rx_oversample_cnt <= rx_oversample_cnt + 1;
          end
        end
      end

      RX_DATA: begin
        if (rx_baud_tick) begin
          // Always sample at center for non-glitch-filter mode
          if (rx_sample_point) begin
            rx_sampled_bit <= rx_sync[1];
          end

          if (rx_oversample_cnt >= OVERSAMPLE - 1) begin
            if (rx_bit_valid) begin
              // Valid bit received
              rx_oversample_cnt <= 0;
              rx_shift_reg <= {rx_bit_value, rx_shift_reg[DATA_WIDTH-1:1]};

              if (rx_bit_cnt >= data_width - 1) begin
                // Check if parity is enabled
                if (parity_mode != 2'b00) begin
                  rx_state <= RX_PARITY;
                end
                else begin
                  rx_state            <= RX_STOP;
                  rx_parity_error_reg <= 1'b0; // No parity error when parity disabled
                  rx_frame_error_reg  <= 1'b0;  // Clear frame error before checking STOP bit
                  rx_stop_bit_cycles  <= calc_stop_bit_cycles(stop_bits_mode);
                end
              end
              else begin
                rx_bit_cnt <= rx_bit_cnt + 1;
              end
            end
            else begin
              // Glitch: discard frame but continue to stay bit-aligned
              rx_discard_frame  <= 1'b1;
              rx_oversample_cnt <= 0;
              rx_shift_reg      <= {rx_bit_value, rx_shift_reg[DATA_WIDTH-1:1]};
              if (rx_bit_cnt >= data_width - 1) begin
                if (parity_mode != 2'b00) begin
                  rx_state <= RX_PARITY;
                end
                else begin
                  rx_state            <= RX_STOP;
                  rx_parity_error_reg <= 1'b0;
                  rx_frame_error_reg  <= 1'b0;
                  rx_stop_bit_cycles  <= calc_stop_bit_cycles(stop_bits_mode);
                end
              end
              else begin
                rx_bit_cnt <= rx_bit_cnt + 1;
              end
            end
          end
          else begin
            rx_oversample_cnt <= rx_oversample_cnt + 1;
          end
        end
      end

      RX_PARITY: begin
        if (rx_baud_tick) begin
          // Always sample at center for non-glitch-filter mode
          if (rx_sample_point) begin
            rx_sampled_bit <= rx_sync[1];
          end
          if (rx_oversample_cnt >= OVERSAMPLE - 1) begin
            if (rx_bit_valid) begin
              rx_oversample_cnt <= 0;
              rx_parity_error_reg <= (rx_bit_value != calc_rx_parity(rx_shift_reg, data_width, parity_mode));
            end
            else begin
              // Glitch: discard frame but continue to stay bit-aligned
              rx_discard_frame  <= 1'b1;
              rx_oversample_cnt <= 0;
            end
            rx_frame_error_reg <= 1'b0;  // Clear frame error before checking STOP bit
            rx_stop_bit_cycles <= calc_stop_bit_cycles(stop_bits_mode);
            rx_state           <= RX_STOP;
          end else begin
            rx_oversample_cnt <= rx_oversample_cnt + 1;
          end
        end
      end
                
      RX_STOP: begin
        if (rx_baud_tick) begin
          // Always sample at center for non-glitch-filter mode
          if (rx_sample_point) begin
            rx_sampled_bit <= rx_sync[1];
          end

          if (rx_oversample_cnt == OVERSAMPLE - 1) begin // end of 1.5 stop bit
            if (rx_bit_valid) begin
              rx_frame_error_reg <= rx_bit_value ? rx_frame_error_reg : 1'b1;
            end
            else begin
              // Glitch: discard frame but continue to end of stop period
              rx_discard_frame <= 1'b1;
            end
          end
          if (rx_oversample_cnt >= rx_stop_bit_cycles  - 1) begin
            if (rx_falling_edge) begin
              // Back-to-back: next frame already started; re-lock and receive it
              rx_state          <= RX_START;
              rx_oversample_cnt <= 0;
              rx_discard_frame  <= 0;
            end
            else begin
              rx_state          <= RX_IDLE;
              rx_oversample_cnt <= 0;
            end
          end
          else begin
            rx_oversample_cnt <= rx_oversample_cnt + 1;
          end
        end
      end
    endcase
  end
end

// RX output: discard if rx_discard_frame was set earlier OR stop-bit glitch detected this cycle
// (rx_discard_frame updates next cycle, so same-cycle stop-bit glitch would otherwise be presented)
logic rx_stop_bit_glitch;
assign rx_stop_bit_glitch =
  (rx_state == RX_STOP && rx_oversample_cnt >= rx_stop_bit_cycles - 1 && rx_baud_tick) &&
  (rx_oversample_cnt >= OVERSAMPLE - 1) && !rx_bit_valid;

logic                  rx_data_valid;
logic [DATA_WIDTH-1:0] rx_data_reg;

always_ff @(posedge clk) begin
  if (!rst_n) begin
    rx_data_valid <= 0;
    rx_data_reg   <= 0;
  end
  else begin
    if (rx_state == RX_STOP && rx_oversample_cnt >= rx_stop_bit_cycles - 1 && rx_baud_tick) begin
      // Only present frame when not discarded (start/data/parity glitch or stop-bit glitch)
      rx_data_valid <= !rx_discard_frame && !rx_stop_bit_glitch;
      rx_data_reg   <= rx_shift_reg;
    end
    else if (rx_ready) begin
      rx_data_valid <= 0;
    end
  end
end

// Mask unused bits based on data width
logic [DATA_WIDTH-1:0] rx_data_masked;
always_comb begin
    case (data_width)
        4'd5: rx_data_masked    = {3'b000, rx_data_reg[7:3]};
        4'd6: rx_data_masked    = {2'b00, rx_data_reg[7:2]};
        4'd7: rx_data_masked    = {1'b0, rx_data_reg[7:1]};
        4'd8: rx_data_masked    = rx_data_reg[7:0];
        default: rx_data_masked = rx_data_reg[7:0];
    endcase
end

assign rx_data      = rx_data_masked;
assign rx_valid     = rx_data_valid;
assign rx_parity_error = rx_parity_error_reg && rx_data_valid; // Only signal parity error when data is valid
assign rx_frame_error = rx_frame_error_reg && rx_data_valid;   // Only signal frame error when data is valid
assign glitch_error = rx_glitch_error;
endmodule
