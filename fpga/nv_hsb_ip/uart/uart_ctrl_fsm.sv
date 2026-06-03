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

module uart_ctrl_fsm
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter NUM_INST        = 1,
  parameter FIFO_DEPTH      = 256,
  parameter UART_ADDR_WIDTH = 9
)(
  input                           i_aclk,
  input                           i_arst,

  // APB Interface
  input  apb_m2s                  i_apb_m2s,
  output apb_s2m                  o_apb_s2m,

  // UART Interface
  output                          uart_tx,
  input                           uart_rx,

  // Flow control pins (optional)
  output                          uart_rts,     // Request to Send (output)
  input                           uart_cts,     // Clear to Send (input)

  // Enable Override
  input                           uart_enable_override,

  // Status
  output                          uart_busy,

  // Interrupt
  output                          uart_interrupt
);

// Register definitions
logic [31:0] ctrl_reg [5];
logic [31:0] stat_reg [2];

// APB interface splitting
logic        isCtrlAddr;
apb_m2s      ctrl_apb_m2s;
apb_s2m      ctrl_apb_s2m;
apb_m2s      data_apb_m2s;
apb_s2m      data_apb_s2m;

// Clock and reset mapping
logic clk, rst_n;
assign clk = i_aclk;
assign rst_n = !i_arst;

// s_apb_reg for control and status registers
s_apb_reg #(
  .N_CTRL    ( 5              ),
  .N_STAT    ( 2              ),
  .W_OFST    ( 8              ), // 8-bit offset for 256 byte address space
  .SAME_CLK  ( 1              )
) u_reg_map (
  // APB Interface
  .i_aclk    ( i_aclk         ),
  .i_arst    ( i_arst         ),
  .i_apb_m2s ( ctrl_apb_m2s   ),
  .o_apb_s2m ( ctrl_apb_s2m   ),
  // User Control Signals
  .i_pclk    ( i_aclk         ),
  .i_prst    ( i_arst         ),
  .o_ctrl    ( ctrl_reg       ),
  .i_stat    ( stat_reg       )
);

// Address decoding: bit 8 determines control (0) vs data (1) registers
assign isCtrlAddr = !i_apb_m2s.paddr[8];

// APB signal routing
assign ctrl_apb_m2s.psel    = (isCtrlAddr && i_apb_m2s.psel);
assign ctrl_apb_m2s.penable = i_apb_m2s.penable;
assign ctrl_apb_m2s.paddr   = i_apb_m2s.paddr;
assign ctrl_apb_m2s.pwdata  = i_apb_m2s.pwdata;
assign ctrl_apb_m2s.pwrite  = i_apb_m2s.pwrite;

assign data_apb_m2s.psel    = (!isCtrlAddr && i_apb_m2s.psel);
assign data_apb_m2s.penable = i_apb_m2s.penable;
assign data_apb_m2s.paddr   = {24'h0, i_apb_m2s.paddr[7:0]};
assign data_apb_m2s.pwdata  = i_apb_m2s.pwdata;
assign data_apb_m2s.pwrite  = i_apb_m2s.pwrite;

assign o_apb_s2m = (isCtrlAddr) ? ctrl_apb_s2m : data_apb_s2m;

// Register map
logic        uart_enable;
logic        tx_enable;
logic        rx_enable;
logic        loopback_en;
logic [15:0] baud_div;
logic        tx_fifo_reset;
logic        rx_fifo_reset;

// Status signals
logic        tx_fifo_full;
logic        tx_fifo_empty, tx_fifo_empty_sticky;
logic        rx_fifo_full;
logic        rx_fifo_empty, rx_fifo_not_empty;
logic        last_rx_clear;

// Control register assignments with override capability
assign uart_enable    = ctrl_reg[0][0] | uart_enable_override;
assign tx_enable      = ctrl_reg[0][1] | uart_enable_override;
assign rx_enable      = ctrl_reg[0][2] | uart_enable_override;
assign loopback_en    = ctrl_reg[0][3];
assign tx_fifo_reset  = ctrl_reg[0][8];
assign rx_fifo_reset  = ctrl_reg[0][9];
assign last_rx_clear  = ctrl_reg[0][10];
assign baud_div       = ctrl_reg[1][15:0];

// Parity configuration: 00=NONE(default), 01=ODD, 10=EVEN, 11=reserved
logic [1:0] parity_mode;
assign parity_mode = ctrl_reg[0][7:6];

// Stop bits configuration: 00=1 stop bit(default), 01=1.5 stop bits, 10=2 stop bits, 11=reserved
logic [1:0] stop_bits_mode;
assign stop_bits_mode = ctrl_reg[0][13:12];

// Flow control configuration
logic flow_control_en;   // Enable flow control functionality 
assign flow_control_en = ctrl_reg[0][14];

// FIFO threshold configuration (programmable)
logic [7:0] rx_afull_threshold;   // RX almost full threshold (0-255)
logic [7:0] rx_aempty_threshold;  // RX almost empty threshold (0-255)
logic [7:0] tx_afull_threshold;   // TX almost full threshold (0-255)
logic [7:0] tx_aempty_threshold;  // TX almost empty threshold (0-255)
logic [7:0] rx_afull_reg;         // RX almost full register value
logic [7:0] rx_aempty_reg;        // RX almost empty register value
logic [7:0] tx_afull_reg;         // TX almost full register value
logic [7:0] tx_aempty_reg;        // TX almost empty register value

// Extract threshold register values
assign rx_afull_reg   = ctrl_reg[0][23:16];
assign rx_aempty_reg  = ctrl_reg[0][31:24];
assign tx_afull_reg   = ctrl_reg[1][23:16];
assign tx_aempty_reg  = ctrl_reg[1][31:24];

// Apply defaults for backward compatibility (when register value is 0)
// Default values match original fixed thresholds: almost_full=252, almost_empty=4
assign rx_afull_threshold  = (rx_afull_reg == 8'h00)  ? 8'd252 : rx_afull_reg;
assign rx_aempty_threshold = (rx_aempty_reg == 8'h00) ? 8'd4   : rx_aempty_reg;
assign tx_afull_threshold  = (tx_afull_reg == 8'h00)  ? 8'd252 : tx_afull_reg;
assign tx_aempty_threshold = (tx_aempty_reg == 8'h00) ? 8'd4   : tx_aempty_reg;

// Data width selection: 00=8bits(default), 01=7bits, 10=6bits, 11=5bits
logic [1:0] data_width_sel;
logic [3:0] data_width_actual;
assign data_width_sel = ctrl_reg[0][5:4];

always_comb begin
    case (data_width_sel)
        2'b00: data_width_actual = 4'd8; // Default: 8 bits
        2'b01: data_width_actual = 4'd7; // 7 bits
        2'b10: data_width_actual = 4'd6; // 6 bits
        2'b11: data_width_actual = 4'd5; // 5 bits
    endcase
end
    
// Interrupt control signals
logic tx_empty_int_en, tx_afull_int_en, tx_aempty_int_en, rx_not_empty_int_en, rx_glitch_error_int_en;


// Interrupt status signals
logic tx_empty_int_status, tx_afull_int_status, tx_aempty_int_status, rx_glitch_error_int_status;
logic rx_not_empty_int_status;

// Error detection
logic tx_overflow, tx_overflow_sticky;
logic rx_overflow, rx_overflow_sticky;
logic rx_parity_error, rx_parity_error_sticky;
logic rx_frame_error, rx_frame_error_sticky;
logic rx_glitch_error, rx_glitch_error_sticky;

// Busy signals
logic uart_tx_busy, uart_rx_busy;

// Last RX byte register - stores most recent byte read from RX FIFO
logic [7:0]  last_rx_byte;
logic        last_rx_valid;


// Interrupt control register (ctrl_reg[2]) - bit assignments
assign tx_empty_int_en        = ctrl_reg[2][0];
assign tx_afull_int_en        = ctrl_reg[2][1];
assign tx_aempty_int_en       = ctrl_reg[2][2];
assign rx_not_empty_int_en    = ctrl_reg[2][3];
assign rx_glitch_error_int_en = ctrl_reg[2][4];

// Glitch filter configuration
logic [3:0] glitch_window_width;
logic [3:0] glitch_start_bit_majority;
logic [3:0] glitch_data_bit_majority;
logic       glitch_filter_en;

// Extract from control register (suggest CTRL2 or CTRL3)
assign glitch_window_width        = ctrl_reg[4][3:0];
assign glitch_start_bit_majority  = ctrl_reg[4][7:4];
assign glitch_data_bit_majority   = ctrl_reg[4][11:8];
assign glitch_filter_en           = ctrl_reg[4][16];


// Status register assignments
assign stat_reg[0][0]     = uart_tx_busy;
assign stat_reg[0][1]     = uart_rx_busy;
assign stat_reg[0][2]     = tx_fifo_full;
assign stat_reg[0][3]     = tx_fifo_empty;
assign stat_reg[0][4]     = rx_fifo_full;
assign stat_reg[0][5]     = rx_fifo_not_empty;
assign stat_reg[0][6]     = rx_parity_error_sticky;
assign stat_reg[0][7]     = rx_frame_error_sticky;
assign stat_reg[0][8]     = tx_overflow_sticky;
assign stat_reg[0][9]     = rx_overflow_sticky;
assign stat_reg[0][10]    = last_rx_valid;
assign stat_reg[0][11]    = uart_cts;  // CTS input status (read-only)
assign stat_reg[0][12]    = uart_rts;  // RTS output status (read-only)
assign stat_reg[0][13]    = rx_glitch_error_sticky;
assign stat_reg[0][31:14] = '0;

assign stat_reg[1][0]    = tx_empty_int_status;
assign stat_reg[1][1]    = tx_afull_int_status;
assign stat_reg[1][2]    = tx_aempty_int_status;
assign stat_reg[1][3]    = rx_not_empty_int_status;
assign stat_reg[1][4]    = rx_glitch_error_int_status;
assign stat_reg[1][31:5] = '0; // Reserved for future use

// TX FIFO
logic [7:0] tx_fifo_din;
logic       tx_fifo_wr;
logic       tx_fifo_wr_safe;  // Safe write (prevents overflow)
logic [7:0] tx_fifo_dout;
logic       tx_fifo_rd;
logic       tx_fifo_rd_reg;
logic       tx_fifo_dval;
logic       tx_fifo_afull, tx_fifo_afull_sticky;     // Programmable almost full
logic       tx_fifo_aempty, tx_fifo_aempty_sticky;    // Programmable almost empty
logic       tx_fifo_afull_hw;  // Hardware almost full (from FIFO)
logic       tx_fifo_aempty_hw; // Hardware almost empty (from FIFO)
logic [8:0] tx_fifo_level;     // Current FIFO level (0 to FIFO_DEPTH)

sc_fifo #(
  .DATA_WIDTH   ( 8            ),
  .FIFO_DEPTH   ( FIFO_DEPTH   ),
  .ALMOST_FULL  ( FIFO_DEPTH-1 ),    // Set to max sensitivity (will be overridden)
  .ALMOST_EMPTY ( 1            ),    // Set to max sensitivity (will be overridden)
  .MEM_STYLE    ( "BLOCK"      )
) tx_fifo (
  .clk    ( clk                     ),
  .rst    ( !rst_n || tx_fifo_reset ),
  .wr     ( tx_fifo_wr_safe         ),
  .din    ( tx_fifo_din             ),
  .full   ( tx_fifo_full            ),
  .afull  ( tx_fifo_afull_hw        ),
  .over   (                         ),
  .rd     ( tx_fifo_rd              ),
  .dout   ( tx_fifo_dout            ),
  .dval   ( tx_fifo_dval            ),
  .empty  ( tx_fifo_empty           ),
  .aempty ( tx_fifo_aempty_hw       ),
  .under  (                         ),
  .count  (                         )
);

// TX FIFO level counter
always_ff @(posedge clk) begin
  if (!rst_n || tx_fifo_reset) begin
    tx_fifo_level <= 0;
  end
  else begin
    case ({tx_fifo_wr_safe, tx_fifo_rd})
      2'b01: tx_fifo_level <= tx_fifo_level - 1;  // Read only
      2'b10: tx_fifo_level <= tx_fifo_level + 1;  // Write only
      2'b11: tx_fifo_level <= tx_fifo_level;      // Read and write
      2'b00: tx_fifo_level <= tx_fifo_level;      // No operation
    endcase
  end
end

// Programmable TX FIFO threshold comparisons
assign tx_fifo_afull  = (tx_fifo_level >= tx_afull_threshold);
assign tx_fifo_aempty = (tx_fifo_level <= tx_aempty_threshold);

// RX FIFO
logic [7:0] rx_fifo_din;
logic       rx_fifo_wr;
logic       rx_fifo_wr_safe;  // Safe write (prevents overflow)
logic [7:0] rx_fifo_dout;
logic       rx_fifo_rd;
logic       rx_fifo_dval;
logic       rx_fifo_afull;     // Programmable almost full
logic       rx_fifo_aempty;    // Programmable almost empty
logic       rx_fifo_afull_hw;  // Hardware almost full (from FIFO)
logic       rx_fifo_aempty_hw; // Hardware almost empty (from FIFO)
logic [8:0] rx_fifo_level;     // Current FIFO level (0 to FIFO_DEPTH)

sc_fifo #(
  .DATA_WIDTH   ( 8            ),
  .FIFO_DEPTH   ( FIFO_DEPTH   ),
  .ALMOST_FULL  ( FIFO_DEPTH-1 ),    // Set to max sensitivity (will be overridden)
  .ALMOST_EMPTY ( 1            ),    // Set to max sensitivity (will be overridden)
  .MEM_STYLE    ( "BLOCK"      )
) rx_fifo (
  .clk    ( clk                     ),
  .rst    ( !rst_n || rx_fifo_reset ),
  .wr     ( rx_fifo_wr_safe         ),
  .din    ( rx_fifo_din             ),
  .full   ( rx_fifo_full            ),
  .afull  ( rx_fifo_afull_hw        ),
  .over   (                         ),
  .rd     ( rx_fifo_rd              ),
  .dout   ( rx_fifo_dout            ),
  .dval   ( rx_fifo_dval            ),
  .empty  ( rx_fifo_empty           ),
  .aempty ( rx_fifo_aempty_hw       ),
  .under  (                         ),
  .count  (                         )
);

// RX FIFO level counter
always_ff @(posedge clk) begin
  if (!rst_n || rx_fifo_reset) begin
    rx_fifo_level <= 0;
  end
  else begin
    case ({rx_fifo_wr_safe, rx_fifo_rd})
      2'b01: rx_fifo_level <= rx_fifo_level - 1;  // Read only
      2'b10: rx_fifo_level <= rx_fifo_level + 1;  // Write only
      2'b11: rx_fifo_level <= rx_fifo_level;      // Read and write
      2'b00: rx_fifo_level <= rx_fifo_level;      // No operation
    endcase
  end
end

// Programmable RX FIFO threshold comparisons
assign rx_fifo_afull  = (rx_fifo_level >= rx_afull_threshold);
assign rx_fifo_aempty = (rx_fifo_level <= rx_aempty_threshold);

assign rx_fifo_not_empty = !rx_fifo_empty;

// UART Core
logic [7:0] uart_tx_data;
logic       uart_tx_valid;
logic       uart_tx_ready;
logic       uart_tx_done;
logic [7:0] uart_rx_data;
logic       uart_rx_valid;
logic       uart_rx_ready;
logic       uart_rx_done;

uart #(
  .OVERSAMPLE                ( 8                         ),
  .DATA_WIDTH                ( 8                         )
) uart_core (
  .clk                       ( clk                       ),
  .rst_n                     ( rst_n                     ),
  .baud_div                  ( baud_div                  ),
  .tx_en                     ( tx_enable && uart_enable  ),
  .rx_en                     ( rx_enable && uart_enable  ),
  .loopback_en               ( loopback_en               ),
  .data_width                ( data_width_actual         ),
  .parity_mode               ( parity_mode               ),
  .stop_bits_mode            ( stop_bits_mode            ),
  .tx_data                   ( uart_tx_data              ),
  .tx_valid                  ( uart_tx_valid             ),
  .tx_ready                  ( uart_tx_ready             ),
  .rx_data                   ( uart_rx_data              ),
  .rx_valid                  ( uart_rx_valid             ),
  .rx_ready                  ( uart_rx_ready             ),
  .rx_parity_error           ( rx_parity_error           ),
  .rx_frame_error            ( rx_frame_error            ),
  .uart_tx                   ( uart_tx                   ),
  .uart_rx                   ( uart_rx                   ),
  .rts_n                     ( uart_rts                ),
  .glitch_window_width       ( glitch_window_width       ),
  .glitch_start_bit_majority ( glitch_start_bit_majority ),
  .glitch_data_bit_majority  ( glitch_data_bit_majority  ),
  .glitch_filter_en          ( glitch_filter_en          ),
  .glitch_error              ( rx_glitch_error           )
);


// TX FIFO to UART connection
// CTS (Clear To Send) check: Active LOW - CTS=0 means remote ready, CTS=1 means not ready
// When flow control enabled: only transmit when CTS=0 (remote is ready)
// When flow control disabled: always transmit
logic cts_ok;
assign cts_ok = flow_control_en ? !uart_cts : 1'b1;
assign uart_tx_data = tx_fifo_dout;
assign uart_tx_valid = tx_fifo_dval && cts_ok;  // Gate transmission by CTS
assign tx_fifo_rd = uart_tx_ready && !tx_fifo_empty && !tx_fifo_rd_reg && cts_ok;

// Overflow detection and safe write logic
assign tx_overflow = tx_fifo_wr && tx_fifo_full; // Attempted write to full FIFO
assign rx_overflow = rx_fifo_wr && rx_fifo_full; // Attempted write to full FIFO
assign tx_fifo_wr_safe = tx_fifo_wr && !tx_fifo_full;  // Only write when not full
assign rx_fifo_wr_safe = rx_fifo_wr && !rx_fifo_full;  // Only write when not full

// RX UART to FIFO connection
assign rx_fifo_din = uart_rx_data;
assign rx_fifo_wr = uart_rx_valid;  // Always attempt write, but safe logic prevents overflow
assign uart_rx_ready = rx_fifo_wr_safe;

// Interrupt status generation and clearing
// Overflow, parity error, and frame error interrupts are sticky (latched) - need to be cleared explicitly
// logic tx_overflow_sticky, rx_overflow_sticky;
logic clear_tx_fifo_empty_sticky;
logic clear_tx_fifo_afull_sticky;
logic clear_tx_fifo_aempty_sticky;
logic clear_rx_parity_error_sticky;
logic clear_rx_frame_error_sticky;
logic clear_tx_overflow_sticky;
logic clear_rx_overflow_sticky;
logic clear_rx_glitch_error_sticky;

// Interrupt clear register (ctrl_reg[3]) - write-only, user clears
assign clear_tx_fifo_empty_sticky   = ctrl_reg[3][0];
assign clear_tx_fifo_afull_sticky   = ctrl_reg[3][1];
assign clear_tx_fifo_aempty_sticky  = ctrl_reg[3][2];
assign clear_rx_parity_error_sticky = ctrl_reg[3][3];
assign clear_rx_frame_error_sticky  = ctrl_reg[3][4];
assign clear_tx_overflow_sticky     = ctrl_reg[3][5];
assign clear_rx_overflow_sticky     = ctrl_reg[3][6];
assign clear_rx_glitch_error_sticky = ctrl_reg[3][7];

// Sticky overflow, parity error, and frame error interrupts
always_ff @(posedge clk) begin
  if (!rst_n) begin
    tx_fifo_empty_sticky     <= 1'b0;
    tx_fifo_afull_sticky     <= 1'b0;
    tx_fifo_aempty_sticky    <= 1'b0;
    rx_parity_error_sticky   <= 1'b0;
    rx_frame_error_sticky    <= 1'b0;
    tx_overflow_sticky       <= 1'b0;
    rx_overflow_sticky       <= 1'b0;
    rx_glitch_error_sticky   <= 1'b0;
  end
  else begin
    if (tx_fifo_empty)
      tx_fifo_empty_sticky <= 1'b1;
    else if (clear_tx_fifo_empty_sticky || tx_fifo_reset)
      tx_fifo_empty_sticky <= 1'b0;
    if (tx_fifo_afull)
      tx_fifo_afull_sticky <= 1'b1;
    else if (clear_tx_fifo_afull_sticky || tx_fifo_reset)
      tx_fifo_afull_sticky <= 1'b0;
    if (tx_fifo_aempty)
      tx_fifo_aempty_sticky <= 1'b1;
    else if (clear_tx_fifo_aempty_sticky || tx_fifo_reset)
      tx_fifo_aempty_sticky <= 1'b0;
    if (rx_parity_error)
      rx_parity_error_sticky <= 1'b1;
    else if (clear_rx_parity_error_sticky)
      rx_parity_error_sticky <= 1'b0;
    if (rx_frame_error)
      rx_frame_error_sticky <= 1'b1;
    else if (clear_rx_frame_error_sticky)
      rx_frame_error_sticky <= 1'b0;
    if (tx_overflow)
      tx_overflow_sticky <= 1'b1;
    else if (clear_tx_overflow_sticky || tx_fifo_reset)
      tx_overflow_sticky <= 1'b0;
    if (rx_overflow)
      rx_overflow_sticky <= 1'b1;
    else if (clear_rx_overflow_sticky || rx_fifo_reset)
      rx_overflow_sticky <= 1'b0;
    if (rx_glitch_error)
      rx_glitch_error_sticky <= 1'b1;
    else if (clear_rx_glitch_error_sticky)
      rx_glitch_error_sticky <= 1'b0;
  end
end

// Interrupt status generation (level-sensitive for FIFO status, sticky for errors)
assign tx_empty_int_status         = tx_fifo_empty_sticky     & tx_empty_int_en        ;
assign tx_afull_int_status         = tx_fifo_afull_sticky     & tx_afull_int_en        ;
assign tx_aempty_int_status        = tx_fifo_aempty_sticky    & tx_aempty_int_en       ;
assign rx_not_empty_int_status     = rx_fifo_not_empty        & rx_not_empty_int_en    ;
assign rx_glitch_error_int_status  = rx_glitch_error_sticky   & rx_glitch_error_int_en ;

// Combined interrupt output
logic [4:0] uart_int_status_pulse;
edge_to_pulse #(
  .WIDTH(5),
  .PULSE_WIDTH(1),
  .EDGE_TYPE("RISING")
) uart_int_pulse_gen (
  .clk(clk),
  .rst(rst_n),
  .i_edge({tx_empty_int_status, tx_afull_int_status, tx_aempty_int_status, rx_not_empty_int_status, rx_glitch_error_int_status}),
  .o_pulse(uart_int_status_pulse)
);

assign uart_interrupt = |uart_int_status_pulse;

// Flow control output assignment
// RTS (Request to Send) - indicates readiness to receive data
// Active LOW: RTS=0 means we're ready to receive, RTS=1 means stop sending
// When UART/RX disabled (including reset default): RTS = '1' (not ready, active low)
// When flow control enabled: 
//   - RTS = '0' (ready) when RX FIFO has space (below almost full threshold)
//   - RTS = '1' (not ready) when RX FIFO is almost full/full or UART/RX disabled
logic rx_almost_full;
assign rx_almost_full = (rx_fifo_level >= rx_afull_threshold);
assign uart_rts = flow_control_en ? rx_almost_full || !(rx_enable && uart_enable) : !(rx_enable && uart_enable);

// Data Register Access (TX/RX Data registers)
logic [31:0] data_read_data;
logic        tx_fifo_wr_reg;
logic        rx_fifo_rd_reg;
logic        data_pready;
logic [1:0]  data_pready_reg;
logic        apb_data_active_reg;
logic        apb_data_edge;

// Detect rising edge of APB data transaction
always_ff @(posedge clk) begin
  if (!rst_n) begin
    apb_data_active_reg <= 1'b0;
  end
  else begin
    apb_data_active_reg <= (data_apb_m2s.psel && data_apb_m2s.penable);
  end
end

// Last RX byte capture logic
always_ff @(posedge clk) begin
  if (!rst_n) begin
    last_rx_byte  <= 8'h00;
    last_rx_valid <= 1'b0;
  end
  else begin
    // Clear last RX valid if clear bit is set
    if (last_rx_clear) begin
      last_rx_valid <= 1'b0;
      last_rx_byte  <= 8'h00;
    // Capture the byte whenever we successfully read from RX FIFO
    end
    else if (rx_fifo_dval) begin
      last_rx_byte  <= rx_fifo_dout;
      last_rx_valid <= 1'b1;
    end
  end
end

assign apb_data_edge = (data_apb_m2s.psel && data_apb_m2s.penable) && !apb_data_active_reg;

always_ff @(posedge clk) begin
  if (!rst_n) begin
    data_read_data  <= 32'h0;
    tx_fifo_wr_reg  <= 1'b0;
    rx_fifo_rd_reg  <= 1'b0;
    data_pready_reg <= 'b0;
    data_pready     <= 1'b0;
    tx_fifo_rd_reg  <= 1'b0;
  end
  else begin
    data_read_data  <= 32'h0;
    tx_fifo_wr_reg  <= 1'b0;
    rx_fifo_rd_reg  <= 1'b0;
    data_pready_reg <= 'b0;
    data_pready     <= 1'b0;
    tx_fifo_rd_reg  <= tx_fifo_rd;

    if (data_apb_m2s.psel && data_apb_m2s.penable) begin
      data_pready     <= data_pready_reg[1];
      data_pready_reg <= {data_pready_reg[0], 1'b1};
      case (data_apb_m2s.paddr[7:0])
        8'h40: begin // TX Data register
          if (data_apb_m2s.pwrite && apb_data_edge) begin
            tx_fifo_wr_reg <= 1'b1;  // Always set on write, overflow detection will handle full case
          end
          else if (!data_apb_m2s.pwrite) begin
            data_read_data <= {24'h0, tx_fifo_full ? 8'hFF : 8'h00};
          end
        end
        8'h44: begin // RX Data register
          if (!data_apb_m2s.pwrite && !rx_fifo_empty && apb_data_edge) begin
            rx_fifo_rd_reg <= 1'b1;
          end
          else if (!data_apb_m2s.pwrite) begin
            data_read_data <= {24'h0, rx_fifo_dout};
          end
        end
        8'h48: begin // Last RX Data register (read-only)
          if (!data_apb_m2s.pwrite) begin
            data_read_data <= last_rx_valid ? {24'h0, last_rx_byte} : 32'h0;
          end
        end
        default: begin
          if (!data_apb_m2s.pwrite) begin
            data_read_data <= 32'hDEADBEEF;
          end
        end
      endcase
    end
  end
end

// FIFO write/read signals
assign tx_fifo_wr  = tx_fifo_wr_reg;  // This will be protected by tx_fifo_wr_safe in FIFO instantiation
assign tx_fifo_din = data_apb_m2s.pwdata[7:0];
assign rx_fifo_rd  = rx_fifo_rd_reg;

// Data APB response
assign data_apb_s2m.pready = data_pready;
assign data_apb_s2m.prdata = data_read_data;
assign data_apb_s2m.pserr  = 1'b0;

// UART TX is busy if FSM not idle, FIFO read valid, or FIFO is not empty
assign uart_tx_busy = !uart_tx_ready || uart_tx_valid || !tx_fifo_empty;
assign uart_rx_busy = (uart_rx_valid || !rx_fifo_empty); // FIXME: @hderbyshire
assign uart_busy    = uart_tx_busy || uart_rx_busy;

endmodule
