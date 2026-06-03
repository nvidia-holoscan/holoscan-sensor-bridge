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

module eeprom_info
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter CLK_FREQ      = 100000000,
  parameter I2C_CLK_FREQ  = 400000,
  parameter REG_ADDR_BITS = 8
) (
  // clock and reset
  input              i_aclk,
  input              i_arst,
  // control
  input              i_init,
  // APB Interface
  output apb_m2s     o_apb_m2s,
  input  apb_s2m     i_apb_s2m,
  // EEPROM Fields
  output             eeprom_dval,
  output reg [47 :0] mac_addr,
  output reg [31 :0] ip_addr,
  output reg         ip_addr_vld,
  output reg [159:0] board_version,
  output reg [55 :0] board_sn,
  output             eeprom_crc_err
);


localparam [7:0] CLK_DIV_CNT = ((CLK_FREQ-1) / I2C_CLK_FREQ / 2) + 1;

`ifdef DV_REDUCED
  parameter SCALE_TIMERS_BY = 11;
`else
  parameter SCALE_TIMERS_BY = 0;
`endif

localparam                           FIVE_MS_CNT = CLK_FREQ / 150;
localparam [$clog2(FIVE_MS_CNT)-1:0] FIVE_MS     = FIVE_MS_CNT >> SCALE_TIMERS_BY;

localparam [15:0] NUM_WR_BYTE_RD = $ceil(REG_ADDR_BITS/8);

//------------------------------------------------------------------------------------------------//
// Sequence to fetch the ethernet address from the EEPROM
//------------------------------------------------------------------------------------------------//

logic [W_DATA-1:0] eth_data;
logic [W_ADDR-1:0] eth_addr;
logic              eth_wren;

apb_m2s            eth_apb_m2s;
apb_m2s            sys_apb_m2s;

assign sys_apb_m2s = 0;

typedef enum logic [2:0] {IDLE, SEQ_START, SEQ_BYTES, SEQ_WAIT, SEQ_DONE} states;
states state;

logic                           reg_err;
logic [3:0]                     reg_seq;
logic [$clog2(FIVE_MS_CNT)-1:0] wait_cntr;
logic [7:0]                     byte_cntr;
logic [31:0]                    read_data;
logic [7:0]                     eeprom_crc;
logic [31:0]                    reg_addr;

logic                           psel;
logic                           penable;
logic              [W_ADDR-1:0] paddr;
logic              [W_ADDR-1:0] pwdata;
logic                           pwrite;
logic                           pserr;

logic              [7:0]        init_delay;
logic                           init_en;

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    init_delay <= 32;
    init_en    <= 1'b0;
  end
  else begin
    if (i_init) begin
      init_delay <= (|init_delay) ? init_delay - 1'b1 : init_delay;
      init_en    <= (init_delay == 1);
    end
    else begin
      init_delay <= 32;
      init_en    <= 1'b0;
    end
  end
end

assign eeprom_dval = (state == SEQ_DONE) | ((init_delay == 1) & !i_init);

always_comb begin
  if (NUM_WR_BYTE_RD == 16'd2) begin
    reg_addr = {16'h0000, byte_cntr[7:0], 8'h00};
  end
  else begin
    reg_addr = {24'h0000_00, byte_cntr[7:0]};
  end
end

//------------------------------------------------------------------------------------------------//
// Register Sequence to set up the I2C to read 28 bytes at a time from the EEPROM
//------------------------------------------------------------------------------------------------//

assign eth_addr = (reg_seq == 4'h0) ? 32'h0300_0200 :  // Clear Start
                  (reg_seq == 4'h1) ? 32'h0300_020C :
                  (reg_seq == 4'h2) ? 32'h0300_0208 :
                  (reg_seq == 4'h3) ? 32'h0300_0210 : // Timeout
                  (reg_seq == 4'h4) ? 32'h0300_0300 :
                  (reg_seq == 4'h5) ? 32'h0300_0200 :
                  (reg_seq == 4'h6) ? 32'h0300_0280 : //Status
                  (reg_seq == 4'h7) ? 32'h0300_0300 :
                  (reg_seq == 4'h8) ? 32'h0300_0304 :
                  (reg_seq == 4'h9) ? 32'h0300_0308 :
                  (reg_seq == 4'hA) ? 32'h0300_030C :
                  (reg_seq == 4'hB) ? 32'h0300_0310 :
                  (reg_seq == 4'hC) ? 32'h0300_0314 :
                  (reg_seq == 4'hD) ? 32'h0300_0318 :
                                      32'h0300_031C ;

assign eth_data = (reg_seq == 4'h0) ? {32'h0000_0000} :                // clear Start
                  (reg_seq == 4'h1) ? {32'h0000_00, CLK_DIV_CNT} :     // clk_cnt
                  (reg_seq == 4'h2) ? {16'h0020, NUM_WR_BYTE_RD} :     // {num_rd_bytes, num_wr_bytes} read 28 bytes
                  (reg_seq == 4'h3) ? {32'h004C4B40} :                 // Timeout
                  (reg_seq == 4'h4) ? reg_addr :                       // EEPROM Read from address (Gigabit Ethernet MAC byte 0)
                  (reg_seq == 4'h5) ? 32'h0050_0001 :                  // Write CONTROL with device addr and start
                                      32'h0000_0000;                   // Read STATUS[0] Busy

assign eth_wren = (reg_seq == 4'h0) ? 1'b1 :  // clk_cnt
                  (reg_seq == 4'h1) ? 1'b1 :  // clk_cnt
                  (reg_seq == 4'h2) ? 1'b1 :  // {num_rd_bytes, num_wr_bytes} read 28 bytes
                  (reg_seq == 4'h3) ? 1'b1 :  // EEPROM Read from address (Gigabit Ethernet MAC byte 0)
                  (reg_seq == 4'h4) ? 1'b1 :  // Timeout
                  (reg_seq == 4'h5) ? 1'b1 :  // Write CONTROL with device addr and start
                                      1'b0;   // Read STATUS[0] Busy


always_ff @(posedge i_aclk) begin
  if (i_arst) begin

    state     <= IDLE;
    reg_seq   <= 0;
    wait_cntr <= 0;
    byte_cntr <= 0;

    psel      <= 0;
    penable   <= 0;
    paddr     <= 0;
    pwdata    <= 0;
    pwrite    <= 0;
    pserr     <= 0;

  end
  else begin

    pserr    <= i_apb_s2m.pserr & i_apb_s2m.pready;

    case (state)

      IDLE: begin
        reg_seq   <= 0;
        wait_cntr <= 0;
        byte_cntr <= 0;
        psel      <= 0;
        penable   <= 0;
        pwrite    <= 0;
        paddr     <= 0;
        pwdata    <= 0;
        if (init_en) begin
          state  <= SEQ_START;
          psel   <= 1;
          pwrite <= eth_wren;
          paddr  <= eth_addr;
          pwdata <= eth_data;
        end
      end

      SEQ_START: begin
        penable   <= 1 ^ i_apb_s2m.pready;
        if (penable & i_apb_s2m.pready) begin
          psel      <= 0;
          penable   <= 0;
          pwrite    <= 0;
          paddr     <= 0;
          pwdata    <= 0;
          wait_cntr <= 32;
          if (reg_seq == 4'h0 && byte_cntr[7:0] == 8'b11111111) begin
            state   <= SEQ_DONE;
          end
          else if (reg_seq == 4'd6 && i_apb_s2m.prdata[0] == 1) begin
            state   <= SEQ_WAIT;
          end
          else if (reg_seq > 4'h6) begin
            state   <= SEQ_BYTES;
            reg_seq <= reg_seq + 1'b1;
          end
          else begin
            state   <= SEQ_WAIT;
            reg_seq <= reg_seq + 1'b1;
          end
        end
      end

      SEQ_BYTES: begin
        byte_cntr <= byte_cntr + 1'b1;

        if (byte_cntr[7:0] == 8'b11111111) begin
          byte_cntr <= byte_cntr;
          state     <= SEQ_WAIT;
        end
        else if (byte_cntr[1:0] == 2'b11) begin
          state <= SEQ_WAIT;
        end
        if (reg_seq == 4'hF) begin
          reg_seq <= 4'h0;
        end
      end

      SEQ_WAIT: begin
        if (|wait_cntr) begin
          wait_cntr    <= wait_cntr - 1'b1;
        end
        if (~|wait_cntr) begin
          state   <= SEQ_START;
          psel    <= 1'b1;
          pwrite  <= eth_wren;
          paddr   <= eth_addr;
          pwdata  <= eth_data;
        end
      end

      SEQ_DONE: begin
        state          <= SEQ_DONE;
      end

      default: state <= IDLE;

    endcase
  end
end

assign eth_apb_m2s.psel    = psel;
assign eth_apb_m2s.penable = penable;
assign eth_apb_m2s.paddr   = paddr;
assign eth_apb_m2s.pwdata  = pwdata;
assign eth_apb_m2s.pwrite  = pwrite;

assign o_apb_m2s = (state != SEQ_DONE) ? eth_apb_m2s : sys_apb_m2s;

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    read_data <= 32'h00000000;
  end
  else if (penable & i_apb_s2m.pready) begin
    read_data <= i_apb_s2m.prdata[31:0];
  end
end

//------------------------------------------------------------------------------------------------//
// CRC Calculation
//------------------------------------------------------------------------------------------------//
  // CRC polynomial coefficients: x^8 + x^3 + x^2 + 1
  //                              0xB0 (hex)
  // CRC width:                   8 bits
  // CRC shift direction:         right (little endian)
  // Input word width:            8 bits

function automatic [7:0] nextCRC;
  input [7:0] data;
  input [7:0] crcIn;
  logic [7:0] odd;
  begin
    for (int i=0; i<8; i++) begin
      odd = (data ^ crcIn);
      crcIn >>= 1;
      data >>= 1;
      if (odd[0] == 1'b1) begin
        crcIn ^= 8'h8C;
      end
    end
    nextCRC = crcIn;
  end
endfunction

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    eeprom_crc <= 8'h00;
  end
  else if ((state == SEQ_BYTES) && (byte_cntr[1:0] == 2'b00)) begin
    eeprom_crc <= nextCRC(read_data[7:0], eeprom_crc);
  end
  else if ((state == SEQ_BYTES) && (byte_cntr[1:0] == 2'b01)) begin
    eeprom_crc <= nextCRC(read_data[15:8], eeprom_crc);
  end
  else if ((state == SEQ_BYTES) && (byte_cntr[1:0] == 2'b10)) begin
    eeprom_crc <= nextCRC(read_data[23:16], eeprom_crc);
  end
  else if ((state == SEQ_BYTES) && (byte_cntr[1:0] == 2'b11)) begin
    //Don't include last byte in CRC calc
    if (!(byte_cntr[7:2]==6'b111111)) begin
      eeprom_crc <= nextCRC(read_data[31:24], eeprom_crc);
    end
  end
end

assign eeprom_crc_err = (|eeprom_crc);

//------------------------------------------------------------------------------------------------//
// Board Version, EEPROM Location 20 to 39 MSB First
//------------------------------------------------------------------------------------------------//

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    board_version <= 160'h00000000_00000000_00000000_00000000_00000000;
  end
  else if ((byte_cntr[7:5] == 3'h0) && (byte_cntr[4:2] == 3'h5) && penable && i_apb_s2m.pready) begin
    board_version[159:152] <= i_apb_s2m.prdata[7:0];
    board_version[151:144] <= i_apb_s2m.prdata[15:8];
    board_version[143:136] <= i_apb_s2m.prdata[23:16];
    board_version[135:128] <= i_apb_s2m.prdata[31:24];
  end
  else if ((byte_cntr[7:5] == 3'h0) && (byte_cntr[4:2] == 3'h6) && penable && i_apb_s2m.pready) begin
    board_version[127:120] <= i_apb_s2m.prdata[7:0];
    board_version[119:112] <= i_apb_s2m.prdata[15:8];
    board_version[111:104] <= i_apb_s2m.prdata[23:16];
    board_version[103:96]  <= i_apb_s2m.prdata[31:24];
  end
  else if ((byte_cntr[7:5] == 3'h0) && (byte_cntr[4:2] == 3'h7) && penable && i_apb_s2m.pready) begin
    board_version[95:88]   <= i_apb_s2m.prdata[7:0];
    board_version[87:80]   <= i_apb_s2m.prdata[15:8];
    board_version[79:72]   <= i_apb_s2m.prdata[23:16];
    board_version[71:64]   <= i_apb_s2m.prdata[31:24];
  end
  else if ((byte_cntr[7:5] == 3'h1) && (byte_cntr[4:2] == 3'h0) && penable && i_apb_s2m.pready) begin
    board_version[63:56]   <= i_apb_s2m.prdata[7:0];
    board_version[55:48]   <= i_apb_s2m.prdata[15:8];
    board_version[47:40]   <= i_apb_s2m.prdata[23:16];
    board_version[39:32]   <= i_apb_s2m.prdata[31:24];
  end
  else if ((byte_cntr[7:5] == 3'h1) && (byte_cntr[4:2] == 3'h1) && penable && i_apb_s2m.pready) begin
    board_version[31:24]   <= i_apb_s2m.prdata[7:0];
    board_version[23:16]   <= i_apb_s2m.prdata[15:8];
    board_version[15:8]    <= i_apb_s2m.prdata[23:16];
    board_version[7:0]     <= i_apb_s2m.prdata[31:24];
  end
end

//------------------------------------------------------------------------------------------------//
// IP Address, EEPROM Location 60 to 64 MSB First
//------------------------------------------------------------------------------------------------//

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    ip_addr     <= 32'h00000000;
    ip_addr_vld <= 1'b0;
  end
  else if ((byte_cntr[7:5] == 3'h1) && (byte_cntr[4:2] == 3'h7) && penable && i_apb_s2m.pready) begin
    ip_addr[31:24] <= i_apb_s2m.prdata[7:0];
    ip_addr[23:16] <= i_apb_s2m.prdata[15:8];
    ip_addr[15:8]  <= i_apb_s2m.prdata[23:16];
    ip_addr[7:0]   <= i_apb_s2m.prdata[31:24];
  end
  else if ((byte_cntr[7:5] == 3'h2) && (byte_cntr[4:2] == 3'h0) && penable && i_apb_s2m.pready) begin
    ip_addr_vld <= i_apb_s2m.prdata[7:0] == 8'h17;
  end
end

//------------------------------------------------------------------------------------------------//
// MAC Address, EEPROM Location 68 to 73 MSB First
//------------------------------------------------------------------------------------------------//

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    mac_addr <= 48'hCAFEC0FFEE00;
  end
  else if ((byte_cntr[7:5] == 3'h2) && (byte_cntr[4:2] == 3'h1) && penable && i_apb_s2m.pready) begin
    mac_addr[47:40] <= i_apb_s2m.prdata[7:0];
    mac_addr[39:32] <= i_apb_s2m.prdata[15:8];
    mac_addr[31:24] <= i_apb_s2m.prdata[23:16];
    mac_addr[23:16] <= i_apb_s2m.prdata[31:24];
  end
  else if ((byte_cntr[7:5] == 3'h2) && (byte_cntr[4:2] == 3'h2) && penable && i_apb_s2m.pready) begin
    mac_addr[15:8] <= i_apb_s2m.prdata[7:0];
    mac_addr[7:0]  <= i_apb_s2m.prdata[15:8];
  end
end

//------------------------------------------------------------------------------------------------//
// Board Serial Number, EEPROM Location 74 to 80 MSB First
//------------------------------------------------------------------------------------------------//

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    board_sn <= 56'h563412_78563412;
  end
  else if ((byte_cntr[7:5] == 3'h2) && (byte_cntr[4:2] == 3'h2) && penable && i_apb_s2m.pready) begin
    board_sn[55:48] <= i_apb_s2m.prdata[23:16];
    board_sn[47:40] <= i_apb_s2m.prdata[31:24];
  end
  else if ((byte_cntr[7:5] == 3'h2) && (byte_cntr[4:2] == 3'h3) && penable && i_apb_s2m.pready) begin
    board_sn[39:32] <= i_apb_s2m.prdata[7:0];
    board_sn[31:24] <= i_apb_s2m.prdata[15:8];
    board_sn[23:16] <= i_apb_s2m.prdata[23:16];
    board_sn[15:8]  <= i_apb_s2m.prdata[31:24];
  end
  else if ((byte_cntr[7:5] == 3'h2) && (byte_cntr[4:2] == 3'h4) && penable && i_apb_s2m.pready) begin
    board_sn[7:0] <= i_apb_s2m.prdata[7:0];
  end
end


endmodule

