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

module spi_ctrl_fsm
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter NUM_INST  = 1,
  parameter RAM_DEPTH = 128
)
(
  input                           i_aclk,
  input                           i_arst,
  input  apb_m2s                  i_apb_m2s,
  output apb_s2m                  o_apb_s2m,

  output                          o_busy,
  output [$clog2(NUM_INST)-1:0]   o_bus_en,
  //Spi Ports
  output                          CS_N,
  output                          SCK,
  output                          SDIO_en,
  input  [3:0]                    SDIO_in,
  output [3:0]                    SDIO_out
);

logic [31:0] ctrl_reg [5];
logic [31:0] stat_reg [1];

logic     isCtrlAddr;
apb_m2s   ctrl_apb_m2s;
apb_s2m   ctrl_apb_s2m;
apb_m2s   db_apb_m2s;
apb_s2m   db_apb_s2m;

s_apb_reg #(
  .N_CTRL    ( 5              ),
  .N_STAT    ( 1              ),
  .W_OFST    ( w_ofst         ),
  .SAME_CLK  ( 1              )
) u_reg_map  (
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

typedef enum logic [3:0] {
  SPI_IDLE,
  SPI_CMD,
  SPI_WRITE,
  SPI_TURN,
  SPI_READ,
  SPI_DONE
} spi_states;
spi_states state,state_nxt;


//------------------------------------------------------------------------------------------------//
// Register Map
//------------------------------------------------------------------------------------------------//

logic [15:0]   num_wr_bytes   ;
logic [15:0]   num_rd_bytes   ;
logic [15:0]   clk_cnt        ;
logic          fsm_start      ;
logic          cmd_valid      ;
logic          spi_done       ;
logic          cmd_err        ;


// Write Spi Interface signals
logic [7:0] num_cmd_bytes;
logic [7:0] turnaround_len ;

logic [3:0]   prescaler      ;
logic [1:0]   spi_mode       ;
logic [2:0]   targ_sel       ;
logic [1:0]   spi_width      ;

logic  [$clog2(NUM_INST)-1:0] reg_bus_en;

assign fsm_start         = ctrl_reg[0][0];
assign reg_bus_en        = ctrl_reg[1][$clog2(NUM_INST):0];
assign num_wr_bytes      = ctrl_reg[2][15:0];
assign num_rd_bytes      = ctrl_reg[2][31:16];
assign prescaler         = ctrl_reg[3][3:0];
assign spi_mode          = ctrl_reg[3][5:4];
assign spi_width         = ((state == SPI_WRITE) || (state == SPI_READ)) ? ctrl_reg[3][9:8] : 2'b0;
assign turnaround_len    = ctrl_reg[4][7:0];
assign num_cmd_bytes     = ctrl_reg[4][15:8];

assign stat_reg[0][0]    = o_busy;
assign stat_reg[0][1]    = cmd_err;
assign stat_reg[0][4]    = spi_done;
assign stat_reg[0][31:5] = '0;
assign stat_reg[0][3:2]  = '0;

assign ctrl_apb_m2s.psel    = (isCtrlAddr && i_apb_m2s.psel);
assign ctrl_apb_m2s.penable = i_apb_m2s.penable;
assign ctrl_apb_m2s.paddr   = i_apb_m2s.paddr  ;
assign ctrl_apb_m2s.pwdata  = i_apb_m2s.pwdata ;
assign ctrl_apb_m2s.pwrite  = i_apb_m2s.pwrite ;

assign db_apb_m2s.psel      = (!isCtrlAddr && i_apb_m2s.psel);
assign db_apb_m2s.penable   = i_apb_m2s.penable;
assign db_apb_m2s.paddr     = {24'h0,i_apb_m2s.paddr[7:0]};
assign db_apb_m2s.pwdata    = i_apb_m2s.pwdata ;
assign db_apb_m2s.pwrite    = i_apb_m2s.pwrite ;
assign o_apb_s2m            = (isCtrlAddr) ? ctrl_apb_s2m : db_apb_s2m;
assign o_bus_en = reg_bus_en;


//------------------------------------------------------------------------------------------------//
// Data Buffer
//------------------------------------------------------------------------------------------------//
logic        db_wr_en;
logic [31:0] db_wr_data;
logic [6:0]  db_addr;
logic [31:0] db_rd_data;
logic        rd_data_valid;
logic        rd_data_valid_q;
logic [7:0]  rd_data;
logic [8:0]  spi_db_addr        ;
logic [8:0]  spi_db_rd_addr     ;
logic [8:0]  transaction_cnt    ;
logic [8:0]  rd_transaction_cnt ;
logic [7:0]  spi_db_data        ;
logic        ack_rise ;
logic        cmd_ack_prev;
logic        busy;
logic        spi_busy;
logic        fsm_done;
logic        cmd_ack;
logic        read;
logic        db_pready;

always_comb begin
  db_wr_en = ((db_apb_m2s.psel && db_apb_m2s.penable && db_apb_m2s.pwrite) || (rd_data_valid_q));
  if (o_busy || fsm_start) begin
    if (rd_data_valid_q) begin
      db_wr_data = (spi_db_rd_addr[1:0] == 2'b11) ? {rd_data, db_rd_data[23:0]} :
                    (spi_db_rd_addr[1:0] == 2'b10) ? {db_rd_data[31:24], rd_data, db_rd_data[15:0]} :
                    (spi_db_rd_addr[1:0] == 2'b01) ? {db_rd_data[31:16], rd_data, db_rd_data[7:0]} :
                                                    {db_rd_data[31:8],  rd_data} ;
    end
    else begin
      db_wr_data = 8'h0;
    end
    db_addr = (rd_data_valid || rd_data_valid_q) ? spi_db_rd_addr[7:2] : spi_db_addr[7:2];
  end
  else begin
    db_addr    = db_apb_m2s.paddr[7:2];
    db_wr_data = db_apb_m2s.pwdata;
  end
end

always@(posedge i_aclk) begin
  if (i_arst) begin
    db_pready       <= 1'b0;
    rd_data_valid_q <= '0;
  end
  else begin
    db_pready       <= (db_apb_m2s.psel && db_apb_m2s.penable);
    rd_data_valid_q <= rd_data_valid;
  end
end

dp_ram #(
  .DATA_WIDTH ( 32                              ),
  .RAM_DEPTH  ( RAM_DEPTH                       ),
  .RAM_TYPE   ( "SIMPLE"                        ),
  .MEM_STYLE  ( "BLOCK"                         )
) data_buffer (
  .clk_a      ( i_aclk                          ),
  .en_a       ( !i_arst                         ),
  .we_a       ( db_wr_en                        ),
  .din_a      ( db_wr_data                      ),
  .addr_a     ( db_addr[$clog2(RAM_DEPTH)-1:0]  ),
  .dout_a     ( db_rd_data                      ),
  .clk_b      ( '0                              ),
  .en_b       ( '0                              ),
  .we_b       ( '0                              ),
  .din_b      ( '0                              ),
  .addr_b     ( '0                              ),
  .dout_b     (                                 )
);
assign isCtrlAddr    = !i_apb_m2s.paddr[8];
assign db_apb_s2m.pready = db_pready && (db_apb_m2s.psel && db_apb_m2s.penable); // Delay 1 clock cycle
assign db_apb_s2m.prdata = (!o_busy && (db_apb_m2s.psel && db_apb_m2s.penable)) ? db_rd_data : '0;
assign db_apb_s2m.pserr  = '0;
//------------------------------------------------------------------------------------------------//
// SPI Engine
//------------------------------------------------------------------------------------------------//

// Read for SPI interface
assign spi_db_data = db_rd_data[spi_db_addr[1:0]*8+:8];  //spi_db_data[7:2] selects the dword, spi_db_data[1:0] selects the byte
assign spi_db_rd_addr = spi_db_addr - 1'b1;

// Interface
logic        start          ;
logic        write          ;
logic        turnaround     ;
logic        stop           ;

spi #(
  .NUM_TARGETS     ( 1                )
) spi (
  .clk             ( i_aclk           ),
  .rst_n           ( !i_arst          ),
  .prescaler       ( prescaler        ),
  .spi_mode        ( spi_mode         ),
  .targ_sel        ( '0               ),
  .start           ( start            ),
  .write           ( write            ),
  .turnaround      ( turnaround       ),
  .read            ( read             ),
  .stop            ( stop             ),
  .spi_width       ( spi_width        ),
  .turnaround_len  ( turnaround_len   ),
  .wr_data         ( spi_db_data      ),
  .cmd_ack         ( cmd_ack          ),
  .rd_data         ( rd_data          ),
  .rd_data_valid   ( rd_data_valid    ),
  .busy            ( spi_busy         ),
  .CS_N            ( CS_N             ),
  .SCK             ( SCK              ),
  .SDIO_en         ( SDIO_en          ),
  .SDIO_in         ( SDIO_in          ),
  .SDIO_out        ( SDIO_out         )
);

//------------------------------------------------------------------------------------------------//
// SPI Interface FSM
//------------------------------------------------------------------------------------------------//

assign ack_rise = ({cmd_ack_prev,cmd_ack} == 2'b01);
assign cmd_valid  = (num_wr_bytes >= num_cmd_bytes);

assign write      = ((state == SPI_CMD) || (state == SPI_WRITE));
assign turnaround = (state == SPI_TURN);
assign read       = ((state == SPI_CMD) || (state == SPI_READ));
assign start      = ((state == SPI_CMD) || (state == SPI_WRITE));

always_comb begin
  state_nxt = state;
  case(state)
    SPI_IDLE: begin
      if (fsm_start) begin
        if (cmd_valid) begin
          state_nxt = (num_cmd_bytes == 3'h0) ? 
                      (num_wr_bytes == '0)    ? SPI_READ  : 
                                                SPI_WRITE :
                                                 SPI_CMD  ;
        end
        else begin
          state_nxt = SPI_DONE;
        end
      end
    end
    SPI_CMD: begin
      if ((1+transaction_cnt) == num_cmd_bytes) begin
        state_nxt = (num_cmd_bytes != num_wr_bytes) ? SPI_WRITE :
                    (turnaround_len != '0 )         ? SPI_TURN  :
                    (num_rd_bytes != '0)            ? SPI_READ  :
                                                      SPI_DONE  ;
      end
    end
    SPI_WRITE: begin
      if ((1+transaction_cnt) == num_wr_bytes) begin
        state_nxt = (turnaround_len != 4'h0)        ? SPI_TURN  :
                    (num_rd_bytes != 8'h0)          ? SPI_READ  :
                                                      SPI_DONE  ;
      end
    end
    SPI_TURN: begin
      state_nxt =  (num_rd_bytes != 8'h0)          ? SPI_READ  :
                                                      SPI_DONE  ;
    end
    SPI_READ: begin
      if ((rd_transaction_cnt+1) == (num_rd_bytes+num_cmd_bytes)) begin
        state_nxt = SPI_DONE;
      end
    end
    SPI_DONE: begin
      if (!spi_busy && !fsm_start) begin
        state_nxt = SPI_IDLE;
      end
    end
  endcase
end



always @(posedge i_aclk) begin
  if (i_arst) begin
    state              <=  SPI_IDLE ;
    transaction_cnt    <= '0;
    cmd_ack_prev       <= '0;
    cmd_err            <= '0;
    busy               <= '0;
    rd_transaction_cnt <= '0;
  end
  else begin
    state              <= ((ack_rise) || ((state == SPI_IDLE) || (state == SPI_DONE))) ? state_nxt : state;
    transaction_cnt    <= ((state != SPI_WRITE) && (state!=SPI_CMD) && (state != SPI_READ)) ? '0 : transaction_cnt + ack_rise;
    rd_transaction_cnt <= (state == SPI_IDLE)                       ? '0:
                          (state == SPI_CMD) || (state == SPI_READ) ? rd_transaction_cnt + ack_rise:
                                                                      rd_transaction_cnt;
    cmd_ack_prev       <= cmd_ack;
    cmd_err            <= (!cmd_valid && fsm_start) ? '1 :
                            (state == SPI_IDLE)      ? '0 : cmd_err;
    busy               <= (!busy && (state == SPI_IDLE) && fsm_start && cmd_valid)          ? '1 :
                            (busy && (state == SPI_DONE) && !spi_busy) ? '0 : busy;
  end
end


assign spi_db_addr = ((state == SPI_READ) || (state == SPI_DONE)) ? rd_transaction_cnt : transaction_cnt;
assign stop = (state_nxt == SPI_DONE);
assign o_busy = busy;
assign spi_done = (state == SPI_DONE) && (!spi_busy);

endmodule

