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


module i2c_ctrl_fsm
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter NUM_INST  = 1,
  parameter RAM_DEPTH = 128
)
(
  // Interface
  input                           i_aclk,
  input                           i_arst,
  input  apb_m2s                  i_apb_m2s,
  output apb_s2m                  o_apb_s2m,

  output                          o_busy,
  output [$clog2(NUM_INST)-1:0]   o_bus_en,

  input                           i_start,
  output [7:0]                    o_data,
  output                          o_data_valid,
  //I2C Ports
  input                           scl_i,
  output                          scl_o,
  input                           sda_i,
  output                          sda_o
);

//------------------------------------------------------------------------------------------------//
// Register Map
//------------------------------------------------------------------------------------------------//

logic [31:0] ctrl_reg [6];
logic [31:0] stat_reg [1];

logic     isCtrlAddr;
apb_m2s   ctrl_apb_m2s;
apb_s2m   ctrl_apb_s2m;
apb_m2s   db_apb_m2s;
apb_s2m   db_apb_s2m;

localparam  [(6*32)-1:0] RST_VAL = {'0,32'h004C_4B40,{(4*32){1'b0}}};

s_apb_reg #(
  .N_CTRL    ( 6              ),
  .N_STAT    ( 1              ),
  .W_OFST    ( w_ofst         ),
  .RST_VAL   ( RST_VAL        ),
  .SAME_CLK  ( 1              )
) u_reg_map  (
  .i_aclk    ( i_aclk         ),
  .i_arst    ( i_arst         ),
  .i_apb_m2s ( ctrl_apb_m2s   ),
  .o_apb_s2m ( ctrl_apb_s2m   ),
  .i_pclk    ( i_aclk         ),
  .i_prst    ( i_arst         ),
  .o_ctrl    ( ctrl_reg       ),
  .i_stat    ( stat_reg       )
);

logic [15:0]   num_wr_bytes   ;
logic [15:0]   num_rd_bytes   ;
logic [9:0]    device_address ;
logic [15:0]   clk_cnt        ;
logic [31:0]   i2c_scl_timeout;
logic          is_10b_addr    ;
logic          fsm_start      ;
logic          i2c_nack       ;
logic          i2c_al_err     ;
logic          cmd_valid      ;
logic          i2c_done       ;
logic [7:0]    num_trans      ;


logic  [$clog2(NUM_INST)-1:0] reg_bus_en;

assign fsm_start         = ctrl_reg[0][0] && i_start;
assign is_10b_addr       = ctrl_reg[0][1];
assign device_address    = ctrl_reg[0][25:16];
assign reg_bus_en        = ctrl_reg[1][$clog2(NUM_INST):0];
assign num_wr_bytes      = ctrl_reg[2][15:0];
assign num_rd_bytes      = ctrl_reg[2][31:16];
assign clk_cnt           = ctrl_reg[3][15:0];
assign i2c_scl_timeout   = ctrl_reg[4];
assign num_trans         = ctrl_reg[5][7:0];

assign stat_reg[0][0]    = o_busy;
assign stat_reg[0][1]    = '0;
assign stat_reg[0][2]    = i2c_al_err;
assign stat_reg[0][3]    = i2c_nack;
assign stat_reg[0][4]    = i2c_done;
assign stat_reg[0][31:5] = '0;


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

typedef enum logic [3:0] {
  I2C_IDLE,
  I2C_10b_DEV_ADDR,
  I2C_DEV_ADDR,
  I2C_WRITE,
  I2C_RS_10b_DEV_ADDR,
  I2C_RS_DEV_ADDR,
  I2C_READ,
  I2C_PERI_NACK,
  I2C_DONE
} i2c_states;
i2c_states state,state_nxt, state_prev;

//------------------------------------------------------------------------------------------------//
// Data Buffer
//------------------------------------------------------------------------------------------------//
logic        db_wr_en;
logic        wr_en;
logic [31:0] db_wr_data;
logic [6:0]  db_addr;
logic [31:0] db_rd_data;
logic        rd_data_valid;
logic [7:0]  i2c_db_addr        ;
logic [8:0]  transaction_cnt    ;
logic [8:0]  wr_ptr             ;
logic [8:0]  rd_ptr             ;
logic [7:0]  i2c_db_data        ;
logic [7:0]  rd_data            ;
logic        ack_rise ;
logic        cmd_ack_prev;
logic        i2c_busy;
logic        idle;
logic        fsm_done;
logic        i2c_al;
logic        cmd_ack;
logic        read;
logic        db_pready;
logic [7:0]  trans_cnt;

always_comb begin
  db_wr_en = ((db_apb_m2s.psel && db_apb_m2s.penable && db_apb_m2s.pwrite) || (rd_data_valid));
  if (o_busy || (fsm_start && (state == I2C_IDLE))) begin
    if (cmd_ack && read) begin
      db_wr_data = (i2c_db_addr[1:0] == 2'b11) ? {rd_data, db_rd_data[23:0]} :
                    (i2c_db_addr[1:0] == 2'b10) ? {db_rd_data[31:24], rd_data, db_rd_data[15:0]} :
                    (i2c_db_addr[1:0] == 2'b01) ? {db_rd_data[31:16], rd_data, db_rd_data[7:0]} :
                                                  {db_rd_data[31:8],  rd_data} ;
    end
    else begin
      db_wr_data = 8'h0;
    end
    db_addr = i2c_db_addr[7:2];
  end
  else begin
    db_addr    = (db_apb_m2s.paddr[7:2]);
    db_wr_data = db_apb_m2s.pwdata;
  end
end

assign i2c_db_addr = (state == I2C_WRITE) ? wr_ptr : rd_ptr;

always @(posedge i_aclk) begin
  if (i_arst) begin
    db_pready <= 1'b0;
  end
  else begin
    db_pready <= (db_apb_m2s.psel && db_apb_m2s.penable);
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

assign isCtrlAddr    = (!i_apb_m2s.paddr[8]);
assign rd_data_valid = (ack_rise && read);
assign db_apb_s2m.pready = (db_apb_m2s.psel && db_apb_m2s.penable) && db_pready;  // Delay 1 clock cycle
assign db_apb_s2m.prdata = (!o_busy && (db_apb_m2s.psel && db_apb_m2s.penable)) ? db_rd_data : '0;
assign db_apb_s2m.pserr  = '0;

//------------------------------------------------------------------------------------------------//
// I2C Engine
//------------------------------------------------------------------------------------------------//

// Ctrl Inputs
logic       start;
logic       stop;
logic       write;
logic       ack_stretch;
logic       ack_in;
logic [7:0] din;
logic       busy;

// Status Outputs
logic       ack_out;

// Read for I2C interface
assign din = (state == I2C_10b_DEV_ADDR)    ? {5'b11110, device_address[9:8],1'b0}   : // CMD code + Addr + Write
             (state == I2C_DEV_ADDR)        ? ((is_10b_addr) ? {device_address[7:0]} : {device_address[6:0],1'b0}): // Addr + Write
             (state == I2C_RS_10b_DEV_ADDR) ? {5'b11110, device_address[9:8],1'b1}   :  // Addr + Read
             (state == I2C_RS_DEV_ADDR)     ? ((is_10b_addr) ? {device_address[7:0]} : {device_address[6:0],1'b1}): // Addr + Read
                                                         db_rd_data[i2c_db_addr[1:0]*8+:8];

i2c i2c_ctrl_byte_inst (
  .clk            ( i_aclk          ), // clock
  .rst            ( 1'b0            ), // synchronous active high reset
  .nReset         ( !i_arst         ), // asynchronous active low reset
  .ena            ( '1              ), // core enable signal
  .clk_cnt        ( clk_cnt         ), // 4x SCL (default)
  // control inputs
  .start          ( start           ),
  .stop           ( stop            ),
  .read           ( read            ),
  .write          ( write           ),
  .ack_stretch    ( '0              ),
  .ack_in         ( ack_in          ),
  .din            ( din             ),
  .i2c_scl_timeout( i2c_scl_timeout ),
  // status outputs
  .cmd_ack        ( cmd_ack         ),
  .ack_out        ( ack_out         ),
  .i2c_busy       ( i2c_busy        ),
  .i2c_al         ( i2c_al          ),
  .idle           ( idle            ),
  .dout           ( rd_data         ),
  // I2C signals
  .scl_i          ( scl_i           ),
  .scl_o          (                 ),
  .scl_oen        ( scl_o           ),
  .sda_i          ( sda_i           ),
  .sda_o          (                 ),
  .sda_oen        ( sda_o           )
);

//------------------------------------------------------------------------------------------------//
// I2C Interface FSM
//------------------------------------------------------------------------------------------------//


assign ack_rise = ({cmd_ack_prev,cmd_ack} == 2'b01);

assign cmd_valid   = (num_trans != '0) ? (num_wr_bytes >= num_rd_bytes) : '1;
assign write       = ((state == I2C_WRITE)    || (state == I2C_10b_DEV_ADDR) || (state == I2C_RS_10b_DEV_ADDR) ||
                      (state == I2C_DEV_ADDR) || (state == I2C_RS_DEV_ADDR));

assign read        = (state == I2C_READ);
assign start       = (is_10b_addr) ? ((state == I2C_10b_DEV_ADDR) || (state == I2C_RS_10b_DEV_ADDR)):
                                     ((state == I2C_DEV_ADDR    ) || (state == I2C_RS_DEV_ADDR));

always_comb begin
  state_nxt = state;
  if (ack_out && cmd_ack && (state != I2C_IDLE) && (state != I2C_DONE) && !read) begin
    state_nxt = (state == I2C_PERI_NACK) ? I2C_DONE : I2C_PERI_NACK;
  end
  else begin
    case(state)
      I2C_IDLE: begin
        if (fsm_start && cmd_valid) begin
          state_nxt  = (num_wr_bytes == '0) ?
                       (is_10b_addr       ) ? I2C_RS_10b_DEV_ADDR : I2C_RS_DEV_ADDR:
                       (is_10b_addr       ) ? I2C_10b_DEV_ADDR    : I2C_DEV_ADDR   ;
        end
      end
      I2C_10b_DEV_ADDR: begin
        state_nxt = I2C_DEV_ADDR;
      end
      I2C_DEV_ADDR: begin
        state_nxt = I2C_WRITE;
      end
      I2C_WRITE: begin
        if (transaction_cnt == (num_wr_bytes - 1)) begin
          state_nxt = (num_rd_bytes != '0) ? (is_10b_addr) ?  I2C_RS_10b_DEV_ADDR : I2C_RS_DEV_ADDR : I2C_DONE;
        end
      end
      I2C_RS_10b_DEV_ADDR: begin
        state_nxt = I2C_RS_DEV_ADDR;
      end
      I2C_RS_DEV_ADDR: begin
        state_nxt =  I2C_READ;
      end
      I2C_READ: begin
        state_nxt = (transaction_cnt == (num_rd_bytes-1)) ? I2C_DONE : I2C_READ;
      end
      I2C_PERI_NACK: begin
        state_nxt  =  I2C_DONE;
      end
      I2C_DONE: begin
        if (idle && (trans_cnt < num_trans)) begin
          state_nxt = I2C_IDLE;
        end
        else if (idle && !fsm_start) begin
          state_nxt       =  I2C_IDLE;
        end
        else begin
          state_nxt       =  I2C_DONE;
        end
      end
    endcase
  end
end


always@(posedge i_aclk) begin
  if (i_arst) begin
    state           <= I2C_IDLE;
    state_prev      <= I2C_IDLE;
    transaction_cnt <= '0;
    wr_ptr          <= '0;
    rd_ptr          <= '0;
    i2c_nack        <= '0;
    i2c_al_err      <= '0;
    cmd_ack_prev    <= '0;
    trans_cnt       <= '0;
    busy            <= '0;
  end
  else begin
    state           <= (i2c_al) ? I2C_DONE  : ((ack_rise) || ((state == I2C_IDLE) || (state == I2C_DONE) || (state == I2C_PERI_NACK)))
                                ? state_nxt : state;
    state_prev      <= state;
    transaction_cnt <= ((state != I2C_WRITE) && (state != I2C_READ)) ? '0 : transaction_cnt + ack_rise;
    wr_ptr          <= (state == I2C_WRITE) ? wr_ptr + ack_rise : (!fsm_start && !busy) ? '0 : wr_ptr;
    rd_ptr          <= (state == I2C_READ)  ? rd_ptr + ack_rise : (!fsm_start && !busy) ? '0 : rd_ptr;
    i2c_nack        <= (state == I2C_PERI_NACK) ? 1'b1 : (state == I2C_IDLE) ? 1'b0 : i2c_nack;
    i2c_al_err      <= (i2c_al) ? 1'b1 : (state == I2C_IDLE) ? 1'b0 : i2c_al_err;
    cmd_ack_prev    <= cmd_ack;
    trans_cnt       <=  ((state == I2C_IDLE) && !fsm_start            ) ? 'd1            :
                        ((state == I2C_IDLE && state_prev == I2C_DONE)) ? trans_cnt + 1  : trans_cnt;
    busy            <= (!busy && (state == I2C_IDLE) && fsm_start && cmd_valid)           ? '1 :
                        (busy && (state == I2C_DONE) && idle && (trans_cnt >= num_trans)) ? '0 : busy;
  end
end

assign stop = ((state_nxt == I2C_DONE) || (state_nxt == I2C_PERI_NACK)) && (state_nxt != state);
assign ack_in = ((state_nxt == I2C_DONE) && (state == I2C_READ));
assign o_busy = busy;
assign i2c_done = (state == I2C_DONE) && idle;

assign o_data = rd_data;
assign o_data_valid = rd_data_valid;


endmodule
