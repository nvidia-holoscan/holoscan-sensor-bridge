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

module axis_hdr #(
    parameter HDR_WIDTH = 224,
    parameter DWIDTH    = 64,
    parameter W_USER    = 1,
    parameter W_KEEP    = DWIDTH/8
)
(
  input   logic                               clk,
  input   logic                               rst,

  input   logic   [HDR_WIDTH-1:0]             i_hdr,
  input   logic                               i_hdr_en,

  input   logic                               i_axis_rx_tvalid,
  input   logic   [DWIDTH-1:0]                i_axis_rx_tdata,
  input   logic                               i_axis_rx_tlast,
  input   logic   [W_USER-1:0]                i_axis_rx_tuser,
  input   logic   [W_KEEP-1:0]                i_axis_rx_tkeep,
  output  logic                               o_axis_rx_tready,

  output  logic                               o_axis_tx_tvalid,
  output  logic   [DWIDTH-1:0]                o_axis_tx_tdata,
  output  logic                               o_axis_tx_tlast,
  output  logic   [W_USER-1:0]                o_axis_tx_tuser,
  output  logic   [W_KEEP-1:0]                o_axis_tx_tkeep,
  input   logic                               i_axis_tx_tready
);

localparam HDR_CYCLES     = (HDR_WIDTH-1)/DWIDTH + 1;
localparam HDR_DATA_BITS  = (HDR_WIDTH%DWIDTH) == 0 ? DWIDTH : (HDR_WIDTH%DWIDTH);
localparam HDR_PAD_BITS   = DWIDTH - HDR_DATA_BITS;
localparam VAL_BYTES      = HDR_DATA_BITS/8;
localparam CNT_WIDTH      = $clog2(HDR_CYCLES);

localparam [DWIDTH/8-1:0] TKEEP_VAL = {'0,{(HDR_DATA_BITS/8){1'b1}}};

logic [W_KEEP*2-1:0] tkeep_final;   //tkeep for final 2 cycles
logic [W_KEEP-1:0]   r_tkeep;

logic                  hdr_axis_tvalid;
logic                  hdr_axis_tlast;
logic [DWIDTH-1:0]     hdr_axis_tdata;
logic [W_KEEP-1:0]     hdr_axis_tkeep;
logic [W_USER-1:0]     hdr_axis_tuser;
logic                  hdr_axis_tready;

//------------------------------------------------------------------------------------------------//
// Register Buffer
//------------------------------------------------------------------------------------------------//

logic                  append_tready;
logic                  tlast_flag;
logic                  hdr_bypass;

logic [W_USER-1:0] r_axis_tuser;
logic [DWIDTH-1:0] r_axis_tdata;

always_ff @(posedge clk) begin
  if (hdr_axis_tready) begin
    r_axis_tdata <= i_axis_rx_tdata;
  end
end

assign tkeep_final      = (TKEEP_VAL | (i_axis_rx_tkeep << VAL_BYTES));

//------------------------------------------------------------------------------------------------//
// Data MUX
//------------------------------------------------------------------------------------------------//

typedef enum logic [5:0] {
  APPEND_IDLE      = 6'b000001,
  APPEND_HDR       = 6'b000010,
  APPEND_HDR_DATA  = 6'b000100,
  APPEND_DATA      = 6'b001000,
  APPEND_DATA_PAD  = 6'b010000,
  APPEND_DATA_EVEN = 6'b100000
} append_states;
append_states append_state, append_state_nxt;

logic [HDR_WIDTH-1:0]  hdr;
logic [CNT_WIDTH  :0]  cnt;
logic                  start_fsm;
logic                  start_fsm_buf;
logic                  is_packet;


generate
  if (HDR_WIDTH < DWIDTH) begin  // If first cycle is hdr+data
    assign hdr_axis_tdata= (hdr_bypass)                     ? i_axis_rx_tdata:
                           (start_fsm_buf)                  ? {i_axis_rx_tdata[0+:HDR_PAD_BITS],hdr[cnt*DWIDTH+:HDR_DATA_BITS]}:
                           (append_state == APPEND_HDR_DATA)? {i_axis_rx_tdata[0+:HDR_PAD_BITS],hdr[cnt*DWIDTH+:HDR_DATA_BITS]}:
                           (append_state == APPEND_DATA)    ? {i_axis_rx_tdata[0+:HDR_PAD_BITS],r_axis_tdata[HDR_PAD_BITS+:HDR_DATA_BITS]}:
                           (append_state == APPEND_DATA_PAD)? {'0,r_axis_tdata[HDR_PAD_BITS+:HDR_DATA_BITS]}:
                                                              '0;
  end
  else if (HDR_DATA_BITS != DWIDTH) begin
    assign hdr_axis_tdata= (hdr_bypass)                      ? i_axis_rx_tdata:
                           (start_fsm_buf)                   ? hdr[cnt*DWIDTH+:DWIDTH]:
                           (append_state == APPEND_HDR)      ? hdr[cnt*DWIDTH+:DWIDTH]:
                           (append_state == APPEND_HDR_DATA) ? {i_axis_rx_tdata[0+:HDR_PAD_BITS],hdr[cnt*DWIDTH+:HDR_DATA_BITS]}:
                           (append_state == APPEND_DATA)     ? {i_axis_rx_tdata[0+:HDR_PAD_BITS],r_axis_tdata[HDR_PAD_BITS+:HDR_DATA_BITS]}:
                           (append_state == APPEND_DATA_PAD) ? {'0,r_axis_tdata[HDR_PAD_BITS+:HDR_DATA_BITS]}:
                                                              '0;
  end
  else begin
    assign hdr_axis_tdata  = (hdr_bypass)                          ? i_axis_rx_tdata:
                              (start_fsm_buf)                       ? hdr[cnt*DWIDTH+:DWIDTH]:
                              (append_state == APPEND_HDR)          ? hdr[cnt*DWIDTH+:DWIDTH]:
                              (append_state == APPEND_DATA_EVEN)    ? {i_axis_rx_tdata}:
                                                                    '0;
  end
endgenerate

assign start_fsm = ((append_state == APPEND_IDLE) && (append_state_nxt != APPEND_IDLE));
//------------------------------------------------------------------------------------------------//
// FSM
//------------------------------------------------------------------------------------------------//


always_comb begin
  append_state_nxt = append_state;
  case (append_state)
    APPEND_IDLE: begin
      if (i_axis_rx_tvalid && i_hdr_en) begin
        append_state_nxt = (HDR_WIDTH < DWIDTH) ? APPEND_HDR_DATA  : APPEND_HDR       ;
      end
    end
    APPEND_HDR: begin
      if (HDR_DATA_BITS == DWIDTH) begin
        if (cnt == (HDR_CYCLES-1)) begin
          append_state_nxt = APPEND_DATA_EVEN;
        end
      end
      else begin
        if (cnt == (HDR_CYCLES-2)) begin
          append_state_nxt =  APPEND_HDR_DATA;
        end
      end
    end
    APPEND_HDR_DATA: begin
      if (i_axis_rx_tlast) begin
        append_state_nxt = (tkeep_final[W_KEEP+:W_KEEP] == '0) ? APPEND_IDLE : APPEND_DATA_PAD;
      end
      else begin
        append_state_nxt = APPEND_DATA;
      end
    end
    APPEND_DATA_EVEN: begin
      if (i_axis_rx_tlast) begin
        append_state_nxt = APPEND_IDLE;
      end
    end
    APPEND_DATA: begin
      if (i_axis_rx_tlast) begin
        append_state_nxt = (tkeep_final[W_KEEP+:W_KEEP] == '0) ? APPEND_IDLE : APPEND_DATA_PAD;
      end
    end
    APPEND_DATA_PAD: begin
      append_state_nxt = APPEND_IDLE;
    end
    default: begin
      append_state_nxt = APPEND_IDLE;
    end
  endcase
end


always_ff @(posedge clk) begin
  if (rst) begin
    hdr           <= '0;
    cnt           <= '0;
    append_state  <= APPEND_IDLE;
    tlast_flag    <= '0;
    r_tkeep       <= '0;
    start_fsm_buf <= '0;
    r_axis_tuser  <= '0;
    is_packet     <= '0;
  end
  else begin
    start_fsm_buf <= start_fsm && (!hdr_axis_tready);
    if (hdr_axis_tready) begin
      append_state <= append_state_nxt;
      if ((append_state == APPEND_IDLE) || (append_state_nxt == APPEND_IDLE)) begin
        cnt <= '0;
      end
      else begin
        cnt <= cnt + 1'b1;
      end

      if (i_axis_rx_tlast && i_axis_rx_tvalid && (append_state != APPEND_HDR) && (append_state != APPEND_IDLE)) begin
        tlast_flag <= 1'b1;
      end
      else if (append_state == APPEND_IDLE) begin
        tlast_flag <= 1'b0;
      end

      if (i_axis_rx_tlast) begin
        r_tkeep      <= tkeep_final[W_KEEP+:W_KEEP];
      end
    end
    if (i_axis_rx_tvalid && !is_packet) begin
      hdr          <= i_hdr;
      is_packet    <= !(i_axis_rx_tlast && o_axis_rx_tready);
    end
    else if (i_axis_rx_tvalid && i_axis_rx_tlast && o_axis_rx_tready && is_packet) begin
      is_packet <= '0;
    end
    if (start_fsm) begin
      r_axis_tuser  <= i_axis_rx_tuser;
    end
  end
end

assign hdr_bypass        = (append_state == APPEND_IDLE) && !i_hdr_en;


assign hdr_axis_tvalid  = (hdr_bypass) ? i_axis_rx_tvalid : (append_state != APPEND_IDLE);
assign hdr_axis_tuser   = ((hdr_bypass) || (append_state == APPEND_IDLE)) ? i_axis_rx_tuser : r_axis_tuser;
assign hdr_axis_tlast   = (hdr_bypass) ? i_axis_rx_tlast  : (append_state_nxt == APPEND_IDLE && append_state != APPEND_IDLE);
assign hdr_axis_tkeep   = (hdr_bypass) ? i_axis_rx_tkeep  :
                          (!hdr_axis_tlast)                 ? '1                            :
                          (append_state == APPEND_DATA_PAD) ? r_tkeep                       :
                                                              tkeep_final[0+:W_KEEP]        ;

assign o_axis_rx_tready = (hdr_bypass)                  ? hdr_axis_tready :
                          (tlast_flag)                  ? '0              :
                          (append_state == APPEND_IDLE) ? '0              :
                          (append_state == APPEND_HDR)  ? '0              :
                                                          hdr_axis_tready ;

//------------------------------------------------------------------------------------------------//
// Register output
//------------------------------------------------------------------------------------------------//

axis_reg # (
  .DWIDTH             ( DWIDTH + W_KEEP + W_USER + 1                                     )
) u_axis_reg (
  .clk                ( clk                                                              ),
  .rst                ( rst                                                              ),
  .i_axis_rx_tvalid   ( hdr_axis_tvalid                                                  ),
  .i_axis_rx_tdata    ( {hdr_axis_tdata,hdr_axis_tlast,hdr_axis_tuser,hdr_axis_tkeep}    ),
  .o_axis_rx_tready   ( hdr_axis_tready                                                  ),
  .o_axis_tx_tvalid   ( o_axis_tx_tvalid                                                 ),
  .o_axis_tx_tdata    ( {o_axis_tx_tdata,o_axis_tx_tlast,o_axis_tx_tuser,o_axis_tx_tkeep}),
  .i_axis_tx_tready   ( i_axis_tx_tready                                                 )
);


endmodule
