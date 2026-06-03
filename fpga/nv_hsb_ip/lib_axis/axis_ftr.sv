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


module axis_ftr #(
  parameter FTR_WIDTH = 32,
  parameter DWIDTH    = 64,
  parameter W_KEEP    = DWIDTH/8,
  parameter W_USER    = 1
)
(
  input                                       clk,
  input                                       rst,

  input           [FTR_WIDTH-1:0]             i_ftr,    // hold constant until tlast, FTR<=DWIDTH
  input                                       i_ftr_val,
  input                                       i_ftr_en,

  input                                       i_axis_rx_tvalid,
  input           [DWIDTH-1:0]                i_axis_rx_tdata,
  input                                       i_axis_rx_tlast,
  input           [W_USER-1:0]                i_axis_rx_tuser,
  input           [W_KEEP-1:0]                i_axis_rx_tkeep,
  output                                      o_axis_rx_tready,

  output                                      o_axis_tx_tvalid,
  output          [DWIDTH-1:0]                o_axis_tx_tdata,
  output                                      o_axis_tx_tlast,
  output          [W_USER-1:0]                o_axis_tx_tuser,
  output          [W_KEEP-1:0]                o_axis_tx_tkeep,
  input                                       i_axis_tx_tready
);

localparam                  FTR_BYTES  = FTR_WIDTH/8;
localparam                  FTR_CYCLES = ((FTR_BYTES-1)/W_KEEP)+1;
localparam                  NXT_KEEP_W = (W_KEEP<FTR_BYTES) ? FTR_BYTES + W_KEEP : W_KEEP+W_KEEP;
localparam [NXT_KEEP_W-1:0] TKEEP_VAL  = {'0,{(FTR_BYTES){1'b1}}};

logic [NXT_KEEP_W-1:0]     tkeep_final;   //tkeep track for first cycle of ftr+tlast
logic [NXT_KEEP_W-1:0]     r_tkeep;

assign tkeep_final = (TKEEP_VAL | (i_axis_rx_tkeep << FTR_BYTES));

//------------------------------------------------------------------------------------------------//
// FSM
//------------------------------------------------------------------------------------------------//

typedef enum logic [3:0] {
  APPEND_IDLE,
  APPEND_WAIT,
  APPEND_FTR
} append_states;
append_states append_state, append_state_nxt;

logic [9:0]             idx;
logic [W_KEEP-1:0]      tkeep;
logic [9:0]             r_idx;
logic [DWIDTH-1:0]      ftr;
logic                   tlast;
logic                   ftr_en_seen;
logic                   incr_state;
logic                   is_done;
logic [7:0]             cnt;

integer i;

always_comb begin
  idx = '0;
  for (i=0;i<W_KEEP;i=i+1) begin
    idx = idx + i_axis_rx_tkeep[i];
  end
end

always_comb begin
  append_state_nxt = append_state;
  case (append_state)
    APPEND_IDLE: begin
      if (i_axis_rx_tlast && i_axis_rx_tvalid && i_ftr_en) begin
        append_state_nxt = (!i_ftr_val && !ftr_en_seen)  ? APPEND_WAIT  :
                            (is_done)                     ? APPEND_IDLE  :   // Sent in same cycle
                                                            APPEND_FTR   ;   // Needs extra cycles
      end
    end
    APPEND_WAIT: begin
      append_state_nxt = (i_ftr_val || ftr_en_seen) ? (is_done) ? APPEND_IDLE :
                                                                  APPEND_FTR  :
                                                                  APPEND_WAIT ;
    end
    APPEND_FTR: begin
      append_state_nxt = (r_tkeep[W_KEEP+:W_KEEP] == '0) ? APPEND_IDLE : APPEND_FTR;
    end
    default: begin
      append_state_nxt = APPEND_IDLE;
    end
  endcase
end

always_comb begin
  ftr = '0;
  if ((append_state == APPEND_IDLE) && i_axis_rx_tlast && i_ftr_en) begin
    ftr = i_ftr << (idx << 3);
  end
  else if (append_state == APPEND_FTR) begin
    ftr = i_ftr >> (r_idx << 3);
  end
end


always_ff @(posedge clk) begin
  if (rst) begin
    append_state <= APPEND_IDLE;
    r_idx        <= '0;
    r_tkeep      <= '0;
    ftr_en_seen  <= '0;
    cnt          <= '0;
  end
  else begin
    if (o_axis_tx_tlast && i_axis_tx_tready) begin
      ftr_en_seen <= 1'b0;
    end
    else if (i_ftr_val) begin
      ftr_en_seen <= 1'b1;
    end

    if (incr_state) begin
      append_state <= append_state_nxt;
    end
    if (append_state == APPEND_FTR) begin
      if (i_axis_tx_tready) begin
        r_tkeep  <= r_tkeep >> W_KEEP;
        cnt      <= cnt + 1'b1;
        r_idx    <= (cnt*W_KEEP) + (W_KEEP-idx);
      end
    end
    else begin
      r_tkeep  <= tkeep_final >> W_KEEP;
      cnt      <= 'd1;
      r_idx    <= (W_KEEP-idx);
    end
  end
end

assign incr_state       = (i_axis_tx_tready && (i_axis_rx_tvalid || o_axis_tx_tlast));
assign is_done          = (tkeep_final[W_KEEP+:(NXT_KEEP_W-W_KEEP)] == '0);
assign o_axis_rx_tready = (append_state == APPEND_WAIT) ? (incr_state && (append_state_nxt != APPEND_WAIT))                       :
                                                          (i_axis_tx_tready && (append_state_nxt == APPEND_IDLE))                 ;
assign o_axis_tx_tuser  = i_axis_rx_tuser;
assign o_axis_tx_tlast  = (append_state_nxt == APPEND_IDLE) && ((i_axis_rx_tlast && i_axis_rx_tvalid) || (append_state != APPEND_IDLE));
assign o_axis_tx_tdata  = (((append_state != APPEND_FTR) ? i_axis_rx_tdata : '0) | ftr);
assign o_axis_tx_tkeep  = (append_state == APPEND_FTR)                      ? r_tkeep[0+:W_KEEP] :
                          (i_axis_rx_tlast && i_axis_rx_tvalid && i_ftr_en) ? tkeep_final[0+:W_KEEP]:
                                                                              i_axis_rx_tkeep;
assign o_axis_tx_tvalid = ((append_state == APPEND_IDLE) && (append_state_nxt != APPEND_WAIT))        ? i_axis_rx_tvalid:
                          (append_state == APPEND_WAIT) && (i_ftr_val || ftr_en_seen) && (incr_state) ? 1'b1            :
                          (append_state == APPEND_FTR)                                                ? 1'b1            :
                                                                                                        1'b0            ;

endmodule
