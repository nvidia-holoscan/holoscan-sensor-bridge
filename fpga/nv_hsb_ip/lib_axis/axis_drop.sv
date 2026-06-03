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

module axis_drop #(
    parameter DROP_WIDTH = 32,
    parameter DWIDTH     = 64,
    parameter W_KEEP     = DWIDTH/8,
    parameter W_USER     = 1
)
(
  input   logic                               clk,
  input   logic                               rst,

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

localparam                DROP_CYCLES = (DROP_WIDTH-1)/DWIDTH + 1;
localparam                VAL_BITS    = (DROP_WIDTH%DWIDTH) == 0 ? DWIDTH : (DROP_WIDTH%DWIDTH);
localparam                NXT_BITS    = DWIDTH-VAL_BITS;
localparam                NXT_BYTES   = NXT_BITS/8;
localparam [DWIDTH/8-1:0] TKEEP_VAL   = {'0,{(NXT_BITS/8){1'b1}}};

logic [W_KEEP*2-1:0] tkeep_final;   //tkeep for final 2 cycles
logic [W_KEEP-1:0]   r_tkeep;
logic [W_USER-1:0]   r_tuser;

assign tkeep_final = (TKEEP_VAL | (i_axis_rx_tkeep << NXT_BYTES));

//------------------------------------------------------------------------------------------------//
// FSM
//------------------------------------------------------------------------------------------------//

typedef enum logic [3:0] {
  STATE_IDLE,
  STATE_DROP,
  STATE_SEND_EVEN,
  STATE_SEND,
  STATE_SEND_PAD
} drop_states;
drop_states drop_state, drop_state_nxt;

logic [7:0]        cnt;
logic              one_cycle;
logic [DWIDTH-1:0] data_buffer;



always_comb begin
  drop_state_nxt = drop_state;
  case (drop_state)
    STATE_IDLE: begin
      drop_state_nxt = (i_axis_rx_tlast)        ? STATE_IDLE :
                        ((DROP_WIDTH >= DWIDTH)) ? STATE_DROP :
                                                  STATE_SEND ;
    end
    STATE_DROP: begin
      if (VAL_BITS == DWIDTH) begin
        drop_state_nxt = (i_axis_rx_tlast)        ? STATE_SEND_PAD  :
                          (cnt >= (DROP_CYCLES-2)) ? STATE_SEND_EVEN :
                                                    STATE_DROP      ;
      end
      else begin
        drop_state_nxt = (i_axis_rx_tlast)        ? STATE_SEND_PAD :
                          (cnt >= (DROP_CYCLES-2)) ? STATE_SEND     :
                                                    STATE_DROP     ;
      end
    end
    STATE_SEND_EVEN: begin
      if (i_axis_rx_tlast) begin
        drop_state_nxt = STATE_IDLE;
      end
    end
    STATE_SEND: begin
      if (i_axis_rx_tlast) begin
        drop_state_nxt = (tkeep_final[W_KEEP+:W_KEEP] == '0) ? STATE_IDLE : STATE_SEND_PAD;
      end
    end
    STATE_SEND_PAD: begin
      if (i_axis_tx_tready) begin
        drop_state_nxt = (!i_axis_rx_tvalid)                        ? STATE_IDLE     :
                          ((DROP_WIDTH >= DWIDTH))                   ? STATE_DROP     :
                          (i_axis_rx_tlast && (DROP_WIDTH < DWIDTH)) ? STATE_SEND_PAD : // back-to-back 1-beat pkt: output its data next cycle as pad
                                                                      STATE_SEND     ;
      end
    end
    default: begin
      drop_state_nxt = STATE_IDLE;
    end
  endcase
end


generate
  if (VAL_BITS == DWIDTH) begin
    assign o_axis_tx_tdata = (drop_state == STATE_IDLE)      ? '0                                                               :
                              (drop_state == STATE_DROP)      ? '0                                                               :
                              (drop_state == STATE_SEND_EVEN) ? i_axis_rx_tdata                                                  :
                                                                '0                                                               ;
  end
  else begin
    assign o_axis_tx_tdata = (one_cycle                   ) ? {'0,i_axis_rx_tdata[VAL_BITS+:NXT_BITS]}                         :
                              (drop_state == STATE_IDLE    ) ? '0                                                               :
                              (drop_state == STATE_DROP    ) ? '0                                                               :
                              (drop_state == STATE_SEND    ) ? {i_axis_rx_tdata[0+:VAL_BITS],data_buffer[VAL_BITS+:NXT_BITS]}   :
                              (drop_state == STATE_SEND_PAD) ? {'0,                          data_buffer[VAL_BITS+:NXT_BITS]}   :
                                                              '0                                                                ;
  end
endgenerate


// Control path with synchronous reset
always_ff @(posedge clk) begin
  if (rst) begin
    drop_state  <= STATE_IDLE;
    cnt         <= '0;
  end
  else begin
    if ((i_axis_rx_tvalid || (drop_state == STATE_SEND_PAD)) && o_axis_rx_tready) begin
      drop_state   <= drop_state_nxt;
      cnt          <= (drop_state == STATE_IDLE)     ? '0        :
                      (drop_state == STATE_SEND_PAD) ? 8'h0      : // back-to-back: beat during SEND_PAD not counted toward drop
                                                        cnt + 1'b1;
    end
  end
end

// Data path without reset
always_ff @(posedge clk) begin
  if (i_axis_rx_tvalid && i_axis_rx_tlast) begin
    r_tkeep      <= tkeep_final[W_KEEP+:W_KEEP];
    r_tuser      <= i_axis_rx_tuser;
  end
  if ((i_axis_rx_tvalid || (drop_state == STATE_SEND_PAD)) && o_axis_rx_tready) begin
    data_buffer  <= i_axis_rx_tdata;
  end
end

// Should synth away if drop width is greater than dwidth
assign one_cycle        = ((drop_state == STATE_IDLE) && i_axis_tx_tready && i_axis_rx_tlast && i_axis_rx_tvalid && (DROP_WIDTH < DWIDTH));

assign o_axis_rx_tready = i_axis_tx_tready;
assign o_axis_tx_tvalid = ((((drop_state == STATE_SEND) || (drop_state==STATE_SEND_EVEN)) &&
                          i_axis_rx_tvalid) || (drop_state == STATE_SEND_PAD) || one_cycle);
assign o_axis_tx_tuser  = (drop_state == STATE_SEND_PAD) ? r_tuser : i_axis_rx_tuser;
assign o_axis_tx_tlast  = ((drop_state_nxt == STATE_IDLE && drop_state != STATE_IDLE) || one_cycle || (drop_state == STATE_SEND_PAD));
assign o_axis_tx_tkeep  = one_cycle                      ? tkeep_final[W_KEEP+:W_KEEP] :
                          (drop_state == STATE_SEND_PAD) ? r_tkeep :
                                                           tkeep_final[0+:W_KEEP];


endmodule
