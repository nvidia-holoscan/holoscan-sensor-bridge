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


module axis_mask #(
    parameter MASK_WIDTH = 32,
    parameter DWIDTH     = 64,
    parameter W_KEEP     = DWIDTH/8,
    parameter OP         = "OR"
)
(
  input   logic                               clk,
  input   logic                               rst,

  input   logic   [MASK_WIDTH-1:0]            i_mask,
  input   logic                               i_mask_en,

  input   logic                               i_axis_rx_tvalid,
  input   logic   [DWIDTH-1:0]                i_axis_rx_tdata,
  input   logic                               i_axis_rx_tlast,
  input   logic                               i_axis_rx_tuser,
  input   logic   [W_KEEP-1:0]                i_axis_rx_tkeep,
  output  logic                               o_axis_rx_tready,

  output  logic                               o_axis_tx_tvalid,
  output  logic   [DWIDTH-1:0]                o_axis_tx_tdata,
  output  logic                               o_axis_tx_tlast,
  output  logic                               o_axis_tx_tuser,
  output  logic   [W_KEEP-1:0]                o_axis_tx_tkeep,
  input   logic                               i_axis_tx_tready
);


localparam MASK_CYCLES     = (MASK_WIDTH-1)/DWIDTH + 1;
localparam W_MASK_CYCLES   = (MASK_CYCLES == 1) ? 1 : $clog2(MASK_CYCLES);

logic [MASK_CYCLES*DWIDTH-1:0] mask;
assign mask = {'0,i_mask};

//------------------------------------------------------------------------------------------------//
// FSM
//------------------------------------------------------------------------------------------------//

typedef enum logic {
  STATE_MASK,
  STATE_SEND
} mask_states;
mask_states mask_state, mask_state_nxt;

integer i;
logic [W_MASK_CYCLES-1:0] cnt;


always_comb begin
  mask_state_nxt = mask_state;
  case (mask_state)
    STATE_MASK: begin
      if (i_axis_rx_tvalid && i_axis_tx_tready && i_mask_en) begin
        mask_state_nxt = ((cnt >= (MASK_CYCLES-1)) && (!i_axis_rx_tlast)) ? STATE_SEND : STATE_MASK;
      end
    end
    STATE_SEND: begin
      if (i_axis_rx_tvalid && i_axis_tx_tready && i_axis_rx_tlast) begin
        mask_state_nxt = STATE_MASK;
      end
    end
    default: begin
      mask_state_nxt = STATE_MASK;
    end
  endcase
end



always_comb begin
  o_axis_tx_tdata = i_axis_rx_tdata;
  if ((mask_state == STATE_MASK) && i_mask_en) begin
    if (OP == "OR") begin
      o_axis_tx_tdata |= mask[cnt*DWIDTH+:DWIDTH];
    end
    else if (OP == "AND") begin
      o_axis_tx_tdata &= mask[cnt*DWIDTH+:DWIDTH];
    end
    else begin
      o_axis_tx_tdata ^= mask[cnt*DWIDTH+:DWIDTH];
    end
  end
end

always_ff @(posedge clk) begin
  if (rst) begin
    mask_state <= STATE_MASK;
    cnt        <= '0;
  end
  else begin
    mask_state   <= mask_state_nxt;
    if (i_axis_rx_tvalid && o_axis_rx_tready && i_mask_en) begin
      cnt          <= ((mask_state == STATE_SEND) || (i_axis_rx_tlast)) ? '0 : cnt + 1'b1;
    end
  end
end


assign o_axis_rx_tready = i_axis_tx_tready;
assign o_axis_tx_tvalid = i_axis_rx_tvalid;
assign o_axis_tx_tuser  = i_axis_rx_tuser;
assign o_axis_tx_tlast  = i_axis_rx_tlast;
assign o_axis_tx_tkeep  = i_axis_rx_tkeep;


endmodule
