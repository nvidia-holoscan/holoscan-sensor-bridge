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

module vec_to_axis_dyn_len
#(
    parameter                               AXI_DWIDTH   = 64,
    parameter                               AXI_TKEEP    = AXI_DWIDTH/8,
    parameter                               DATA_WIDTH   = 288
)(
    input                                   clk,
    input                                   rst,

    input                                   trigger,
    input   [DATA_WIDTH-1:0]                data,
    input   [$clog2(DATA_WIDTH/8)-1:0]      byte_len,
    output                                  is_busy,
  //AXIS Interface
    output  logic                           o_axis_tx_tvalid,
    output  logic   [AXI_DWIDTH-1:0]        o_axis_tx_tdata,
    output  logic                           o_axis_tx_tlast,
    output  logic                           o_axis_tx_tuser,
    output  logic   [(AXI_DWIDTH/8)-1:0]    o_axis_tx_tkeep,
    input                                   i_axis_tx_tready
);

localparam D_CYCLES        = (DATA_WIDTH-1)/AXI_DWIDTH + 1;
localparam AXI_TKEEP_WIDTH = AXI_TKEEP>1 ? $clog2(AXI_TKEEP) : 1;


typedef enum logic [1:0] {
  AXI_IDLE,
  AXI_SEND,
  AXI_LAST
} vec_states;
vec_states vec_state;

logic [D_CYCLES*AXI_DWIDTH-1:0] r_data;
logic [$clog2(D_CYCLES+1)+1:0]  cnt;
logic                           r_trigger;
logic                           r_busy;
logic                           r_is_busy;
logic [$clog2(D_CYCLES+1)-1:0]  r_d_cycles;
logic [AXI_DWIDTH/8-1:0]        r_tkeep_val;
logic                           extra_cycle;
assign extra_cycle = AXI_TKEEP>1 ? byte_len[AXI_TKEEP_WIDTH-1:0]!='d0 : 1'b0;

always_ff @(posedge clk) begin
  if (rst) begin
    vec_state               <= AXI_IDLE;
    cnt                     <= '0;
    r_d_cycles              <= '0;
    r_data                  <= '0;
    r_trigger               <= '0;
    r_busy                  <= '0;
    r_is_busy               <= '0;
    r_tkeep_val             <= '1;
  end
  else begin
    r_trigger <= trigger;
    r_is_busy <= (r_busy || ({r_trigger,trigger} == 2'b01));
    case(vec_state)
      AXI_IDLE:   begin
        cnt                     <= '0;
        r_data                  <= {'0,data};
        r_d_cycles              <= byte_len <= AXI_TKEEP ? 1 : (byte_len>>$clog2(AXI_TKEEP)) + extra_cycle;
        if ({r_trigger,trigger} == 2'b01) begin
          vec_state           <=  (r_d_cycles == 1) ? AXI_LAST : AXI_SEND;
          r_tkeep_val         <= ~('1 << (byte_len[AXI_TKEEP_WIDTH-1:0]));
          r_busy              <= '1;
        end
      end
      AXI_SEND: begin
        if (i_axis_tx_tready) begin
          cnt                     <= cnt + 1'b1;
          vec_state               <= (cnt == r_d_cycles-2) ? AXI_LAST : vec_state ;
        end
      end
      AXI_LAST: begin
        vec_state               <= i_axis_tx_tready ? AXI_IDLE : vec_state;
        r_busy                  <= !i_axis_tx_tready;
      end
      default: begin
        vec_state               <= AXI_IDLE;
      end
    endcase
  end
end

assign o_axis_tx_tdata  = (vec_state == AXI_IDLE) ? '0 : {r_data[AXI_DWIDTH*cnt+:AXI_DWIDTH]};
assign o_axis_tx_tlast  = (vec_state == AXI_LAST);
assign o_axis_tx_tvalid = (vec_state != AXI_IDLE);
assign o_axis_tx_tkeep  = o_axis_tx_tlast ? (byte_len[AXI_TKEEP_WIDTH-1:0]!='d0) ? r_tkeep_val : '1
                                          : '1;
assign o_axis_tx_tuser  = 1'b0;
assign is_busy          = r_is_busy;

endmodule
