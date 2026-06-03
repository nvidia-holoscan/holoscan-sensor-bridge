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

module axis_packer #(
    parameter DWIDTH = 64,
    parameter W_KEEP = DWIDTH/8,
    parameter W_USER = 1
)
(
  input   logic                               clk,
  input   logic                               rst,

  input   logic                               i_axis_tvalid,
  input   logic   [DWIDTH-1:0]                i_axis_tdata,
  input   logic                               i_axis_tlast,
  input   logic   [W_USER-1:0]                i_axis_tuser,
  input   logic   [W_KEEP-1:0]                i_axis_tkeep,
  output  logic                               o_axis_tready,

  output  logic                               o_axis_tvalid,
  output  logic   [DWIDTH-1:0]                o_axis_tdata,
  output  logic                               o_axis_tlast,
  output  logic   [W_USER-1:0]                o_axis_tuser,
  output  logic   [W_KEEP-1:0]                o_axis_tkeep,
  input   logic                               i_axis_tready
);

logic   [(DWIDTH*2)-1:0] tdata;
logic   [DWIDTH-1    :0] wdata;
logic   [(W_KEEP*2)-1:0] tkeep;
logic   [           1:0] tlast;

logic   [DWIDTH-1    :0] tdata_r;
logic   [W_KEEP-1    :0] tkeep_r;
logic                    tlast_r;
logic                    tlast_prev;

logic [($clog2(W_KEEP))-1:0] idx;

always_comb begin
  tdata = tdata_r;
  tkeep = tkeep_r;
  tlast = tlast_r;
  if (i_axis_tvalid && o_axis_tready) begin
    tdata = (tdata_r | (wdata << (idx*8)));
    tkeep = (tkeep_r | (i_axis_tkeep << (idx)));
    tlast[0] = (i_axis_tlast && tkeep[W_KEEP+:W_KEEP] == '0);
    tlast[1] = (i_axis_tlast && tkeep[W_KEEP+:W_KEEP] != '0);
  end
end

integer i;
always_comb begin
  idx = '0;
  for (i=0;i<W_KEEP;i=i+1) begin
    idx = idx + tkeep_r[i];
  end
end

genvar j;
generate
  for (j=0;j<W_KEEP;j=j+1) begin
    assign wdata[j*8+:8] = (i_axis_tkeep[j]) ? i_axis_tdata[j*8+:8] : '0;
  end
endgenerate


always_ff @(posedge clk) begin
  if (rst) begin
    tdata_r     <= '0;
    tkeep_r     <= '0;
    tlast_r     <= '0;
    tlast_prev  <= '0;
  end
  else begin
    tdata_r    <= (o_axis_tvalid) ? tdata[DWIDTH+:DWIDTH] : tdata[0+:DWIDTH];
    tkeep_r    <= (o_axis_tvalid) ? tkeep[W_KEEP+:W_KEEP] : tkeep[0+:W_KEEP];
    tlast_r    <= (o_axis_tvalid) ? tlast[1]              : tlast[0];
    tlast_prev <= tlast[1];
  end
end

assign o_axis_tvalid = (o_axis_tkeep == '1) || (o_axis_tlast);
assign o_axis_tdata  = tdata[0+:DWIDTH];
assign o_axis_tlast  = tlast[0];
assign o_axis_tuser  = i_axis_tuser;
assign o_axis_tkeep  = tkeep[0+:W_KEEP];
assign o_axis_tready = i_axis_tready && (!tlast_prev);

endmodule