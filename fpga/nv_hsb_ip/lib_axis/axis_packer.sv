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

logic   [W_KEEP-1    :0] ins_sel;
logic   [(DWIDTH*2)-1:0] tdata_ins;
logic   [(W_KEEP*2)-1:0] tkeep_ins;

always_comb begin
  tdata = tdata_r;
  tkeep = tkeep_r;
  tlast = tlast_r;
  if (i_axis_tvalid && o_axis_tready) begin
    tdata = (tdata_r | tdata_ins);
    tkeep = (tkeep_r | tkeep_ins);
    tlast[0] = (i_axis_tlast && !tkeep[W_KEEP]);
    tlast[1] = (i_axis_tlast &&  tkeep[W_KEEP]);
  end
end


// Because tkeep_r is packed contiguously from bit 0, the byte position to insert
// at is just its single 1->0 boundary, so it decodes one-hot in one gate per bit.
// tkeep_r is never full -- a full low half asserts o_axis_tvalid, which reloads
// tkeep_r from the upper half -- so position W_KEEP needs no slot.
always_comb begin
  ins_sel[0] = !tkeep_r[0];
  for (int k = 1; k < W_KEEP; k++) begin
    ins_sel[k] = tkeep_r[k-1] && !tkeep_r[k];
  end
end

// Insert at the selected position. Each term shifts by a constant, so the shifts
// are wiring and the only logic is the one-hot OR -- no binary shift amount, and
// so no popcount feeding a barrel shifter.
always_comb begin
  tdata_ins = '0;
  tkeep_ins = '0;
  for (int k = 0; k < W_KEEP; k++) begin
    if (ins_sel[k]) begin
      tdata_ins = tdata_ins | ({{DWIDTH{1'b0}}, wdata}         << (k*8));
      tkeep_ins = tkeep_ins | ({{W_KEEP{1'b0}}, i_axis_tkeep}  <<  k   );
    end
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

// Contiguous tkeep (see above) means the low half is full exactly when its top
// bit is set, so this replaces a W_KEEP-wide AND with a single bit test.
assign o_axis_tvalid = (o_axis_tkeep[W_KEEP-1]) || (o_axis_tlast);
assign o_axis_tdata  = tdata[0+:DWIDTH];
assign o_axis_tlast  = tlast[0];
assign o_axis_tuser  = i_axis_tuser;
assign o_axis_tkeep  = tkeep[0+:W_KEEP];
assign o_axis_tready = i_axis_tready && (!tlast_prev);

endmodule