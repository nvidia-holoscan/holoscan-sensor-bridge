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


module cam #(
  parameter  W_DATA  = 1,  // Width of Search Data
  parameter  W_KEY   = 48,  // Width of Return Data
  parameter  DEPTH   = 32, // Number of CAM Cells
  localparam W_ADDR  = $clog2(DEPTH)
)(
  input                 i_clk,
  input                 i_rst,

  // Ctrl
  input                 i_wr,
  input                 i_del,
  input  [W_DATA-1:0]   i_data,
  input  [W_KEY-1:0]    i_key,
  output [W_DATA-1:0]   o_data,
  output                o_data_val
);

localparam RAM_W_DATA = W_DATA + W_KEY + 1; // Search + Key + valid
localparam VALID_IDX  = RAM_W_DATA-1;

logic [RAM_W_DATA-1:0] ram [DEPTH];
logic [W_ADDR-1:0]     match_idx;
logic                  match_val;
logic [W_ADDR-1:0]     match_idx_reg;
logic                  match_val_reg;
logic                  wr_reg;

logic [W_ADDR-1:0]     empty_idx;
logic                  empty_val;
logic [W_ADDR-1:0]     empty_idx_reg;
logic                  empty_val_reg;

logic [W_DATA-1:0]     data_reg;
logic [W_KEY-1:0]      key_reg;

integer i,j;
always_comb begin
  match_val = 0;
  match_idx = 0;
  for (i=0;i<DEPTH;i=i+1) begin
    if (ram[i][VALID_IDX] && (ram[i][0+:W_KEY] == i_key )) begin
      match_val = 1;
      match_idx = i;
    end
  end
end

assign o_data = ram[match_idx_reg][W_KEY+:W_DATA];
assign o_data_val = match_val_reg;

always_comb begin
  empty_val = 0;
  empty_idx = 0;
  for (j=0;j<DEPTH;j=j+1) begin
    if (!ram[j][VALID_IDX]) begin
      empty_val = 1;
      empty_idx = j;
    end
  end
end

always_ff @(posedge i_clk) begin
if (i_rst) begin
  match_val_reg     <= 0;
  match_idx_reg     <= 0;
  empty_val_reg     <= 0;
  empty_idx_reg     <= 0;
  wr_reg            <= 0;
  data_reg          <= 0;
  key_reg           <= 0;
end
else begin
  match_val_reg     <= match_val;
  match_idx_reg     <= match_idx;
  empty_val_reg     <= empty_val;
  empty_idx_reg     <= empty_idx;
  wr_reg            <= i_wr;
  data_reg          <= i_data;
  key_reg           <= i_key;
end
end

always_ff @(posedge i_clk) begin
if (i_rst) begin
  ram <= '{default:0};
end
else begin
  if (wr_reg && empty_val && !match_val_reg) begin
    ram[empty_idx] <= {1'b1,data_reg,key_reg};
  end
  else if (i_del && match_val_reg) begin
    ram[match_idx_reg] <= '0;
  end
end
end


endmodule
