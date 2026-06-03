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

module reset_sync (
  input  i_clk,
  input  i_arst_n,
  input  i_srst,
  input  i_locked,
  output o_arst,
  output o_arst_n,
  output o_srst,
  output o_srst_n
);

logic aasr_rst;   // async applied sync released active high reset
logic aasr_rst_q; // async applied sync released active high reset
logic arst;
logic arst_n;
// asynchronous reset output
always_ff @(posedge i_clk or negedge i_arst_n) begin
  if (!i_arst_n) begin
    aasr_rst   <=  1;
    aasr_rst_q <=  1;
    arst       <=  1;
    arst_n     <=  0;
  end
  else begin
    aasr_rst   <=  i_srst | ~i_locked;
    aasr_rst_q <=  aasr_rst;
    arst       <=  aasr_rst_q;
    arst_n     <= ~aasr_rst_q;
  end
end

assign o_arst   = arst;
assign o_arst_n = arst_n;

reg [1:0] srst   = '1;
reg [1:0] srst_n = '0;
// synchronous reset output
always @ (posedge i_clk) begin
  srst   <= {srst  [0],  aasr_rst_q};
  srst_n <= {srst_n[0], ~aasr_rst_q};
end

assign o_srst   = srst  [1];
assign o_srst_n = srst_n[1];

endmodule
