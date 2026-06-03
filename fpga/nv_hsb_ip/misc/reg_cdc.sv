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

module reg_cdc #(
  parameter             NBITS         = 32,
  parameter [NBITS-1:0] REG_RST_VALUE = '0
)(
  // Source Clock Domain
  input              i_a_clk,
  input              i_a_rst,
  input              i_a_val,
  input  [NBITS-1:0] i_a_reg,
  // Destination Clock Domain
  input              i_b_clk,
  input              i_b_rst,
  output             o_b_val,
  output [NBITS-1:0] o_b_reg
);

//------------------------------------------------------------------------------------------------//
// A clock domain
//------------------------------------------------------------------------------------------------//

logic             a_val_q;
logic             a_val_qq;
logic [NBITS-1:0] a_reg_q;
logic [NBITS-1:0] a_reg_qq;
logic             b2a_ack;
logic [1      :0] b2a_ack_sync;

always_ff @(posedge i_a_clk) begin
  if (i_a_rst) begin
    a_val_q      <= '0;
    a_val_qq     <= '0;
    b2a_ack_sync <= '0;
    a_reg_q      <= '0;
    a_reg_qq     <= '0;
  end
  else begin
    a_val_q      <= i_a_val;
    a_val_qq     <= (a_val_q | a_val_qq) & ~b2a_ack_sync[1];
    b2a_ack_sync <= {b2a_ack_sync[0], b2a_ack};
    a_reg_q      <= i_a_reg;
    if (a_val_q && !a_val_qq) begin
      a_reg_qq   <= a_reg_q;
    end
  end
end

//------------------------------------------------------------------------------------------------//
// B clock domain
//------------------------------------------------------------------------------------------------//

logic [2      :0] a2b_val_sync;
logic [NBITS-1:0] a2b_reg_sync;

always_ff @(posedge i_b_clk) begin
  if (i_b_rst) begin
    a2b_val_sync   <= '0;
    a2b_reg_sync   <= REG_RST_VALUE;
    b2a_ack        <= '0;
  end
  else begin
    a2b_val_sync   <= {a2b_val_sync[1:0], a_val_qq};
    if (a2b_val_sync[1] && !a2b_val_sync[2]) begin
      a2b_reg_sync <= a_reg_qq;
    end
    b2a_ack        <= a2b_val_sync[1];
  end
end

assign o_b_val = a2b_val_sync[2];
assign o_b_reg = a2b_reg_sync;

endmodule