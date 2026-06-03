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

module str_fwd_fifo #(
    parameter           WIDTH   =   512,
    parameter           DEPTH   =  1024,
    parameter           AFULL   = DEPTH-4,
    localparam          MAX_BIT = WIDTH - 1,
    localparam          NBITS   = $clog2(DEPTH),
    localparam          MBIT    = NBITS -1
)(
    input               clk,
    input               rst,
    input  [MAX_BIT:0]  wr_data,
    input               wrreq,
    input               wr_commit,
    input               wr_reject,
    input               rdreq,
    output reg          afull,
    output              empty,
    output              full,
    output              rd_rdy,
    output [MAX_BIT:0]  rd_data
);

logic [MBIT:0] wr_ptr;
logic [MBIT:0] wr_ptr_cmt;
logic [MBIT:0] wr_ptr_nxt;
logic [MBIT:0] rd_ptr;
logic [MBIT:0] rd_ptr_nxt;
logic [MBIT:0] wr_cnt;

assign wr_ptr_nxt = wr_ptr + 1'b1;
assign rd_ptr_nxt = rd_ptr + 1'b1;

assign wr_cnt     = (wr_ptr >= rd_ptr) ? wr_ptr - rd_ptr : DEPTH - rd_ptr + wr_ptr;

assign full       = (wr_cnt == DEPTH-1);
assign empty      = (rd_ptr == wr_ptr_cmt);

assign rd_rdy   = !empty;

always@(posedge clk or posedge rst) begin
  if (rst) begin
    afull <= '1;
  end
  else begin
    afull <= wr_cnt >= AFULL-1;
  end
end

always@(posedge clk or posedge rst) begin
  if (rst) begin
    wr_ptr      <=  '0;
    wr_ptr_cmt  <=  '0;
  end
  else begin
    if (wrreq && wr_commit && !full) begin
      wr_ptr      <= wr_ptr_nxt;
      wr_ptr_cmt  <= wr_ptr_nxt;
    end
    else if (wrreq && !full) begin
      wr_ptr      <= wr_ptr_nxt;
    end
    else if (wr_commit) begin
      wr_ptr_cmt  <= wr_ptr;
    end
    else if (wr_reject) begin
      wr_ptr      <= wr_ptr_cmt;
    end
  end
end

always@(posedge clk or posedge rst) begin
  if (rst) begin
    rd_ptr      <=  '0;
  end
  else begin
    if (!empty && rdreq) begin
      rd_ptr  <= rd_ptr_nxt;
    end
  end
end

logic wr_ram;

assign wr_ram = wrreq && !full;

dp_ram #(
  .DATA_WIDTH ( WIDTH    ), // bit width of data
  .RAM_DEPTH  ( DEPTH    ), // depth of internal ram addressing
  .RAM_TYPE   ( "SIMPLE" ), // "SIMPLE" or "TRUE"
  .MEM_STYLE  ( "AUTO"   )  // "LUT" / "BLOCK" / "AUTO"
) u_dp_ram (
  // Port A
  .clk_a      ( clk     ),
  .en_a       ( '1      ),
  .we_a       ( wr_ram  ),
  .din_a      ( wr_data ),
  .addr_a     ( wr_ptr  ),
  .dout_a     (         ),
  // Port B
  .clk_b      ( clk     ),
  .en_b       ( '1      ),
  .we_b       ( '0      ),
  .din_b      ( '0      ),
  .addr_b     ( rd_ptr  ),
  .dout_b     ( rd_data )
);

endmodule


