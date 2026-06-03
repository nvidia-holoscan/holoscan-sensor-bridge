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


module sc_fifo #(
  parameter  DATA_WIDTH   = 16,
  parameter  FIFO_DEPTH   = 512,
  localparam ADDR_WIDTH   = ($clog2(FIFO_DEPTH))+1,
  parameter  ALMOST_FULL  = FIFO_DEPTH-4,
  parameter  ALMOST_EMPTY = 4,
  parameter  SYNC_DEPTH   = 2,
  parameter  RST_SYNC     = "TRUE", // "TRUE"|"FALSE"
  parameter  MEM_STYLE    = "BLOCK" // "LUT"|"BLOCK"
)(
  input                       clk,
  input                       rst,
  input                       wr,
  input      [DATA_WIDTH-1:0] din,
  output reg                  full,
  output reg                  afull,
  output                      over,
  input                       rd,
  output reg [DATA_WIDTH-1:0] dout,
  output reg                  dval,
  output reg                  empty,
  output reg                  aempty,
  output                      under,
  output reg [ADDR_WIDTH-1:0] count
);

localparam BUF_DEPTH = 2**($clog2(FIFO_DEPTH));

logic [DATA_WIDTH-1:0] mem     [BUF_DEPTH];
(* ram_style = "block" *)       logic [DATA_WIDTH-1:0] blk_mem [BUF_DEPTH] /*synthesis syn_ramstyle = "block_ram"*/;
(* ram_style = "distributed" *) logic [DATA_WIDTH-1:0] lut_mem [BUF_DEPTH] /*synthesis syn_ramstyle = "distributed"*/;

//------------------------------------------------------------------------------------------------//
// Write
//------------------------------------------------------------------------------------------------//
logic [ADDR_WIDTH-1:0] wr_bin;
logic [ADDR_WIDTH-1:0] wr_bin_nxt;
logic                  wr_en;

logic [ADDR_WIDTH-1:0] rd_bin;
logic [ADDR_WIDTH-1:0] rd_bin_nxt;
logic                  rd_en;

`ifdef SIMULATION
  initial begin
    wr_bin      = '0;
  end
`endif

generate
if (RST_SYNC == "TRUE") begin
  always @ (posedge clk) begin
    if (rst) begin
      wr_bin  <= '0;
      rd_bin  <= '0;
      dval    <= '0;
    end
    else begin
      if (wr_en) begin
        wr_bin  <= wr_bin_nxt;
      end
      if (rd_en) begin
        rd_bin  <= rd_bin_nxt;
      end
      dval    <= rd_en;
    end
  end
end
else begin
  always @ (posedge clk or posedge rst) begin
    if (rst) begin
      wr_bin  <= '0;
      rd_bin  <= '0;
      dval    <= '0;
    end
    else begin
      if (rd_en) begin
        rd_bin  <= rd_bin_nxt;
      end
      dval    <= rd_en;
    end
  end
end
endgenerate

assign wr_bin_nxt  = wr_bin + 1;
assign rd_bin_nxt  = rd_bin + 1;

assign over        = wr &  full;
assign wr_en       = wr & ~full;

assign under       = rd &  empty;
assign rd_en       = rd & ~empty;

//------------------------------------------------------------------------------------------------//
// Flags
//------------------------------------------------------------------------------------------------//

logic [ADDR_WIDTH-1:0] count_nxt;

generate
if (RST_SYNC == "TRUE") begin
  always_ff @(posedge clk) begin
    if (rst) begin
      count    <= '0;
      aempty   <= '1;
      empty    <= '1;
      afull    <= '1;
      full     <= '1;
    end
    else begin
      count    <= count_nxt;
      aempty   <= (count_nxt < ALMOST_EMPTY+1);
      empty    <= (count_nxt == 1) ? rd : ~|count_nxt;
      afull    <= (count_nxt > (ALMOST_FULL-1));
      full     <= (count_nxt == BUF_DEPTH-1) ? wr : (count_nxt == BUF_DEPTH);
    end
  end
end
else begin
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      count    <= '0;
      aempty   <= '1;
      empty    <= '1;
      afull    <= '1;
      full     <= '1;
    end
    else begin
      count    <= count_nxt;
      aempty   <= (count_nxt < ALMOST_EMPTY+1);
      empty    <= (count_nxt == 1) ? rd : ~|count_nxt;
      afull    <= (count_nxt > (ALMOST_FULL-1));
      full     <= (count_nxt == BUF_DEPTH-1) ? wr : (count_nxt == BUF_DEPTH);
    end
  end
end
endgenerate

assign count_nxt = wr_bin - rd_bin;

//------------------------------------------------------------------------------------------------//
// RAM
//------------------------------------------------------------------------------------------------//

logic [ADDR_WIDTH-2:0] wr_ptr;
logic [ADDR_WIDTH-2:0] rd_ptr;

assign wr_ptr = wr_bin[ADDR_WIDTH-2:0];
assign rd_ptr = rd_bin[ADDR_WIDTH-2:0];


// RAM synthesis style
generate
// Distributed RAM
if (((BUF_DEPTH < 128) && (MEM_STYLE == "AUTO")) || MEM_STYLE == "LUT") begin : DISTRIBUTED_RAM
  always_ff @(posedge clk) begin
    if (wr_en) begin
      lut_mem[wr_ptr] <= din;
    end
    if (rd_en) begin
      dout <= lut_mem[rd_ptr];
    end
  end
  // Block RAM
end
else if (((BUF_DEPTH >= 128) && (MEM_STYLE == "AUTO")) || MEM_STYLE == "BLOCK") begin : BLOCK_RAM
  always_ff @(posedge clk) begin
    if (wr_en) begin
      blk_mem[wr_ptr] <= din;
    end
    if (rd_en) begin
      dout <= blk_mem[rd_ptr];
    end
  end
// Auto RAM
end
else begin : AUTO_RAM
  always_ff @(posedge clk) begin
    if (wr_en) begin
      mem[wr_ptr] <= din;
    end
    if (rd_en) begin
      dout <= mem[rd_ptr];
    end
  end
end
endgenerate


endmodule
