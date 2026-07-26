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

// NOTE (RST_SYNC == "TRUE"):
// - wrrst and rdrst are *synchronous* resets to wrclk/rdclk (not async).
// - BOTH sides MUST be reset (assert wrrst AND rdrst) to properly reset the DC FIFO.
//   If the write/input side is not reset, the reset-handshake never completes and the read/output
//   side is intentionally held/gated idle (no reads/dval) to prevent unsafe operation.

module dc_fifo_generic #(
  parameter  DATA_WIDTH   = 16,
  parameter  FIFO_DEPTH   = 512,
  localparam ADDR_WIDTH   = ($clog2(FIFO_DEPTH))+1,
  parameter  ALMOST_FULL  = FIFO_DEPTH-4,
  parameter  ALMOST_EMPTY = 4,
  parameter  SYNC_DEPTH   = 2,
  parameter  RST_SYNC     = "TRUE", // "TRUE"|"FALSE"
  parameter  MEM_STYLE    = "BLOCK" // "LUT"|"BLOCK"
)(
  // asynchronous reset
  input                         rst,
  // wr domain signals
  input                         wrclk,
  input                         wrrst,
  input                         wr,
  input        [DATA_WIDTH-1:0] din,
  output logic [ADDR_WIDTH-1:0] wrcount,
  output logic                  full,
  output logic                  afull,
  output                        over,
  // rd domain signals
  input                         rdclk,
  input                         rdrst,
  input                         rd,
  output logic [DATA_WIDTH-1:0] dout,
  output logic                  dval,
  output logic [ADDR_WIDTH-1:0] rdcount,
  output logic                  empty,
  output logic                  aempty,
  output                        under
);

localparam BUF_DEPTH = 2**($clog2(FIFO_DEPTH));

logic [DATA_WIDTH-1:0] mem     [BUF_DEPTH];
(* ram_style = "block" *)       logic [DATA_WIDTH-1:0] blk_mem [BUF_DEPTH] /*synthesis syn_ramstyle = "block_ram"*/;
(* ram_style = "distributed" *) logic [DATA_WIDTH-1:0] lut_mem [BUF_DEPTH] /*synthesis syn_ramstyle = "distributed"*/;


//------------------------------------------------------------------------------------------------//
// Reset CDC
//------------------------------------------------------------------------------------------------//

// CDC reset handshake ensures rd waits for wr to complete reset.
// rd won't assert rdval until wr is ready.

logic wr_init_done;
logic rd_init_done;

generate
  if (RST_SYNC == "TRUE") begin
    cdc_reset_handshake u_cdc_reset_handshake (
      .src_clk        ( wrclk        ),
      .src_rst        ( wrrst        ),
      .dst_clk        ( rdclk        ),
      .dst_rst        ( rdrst        ),
      .o_src_init_done( wr_init_done ),
      .o_dst_init_done( rd_init_done )
    );
  end
  else begin
    assign wr_init_done = '1;
    assign rd_init_done = '1;
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// Wr Clock Domain
//------------------------------------------------------------------------------------------------//
logic [ADDR_WIDTH-1:0] wr_bin;
logic [ADDR_WIDTH-1:0] wr_bin_nxt;

logic [ADDR_WIDTH-1:0] wr_gray;
logic [ADDR_WIDTH-1:0] wr_gray_nxt;

`ifdef SIMULATION
  initial begin
    wr_bin      = '0;
    wr_gray     = '0;
  end
`endif

logic                wr_en;

generate
if (RST_SYNC == "TRUE") begin
  always @ (posedge wrclk) begin
    if (wrrst) begin
      wr_bin  <= '0;
      wr_gray <= '0;
    end
    else begin
      if (wr_en) begin
        wr_bin  <= wr_bin_nxt;
        wr_gray <= wr_gray_nxt;
      end
    end
  end
end
else begin
  always @ (posedge wrclk or posedge rst) begin
    if (rst) begin
      wr_bin  <= '0;
      wr_gray <= '0;
    end
    else begin
      if (wr_en) begin
        wr_bin  <= wr_bin_nxt;
        wr_gray <= wr_gray_nxt;
      end
    end
  end
end
endgenerate

assign over        = wr &  full;
assign wr_en       = wr & ~full & wr_init_done;

assign wr_bin_nxt  = wr_bin + 1;
assign wr_gray_nxt = (wr_bin_nxt >> 1) ^ wr_bin_nxt;

//------------------------------------------------------------------------------------------------//
// Rd Clock Domain
//------------------------------------------------------------------------------------------------//

logic [ADDR_WIDTH-1:0] rd_bin;
logic [ADDR_WIDTH-1:0] rd_bin_nxt;

logic [ADDR_WIDTH-1:0] rd_gray;
logic [ADDR_WIDTH-1:0] rd_gray_nxt;

logic                rd_en;

generate
if (RST_SYNC == "TRUE") begin
  always_ff @(posedge rdclk) begin
    if (rdrst) begin
      rd_bin  <= '0;
      rd_gray <= '0;
      dval    <= '0;
    end
    else begin
      if (rd_en) begin
        rd_bin  <= rd_bin_nxt;
        rd_gray <= rd_gray_nxt;
      end
      dval    <= rd_en;
    end
  end
end
else begin
  always_ff @(posedge rdclk or posedge rst) begin
    if (rst) begin
      rd_bin  <= '0;
      rd_gray <= '0;
      dval    <= '0;
    end
      else begin
      if (rd_en) begin
        rd_bin  <= rd_bin_nxt;
        rd_gray <= rd_gray_nxt;
      end
      dval    <= rd_en;
    end
  end
end
endgenerate

assign rd_en       = rd & ~empty & rd_init_done;
assign under       = rd &  empty;

assign rd_bin_nxt  = rd_bin + 1;
assign rd_gray_nxt = (rd_bin_nxt >> 1) ^ rd_bin_nxt;

//------------------------------------------------------------------------------------------------//
// Pointer CDC
//------------------------------------------------------------------------------------------------//

logic [ADDR_WIDTH-1:0] rd_gray_wrsync;
logic [ADDR_WIDTH-1:0] rd_bin_wrsync;

logic [ADDR_WIDTH-1:0] wr_gray_rdsync;
logic [ADDR_WIDTH-1:0] wr_bin_rdsync;

generate
if (RST_SYNC == "TRUE") begin
  data_sync #(
    .DATA_WIDTH ( ADDR_WIDTH     ),
    .SYNC_DEPTH ( SYNC_DEPTH     ),
    .RST_SYNC   ( RST_SYNC       )
  ) rd_ptr_sync_inst (
    .clk        ( wrclk          ),
    .rst_n      ( !wrrst         ),
    .sync_in    ( rd_gray        ),
    .sync_out   ( rd_gray_wrsync )
  );

  data_sync #(
    .DATA_WIDTH ( ADDR_WIDTH     ),
    .SYNC_DEPTH ( SYNC_DEPTH     ),
    .RST_SYNC   ( RST_SYNC       )
  ) wr_ptr_sync_inst (
    .clk        ( rdclk          ),
    .rst_n      ( !rdrst         ),
    .sync_in    ( wr_gray        ),
    .sync_out   ( wr_gray_rdsync )
  );
end
else begin
  data_sync #(
    .DATA_WIDTH ( ADDR_WIDTH     ),
    .SYNC_DEPTH ( SYNC_DEPTH     ),
    .RST_SYNC   ( RST_SYNC       )
  ) rd_ptr_sync_inst (
    .clk        ( wrclk          ),
    .rst_n      ( !rst           ),
    .sync_in    ( rd_gray        ),
    .sync_out   ( rd_gray_wrsync )
  );

  data_sync #(
    .DATA_WIDTH ( ADDR_WIDTH     ),
    .SYNC_DEPTH ( SYNC_DEPTH     ),
    .RST_SYNC   ( RST_SYNC       )
  ) wr_ptr_sync_inst (
    .clk        ( rdclk          ),
    .rst_n      ( !rst           ),
    .sync_in    ( wr_gray        ),
    .sync_out   ( wr_gray_rdsync )
  );
end
endgenerate

// Gray to Binary
genvar i;
generate
for(i=0;i<ADDR_WIDTH;i++) begin
  assign rd_bin_wrsync[i] = ^(rd_gray_wrsync >> i);
  assign wr_bin_rdsync[i] = ^(wr_gray_rdsync >> i);
end
endgenerate

//------------------------------------------------------------------------------------------------//
// Rd Flags
//------------------------------------------------------------------------------------------------//

logic [ADDR_WIDTH-1:0] rdcount_nxt;

generate
if (RST_SYNC == "TRUE") begin
  always_ff @(posedge rdclk) begin
    if (rdrst) begin
      rdcount  <= '0;
      aempty   <= '1;
      empty    <= '1;
    end
    else begin
      rdcount  <= !rd_init_done ? '0 : rdcount_nxt;
      aempty   <= !rd_init_done ? '1 : (rdcount_nxt == ALMOST_EMPTY+1) ? rd : (rdcount_nxt < ALMOST_EMPTY+1);
      empty    <= !rd_init_done ? '1 : ((rdcount_nxt == 1) ? rd : ~|rdcount_nxt);
    end
  end
end
else begin
  always_ff @(posedge rdclk or posedge rst) begin
    if (rst) begin
      rdcount  <= '0;
      aempty   <= '1;
      empty    <= '1;
    end
    else begin
      rdcount  <= rdcount_nxt;
      aempty   <= (rdcount_nxt == ALMOST_EMPTY+1) ? rd : (rdcount_nxt < ALMOST_EMPTY+1);
      empty    <= ((rdcount_nxt == 1) ? rd : ~|rdcount_nxt);
    end
  end
end
endgenerate

assign rdcount_nxt = (wr_bin_rdsync - rd_bin);

//------------------------------------------------------------------------------------------------//
// Wr Flags
//------------------------------------------------------------------------------------------------//

logic [ADDR_WIDTH-1:0] wrcount_nxt;

generate
if (RST_SYNC == "TRUE") begin
  always_ff @(posedge wrclk) begin
    if (wrrst) begin
      wrcount  <= '0;
      afull    <= '1;
      full     <= '1;
    end
    else begin
      wrcount  <= !wr_init_done ? '0 : wrcount_nxt;
      afull    <= !wr_init_done ? '1 : (wrcount_nxt == (ALMOST_FULL-1) ? wr : (wrcount_nxt > (ALMOST_FULL-1)));
      full     <= !wr_init_done ? '1 : (wrcount_nxt == BUF_DEPTH-1) ? wr : (wrcount_nxt == BUF_DEPTH);
    end
  end
end
else begin
  always_ff @(posedge wrclk or posedge rst) begin
    if (rst) begin
      wrcount  <= '0;
      afull    <= '1;
      full     <= '1;
    end
    else begin
      wrcount  <= wrcount_nxt;
      afull    <= (wrcount_nxt == (ALMOST_FULL-1) ? wr : (wrcount_nxt > (ALMOST_FULL-1)));
      full     <= (wrcount_nxt == BUF_DEPTH-1) ? wr : (wrcount_nxt == BUF_DEPTH);
    end
  end
end
endgenerate

assign wrcount_nxt = wr_bin - rd_bin_wrsync;

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
    always_ff @(posedge wrclk) begin
      if (wr_en) begin
        lut_mem[wr_ptr] <= din;
      end
    end
    always_ff @(posedge rdclk) begin
      if (rd_en) begin
        dout <= lut_mem[rd_ptr];
      end
    end
    // Block RAM
  end
  else if (((FIFO_DEPTH >= 128) && (MEM_STYLE == "AUTO")) || MEM_STYLE == "BLOCK") begin : BLOCK_RAM
    always_ff @(posedge wrclk) begin
      if (wr_en) begin
        blk_mem[wr_ptr] <= din;
      end
    end
    always_ff @(posedge rdclk) begin
      if (rd_en) begin
        dout <= blk_mem[rd_ptr];
      end
    end
  // Auto RAM
  end
  else begin : AUTO_RAM
    always_ff @(posedge wrclk) begin
      if (wr_en) begin
        mem[wr_ptr] <= din;
      end
    end
    always_ff @(posedge rdclk) begin
      if (rd_en) begin
        dout <= mem[rd_ptr];
      end
    end
  end
endgenerate


endmodule
