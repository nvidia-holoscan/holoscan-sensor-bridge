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

module streaming_cdc #(
    parameter DATA_WIDTH = 64,
    parameter SRC_FREQ   = 100000000,
    parameter DST_FREQ   = 100000000
)(
    input                   i_src_clk,
    input                   i_dst_clk,
    input                   i_src_rst,
    input                   i_dst_rst,
    input  [DATA_WIDTH-1:0] i_src_data,
    output [DATA_WIDTH-1:0] o_dst_data,
    output                  o_dst_valid
);


//------------------------------------------------------------------------------------------------//
// Rate Limiting
//------------------------------------------------------------------------------------------------//


// We want to write at a rate that is just under DST_FREQ to prevent accumulation
// Target write rate = DST_FREQ * 0.99 (99% of destination frequency)

localparam DST_FREQ_REDUCED = DST_FREQ/1000;
localparam SRC_FREQ_REDUCED = SRC_FREQ/1000;

localparam PRECISION_BITS    = 8;  // 8-bit precision for fractional part
localparam TARGET_WRITE_FREQ = (DST_FREQ_REDUCED * 99) / 100;  // 99% of destination frequency
localparam PHASE_INCREMENT   = (TARGET_WRITE_FREQ << PRECISION_BITS) / SRC_FREQ_REDUCED;

logic [PRECISION_BITS-1:0] phase_accumulator;

logic src_valid;
logic dst_ready;
logic dst_ready_sync;

logic fifo_afull;
generate
  if (SRC_FREQ <= DST_FREQ) begin
    // DST is faster or equal to SRC - can always write
    always_ff @(posedge i_src_clk) begin
      if (i_src_rst) begin
        src_valid <= '0;
      end
      else begin
        src_valid <= dst_ready_sync && !fifo_afull;
      end
    end
  end
  else begin
    // Source is faster - need to throttle writes using precise conservative rate control
    always_ff @(posedge i_src_clk) begin
      if (i_src_rst) begin
        phase_accumulator <= '0;
        src_valid         <= '0;
      end
      else begin
        if (dst_ready_sync && !fifo_afull) begin
          {src_valid, phase_accumulator} <= phase_accumulator + PHASE_INCREMENT;
        end
        else begin
          phase_accumulator <= '0;
          src_valid         <= '0;
        end
      end
    end
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// Destination Ready
//------------------------------------------------------------------------------------------------//

always_ff @(posedge i_dst_clk) begin
  if (i_dst_rst) begin
    dst_ready <= '0;
  end
  else begin
    dst_ready <= '1;
  end
end

data_sync #(
  .DATA_WIDTH ( 1              ),
  .RESET_VALUE( 1'b0           ),
  .SYNC_DEPTH ( 2              ),
  .RST_SYNC   ( "TRUE"         )
) u_dst_ready_sync (
  .clk        ( i_src_clk      ),
  .rst_n      ( !i_src_rst     ),
  .sync_in    ( dst_ready      ),
  .sync_out   ( dst_ready_sync )
);

//------------------------------------------------------------------------------------------------//
// DC FIFO
//------------------------------------------------------------------------------------------------//

logic fifo_rd_en;
logic fifo_empty;
logic fifo_aempty;

dc_fifo_stub #(
  .DATA_WIDTH  ( DATA_WIDTH   ),
  .ALMOST_FULL ( 7            ),
  .FIFO_DEPTH  ( 16           )
) u_dc_fifo_stub (
  .wrrst  ( i_src_rst         ),
  .rdrst  ( i_dst_rst         ),
  .wrclk  ( i_src_clk         ),
  .wren   ( src_valid         ),
  .wrdin  ( i_src_data        ),
  .wrcount(                   ),
  .full   (                   ),
  .afull  ( fifo_afull        ),
  .over   (                   ),
  .rdclk  ( i_dst_clk         ),
  .rden   ( fifo_rd_en        ),
  .rddout ( o_dst_data        ),
  .rdval  ( o_dst_valid       ),
  .rdcount(                   ),
  .empty  ( fifo_empty        ),
  .aempty ( fifo_aempty       ),
  .under  (                   )
);

assign fifo_rd_en = !fifo_empty;

endmodule







