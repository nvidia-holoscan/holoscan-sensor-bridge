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

// NOTE:
// - wrrst and rdrst are *synchronous* resets to wrclk/rdclk (not async).
// - BOTH sides MUST be reset (assert wrrst AND rdrst) to properly reset the DC FIFO.
//   If the write/input side is not reset, the reset-handshake never completes and the read/output
//   side is intentionally held/gated idle (no reads/dval) to prevent unsafe operation.

`include "HOLOLINK_def.svh"
module dc_fifo_stub #(
  parameter  DATA_WIDTH   = 8,
  parameter  FIFO_DEPTH   = 512,
  localparam ADDR_WIDTH   = ($clog2(FIFO_DEPTH))+1,
  parameter  ALMOST_FULL  = FIFO_DEPTH-4,
  parameter  ALMOST_EMPTY = 4,
  parameter  MEM_STYLE    = "LUT" // "LUT"|"BLOCK"
)(
  // wr domain signals
  input                   wrclk,
  input                   wrrst,
  input                   wren,
  input  [DATA_WIDTH-1:0] wrdin,
  output [ADDR_WIDTH-1:0] wrcount,
  output logic            full,
  output logic            afull,
  output                  over,
  // rd domain signals
  input                   rdclk,
  input                   rdrst,
  input                   rden,
  output [DATA_WIDTH-1:0] rddout,
  output logic            rdval,
  output [ADDR_WIDTH-1:0] rdcount,
  output                  empty,
  output logic            aempty,
  output                  under
);

dc_fifo_generic #(
  .DATA_WIDTH   ( DATA_WIDTH   ),
  .FIFO_DEPTH   ( FIFO_DEPTH   ),
  .ALMOST_FULL  ( ALMOST_FULL  ),
  .ALMOST_EMPTY ( ALMOST_EMPTY ),
  .MEM_STYLE    ( MEM_STYLE    )
) u_dc_fifo (
  .rst           ( '0         ),
  .wrclk         ( wrclk      ),
  .wrrst         ( wrrst      ),
  .wr            ( wren       ),
  .din           ( wrdin      ),
  .wrcount       ( wrcount    ),
  .full          ( full       ),
  .afull         ( afull      ),
  .over          ( over       ),
  .rdclk         ( rdclk      ),
  .rdrst         ( rdrst      ),
  .rd            ( rden       ),
  .dout          ( rddout     ),
  .dval          ( rdval      ),
  .rdcount       ( rdcount    ),
  .empty         ( empty      ),
  .aempty        ( aempty     ),
  .under         ( under      )
);

`ifdef ASSERT_ON
  //Assertions
  assert_dc_fifo_overflow: assert property ( @(posedge wrclk) disable iff (wrrst)
        (!over));

  assert_dc_fifo_underflow: assert property ( @(posedge rdclk) disable iff (rdrst)
        (!under));
`endif

endmodule
