// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Timestamp ILA lanes 5 and 6 (sample bits [223:160]) are the fixed marker
// 82'h123456789ABCDEF. Answer those APB windows here so the sample RAMs are
// not built. The external 8-lane map is unchanged: a read of lane 7 is
// rewritten onto stored lane 5, which holds the counter.
//
// Stored lane count is 6. $clog2(6) == $clog2(8), so the control/RAM split
// and the lane select bits stay where the 256-bit ILA put them.

module ts_ila_apb_shim
  import apb_pkg::*;
#(
  parameter DEPTH = 16384
) (
  input  apb_m2s i_apb_m2s,
  output apb_s2m o_apb_s2m,
  output apb_m2s o_ila_apb_m2s,
  input  apb_s2m i_ila_apb_s2m
);

  localparam W_ADDR   = $clog2(DEPTH);
  localparam W_RAM    = 3;
  localparam W_OFSET  = W_ADDR + W_RAM + 2;
  localparam BANK_LSB = W_ADDR + 2;

  // Same assignment as sensor_if_path: zero-extended into bits [223:142].
  localparam logic [81:0] MARKER = 82'h123456789ABCDEF;
  localparam logic [31:0] LANE5  = MARKER[49:18];
  localparam logic [31:0] LANE6  = MARKER[81:50];

  wire [31:0] region = i_apb_m2s.paddr[31:W_OFSET];
  wire [2:0]  bank   = i_apb_m2s.paddr[BANK_LSB +: 3];
  wire        ram_region   = (region == 32'd1);
  wire        const_lane   = ram_region && ((bank == 3'd5) || (bank == 3'd6));
  wire        counter_lane = ram_region && (bank == 3'd7);

  always_comb begin
    o_ila_apb_m2s      = i_apb_m2s;
    o_ila_apb_m2s.psel = i_apb_m2s.psel && !const_lane;
    if (counter_lane) begin
      o_ila_apb_m2s.paddr[BANK_LSB +: 3] = 3'd5;
    end

    if (const_lane && i_apb_m2s.psel) begin
      o_apb_s2m.pready = i_apb_m2s.penable;
      o_apb_s2m.prdata = (bank == 3'd5) ? LANE5 : LANE6;
      o_apb_s2m.pserr  = 1'b0;
    end
    else begin
      o_apb_s2m = i_ila_apb_s2m;
    end
  end

endmodule
