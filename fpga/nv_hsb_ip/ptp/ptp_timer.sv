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

module ptp_timer
  import ptp_pkg::*;
#(
  parameter HIF_CLK_FREQ    = 156250000, // clock period in Hz
  parameter PTP_CLK_FREQ    = 10000000,
  parameter AXI_DWIDTH      = 64,
  parameter W_DLY           = 32,
  parameter W_FA            = 32
)(
  input                      i_pclk,
  input                      i_prst,
  input                      i_enable,

  input          [79:0]      i_sync_ts,
  input                      i_sync_ts_vld,

  input   signed [W_FA-1:0]  i_fa_adj,
  input                      i_fa_adj_vld,

  output                     o_pps,           // pulse per second
  output         [47:0]      o_sec,           // v2 PTP seconds
  output         [31:0]      o_nano_sec,      // v2 PTP nano seconds round to the nearest nano second
  output         [47:0]      o_frac_nano_sec, // v2 PTP fraction nano seconds
  output         [47:0]      o_inc,

  output         [W_DLY-1:0] o_dly_asymm_ns
);

localparam            W_FRAC_NS   = 24;
localparam            W_NS        = 31 + W_FRAC_NS;

localparam            W_INC_NS    = $clog2((10**9/PTP_CLK_FREQ) + 1);
localparam            W_INC       = W_INC_NS + W_FRAC_NS;

localparam   [W_NS:0] default_inc = 10**9 * (2**W_FRAC_NS) / PTP_CLK_FREQ ;

localparam            W_HIF_INC_NS    = $clog2((10**9/HIF_CLK_FREQ) + 1);
localparam            W_HIF_INC       = W_HIF_INC_NS + W_FRAC_NS;

localparam   [W_NS:0] default_hif_inc = 10**9 * (2**W_FRAC_NS) / HIF_CLK_FREQ ;


`ifdef SIMULATION
  localparam W_GT_BIL_COMP = 30;
`else
  localparam W_GT_BIL_COMP = 10;
`endif

logic signed [W_INC:0] inc;

logic                  pps;
logic           [31:0] sec;
logic           [31:0] sec_inc;

assign sec_inc = sec + 1'b1;

logic signed [W_FA-1:0] fa_adj;
logic                   fa_adj_vld;

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    fa_adj     <= 0;
    fa_adj_vld <= 0;
  end
  else begin
    fa_adj     <= i_fa_adj;
    fa_adj_vld <= i_fa_adj_vld;
  end
end

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    inc     <= {1'b0, default_inc[W_INC-1:0]};
  end
  else begin
    if (fa_adj_vld) begin
      inc <= signed'({1'b0, default_inc[W_INC-1:0]}) + signed'(fa_adj);
    end
  end
end

//------------------------------------------------------------------------------------------------//
// DELAY ASYMMETRY
//------------------------------------------------------------------------------------------------//

localparam AXIS_BYTE              = AXI_DWIDTH/8;
localparam RX_PARSER_AXIS_BUF_MOD = 64%AXIS_BYTE != 0 ? 1 : 0;
localparam PTP_AXIS_DROP_MOD      = 14%AXIS_BYTE != 0 ? 1 : 0;
localparam PTP_AXIS_VEC_MOD       = 54%AXIS_BYTE != 0 ? 1 : 0;

localparam PTP_IS_LOWER_FREQ = PTP_CLK_FREQ < HIF_CLK_FREQ ? 1 : 0;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                   |                                PTP                         |
//                                                   |  AXIS_BUFFER   |            AXIS_TO_VEC              | REG |
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
localparam NUM_CLK_DLY_ASYMM_PTP = PTP_IS_LOWER_FREQ ? 2 +             (50/AXIS_BYTE) + PTP_AXIS_VEC_MOD + 1 + 1
                                                      : 6                                                + 1;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                   |                          RX PARSER                   |                          PTP                      |
//                                                   |               AXIS_BUFFER               | IPV4 CHECK |           AXIS_DROP                | AXIS_BUFFER  |
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
localparam NUM_CLK_DLY_ASYMM_HIF = PTP_IS_LOWER_FREQ ? (64/AXIS_BYTE) + RX_PARSER_AXIS_BUF_MOD  +      8     + (14/AXIS_BYTE) + PTP_AXIS_DROP_MOD + 2 
                                                      : (64/AXIS_BYTE) + RX_PARSER_AXIS_BUF_MOD +      8     + (64/AXIS_BYTE)+ RX_PARSER_AXIS_BUF_MOD;

localparam W_DLY_ASYMM_PTP = $clog2(NUM_CLK_DLY_ASYMM_PTP+1);
localparam W_DLY_ASYMM_HIF = $clog2(NUM_CLK_DLY_ASYMM_HIF+1);

localparam W_INC_TOT = W_INC+W_DLY_ASYMM_PTP > W_HIF_INC+W_DLY_ASYMM_HIF ? W_INC+W_DLY_ASYMM_PTP : W_HIF_INC+W_DLY_ASYMM_HIF;

logic [W_INC     + W_DLY_ASYMM_PTP:0] dly_asymm_ptp;
logic [W_HIF_INC + W_DLY_ASYMM_HIF:0] dly_asymm_hif;
logic [W_INC_TOT:0]                   dly_asymm_tot;
logic [W_INC_NS  + W_DLY_ASYMM_PTP:0] dly_asymm_ns;

assign dly_asymm_ptp  = NUM_CLK_DLY_ASYMM_PTP * default_inc[W_INC-1:0];
assign dly_asymm_hif  = NUM_CLK_DLY_ASYMM_HIF * default_hif_inc[W_HIF_INC-1:0];
assign dly_asymm_tot  = dly_asymm_ptp + dly_asymm_hif;

assign o_dly_asymm_ns = { {W_DLY-(W_INC_TOT-W_FRAC_NS){1'b0}}, dly_asymm_tot[W_INC_TOT:W_FRAC_NS]};

//------------------------------------------------------------------------------------------------//
// ROLLOVER LOGIC
//------------------------------------------------------------------------------------------------//

logic [W_NS-1:0] nano_sec; //nano seconds incl frac nanosec
logic [W_NS-1:0] nano_sec_inc;
logic [W_NS-1:0] nano_sec_2inc;
logic [W_NS-1:0] nano_sec_2inc_r;

assign nano_sec_inc [W_NS-1:0] = nano_sec[W_NS-1:0] + unsigned'(inc[W_INC-1:0]); //Next Clock value
assign nano_sec_2inc[W_NS-1:1] = nano_sec[W_NS-1:1] + unsigned'(inc[W_INC-1:0]); //Next Next Clock value
assign nano_sec_2inc[0]        = nano_sec[0];

logic nano_sec_gt_bil;
logic nano_sec_gt_bil_msb;
logic nano_sec_gt_bil_lsb;

assign nano_sec_gt_bil_lsb = (nano_sec_2inc_r[W_FRAC_NS+W_GT_BIL_COMP-1:W_FRAC_NS+8] >= BILLION[W_GT_BIL_COMP-1:8]) ? 1'b1 : 1'b0;
assign nano_sec_gt_bil     = nano_sec_gt_bil_msb && nano_sec_gt_bil_lsb;

logic [29:0] nano_sec_sub_bil;
assign nano_sec_sub_bil = nano_sec_2inc_r[W_NS-2:W_FRAC_NS] - BILLION[29:0];

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    pps                 <= 1'b0;
    sec                 <= 0;
    nano_sec            <= 0;
    nano_sec_2inc_r     <= 0;
    nano_sec_gt_bil_msb <= 1'b0;
  end
  else begin
    pps   <= nano_sec_gt_bil ? 1'b1 :
                                1'b0;

    sec   <= i_sync_ts_vld   ? i_sync_ts[79:32] :
              nano_sec_gt_bil ? sec_inc          :
                                sec;

    nano_sec_gt_bil_msb <= (nano_sec_2inc_r[W_NS-1:W_FRAC_NS+W_GT_BIL_COMP] >= BILLION[30:W_GT_BIL_COMP]) ? 1'b1 : 1'b0;

    nano_sec[W_NS     -1:W_FRAC_NS]        <= i_sync_ts_vld   ? i_sync_ts[31:0]  :
                                              nano_sec_gt_bil ? nano_sec_sub_bil :
                                                                nano_sec_inc[W_NS-1:W_FRAC_NS];

    nano_sec[W_FRAC_NS-1:        0]        <= i_sync_ts_vld   ? 0                              :
                                              nano_sec_gt_bil ? nano_sec_2inc_r[W_FRAC_NS-1:0] :
                                                                nano_sec_inc[W_FRAC_NS-1:0];

    nano_sec_2inc_r[W_NS     -1:W_FRAC_NS] <= i_sync_ts_vld   ? i_sync_ts[31:0] + unsigned'(inc[W_INC-1:W_FRAC_NS]) :
                                              nano_sec_gt_bil ? 0                                                   :
                                                                nano_sec_2inc[W_NS-1:W_FRAC_NS];

    nano_sec_2inc_r[W_FRAC_NS-1:        0] <= i_sync_ts_vld   ? unsigned'(inc[W_FRAC_NS-1:0])  :
                                              nano_sec_gt_bil ? 0                              :
                                                                nano_sec_2inc[W_FRAC_NS-1:0];
  end
end

assign o_pps           = pps;
assign o_sec           = {16'd0, sec};
assign o_nano_sec      = {2'h0, nano_sec[W_NS-1:W_FRAC_NS]};
assign o_frac_nano_sec = nano_sec[W_FRAC_NS-1:0];
assign o_inc           = inc;

endmodule

