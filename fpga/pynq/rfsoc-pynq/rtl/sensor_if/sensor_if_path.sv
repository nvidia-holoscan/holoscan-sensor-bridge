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

`include "HOLOLINK_def.svh"

module sensor_if_path
  import HOLOLINK_pkg::*;
  import apb_pkg::*;
#(
  parameter ILA_DATA_WIDTH     = 256,
  parameter SIF_ILA_DATA_WIDTH = 512 + 2 + 7 + 64,
  parameter SIF_ILA_WSTRB      = $clog2(`DATAPATH_WIDTH/8),
  parameter ILA_DEPTH          = 16384,
  parameter SIF_ILA_DEPTH      = 8192
)(
  input  logic                          i_apb_clk,
  input  logic                          i_apb_rst,
  input  logic                          i_sif_clk,
  input  logic                          i_sif_rst,

  input  apb_m2s                        i_ila_apb_m2s,
  output apb_s2m                        o_ila_apb_s2m,
  input  apb_m2s                        i_sif_ila_apb_m2s,
  output apb_s2m                        o_sif_ila_apb_s2m,
  input  apb_m2s                        i_ram_apb_m2s,
  output apb_s2m                        o_ram_apb_s2m,

  input  logic [47:0]                   i_ptp_sec,
  input  logic [31:0]                   i_ptp_nsec,
  input  logic [31:0]                   i_cnt,

  input  logic                          i_sif_tx_axis_tvalid,
  input  logic                          i_sif_tx_axis_tlast,
  input  logic [`DATAPATH_WIDTH-1:0]    i_sif_tx_axis_tdata,
  input  logic [`DATAKEEP_WIDTH-1:0]    i_sif_tx_axis_tkeep,

  output logic                          o_sif_rx_axis_tvalid,
  output logic                          o_sif_rx_axis_tlast,
  output logic [`DATAPATH_WIDTH-1:0]    o_sif_rx_axis_tdata,
  output logic [`DATAKEEP_WIDTH-1:0]    o_sif_rx_axis_tkeep,
  output logic [`DATAUSER_WIDTH-1:0]    o_sif_rx_axis_tuser,
  input  logic                          i_sif_rx_axis_tready
);

logic                sof;
logic                eof;
logic [79:0]         ptp_ts;
logic                ptp_ts_en;
logic [11:0]         frame_cnt;
logic [ILA_DATA_WIDTH-1:0] ila_wr_data;

always_ff @ (posedge i_sif_clk) begin
  if (i_sif_rst) begin
    ptp_ts    <= '0;
    ptp_ts_en <= '0;
    frame_cnt <= '0;
  end
  else begin
    ptp_ts    <= (sof) ? i_sif_tx_axis_tdata[79:0]  : ptp_ts;
    frame_cnt <= (sof) ? i_sif_tx_axis_tdata[91:80] : frame_cnt;
    ptp_ts_en <= sof;
  end
end

assign sof = i_sif_tx_axis_tvalid;
assign eof = i_sif_tx_axis_tlast;

assign ila_wr_data[63:0]    = ptp_ts[63:0];
assign ila_wr_data[127:64]  = {i_ptp_sec[31:0], i_ptp_nsec[31:0]};
assign ila_wr_data[139:128] = frame_cnt;
assign ila_wr_data[140]     = sof;
assign ila_wr_data[141]     = eof;
assign ila_wr_data[223:142] = 'h123456789ABCDEF;
assign ila_wr_data[255:224] = i_cnt;

// Lanes 5 and 6 are the marker and are answered by the shim. The counter
// moves into stored lane 5; the shim puts lane-7 accesses back on that lane.
localparam ILA_STORED_WIDTH = 192;
logic [ILA_STORED_WIDTH-1:0] ila_stored_data;
assign ila_stored_data[159:0]   = ila_wr_data[159:0];
assign ila_stored_data[191:160] = ila_wr_data[255:224];

apb_m2s ila_stored_apb_m2s;
apb_s2m ila_stored_apb_s2m;

ts_ila_apb_shim #(
  .DEPTH ( ILA_DEPTH )
) u_ts_ila_apb_shim (
  .i_apb_m2s     ( i_ila_apb_m2s      ),
  .o_apb_s2m     ( o_ila_apb_s2m      ),
  .o_ila_apb_m2s ( ila_stored_apb_m2s ),
  .i_ila_apb_s2m ( ila_stored_apb_s2m )
);

s_apb_ila #(
  .DEPTH            ( ILA_DEPTH                      ),
  .W_DATA           ( ILA_STORED_WIDTH               )
) u_apb_ila (
  .i_aclk           ( i_apb_clk                      ),
  .i_arst           ( i_apb_rst                      ),
  .i_apb_m2s        ( ila_stored_apb_m2s             ),
  .o_apb_s2m        ( ila_stored_apb_s2m             ),
  .i_pclk           ( i_sif_clk                      ),
  .i_prst           ( i_sif_rst                      ),
  .i_trigger        ( '1                             ),
  .i_enable         ( '1                             ),
  .i_wr_data        ( ila_stored_data                ),
  .i_wr_en          ( ptp_ts_en                      ),
  .o_ctrl_reg       (                                )
);

logic [SIF_ILA_DATA_WIDTH-1:0] sif_ila_wr_data;
logic [SIF_ILA_WSTRB-1:0] sif_ila_wr_tcnt;

assign sif_ila_wr_data[511:0]   = i_sif_tx_axis_tdata;
assign sif_ila_wr_data[512]     = i_sif_tx_axis_tvalid;
assign sif_ila_wr_data[513]     = i_sif_tx_axis_tlast;
assign sif_ila_wr_data[520:514] = sif_ila_wr_tcnt;
assign sif_ila_wr_data[584:521] = {i_ptp_sec[31:0], i_ptp_nsec[31:0]};

always_comb begin
  sif_ila_wr_tcnt = '0;
  for (int j=0;j<(`DATAPATH_WIDTH/8);j=j+1) begin
    if (i_sif_tx_axis_tkeep[j]) begin
      sif_ila_wr_tcnt = j;
    end
  end
end

s_apb_ila #(
  .DEPTH            ( SIF_ILA_DEPTH                  ),
  .W_DATA           ( SIF_ILA_DATA_WIDTH             )
) u_apb_sif_ila (
  .i_aclk           ( i_apb_clk                      ),
  .i_arst           ( i_apb_rst                      ),
  .i_apb_m2s        ( i_sif_ila_apb_m2s              ),
  .o_apb_s2m        ( o_sif_ila_apb_s2m              ),
  .i_pclk           ( i_sif_clk                      ),
  .i_prst           ( i_sif_rst                      ),
  .i_trigger        ( '1                             ),
  .i_enable         ( '1                             ),
  .i_wr_data        ( sif_ila_wr_data                ),
  .i_wr_en          ( i_sif_tx_axis_tvalid           ),
  .o_ctrl_reg       (                                )
);

ram_player #(
  .W_DATA           ( 512                            )
) u_ram_player (
  .i_apb_clk        ( i_apb_clk                      ),
  .i_apb_rst        ( i_apb_rst                      ),
  .i_sif_clk        ( i_sif_clk                      ),
  .i_sif_rst        ( i_sif_rst                      ),
  .i_apb_m2s        ( i_ram_apb_m2s                  ),
  .o_apb_s2m        ( o_ram_apb_s2m                  ),
  .o_ram_axis_mux   (                                ),
  .i_ptp            ( {i_ptp_sec, i_ptp_nsec}        ),
  .o_axis_tvalid    ( o_sif_rx_axis_tvalid           ),
  .o_axis_tdata     ( o_sif_rx_axis_tdata            ),
  .o_axis_tkeep     ( o_sif_rx_axis_tkeep            ),
  .o_axis_tuser     ( o_sif_rx_axis_tuser            ),
  .o_axis_tlast     ( o_sif_rx_axis_tlast            ),
  .i_axis_tready    ( i_sif_rx_axis_tready           )
);

endmodule
