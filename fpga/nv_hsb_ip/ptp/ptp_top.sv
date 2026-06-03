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

module ptp_top
  import ptp_pkg::*;
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter HIF_CLK_FREQ = 156250000, // clock period in Hz
  parameter PTP_CLK_FREQ = 100000000, // clock period in Hz
  parameter AXI_DWIDTH   = 64,
  parameter NUM_HOST     = 2,
  parameter W_NUM_HOST   = NUM_HOST>1 ? $clog2(NUM_HOST) : 1,
  parameter KEEP_WIDTH   = AXI_DWIDTH/8,
  parameter SYNC_CLK     = 0
)(
  input                   i_pclk,
  input                   i_prst,

  input                   i_apb_clk,
  input                   i_apb_rst,
  input  apb_m2s          i_apb_m2s,
  output apb_s2m          o_apb_s2m,

  ///////////////////////////////////////////////////////////
  // AXIS
  ///////////////////////////////////////////////////////////
  input                   i_hif_clk,
  input                   i_hif_rst,
  input  [AXI_DWIDTH-1:0] i_axis_tdata,
  input  [KEEP_WIDTH-1:0] i_axis_tkeep,
  input                   i_axis_tvalid,
  input                   i_axis_tuser,
  input                   i_axis_tlast,
  output                  o_axis_tready,

  output [AXI_DWIDTH-1:0] o_axis_tdata,
  output [KEEP_WIDTH-1:0] o_axis_tkeep,
  output                  o_axis_tvalid,
  output                  o_axis_tuser,
  output                  o_axis_tlast,
  input                   i_axis_tready,

  input  [          47:0] i_dev_mac_addr,
  input                   i_ptp_tx_ts_vld, //Stamp TX Timestamp valid
  output                  o_ptp_en_sync_rx, //Enable Sync RX
  ////////////////////////////////////////////////////////////
  // PTP
  ////////////////////////////////////////////////////////////
  input                   i_enable,
  output                  o_pps,           // pulse per second
  output [          47:0] o_sec,           // v2 PTP seconds
  output [          31:0] o_nano_sec,      // v2 PTP nano seconds round to the nearest nano second
  output [          47:0] o_frac_nano_sec  // v2 PTP fraction nano seconds
);

localparam PTP_EGRESS_WIDTH   = 68*8; //Ethernet L2 is added in EGRESS module
localparam PTP_INGRESS_WIDTH  = 76*8; //1588 Delay-Resp msg = 68 bytes

//Ingress wires
logic [           7:0] ptp_rx_data [PTP_INGRESS_WIDTH/8];
logic                  ptp_rx_vld;

//Parse wires
logic        sync_msg_vld;
logic        follow_up_msg_vld;
logic        dly_resp_msg_vld;
logic        pdly_req_msg_vld;
logic        pdly_resp_msg_vld;
logic        pdly_resp_follow_up_msg_vld;

logic [15:0] flag;
logic        two_step;
assign       two_step = flag[9];
logic [47:0] cf_ns;
logic [63:0] src_clkid;
logic [15:0] src_portid;
logic [15:0] seq_id;
logic [79:0] timestamp;
logic [63:0] req_src_portidentity;
logic [15:0] req_src_portid;

//Sync wires
logic        dly_req;
logic [79:0] sync_t1_ts;
logic [79:0] sync_t2_ts;
logic [47:0] sync_cf_ns;
logic        sync_ld;

//P2P wires
logic [79:0] dly_t1_ts;
logic [79:0] dly_t2_ts;
logic [79:0] dly_t3_ts;
logic [79:0] dly_t4_ts;
logic [47:0] dly_cf_ns;
logic        dly_ld;

//First Rx Logic
logic        first_sync_rx;
logic        first_sync_rx_dpll_en;
logic        first_dly_rx;
logic        first_dly_rx_dpll_en;

logic signed [31:0] ofm;
logic               ofm_vld;
logic signed [31:0] mean_dly;
logic               mean_dly_vld;

//Timer wires
logic        pps;
logic [47:0] sec;
logic [31:0] nano_sec;
logic [47:0] inc;

//Frequency Adjustment wires
logic [31:0] fa_adj;
logic        fa_adj_vld;

//Egress wires
logic [PTP_EGRESS_WIDTH-1:0]           ptp_tx_data;
logic                                  ptp_tx_vld;
logic [$clog2(PTP_EGRESS_WIDTH/8)-1:0] ptp_tx_len;

//IP specific Delay Asymmetry
logic [W_DLY-1:0] ip_dly_asymm;

//------------------------------------------------------------------------------------------------//
// Register Map
//------------------------------------------------------------------------------------------------//
logic [31:0] ctrl_reg [ptp_nctrl];
logic [31:0] stat_reg [ptp_nstat];

logic        dpll_cfg1_en;
logic        dpll_cfg2_en;
logic        dpll_en;
logic [31:0] dpll_cfg1;
logic [31:0] dpll_cfg2;
logic [31:0] dly_asymm;
logic [ 1:0] avg_fact;
logic [ 1:0] ptp_profile;
logic        is_gPTP;
logic        is_1588_e2e;
logic        is_1588_p2p;

//Control Registers
assign dpll_cfg1_en    = ctrl_reg[ptp_ctrl][0];
assign dpll_cfg2_en    = ctrl_reg[ptp_ctrl][1];
assign dpll_en         = dpll_cfg1_en || dpll_cfg2_en;
assign dly_asymm       = ctrl_reg[ptp_ctrl_dly];
assign dpll_cfg1       = ctrl_reg[ptp_ctrl_dpll_cfg1];
assign dpll_cfg2       = ctrl_reg[ptp_ctrl_dpll_cfg2];
assign avg_fact        = ctrl_reg[ptp_ctrl_avg_fact][1:0];
assign ptp_profile     = ctrl_reg[ptp_ctrl_profile][1:0];

assign is_1588_e2e = (ptp_profile == 2'h0) ? 1'b1 : 1'b0;
assign is_1588_p2p = (ptp_profile == 2'h2) ? 1'b1 : 1'b0;
assign is_gPTP     = (ptp_profile == 2'h1) ? 1'b1 : 1'b0;

//Status Registers
assign stat_reg[ptp_sync_ts_0 - stat_ofst] = sync_t1_ts[31:0];
assign stat_reg[ptp_sync_cf_0 - stat_ofst] = sync_cf_ns[31:0];
assign stat_reg[ptp_sync_stat - stat_ofst] = {28'd0,first_dly_rx_dpll_en,first_sync_rx_dpll_en,first_dly_rx,first_sync_rx};
assign stat_reg[ptp_ofm       - stat_ofst] = { {32-W_OFM{ofm[W_OFM-1]}}, ofm[W_OFM-1:0]};
assign stat_reg[ptp_mean_dly  - stat_ofst] = { {32-W_DLY{mean_dly[W_DLY-1]}}, mean_dly[W_DLY-1:0]};
assign stat_reg[ptp_inc       - stat_ofst] = inc[31:0];
assign stat_reg[ptp_fa_adj    - stat_ofst] = { {32-W_FA{fa_adj[W_FA-1]}}, fa_adj[W_FA-1:0]};
assign stat_reg[ptp_dly_cf_0  - stat_ofst] = dly_cf_ns[31:0];
assign stat_reg[ptp_ip_dly_asymm - stat_ofst] = ip_dly_asymm;

localparam  [(ptp_nctrl*32)-1:0] RST_VAL = {'0,32'h3,32'h2,32'h2,32'h38,32'h0,32'h3,32'h0};

s_apb_reg #(
  .N_CTRL           ( ptp_nctrl         ),
  .N_STAT           ( ptp_nstat         ),
  .W_OFST           ( w_ofst            ),
  .RST_VAL          ( RST_VAL           ),
  .SYNC_CLK         ( SYNC_CLK          )
) u_reg_map  (
  // APB Interface
  .i_aclk           ( i_apb_clk         ),
  .i_arst           ( i_apb_rst         ),
  .i_apb_m2s        ( i_apb_m2s         ),
  .o_apb_s2m        ( o_apb_s2m         ),
  // User Control Signals
  .i_pclk           ( i_pclk            ),
  .i_prst           ( i_prst            ),
  .o_ctrl           ( ctrl_reg          ),
  .i_stat           ( stat_reg          )
);

//------------------------------------------------------------------------------------------------//
// First Sync Logic
//------------------------------------------------------------------------------------------------//
logic [2:0] pps_det;
logic       latch_sync;
logic       latch_dly;
logic       dpll_en_ff;
logic       dpll_en_posedge;

assign dpll_en_posedge = dpll_en && !dpll_en_ff;

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    pps_det               <= 'd0;
    first_sync_rx         <= 1'b0;
    first_dly_rx          <= 1'b0;
    first_sync_rx_dpll_en <= 1'b0;
    first_dly_rx_dpll_en  <= 1'b0;
    latch_sync            <= 1'b0;
    latch_dly             <= 1'b0;
    dpll_en_ff            <= 1'b0;
  end
  else begin
    dpll_en_ff <= dpll_en;

    latch_sync            <= sync_ld && latch_sync   ? 1'b0 :
                              pps_det[2] && dpll_en   ? 1'b1 :
                              dpll_en_posedge         ? 1'b1 :
                              !dpll_en                ? 1'b0 :
                              latch_sync;

    latch_dly             <= dly_ld && latch_dly     ? 1'b0 :
                              pps_det[2] && dpll_en   ? 1'b1 :
                              dpll_en_posedge         ? 1'b1 :
                              !dpll_en                ? 1'b0 :
                              latch_dly;

    first_sync_rx         <= sync_ld                 ? 1'b1 :
                              pps_det[2]              ? 1'b0 :
                              first_sync_rx;

    first_dly_rx          <= dly_ld                  ? 1'b1 :
                              pps_det[2]              ? 1'b0 :
                              first_dly_rx;

    first_sync_rx_dpll_en <= sync_ld && latch_sync   ? 1'b1 :
                              pps_det[2] || !dpll_en  ? 1'b0 :
                              first_sync_rx_dpll_en;

    first_dly_rx_dpll_en  <= dly_ld && latch_dly     ? 1'b1 :
                              pps_det[2] || !dpll_en  ? 1'b0 :
                              first_dly_rx_dpll_en;

    //If SYNC msg is not received in 3 consec PPS, need to re-latch
    pps_det <= sync_ld  ? 'd0                  :
                pps      ? {pps_det[1:0], 1'b1} :
                pps_det;
  end
end

logic tx_ts_vld_sync;

pulse_sync u_tx_vld_pulse_sync (
  .src_clk       ( i_hif_clk           ),
  .src_rst       ( i_hif_rst           ),
  .dst_clk       ( i_pclk              ),
  .dst_rst       ( i_prst              ),
  .i_src_pulse   ( i_ptp_tx_ts_vld     ),
  .o_dst_pulse   ( tx_ts_vld_sync   )
);

assign o_ptp_en_sync_rx = first_sync_rx_dpll_en;

//------------------------------------------------------------------------------------------------//
// Ingress
//------------------------------------------------------------------------------------------------//
ptp_ingress #(
  .AXI_DWIDTH         ( AXI_DWIDTH        ),
  .KEEP_WIDTH         ( KEEP_WIDTH        ),
  .HIF_CLK_FREQ       ( HIF_CLK_FREQ      ),
  .PTP_CLK_FREQ       ( PTP_CLK_FREQ      ),
  .PTP_INGRESS_WIDTH  ( PTP_INGRESS_WIDTH )
) u_ptp_ingress (
  .i_pclk             ( i_pclk            ),
  .i_prst             ( i_prst            ),
  .i_hif_clk          ( i_hif_clk         ),
  .i_hif_rst          ( i_hif_rst         ),
  .i_axis_tdata       ( i_axis_tdata      ),
  .i_axis_tkeep       ( i_axis_tkeep      ),
  .i_axis_tvalid      ( i_axis_tvalid     ),
  .i_axis_tuser       ( i_axis_tuser      ),
  .i_axis_tlast       ( i_axis_tlast      ),
  .o_axis_tready      ( o_axis_tready     ),
  .o_ptp_ingress_data ( ptp_rx_data       ),
  .o_ptp_ingress_vld  ( ptp_rx_vld        )
);

//------------------------------------------------------------------------------------------------//
// PARSE
//------------------------------------------------------------------------------------------------//
ptp_parser #(
  .PTP_INGRESS_WIDTH             ( PTP_INGRESS_WIDTH           )
) u_ptp_parser (
  .i_clk                         ( i_pclk                      ),
  .i_rst                         ( i_prst                      ),
  .i_is_gPTP                     ( is_gPTP                     ),
  .i_ptp_rx_data                 ( ptp_rx_data                 ),
  .i_ptp_rx_vld                  ( ptp_rx_vld                  ),
  .o_sync_msg_vld                ( sync_msg_vld                ),
  .o_follow_up_msg_vld           ( follow_up_msg_vld           ),
  .o_dly_resp_msg_vld            ( dly_resp_msg_vld            ),
  .o_pdly_req_msg_vld            ( pdly_req_msg_vld            ),
  .o_pdly_resp_msg_vld           ( pdly_resp_msg_vld           ),
  .o_pdly_resp_follow_up_msg_vld ( pdly_resp_follow_up_msg_vld ),
  .o_flag                        ( flag                        ),
  .o_cf_ns                       ( cf_ns                       ),
  .o_src_clkid                   ( src_clkid                   ),
  .o_src_portid                  ( src_portid                  ),
  .o_seq_id                      ( seq_id                      ),
  .o_timestamp                   ( timestamp                   ),
  .o_req_src_portidentity        ( req_src_portidentity        ),
  .o_req_src_portid              ( req_src_portid              )
);

//------------------------------------------------------------------------------------------------//
// PTP TIMER
//------------------------------------------------------------------------------------------------//
ptp_timer #(
  .HIF_CLK_FREQ       ( HIF_CLK_FREQ            ),
  .PTP_CLK_FREQ       ( PTP_CLK_FREQ            ),
  .AXI_DWIDTH         ( AXI_DWIDTH              ),
  .W_DLY              ( W_DLY                   ),
  .W_FA               ( W_FA                    )
) u_ptp_timer (
  .i_pclk             ( i_pclk                  ),
  .i_prst             ( i_prst                  ),
  .i_enable           ( i_enable                ),
  .i_sync_ts          ( sync_t1_ts              ),
  .i_sync_ts_vld      ( sync_ld && latch_sync   ),
  .i_fa_adj           ( fa_adj [W_FA-1:0]       ),
  .i_fa_adj_vld       ( fa_adj_vld              ),
  .o_pps              ( pps                     ),
  .o_sec              ( sec                     ),
  .o_nano_sec         ( nano_sec                ),
  .o_frac_nano_sec    ( o_frac_nano_sec         ),
  .o_inc              ( inc                     ),
  .o_dly_asymm_ns     ( ip_dly_asymm[W_DLY-1:0] )
);

//------------------------------------------------------------------------------------------------//
// PTP SYNCHRONIZATION
//------------------------------------------------------------------------------------------------//
ptp_sync u_ptp_sync   (
  .i_pclk              ( i_pclk            ),
  .i_prst              ( i_prst            ),
  .i_sec               ( sec               ),
  .i_nano_sec          ( nano_sec          ),
  .i_sync_msg_vld      ( sync_msg_vld      ),
  .i_follow_up_msg_vld ( follow_up_msg_vld ),
  .i_flag              ( flag              ),
  .i_cf_ns             ( cf_ns             ),
  .i_timestamp         ( timestamp         ),
  .i_seq_id            ( seq_id            ),
  .i_src_clkid         ( src_clkid         ),
  .i_src_portid        ( src_portid        ),
  .o_dly_req           ( dly_req           ),
  .o_sync_t1_ts        ( sync_t1_ts        ),
  .o_sync_t2_ts        ( sync_t2_ts        ),
  .o_sync_cf_ns        ( sync_cf_ns        ),
  .o_sync_ld           ( sync_ld           )
);

//------------------------------------------------------------------------------------------------//
// PTP DELAY MECH
//------------------------------------------------------------------------------------------------//
ptp_dly #(
  .PTP_EGRESS_WIDTH              ( PTP_EGRESS_WIDTH            ),
  .NUM_HOST                      ( NUM_HOST                    )
) u_ptp_dly  (
  .i_pclk                        ( i_pclk                      ),
  .i_prst                        ( i_prst                      ),
  .i_hif_clk                     ( i_hif_clk                   ),
  .i_hif_rst                     ( i_hif_rst                   ),
  .i_dev_mac_addr                ( i_dev_mac_addr              ),
  .i_tx_ts_vld                   ( tx_ts_vld_sync              ),
  .o_ptp_tx_data                 ( ptp_tx_data                 ),
  .o_ptp_tx_vld                  ( ptp_tx_vld                  ),
  .i_is_1588_e2e                 ( is_1588_e2e                 ),
  .i_is_1588_p2p                 ( is_1588_p2p                 ),
  .i_is_gPTP                     ( is_gPTP                     ),
  .i_dly_resp_msg_vld            ( dly_resp_msg_vld            ),
  .i_pdly_req_msg_vld            ( pdly_req_msg_vld            ),
  .i_pdly_resp_msg_vld           ( pdly_resp_msg_vld           ),
  .i_pdly_resp_follow_up_msg_vld ( pdly_resp_follow_up_msg_vld ),
  .i_two_step                    ( two_step                    ),
  .i_cf_ns                       ( cf_ns                       ),
  .i_timestamp                   ( timestamp                   ),
  .i_seq_id                      ( seq_id                      ),
  .i_src_clkid                   ( src_clkid                   ),
  .i_src_portid                  ( src_portid                  ),
  .i_req_src_portidentity        ( req_src_portidentity        ),
  .i_req_src_portid              ( req_src_portid              ),
  .i_dly_req                     ( dly_req                     ),
  .i_pps                         ( pps                         ),
  .i_sec                         ( sec                         ),
  .i_nano_sec                    ( nano_sec                    ),
  .o_ptp_tx_len                  ( ptp_tx_len                  ),
  .o_dly_t1_ts                   ( dly_t1_ts                   ),
  .o_dly_t2_ts                   ( dly_t2_ts                   ),
  .o_dly_t3_ts                   ( dly_t3_ts                   ),
  .o_dly_t4_ts                   ( dly_t4_ts                   ),
  .o_dly_cf_ns                   ( dly_cf_ns                   ),
  .o_dly_ld                      ( dly_ld                      )
);

//------------------------------------------------------------------------------------------------//
// PTP DELAY/OFFSET CALCULATION
//------------------------------------------------------------------------------------------------//
ptp_calc #(
  .W_DLY              ( W_DLY                            ),
  .W_OFM              ( W_OFM                            )
) u_ptp_calc  (
  .i_pclk             ( i_pclk                           ),
  .i_prst             ( i_prst                           ),
  .i_cfg_avg_fact     ( avg_fact                         ),
  .i_cfg_dly_asymm    ( dly_asymm                        ),
  .i_ip_dly_asymm     ( ip_dly_asymm                     ),
  .i_P2P              ( is_gPTP || is_1588_p2p           ),
  .i_sync_t1_ts       ( sync_t1_ts                       ),
  .i_sync_t2_ts       ( sync_t2_ts                       ),
  .i_sync_cf_ns       ( sync_cf_ns                       ),
  .i_sync_ld          ( sync_ld && first_sync_rx_dpll_en ),
  .o_ofm              ( ofm      [W_OFM-1:0]             ),
  .o_ofm_vld          ( ofm_vld                          ),
  .i_dly_t1_ts        ( dly_t1_ts                        ),
  .i_dly_t2_ts        ( dly_t2_ts                        ),
  .i_dly_t3_ts        ( dly_t3_ts                        ),
  .i_dly_t4_ts        ( dly_t4_ts                        ),
  .i_dly_cf_ns        ( dly_cf_ns                        ),
  .i_dly_ld           ( dly_ld && first_dly_rx_dpll_en   ),
  .o_mean_dly         ( mean_dly [W_DLY-1:0]             ),
  .o_mean_dly_vld     ( mean_dly_vld                     )
);

//------------------------------------------------------------------------------------------------//
// DIGITAL PLL FOR PHASE ADJUSTMENT
//------------------------------------------------------------------------------------------------//
dpll #(
  .W_FA                ( W_FA               )
) u_dpll (
  .i_clk               ( i_pclk             ),
  .i_rst               ( i_prst             ),
  .cfg_gain_1_en       ( dpll_cfg1_en       ),
  .cfg_gain_2_en       ( dpll_cfg2_en       ),
  .cfg_gain_1          ( dpll_cfg1          ),
  .cfg_gain_2          ( dpll_cfg2          ),
  .i_pd                ( ofm   [W_FA-1:0]   ),
  .i_pd_vld            ( ofm_vld && dpll_en ),
  .o_fa                ( fa_adj [W_FA-1:0]  ),
  .o_fa_vld            ( fa_adj_vld         )
);

//------------------------------------------------------------------------------------------------//
// Egress
//------------------------------------------------------------------------------------------------//

ptp_egress #(
  .AXI_DWIDTH        ( AXI_DWIDTH       ),
  .NUM_HOST          ( NUM_HOST         ),
  .KEEP_WIDTH        ( KEEP_WIDTH       ),
  .PTP_EGRESS_WIDTH  ( PTP_EGRESS_WIDTH )
) u_ptp_egress (
  .i_pclk            ( i_pclk           ),
  .i_prst            ( i_prst           ),
  .i_ptp_egress_data ( ptp_tx_data      ),
  .i_ptp_egress_vld  ( ptp_tx_vld       ),
  .i_ptp_egress_len  ( ptp_tx_len       ),
  .o_ptp_egress_busy (                  ),
  .i_hif_clk         ( i_hif_clk        ),
  .i_hif_rst         ( i_hif_rst        ),
  .o_axis_tdata      ( o_axis_tdata     ),
  .o_axis_tkeep      ( o_axis_tkeep     ),
  .o_axis_tvalid     ( o_axis_tvalid    ),
  .o_axis_tuser      ( o_axis_tuser     ),
  .o_axis_tlast      ( o_axis_tlast     ),
  .i_axis_tready     ( i_axis_tready    )
);

assign o_pps      = pps;
assign o_sec      = sec;
assign o_nano_sec = nano_sec;

endmodule

