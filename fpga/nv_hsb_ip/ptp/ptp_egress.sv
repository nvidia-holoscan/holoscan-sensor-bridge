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

module ptp_egress
#(
  parameter  AXI_DWIDTH       = 64,
  parameter  NUM_HOST         = 2,
  parameter  W_NUM_HOST       = NUM_HOST>1 ? $clog2(NUM_HOST) : 1,
  parameter  KEEP_WIDTH       = AXI_DWIDTH/8,
  parameter  PTP_EGRESS_WIDTH = 64*8,
  localparam PTP_WIDTH        = $clog2(PTP_EGRESS_WIDTH/8)
)(
  input                          i_pclk,
  input                          i_prst,

  input  [PTP_EGRESS_WIDTH-1:0]  i_ptp_egress_data,
  input                          i_ptp_egress_vld,
  input  [PTP_WIDTH-1:0]         i_ptp_egress_len,
  output                         o_ptp_egress_busy,

  input                          i_hif_clk,
  input                          i_hif_rst,

  output [AXI_DWIDTH-1:0]        o_axis_tdata,
  output [KEEP_WIDTH-1:0]        o_axis_tkeep,
  output                         o_axis_tvalid,
  output                         o_axis_tuser,
  output                         o_axis_tlast,
  input                          i_axis_tready
);

localparam BUF_DEPTH = (2**($clog2(PTP_EGRESS_WIDTH/AXI_DWIDTH)+1)) < 8 ? 8 : 2**($clog2(PTP_EGRESS_WIDTH/AXI_DWIDTH)+1);

logic                      axis_is_busy;
logic [AXI_DWIDTH-1:0]     req_axis_tdata;
logic [(AXI_DWIDTH/8)-1:0] req_axis_tkeep;
logic                      req_axis_tvalid;
logic                      req_axis_tuser;
logic                      req_axis_tlast;
logic                      req_axis_tready;

logic [AXI_DWIDTH-1:0]     ptp_tx_axis_tdata;
logic [(AXI_DWIDTH/8)-1:0] ptp_tx_axis_tkeep;
logic                      ptp_tx_axis_tvalid;
logic                      ptp_tx_axis_tuser;
logic                      ptp_tx_axis_tlast;
logic                      ptp_tx_axis_tready;


vec_to_axis_dyn_len #(
  .AXI_DWIDTH       ( AXI_DWIDTH         ),
  .DATA_WIDTH       ( PTP_EGRESS_WIDTH   )
) req_to_axis (
  .clk              ( i_hif_clk          ),
  .rst              ( i_hif_rst          ),
  .trigger          ( i_ptp_egress_vld   ),
  .data             ( i_ptp_egress_data  ),
  .byte_len         ( i_ptp_egress_len   ),
  .is_busy          ( axis_is_busy       ),
  .o_axis_tx_tvalid ( ptp_tx_axis_tvalid ),
  .o_axis_tx_tdata  ( ptp_tx_axis_tdata  ),
  .o_axis_tx_tlast  ( ptp_tx_axis_tlast  ),
  .o_axis_tx_tuser  ( ptp_tx_axis_tuser  ),
  .o_axis_tx_tkeep  ( ptp_tx_axis_tkeep  ),
  .i_axis_tx_tready ( ptp_tx_axis_tready )
);

assign o_axis_tvalid = ptp_tx_axis_tvalid;
assign o_axis_tdata  = ptp_tx_axis_tdata;
assign o_axis_tlast  = ptp_tx_axis_tlast;
assign o_axis_tuser  = ptp_tx_axis_tuser;
assign o_axis_tkeep  = ptp_tx_axis_tkeep;


assign ptp_tx_axis_tready = i_axis_tready;
assign o_ptp_egress_busy = axis_is_busy;

`ifdef ASSERT_ON
  axis_checker #(
  .STBL_CHECK  (1),
  .NLST_BT_B2B (1),
  .MIN_PKTL_CHK (1),
  .MAX_PKTL_CHK (1),
  .AXI_TDATA   (AXI_DWIDTH),
  .AXI_TUSER   (1),
`ifdef SIMULATION
    .SIMULATION(1),
`endif
  .PKT_MIN_LENGTH  (58),
  .PKT_MAX_LENGTH  (68)
  ) assert_output_axis (
  .clk            (i_hif_clk),
  .rst            (i_hif_rst),
  .axis_tvalid    (o_axis_tvalid),
  .axis_tlast     (o_axis_tlast),
  .axis_tkeep     (o_axis_tkeep),
  .axis_tdata     (o_axis_tdata),
  .axis_tuser     (o_axis_tuser),
  .axis_tready    (i_axis_tready),
  .byte_count     (),
  .byte_count_nxt ()
  );
`endif

endmodule
