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

module ptp_ingress
  import axis_pkg::*;
#(
  parameter  AXI_DWIDTH        = 64,
  parameter  KEEP_WIDTH        = AXI_DWIDTH/8,
  parameter  HIF_CLK_FREQ      = 156250000,
  parameter  PTP_CLK_FREQ      = 100000000,
  parameter  PTP_INGRESS_WIDTH = 64*8,
  localparam PTP_WIDTH         = $clog2(PTP_INGRESS_WIDTH/8)
)(
  input                           i_pclk,
  input                           i_prst,

  input                           i_hif_clk,
  input                           i_hif_rst,

  input  [AXI_DWIDTH-1:0]         i_axis_tdata,
  input  [KEEP_WIDTH-1:0]         i_axis_tkeep,
  input                           i_axis_tvalid,
  input                           i_axis_tuser,
  input                           i_axis_tlast,
  output                          o_axis_tready,

  output [                   7:0] o_ptp_ingress_data   [PTP_INGRESS_WIDTH/8],
  output                          o_ptp_ingress_vld
);

localparam BUF_DEPTH = PTP_CLK_FREQ >  HIF_CLK_FREQ ? 16 :
                      AXI_DWIDTH   >= 256          ? 8  :
                      AXI_DWIDTH   >= 32           ? 32 :
                      AXI_DWIDTH   >= 16           ? 64 :
                                                      128;

logic [        AXI_DWIDTH-1:0] inp_axis_tdata;
logic [    (AXI_DWIDTH/8)-1:0] inp_axis_tkeep;
logic                          inp_axis_tvalid;
logic                          inp_axis_tuser;
logic                          inp_axis_tlast;
logic                          inp_axis_tready;

logic [        AXI_DWIDTH-1:0] cdc_axis_tdata;
logic [    (AXI_DWIDTH/8)-1:0] cdc_axis_tkeep;
logic                          cdc_axis_tvalid;
logic                          cdc_axis_tuser;
logic                          cdc_axis_tlast;
logic                          cdc_axis_tready;

logic [        AXI_DWIDTH-1:0] drp_axis_tdata;
logic [    (AXI_DWIDTH/8)-1:0] drp_axis_tkeep;
logic                          drp_axis_tvalid;
logic                          drp_axis_tuser;
logic                          drp_axis_tlast;
logic                          drp_axis_tready;

logic [        AXI_DWIDTH-1:0] ptp_axis_reg_tdata;
logic [    (AXI_DWIDTH/8)-1:0] ptp_axis_reg_tkeep;
logic                          ptp_axis_reg_tvalid;
logic                          ptp_axis_reg_tuser;
logic                          ptp_axis_reg_tlast;
logic                          ptp_axis_reg_tready;

logic                          axis_busy;
logic [ PTP_INGRESS_WIDTH-1:0] ptp_ingress_data;
logic [                   7:0] ptp_ingress_byte [PTP_INGRESS_WIDTH/8];
logic                          ptp_ingress_vld;
logic                          ptp_done;

assign o_axis_tready    = inp_axis_tready;
assign inp_axis_tdata   = i_axis_tdata;
assign inp_axis_tkeep   = i_axis_tkeep;
assign inp_axis_tvalid  = i_axis_tvalid;
assign inp_axis_tuser   = i_axis_tuser;
assign inp_axis_tlast   = i_axis_tlast;

axis_drop #(
  .DROP_WIDTH         ( 14*8                 ),
  .DWIDTH             ( AXI_DWIDTH           )
) ptp_axis_drop (
  .clk                ( i_hif_clk            ),
  .rst                ( i_hif_rst            ),
  .i_axis_rx_tvalid   ( inp_axis_tvalid      ),
  .i_axis_rx_tdata    ( inp_axis_tdata       ),
  .i_axis_rx_tlast    ( inp_axis_tlast       ),
  .i_axis_rx_tuser    ( inp_axis_tuser       ),
  .i_axis_rx_tkeep    ( inp_axis_tkeep       ),
  .o_axis_rx_tready   ( inp_axis_tready      ),
  .o_axis_tx_tvalid   ( drp_axis_tvalid      ),
  .o_axis_tx_tdata    ( drp_axis_tdata       ),
  .o_axis_tx_tlast    ( drp_axis_tlast       ),
  .o_axis_tx_tuser    ( drp_axis_tuser       ),
  .o_axis_tx_tkeep    ( drp_axis_tkeep       ),
  .i_axis_tx_tready   ( drp_axis_tready      )
);

axis_buffer # (
  .IN_DWIDTH  ( AXI_DWIDTH     ),
  .OUT_DWIDTH ( AXI_DWIDTH     ),
  .DUAL_CLOCK ( 1              ),
  .BUF_DEPTH  ( BUF_DEPTH      ),
  .W_USER     ( 1              )
) u_axis_buffer (
  .in_clk            ( i_hif_clk             ),
  .in_rst            ( i_hif_rst             ),
  .out_clk           ( i_pclk                ),
  .out_rst           ( i_prst                ),
  .i_axis_rx_tvalid  ( drp_axis_tvalid       ),
  .i_axis_rx_tdata   ( drp_axis_tdata        ),
  .i_axis_rx_tlast   ( drp_axis_tlast        ),
  .i_axis_rx_tuser   ( drp_axis_tuser        ),
  .i_axis_rx_tkeep   ( drp_axis_tkeep        ),
  .o_axis_rx_tready  ( drp_axis_tready       ),
  .o_fifo_aempty     (                       ),
  .o_fifo_afull      (                       ),
  .o_fifo_empty      (                       ),
  .o_fifo_full       (                       ),
  .o_axis_tx_tvalid  ( cdc_axis_tvalid       ),
  .o_axis_tx_tdata   ( cdc_axis_tdata        ),
  .o_axis_tx_tlast   ( cdc_axis_tlast        ),
  .o_axis_tx_tuser   ( cdc_axis_tuser        ),
  .o_axis_tx_tkeep   ( cdc_axis_tkeep        ),
  .i_axis_tx_tready  ( cdc_axis_tready       )
);

axis_reg #(
  .DWIDTH                       ( AXI_DWIDTH+KEEP_WIDTH+1+1                                                        ),
  .SKID                         ( 1                                                                                )
) ptp_axis_reg (
  .clk                          ( i_pclk                                                                           ),
  .rst                          ( i_prst                                                                           ),
  .i_axis_rx_tvalid             ( cdc_axis_tvalid                                                                  ),
  .i_axis_rx_tdata              ( {cdc_axis_tdata, cdc_axis_tlast, cdc_axis_tuser, cdc_axis_tkeep}                 ),
  .o_axis_rx_tready             ( cdc_axis_tready                                                                  ),
  .o_axis_tx_tvalid             ( ptp_axis_reg_tvalid                                                              ),
  .o_axis_tx_tdata              ( {ptp_axis_reg_tdata, ptp_axis_reg_tlast, ptp_axis_reg_tuser, ptp_axis_reg_tkeep} ),
  .i_axis_tx_tready             ( ptp_axis_reg_tready                                                              )
);

axis_to_vec #(
  .AXI_DWIDTH       ( AXI_DWIDTH           ),
  .DATA_WIDTH       ( PTP_INGRESS_WIDTH    )
) axis_to_ptp (
  .clk              ( i_pclk               ),
  .rst              ( i_prst               ),
  .i_axis_rx_tvalid ( ptp_axis_reg_tvalid  ),
  .i_axis_rx_tdata  ( ptp_axis_reg_tdata   ),
  .i_axis_rx_tlast  ( ptp_axis_reg_tlast   ),
  .i_axis_rx_tuser  ( ptp_axis_reg_tuser   ),
  .i_axis_rx_tkeep  ( ptp_axis_reg_tkeep   ),
  .o_axis_rx_tready ( ptp_axis_reg_tready  ),
  .i_done           ( ptp_done             ),
  .o_data           ( ptp_ingress_data     ),
  .o_valid          ( ptp_ingress_vld      ),
  .o_busy           ( axis_busy            ),
  .o_byte_cnt       (                      )
);

always_comb begin
  for (int i=0; i<PTP_INGRESS_WIDTH/8; i++) begin
    ptp_ingress_byte [i] = ptp_ingress_data[(i*8)+:8];
  end
end

assign o_ptp_ingress_data = ptp_ingress_byte;
assign o_ptp_ingress_vld  = ptp_ingress_vld;
assign ptp_done   = 1'b1;

`ifdef ASSERT_ON
  axis_checker #(
  .STBL_CHECK  (1),
  .NLST_BT_B2B (1),
  .MIN_PKTL_CHK (0),
  .MAX_PKTL_CHK (0),
  .AXI_TDATA   (AXI_DWIDTH),
  .AXI_TUSER   (1),
`ifdef SIMULATION
    .SIMULATION(1),
`endif
  .PKT_MIN_LENGTH  (58),
  .PKT_MAX_LENGTH  (`HOST_MTU)
  ) assert_input_axis (
  .clk            (i_hif_clk),
  .rst            (i_hif_rst),
  .axis_tvalid    (i_axis_tvalid),
  .axis_tlast     (i_axis_tlast),
  .axis_tkeep     (i_axis_tkeep),
  .axis_tdata     (i_axis_tdata),
  .axis_tuser     (i_axis_tuser),
  .axis_tready    (o_axis_tready),
  .byte_count     (),
  .byte_count_nxt ()
  );
`endif

endmodule
