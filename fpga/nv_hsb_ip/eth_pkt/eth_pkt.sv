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

module eth_pkt
  import apb_pkg::*;
  import regmap_pkg::*;
#(// port 0 = data; port 1 = cmd, port 2 = error, port 3 = enum_int, port 4 = evt_int, port 5 = bootp, port 6 = rsp
  parameter           N_INPT = 7,
  parameter           W_DATA = 64,
  localparam          W_KEEP = W_DATA/8,
  parameter           SYNC_CLK = 0
)(
  input               i_pclk,
  input               i_prst,
  // Register Map, abp clk domain
  input               i_aclk,
  input               i_arst,
  input  apb_m2s      i_apb_m2s,
  output apb_s2m      o_apb_s2m,
  // PTP Timestamp
  output              o_ptp_val,
  output              o_del_req_val,
  output logic        o_pkt_inc,
  // AXIS From Multiple Sources
  input  [N_INPT-1:0] i_axis_tvalid,
  input  [N_INPT-1:0] i_axis_tlast,
  input  [W_DATA-1:0] i_axis_tdata    [N_INPT],
  input  [W_KEEP-1:0] i_axis_tkeep    [N_INPT],
  input  [N_INPT-1:0] i_axis_tuser,
  output [N_INPT-1:0] o_axis_tready,
  // AXIS to MAC
  output              o_axis_tvalid,
  output              o_axis_tlast,
  output [W_DATA-1:0] o_axis_tdata,
  output [W_KEEP-1:0] o_axis_tkeep,
  output              o_axis_tuser,
  input               i_axis_tready
);

logic [31:0] ctrl_reg [eth_pkt_nctrl];
logic [31:0] stat_reg [eth_pkt_nstat];

assign stat_reg[eth_pkt_stat-stat_ofst] = '{default:0};

s_apb_reg #(
  .N_CTRL    ( eth_pkt_nctrl  ),
  .N_STAT    ( eth_pkt_nstat  ),
  .W_OFST    ( w_ofst         ),
  .SYNC_CLK  ( SYNC_CLK       )
) u_reg_map  (
  // APB Interface
  .i_aclk    ( i_aclk         ),
  .i_arst    ( i_arst         ),
  .i_apb_m2s ( i_apb_m2s      ),
  .o_apb_s2m ( o_apb_s2m      ),
  // User Control Signals
  .i_pclk    ( i_pclk         ),
  .i_prst    ( i_prst         ),
  .o_ctrl    ( ctrl_reg       ),
  .i_stat    ( stat_reg       )
);

//------------------------------------------------------------------------------------------------//
// Data and Command AXIS Arbitration
//------------------------------------------------------------------------------------------------//


logic              arb_axis_tvalid;
logic              arb_axis_tlast;
logic [W_DATA-1:0] arb_axis_tdata;
logic [W_KEEP-1:0] arb_axis_tkeep;
logic              arb_axis_tuser;
logic              arb_axis_tready;


logic [2       :0] arb_idx;
logic [2:0]        arb_pipe [4:0];

axis_arb #(
  .N_INPT        ( N_INPT                     ),
  .W_DATA        ( W_DATA                     ),
  .W_USER        ( 1                          ),
  .ARB_TYPE      ( "PRIORITY"                 ),
  .SKID          ( 5'b01011                   ) // SKID all high speed paths: data plane, ptp, pause, and bridge
) u_data_cmd_arb (
  .i_clk         ( i_pclk                     ),
  .i_rst         ( i_prst                     ),
  .i_axis_tvalid ( i_axis_tvalid              ),
  .i_axis_tlast  ( i_axis_tlast               ),
  .i_axis_tdata  ( i_axis_tdata               ),
  .i_axis_tkeep  ( i_axis_tkeep               ),
  .o_axis_tready ( o_axis_tready              ),
  .i_axis_tuser  ( '{default:0}               ),
  .o_axis_idx    ( arb_idx                    ),
  .o_axis_tvalid ( arb_axis_tvalid            ),
  .o_axis_tlast  ( arb_axis_tlast             ),
  .o_axis_tdata  ( arb_axis_tdata             ),
  .o_axis_tkeep  ( arb_axis_tkeep             ),
  .o_axis_tuser  ( arb_axis_tuser             ),
  .i_axis_tready ( arb_axis_tready            )
);

logic ptp_val;
always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    ptp_val <= 1'b0;
  end
  else begin
    if (arb_axis_tvalid && arb_axis_tready && arb_axis_tlast && arb_idx == 3'd1) begin
      ptp_val <= 1'b1;
    end
    else begin
      ptp_val <= 1'b0;
    end
  end
end

assign o_ptp_val = ptp_val;


//------------------------------------------------------------------------------------------------//
// Register output
//------------------------------------------------------------------------------------------------//

localparam PTP_ETH_TYPE          = 16'hF788;
localparam PTP_DEL_REQ_TYPE      = 4'h1;

logic sop;
logic sop_r;
logic drop_pkt;
`ifdef SIMULATION
  logic [11:0] tready_cnt;
`else
  logic [15:0] tready_cnt;
`endif
logic w_arb_axis_tready;
always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    sop         <= '1;
    tready_cnt  <= '0;
    drop_pkt    <= '0;
    sop_r       <= '0;
  end
  else begin
    if (sop) begin
      sop         <= !(o_axis_tvalid && !o_axis_tlast && i_axis_tready);
    end
    else begin
      sop         <= (o_axis_tvalid && o_axis_tlast && i_axis_tready);
    end
    sop_r    <= sop;
    drop_pkt <= (tready_cnt == '1);
    if (!w_arb_axis_tready) begin  // If tready hasn't been seen for some duration
      if (tready_cnt == '1) begin
        tready_cnt  <= '1;
      end
      else begin
        tready_cnt  <= tready_cnt + 1;
      end
    end
    else begin
      tready_cnt <= '0;
    end
  end
end


assign arb_axis_tready = drop_pkt ? '1 : w_arb_axis_tready;

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    o_pkt_inc <= '0;
  end
  else begin
    o_pkt_inc <= (w_arb_axis_tready && arb_axis_tvalid && arb_axis_tlast);
  end
end

axis_reg # (
  .DWIDTH              ( W_DATA + W_KEEP + 1 + 1 + 3)
) u_axis_reg (
  .clk                ( i_pclk                                                                    ),
  .rst                ( i_prst                                                                    ),
  .i_axis_rx_tvalid   ( arb_axis_tvalid                                                           ),
  .i_axis_rx_tdata    ( {arb_axis_tdata,arb_axis_tlast,arb_axis_tuser,arb_axis_tkeep,arb_idx}     ),
  .o_axis_rx_tready   ( w_arb_axis_tready                                                         ),
  .o_axis_tx_tvalid   ( o_axis_tvalid                                                             ),
  .o_axis_tx_tdata    ( {o_axis_tdata,o_axis_tlast,o_axis_tuser,o_axis_tkeep,arb_pipe[0]  }       ),
  .i_axis_tx_tready   ( i_axis_tready                                                             )
);

logic is_ptp_del_req;
generate
  if (W_DATA == 64) begin
    assign is_ptp_del_req = (o_axis_tdata[32+:16] == PTP_ETH_TYPE) && (o_axis_tdata[48+:4] == PTP_DEL_REQ_TYPE);
    assign o_del_req_val = (arb_pipe[0] == 'd4) && o_axis_tvalid && i_axis_tready && sop_r && is_ptp_del_req;
  end
  else if (W_DATA >= 128) begin
    assign is_ptp_del_req = (o_axis_tdata[(12*8)+:16] == PTP_ETH_TYPE) && (o_axis_tdata[(14*8)+:4] == PTP_DEL_REQ_TYPE);
    assign o_del_req_val = (arb_pipe[0] == 'd4) && o_axis_tvalid && i_axis_tready && sop && is_ptp_del_req;
  end
  else begin
    assign is_ptp_del_req = 1'b0;
    assign o_del_req_val = 1'b0;
  end
endgenerate




`ifdef ASSERT_ON
  //Input AXIS Assertions
  logic [31:0] fv_axis_inp_byt_cnt       [N_INPT];
  logic [31:0] fv_axis_inp_byt_cnt_nxt   [N_INPT];
  generate
  for (genvar i = 0; i < N_INPT; i++) begin
    axis_checker #(
    .STBL_CHECK  (1),
    .NLST_BT_B2B (1),
    .MIN_PKTL_CHK (0),
    .MAX_PKTL_CHK (0),
    .AXI_TDATA   (W_DATA),
    .AXI_TUSER   (1),
`ifdef SIMULATION
    .SIMULATION(1),
`endif
    .PKT_MIN_LENGTH  (58),
    .PKT_MAX_LENGTH  (`HOST_MTU)
    ) assert_input_axis (
    .clk            (i_pclk),
    .rst            (i_prst),
    .axis_tvalid    (i_axis_tvalid[i]),
    .axis_tlast     (i_axis_tlast[i]),
    .axis_tkeep     (i_axis_tkeep[i]),
    .axis_tdata     (i_axis_tdata[i]),
    .axis_tuser     (i_axis_tuser[i]),
    .axis_tready    (o_axis_tready[i]),
    .byte_count     (fv_axis_inp_byt_cnt[i]),
    .byte_count_nxt (fv_axis_inp_byt_cnt_nxt[i])
    );
  end
  endgenerate

  //Output AXIS Assertions
  logic [31:0] fv_axis_out_byt_cnt;
  logic [31:0] fv_axis_out_byt_cnt_nxt;

  // Tlast must be zero if tvalid is low
  assert_output_axis_tlast_tvalid: assert property ( @(posedge i_pclk) disable iff (i_prst)
    (!o_axis_tlast || (o_axis_tvalid && o_axis_tlast)));

  axis_checker #(
  .STBL_CHECK  (1),
  .NLST_BT_B2B (1),
  .MIN_PKTL_CHK (0),
  .MAX_PKTL_CHK (0),
  .AXI_TDATA   (W_DATA),
  .AXI_TUSER   (1),
`ifdef SIMULATION
    .SIMULATION(1),
`endif
  .PKT_MIN_LENGTH  (58),
  .PKT_MAX_LENGTH  (`HOST_MTU)
  ) assert_output_axis (
  .clk            (i_pclk),
  .rst            (i_prst),
  .axis_tvalid    (o_axis_tvalid),
  .axis_tlast     (o_axis_tlast),
  .axis_tkeep     (o_axis_tkeep),
  .axis_tdata     (o_axis_tdata),
  .axis_tuser     (o_axis_tuser),
  .axis_tready    (i_axis_tready),
  .byte_count     (fv_axis_out_byt_cnt),
  .byte_count_nxt (fv_axis_out_byt_cnt_nxt)
  );
`endif


endmodule
