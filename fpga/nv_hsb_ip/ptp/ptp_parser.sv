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

module ptp_parser #(
  parameter PTP_INGRESS_WIDTH = 64*8
)(
  input           i_clk,
  input           i_rst,

  input           i_is_gPTP,
  input  [ 7:0]   i_ptp_rx_data [PTP_INGRESS_WIDTH/8],
  input           i_ptp_rx_vld,

  output          o_sync_msg_vld,
  output          o_follow_up_msg_vld,
  output          o_dly_resp_msg_vld,
  output          o_pdly_req_msg_vld,
  output          o_pdly_resp_msg_vld,
  output          o_pdly_resp_follow_up_msg_vld,

  output [15:0]   o_flag,
  output [47:0]   o_cf_ns,
  output [63:0]   o_src_clkid,
  output [15:0]   o_src_portid,
  output [15:0]   o_seq_id,
  output [79:0]   o_timestamp,
  output [63:0]   o_req_src_portidentity,
  output [15:0]   o_req_src_portid

);

//------------------------------------------------------------------------------------------------//
// PTP MESSAGE TYPES
//------------------------------------------------------------------------------------------------//

localparam MSG_SYNC                  = 4'h0;
localparam MSG_DELAY_REQ             = 4'h1;
localparam MSG_PDELAY_REQ            = 4'h2;
localparam MSG_PDELAY_RESP           = 4'h3;
localparam MSG_FOLLOW_UP             = 4'h8;
localparam MSG_DELAY_RESP            = 4'h9;
localparam MSG_PDELAY_RESP_FOLLOW_UP = 4'hA;

localparam VERSION_PTP               = 4'd2;

//------------------------------------------------------------------------------------------------//
// PTP HEADER
//------------------------------------------------------------------------------------------------//

logic [ 3:0] rx_majorsdoid;
logic [ 3:0] rx_msg_type;
logic [15:0] rx_msg_len;
logic [3:0]  rx_versionPTP;

assign rx_msg_type   = i_ptp_rx_data[0][3:0];
assign rx_majorsdoid = i_ptp_rx_data[0][7:4]; //MajorSdoId
assign rx_versionPTP = i_ptp_rx_data[1][3:0];
assign rx_msg_len    = {i_ptp_rx_data[2] ,i_ptp_rx_data[3]};
assign o_flag        = {i_ptp_rx_data[6] ,i_ptp_rx_data[7]};
assign o_cf_ns       = {i_ptp_rx_data[8] ,i_ptp_rx_data[9],i_ptp_rx_data[10],i_ptp_rx_data[11],i_ptp_rx_data[12],i_ptp_rx_data[13]};
assign o_src_clkid   = {i_ptp_rx_data[20],i_ptp_rx_data[21],i_ptp_rx_data[22],i_ptp_rx_data[23],
                        i_ptp_rx_data[24],i_ptp_rx_data[25],i_ptp_rx_data[26],i_ptp_rx_data[27]};
assign o_src_portid  = {i_ptp_rx_data[28],i_ptp_rx_data[29]};
assign o_seq_id      = {i_ptp_rx_data[30],i_ptp_rx_data[31]};

logic is_versionPTP;
logic is_profile;
logic is_sync_msg;
logic is_follow_up_msg;
logic is_dly_resp_msg;
logic is_ptp_rx_vld;
logic is_pdly_req_msg;
logic is_pdly_resp_msg;
logic is_pdly_resp_follow_up_msg;

assign is_versionPTP              = (rx_versionPTP == VERSION_PTP)               ? 1'b1 : 1'b0;
assign is_profile                 = i_is_gPTP ? (rx_majorsdoid == 4'h1)          ? 1'b1 : 1'b0:
                                                (rx_majorsdoid == 4'h0)          ? 1'b1 : 1'b0;
assign is_sync_msg                = (rx_msg_type   == MSG_SYNC)                  ? 1'b1 : 1'b0;
assign is_follow_up_msg           = (rx_msg_type   == MSG_FOLLOW_UP)             ? 1'b1 : 1'b0;
assign is_dly_resp_msg            = (rx_msg_type   == MSG_DELAY_RESP)            ? 1'b1 : 1'b0;
assign is_pdly_req_msg            = (rx_msg_type   == MSG_PDELAY_REQ)            ? 1'b1 : 1'b0;
assign is_pdly_resp_msg           = (rx_msg_type   == MSG_PDELAY_RESP)           ? 1'b1 : 1'b0;
assign is_pdly_resp_follow_up_msg = (rx_msg_type   == MSG_PDELAY_RESP_FOLLOW_UP) ? 1'b1 : 1'b0;
assign is_ptp_rx_vld              = i_ptp_rx_vld && is_versionPTP && is_profile;

assign o_sync_msg_vld                = is_ptp_rx_vld && is_sync_msg;
assign o_follow_up_msg_vld           = is_ptp_rx_vld && is_follow_up_msg;
assign o_dly_resp_msg_vld            = is_ptp_rx_vld && is_dly_resp_msg;
assign o_pdly_req_msg_vld            = is_ptp_rx_vld && is_pdly_req_msg;
assign o_pdly_resp_msg_vld           = is_ptp_rx_vld && is_pdly_resp_msg;
assign o_pdly_resp_follow_up_msg_vld = is_ptp_rx_vld && is_pdly_resp_follow_up_msg;

//------------------------------------------------------------------------------------------------//
// PTP CONTENT
//------------------------------------------------------------------------------------------------//

assign o_timestamp            = {i_ptp_rx_data[34],i_ptp_rx_data[35],i_ptp_rx_data[36],i_ptp_rx_data[37],i_ptp_rx_data[38],
                                  i_ptp_rx_data[39],i_ptp_rx_data[40],i_ptp_rx_data[41],i_ptp_rx_data[42],i_ptp_rx_data[43]};
assign o_req_src_portidentity = {i_ptp_rx_data[44],i_ptp_rx_data[45],i_ptp_rx_data[46],i_ptp_rx_data[47],i_ptp_rx_data[48],
                                  i_ptp_rx_data[49],i_ptp_rx_data[50],i_ptp_rx_data[51]};
assign o_req_src_portid       = {i_ptp_rx_data[52], i_ptp_rx_data[53]};

endmodule
