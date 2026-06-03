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

module ptp_dly
  import ptp_pkg::*;
#(
  parameter  PTP_EGRESS_WIDTH = 64*8,
  parameter  NUM_HOST         = 2,
  parameter  W_NUM_HOST       = NUM_HOST>1 ? $clog2(NUM_HOST) : 1,
  localparam PTP_TX_WIDTH     = $clog2(PTP_EGRESS_WIDTH/8)
)(
  input                           i_pclk,
  input                           i_prst,

  input                           i_hif_clk,
  input                           i_hif_rst,

  input  [47:0]                   i_dev_mac_addr,
  output [PTP_EGRESS_WIDTH-1:0]   o_ptp_tx_data,
  output                          o_ptp_tx_vld,
  output [PTP_TX_WIDTH-1:0]       o_ptp_tx_len,

  input                           i_is_1588_e2e,
  input                           i_is_1588_p2p,
  input                           i_is_gPTP,
  input                           i_dly_resp_msg_vld,
  input                           i_pdly_req_msg_vld,
  input                           i_pdly_resp_msg_vld,
  input                           i_pdly_resp_follow_up_msg_vld,

  input                           i_two_step,
  input  [47:0]                   i_cf_ns,
  input  [79:0]                   i_timestamp,
  input  [15:0]                   i_seq_id,
  input  [63:0]                   i_src_clkid,
  input  [15:0]                   i_src_portid,
  input  [63:0]                   i_req_src_portidentity,
  input  [15:0]                   i_req_src_portid,

  input                           i_dly_req,
  input                           i_pps,
  input  [47:0]                   i_sec,
  input  [31:0]                   i_nano_sec,
  input                           i_tx_ts_vld,

  output [79:0]                   o_dly_t1_ts,
  output [79:0]                   o_dly_t2_ts,
  output [79:0]                   o_dly_t3_ts,
  output [79:0]                   o_dly_t4_ts,
  output [47:0]                   o_dly_cf_ns,
  output                          o_dly_ld
);

logic [$clog2(PTP_EGRESS_WIDTH/8)-1:0] ptp_tx_len;
logic [$clog2(PTP_EGRESS_WIDTH/8)-1:0] ptp_tx_len_sync;

//------------------------------------------------------------------------------------------------//
// TX ETHERNET L2 LAYER
//------------------------------------------------------------------------------------------------//
logic      [ETH_L2_HDR_WIDTH-1:0]   eth_l2_header;
logic      [PTP_EGRESS_WIDTH-1:0]   eth_tx_data;
logic      [PTP_EGRESS_WIDTH-1:0]   eth_tx_data_be;

logic      [47:0]                   eth_mac_addr;
logic      [47:0]                   eth_mac_addr_sync;

assign eth_l2_header = {eth_mac_addr_sync, i_dev_mac_addr, PTP_ETH_TYPE};

//------------------------------------------------------------------------------------------------//
// PTP TX VECTOR
//------------------------------------------------------------------------------------------------//
logic [PTP_EGRESS_WIDTH-ETH_L2_HDR_WIDTH-1:0] ptp_tx_data;

logic        pdly_req_tx_vld;
logic        pdly_req_tx_vld_sync;
logic        pdly_req_tx_vld_sync_ff0;
logic        pdly_req_tx_vld_sync_ff1;
logic        pdly_req_tx_vld_sync_posedge;
logic [15:0] pdly_req_tx_seqId;

enum logic [2:0] {
  WAIT_FOR_PDLY_REQ_EN,
  TX_PDLY_REQ,
  WAIT_FOR_TX_PDLY_REQ_STAMP,
  WAIT_FOR_RX_PDLY_RESP,
  WAIT_FOR_RX_PDLY_RESP_FOLLOW_UP,
  INC_PDLY_REQ_TX_SEQID
} pdly_req_state;

enum logic [1:0] {
  WAIT_FOR_PDLY_RESP,
  WAIT_FOR_PDLY_RESP_FOLLOW_UP
} pdly_resp_state;

logic [3:0]  msg_type;
logic        majorSdoId;
logic [7:0]  minorSdoId;
logic [3:0]  versionPTP;
logic [15:0] messageLength;
logic [7:0]  domainNumber;
logic        twoStep;
logic [15:0] flagField;
logic [63:0] correctionField;
logic [63:0] clockIdentity;
logic [15:0] sourcePortID;
logic [15:0] sequenceId;
logic [7:0]  controlField;
logic [7:0]  logMessageInterval;
logic [79:0] timestamp;
logic [79:0] req_src_portid;

logic        majorSdoId_sync;
logic        twoStep_sync;
logic [3:0]  msg_type_sync;
logic [15:0] messageLength_sync;
logic [15:0] sequenceId_sync;
logic [7:0]  controlField_sync;
logic [7:0]  logMessageInterval_sync;
logic [79:0] timestamp_sync;
logic [79:0] req_src_portid_sync;

assign majorSdoId         = i_is_gPTP ? 1'h1 : 1'h0;
assign versionPTP         = 4'h2;
assign domainNumber       = 8'h0;
assign minorSdoId         = 8'h0;
assign flagField          = {6'd0, twoStep_sync, 9'd0};
assign correctionField    = 64'h0;
assign clockIdentity      = {i_dev_mac_addr[47:24], 16'hFFFE, i_dev_mac_addr[23:0]};
assign sourcePortID       = 16'd1;

assign ptp_tx_data = {
  // Bytes 53-50
  3'h0, majorSdoId_sync, msg_type_sync, 4'h1, versionPTP, messageLength_sync,
  // Bytes 49-46
  domainNumber, minorSdoId, flagField,
  // Bytes 45-48
  correctionField,
  // Bytes 37-34
  32'd0,
  // Bytes 33-24
  clockIdentity, sourcePortID,
  // Bytes 23-20
  sequenceId_sync, controlField_sync, logMessageInterval_sync,
  // Bytes 19-10
  timestamp_sync,
  // Bytes 9-0
  req_src_portid_sync
};

reg_cdc #(
  .NBITS(52+80+80+$clog2(PTP_EGRESS_WIDTH/8) +1+1+48)
) u_ptp_egress_data_cdc (
  .i_a_clk( i_pclk                                                                                      ),
  .i_a_rst( i_prst                                                                                      ),
  .i_a_val( pdly_req_tx_vld                                                                             ),
  .i_a_reg( {msg_type,messageLength,sequenceId,controlField,logMessageInterval,timestamp,
              req_src_portid,ptp_tx_len, majorSdoId, twoStep, eth_mac_addr}),
  .i_b_clk( i_hif_clk                                                                                   ),
  .i_b_rst( i_hif_rst                                                                                   ),
  .o_b_val( pdly_req_tx_vld_sync                                                                        ),
  .o_b_reg( {msg_type_sync,messageLength_sync,sequenceId_sync,controlField_sync,logMessageInterval_sync,timestamp_sync,
              req_src_portid_sync,ptp_tx_len_sync, majorSdoId_sync, twoStep_sync, eth_mac_addr_sync})
);

always_ff @(posedge i_hif_clk) begin
  if (i_hif_rst) begin
    pdly_req_tx_vld_sync_ff0 <= 1'b0;
    pdly_req_tx_vld_sync_ff1 <= 1'b0;
  end
  else begin
    pdly_req_tx_vld_sync_ff0 <= pdly_req_tx_vld_sync;
    pdly_req_tx_vld_sync_ff1 <= pdly_req_tx_vld_sync_ff0;
  end
end
assign pdly_req_tx_vld_sync_posedge = ~pdly_req_tx_vld_sync_ff1 & pdly_req_tx_vld_sync_ff0;

assign eth_tx_data = {eth_l2_header, ptp_tx_data};
genvar j;
generate
  for (j=0; j<PTP_EGRESS_WIDTH/8; j++) begin
    assign eth_tx_data_be[j*8+:8] = eth_tx_data[(PTP_EGRESS_WIDTH/8-1-j)*8+:8];
  end
endgenerate

assign o_ptp_tx_data = eth_tx_data_be;
assign o_ptp_tx_vld  = pdly_req_tx_vld_sync_posedge;

//------------------------------------------------------------------------------------------------//
// PEER DELAY REQUEST STATE MACHINE
//------------------------------------------------------------------------------------------------//

logic [79:0] pdly_t1_ts;
logic [79:0] pdly_t2_ts;
logic [79:0] pdly_t3_ts;
logic [79:0] pdly_t4_ts;
logic [47:0] dly_cf_ns;
logic        pdly_ts_ld;
logic [79:0] pdly_req_rx_ts;
logic [15:0] pdly_req_rx_seqId;
logic [79:0] pdly_req_rx_src_portid;
logic        pdly_req_rx;
logic        pdly_req_tx;
logic        dly_req_tx;
logic        dly_req_tx_ff;
logic        dly_req_tx_posedge;
logic        is_src_resp_match;
logic        first_dly_req_tx;

assign is_src_resp_match = (pdly_req_tx_seqId==i_seq_id) && (sourcePortID==i_req_src_portid) && (clockIdentity==i_req_src_portidentity);

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    pdly_req_tx <= '0;
  end
  else begin
    if (i_pps && (i_is_gPTP||i_is_1588_p2p)) begin
      pdly_req_tx <= 1'b1;
    end
    else if (pdly_req_tx && msg_type == MSG_PEER_DELAY_REQ) begin
      pdly_req_tx <= 1'b0;
    end
  end
end

assign dly_req_tx = (i_is_gPTP||i_is_1588_p2p) ? pdly_req_tx : i_dly_req;
assign dly_req_tx_posedge = dly_req_tx && !dly_req_tx_ff;


always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    dly_req_tx_ff     <= 1'b0;
    pdly_req_tx_seqId <= 1'b0;
    first_dly_req_tx  <= 1'b0;
  end
  else begin
    dly_req_tx_ff <= dly_req_tx;

    if (dly_req_tx) begin
      first_dly_req_tx <= 1'b1;
    end

    if (dly_req_tx_posedge && first_dly_req_tx) begin
      pdly_req_tx_seqId <= pdly_req_tx_seqId + 1'b1;
    end
  end
end

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    pdly_req_rx_ts         <= '0;
    pdly_req_rx_seqId      <= '0;
    pdly_req_rx_src_portid <= '0;
    pdly_req_rx            <= 1'b0;
  end
  else begin
    if (i_pdly_req_msg_vld && (i_is_gPTP||i_is_1588_p2p)) begin
      pdly_req_rx_ts         <= {i_sec, i_nano_sec};
      pdly_req_rx_seqId      <= i_seq_id;
      pdly_req_rx_src_portid <= {i_src_clkid, i_src_portid};
      pdly_req_rx            <= 1'b1;
    end
    else if (pdly_req_rx && msg_type == MSG_PEER_DELAY_RESP) begin
      pdly_req_rx            <= 1'b0;
    end
  end
end

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    pdly_req_tx_vld     <= 1'b0;
    pdly_t1_ts          <= '0;
    eth_mac_addr        <= '0;
    msg_type            <= '0;
    messageLength       <= '0;
    timestamp           <= '0;
    twoStep             <= 1'b0;
    sequenceId          <= '0;
    req_src_portid      <= '0;
    logMessageInterval  <= '0;
    ptp_tx_len          <= '0;
    pdly_req_state      <= WAIT_FOR_PDLY_REQ_EN;
    controlField        <= '0;
  end
  else begin
  case (pdly_req_state)
    WAIT_FOR_PDLY_REQ_EN: begin
      pdly_req_tx_vld    <= 1'b0;
      timestamp          <= '0;
      twoStep            <= 1'b0;
      req_src_portid     <= '0;
      logMessageInterval <= 8'h7F;
      msg_type           <= '0;
      if (i_is_gPTP || i_is_1588_p2p) begin
        eth_mac_addr       <= PTP_NON_FW_MULTI_ADDR;
      end
      else begin
        eth_mac_addr       <= PTP_FW_MULTI_ADDR;
      end
      if (i_is_gPTP || i_is_1588_p2p) begin
        controlField       <= 8'h05;
        messageLength      <= 8'd54;
        ptp_tx_len         <= 'd68;
      end
      else begin
        controlField       <= 8'h01;
        messageLength      <= 8'd44;
        ptp_tx_len         <= 'd60;
      end
      if (dly_req_tx_ff) begin
        sequenceId     <= pdly_req_tx_seqId;
        if (i_is_gPTP || i_is_1588_p2p) begin
          logMessageInterval <= 8'h00;
          msg_type           <= MSG_PEER_DELAY_REQ;
        end
        else begin
          msg_type           <= MSG_DELAY_REQ;
        end
        pdly_req_state <= TX_PDLY_REQ;
      end
      else if (pdly_req_rx) begin
        sequenceId     <= pdly_req_rx_seqId;
        twoStep        <= 1'b1;
        timestamp      <= pdly_req_rx_ts;
        req_src_portid <= pdly_req_rx_src_portid;
        msg_type       <= MSG_PEER_DELAY_RESP;
        pdly_req_state <= TX_PDLY_REQ;
      end
    end
    TX_PDLY_REQ: begin
      pdly_req_tx_vld  <= 1'b1;
      pdly_req_state   <= WAIT_FOR_TX_PDLY_REQ_STAMP;
    end
    WAIT_FOR_TX_PDLY_REQ_STAMP: begin
      pdly_req_tx_vld <= 1'b0;
      if (i_tx_ts_vld) begin
        msg_type         <= 'd0;
        if ((i_is_gPTP || i_is_1588_p2p) && msg_type == MSG_PEER_DELAY_RESP) begin
          timestamp      <= {i_sec, i_nano_sec};
          twoStep        <= 1'b0;
          msg_type       <= MSG_PEER_DELAY_RESP_FOLLOW_UP;
          pdly_req_state <= TX_PDLY_REQ;
        end
        else if ((i_is_gPTP || i_is_1588_p2p) && msg_type == MSG_PEER_DELAY_RESP_FOLLOW_UP) begin
          pdly_req_state <= WAIT_FOR_PDLY_REQ_EN;
        end
        else begin
          pdly_t1_ts     <= {i_sec, i_nano_sec};
          pdly_req_state <= WAIT_FOR_PDLY_REQ_EN;
        end
      end else if (dly_req_tx && msg_type == MSG_DELAY_REQ) begin
        msg_type         <= 'd0;
        pdly_req_state   <= WAIT_FOR_PDLY_REQ_EN;
      end else if (i_pdly_req_msg_vld && (msg_type==MSG_PEER_DELAY_RESP || msg_type==MSG_PEER_DELAY_RESP_FOLLOW_UP)) begin
        msg_type         <= 'd0;
        pdly_req_state   <= WAIT_FOR_PDLY_REQ_EN;
      end
    end
    default: begin
      pdly_req_state <= WAIT_FOR_PDLY_REQ_EN;
    end
    endcase
  end
end

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    pdly_resp_state   <= WAIT_FOR_PDLY_RESP;
    dly_cf_ns         <= '0;
    pdly_t2_ts        <= '0;
    pdly_t3_ts        <= '0;
    pdly_t4_ts        <= '0;
    pdly_ts_ld        <= 1'b0;
  end
  else begin
    case (pdly_resp_state)
      WAIT_FOR_PDLY_RESP: begin
        pdly_ts_ld <= 1'b0;
        if (dly_req_tx_posedge) begin
          pdly_resp_state <= WAIT_FOR_PDLY_RESP;
        end
        else if (i_dly_resp_msg_vld || i_pdly_resp_msg_vld) begin
          if (is_src_resp_match) begin
            pdly_t2_ts     <= i_timestamp;
            dly_cf_ns      <= i_cf_ns;
            if (i_two_step) begin
              pdly_t4_ts      <= {i_sec, i_nano_sec};
              pdly_resp_state <= WAIT_FOR_PDLY_RESP_FOLLOW_UP;
            end
            else begin
              pdly_ts_ld        <= 1'b1;
              dly_cf_ns         <= i_cf_ns;
              pdly_resp_state   <= WAIT_FOR_PDLY_RESP;
            end
          end
        end
      end
      WAIT_FOR_PDLY_RESP_FOLLOW_UP: begin
        if (dly_req_tx_posedge) begin
          pdly_resp_state <= WAIT_FOR_PDLY_RESP;
        end
        else if (i_pdly_resp_follow_up_msg_vld) begin
          if (is_src_resp_match) begin
            pdly_ts_ld        <= 1'b1;
            pdly_t3_ts        <= i_timestamp;
            pdly_resp_state   <= WAIT_FOR_PDLY_RESP;
          end
        end
      end
    endcase
  end
end

assign o_dly_t1_ts  = pdly_t1_ts;
assign o_dly_t2_ts  = pdly_t2_ts;
assign o_dly_t3_ts  = pdly_t3_ts;
assign o_dly_t4_ts  = pdly_t4_ts;
assign o_dly_cf_ns  = dly_cf_ns;
assign o_dly_ld     = pdly_ts_ld;
assign o_ptp_tx_len = ptp_tx_len_sync;

endmodule
