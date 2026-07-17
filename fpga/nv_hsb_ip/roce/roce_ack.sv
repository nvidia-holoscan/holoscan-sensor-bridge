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

module roce_ack
  import rx_parser_pkg::*;
  import axis_pkg::*;
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter  W_DATA          = 64,
  parameter  W_USER          = 1,
  parameter  HDR_BYTES       = 54,
  parameter  META_FIFO_DEPTH = 16,
  parameter  N_QP            = 32,
  parameter  FIFO_DEPTH      = 512,
  localparam W_KEEP          = W_DATA/8,
  localparam W_QP            = $clog2(N_QP)+1
)(
  input                   i_pclk,
  input                   i_prst,
  input                   i_aclk,
  input                   i_arst,

  // Packet Valid Interface
  output                  o_empty,
  output                  o_is_err,
  input                   i_rd_en,
  input                   i_pkt_done,
  input                   i_pkt_is_err,

  // Header structs
  input                   i_hdr_valid,
  input                   i_is_stx,
  input   eth_hdr         i_eth_hdr,
  input   ip_hdr          i_ip_hdr,
  input   udp_hdr         i_udp_hdr,
  input   bth_hdr         i_bth_hdr,
  input  [W_USER-1:0]     i_vlan_tuser,

  // Sensor TX Backpressure
  input   [N_QP-1:0]      i_stx_ready,

  // Registers
  input   [31:0]          i_ctrl_reg,

  //Register APB Interfaces
  input   apb_m2s         i_apb_m2s,
  output  apb_s2m         o_apb_s2m,

  // AXIS Interface
  output                  o_axis_tvalid,
  output [W_DATA-1:0]     o_axis_tdata,
  output                  o_axis_tlast,
  output [W_USER-1:0]     o_axis_tuser,
  output [W_KEEP-1:0]     o_axis_tkeep,
  input                   i_axis_tready
);

localparam ETH_BYTES             = 14;
localparam IP_BYTES              = 20;
localparam UDP_BYTES             = 8;
localparam BTH_BYTES             = 12;
localparam AETH_BYTES            = 4;
localparam ICRC_BYTES            = 4;

localparam RESP_BYTES = ETH_BYTES + IP_BYTES + UDP_BYTES + BTH_BYTES + AETH_BYTES;

localparam W_RAM_ADDR         = $clog2(N_QP)+1;

localparam BTH_SEND_FIRST   = 4'h0;
localparam BTH_SEND_MIDDLE  = 4'h1;
localparam BTH_SEND_LAST    = 4'h2;
localparam BTH_SEND_ONLY    = 4'h4;

localparam BTH_WRITE_FIRST  = 4'h6;
localparam BTH_WRITE_MIDDLE = 4'h7;
localparam BTH_WRITE_LAST   = 4'h8;
localparam BTH_WRITE_ONLY   = 4'hA;

localparam BTH_UC     = 4'h2;
localparam BTH_RC     = 4'h0;


//------------------------------------------------------------------------------------------------//
// header Buffer
//------------------------------------------------------------------------------------------------//

eth_hdr  eth_hdr_r;
ip_hdr   ip_hdr_r;
udp_hdr  udp_hdr_r;
bth_hdr  bth_hdr_r;
logic [W_USER-1:0] vlan_tuser_r;


always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    eth_hdr_r <= '{default:0};
    ip_hdr_r  <= '{default:0};
    udp_hdr_r <= '{default:0};
    bth_hdr_r <= '{default:0};
    vlan_tuser_r <= '0;
  end
  else begin
    if (i_hdr_valid) begin
      eth_hdr_r <= i_eth_hdr;
      ip_hdr_r  <= i_ip_hdr;
      udp_hdr_r <= i_udp_hdr;
      bth_hdr_r <= i_bth_hdr;
      vlan_tuser_r <= i_vlan_tuser;
    end
  end
end


//------------------------------------------------------------------------------------------------//
// Registers
//------------------------------------------------------------------------------------------------//

logic ack_enable;
logic inj_err_en;
logic inj_err_rnr_en;
logic psn_check_enable;
logic ack_first;
logic enable_rnr;

assign ack_enable        = i_ctrl_reg[0];
assign inj_err_en        = i_ctrl_reg[1];
assign psn_check_enable  = i_ctrl_reg[2];
assign ack_first         = i_ctrl_reg[3];
assign inj_err_rnr_en    = i_ctrl_reg[4];
assign enable_rnr        = i_ctrl_reg[5];

//------------------------------------------------------------------------------------------------//
// PSN RAM
//------------------------------------------------------------------------------------------------//

localparam PSN_RAM_SIZE = 24 + 24 + 1 + 1;

logic [PSN_RAM_SIZE-1:0] sts_ram_data;
logic [W_RAM_ADDR-1:0]   sts_ram_addr;
logic [W_RAM_ADDR-1:0]   sts_ram_wraddr;
logic [PSN_RAM_SIZE-1:0] sts_ram_wrdata;
logic                    sts_ram_wren = '0;
logic                    sts_ram_rden;

(* ram_style = "block" *) logic [PSN_RAM_SIZE-1:0] sts_ram [N_QP] = '{default:'0}/*synthesis syn_ramstyle = "block_ram"*/;

always @ (posedge i_pclk) begin
  if (sts_ram_wren) begin
    sts_ram[sts_ram_addr] <= sts_ram_wrdata;
  end
  if (sts_ram_rden) begin
    sts_ram_data <= sts_ram[sts_ram_addr];
  end
end




//------------------------------------------------------------------------------------------------//
// PSN Pipeline
//------------------------------------------------------------------------------------------------//

logic [23:0] psn_exp;
logic [23:0] msn;
logic [23:0] psn_next;
logic [23:0] msn_next;

logic [W_QP-1:0] sif_idx;

logic resp_trigger;
logic ack;
logic rnr_nak;

logic is_last_pkt;
logic is_first_pkt;
logic pkt_err;
logic pkt_err_wr;
logic inj_err;
logic is_rc;
logic ack_req;
logic has_nacked;
logic is_nacked;
logic has_inited; 


logic              pending_valid;
logic              pending_is_first;
logic              pending_is_last;
logic [W_QP-1:0]   pending_sif_idx;
logic              pending_is_stx;
logic              pending_is_stx_skid;
logic              pending_is_stx_skid_valid;
logic              pkt_done_d1;
logic              pkt_is_err_d1;
logic              pkt_done_valid;
logic              pkt_done_is_err;



always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    pending_valid    <= '0;
    pending_is_first <= '0;
    pending_is_last  <= '0;
    pending_sif_idx  <= '0;
    pending_is_stx_skid       <= '0;
    pending_is_stx_skid_valid <= '0;
    pkt_done_d1      <= '0;
    pkt_is_err_d1    <= '0;
  end
  else begin
    pkt_done_d1   <= i_pkt_done;
    pkt_is_err_d1 <= i_pkt_is_err;

    if (pending_valid && !pkt_done_valid) begin
      pending_valid    <= pending_valid;
      pending_is_first <= pending_is_first;
      pending_is_last  <= pending_is_last;
      pending_sif_idx  <= pending_sif_idx;
      if (!pending_is_stx_skid_valid) begin
        pending_is_stx_skid       <= i_is_stx && (i_bth_hdr.opcode[7:4] == BTH_RC);
        pending_is_stx_skid_valid <= '1;
      end
    end
    else begin
      pending_valid    <= i_hdr_valid;
      pending_is_first <= is_first_pkt;
      pending_is_last  <= is_last_pkt;
      pending_sif_idx  <= sif_idx;
      pending_is_stx_skid       <= '0;
      pending_is_stx_skid_valid <= '0;
    end
  end
end


always @(posedge i_pclk) begin
  if (i_prst) begin
    msn_next     <= '0;
    psn_next     <= '0;
    sts_ram_wraddr <= '0;
    resp_trigger <= '0;
    ack          <= '0;
    rnr_nak      <= '0;
    pkt_err      <= '0;
    sts_ram_wren <= '0;
    pkt_err_wr   <= '0;
  end
  else begin
    if (pending_valid) begin
      if (!pending_is_stx) begin
        msn_next     <= msn;
        psn_next     <= psn_exp;
        resp_trigger <= '0;
        ack          <= '0;
        rnr_nak      <= '0;
        pkt_err      <= '0;
        sts_ram_wren <= '0;
        pkt_err_wr   <= pkt_done_valid;
      end
      else if (!has_inited) begin // Initialize the PSN RAM
        msn_next       <= pending_is_last;
        psn_next       <= bth_hdr_r.psn + 1;
        sts_ram_wraddr <= pending_sif_idx[W_RAM_ADDR-1:0];
        resp_trigger   <= pkt_done_valid && !pkt_done_is_err && ack_req;
        ack            <= '1;
        rnr_nak        <= '0;
        pkt_err        <= '0;
        sts_ram_wren   <= pkt_done_valid && !pkt_done_is_err;
        pkt_err_wr     <= pkt_done_valid;
      end
      else if (inj_err) begin // Error Injection
        msn_next       <= msn;
        psn_next       <= psn_exp;
        sts_ram_wraddr <= pending_sif_idx[W_RAM_ADDR-1:0];
        resp_trigger   <= pkt_done_valid && !pkt_done_is_err && !has_nacked;
        ack            <= '0;
        rnr_nak        <= inj_err_rnr_en;
        pkt_err        <= '1;
        sts_ram_wren   <= pkt_done_valid && !pkt_done_is_err;
        pkt_err_wr     <= pkt_done_valid;
      end
      else if (!i_stx_ready[pending_sif_idx] && enable_rnr) begin
        msn_next       <= msn;
        psn_next       <= psn_exp;
        sts_ram_wraddr <= pending_sif_idx[W_RAM_ADDR-1:0];
        resp_trigger   <= pkt_done_valid && !pkt_done_is_err && !has_nacked;
        ack            <= '0;
        rnr_nak        <= '1;
        pkt_err        <= '1;
        sts_ram_wren   <= pkt_done_valid && !pkt_done_is_err;
        pkt_err_wr     <= pkt_done_valid;
      end
      else if (bth_hdr_r.psn == psn_exp) begin // Ack the packet if is next in sequence
        msn_next       <= msn + pending_is_last;
        psn_next       <= psn_exp + 1;
        sts_ram_wraddr <= pending_sif_idx[W_RAM_ADDR-1:0];
        resp_trigger   <= pkt_done_valid && !pkt_done_is_err && ack_req;
        ack            <= '1;
        rnr_nak        <= '0;
        pkt_err        <= '0;
        sts_ram_wren   <= pkt_done_valid && !pkt_done_is_err;
        pkt_err_wr     <= pkt_done_valid;
      end
      else if (bth_hdr_r.psn > psn_exp) begin // Drop the packet if out of order
        msn_next       <= msn;
        psn_next       <= psn_exp;
        sts_ram_wraddr <= pending_sif_idx[W_RAM_ADDR-1:0];
        resp_trigger   <= pkt_done_valid && !pkt_done_is_err && !has_nacked;
        ack            <= '0;
        rnr_nak        <= '0;
        pkt_err        <= '1;
        sts_ram_wren   <= pkt_done_valid && !pkt_done_is_err;
        pkt_err_wr     <= pkt_done_valid;
      end
      else begin  // Drop the packet if duplicate
        msn_next       <= msn;
        psn_next       <= psn_exp;
        sts_ram_wraddr <= pending_sif_idx[W_RAM_ADDR-1:0];
        resp_trigger   <= '0;
        ack            <= '0;
        rnr_nak        <= '0;
        pkt_err        <= '1;
        sts_ram_wren   <= pkt_done_valid && !pkt_done_is_err;
        pkt_err_wr     <= pkt_done_valid;
      end
    end
    else begin
      sts_ram_wren <= '0;
      pkt_err_wr   <= '0;
      msn_next     <= msn;
      psn_next     <= psn_exp;
      resp_trigger <= '0;
      ack          <= '0;
      rnr_nak      <= '0;
      pkt_err      <= '0;
    end
  end
end

assign sif_idx        = (i_bth_hdr.dest_qp[7:1] - 1);
assign sts_ram_addr   = sts_ram_wren ? sts_ram_wraddr : sif_idx[W_RAM_ADDR-1:0];
assign sts_ram_rden   = i_hdr_valid;
assign sts_ram_wrdata = {'1, is_nacked, msn_next, psn_next};
assign pending_is_stx = pending_is_stx_skid_valid ? pending_is_stx_skid : (i_is_stx && (i_bth_hdr.opcode[7:4] == BTH_RC));
assign pkt_done_valid = pkt_done_d1 || i_pkt_done;
assign pkt_done_is_err = pkt_done_d1 ? pkt_is_err_d1 : i_pkt_is_err;

assign ack_req = (bth_hdr_r.a_rsv7[7]);
assign psn_exp = sts_ram_data[23:0];

assign is_last_pkt  = (i_bth_hdr.opcode[3:0] inside {BTH_SEND_LAST, BTH_SEND_ONLY, BTH_WRITE_LAST, BTH_WRITE_ONLY});
assign is_first_pkt = (i_bth_hdr.opcode[3:0] inside {BTH_SEND_FIRST, BTH_SEND_ONLY, BTH_WRITE_FIRST, BTH_WRITE_ONLY});
assign is_rc        = (bth_hdr_r.opcode[7:4] == BTH_RC);
assign msn          = sts_ram_data[47:24];
assign has_nacked   = sts_ram_data[48]; 
assign is_nacked    = (ack) ? '0 : (resp_trigger && !ack) || has_nacked;
assign has_inited   = sts_ram_data[49];

//------------------------------------------------------------------------------------------------//
// Error injection
//------------------------------------------------------------------------------------------------//

logic [7:0] err_cnt;

// Inject an error if the bit is enabled. Inject on the 8th packet of each sequence.
always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    inj_err <= '0;
    err_cnt <= '0;
  end
  else begin
    if (i_hdr_valid && i_is_stx && (i_bth_hdr.opcode[7:4] == BTH_RC) && inj_err_en) begin
      if (is_first_pkt) begin
        inj_err <= '0;
        err_cnt <= '0;
      end
      else begin
        inj_err <= (err_cnt == '1);
        err_cnt <= err_cnt + 1;
      end
    end
  end
end

//------------------------------------------------------------------------------------------------//
// Packet Error FIFO
//------------------------------------------------------------------------------------------------//

logic is_err;

sc_fifo #(
  .DATA_WIDTH ( 1                ),
  .FIFO_DEPTH ( META_FIFO_DEPTH  ),
  .MEM_STYLE  ( "LUT"            )
) err_fifo (
  .clk        ( i_pclk           ),
  .rst        ( i_prst           ),
  .wr         ( pkt_err_wr       ),
  .din        ( pkt_err          ),
  .full       (                  ),
  .afull      (                  ),
  .over       (                  ),
  .rd         ( i_rd_en          ),
  .dout       ( is_err           ),
  .dval       (                  ),
  .empty      ( o_empty          ),
  .aempty     (                  ),
  .under      (                  ),
  .count      (                  )
);

assign o_is_err = is_err & psn_check_enable;

//------------------------------------------------------------------------------------------------//
// Response AXIS To Vector
//------------------------------------------------------------------------------------------------//

logic resp_valid;

logic [RESP_BYTES*8-1:0] resp_data;
logic [RESP_BYTES*8-1:0] resp_data_be;

logic resp_busy;

logic              resp_tvalid;
logic              resp_tlast;
logic [W_USER-1:0] resp_tuser;
logic [W_KEEP-1:0] resp_tkeep;
logic [W_DATA-1:0] resp_tdata;
logic              resp_tready;

vec_to_axis #(
  .AXI_DWIDTH       ( W_DATA               ),
  .W_USER           ( W_USER               ),
  .DATA_WIDTH       ( RESP_BYTES*8         )
) roce_ack_to_axis (
  .clk              ( i_pclk               ),
  .rst              ( i_prst               ),
  .trigger          ( resp_valid           ),
  .data             ( resp_data_be         ),
  .tuser            ( vlan_tuser_r         ),
  .is_busy          ( resp_busy            ),
  .o_axis_tx_tvalid ( resp_tvalid          ),
  .o_axis_tx_tdata  ( resp_tdata           ),
  .o_axis_tx_tlast  ( resp_tlast           ),
  .o_axis_tx_tuser  ( resp_tuser           ),
  .o_axis_tx_tkeep  ( resp_tkeep           ),
  .i_axis_tx_tready ( resp_tready          )
);

assign resp_valid = resp_trigger & ack_enable;

//------------------------------------------------------------------------------------------------//
// Response Data
//------------------------------------------------------------------------------------------------//

logic [15:0] udp_len;
logic [7:0]  rsp_opcode;
logic [7:0]  ack_syndrome;
logic [4:0]  rnr_timer;
logic [23:0] host_qp;

logic [15:0] ipv4_chksum;
logic [15:0] ipv4_len;

assign ipv4_len      = udp_len + 20;

assign resp_data = {
  // Ethernet Header 14 Bytes
  eth_hdr_r.src_mac,eth_hdr_r.dest_mac,  // Swap src and dest mac addresses
  eth_hdr_r.ethertype,
  // IPv4 Header 20 Bytes
  ip_hdr_r.ver_ihl, ip_hdr_r.dscp_ecn, ipv4_len,
  ip_hdr_r.id, ip_hdr_r.flags_frag,
  ip_hdr_r.ttl, ip_hdr_r.protocol, ipv4_chksum,
  ip_hdr_r.dest_addr,// Swap src and dest addresses
  ip_hdr_r.src_addr, 
  // UDP Header 8 bytes
  udp_hdr_r.src_port, udp_hdr_r.dest_port,  // Don't swap src and dest ports for 0x12b7 RoCE udp
  udp_len, 16'h0,
  // BTH Header 12 Bytes
  rsp_opcode,bth_hdr_r.s_m_pad_tver,bth_hdr_r.pkey,8'h0,host_qp,
  8'h0,psn_exp,
  // AETH Header 4 Bytes
  ack_syndrome, msn_next
};

assign udp_len           = BTH_BYTES + AETH_BYTES + ICRC_BYTES + UDP_BYTES;
assign rsp_opcode        = 8'h11;
assign ack_syndrome[7:5] = ack ? 3'h0 : rnr_nak ? 3'h1 : 3'h3; // ACK or NAK
assign ack_syndrome[4:0] = ack ? '1 : (rnr_nak ? rnr_timer : 5'h00); // Credit Disable ACK, RNR timer, or NAK code 0x00

genvar k;
generate
  for (k=0; k<RESP_BYTES; k++) begin : gen_resp_data_byte_align
    assign resp_data_be[k*8+:8] = resp_data[(RESP_BYTES-1-k)*8+:8];
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// Ipv4 Checksum
//------------------------------------------------------------------------------------------------//

logic [16:0]                          ipv4_calc_stage1;
logic [16:0]                          ipv4_calc_stage2;

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    ipv4_calc_stage1                 <= '0;
    ipv4_calc_stage2                 <= '0;
  end
  else begin
    ipv4_calc_stage1               <= {1'b0,ip_hdr_r.chksum[15:0]} + {1'b0,ip_hdr_r.length[15:0]} - {1'b0,ipv4_len[15:0]};
    ipv4_calc_stage2               <= ipv4_calc_stage1[15:0] + ipv4_calc_stage1[16];
  end
end

assign ipv4_chksum = ipv4_calc_stage2[15:0];

//------------------------------------------------------------------------------------------------//
// Roce Host QP RAM
//------------------------------------------------------------------------------------------------//

logic [W_RAM_ADDR-1:0] host_qp_addr;
logic [31:0]           host_qp_data;
logic                  host_qp_rd_valid;
logic                  host_qp_rd_en;

s_apb_ram #(
  .R_CTRL           ( 2                 ),
  .R_WIDTH          ( 32                ),
  .R_TOTL           ( 2*4            )
) u_host_qp_ram  (
  .i_aclk           ( i_aclk            ),
  .i_arst           ( i_arst            ),
  .i_apb_m2s        ( i_apb_m2s         ),
  .o_apb_s2m        ( o_apb_s2m         ),
  // User Control Signals
  .i_pclk           ( i_pclk            ),
  .i_prst           ( i_prst            ),
  .i_addr           ( host_qp_addr      ),
  .o_rd_data        ( host_qp_data      ),
  .o_rd_data_valid  ( host_qp_rd_valid  ),
  .i_wr_data        ( '0                ),
  .i_wr_en          ( '0                ),
  .i_rd_en          ( host_qp_rd_en     )
);

assign host_qp_addr   = sif_idx;
assign host_qp_rd_en  = i_hdr_valid;
assign host_qp        = host_qp_data[23:0];
assign rnr_timer      = host_qp_data[28:24];
//------------------------------------------------------------------------------------------------//
// Roce ICRC
//------------------------------------------------------------------------------------------------//


logic              crc_tvalid;
logic              crc_tlast;
logic [W_USER-1:0] crc_tuser;
logic [W_KEEP-1:0] crc_tkeep;
logic [W_DATA-1:0] crc_tdata;
logic              crc_tready;

roce_icrc #(  
  .W_DATA           ( W_DATA        ),
  .W_USER           ( W_USER        )
) u_roce_icrc (
  .pclk             ( i_pclk        ),
  .prst             ( i_prst        ),
  .i_crc_en         ( '1            ),
  .i_axis_rx_tvalid ( resp_tvalid   ),
  .i_axis_rx_tlast  ( resp_tlast    ),
  .i_axis_rx_tkeep  ( resp_tkeep    ),
  .i_axis_rx_tdata  ( resp_tdata    ),
  .i_axis_rx_tuser  ( resp_tuser    ),
  .o_axis_rx_tready ( resp_tready   ),
  .o_axis_tx_tvalid ( crc_tvalid    ),
  .o_axis_tx_tlast  ( crc_tlast     ),
  .o_axis_tx_tkeep  ( crc_tkeep     ),
  .o_axis_tx_tdata  ( crc_tdata     ),
  .o_axis_tx_tuser  ( crc_tuser     ),
  .i_axis_tx_tready ( crc_tready    )
);


//------------------------------------------------------------------------------------------------//
// Response Buffer
//------------------------------------------------------------------------------------------------//


axis_buffer # (
  .IN_DWIDTH         ( W_DATA                ),
  .OUT_DWIDTH        ( W_DATA                ),
  .WAIT2SEND         ( 0                     ),
  .BUF_DEPTH         ( FIFO_DEPTH            ),
  .OUTPUT_SKID       ( 0                     ),
  .W_USER            ( W_USER                ),
  .NO_BACKPRESSURE   ( 0                     )
) u_axis_out_buffer (
  .in_clk            ( i_pclk                ),
  .in_rst            ( i_prst                ),
  .out_clk           ( i_pclk                ),
  .out_rst           ( i_prst                ),
  .i_axis_rx_tvalid  ( crc_tvalid            ),
  .i_axis_rx_tdata   ( crc_tdata             ),
  .i_axis_rx_tlast   ( crc_tlast             ),
  .i_axis_rx_tuser   ( crc_tuser             ),
  .i_axis_rx_tkeep   ( crc_tkeep             ),
  .o_axis_rx_tready  ( crc_tready            ),
  .o_fifo_aempty     (                       ),
  .o_fifo_afull      (                       ),
  .o_fifo_empty      (                       ),
  .o_fifo_full       (                       ),
  .o_axis_tx_tvalid  ( o_axis_tvalid         ),
  .o_axis_tx_tdata   ( o_axis_tdata          ),
  .o_axis_tx_tlast   ( o_axis_tlast          ),
  .o_axis_tx_tuser   ( o_axis_tuser          ),
  .o_axis_tx_tkeep   ( o_axis_tkeep          ),
  .i_axis_tx_tready  ( i_axis_tready         ) 
);

endmodule
