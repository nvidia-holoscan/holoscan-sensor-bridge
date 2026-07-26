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


module rx_parser
  import ptp_pkg::*;
  import rx_parser_pkg::*;
  import axis_pkg::*;
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter   AXI_DWIDTH       = 64,
  parameter   AXI_LS_DWIDTH    = 8,
  parameter   USER_WIDTH       = 17,
  localparam  KEEP_WIDTH       = (AXI_DWIDTH / 8),
  localparam  KEEP_LS_WIDTH    = (AXI_LS_DWIDTH / 8),
  parameter   NUM_LS_INST      = 6,
  parameter   MTU              = 1500,           //Maximum packet size in bytes
  parameter   N_QP             = 1,
  parameter   ENABLE_ROCE_ACK  = 0,
  parameter   SYNC_CLK         = 0
)(

  input   logic                           host_clk,
  input   logic                           host_rst,
  input   logic                           apb_clk,
  input   logic                           apb_rst,

  //Configuration
  input   logic   [47:0]                  i_dev_mac_addr,
  input   logic   [31:0]                  i_dev_ip_addr,
  //Register APB Interfaces
  input   apb_m2s                         i_apb_m2s,
  output  apb_s2m                         o_apb_s2m,
  //Host QP RAM APB Interfaces
  input   apb_m2s                         i_apb_m2s_qp_ram,
  output  apb_s2m                         o_apb_s2m_qp_ram,

  // Output Data
  output logic [71:0]                     o_dest_info,
  output logic                            o_ptp_sync_msg,
  input  logic [N_QP-1:0]                 i_stx_ready,

  //AXI RX Interface inbound Ethernet MAC
  input   logic                           i_axis_rx_tvalid,
  input   logic   [AXI_DWIDTH-1:0]        i_axis_rx_tdata,
  input   logic                           i_axis_rx_tlast,
  input   logic                           i_axis_rx_tuser,
  input   logic   [KEEP_WIDTH-1:0]        i_axis_rx_tkeep,


  //AXIS Sensor TX Interface
  output  logic                           o_stx_axis_tvalid,
  output  logic   [AXI_DWIDTH-1:0]        o_stx_axis_tdata,
  output  logic                           o_stx_axis_tlast,
  output  logic                           o_stx_axis_tuser,
  output  logic   [KEEP_WIDTH-1:0]        o_stx_axis_tkeep,

  //AXIS PTP Interface
  output  logic                           o_ptp_axis_tvalid,
  output  logic   [AXI_DWIDTH-1:0]        o_ptp_axis_tdata,
  output  logic                           o_ptp_axis_tlast,
  output  logic   [USER_WIDTH-1:0]        o_ptp_axis_tuser,
  output  logic   [KEEP_WIDTH-1:0]        o_ptp_axis_tkeep,

  //AXIS LS Interface
  output  logic                           o_ls_axis_tvalid,
  output  logic   [AXI_LS_DWIDTH-1:0]     o_ls_axis_tdata,
  output  logic                           o_ls_axis_tlast,
  output  logic   [USER_WIDTH+NUM_LS_INST-1:0] o_ls_axis_tuser,
  output  logic   [KEEP_LS_WIDTH-1:0]     o_ls_axis_tkeep,
  input   logic                           i_ls_axis_tready,

  //Bridge TX AXIS Interface to datapath
  output  logic                           o_btx_axis_tvalid,
  output  logic   [AXI_DWIDTH-1:0]        o_btx_axis_tdata,
  output  logic                           o_btx_axis_tlast,
  output  logic   [           1:0]        o_btx_axis_tuser,
  output  logic   [KEEP_WIDTH-1:0]        o_btx_axis_tkeep,
  input   logic                           i_btx_axis_tready,
  //RoCE ACK AXIS Interface to datapath
  output  logic                           o_ack_axis_tvalid,
  output  logic   [AXI_DWIDTH-1:0]        o_ack_axis_tdata,
  output  logic                           o_ack_axis_tlast,
  output  logic   [USER_WIDTH-1:0]        o_ack_axis_tuser,
  output  logic   [KEEP_WIDTH-1:0]        o_ack_axis_tkeep,
  input   logic                           i_ack_axis_tready

);

localparam ETH_BYTES             = 14;
localparam VLAN_BYTES            = 4;
localparam IP_BYTES              = 20;
localparam UDP_BYTES             = 8;
localparam BTH_BYTES             = 12;
localparam HDR_BYTES             = ETH_BYTES + VLAN_BYTES + IP_BYTES + UDP_BYTES + BTH_BYTES;
localparam HDR_DECODE_BEAT       = ((ETH_BYTES + VLAN_BYTES + IP_BYTES + UDP_BYTES) / KEEP_WIDTH);  //Data input beat that the header is completed
localparam AXI_PIPE_WIDTH        = AXI_DWIDTH + KEEP_WIDTH + 3;
localparam AXI_HDR_WIDTH         = 85+USER_WIDTH;
localparam IPV4_STAGES           = 5;
localparam BOOTP_UDP_PORT        = 16'd12268;
localparam LOOPBCK_UDP_PORT      = 16'h1000;
localparam ICMP                  = 8'h01;
localparam UDP                   = 8'h11;

// The NO_BACKPRESSURE input path needs the almost-full threshold plus one full
// packet of headroom, so FIFO_DEPTH is normally RLS_AFULL_DEPTH + 1 packet
// (3 packets deep, AFULL at 2). On wide host buses MTU/KEEP_WIDTH is small and
// this is cheap. On narrow buses (e.g. 8-bit nano) MTU/KEEP_WIDTH is large and a
// 3-packet depth crosses the 4096-entry LSRAM boundary, forcing each block from
// 4Kx4 into 8Kx2 and doubling RAM1K18 usage (overflows small devices). When
// 3 packets will not fit in a 4K-deep LSRAM, fall back to AFULL=1/depth=2
// packets, which keeps the AFULL+1-packet invariant within a single 4K LSRAM.
localparam PKT_DEPTH       = (MTU / KEEP_WIDTH);
localparam LSRAM_4K_DEPTH  = 4096;
localparam FIT_3PKT_IN_4K  = ((PKT_DEPTH * 3) <= LSRAM_4K_DEPTH);
localparam FIFO_DEPTH      = FIT_3PKT_IN_4K ? (PKT_DEPTH * 3) : (PKT_DEPTH * 2); // 3 full packets (2 on narrow buses)
localparam RLS_AFULL_DEPTH = FIT_3PKT_IN_4K ? (PKT_DEPTH * 2) : (PKT_DEPTH * 1); // AFULL after 2 packets (1 on narrow buses)
localparam META_FIFO_DEPTH = (AXI_DWIDTH < 32) ? 8 : 31;


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
// Registers
//------------------------------------------------------------------------------------------------//

logic [31:0]                          ctrl_reg [inst_dec_nctrl];
logic [31:0]                          stat_reg [inst_dec_nstat];

logic [32:0]                          parser_ipv4_chksum_err_cnt;
logic                                 parser_ipv4_chksum_iserr;
logic                                 inst_dec_stat_rst;
logic [15:0]                          cfg_ecb_udp_port;
logic [15:0]                          cfg_lpbk_udp_port;
logic [15:0]                          cfg_stx_udp_port;
logic [23:0]                          cfg_roce_ecb_dest_qp;
logic [31:0]                          cfg_ack_ctrl;

localparam  [(inst_dec_nctrl*32)-1:0] RST_VAL = {'0,32'h12B7,32'h2000,{(4*32){1'b0}},32'h1000,{(3*32){1'b0}}};

s_apb_reg #(
  .N_CTRL    ( inst_dec_nctrl   ),
  .N_STAT    ( inst_dec_nstat   ),
  .W_OFST    ( w_ofst           ),
  .RST_VAL   ( RST_VAL          ),
  .SYNC_CLK  ( SYNC_CLK         )
) u_reg_map (
  // APB Interface
  .i_aclk    ( apb_clk        ), // Slow Clock
  .i_arst    ( apb_rst        ),
  .i_apb_m2s ( i_apb_m2s      ),
  .o_apb_s2m ( o_apb_s2m      ),
  // User Control Signals
  .i_pclk    ( host_clk       ), // Fast Clock
  .i_prst    ( host_rst       ),
  .o_ctrl    ( ctrl_reg       ),
  .i_stat    ( stat_reg       )
);

assign cfg_ecb_udp_port     = ctrl_reg[ecb_udp_port][15:0];
assign cfg_lpbk_udp_port    = ctrl_reg[lpbk_udp_port][15:0];
assign cfg_stx_udp_port     = ctrl_reg[stx_udp_port][15:0];
assign cfg_roce_ecb_dest_qp = ctrl_reg[roce_ecb_dest_qp][23:0];
assign cfg_ack_ctrl         = ctrl_reg[inst_dec_ack_ctrl][31:0];

assign stat_reg[inst_dec_stat-stat_ofst]        = '{default:0};
assign stat_reg[ipv4_chksum_err_cnt-stat_ofst]  = parser_ipv4_chksum_err_cnt[31:0];
assign inst_dec_stat_rst                        = ctrl_reg[eth_pkt_ctrl][8];

//------------------------------------------------------------------------------------------------//
// Flatten Header
//------------------------------------------------------------------------------------------------//

logic [((HDR_BYTES*8)-1):0] hdr;
logic [7:0]                 hdr_array [HDR_BYTES-1:0];
logic                       hdr_valid;

logic                       sop;
logic                       eop;
logic                       is_vlan;
logic [15:0]                vlan_tci;
logic                       r_is_vlan;
logic [15:0]                r_vlan_tci;

//Header structs
eth_hdr     eth_hdr_i;
ip_hdr      ip_hdr_i;
udp_hdr     udp_hdr_i;
arp_hdr     arp_hdr_i;
ping_hdr    ping_hdr_i;
bth_hdr     bth_hdr_i;

axis_to_vec #(
  .AXI_DWIDTH       ( AXI_DWIDTH          ),
  .DATA_WIDTH       ( HDR_BYTES*8         )
) axis_to_hdr (
  .clk              ( host_clk            ),
  .rst              ( host_rst            ),
  .i_axis_rx_tvalid ( i_axis_rx_tvalid    ),
  .i_axis_rx_tdata  ( i_axis_rx_tdata     ),
  .i_axis_rx_tlast  ( i_axis_rx_tlast     ),
  .i_axis_rx_tuser  ( i_axis_rx_tuser     ),
  .i_axis_rx_tkeep  ( i_axis_rx_tkeep     ),
  .o_axis_rx_tready (                     ),
  .i_done           ( '1                  ),
  .o_busy           (                     ),
  .o_data           ( hdr                 ),
  .o_valid          ( hdr_valid           ),
  .o_byte_cnt       (                     )
);

genvar k;
generate
  for (k=0; k<HDR_BYTES; k=k+1) begin
    assign hdr_array[k] = hdr[k*8+:8];
  end
endgenerate

assign is_vlan                          = ({hdr_array[12],hdr_array[13]} == 16'h8100);
assign vlan_tci                         = {hdr_array[14], hdr_array[15]};

assign eth_hdr_i.dest_mac               = {hdr_array[00], hdr_array[01], hdr_array[02], hdr_array[03], hdr_array[04], hdr_array[05]};
assign eth_hdr_i.src_mac                = {hdr_array[06], hdr_array[07], hdr_array[08], hdr_array[09], hdr_array[10], hdr_array[11]};
assign eth_hdr_i.ethertype              = is_vlan ? {hdr_array[16], hdr_array[17]} : {hdr_array[12], hdr_array[13]};
assign ip_hdr_i.ver_ihl                 = is_vlan ? {hdr_array[18]} : {hdr_array[14]};
assign ip_hdr_i.dscp_ecn                = is_vlan ? {hdr_array[19]} : {hdr_array[15]};
assign ip_hdr_i.length                  = is_vlan ? {hdr_array[20], hdr_array[21]} : {hdr_array[16], hdr_array[17]};
assign ip_hdr_i.id                      = is_vlan ? {hdr_array[22], hdr_array[23]} : {hdr_array[18], hdr_array[19]};
assign ip_hdr_i.flags_frag              = is_vlan ? {hdr_array[24], hdr_array[25]} : {hdr_array[20], hdr_array[21]};
assign ip_hdr_i.ttl                     = is_vlan ? {hdr_array[26]} : {hdr_array[22]};
assign ip_hdr_i.protocol                = is_vlan ? {hdr_array[27]} : {hdr_array[23]};
assign ip_hdr_i.chksum                  = is_vlan ? {hdr_array[28], hdr_array[29]} : {hdr_array[24], hdr_array[25]};
assign ip_hdr_i.src_addr                = is_vlan ? {hdr_array[30], hdr_array[31], hdr_array[32], hdr_array[33]} : {hdr_array[26], hdr_array[27], hdr_array[28], hdr_array[29]};
assign ip_hdr_i.dest_addr               = is_vlan ? {hdr_array[34], hdr_array[35], hdr_array[36], hdr_array[37]} : {hdr_array[30], hdr_array[31], hdr_array[32], hdr_array[33]};
assign udp_hdr_i.src_port               = is_vlan ? {hdr_array[38], hdr_array[39]} : {hdr_array[34], hdr_array[35]};
assign udp_hdr_i.dest_port              = is_vlan ? {hdr_array[40], hdr_array[41]} : {hdr_array[36], hdr_array[37]};
assign udp_hdr_i.length                 = is_vlan ? {hdr_array[42], hdr_array[43]} : {hdr_array[38], hdr_array[39]};
assign udp_hdr_i.chksum                 = is_vlan ? {hdr_array[44], hdr_array[45]} : {hdr_array[40], hdr_array[41]};
assign arp_hdr_i.tpa                    = is_vlan ? {hdr_array[42],hdr_array[43],hdr_array[44],hdr_array[45]} : {hdr_array[38],hdr_array[39],hdr_array[40],hdr_array[41]};
assign ping_hdr_i.icmp_type             = is_vlan ? {hdr_array[38]} : {hdr_array[34]};

assign bth_hdr_i.opcode                 = is_vlan ? {hdr_array[46]} : {hdr_array[42]};
assign bth_hdr_i.s_m_pad_tver           = is_vlan ? {hdr_array[47]} : {hdr_array[43]};
assign bth_hdr_i.pkey                   = is_vlan ? {hdr_array[48], hdr_array[49]} : {hdr_array[44], hdr_array[45]};
assign bth_hdr_i.f_b_rsv6               = is_vlan ? {hdr_array[50]} : {hdr_array[46]};
assign bth_hdr_i.dest_qp                = is_vlan ? {hdr_array[51], hdr_array[52], hdr_array[53]} : {hdr_array[47], hdr_array[48], hdr_array[49]};
assign bth_hdr_i.a_rsv7                 = is_vlan ? {hdr_array[54]} : {hdr_array[50]};
assign bth_hdr_i.psn                    = is_vlan ? {hdr_array[55], hdr_array[56], hdr_array[57]} : {hdr_array[51], hdr_array[52], hdr_array[53]};

//------------------------------------------------------------------------------------------------//
// Pipeline
//------------------------------------------------------------------------------------------------//

logic [IPV4_STAGES-1:0]    proc_pipeline;
logic                      axis_rx_tlast_prev;
logic                      one_cycle;

//Raise packet active after the first tvalid following a tlast.
always_ff @(posedge host_clk) begin
  if (host_rst) begin
    proc_pipeline               <= '0;
    axis_rx_tlast_prev          <= '0;
    one_cycle                   <= '0;
  end
  else begin
    proc_pipeline[IPV4_STAGES-1:1] <= {proc_pipeline[IPV4_STAGES-2:0]};
    proc_pipeline[0]               <= (proc_pipeline[0] && !one_cycle) ? '0 : hdr_valid;  // Pulse hdr_valid
    axis_rx_tlast_prev             <= i_axis_rx_tlast && i_axis_rx_tvalid;
    // Back to back single-cycle packets
    one_cycle                      <= (HDR_DECODE_BEAT != 0) ? '0 : (i_axis_rx_tvalid && axis_rx_tlast_prev);
  end
end

//------------------------------------------------------------------------------------------------//
// IPV4
//------------------------------------------------------------------------------------------------//

//IPv4 Checksum
logic [16:0]                          ipv4_calc_stage1  [5];
logic [16:0]                          ipv4_calc_stage2  [3];
logic [16:0]                          ipv4_calc_stage3  [2];
logic [16:0]                          ipv4_calc_stage4  [1];
logic                                 ipv4_iserr;
logic                                 ipv4_empty;
logic                                 fcs_empty;
logic                                 hdr_rden;
logic [IPV4_STAGES-2:0]               is_ipv4_type;
logic                                 fcs_iserr;
logic                                 fcs_err_data;
logic                                 ack_pkt_done;
logic                                 ack_pkt_is_err;
logic                                 buf_rls_afull;


//calculated value should be 0xFFFF.
always_ff @(posedge host_clk) begin
  if (host_rst) begin
    ipv4_calc_stage1                    <= '{default:0};
    ipv4_calc_stage2                    <= '{default:0};
    ipv4_calc_stage3                    <= '{default:0};
    ipv4_calc_stage4                    <= '{default:0};
    parser_ipv4_chksum_err_cnt          <= '0;
    parser_ipv4_chksum_iserr            <= '0;
    is_ipv4_type                        <= '0;
    ack_pkt_done                        <= '0;
    ack_pkt_is_err                      <= '0;
  end
  else begin
    is_ipv4_type <= {is_ipv4_type[IPV4_STAGES-3:0],(eth_hdr_i.ethertype == 16'h0800)};
    ack_pkt_done   <= i_axis_rx_tvalid && i_axis_rx_tlast;
    ack_pkt_is_err <= fcs_err_data;
    //If the packet is IPv4 and there is a IPv4 checksum failure, add to stat counter. Reset on sync'ed SW signal.
    if (inst_dec_stat_rst) begin
      parser_ipv4_chksum_err_cnt          <= '0;
    end
    else begin
      parser_ipv4_chksum_err_cnt          <= (parser_ipv4_chksum_err_cnt[32]) ? '1 : parser_ipv4_chksum_err_cnt + parser_ipv4_chksum_iserr;
    end

    if (hdr_valid) begin
      ipv4_calc_stage1[0]               <= {ip_hdr_i.ver_ihl, ip_hdr_i.dscp_ecn} + ip_hdr_i.length;
      ipv4_calc_stage1[1]               <= ip_hdr_i.id + ip_hdr_i.flags_frag;
      ipv4_calc_stage1[2]               <= {ip_hdr_i.ttl, ip_hdr_i.protocol} + ip_hdr_i.chksum;
      ipv4_calc_stage1[3]               <= ip_hdr_i.src_addr[31:16] + ip_hdr_i.src_addr[15:0];
      ipv4_calc_stage1[4]               <= ip_hdr_i.dest_addr[31:16] + ip_hdr_i.dest_addr[15:0];
    end

    if (proc_pipeline[0]) begin
      ipv4_calc_stage2[0]               <= ipv4_calc_stage1[0][15:0] + ipv4_calc_stage1[1][15:0];
      ipv4_calc_stage2[1]               <= ipv4_calc_stage1[2][15:0] + ipv4_calc_stage1[3][15:0];
      ipv4_calc_stage2[2]               <= ipv4_calc_stage1[4][15:0] + (ipv4_calc_stage1[0][16] + ipv4_calc_stage1[1][16] + ipv4_calc_stage1[2][16] + ipv4_calc_stage1[3][16] + ipv4_calc_stage1[4][16]);
    end

    if (proc_pipeline[1]) begin
      ipv4_calc_stage3[0]               <= ipv4_calc_stage2[0][15:0] + ipv4_calc_stage2[1][15:0];
      ipv4_calc_stage3[1]               <= ipv4_calc_stage2[2][15:0] + (ipv4_calc_stage2[0][16] + ipv4_calc_stage2[1][16] + ipv4_calc_stage2[2][16]);
    end

    if (proc_pipeline[2]) begin
      ipv4_calc_stage4[0]               <= (ipv4_calc_stage3[0][16] + ipv4_calc_stage3[1][16]) + ipv4_calc_stage3[0][15:0] + ipv4_calc_stage3[1][15:0];
    end

    if (proc_pipeline[3] && (is_ipv4_type[3])) begin
      parser_ipv4_chksum_iserr        <= !(&(ipv4_calc_stage4[0][15:0] + ipv4_calc_stage4[0][16]));
    end
    else begin
      parser_ipv4_chksum_iserr        <= '0;
    end
  end
end

sc_fifo #(
  .DATA_WIDTH                 ( 1               ),
  .FIFO_DEPTH                 ( META_FIFO_DEPTH )
) ipv4_buffer (
  .clk      ( host_clk                 ),
  .rst      ( host_rst                 ),
  .wr       ( proc_pipeline[4]         ),
  .din      ( parser_ipv4_chksum_iserr ),
  .full     (                          ),
  .afull    (                          ),
  .over     (                          ),
  .rd       ( hdr_rden                 ),
  .dout     ( ipv4_iserr               ),
  .dval     (                          ),
  .empty    ( ipv4_empty               ),
  .aempty   (                          ),
  .under    (                          ),
  .count    (                          )
);

sc_fifo #(
  .DATA_WIDTH   ( 1                                                ),
  .FIFO_DEPTH   ( META_FIFO_DEPTH                                  )
) fcs_buffer (
  .clk          ( host_clk                                         ),
  .rst          ( host_rst                                         ),
  .wr           ( (i_axis_rx_tvalid & i_axis_rx_tlast)             ),
  .din          ( fcs_err_data                                     ),
  .full         (                                                  ),
  .afull        (                                                  ),
  .over         (                                                  ),
  .rd           ( hdr_rden                                         ),
  .dout         ( fcs_iserr                                        ),
  .dval         (                                                  ),
  .empty        ( fcs_empty                                        ),
  .aempty       (                                                  ),
  .under        (                                                  ),
  .count        (                                                  )
);

assign fcs_err_data = i_axis_rx_tuser || buf_rls_afull;

//------------------------------------------------------------------------------------------------//
// PTP
//------------------------------------------------------------------------------------------------//

localparam PTP_SYNC_TYPE     = 4'h0;
localparam PTP_FOLLOWUP_TYPE = 4'h8;
localparam PTP_DEL_RESP_TYPE = 4'h9;

logic       r_is_ptp;
logic [3:0] rx_msg_type_ptp_hdr, rx_msg_type_ptp;
logic       is_sync_msg;
logic       is_followup_msg;
logic       is_del_resp_msg;

assign is_sync_msg = (rx_msg_type_ptp == PTP_SYNC_TYPE);
assign is_followup_msg = (rx_msg_type_ptp == PTP_FOLLOWUP_TYPE);
assign is_del_resp_msg = (rx_msg_type_ptp == PTP_DEL_RESP_TYPE);

//------------------------------------------------------------------------------------------------//
// Header Parser
//------------------------------------------------------------------------------------------------//


logic        eth_mac_dest_match;
logic        eth_mac_dest_match_lsb;
logic        eth_mac_dest_match_msb;
logic        eth_mac_dest_bcast;
logic        eth_mac_ptp_match;
logic        eth_type_arp;
logic        eth_type_ptp;
logic        ip_dest_addr_match;
logic        ip_dest_addr_match_lsb;
logic        ip_dest_addr_match_msb;
logic        prot_is_udp;
logic        prot_is_ping;
logic        ecb_port_match;
logic        stx_port_match;
logic        bootp_port_match;
logic        loopback_port_match;
logic        roce_ecb_dest_qp_match;
logic        arp_tpa_match;
logic        bth_tx_opcode_match;
logic        opcode_is_rc;
logic        opcode_is_uc;
logic        opcode_is_send;
logic        opcode_is_write;
logic        stx_not_empty;

logic [23:0] bth_dest_qp;

logic hdr_empty;
logic is_stx;
logic r_is_stx;
logic is_btx;
logic r_is_btx;
logic is_loopbk;
logic is_bootp;
logic is_arp;
logic is_ptp;
logic is_ecb;
logic is_roce_ecb;
logic is_ping;
logic is_pkt;

logic [47:0]              hdr_din_mac;
logic [AXI_HDR_WIDTH-1:0] hdr_din;
logic [AXI_HDR_WIDTH-1:0] hdr_dout;

// Parse Header
always_ff @(posedge host_clk) begin
  if (host_rst) begin
    eth_mac_dest_match_lsb      <= '0;
    eth_mac_dest_match_msb      <= '0;
    eth_mac_dest_bcast          <= '0;
    eth_mac_ptp_match           <= '0;
    eth_type_arp                <= '0;
    eth_type_ptp                <= '0;
    rx_msg_type_ptp_hdr         <= '0;
    ip_dest_addr_match_lsb      <= '0;
    ip_dest_addr_match_msb      <= '0;
    prot_is_udp                 <= '0;
    prot_is_ping                <= '0;
    ecb_port_match              <= '0;
    stx_port_match              <= '0;
    roce_ecb_dest_qp_match      <= '0;
    bootp_port_match            <= '0;
    loopback_port_match         <= '0;
    arp_tpa_match               <= '0;
    hdr_din_mac                 <= '0;
    bth_dest_qp                 <= '0;
    bth_tx_opcode_match         <= '0;
    stx_not_empty               <= '0;
    r_is_vlan                   <= '0;
    r_vlan_tci                  <= '0;
  end
  else begin
    if (hdr_valid) begin
      eth_mac_dest_match_lsb <= eth_hdr_i.dest_mac[23:0] == i_dev_mac_addr[23:0];
      eth_mac_dest_match_msb <= eth_hdr_i.dest_mac[47:24] == i_dev_mac_addr[47:24];
      eth_mac_dest_bcast     <= &eth_hdr_i.dest_mac;
      eth_type_arp           <= (eth_hdr_i.ethertype == 16'h0806);
      eth_mac_ptp_match      <= (eth_hdr_i.dest_mac == PTP_FW_MULTI_ADDR) || (eth_hdr_i.dest_mac == PTP_NON_FW_MULTI_ADDR) ;
      ip_dest_addr_match_lsb <= ip_hdr_i.dest_addr[15:0] == i_dev_ip_addr[15:0];
      ip_dest_addr_match_msb <= ip_hdr_i.dest_addr[31:16] == i_dev_ip_addr[31:16];
      prot_is_udp            <= (ip_hdr_i.protocol == UDP);
      prot_is_ping           <= (ip_hdr_i.protocol == ICMP) && (ping_hdr_i.icmp_type == 8'h8);
      ecb_port_match         <= (udp_hdr_i.dest_port == cfg_ecb_udp_port);
      stx_port_match         <= udp_hdr_i.dest_port == cfg_stx_udp_port;
      bootp_port_match       <= (udp_hdr_i.dest_port == BOOTP_UDP_PORT);
      loopback_port_match    <= (udp_hdr_i.dest_port == LOOPBCK_UDP_PORT);
      roce_ecb_dest_qp_match <= (bth_hdr_i.dest_qp == cfg_roce_ecb_dest_qp);
      eth_type_ptp           <= (eth_hdr_i.ethertype == 16'h88F7);
      rx_msg_type_ptp_hdr    <= hdr_array[14][3:0];
      arp_tpa_match          <= (arp_hdr_i.tpa == i_dev_ip_addr);
      hdr_din_mac            <= {eth_hdr_i.dest_mac[7:0], eth_hdr_i.dest_mac[15:8], eth_hdr_i.dest_mac[23:16],
                               eth_hdr_i.dest_mac[31:24], eth_hdr_i.dest_mac[39:32], eth_hdr_i.dest_mac[47:40]};
      bth_dest_qp            <= bth_hdr_i.dest_qp;
      bth_tx_opcode_match    <= (opcode_is_uc && opcode_is_send) || (opcode_is_rc && (opcode_is_write || opcode_is_send) && ENABLE_ROCE_ACK);
      stx_not_empty          <= udp_hdr_i.length > 'd16; // Is not empty packet
      r_is_vlan              <= is_vlan;
      r_vlan_tci             <= vlan_tci;
    end
  end
end


assign opcode_is_rc    = bth_hdr_i.opcode[7:4] == BTH_RC;
assign opcode_is_uc    = bth_hdr_i.opcode[7:4] == BTH_UC;
assign opcode_is_send  = bth_hdr_i.opcode[3:0] inside {BTH_SEND_FIRST, BTH_SEND_MIDDLE, BTH_SEND_LAST, BTH_SEND_ONLY};
assign opcode_is_write = bth_hdr_i.opcode[3:0] inside {BTH_WRITE_FIRST, BTH_WRITE_MIDDLE, BTH_WRITE_LAST, BTH_WRITE_ONLY};

assign eth_mac_dest_match   = eth_mac_dest_match_lsb && eth_mac_dest_match_msb;
assign ip_dest_addr_match   = ip_dest_addr_match_lsb && ip_dest_addr_match_msb;
assign is_ecb               = prot_is_udp && ip_dest_addr_match && eth_mac_dest_match && ecb_port_match;
assign is_bootp             = prot_is_udp && eth_mac_dest_match && bootp_port_match;
assign is_ping              = prot_is_ping && ip_dest_addr_match && eth_mac_dest_match;
assign is_stx               = prot_is_udp && ip_dest_addr_match && eth_mac_dest_match && stx_port_match && !roce_ecb_dest_qp_match && bth_tx_opcode_match; 
assign is_roce_ecb          = prot_is_udp && ip_dest_addr_match && eth_mac_dest_match && stx_port_match && roce_ecb_dest_qp_match && bth_tx_opcode_match;
assign is_loopbk            = prot_is_udp && ip_dest_addr_match && eth_mac_dest_match && loopback_port_match;
assign is_arp               = (eth_mac_dest_bcast || eth_mac_dest_match) && eth_type_arp && arp_tpa_match;
assign is_ptp               = (eth_type_ptp && eth_mac_ptp_match);
assign is_btx               = !eth_mac_dest_match && !is_arp;

sc_fifo #(
  .DATA_WIDTH ( AXI_HDR_WIDTH   ),
  .FIFO_DEPTH ( META_FIFO_DEPTH )
) hdr_buffer (
  .clk      (host_clk         ),
  .rst      (host_rst         ),
  .wr       (proc_pipeline[0] ),
  .din      (hdr_din          ),
  .full     (                 ),
  .afull    (                 ),
  .over     (                 ),
  .rd       (hdr_rden         ),
  .dout     (hdr_dout         ),
  .dval     (                 ),
  .empty    (hdr_empty        ),
  .aempty   (                 ),
  .under    (                 ),
  .count    (                 )
);


assign hdr_din = {r_is_vlan,r_vlan_tci,rx_msg_type_ptp_hdr,bth_dest_qp,hdr_din_mac,is_roce_ecb,is_arp,is_ping,is_ecb,is_bootp,is_loopbk,is_ptp,is_btx,is_stx};
assign o_dest_info          = hdr_dout[80:9];
assign o_ptp_sync_msg = proc_pipeline[0] && is_ptp && (rx_msg_type_ptp_hdr == PTP_SYNC_TYPE);


//------------------------------------------------------------------------------------------------//
// RoCE ACK logic
//------------------------------------------------------------------------------------------------//

logic ack_empty;
logic ack_is_err;

generate
  if (ENABLE_ROCE_ACK) begin
    roce_ack #(
      .W_DATA           ( AXI_DWIDTH         ),
      .W_USER           ( USER_WIDTH         ),
      .HDR_BYTES        ( 54                 ),
      .N_QP             ( N_QP               ),
      .META_FIFO_DEPTH  ( META_FIFO_DEPTH    ),
      .FIFO_DEPTH       ( FIFO_DEPTH         )
    ) u_roce_ack (
      .i_pclk           ( host_clk           ),
      .i_prst           ( host_rst           ),
      //Register APB Interfaces
      .i_aclk           ( apb_clk            ),
      .i_arst           ( apb_rst            ),
      //Host QP RAM APB Interfaces
      .i_apb_m2s        ( i_apb_m2s_qp_ram   ),
      .o_apb_s2m        ( o_apb_s2m_qp_ram   ),
      // Packet Valid Interface
      .o_empty          ( ack_empty          ),
      .o_is_err         ( ack_is_err         ),
      .i_rd_en          ( hdr_rden           ),
      .i_pkt_done       ( ack_pkt_done       ),
      .i_pkt_is_err     ( ack_pkt_is_err     ),
      // Sensor TX Backpressure
      .i_stx_ready      ( i_stx_ready        ),
      // Packet Header 
      .i_eth_hdr        ( eth_hdr_i          ),
      .i_ip_hdr         ( ip_hdr_i           ),
      .i_udp_hdr        ( udp_hdr_i          ),
      .i_bth_hdr        ( bth_hdr_i          ),
      .i_hdr_valid      ( hdr_valid          ),
      .i_vlan_tuser     ( {r_is_vlan,r_vlan_tci}),
      // Backpressure
      .i_is_stx         ( is_stx             ),
      // Registers
      .i_ctrl_reg       ( cfg_ack_ctrl       ),
      // AXIS Interface
      .o_axis_tvalid    ( o_ack_axis_tvalid  ),
      .o_axis_tdata     ( o_ack_axis_tdata   ),
      .o_axis_tlast     ( o_ack_axis_tlast   ),
      .o_axis_tuser     ( o_ack_axis_tuser   ),
      .o_axis_tkeep     ( o_ack_axis_tkeep   ),
      .i_axis_tready    ( i_ack_axis_tready  )
    );
  end
  else begin
    assign o_ack_axis_tvalid = '0;
    assign o_ack_axis_tdata  = '0;
    assign o_ack_axis_tlast  = '0;
    assign o_ack_axis_tuser  = '0;
    assign o_ack_axis_tkeep  = '0;
    assign ack_empty         = '0;
    assign ack_is_err        = '0;
    assign o_apb_s2m_qp_ram = '{default:0};
  end
endgenerate


//------------------------------------------------------------------------------------------------//
// AXIS Buffer
//------------------------------------------------------------------------------------------------//

logic                   buf_axis_tvalid;
logic                   buf_axis_tlast;
logic [AXI_DWIDTH-1:0]  buf_axis_tdata;
logic [KEEP_WIDTH-1:0]  buf_axis_tkeep;
logic [NUM_LS_INST+USER_WIDTH-1:0] buf_axis_tuser;
logic                   buf_axis_tready;
logic                   w_buf_axis_tready;
logic                   w_buf_axis_tvalid;
logic                   ipv4_invld;

axis_buffer # (
  .IN_DWIDTH       ( AXI_DWIDTH     ),
  .OUT_DWIDTH      ( AXI_DWIDTH     ),
  .WAIT2SEND       ( 1              ),
  .BUF_DEPTH       ( FIFO_DEPTH     ),
  .OUTPUT_SKID     ( '0             ),
  .NO_BACKPRESSURE ( 1              )
) u_axis_inp_buffer (
  .in_clk            ( host_clk              ),
  .in_rst            ( host_rst              ),
  .out_clk           ( host_clk              ),
  .out_rst           ( host_rst              ),
  .i_axis_rx_tvalid  ( i_axis_rx_tvalid      ),
  .i_axis_rx_tdata   ( i_axis_rx_tdata       ),
  .i_axis_rx_tlast   ( i_axis_rx_tlast       ),
  .i_axis_rx_tuser   ( i_axis_rx_tuser       ),
  .i_axis_rx_tkeep   ( i_axis_rx_tkeep       ),
  .o_axis_rx_tready  (                       ),
  .o_fifo_aempty     (                       ),
  .o_fifo_afull      (                       ),
  .o_fifo_empty      (                       ),
  .o_fifo_full       (                       ),
  .o_axis_tx_tvalid  ( w_buf_axis_tvalid     ),
  .o_axis_tx_tdata   ( buf_axis_tdata        ),
  .o_axis_tx_tlast   ( buf_axis_tlast        ),
  .o_axis_tx_tuser   (                       ),
  .o_axis_tx_tkeep   ( buf_axis_tkeep        ),
  .i_axis_tx_tready  ( w_buf_axis_tready     ) // Cannot backpressure into ethernet host, however, use buffer as a pipeline reg
);


logic hdr_ready;
logic buf_axis_eop;
logic                    buf_rmv_axis_tvalid;
logic                    buf_rmv_axis_tlast;
logic [ AXI_DWIDTH-1:0]  buf_rmv_axis_tdata;
logic [ KEEP_WIDTH-1:0]  buf_rmv_axis_tkeep;
logic [NUM_LS_INST+USER_WIDTH-1:0] buf_rmv_axis_tuser;

// Backpressure
always_ff @(posedge host_clk) begin
  if (host_rst) begin
    buf_axis_tready <= '0;
    hdr_ready       <= '0;
    is_pkt          <= '0;
  end
  else begin
    is_pkt <= (sop && eop) ? '0   :
              (sop       ) ? '1   :
              (eop       ) ? '0   :
                            is_pkt;
    hdr_ready <= (!ipv4_empty && !fcs_empty && !hdr_empty && !hdr_rden && !ack_empty);
    if (buf_axis_tready) begin
      buf_axis_tready <= !buf_axis_eop;
    end
    else begin
      buf_axis_tready <= (hdr_ready);
    end
  end
end

assign hdr_rden = (hdr_ready && !buf_axis_tready);
assign ipv4_invld = ipv4_iserr | fcs_iserr | ack_is_err;

assign sop                  = (w_buf_axis_tvalid && w_buf_axis_tready) && !is_pkt;
assign buf_axis_eop         = (buf_axis_tlast && w_buf_axis_tvalid && w_buf_axis_tready);
assign eop                  = (buf_rmv_axis_tlast && buf_rmv_axis_tvalid) ||
                              (buf_axis_tlast && w_buf_axis_tvalid && w_buf_axis_tready && ipv4_invld);

assign r_is_stx             = hdr_dout[0];
assign r_is_btx             = hdr_dout[1];
assign r_is_ptp             = hdr_dout[2];
assign buf_axis_tuser       = {hdr_dout[101:85], hdr_dout[8:3]};
assign rx_msg_type_ptp      = hdr_dout[84:81];
assign buf_axis_tvalid      = (ipv4_invld) ? '0 : w_buf_axis_tvalid && buf_axis_tready;
// Light backpressure from bridge pkt proc due to short back to back packets
assign w_buf_axis_tready    = buf_axis_tready && (i_btx_axis_tready || !r_is_btx);

logic            [31:0] vlan_tag;
logic            [31:0] vlan_tag_be;

axis_remove #(
  .AXI_DWIDTH         ( AXI_DWIDTH             ),
  .W_USER             ( NUM_LS_INST+USER_WIDTH ),
  .REMOVE_OFFSET      ( 96                     ),
  .REMOVE_WIDTH       ( 32                     )
) u_vlan_remove (
  .clk                ( host_clk               ),
  .rst                ( host_rst               ),
  .i_enable           ( buf_axis_tuser[USER_WIDTH+NUM_LS_INST-1] && !r_is_btx),
  .o_removed_data     ( vlan_tag               ),
  .o_gated_enable     (                        ),
  .i_axis_rx_tvalid   ( buf_axis_tvalid        ),
  .i_axis_rx_tdata    ( buf_axis_tdata         ),
  .i_axis_rx_tlast    ( buf_axis_tlast         ),
  .i_axis_rx_tuser    ( buf_axis_tuser         ),
  .i_axis_rx_tkeep    ( buf_axis_tkeep         ),
  .o_axis_rx_tready   (                        ),
  .o_axis_tx_tvalid   ( buf_rmv_axis_tvalid    ),
  .o_axis_tx_tdata    ( buf_rmv_axis_tdata     ),
  .o_axis_tx_tlast    ( buf_rmv_axis_tlast     ),
  .o_axis_tx_tuser    ( buf_rmv_axis_tuser     ),
  .o_axis_tx_tkeep    ( buf_rmv_axis_tkeep     ),
  .i_axis_tx_tready   ( 1'b1                   )
);

generate
  for (genvar j=0; j<4; j++) begin
    assign vlan_tag_be[j*8+:8] = vlan_tag[(4-1-j)*8+:8];
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// Low Speed Packet Buffer
//------------------------------------------------------------------------------------------------//

logic                   is_ls;

logic                         rls_axis_tvalid;
logic [AXI_LS_DWIDTH-1:0]     rls_axis_tdata;
logic                         rls_axis_tlast;
logic [USER_WIDTH+NUM_LS_INST-1:0] rls_axis_tuser;
logic [(AXI_LS_DWIDTH/8)-1:0] rls_axis_tkeep;
logic                         rls_axis_tready;

axis_buffer # (
  .IN_DWIDTH         ( AXI_DWIDTH                 ),
  .OUT_DWIDTH        ( AXI_LS_DWIDTH              ),
  .WAIT2SEND         ( 0                          ),
  .BUF_DEPTH         ( FIFO_DEPTH                 ),
  .W_USER            ( 17 + NUM_LS_INST           ),
  .OUTPUT_SKID       ( '1                         ),
  .ALMOST_FULL_DEPTH ( RLS_AFULL_DEPTH            ),
  .NO_BACKPRESSURE   ( 1                          )
) u_axis_out_buffer (
  .in_clk            ( host_clk                   ),
  .in_rst            ( host_rst                   ),
  .out_clk           ( host_clk                   ),
  .out_rst           ( host_rst                   ),
  .i_axis_rx_tvalid  ( (buf_rmv_axis_tvalid & is_ls) ),
  .i_axis_rx_tdata   ( buf_rmv_axis_tdata            ),
  .i_axis_rx_tlast   ( buf_rmv_axis_tlast            ),
  .i_axis_rx_tuser   ( buf_rmv_axis_tuser            ),
  .i_axis_rx_tkeep   ( buf_rmv_axis_tkeep            ),
  .o_axis_rx_tready  (                               ),
  .o_fifo_aempty     (                            ),
  .o_fifo_afull      ( buf_rls_afull              ),
  .o_fifo_empty      (                            ),
  .o_fifo_full       (                            ),
  .o_axis_tx_tvalid  ( rls_axis_tvalid            ),
  .o_axis_tx_tdata   ( rls_axis_tdata             ),
  .o_axis_tx_tlast   ( rls_axis_tlast             ),
  .o_axis_tx_tuser   ( rls_axis_tuser             ),
  .o_axis_tx_tkeep   ( rls_axis_tkeep             ),
  .i_axis_tx_tready  ( rls_axis_tready            )
);

axis_reg # (
  .DWIDTH             ( AXI_LS_DWIDTH + (AXI_LS_DWIDTH/8) + NUM_LS_INST + 1 + USER_WIDTH)
) u_ls_axis_reg (
  .clk                ( host_clk                                                          ),
  .rst                ( host_rst                                                          ),
  .i_axis_rx_tvalid   ( rls_axis_tvalid                                                   ),
  .i_axis_rx_tdata    ( {rls_axis_tdata,rls_axis_tlast,rls_axis_tuser,rls_axis_tkeep}     ),
  .o_axis_rx_tready   ( rls_axis_tready                                                   ),
  .o_axis_tx_tvalid   ( o_ls_axis_tvalid                                                  ),
  .o_axis_tx_tdata    ( {o_ls_axis_tdata,o_ls_axis_tlast,o_ls_axis_tuser,o_ls_axis_tkeep} ),
  .i_axis_tx_tready   ( i_ls_axis_tready                                                  )
);

assign is_ls = (|buf_rmv_axis_tuser[NUM_LS_INST-1:0]);

//------------------------------------------------------------------------------------------------//
// UDP Loopback
//------------------------------------------------------------------------------------------------//

assign o_stx_axis_tvalid  = (buf_rmv_axis_tvalid && r_is_stx);
assign o_stx_axis_tdata   = buf_rmv_axis_tdata;
assign o_stx_axis_tlast   = buf_rmv_axis_tlast;
assign o_stx_axis_tuser   = '0;
assign o_stx_axis_tkeep   = buf_rmv_axis_tkeep;

assign o_ptp_axis_tvalid  = buf_rmv_axis_tvalid && r_is_ptp;
assign o_ptp_axis_tdata   = buf_rmv_axis_tdata;
assign o_ptp_axis_tlast   = buf_rmv_axis_tlast;
assign o_ptp_axis_tuser   = buf_rmv_axis_tuser[USER_WIDTH+NUM_LS_INST-1:NUM_LS_INST];
assign o_ptp_axis_tkeep   = buf_rmv_axis_tkeep;

assign o_btx_axis_tvalid  = (buf_rmv_axis_tvalid && r_is_btx);
assign o_btx_axis_tdata   = buf_rmv_axis_tdata;
assign o_btx_axis_tlast   = buf_rmv_axis_tlast;
assign o_btx_axis_tuser   = (r_is_ptp && is_sync_msg    ) ? 'd1:
                            (r_is_ptp && is_followup_msg) ? 'd2:
                            (r_is_ptp && is_del_resp_msg) ? 'd3:
                                                            '0 ;
assign o_btx_axis_tkeep   = buf_rmv_axis_tkeep;


endmodule
