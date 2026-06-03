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


module rx_ls_parser
  import rx_parser_pkg::*;
  import axis_pkg::*;
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter   AXI_DWIDTH      = 64,
  parameter   AXI_LS_DWIDTH   = 8,
  localparam  KEEP_WIDTH      = (AXI_DWIDTH / 8),
  localparam  KEEP_LS_WIDTH   = (AXI_LS_DWIDTH / 8),
  parameter   ENUM_DWIDTH     = 296,
  parameter   NUM_LS_RX_INST  = 4,
  parameter   NUM_LS_TX_INST  = 4,
  parameter   NUM_HOST        = 1,
  parameter   UUID            = 128'h0000_0000_0000_0000_0060_0DCA_FEC0_FFEE,
  parameter   MTU             = 1500,           //Maximum packet size in bytes
  parameter   SYNC_CLK        = 0
)(

  input   logic                           i_pclk,
  input   logic                           i_prst,

  input   logic                           i_aclk,
  input   logic                           i_arst,

  input   apb_m2s                         i_apb_m2s_ecb,
  output  apb_s2m                         o_apb_s2m_ecb,
  input   apb_m2s                         i_apb_m2s_evt,
  output  apb_s2m                         o_apb_s2m_evt,
  output  apb_m2s                         o_apb_m2s_evt,
  input   apb_s2m                         i_apb_s2m_evt,
  input   apb_m2s                         i_apb_m2s_ram,
  output  apb_s2m                         o_apb_s2m_ram,

  //AXI LS Interface inbound from RX parser
  input   logic   [NUM_HOST-1:0]          i_axis_tvalid,
  input   logic   [AXI_LS_DWIDTH-1:0]     i_axis_tdata        [NUM_HOST],
  input   logic   [NUM_HOST-1:0]          i_axis_tlast,
  input   logic   [NUM_LS_RX_INST-1:0]    i_axis_tuser        [NUM_HOST],
  input   logic   [KEEP_LS_WIDTH-1:0]     i_axis_tkeep        [NUM_HOST],
  output  logic   [NUM_HOST-1:0]          o_axis_tready,

  // Input Control Data
  input   logic                           i_ptp_clk,
  input   logic                           i_ptp_rst,
  input   logic   [79:0]                  i_ptp,
  input   logic   [79:0]                  i_ptp_hif,

  input   logic                           i_init_done,
  input           [47:0]                  i_dev_mac_addr      [NUM_HOST],
  input           [31:0]                  i_eeprom_ip_addr,
  input   logic                           i_eeprom_ip_addr_vld,
  input   logic   [7:0]                   i_hsb_stat,

  output          [31:0]                  o_dev_ip_addr       [NUM_HOST],
  input   logic   [295:0]                 i_enum_data,
  input   logic   [31:0]                  i_evt_vec,
  input   logic                           i_pps,
  input   logic   [NUM_HOST-1:0]          i_pkt_inc,

  //APB Control Interface
  output  apb_m2s                         o_apb_m2s_ecb,
  input   apb_s2m                         i_apb_s2m_ecb,

  //AXI LS Interface outbound to Ethernet Packet at line rate
  output   logic  [NUM_HOST-1:0]          o_axis_tvalid,
  output   logic  [AXI_DWIDTH-1:0]        o_axis_tdata,
  output   logic                          o_axis_tlast,
  output   logic                          o_axis_tuser,
  output   logic  [KEEP_WIDTH-1:0]        o_axis_tkeep,
  input    logic  [NUM_HOST-1:0]          i_axis_tready
);

localparam ETH_BYTES            = 14;
localparam IP_BYTES             = 20;
localparam UDP_BYTES            = 8;
localparam HDR_BYTES            = ETH_BYTES + IP_BYTES + UDP_BYTES;

localparam FIFO_DEPTH = (MTU / KEEP_WIDTH) * 2; // Can fit 2 full packets

localparam LOOPBACK_RX_IDX = 0;
localparam BOOTP_RX_IDX    = 1;
localparam ECB_RX_IDX      = 2;
localparam PING_RX_IDX     = 3;
localparam ARP_RX_IDX      = 4;
localparam ROCE_ECB_RX_IDX = 5;

localparam BOOTP_TX_IDX     = 2;
localparam ECB_TX_IDX       = 0;
localparam EVT_TX_IDX       = 1;
localparam ARP_PING_TX_IDX  = 3;

localparam W_NUM_HOST = (NUM_HOST==1) ? 1 : $clog2(NUM_HOST);

logic [NUM_LS_TX_INST-1:0]    lso_axis_tvalid;
logic [NUM_LS_TX_INST-1:0]    lso_axis_tlast ;
logic [AXI_LS_DWIDTH-1:0]     lso_axis_tdata  [NUM_LS_TX_INST];
logic [KEEP_LS_WIDTH-1:0]     lso_axis_tkeep  [NUM_LS_TX_INST];
logic [W_NUM_HOST-1   :0]     lso_axis_tuser  [NUM_LS_TX_INST];
logic [NUM_LS_TX_INST-1:0]    lso_axis_tready;

//Header Data
logic   [47:0]   host_mac_addr     [NUM_LS_TX_INST];
logic   [31:0]   host_ip_addr      [NUM_LS_TX_INST];
logic   [15:0]   host_udp_port     [NUM_LS_TX_INST];
logic   [15:0]   dev_udp_port      [NUM_LS_TX_INST];
logic   [15:0]   pld_len           [NUM_LS_TX_INST];

//------------------------------------------------------------------------------------------------//
// Input Round Robin ARB
//------------------------------------------------------------------------------------------------//


logic [NUM_LS_RX_INST-1:0]    lsi_axis_tvalid;
logic                         lsi_axis_tlast ;
logic [AXI_LS_DWIDTH-1:0]     lsi_axis_tdata;
logic [KEEP_LS_WIDTH-1:0]     lsi_axis_tkeep;
logic [W_NUM_HOST-1   :0]     lsi_axis_tuser;
logic [NUM_LS_RX_INST-1:0]    lsi_axis_tready;

logic                         w_lsi_axis_tready;
logic [NUM_LS_RX_INST-1:0]    w_lsi_axis_tuser;
logic                         w_lsi_axis_tvalid;

axis_arb #(
  .N_INPT           ( NUM_HOST          ),
  .W_DATA           ( AXI_LS_DWIDTH     ),
  .W_USER           ( NUM_LS_RX_INST    ),
  .ARB_TYPE         ( "ROUND_ROBIN"     ),
  .SKID             ( '1                )
) u_inp_axis_arb (
  .i_clk            ( i_pclk             ),
  .i_rst            ( i_prst             ),
  .i_axis_tvalid    ( i_axis_tvalid      ),
  .i_axis_tlast     ( i_axis_tlast       ),
  .i_axis_tdata     ( i_axis_tdata       ),
  .i_axis_tkeep     ( i_axis_tkeep       ),
  .i_axis_tuser     ( i_axis_tuser       ),
  .o_axis_tready    ( o_axis_tready      ),
  .o_axis_idx       ( lsi_axis_tuser     ), // Host ARB index
  .o_axis_tvalid    ( w_lsi_axis_tvalid  ),
  .o_axis_tlast     ( lsi_axis_tlast     ),
  .o_axis_tdata     ( lsi_axis_tdata     ),
  .o_axis_tkeep     ( lsi_axis_tkeep     ),
  .o_axis_tuser     ( w_lsi_axis_tuser   ), // RX Inst ARB index
  .i_axis_tready    ( w_lsi_axis_tready  )
);

assign lsi_axis_tvalid = (w_lsi_axis_tvalid) ? w_lsi_axis_tuser : '0;
assign w_lsi_axis_tready = |(w_lsi_axis_tuser & lsi_axis_tready);
//------------------------------------------------------------------------------------------------//
// Ping - RX with TX Response
//------------------------------------------------------------------------------------------------//

logic arp_ping_axis_tready;

arp_ping_pkt_proc
#(
  .AXI_DWIDTH                   ( AXI_LS_DWIDTH                      ),
  .W_USER                       ( W_NUM_HOST                         ),
  .NUM_HOST                     ( NUM_HOST                           )
) u_arp_ping_pkt_proc (
  .i_pclk                       ( i_pclk                             ),
  .i_prst                       ( i_prst                             ),
  .i_ping_axis_tvalid           ( lsi_axis_tvalid [PING_RX_IDX]      ),
  .i_arp_axis_tvalid            ( lsi_axis_tvalid [ARP_RX_IDX]       ),
  .i_lpb_axis_tvalid            ( lsi_axis_tvalid [LOOPBACK_RX_IDX]  ),
  .i_axis_tdata                 ( lsi_axis_tdata                     ),
  .i_axis_tlast                 ( lsi_axis_tlast                     ),
  .i_axis_tuser                 ( lsi_axis_tuser                     ),
  .i_axis_tkeep                 ( lsi_axis_tkeep                     ),
  .o_axis_tready                ( arp_ping_axis_tready               ),
  .i_dev_mac_addr               ( i_dev_mac_addr                     ),
  .o_axis_tvalid                ( lso_axis_tvalid [ARP_PING_TX_IDX]  ),
  .o_axis_tdata                 ( lso_axis_tdata  [ARP_PING_TX_IDX]  ),
  .o_axis_tlast                 ( lso_axis_tlast  [ARP_PING_TX_IDX]  ),
  .o_axis_tuser                 ( lso_axis_tuser  [ARP_PING_TX_IDX]  ),
  .o_axis_tkeep                 ( lso_axis_tkeep  [ARP_PING_TX_IDX]  ),
  .i_axis_tready                ( lso_axis_tready [ARP_PING_TX_IDX]  )
);

assign lsi_axis_tready [PING_RX_IDX]     = arp_ping_axis_tready;
assign lsi_axis_tready [ARP_RX_IDX]      = arp_ping_axis_tready;
assign lsi_axis_tready [LOOPBACK_RX_IDX] = arp_ping_axis_tready;

assign host_mac_addr[ARP_PING_TX_IDX] = '0;
assign host_ip_addr [ARP_PING_TX_IDX] = '0;
assign host_udp_port[ARP_PING_TX_IDX] = '0;
assign dev_udp_port [ARP_PING_TX_IDX] = '0;
assign pld_len      [ARP_PING_TX_IDX] = '0;

//------------------------------------------------------------------------------------------------//
// ECB - RX with TX Response
//------------------------------------------------------------------------------------------------//

logic                     ecb_is_roce;

ecb_rdwr_ctrl #(
  .AXI_DWIDTH         ( AXI_LS_DWIDTH                                                    ),
  .W_USER             ( W_NUM_HOST                                                       ),
  .MTU                ( MTU                                                              ),
  .SYNC_CLK           ( SYNC_CLK                                                         )
) ecb_rdwr_ctrl_inst (
  //Clocks and resets
  .i_pclk             ( i_pclk                                                           ),
  .i_prst             ( i_prst                                                           ),
  .i_aclk             ( i_aclk                                                           ),
  .i_arst             ( i_arst                                                           ),
  .i_apb_m2s          ( i_apb_m2s_ecb                                                    ),
  .o_apb_s2m          ( o_apb_s2m_ecb                                                    ),
  // Input AXIS interface
  .i_axis_tvalid      ( {lsi_axis_tvalid [ROCE_ECB_RX_IDX],lsi_axis_tvalid [ECB_RX_IDX]} ),
  .i_axis_tdata       ( lsi_axis_tdata                                                   ),
  .i_axis_tlast       ( lsi_axis_tlast                                                   ),
  .i_axis_tuser       ( lsi_axis_tuser                                                   ),
  .i_axis_tkeep       ( lsi_axis_tkeep                                                   ),
  .o_axis_tready      ( {lsi_axis_tready [ROCE_ECB_RX_IDX],lsi_axis_tready [ECB_RX_IDX]} ),
  // HDR info
  .o_host_mac_addr    ( host_mac_addr [ECB_TX_IDX]                                       ),
  .o_host_ip_addr     ( host_ip_addr  [ECB_TX_IDX]                                       ),
  .o_host_udp_port    ( host_udp_port [ECB_TX_IDX]                                       ),
  .o_pld_len          ( pld_len       [ECB_TX_IDX]                                       ),
  .o_dev_udp_port     ( dev_udp_port  [ECB_TX_IDX]                                       ),
  //APB Control Interface
  .o_apb_m2s          ( o_apb_m2s_ecb                                                    ),
  .i_apb_s2m          ( i_apb_s2m_ecb                                                    ),
  //ECB Response Interface
  .o_axis_tvalid      ( lso_axis_tvalid [ECB_TX_IDX]                                     ),
  .o_axis_tdata       ( lso_axis_tdata  [ECB_TX_IDX]                                     ),
  .o_axis_tlast       ( lso_axis_tlast  [ECB_TX_IDX]                                     ),
  .o_axis_tuser       ( lso_axis_tuser  [ECB_TX_IDX]                                     ),
  .o_axis_tkeep       ( lso_axis_tkeep  [ECB_TX_IDX]                                     ),
  .o_axis_is_roce     ( ecb_is_roce                                                      ),
  .i_axis_tready      ( lso_axis_tready [ECB_TX_IDX]                                     )
);


//------------------------------------------------------------------------------------------------//
// BOOTP - RX and TX
//------------------------------------------------------------------------------------------------//

bootp #(
  .IP_DEFAULT      ( 32'hC0A80002              ),
  .AXI_DWIDTH      ( AXI_LS_DWIDTH             ),
  .ENUM_DWIDTH     ( ENUM_DWIDTH               ),
  .NUM_HOST        ( NUM_HOST                  ),
  .UUID            ( UUID                      )
) u_bootp (
  .i_clk                ( i_pclk                               ),
  .i_rst                ( i_prst                               ),
  .i_init               ( i_init_done                          ),
  .i_ptp_clk            ( i_ptp_clk                            ),
  .i_ptp_rst            ( i_ptp_rst                            ),
  .i_ptp                ( i_ptp                                ),
  .i_pps                ( i_pps                                ),
  .i_enum_data          ( i_enum_data                          ),
  .i_pkt_inc            ( i_pkt_inc                            ),
  .i_hsb_stat           ( i_hsb_stat                           ),
  .i_dev_mac_addr       ( i_dev_mac_addr                       ),
  .i_eeprom_ip_addr     ( i_eeprom_ip_addr                     ),
  .i_eeprom_ip_addr_vld ( i_eeprom_ip_addr_vld                 ),
  .o_dev_ip_addr        ( o_dev_ip_addr                        ),
  .o_host_mac_addr      ( host_mac_addr         [BOOTP_TX_IDX] ),
  .o_host_ip_addr       ( host_ip_addr          [BOOTP_TX_IDX] ),
  .o_host_udp_port      ( host_udp_port         [BOOTP_TX_IDX] ),
  .o_dev_udp_port       ( dev_udp_port          [BOOTP_TX_IDX] ),
  .o_pld_len            ( pld_len               [BOOTP_TX_IDX] ),
  .i_axis_tvalid        ( lsi_axis_tvalid       [BOOTP_RX_IDX] ),
  .i_axis_tdata         ( lsi_axis_tdata                       ),
  .i_axis_tlast         ( lsi_axis_tlast                       ),
  .i_axis_tkeep         ( lsi_axis_tkeep                       ),
  .i_axis_tuser         ( lsi_axis_tuser                       ),
  .o_axis_tready        ( lsi_axis_tready       [BOOTP_RX_IDX] ),
  .o_axis_tvalid        ( lso_axis_tvalid       [BOOTP_TX_IDX] ),
  .o_axis_tdata         ( lso_axis_tdata        [BOOTP_TX_IDX] ),
  .o_axis_tlast         ( lso_axis_tlast        [BOOTP_TX_IDX] ),
  .o_axis_tkeep         ( lso_axis_tkeep        [BOOTP_TX_IDX] ),
  .o_axis_tuser         ( lso_axis_tuser        [BOOTP_TX_IDX] ),
  .i_axis_tready        ( lso_axis_tready       [BOOTP_TX_IDX] )
);


//------------------------------------------------------------------------------------------------//
// EVT - TX Only
//------------------------------------------------------------------------------------------------//

ctrl_bus_evt_int #(
  .AXI_DWIDTH            ( AXI_LS_DWIDTH                    ),
  .NUM_HOST              ( NUM_HOST                         ),
  .W_EVENT               ( 32                               ),
  .SYNC_CLK              ( SYNC_CLK                         )
) u_evt_int (
  .pclk                  ( i_pclk                           ),
  .rst                   ( i_prst                           ),
  .aclk                  ( i_aclk                           ),
  .arst                  ( i_arst                           ),
  .i_apb_m2s             ( i_apb_m2s_evt                    ),
  .o_apb_s2m             ( o_apb_s2m_evt                    ),
  .i_apb_m2s_ram         ( i_apb_m2s_ram                    ),
  .o_apb_s2m_ram         ( o_apb_s2m_ram                    ),
  .o_apb_m2s             ( o_apb_m2s_evt                    ),
  .i_apb_s2m             ( i_apb_s2m_evt                    ),
  .i_ptp                 ( i_ptp_hif                        ),
  .evt_vec               ( i_evt_vec                        ),
  .o_host_mac_addr       ( host_mac_addr       [EVT_TX_IDX] ),
  .o_host_ip_addr        ( host_ip_addr        [EVT_TX_IDX] ),
  .o_host_udp_port       ( host_udp_port       [EVT_TX_IDX] ),
  .o_dev_udp_port        ( dev_udp_port        [EVT_TX_IDX] ),
  .o_pld_len             ( pld_len             [EVT_TX_IDX] ),
  .o_int_axis_tx_tvalid  ( lso_axis_tvalid     [EVT_TX_IDX] ),
  .o_int_axis_tx_tdata   ( lso_axis_tdata      [EVT_TX_IDX] ),
  .o_int_axis_tx_tlast   ( lso_axis_tlast      [EVT_TX_IDX] ),
  .o_int_axis_tx_tuser   ( lso_axis_tuser      [EVT_TX_IDX] ),
  .o_int_axis_tx_tkeep   ( lso_axis_tkeep      [EVT_TX_IDX] ),
  .i_int_axis_tx_tready  ( lso_axis_tready     [EVT_TX_IDX] )
);


//------------------------------------------------------------------------------------------------//
// Output Round Robin ARB
//------------------------------------------------------------------------------------------------//

logic                      out_axis_tvalid;
logic                      out_axis_tlast;
logic [AXI_LS_DWIDTH-1:0]  out_axis_tdata;
logic [KEEP_LS_WIDTH-1:0]  out_axis_tkeep;
logic [1:0]                out_axis_tuser;
logic [W_NUM_HOST-1:0]     out_axis_thost_idx;
logic                      out_axis_tready;
logic                      w_out_axis_tready;
logic                      hdr_crc;
logic                      hdr_out_crc;

axis_arb #(
  .N_INPT           ( NUM_LS_TX_INST    ),
  .W_DATA           ( AXI_LS_DWIDTH     ),
  .W_USER           ( W_NUM_HOST        ),
  .ARB_TYPE         ( "ROUND_ROBIN"     ),
  .SKID             ( '1                )
) u_out_axis_arb (
  .i_clk            ( i_pclk             ),
  .i_rst            ( i_prst             ),
  .i_axis_tvalid    ( lso_axis_tvalid    ),
  .i_axis_tlast     ( lso_axis_tlast     ),
  .i_axis_tdata     ( lso_axis_tdata     ),
  .i_axis_tkeep     ( lso_axis_tkeep     ),
  .i_axis_tuser     ( lso_axis_tuser     ),
  .o_axis_tready    ( lso_axis_tready    ),
  .o_axis_idx       ( out_axis_tuser     ), // Send IDX over user
  .o_axis_tvalid    ( out_axis_tvalid    ),
  .o_axis_tlast     ( out_axis_tlast     ),
  .o_axis_tdata     ( out_axis_tdata     ),
  .o_axis_tkeep     ( out_axis_tkeep     ),
  .o_axis_tuser     ( out_axis_thost_idx ),
  .i_axis_tready    ( w_out_axis_tready  )
);

assign hdr_crc = (out_axis_tuser == ECB_TX_IDX) ? ecb_is_roce : '0;

//------------------------------------------------------------------------------------------------//
// Eth Header added when Needed
//------------------------------------------------------------------------------------------------//

logic [47:0] hif_mac_addr;
logic [31:0] hif_ip_addr;
logic [15:0] hif_udp_port;
logic [47:0] hdr_dev_mac_addr;
logic [31:0] hdr_dev_ip_addr;
logic [15:0] hdr_dev_udp_port;
logic [15:0] hdr_pld_len;


assign hif_mac_addr      = host_mac_addr [out_axis_tuser];
assign hif_ip_addr       = host_ip_addr  [out_axis_tuser];
assign hif_udp_port      = host_udp_port [out_axis_tuser];
assign hdr_pld_len       = pld_len [out_axis_tuser];
assign hdr_dev_mac_addr = i_dev_mac_addr[out_axis_thost_idx];
assign hdr_dev_ip_addr  = o_dev_ip_addr[out_axis_thost_idx];
assign hdr_dev_udp_port = dev_udp_port [out_axis_tuser];


//------------------------------------------------------------------------------------------------//
// UDP Header FSM
//------------------------------------------------------------------------------------------------//
// UDP
localparam HEADER_WIDTH = 336;

logic [15:0] udp_len;
logic [15:0] udp_chksum;

assign udp_len    = hdr_pld_len + 8;
assign udp_chksum = 16'h0; // set to 0 if unused

// IPV4 Header
logic [ 3:0] ipv4_version;
logic [ 3:0] ipv4_ihl;
logic [ 5:0] ipv4_dscp;
logic [ 1:0] ipv4_ecn;
logic [15:0] ipv4_len;
logic [15:0] ipv4_id;
logic [ 2:0] ipv4_flag;
logic [12:0] ipv4_offset;
logic [ 7:0] ipv4_ttl;
logic [ 7:0] ipv4_protocol;
logic [15:0] ipv4_chksum;
logic [31:0] ipv4_src_addr;
logic [31:0] ipv4_dst_addr;
logic [31:0] w_ipv4_dst_addr;

assign ipv4_version  = 4'h4;
assign ipv4_ihl      = 4'h5;
assign ipv4_dscp     = 6'h0;
assign ipv4_ecn      = 2'h0;
assign ipv4_len      = udp_len + 20;
assign ipv4_id       = 16'h0000;
assign ipv4_flag     = 3'h2;
assign ipv4_offset   = 13'h0;
assign ipv4_ttl      = 8'h40;
assign ipv4_protocol = 8'h11;
assign ipv4_src_addr = hdr_dev_ip_addr;
assign ipv4_dst_addr = hif_ip_addr;

// Ethernet Header

logic [15:0] eth_type;

assign eth_type     = 16'h0800; // IPv4 EtherType

logic [HEADER_WIDTH-1:0] header;
logic [HEADER_WIDTH-1:0] header_be;

assign header = { // Ethernet Header 14 Bytes
                  hif_mac_addr, hdr_dev_mac_addr, eth_type,
                  // IPv4 Header 20 Bytes
                  ipv4_version, ipv4_ihl, ipv4_dscp, ipv4_ecn, ipv4_len,
                  ipv4_id, ipv4_flag, ipv4_offset,
                  ipv4_ttl, ipv4_protocol, ipv4_chksum,
                  ipv4_src_addr,
                  ipv4_dst_addr,
                  // UDP Header 8 bytes
                  hdr_dev_udp_port, hif_udp_port, udp_len, udp_chksum};

genvar j;
generate
  for (j=0; j<HEADER_WIDTH/8; j++) begin
    assign header_be[j*8+:8] = header[(HEADER_WIDTH/8-1-j)*8+:8];
  end
endgenerate


logic [2:0] ipv4_chksum_state;
logic       w_out_axis_tvalid;

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    ipv4_chksum_state <= '0;
  end
  else begin
    if (out_axis_tvalid) begin
      if (ipv4_chksum_state[2]) begin
        ipv4_chksum_state <= (out_axis_tlast && out_axis_tready) ? '0 : ipv4_chksum_state;
      end
      else begin
        ipv4_chksum_state <= ipv4_chksum_state + 1;
      end
    end
    else begin
      ipv4_chksum_state <= ipv4_chksum_state;
    end
  end
end

assign w_out_axis_tvalid = ipv4_chksum_state[2] && out_axis_tvalid;
assign w_out_axis_tready = ipv4_chksum_state[2] && out_axis_tready;

logic [18:0] ipv4_chksum_calc [6];

always_ff @(posedge i_pclk) begin
  // C0
  ipv4_chksum_calc[0] <= ipv4_src_addr[31:16] + ipv4_src_addr[15: 0];
  ipv4_chksum_calc[1] <= 16'hC52D            + hdr_pld_len;
  ipv4_chksum_calc[2] <= ipv4_dst_addr[15:0] + ipv4_dst_addr[31:16];
  // C1
  ipv4_chksum_calc[3] <= ipv4_chksum_calc[0] + ipv4_chksum_calc[1];
  // C2
  ipv4_chksum_calc[4] <= ipv4_chksum_calc[2] + ipv4_chksum_calc[3];
  // C3
  ipv4_chksum_calc[5] <= ipv4_chksum_calc[4][15:0] + ipv4_chksum_calc[4][18:16];
end

assign ipv4_chksum = ~(ipv4_chksum_calc[5][15:0] + ipv4_chksum_calc[5][16]);

//------------------------------------------------------------------------------------------------//
// AXIS Append
//------------------------------------------------------------------------------------------------//

logic                     hdr_axis_tvalid;
logic                     hdr_axis_tlast;
logic [AXI_LS_DWIDTH-1:0] hdr_axis_tdata;
logic [KEEP_LS_WIDTH-1:0] hdr_axis_tkeep;
logic [W_NUM_HOST-1:0]    hdr_axis_tuser;
logic                     hdr_axis_tready;
logic                     hdr_en;

axis_hdr #(
  .HDR_WIDTH         ( 336                              ),
  .DWIDTH            ( AXI_LS_DWIDTH                    ),
  .W_USER            ( W_NUM_HOST + 1                   )
) u_axis_hdr (
  .clk               ( i_pclk                           ),
  .rst               ( i_prst                           ),
  .i_hdr             ( header_be                        ),
  .i_hdr_en          ( hdr_en                           ),
  .i_axis_rx_tvalid  ( w_out_axis_tvalid                ),
  .i_axis_rx_tdata   ( out_axis_tdata                   ),
  .i_axis_rx_tlast   ( out_axis_tlast                   ),
  .i_axis_rx_tuser   ( {hdr_crc,out_axis_thost_idx}     ),
  .i_axis_rx_tkeep   ( out_axis_tkeep                   ),
  .o_axis_rx_tready  ( out_axis_tready                  ),
  .o_axis_tx_tvalid  ( hdr_axis_tvalid                  ),
  .o_axis_tx_tdata   ( hdr_axis_tdata                   ),
  .o_axis_tx_tlast   ( hdr_axis_tlast                   ),
  .o_axis_tx_tuser   ( {hdr_out_crc,hdr_axis_tuser}     ), 
  .o_axis_tx_tkeep   ( hdr_axis_tkeep                   ),
  .i_axis_tx_tready  ( hdr_axis_tready                  )
);

assign hdr_en = (out_axis_tuser != ARP_PING_TX_IDX);

//------------------------------------------------------------------------------------------------//
// RoCE CRC
//------------------------------------------------------------------------------------------------//

  logic                     crc_axis_tvalid; 
  logic                     crc_axis_tlast; 
  logic [AXI_LS_DWIDTH-1:0] crc_axis_tdata; 
  logic [KEEP_LS_WIDTH-1:0] crc_axis_tkeep; 
  logic [W_NUM_HOST-1:0]    crc_axis_tuser; 
  logic                     crc_axis_tready;
  
  roce_icrc #(
    .W_DATA                 ( AXI_LS_DWIDTH        ),
    .W_USER                 ( W_NUM_HOST           )
  ) u_roce_icrc (
    .pclk                   ( i_pclk                ),
    .prst                   ( i_prst                ),
    .i_crc_en               ( hdr_out_crc           ),
    .i_axis_rx_tvalid       ( hdr_axis_tvalid       ),
    .i_axis_rx_tlast        ( hdr_axis_tlast        ), 
    .i_axis_rx_tkeep        ( hdr_axis_tkeep        ), 
    .i_axis_rx_tdata        ( hdr_axis_tdata        ), 
    .i_axis_rx_tuser        ( hdr_axis_tuser        ), 
    .o_axis_rx_tready       ( hdr_axis_tready       ),
    .o_axis_tx_tvalid       ( crc_axis_tvalid       ),
    .o_axis_tx_tlast        ( crc_axis_tlast        ),    
    .o_axis_tx_tkeep        ( crc_axis_tkeep        ), 
    .o_axis_tx_tdata        ( crc_axis_tdata        ), 
    .o_axis_tx_tuser        ( crc_axis_tuser        ), 
    .i_axis_tx_tready       ( crc_axis_tready       ) 
  );


//------------------------------------------------------------------------------------------------//
// High Speed Packet Buffer
//------------------------------------------------------------------------------------------------//

logic                         wout_axis_tvalid;
logic                         wout_axis_tlast ;
logic [AXI_DWIDTH-1:0]        wout_axis_tdata ;
logic [KEEP_WIDTH-1:0]        wout_axis_tkeep ;
logic [W_NUM_HOST-1:0]        wout_axis_thost_idx ;
logic                         wout_axis_tready;


axis_buffer # (
  .IN_DWIDTH  ( AXI_LS_DWIDTH  ),
  .OUT_DWIDTH ( AXI_DWIDTH     ),
  .WAIT2SEND  ( 1              ),
  .BUF_DEPTH  ( FIFO_DEPTH     ),
  .W_USER     ( W_NUM_HOST     )
) u_axis_buffer (
  .in_clk            ( i_pclk                                ),
  .in_rst            ( i_prst                                ),
  .out_clk           ( i_pclk                                ),
  .out_rst           ( i_prst                                ),
  .i_axis_rx_tvalid  ( crc_axis_tvalid                       ),
  .i_axis_rx_tdata   ( crc_axis_tdata                        ),
  .i_axis_rx_tlast   ( crc_axis_tlast                        ),
  .i_axis_rx_tuser   ( crc_axis_tuser                        ),
  .i_axis_rx_tkeep   ( crc_axis_tkeep                        ),
  .o_axis_rx_tready  ( crc_axis_tready                       ),
  .o_fifo_aempty     (                                       ),
  .o_fifo_afull      (                                       ),
  .o_fifo_empty      (                                       ),
  .o_fifo_full       (                                       ),
  .o_axis_tx_tvalid  ( wout_axis_tvalid                      ),
  .o_axis_tx_tdata   ( o_axis_tdata                          ),
  .o_axis_tx_tlast   ( o_axis_tlast                          ),
  .o_axis_tx_tuser   ( wout_axis_thost_idx                   ),
  .o_axis_tx_tkeep   ( o_axis_tkeep                          ),
  .i_axis_tx_tready  ( wout_axis_tready                      )
);

integer i;
always_comb begin
  o_axis_tvalid = '{default:0};
  wout_axis_tready = '0;
  for (i=0;i<NUM_HOST;i=i+1) begin
    if ((i == wout_axis_thost_idx)) begin
      o_axis_tvalid[i] = wout_axis_tvalid;
      wout_axis_tready = i_axis_tready[i];
    end
  end
end

assign o_axis_tuser = '0;

endmodule




