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

module bootp
  import HOLOLINK_pkg::*;
#(
  parameter  IP_DEFAULT  = 0,
  parameter  AXI_DWIDTH  = 64,
  parameter  ENUM_DWIDTH = 296,
  parameter  NUM_HOST    = 1,
  parameter  UUID        = 128'h0000_0000_0000_0000_0060_0DCA_FEC0_FFEE,
  localparam W_NUM_HOST  = (NUM_HOST==1) ? 1 : $clog2(NUM_HOST),
  localparam KEEP_WIDTH  = AXI_DWIDTH/8
)(
  input                    i_clk,
  input                    i_rst,

  input                    i_init,

  input                    i_ptp_clk,
  input                    i_ptp_rst,
  input                    i_pps,
  input  [79:0]            i_ptp,
  input  [NUM_HOST-1:0]    i_pkt_inc,
  input  [ENUM_DWIDTH-1:0] i_enum_data,
  input  [7:0]             i_hsb_stat,

  output [47:0]            o_host_mac_addr,
  output [31:0]            o_host_ip_addr,
  output [15:0]            o_host_udp_port,
  input  [47:0]            i_dev_mac_addr [NUM_HOST],
  input  [31:0]            i_eeprom_ip_addr,
  input                    i_eeprom_ip_addr_vld,
  output [31:0]            o_dev_ip_addr  [NUM_HOST],
  output [15:0]            o_dev_udp_port,
  output [15:0]            o_pld_len,

  input                    i_axis_tvalid,
  input  [AXI_DWIDTH-1:0]  i_axis_tdata,
  input                    i_axis_tlast,
  input  [KEEP_WIDTH-1:0]  i_axis_tkeep,
  input  [W_NUM_HOST-1:0]  i_axis_tuser,
  output                   o_axis_tready,

  output                   o_axis_tvalid,
  output [AXI_DWIDTH-1:0]  o_axis_tdata,
  output                   o_axis_tlast,
  output [KEEP_WIDTH-1:0]  o_axis_tkeep,
  output [W_NUM_HOST-1:0]  o_axis_tuser,
  input                    i_axis_tready
);

logic [W_NUM_HOST-1:0] inp_host_idx;
assign inp_host_idx = i_axis_tuser;
logic [W_NUM_HOST-1:0] out_host_idx;

logic [47:0]  dev_mac_addr [NUM_HOST];
logic [31:0]  r_eeprom_ip_addr [NUM_HOST];
logic         bootp_done;
logic [79:0]  ptp_sync;

//------------------------------------------------------------------------------------------------//
// BOOTP request sequence transition
//------------------------------------------------------------------------------------------------//

logic [15:0] pld /* synthesis syn_keep=1 */;

enum logic [1:0] {IDLE, BOOTP_REQ, BOOTP_RPT,BOOTP_WAIT} state;
// BOOTP request packet
logic         dev_ip_valid;
logic [63:0]  bootp_req [5];
logic [31:0]  ciaddr [NUM_HOST];
logic [271:0] axis_data;
logic         axis_is_busy;
logic         axis_vend_is_busy;
logic         trigger;
logic [6:0]   pps_cnt;
logic         ptp_sync_vld;
logic [15:0]  psn;
logic         eeprom_ip_addr_vld_ff;
logic         eeprom_ip_addr_vld_posedge;

always_ff @(posedge i_clk) begin
  if (i_rst) begin
    eeprom_ip_addr_vld_ff <= 1'b0;
    pld                   <= '0;
  end
  else begin
    eeprom_ip_addr_vld_ff <= i_eeprom_ip_addr_vld;
    pld                   <= 16'h12C;
  end
end
assign eeprom_ip_addr_vld_posedge = i_eeprom_ip_addr_vld && !eeprom_ip_addr_vld_ff;

// Endianness conversion
logic [47:0] chaddr [NUM_HOST];

always_ff @(posedge i_clk) begin
  for (int k=0; k<NUM_HOST; k++) begin
    r_eeprom_ip_addr[k][31:8] <= i_eeprom_ip_addr[31:8];
    r_eeprom_ip_addr[k][7:0]  <= i_eeprom_ip_addr[7:0] + k;
  end
end

always_comb begin
  for (int k=0; k<NUM_HOST; k++) begin
    dev_mac_addr[k] = i_dev_mac_addr[k];
  end
end


//------------------------------------------------------------------------------------------------//
// Packet counter
//------------------------------------------------------------------------------------------------//

logic [23:0] pkt_cnt [NUM_HOST];
genvar m;
generate
  for (m=0; m<NUM_HOST; m++) begin
    always_ff @(posedge i_clk) begin
      if (i_rst) begin
        pkt_cnt[m] <= '0;
      end
      else begin
        pkt_cnt[m] <= i_pkt_inc[m] ? pkt_cnt[m] + 1'b1 : pkt_cnt[m];
      end
    end
  end
endgenerate


//------------------------------------------------------------------------------------------------//
// PTP CDC
//------------------------------------------------------------------------------------------------//
logic        ptp_fifo_rden;
logic [15:0] ptp_fifo_rddata;
logic [15:0] ptp_fifo_q;
logic        ptp_fifo_rdval;
logic        ptp_fifo_empty;

reg_cdc # (
  .NBITS         ( 80     ),
  .REG_RST_VALUE ( '0     )
) u_ctrl_cdc_apb (
  .i_a_clk ( i_ptp_clk    ),
  .i_a_rst ( i_ptp_rst    ),
  .i_a_val ( i_pps        ),
  .i_a_reg ( i_ptp        ),
  .i_b_clk ( i_clk        ),
  .i_b_rst ( i_rst        ),
  .o_b_val ( ptp_sync_vld ),
  .o_b_reg ( ptp_sync     )
);

localparam MAC_BYTE = 6;
genvar i,j;
generate
  for (j=0; j<NUM_HOST; j++) begin
    for (i=0; i<MAC_BYTE; i++) begin
      assign chaddr[j][i*8+:8] = dev_mac_addr[j][(MAC_BYTE-1-i)*8+:8];
    end
  end
endgenerate

localparam IP_BYTE = 4;
logic [31:0] ip_addr [NUM_HOST];
generate
  for (j=0;j<NUM_HOST;j++) begin
    logic [31:0] ip_default;
    assign ip_default = IP_DEFAULT + j;
    for (i=0; i<IP_BYTE; i++) begin
      assign ip_addr[j][i*8+:8] = ip_default[(IP_BYTE-1-i)*8+:8];
    end
  end
endgenerate

logic [31:0] eeprom_ip_addr [NUM_HOST];
generate
  for (i=0; i<NUM_HOST; i++) begin
    for (j=0; j<IP_BYTE; j++) begin
      assign eeprom_ip_addr[i][j*8+:8] = r_eeprom_ip_addr[i][(IP_BYTE-1-j)*8+:8];
    end
  end
endgenerate

logic [7:0] eth_port;
assign      eth_port = out_host_idx;

logic [31:0] xid;
assign       xid = 32'd0;

localparam ID_BYTE = 4;
logic [31:0] dev_id;
logic [31:0] ID;
assign ID = out_host_idx;

generate
  for (i=0; i<ID_BYTE; i++) begin
    assign dev_id[i*8+:8] = ID[(ID_BYTE-1-i)*8+:8];
  end
endgenerate

localparam PTP_BYTE = 10;
logic [79:0] ptp_be;
generate
  for (i=0; i<PTP_BYTE; i++) begin
    assign ptp_be[i*8+:8] = ptp_sync[(PTP_BYTE-1-i)*8+:8];
  end
endgenerate

logic [15:0] time_sec;
assign time_sec = ptp_be[32+:16];

localparam PSN_BYTE = 2;
logic [15:0] psn_be;
generate
  for (i=0; i<PSN_BYTE; i++) begin
    assign psn_be[i*8+:8] = psn[(PSN_BYTE-1-i)*8+:8];
  end
endgenerate

localparam UUID_BYTE = 16;
logic [127:0] uuid_be;
generate
  for (i=0; i<UUID_BYTE; i++) begin
    assign uuid_be[i*8+:8] = UUID[(UUID_BYTE-1-i)*8+:8];
  end
endgenerate

assign bootp_req = '{
// 7-4     3     2     1     0
// xid     hops  hlen  htype opcode
  {dev_id,    8'h0, 8'h6, 8'h1, 8'h1},
// 15-12   11-10     9-8
// ciaddr  flags     secs
  {ciaddr[out_host_idx], 16'h0100, time_sec},
// 23-20  19-16
// siaddr yiaddr
  {32'h0, 32'h0},
// 31-28         27-24
// chaddr        giaddr
  {chaddr[out_host_idx][31:0], 32'h0},
// 39-34  33-32
// chaddrpad    chaddr
  {48'h0,      chaddr[out_host_idx][47:32]}
};

// Add Enumeration Data to Vendor Data field
assign axis_data = {bootp_req[4][15:0],bootp_req[3],bootp_req[2],bootp_req[1],bootp_req[0]};


always_ff @(posedge i_clk) begin
  if (i_rst) begin
    state         <= IDLE;
    pps_cnt       <= 0;
    psn           <= 0;
    trigger       <= 0;
    out_host_idx  <= '0;
    bootp_done    <= '0;
  end
  else begin
    case (state)
      IDLE: begin
        if (i_init) begin
          state       <= BOOTP_REQ;
        end
      end
      BOOTP_REQ: begin
        trigger      <= 1'b1;
        bootp_done   <= (out_host_idx >= (NUM_HOST-1));
        state        <= BOOTP_WAIT;
      end
      BOOTP_WAIT: begin
        trigger        <= 1'b0;
        if (!axis_is_busy && !axis_vend_is_busy && !trigger) begin
          state        <= BOOTP_RPT;
          out_host_idx <= (out_host_idx >= (NUM_HOST-1)) ? '0 : out_host_idx + 1'b1;
          if (bootp_done) begin
            psn <= psn + 1'b1;;
          end
        end
      end
      BOOTP_RPT: begin
        trigger <= 1'b0;
        if (ptp_sync_vld || (!bootp_done)) begin
          state        <= BOOTP_REQ;
          bootp_done   <= '0;
        end
      end
      default: begin
        state <= IDLE;
      end
    endcase

  end
end


logic                  axis_tvalid;
logic [AXI_DWIDTH-1:0] axis_tdata;
logic                  axis_tlast;
logic [KEEP_WIDTH-1:0] axis_tkeep;
logic                  axis_tuser;

logic                  axis_vend_tvalid;
logic [AXI_DWIDTH-1:0] axis_vend_tdata;
logic                  axis_vend_tlast;
logic [KEEP_WIDTH-1:0] axis_vend_tkeep;
logic                  axis_vend_tuser;

vec_to_axis #(
  .AXI_DWIDTH       ( AXI_DWIDTH       ),
  .DATA_WIDTH       ( 272              ),
  .PADDED_WIDTH     ( (300-64)*8       )
) bootp_to_axis (
  .clk              ( i_clk            ),
  .rst              ( i_rst            ),
  .trigger          ( trigger          ),
  .data             ( axis_data        ),
  .is_busy          ( axis_is_busy     ),
  .o_axis_tx_tvalid ( axis_tvalid      ),
  .o_axis_tx_tdata  ( axis_tdata       ),
  .o_axis_tx_tlast  ( axis_tlast       ),
  .o_axis_tx_tuser  ( axis_tuser       ),
  .o_axis_tx_tkeep  ( axis_tkeep       ),
  .i_axis_tx_tready ( i_axis_tready    )
);

//BOOTP VENDOR FIELD
localparam BOOTP_VEND_PREFIX_BYTES = 8;

logic [7:0] tag_len;
assign tag_len = 1 + PSN_BYTE + PTP_BYTE + ENUM_DWIDTH/8 + BOOTP_VEND_PREFIX_BYTES-2;

logic [BOOTP_VEND_PREFIX_BYTES*8-1:0] bootp_vend_prefix;

assign bootp_vend_prefix = {
  8'h02,  // Enum Version
  eth_port,
  8'h41,  // "A"
  8'h44,  // "D"
  8'h56,  // "V"
  8'h4E,  // "N"
  tag_len,// LENGTH
  8'hE0   // TAG
};

localparam BOOTP_VEND_DATA_WIDTH = 8+(PSN_BYTE*8)+(PTP_BYTE*8)+ENUM_DWIDTH+(BOOTP_VEND_PREFIX_BYTES*8);

logic [23:0] pkt_cnt_be;
assign pkt_cnt_be = {pkt_cnt[out_host_idx][7:0],pkt_cnt[out_host_idx][15:8],pkt_cnt[out_host_idx][23:16]};

logic [BOOTP_VEND_DATA_WIDTH-1:0] bootp_vend_data;
assign bootp_vend_data = {
  i_hsb_stat, psn_be, ptp_be, i_enum_data[ENUM_DWIDTH-1:176],8'h0,pkt_cnt_be, uuid_be, 16'd0, bootp_vend_prefix
};

vec_to_axis #(
  .AXI_DWIDTH       ( AXI_DWIDTH                  ),
  .DATA_WIDTH       ( BOOTP_VEND_DATA_WIDTH       ),
  .PADDED_WIDTH     ( 64*8                        )
) bootp_vend_to_axis (
  .clk              ( i_clk                       ),
  .rst              ( i_rst                       ),
  .trigger          ( axis_tlast && i_axis_tready ),
  .data             ( bootp_vend_data             ),
  .is_busy          ( axis_vend_is_busy           ),
  .o_axis_tx_tvalid ( axis_vend_tvalid            ),
  .o_axis_tx_tdata  ( axis_vend_tdata             ),
  .o_axis_tx_tlast  ( axis_vend_tlast             ),
  .o_axis_tx_tuser  ( axis_vend_tuser             ),
  .o_axis_tx_tkeep  ( axis_vend_tkeep             ),
  .i_axis_tx_tready ( i_axis_tready               )
);

assign o_axis_tvalid = axis_tvalid | axis_vend_tvalid;
assign o_axis_tdata  = axis_tdata  | axis_vend_tdata;
assign o_axis_tlast  =               axis_vend_tlast;
assign o_axis_tuser  = out_host_idx;
assign o_axis_tkeep  = axis_tkeep  | axis_vend_tkeep;
assign o_pld_len     = pld;

//------------------------------------------------------------------------------------------------//
// BOOTP response decoding
//------------------------------------------------------------------------------------------------//

localparam   OPCODE_BEAT    = 1;
localparam   CIADDR_OFFSET  = 24;
localparam   BOOTP_BYTES    = 300;
localparam   BOOTP_BEAT     = ((BOOTP_BYTES%KEEP_WIDTH)==0) ? BOOTP_BYTES/KEEP_WIDTH -1 : BOOTP_BYTES/KEEP_WIDTH;
localparam   MAX_BEAT       = (BOOTP_BEAT > 1) ? $clog2(BOOTP_BEAT+1) : 1;

logic                bootp_check;
logic [ 8:0]         bootp_byte;
logic [ 7:0]         bootp_opcode;
logic                bootp_valid;
logic [31:0]         bootp_ciaddr;
logic [495:0]        bootp_hdr;
logic                bootp_hdr_valid;
logic [MAX_BEAT-1:0] beat_cnt;

assign o_axis_tready = 1'b1;

axis_to_vec #(
  .AXI_DWIDTH       ( AXI_DWIDTH          ),
  .DATA_WIDTH       ( 496                 )
) axis_to_ciaddr (
  .clk              ( i_clk               ),
  .rst              ( i_rst               ),
  .i_axis_rx_tvalid ( i_axis_tvalid       ),
  .i_axis_rx_tdata  ( i_axis_tdata        ),
  .i_axis_rx_tlast  ( i_axis_tlast        ),
  .i_axis_rx_tuser  (                     ),
  .i_axis_rx_tkeep  ( i_axis_tkeep        ),
  .o_axis_rx_tready (                     ),
  .i_done           ( '1                  ),
  .o_data           ( bootp_hdr           ),
  .o_valid          ( bootp_hdr_valid     ),
  .o_busy           (                     ),
  .o_byte_cnt       (                     )
);

always_ff @(posedge i_clk) begin
  if (i_rst) begin
    bootp_check     <= '0;
    bootp_byte      <= '0;
    bootp_opcode    <= '0;
    bootp_ciaddr    <= '0;
    bootp_valid     <= '0;
    ciaddr          <= ip_addr;
  end
  else begin
    // Latch BOOTP response opcode
    if (bootp_hdr_valid) begin
      bootp_opcode <= bootp_hdr[343:336];
    end
    // Latch BOOTP response ciaddr
    if (bootp_hdr_valid && (bootp_hdr[343:336] == 8'h2)) begin
      ciaddr[inp_host_idx]      <= bootp_hdr[495:464];
    end
    else if (eeprom_ip_addr_vld_posedge) begin
      for (int k=0; k<NUM_HOST; k++) begin
        ciaddr[k]      <= eeprom_ip_addr[k];
      end
    end
  end
end


assign o_host_mac_addr = '{default:1};
assign o_host_ip_addr  = '{default:1};
assign o_host_udp_port = 16'd12267;
assign o_dev_udp_port  = 16'd12268;

generate
  for (i=0; i<NUM_HOST; i++) begin
    assign o_dev_ip_addr[i]  = {ciaddr[i][7:0], ciaddr[i][15:8], ciaddr[i][23:16], ciaddr[i][31:24]};
  end
endgenerate

endmodule
