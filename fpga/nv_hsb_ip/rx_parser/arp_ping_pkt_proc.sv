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

//Input is full AXIS packet. Module will swap header fields as applicable, calculate the ping packet
//checksum, and put the packet into a FIFO buffer for hookup to the datapath.
module arp_ping_pkt_proc
  import rx_parser_pkg::*;
  import axis_pkg::*;
#(
  parameter   AXI_DWIDTH      = 8,
  parameter   KEEP_WIDTH      = (AXI_DWIDTH / 8),
  parameter   W_USER          = 1,
  parameter   NUM_HOST        = 1
)(
  input   logic                           i_pclk,
  input   logic                           i_prst,
  //AXIS input
  input   logic                           i_arp_axis_tvalid,
  input   logic                           i_ping_axis_tvalid,
  input   logic                           i_lpb_axis_tvalid,
  input   logic  [AXI_DWIDTH-1:0]         i_axis_tdata,
  input   logic                           i_axis_tlast,
  input   logic  [W_USER-1:0]             i_axis_tuser,
  input   logic  [KEEP_WIDTH-1:0]         i_axis_tkeep,
  output  logic                           o_axis_tready,
  //Header Input
  input   logic  [47:0]                   i_dev_mac_addr [NUM_HOST],
  //AXIS output
  output  logic                           o_axis_tvalid,
  output  logic  [AXI_DWIDTH-1:0]         o_axis_tdata,
  output  logic                           o_axis_tlast,
  output  logic  [W_USER-1:0]             o_axis_tuser,
  output  logic  [KEEP_WIDTH-1:0]         o_axis_tkeep,
  input   logic                           i_axis_tready
);


logic                    in_axis_tvalid;
logic  [AXI_DWIDTH-1:0]  in_axis_tdata;
logic                    in_axis_tlast;
logic  [W_USER-1:0]      in_axis_tuser;
logic  [KEEP_WIDTH-1:0]  in_axis_tkeep;
logic                    in_axis_tready;

logic                    w_axis_tvalid;
logic  [AXI_DWIDTH-1:0]  w_axis_tdata;
logic                    w_axis_tlast;
logic  [W_USER-1:0]      w_axis_tuser;
logic  [KEEP_WIDTH-1:0]  w_axis_tkeep;
logic                    w_axis_tready;

logic                    in_arp_axis_tvalid;
logic                    in_ping_axis_tvalid;
logic                    in_lpb_axis_tvalid;

logic                    inp_tvalid;

axis_reg # (
  .DWIDTH             ( AXI_DWIDTH + (AXI_DWIDTH/8) + W_USER + 1 + 3)
) u_axis_in_reg (
  .clk                ( i_pclk                                                                                                                  ),
  .rst                ( i_prst                                                                                                                  ),
  .i_axis_rx_tvalid   ( inp_tvalid                                                                                                              ),
  .i_axis_rx_tdata    ( {i_axis_tdata,i_axis_tlast,i_axis_tuser,i_axis_tkeep,i_arp_axis_tvalid,i_ping_axis_tvalid,i_lpb_axis_tvalid}            ),
  .o_axis_rx_tready   ( o_axis_tready                                                                                                           ),
  .o_axis_tx_tvalid   ( in_axis_tvalid                                                                                                          ),
  .o_axis_tx_tdata    ( {in_axis_tdata,in_axis_tlast,in_axis_tuser,in_axis_tkeep, in_arp_axis_tvalid, in_ping_axis_tvalid, in_lpb_axis_tvalid}  ),
  .i_axis_tx_tready   ( in_axis_tready                                                                                                          )
);

logic [47:0] dev_mac_addr [NUM_HOST];

always_ff @(posedge i_pclk) begin
  for (int k=0; k<NUM_HOST; k++) begin
    dev_mac_addr[k]  <= i_dev_mac_addr[k];
  end
end

assign inp_tvalid = i_ping_axis_tvalid || i_arp_axis_tvalid || i_lpb_axis_tvalid;

//------------------------------------------------------------------------------------------------//
// Flatten Header
//------------------------------------------------------------------------------------------------//

localparam ETH_BYTES            = 14;
localparam IP_BYTES             = 20;
localparam UDP_BYTES            = 8;
localparam HDR_BYTES            = ETH_BYTES + IP_BYTES + UDP_BYTES;

logic [(HDR_BYTES*8)-1:0] hdr;
logic [7:0]               hdr_array [HDR_BYTES];
logic                     hdr_valid;
logic                     hdr_done;
logic                     axis_rx_tready;

axis_to_vec #(
  .AXI_DWIDTH         ( AXI_DWIDTH          ),
  .DATA_WIDTH         ( HDR_BYTES*8         )
) axis_to_ciaddr (
  .clk              ( i_pclk              ),
  .rst              ( i_prst              ),
  .i_axis_rx_tvalid ( in_axis_tvalid      ),
  .i_axis_rx_tdata  ( in_axis_tdata       ),
  .i_axis_rx_tlast  ( in_axis_tlast       ),
  .i_axis_rx_tuser  ( '0                  ),
  .i_axis_rx_tkeep  ( in_axis_tkeep       ),
  .o_axis_rx_tready ( axis_rx_tready      ),
  .i_done           ( hdr_done            ),
  .o_data           ( hdr                 ),
  .o_busy           (                     ),
  .o_valid          ( hdr_valid           ),
  .o_byte_cnt       (                     )
);


assign hdr_done   = (w_axis_tvalid && w_axis_tlast && w_axis_tready);

genvar m;
generate
  for (m=0; m<HDR_BYTES; m=m+1) begin
    assign hdr_array[m] = hdr[m*8+:8];
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// MAP Header
//------------------------------------------------------------------------------------------------//

logic [(HDR_BYTES*8)-1:0] hdr_out_array;
logic [16:0]              ping_chksum;

//Map all the header bytes to the header byte array, swapping fields as applicable.

assign hdr_out_array[00*8+:8]  = hdr_array[06];
assign hdr_out_array[01*8+:8]  = hdr_array[07];
assign hdr_out_array[02*8+:8]  = hdr_array[08];
assign hdr_out_array[03*8+:8]  = hdr_array[09];
assign hdr_out_array[04*8+:8]  = hdr_array[10];
assign hdr_out_array[05*8+:8]  = hdr_array[11];
assign hdr_out_array[06*8+:8]  = dev_mac_addr[in_axis_tuser][47:40];
assign hdr_out_array[07*8+:8]  = dev_mac_addr[in_axis_tuser][39:32];
assign hdr_out_array[08*8+:8]  = dev_mac_addr[in_axis_tuser][31:24];
assign hdr_out_array[09*8+:8]  = dev_mac_addr[in_axis_tuser][23:16];
assign hdr_out_array[10*8+:8]  = dev_mac_addr[in_axis_tuser][15:08];
assign hdr_out_array[11*8+:8]  = dev_mac_addr[in_axis_tuser][07:00];
assign hdr_out_array[12*8+:8]  = hdr_array[12];
assign hdr_out_array[13*8+:8]  = hdr_array[13];
assign hdr_out_array[14*8+:8]  = hdr_array[14];
assign hdr_out_array[15*8+:8]  = hdr_array[15];
assign hdr_out_array[16*8+:8]  = hdr_array[16];
assign hdr_out_array[17*8+:8]  = hdr_array[17];
assign hdr_out_array[18*8+:8]  = hdr_array[18];
assign hdr_out_array[19*8+:8]  = hdr_array[19];
assign hdr_out_array[20*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[20]      : 8'h00;
assign hdr_out_array[21*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[21]      : 8'h02;
assign hdr_out_array[22*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[22]      : dev_mac_addr[in_axis_tuser][47:40];
assign hdr_out_array[23*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[23]      : dev_mac_addr[in_axis_tuser][39:32];
assign hdr_out_array[24*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[24]      : dev_mac_addr[in_axis_tuser][31:24];
assign hdr_out_array[25*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[25]      : dev_mac_addr[in_axis_tuser][23:16];
assign hdr_out_array[26*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[30]      : dev_mac_addr[in_axis_tuser][15:08];
assign hdr_out_array[27*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[31]      : dev_mac_addr[in_axis_tuser][07:00];
assign hdr_out_array[28*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[32]      : hdr_array[38];
assign hdr_out_array[29*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[33]      : hdr_array[39];
assign hdr_out_array[30*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[26]      : hdr_array[40];
assign hdr_out_array[31*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[27]      : hdr_array[41];
assign hdr_out_array[32*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[28]      : hdr_array[22];
assign hdr_out_array[33*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[29]      : hdr_array[23];
assign hdr_out_array[34*8+:8]  = in_ping_axis_tvalid ? '0                 : in_lpb_axis_tvalid ? hdr_array[36] : hdr_array[24]; //Change the ICMP Type to "Response" = 0;
assign hdr_out_array[35*8+:8]  = in_ping_axis_tvalid ? hdr_array[35]      : in_lpb_axis_tvalid ? hdr_array[37] : hdr_array[25];
assign hdr_out_array[36*8+:8]  = in_ping_axis_tvalid ? ping_chksum[15:08] : in_lpb_axis_tvalid ? hdr_array[34] : hdr_array[26]; // Add (ping_hdr_i.chksum + 16'h0800);
assign hdr_out_array[37*8+:8]  = in_ping_axis_tvalid ? ping_chksum[07:00] : in_lpb_axis_tvalid ? hdr_array[35] : hdr_array[27]; // Add (ping_hdr_i.chksum + 16'h0800);
assign hdr_out_array[38*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[38]      : hdr_array[28];
assign hdr_out_array[39*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[39]      : hdr_array[29];
assign hdr_out_array[40*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[40]      : hdr_array[30];
assign hdr_out_array[41*8+:8]  = (in_ping_axis_tvalid | in_lpb_axis_tvalid) ? hdr_array[41]      : hdr_array[31];

always_comb begin
  ping_chksum = {hdr_array[36],hdr_array[37]} + 16'h0800;
  ping_chksum = ping_chksum[16] + ping_chksum[15:0];
end


//------------------------------------------------------------------------------------------------//
// Vector to Axis
//------------------------------------------------------------------------------------------------//

logic                  hdr_axis_tvalid;
logic                  hdr_axis_tlast;
logic [AXI_DWIDTH-1:0] hdr_axis_tdata;
logic [KEEP_WIDTH-1:0] hdr_axis_tkeep;
logic                  hdr_axis_tuser;
logic                  hdr_axis_tready;

vec_to_axis #(
  .AXI_DWIDTH       ( AXI_DWIDTH       ),
  .DATA_WIDTH       ( HDR_BYTES*8      ),
  .REG_DATA         ( 0                )
) hdr_to_axis (
  .clk              ( i_pclk           ),
  .rst              ( i_prst           ),
  .trigger          ( hdr_valid        ),
  .data             ( hdr_out_array    ),
  .is_busy          (                  ),
  .o_axis_tx_tvalid ( hdr_axis_tvalid  ),
  .o_axis_tx_tdata  ( hdr_axis_tdata   ),
  .o_axis_tx_tlast  ( hdr_axis_tlast   ),
  .o_axis_tx_tuser  ( hdr_axis_tuser   ),
  .o_axis_tx_tkeep  ( hdr_axis_tkeep   ),
  .i_axis_tx_tready ( hdr_axis_tready  )
);

//------------------------------------------------------------------------------------------------//
// Output MUX
//------------------------------------------------------------------------------------------------//

logic hdr_active;
//Raise packet active after the first tvalid following a tlast.
always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    hdr_active <= 1'b1;
  end
  else begin
    if (!hdr_active) begin
      hdr_active <= (in_axis_tlast && (in_axis_tvalid) && in_axis_tready);
    end
    else begin
      hdr_active <= !(hdr_axis_tlast && hdr_axis_tvalid && w_axis_tready && !in_axis_tlast);
    end
  end
end


assign w_axis_tvalid = (hdr_active) ? hdr_axis_tvalid                  : in_axis_tvalid;
assign w_axis_tdata  = (hdr_active) ? hdr_axis_tdata                   : in_axis_tdata ;
assign w_axis_tlast  = (hdr_active) ? (in_axis_tlast && hdr_axis_tlast) : in_axis_tlast ;
assign w_axis_tuser  = in_axis_tuser;
assign w_axis_tkeep  = (hdr_active) ? hdr_axis_tkeep                   : in_axis_tkeep ;

assign hdr_axis_tready = w_axis_tready;
assign in_axis_tready   = (hdr_active) ? ((!hdr_valid && !in_axis_tlast) || (hdr_axis_tlast && in_axis_tlast)) : w_axis_tready;


//------------------------------------------------------------------------------------------------//
// Output REG
//------------------------------------------------------------------------------------------------//

axis_reg # (
  .DWIDTH             ( AXI_DWIDTH + (AXI_DWIDTH/8) + W_USER + 1)
) u_axis_reg (
  .clk                ( i_pclk                                                 ),
  .rst                ( i_prst                                                 ),
  .i_axis_rx_tvalid   ( w_axis_tvalid                                          ),
  .i_axis_rx_tdata    ( {w_axis_tdata,w_axis_tlast,w_axis_tuser,w_axis_tkeep}  ),
  .o_axis_rx_tready   ( w_axis_tready                                          ),
  .o_axis_tx_tvalid   ( o_axis_tvalid                                          ),
  .o_axis_tx_tdata    ( {o_axis_tdata,o_axis_tlast,o_axis_tuser,o_axis_tkeep}  ),
  .i_axis_tx_tready   ( i_axis_tready                                          )
);



endmodule
