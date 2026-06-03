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

module routing_table
  import ptp_pkg::*;
#(
  parameter N_INPT  = 2,  // Number of AXI stream inputs
  parameter N_HOST  = 2,  // Number of Host ports
  parameter W_DATA  = 64, // Width of AXI stream interfaces in bits, byte align
  parameter DEPTH   = 64,
  parameter W_KEEP  = W_DATA/8
)(
  input   logic                       i_clk,
  input   logic                       i_rst,

  // AXI inputs
  input   logic [N_INPT-1:0]          i_axis_tvalid,
  input   logic [N_INPT-1:0]          i_axis_tready,
  input   logic [N_INPT-1:0]          i_axis_tlast,
  input   logic [W_DATA-1:0]          i_axis_tdata  [N_INPT-1:0],
  input   logic [W_KEEP-1:0]          i_axis_tkeep  [N_INPT-1:0],
  input   logic [N_INPT-1:0]          i_axis_tuser,

  // Read Interface
  input   logic [47:0]                i_mac_addr    [N_HOST],
  input   logic [N_HOST-1:0]          i_mac_req,
  output  logic [N_INPT+N_HOST-1 :0]  o_dest_port,
  output  logic [N_HOST-1:0]          o_dest_val,

  // PTP Output
  output  logic [N_INPT-1:0]          o_chk_del_req,
  output  logic [N_INPT-1:0]          o_is_del_req

);

//------------------------------------------------------------------------------------------------//
// Parse incoming data for Source MAC addr
//------------------------------------------------------------------------------------------------//

logic  [N_INPT-1:0]      axis_handshake;
logic  [N_INPT-1:0]      hdr_valid, hdr_valid_r;
logic  [127:0]           hdr_data [N_INPT];
logic  [N_INPT-1:0]      mac_valid;
logic  [N_INPT-1:0]      mac_grant;
logic  [47:0]            mac_data  [N_INPT];

// AXIS Parser
genvar i;
generate
for (i=0;i<N_INPT;i=i+1) begin
  assign axis_handshake[i] = i_axis_tvalid[i] && i_axis_tready[i];

  axis_to_vec #(
    .AXI_DWIDTH       ( W_DATA               ),
    .DATA_WIDTH       ( 128                  )
  ) axis_to_destmac (
    .clk              ( i_clk                ),
    .rst              ( i_rst                ),
    .i_axis_rx_tvalid ( axis_handshake  [i]  ),
    .i_axis_rx_tdata  ( i_axis_tdata    [i]  ),
    .i_axis_rx_tlast  ( i_axis_tlast    [i]  ),
    .i_axis_rx_tuser  ( i_axis_tuser    [i]  ),
    .i_axis_rx_tkeep  ( i_axis_tkeep    [i]  ),
    .o_axis_rx_tready (                      ),
    .i_done           ( '1                   ),
    .o_data           ( hdr_data        [i]  ),
    .o_valid          ( hdr_valid       [i]  ),
    .o_byte_cnt       (                      )
  );
    always_ff @(posedge i_clk) begin
      if (i_rst) begin
        mac_data[i]      <= '0;
        mac_valid[i]     <= '0;
        hdr_valid_r[i]   <= '0;
        o_chk_del_req[i] <= '0;
        o_is_del_req [i] <= '0;
      end
      else begin
        mac_data [i]   <= ({hdr_valid_r[i],hdr_valid[i]}==2'b01) ? hdr_data[i][95:48] : mac_data[i];
        mac_valid[i]   <= (mac_grant[i])                         ? 1'b0 :
                          ({hdr_valid_r[i],hdr_valid[i]}==2'b01) ? 1'b1 :
                                                                 mac_valid[i];
        hdr_valid_r[i] <= hdr_valid[i];

        o_chk_del_req[i] <= ({hdr_valid_r[i],hdr_valid[i]}==2'b01) && (hdr_data[i][47:0]  == BE_PTP_FW_MULTI_ADDR);
        o_is_del_req[i]  <=  ((hdr_data[i][(12*8)+:16]  == BE_PTP_ETH_TYPE) &&      // Eth Type
                              (hdr_data[i][(14*8)+:4]   == MSG_DELAY_REQ))  ; // PTP Type
      end
  end

end
endgenerate

//------------------------------------------------------------------------------------------------//
// ARB MAC Source to write into CAM
//------------------------------------------------------------------------------------------------//

// RRARB, input Mac, output Dest
// Write when not reading

logic              cam_wr;
logic              cam_del;
logic [N_INPT-1:0] cam_wr_data;
logic [47:0]       cam_key;
logic [N_INPT-1:0] cam_rd_data;
logic              cam_data_val;
logic [N_HOST-1:0] dest_val;

logic              r_cam_wr;
logic [N_INPT-1:0] r_cam_wr_data;
logic [47:0]       r_cam_key;
logic [47:0]       rr_cam_key;
logic [N_HOST-1:0] r_dest_val;
logic [N_HOST-1:0] rr_dest_val;

logic              rrarb_idle;

rrarb #(
  .WIDTH(N_HOST)
) u_rrarb_host (
  .clk    (  i_clk       ),    // Clock
  .rst_n  (  !i_rst      ),    // Asynchronous reset active low
  .rst    (  1'b0        ),    // Synchronous reset active high
  .idle   (  rrarb_idle  ),    // Only allow new grants when idle. Tie to 1 to grant new req at any time.
  .req    (  i_mac_req   ),    // vector of requests
  .gnt    (  dest_val    )     // onehot0 vector of grants
);

rrarb #(
  .WIDTH(N_INPT)
) u_rrarb_sensor (
  .clk    (  i_clk       ),    // Clock
  .rst_n  (  !i_rst      ),    // Asynchronous reset active low
  .rst    (  1'b0        ),    // Synchronous reset active high
  .idle   (  rrarb_idle  ),    // Only allow new grants when idle. Tie to 1 to grant new req at any time.
  .req    (  mac_valid   ),    // vector of requests
  .gnt    (  mac_grant   )     // onehot0 vector of grants
);

assign rrarb_idle = '1;

logic [7:0] wr_idx;
logic [7:0] rd_idx;

always_comb begin
  wr_idx = '0;
  for (int i = 0; i < N_INPT; i++) begin
    if (mac_grant[i]) begin
      wr_idx = i;
    end
  end
end

always_comb begin
  rd_idx = '0;
  for (int i = 0; i < N_HOST; i++) begin
    if (dest_val[i]) begin
      rd_idx = i;
    end
  end
end



//------------------------------------------------------------------------------------------------//
// CAM inst
//------------------------------------------------------------------------------------------------//

cam #(
  .W_DATA     ( N_INPT        ),
  .W_KEY      ( 48            ),
  .DEPTH      ( DEPTH         )
) routing_cam (
  .i_clk      ( i_clk         ),
  .i_rst      ( i_rst         ),
  .i_wr       ( r_cam_wr      ),
  .i_del      ( cam_del       ),
  .i_data     ( r_cam_wr_data ),
  .i_key      ( r_cam_key     ),
  .o_data     ( cam_rd_data   ),
  .o_data_val ( cam_data_val  )
);

always_comb begin
  cam_wr = '0;
  cam_del = '0;
  if (|i_mac_req) begin // Read priority
    cam_wr = '0;
  end
  else if (|mac_valid) begin
    cam_wr = '1;
  end
end

always_ff @(posedge i_clk) begin
  if (i_rst) begin
    r_cam_wr      <= '0;
    r_cam_key     <= '0;
    rr_cam_key    <= '0;
    r_cam_wr_data <= '0;
    r_dest_val    <= '0;
    rr_dest_val   <= '0;
  end
  else begin
    r_cam_wr      <= cam_wr;
    r_cam_key     <= cam_key;
    rr_cam_key    <= r_cam_key;
    r_cam_wr_data <= cam_wr_data;
    r_dest_val    <= dest_val;
    rr_dest_val   <= r_dest_val;
  end
end


assign cam_key     = (cam_wr) ? mac_data[wr_idx] : i_mac_addr[rd_idx];
assign cam_wr_data = (cam_wr) ? mac_grant        : '0                ;

always_ff @(posedge i_clk) begin
  if (i_rst) begin
    o_dest_port <= '0;
    o_dest_val  <= '0;
  end
  else begin
    o_dest_port <= ((rr_cam_key == '1) || (rr_cam_key == BE_PTP_FW_MULTI_ADDR) || (rr_cam_key == BE_PTP_NON_FW_MULTI_ADDR)) ? '1             : // Broadcast MAC
                    (rr_cam_key == '0)                                                                                      ? '0             : // Drop Packet
                                                                                                                             {'0,cam_rd_data}; // Read CAM
    o_dest_val  <= rr_dest_val;
  end
end

endmodule
