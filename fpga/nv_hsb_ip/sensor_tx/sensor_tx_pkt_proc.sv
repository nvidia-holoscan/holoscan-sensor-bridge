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

//Input is full AXIS packet.


module sensor_tx_pkt_proc
#(
  parameter         AXI_DWIDTH      = 64,
  parameter         NUM_HOST        = 2,
  parameter         NUM_SENSOR      = 8,
  localparam        KEEP_WIDTH      = (AXI_DWIDTH / 8)
)(
  input   logic                     host_clk,
  input   logic                     host_rst,

  input   logic   [  NUM_HOST-1:0]  i_axis_tvalid,
  input   logic   [AXI_DWIDTH-1:0]  i_axis_tdata  [NUM_HOST],
  input   logic   [  NUM_HOST-1:0]  i_axis_tlast,
  input   logic   [  NUM_HOST-1:0]  i_axis_tuser,
  input   logic   [KEEP_WIDTH-1:0]  i_axis_tkeep  [NUM_HOST],

  input            [71:0]           i_dest_info   [NUM_HOST],

  output   logic   [NUM_SENSOR-1:0] o_axis_tvalid,
  output   logic   [AXI_DWIDTH-1:0] o_axis_tdata  [NUM_SENSOR],
  output   logic   [NUM_SENSOR-1:0] o_axis_tlast,
  output   logic   [NUM_SENSOR-1:0] o_axis_tuser,
  output   logic   [KEEP_WIDTH-1:0] o_axis_tkeep  [NUM_SENSOR]
);

localparam ETH_BYTES            = 14;
localparam IP_BYTES             = 20;
localparam UDP_BYTES            = 8;
localparam BTH_BYTES            = 12;
localparam HDR_BYTES            = (ETH_BYTES + IP_BYTES + UDP_BYTES + BTH_BYTES);
localparam ICRC_BYTES           = 4;
localparam ICRC_BEATS           = ((ICRC_BYTES - 1) / (AXI_DWIDTH/8)) + 1;

//------------------------------------------------------------------------------------------------//
// Register for Timing - No ready signal so no need for SKID Buffer
//------------------------------------------------------------------------------------------------//


localparam DEST_USER_WIDTH = 8; // dest_info[56:49] carried via tuser sideband

logic   [  NUM_HOST-1:0]           drp_axis_tvalid           ;
logic   [AXI_DWIDTH-1:0]           drp_axis_tdata  [NUM_HOST];
logic   [  NUM_HOST-1:0]           drp_axis_tlast            ;
logic   [DEST_USER_WIDTH-1:0]      drp_axis_tuser  [NUM_HOST];
logic   [KEEP_WIDTH-1:0]           drp_axis_tkeep  [NUM_HOST];

logic   [  NUM_HOST-1:0]           raw_axis_tvalid           ;
logic   [AXI_DWIDTH-1:0]           raw_axis_tdata  [NUM_HOST];
logic   [  NUM_HOST-1:0]           raw_axis_tlast            ;
logic   [DEST_USER_WIDTH-1:0]      raw_axis_tuser  [NUM_HOST];
logic   [KEEP_WIDTH-1:0]           raw_axis_tkeep  [NUM_HOST];

genvar i,k;

generate
  for (k=0;k<NUM_HOST;k=k+1) begin: gen_stx_pre_proc

    //Drop Header Data - tuser carries dest_info[56:49] through pipeline
    axis_drop #(
      .DROP_WIDTH                     (HDR_BYTES*8),
      .DWIDTH                         (AXI_DWIDTH),
      .W_USER                         (DEST_USER_WIDTH)
    ) stx_axis_drop (
      .clk                            (host_clk                ),
      .rst                            (host_rst                ),
      .i_axis_rx_tvalid               (i_axis_tvalid       [k] ),
      .i_axis_rx_tdata                (i_axis_tdata        [k] ),
      .i_axis_rx_tlast                (i_axis_tlast        [k] ),
      .i_axis_rx_tuser                (i_dest_info  [k][56:49] ),
      .i_axis_rx_tkeep                (i_axis_tkeep        [k] ),
      .o_axis_rx_tready               (                        ),
      .o_axis_tx_tvalid               (drp_axis_tvalid     [k] ),
      .o_axis_tx_tdata                (drp_axis_tdata      [k] ),
      .o_axis_tx_tlast                (drp_axis_tlast      [k] ),
      .o_axis_tx_tuser                (drp_axis_tuser      [k] ),
      .o_axis_tx_tkeep                (drp_axis_tkeep      [k] ),
      .i_axis_tx_tready               (1'b1                    )
    );

    //Drop ICRC - tuser carries dest_info[56:49] through pipeline
    axis_drop_tail #(
      .DROP_WIDTH                     (32                  ),
      .DWIDTH                         (AXI_DWIDTH          ),
      .W_USER                         (DEST_USER_WIDTH     )
    ) stx_axis_drop_tail (
      .i_clk                          (host_clk            ),
      .i_rst                          (host_rst            ),
      .i_axis_rx_tvalid               (drp_axis_tvalid [k] ),
      .i_axis_rx_tdata                (drp_axis_tdata  [k] ),
      .i_axis_rx_tlast                (drp_axis_tlast  [k] ),
      .i_axis_rx_tuser                (drp_axis_tuser  [k] ),
      .i_axis_rx_tkeep                (drp_axis_tkeep  [k] ),
      .o_axis_tx_tvalid               (raw_axis_tvalid [k] ),
      .o_axis_tx_tdata                (raw_axis_tdata  [k] ),
      .o_axis_tx_tlast                (raw_axis_tlast  [k] ),
      .o_axis_tx_tuser                (raw_axis_tuser  [k] ),
      .o_axis_tx_tkeep                (raw_axis_tkeep  [k] )
    );

  end
endgenerate


//------------------------------------------------------------------------------------------------//
// DeMux from Host AXIS bus to Sensor AXIS Bus
// dest_info[56:49] is carried through the pipeline via tuser sideband,
// so raw_axis_tuser is always aligned with the output packet framing.
//------------------------------------------------------------------------------------------------//
generate
  for (i=0;i<NUM_SENSOR;i=i+1) begin: gen_stx_demux
    always_comb begin
      o_axis_tvalid [i] = '0;
      o_axis_tdata  [i] = '0;
      o_axis_tlast  [i] = '0;
      o_axis_tuser  [i] = '0;
      o_axis_tkeep  [i] = '0;
      for (int j=0; j<NUM_HOST; j=j+1) begin
        if ((raw_axis_tuser[j] == (i+1)) && raw_axis_tvalid[j]) begin
          o_axis_tvalid [i] = raw_axis_tvalid [j];
          o_axis_tdata  [i] = raw_axis_tdata  [j];
          o_axis_tlast  [i] = raw_axis_tlast  [j];
          o_axis_tuser  [i] = '0;
          o_axis_tkeep  [i] = raw_axis_tkeep  [j];
        end
      end
    end
  end
endgenerate

endmodule
