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

module axis_arb #(         // Priority/RR arb
  parameter              N_INPT   = 2,  // Number of AXI stream inputs
  parameter              W_DATA   = 64, // Width of AXI stream interfaces in bits, byte align
  parameter              W_USER   = 1,
  parameter              ARB_TYPE = "ROUND_ROBIN", //ROUND_ROBIN or PRIORITY
  parameter [N_INPT-1:0] SKID     = '1,
  parameter [N_INPT-1:0] REG      = '1,
  localparam             W_KEEP   = W_DATA/8,
  localparam             W_INDX   = (N_INPT==1) ? 1 : $clog2(N_INPT)
)(
  input                 i_clk,
  input                 i_rst,
  // AXI inputs
  input  [N_INPT-1:0]   i_axis_tvalid,
  input  [N_INPT-1:0]   i_axis_tlast,
  input  [W_DATA-1:0]   i_axis_tdata  [N_INPT],
  input  [W_KEEP-1:0]   i_axis_tkeep  [N_INPT],
  input  [W_USER-1:0]   i_axis_tuser  [N_INPT],
  output [N_INPT-1:0]   o_axis_tready,
  // Output Index
  output [W_INDX-1:0]   o_axis_idx,
  // AXI output
  output                o_axis_tvalid,
  output                o_axis_tlast,
  output [W_DATA-1:0]   o_axis_tdata,
  output [W_KEEP-1:0]   o_axis_tkeep,
  output [W_USER-1:0]   o_axis_tuser,
  input                 i_axis_tready
);

//------------------------------------------------------------------------------------------------//
// Input Reg
//------------------------------------------------------------------------------------------------//
genvar i;

logic  [N_INPT-1:0]   r_axis_tvalid;
logic  [N_INPT-1:0]   r_axis_tlast;
logic  [W_DATA-1:0]   r_axis_tdata  [N_INPT];
logic  [W_KEEP-1:0]   r_axis_tkeep  [N_INPT];
logic  [W_USER-1:0]   r_axis_tuser  [N_INPT];
logic  [N_INPT-1:0]   r_axis_tready;

generate
  for (i=0;i<N_INPT;i=i+1) begin
    if (REG[i]) begin
    axis_reg # (
      .DWIDTH             ( W_DATA + W_KEEP + 1 + W_USER ),
      .SKID               ( SKID[i]                      )
    ) u_in_axis_reg (
      .clk                ( i_clk                                                            ),
      .rst                ( i_rst                                                            ),
      .i_axis_rx_tvalid   ( i_axis_tvalid[i]                                                 ),
      .i_axis_rx_tdata    ( {i_axis_tdata[i],i_axis_tlast[i],i_axis_tkeep[i],i_axis_tuser[i]}),
      .o_axis_rx_tready   ( o_axis_tready[i]                                                 ),
      .o_axis_tx_tvalid   ( r_axis_tvalid[i]                                                 ),
      .o_axis_tx_tdata    ( {r_axis_tdata[i],r_axis_tlast[i],r_axis_tkeep[i],r_axis_tuser[i]}),
      .i_axis_tx_tready   ( r_axis_tready[i]                                                 )
    );
    end
    else begin
      assign r_axis_tvalid [i]  = i_axis_tvalid [i];
      assign r_axis_tlast  [i]  = i_axis_tlast  [i];
      assign r_axis_tdata  [i]  = i_axis_tdata  [i];
      assign r_axis_tkeep  [i]  = i_axis_tkeep  [i];
      assign r_axis_tuser  [i]  = i_axis_tuser  [i];
      assign o_axis_tready [i]  = r_axis_tready [i];
    end
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// Input Round Robin ARB
//------------------------------------------------------------------------------------------------//

logic                    arb_axis_tvalid;
logic                    arb_axis_tlast;
logic [W_DATA-1:0]       arb_axis_tdata;
logic [W_KEEP-1:0]       arb_axis_tkeep;
logic                    arb_axis_tuser;
logic                    arb_axis_tready;

logic [N_INPT-1:0]       w_axis_tready;
logic [W_INDX-1:0]       gnt_idx;
logic                    pkt_active;


always_ff @(posedge i_clk) begin
  if (i_rst) begin
    pkt_active                  <= 1'b0;
  end
  else begin
    if (r_axis_tvalid[gnt_idx]) begin
      if (r_axis_tlast[gnt_idx]) begin
        pkt_active              <= 1'b0;
      end
      else begin
        pkt_active              <= 1'b1;
      end
    end
  end
end

generate
  if (ARB_TYPE == "ROUND_ROBIN") begin
    rrarb #(
      .WIDTH ( N_INPT               )
    ) axis_arb (
      .clk   ( i_clk                ),
      .rst_n ( !i_rst               ),
      .rst   ( 1'b0                 ),
      .idle  ( !pkt_active          ),
      .req   ( r_axis_tvalid        ),
      .gnt   ( w_axis_tready        )
    );
  end
  else begin
    priority_arb #(
      .WIDTH ( N_INPT               )
    ) axis_arb (
      .clk   ( i_clk                ),
      .rst_n ( !i_rst               ),
      .rst   ( 1'b0                 ),
      .idle  ( !pkt_active          ),
      .req   ( r_axis_tvalid        ),
      .gnt   ( w_axis_tready        )
    );
  end
endgenerate

logic        hp_incr;
logic        hp_rst;
logic [15:0] r_hp_pkt_cnt;

integer j;
always_comb begin
  gnt_idx = '0;
  for (j=0;j<N_INPT;j=j+1) begin
    if (w_axis_tready[j]) begin
      gnt_idx = j;
    end
  end
end

assign r_axis_tready = (arb_axis_tready) ? w_axis_tready : '0;

assign o_axis_tvalid = r_axis_tvalid[gnt_idx];
assign o_axis_tdata  = r_axis_tdata[gnt_idx];
assign o_axis_tlast  = r_axis_tlast[gnt_idx];
assign o_axis_tuser  = r_axis_tuser[gnt_idx];
assign o_axis_tkeep  = r_axis_tkeep[gnt_idx];
assign arb_axis_tready = i_axis_tready;
assign o_axis_idx = gnt_idx;


endmodule
