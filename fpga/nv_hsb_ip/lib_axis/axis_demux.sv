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

module axis_demux
#(
    parameter int N_OUT = 2,
    parameter int W_DATA = 64,
    parameter int W_USER = 2,
    parameter int BUFFER_ENABLE = 0,
    parameter int W_OUT_WIDTH [N_OUT-1:0] = '{default:W_DATA},
    localparam int W_ID = (N_OUT <= 2) ? 1 : $clog2(N_OUT),
    localparam int W_KEEP = W_DATA/8
)
(

    // clock and reset
    input                   i_sclk,
    input                   i_srst,

    // AXIS input
    input                  i_axis_tvalid,
    input                  i_axis_tlast,
    input     [W_DATA-1:0] i_axis_tdata,
    input     [W_KEEP-1:0] i_axis_tkeep,
    input     [W_ID-1:0]   i_axis_tid,
    input     [W_USER-1:0] i_axis_tuser,
    output                 o_axis_tready,

    // AXIS outputs
    output     [N_OUT-1:0]  o_axis_tvalid,
    output     [N_OUT-1:0]  o_axis_tlast,
    output     [W_DATA-1:0] o_axis_tdata [N_OUT-1:0],
    output     [W_KEEP-1:0] o_axis_tkeep [N_OUT-1:0],
    output     [W_USER-1:0] o_axis_tuser [N_OUT-1:0],
    input      [N_OUT-1:0]  i_axis_tready
);

//------------------------------------------------------------------------------
// Input DEMUX
//------------------------------------------------------------------------------

logic [N_OUT-1:0] axis_tvalid;
logic [N_OUT-1:0] axis_tready;
logic [N_OUT-1:0] axis_tlast;
logic [W_DATA-1:0] axis_tdata;
logic [W_KEEP-1:0] axis_tkeep;
logic [W_USER-1:0] axis_tuser;
logic              selected_axis_tready;


always_comb begin : input_demux
    axis_tvalid          = '0;
    selected_axis_tready = 1'b0;
    for (int i = 0; i < N_OUT; i++) begin
        if (i_axis_tid == i) begin
            axis_tvalid[i]       = i_axis_tvalid;
            selected_axis_tready = axis_tready[i];
        end
    end
end

assign axis_tlast     = {N_OUT{i_axis_tlast}};
assign axis_tdata     = i_axis_tdata;
assign axis_tkeep     = i_axis_tkeep;
assign axis_tuser     = i_axis_tuser;
assign o_axis_tready  = selected_axis_tready;

genvar i;

generate
    if (BUFFER_ENABLE) begin
        for (i = 0; i < N_OUT; i=i+1) begin
            localparam int W_OUT_KEEP = W_OUT_WIDTH[i]/8;

            logic [W_OUT_WIDTH[i]-1:0] out_axis_tdata;
            logic [W_OUT_KEEP-1:0]     out_axis_tkeep;

            axis_buffer # (
                .IN_DWIDTH         ( W_DATA                ),
                .OUT_DWIDTH        ( W_OUT_WIDTH   [i]     ),
                .WAIT2SEND         ( 0                     ),
                .W_USER            ( W_USER                )
            ) u_axis_buffer (
                .in_clk            ( i_sclk                ),
                .in_rst            ( i_srst                ),
                .out_clk           ( i_sclk                ),
                .out_rst           ( i_srst                ),
                .i_axis_rx_tvalid  ( axis_tvalid    [i]    ),
                .i_axis_rx_tdata   ( axis_tdata            ),
                .i_axis_rx_tlast   ( axis_tlast     [i]    ),
                .i_axis_rx_tuser   ( axis_tuser            ),
                .i_axis_rx_tkeep   ( axis_tkeep            ),
                .o_axis_rx_tready  ( axis_tready    [i]    ),
                .o_axis_tx_tvalid  ( o_axis_tvalid  [i]    ),
                .o_axis_tx_tdata   ( out_axis_tdata        ),
                .o_axis_tx_tlast   ( o_axis_tlast   [i]    ),
                .o_axis_tx_tuser   ( o_axis_tuser   [i]    ),
                .o_axis_tx_tkeep   ( out_axis_tkeep        ),
                .i_axis_tx_tready  ( i_axis_tready  [i]    )
            );

            if (W_OUT_WIDTH[i] == W_DATA) begin
                assign o_axis_tdata[i] = out_axis_tdata;
                assign o_axis_tkeep[i] = out_axis_tkeep;
            end
            else begin
                assign o_axis_tdata[i] = {{(W_DATA-W_OUT_WIDTH[i]){1'b0}}, out_axis_tdata};
                assign o_axis_tkeep[i] = {{(W_KEEP-W_OUT_KEEP){1'b0}}, out_axis_tkeep};
            end
        end
    end
    else begin
        assign o_axis_tvalid = axis_tvalid;
        assign o_axis_tlast  = axis_tlast;
        assign axis_tready   = i_axis_tready;
        for (i = 0; i < N_OUT; i=i+1) begin
            assign o_axis_tdata[i] = axis_tdata;
            assign o_axis_tkeep[i] = axis_tkeep;
            assign o_axis_tuser[i] = axis_tuser;
        end
    end
endgenerate

endmodule
