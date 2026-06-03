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

//This module is modification of the axis_buffer from lib_axis. The idea is that this module is parameterized such that sensor data can stream out at
//a constant rate once started. The modifications to support TX sensor data are as follows:
// 1) Remove the WAIT2SEND functionality. Sensor data is streaming and not packet based at this level.
// 2) clear_to_send logic was changed to wait for the ready input to the module. Otherwise, prefetching of data could result in a non-constant tvalid output.
// 3) gbx_axis_tready tied to 1'b1. No backpressure in this module since the design must ingest TX sensor data as it comes in.
// 4) FIFO status signals all connected as output at top of module.

// Original axis_buffer description follows:
//This module takes AXIS packets on the input and provides a buffer to store the packets. These packets can be
//output from the buffer in a couple different ways. One way is just to output data as soon as it's in the buffer.
//The other mode is to wait until packets are completely in the buffer before sending out. This module assumes a CDC between input and output.
//A DC FIFO is used at the input side to accomplish this. The output mode to choose depends on the interface the output is connected to. If the downstream
//interface can tolerate the output data valid to deassert during a packet transmission then the output mode can be set
//to output data immediately. If the downstream interface cannot tolerate the data valid deasserting during a transmission
//then the WAIT2SEND mode should be chosen.

// Additionally, the input and output data widths can be different. A gearbox is used to align the input width to a central width.
// The centeral width is defined as the greater value between input and output data width. If the input data width is less than, then the
// input gear box does the alignment before going into the DC FIFO. This is to prevent any backpressure coming from the gearbox.
// When the input data width is greater than, the gearbox happens afterwards, and the DC FIFO aborbs the backpressure so the output data rate
// is constant. A AXIS reg is also added to help timing with the dc fifo, as well as fix the "cannot wait for ready" requirement the dc fifo logic
// implements incorrectly.

// Two gearboxes are instantiated, but only one should be adding logic (the other will be din==dout and will just be directly connected wires)

module axis_buffer_tx #(
  parameter  IN_DWIDTH     = 64,
  parameter  OUT_DWIDTH    = 8,
  parameter  BUF_DEPTH     = 256,
  parameter  DUAL_CLOCK    = 0,
  parameter  W_USER        = 1,
  localparam FIFO_DEPTH_W  = ($clog2(BUF_DEPTH))+1
)
(
  input   logic                               in_clk,
  input   logic                               in_rst,
  input   logic                               out_clk,
  input   logic                               out_rst,

  input   logic                               i_axis_rx_tvalid,
  input   logic   [IN_DWIDTH-1:0]             i_axis_rx_tdata,
  input   logic                               i_axis_rx_tlast,
  input   logic   [W_USER-1:0]                i_axis_rx_tuser,
  input   logic   [(IN_DWIDTH/8)-1:0]         i_axis_rx_tkeep,
  output  logic                               o_axis_rx_tready,

  output  logic                               o_axis_tx_tvalid,
  output  logic   [OUT_DWIDTH-1:0]            o_axis_tx_tdata,
  output  logic                               o_axis_tx_tlast,
  output  logic   [W_USER-1:0]                o_axis_tx_tuser,
  output  logic   [(OUT_DWIDTH/8)-1:0]        o_axis_tx_tkeep,
  input   logic                               i_axis_tx_tready,

  output  logic                               o_buffer_afull,
  output  logic                               o_buffer_aempty,
  output  logic                               o_buffer_full,
  output  logic                               o_buffer_empty,
  output  logic   [FIFO_DEPTH_W-1:0]          o_buffer_wrcnt,
  output  logic   [FIFO_DEPTH_W-1:0]          o_buffer_rdcnt,
  output  logic                               o_buffer_over,
  output  logic                               o_buffer_under
);

localparam CENTER_WIDTH = (IN_DWIDTH <= OUT_DWIDTH) ? OUT_DWIDTH : IN_DWIDTH;


localparam ALMOST_FULL_DEPTH  = BUF_DEPTH * 3 / 4;
localparam ALMOST_EMPTY_DEPTH = BUF_DEPTH / 4  ;

// Store num tkeep instead of tkeep to reduce fifo width
localparam CENTER_KEEP_CNT_WIDTH = (CENTER_WIDTH == 8) ? 1 : $clog2(CENTER_WIDTH/8);

localparam FIFO_DWIDTH       = CENTER_WIDTH + CENTER_KEEP_CNT_WIDTH + 1 + W_USER;

localparam WAIT2SEND = 0;

//------------------------------------------------------------------------------------------------//
// Gear Box
//------------------------------------------------------------------------------------------------//

logic                                 gbx_axis_tlast;
logic   [CENTER_WIDTH-1:0]            gbx_axis_tdata;
logic   [(CENTER_WIDTH/8)-1:0]        gbx_axis_tkeep;
logic                                 gbx_axis_tvalid;
logic   [W_USER-1:0]                  gbx_axis_tuser;
logic                                 gbx_axis_tready;
logic   [CENTER_KEEP_CNT_WIDTH-1:0]   gbx_axis_tkcnt;

axis_gearbox #(
  .DIN_WIDTH  ( IN_DWIDTH    ),
  .DOUT_WIDTH ( CENTER_WIDTH ),
  .W_USER     ( W_USER       )
) u_axis_in_gearbox (
  .clk                ( in_clk            ),
  .rst                ( in_rst            ),
  .i_axis_rx_tvalid   ( i_axis_rx_tvalid  ),
  .i_axis_rx_tdata    ( i_axis_rx_tdata   ),
  .i_axis_rx_tlast    ( i_axis_rx_tlast   ),
  .i_axis_rx_tuser    ( i_axis_rx_tuser   ),
  .i_axis_rx_tkeep    ( i_axis_rx_tkeep   ),
  .o_axis_rx_tready   ( o_axis_rx_tready  ),
  .o_axis_tx_tvalid   ( gbx_axis_tvalid   ),
  .o_axis_tx_tdata    ( gbx_axis_tdata    ),
  .o_axis_tx_tlast    ( gbx_axis_tlast    ),
  .o_axis_tx_tuser    ( gbx_axis_tuser    ),
  .o_axis_tx_tkeep    ( gbx_axis_tkeep    ),
  .i_axis_tx_tready   ( gbx_axis_tready   )
);

//------------------------------------------------------------------------------------------------//
// FIFO
//------------------------------------------------------------------------------------------------//

logic   [FIFO_DWIDTH-1:0]             fifo_din;
logic   [FIFO_DWIDTH-1:0]             fifo_dout;
logic                                 fifo_empty;
logic                                 fifo_full;
logic                                 fifo_rden;
logic                                 fifo_afull;
logic                                 fifo_aempty;
logic                                 fifo_under;
logic                                 fifo_over;
logic   [FIFO_DEPTH_W-1:0]            fifo_wrcnt;
logic   [FIFO_DEPTH_W-1:0]            fifo_rdcnt;

logic                                 fifo_rdval;
logic                                 fifo_wren;

logic                                 cdc_axis_tlast;
logic                                 w_cdc_axis_tlast;
logic   [CENTER_WIDTH-1:0]            cdc_axis_tdata;
logic                                 cdc_axis_tvalid;
logic   [W_USER-1:0]                  cdc_axis_tuser;
logic                                 cdc_axis_tready;
logic   [CENTER_KEEP_CNT_WIDTH-1:0]   cdc_axis_tkcnt;
logic                                 cdc_tvalid_hold;

logic                                 in_pkt_done;


generate
  if (DUAL_CLOCK == 1) begin
    dc_fifo_stub #(
      .DATA_WIDTH    ( FIFO_DWIDTH          ), // data + keep + last
      .FIFO_DEPTH    ( BUF_DEPTH            ), // integer
      .ALMOST_FULL   ( ALMOST_FULL_DEPTH    ), // integer (pmi_almost_full_flag MUST be LESS than pmi_data_depth)
      .ALMOST_EMPTY  ( ALMOST_EMPTY_DEPTH   ), // integer
      .MEM_STYLE     ( "BLOCK"              )
    ) axis_buffer_cdc (
      .wrrst        ( in_rst                ),
      .rdrst        ( out_rst               ),
      // wr domain signals
      .wrclk        ( in_clk                ),
      .wren         ( fifo_wren             ),
      .wrdin        ( fifo_din              ),
      .full         ( fifo_full             ),
      .afull        ( fifo_afull            ),
      .over         ( fifo_over             ),
      .wrcount      ( fifo_wrcnt            ),
      // rd domain signals
      .rdclk        ( out_clk               ),
      .rden         ( fifo_rden             ),
      .rddout       ( fifo_dout             ),
      .rdval        ( fifo_rdval            ),
      .empty        ( fifo_empty            ),
      .aempty       ( fifo_aempty           ),
      .under        ( fifo_under            ),
      .rdcount      ( fifo_rdcnt            )
    );
  end
  else if (DUAL_CLOCK == 0) begin
    sc_fifo #(
      .DATA_WIDTH    ( FIFO_DWIDTH          ), // data + keep + last
      .FIFO_DEPTH    ( BUF_DEPTH            ), // integer
      .ALMOST_FULL   ( ALMOST_FULL_DEPTH    ), // integer (pmi_almost_full_flag MUST be LESS than pmi_data_depth)
      .ALMOST_EMPTY  ( ALMOST_EMPTY_DEPTH   ), // integer
      .MEM_STYLE     ( "BLOCK"              )
    ) axis_buffer_sc (
      .rst          ( in_rst                ),
      .clk          ( in_clk                ),
      // wr signals
      .wr           ( fifo_wren             ),
      .din          ( fifo_din              ),
      .full         ( fifo_full             ),
      .afull        ( fifo_afull            ),
      .over         ( fifo_over             ),
      .count        ( fifo_wrcnt            ),
      // rd signals
      .rd           ( fifo_rden             ),
      .dout         ( fifo_dout             ),
      .dval         ( fifo_rdval            ),
      .empty        ( fifo_empty            ),
      .aempty       ( fifo_aempty           ),
      .under        ( fifo_under            )
    );
    assign fifo_rdcnt  = fifo_wrcnt;
  end
endgenerate

assign o_buffer_full    = fifo_full;
assign o_buffer_empty   = fifo_empty;
assign o_buffer_afull   = fifo_afull;
assign o_buffer_aempty  = fifo_aempty;
assign o_buffer_wrcnt   = fifo_wrcnt;
assign o_buffer_rdcnt   = fifo_rdcnt;
assign o_buffer_over    = fifo_over;
assign o_buffer_under   = fifo_under;

integer i;
always_comb begin
  gbx_axis_tkcnt = '0;
  for (i=0;i<(CENTER_WIDTH/8);i=i+1) begin
    if (gbx_axis_tkeep[i]) begin
      gbx_axis_tkcnt = i;
    end
  end
end

assign fifo_din = {gbx_axis_tlast, gbx_axis_tkcnt, gbx_axis_tdata,gbx_axis_tuser};
assign {w_cdc_axis_tlast, cdc_axis_tkcnt, cdc_axis_tdata,cdc_axis_tuser} = fifo_dout;
assign cdc_axis_tlast = (w_cdc_axis_tlast && cdc_axis_tvalid);
assign cdc_axis_tvalid = (fifo_rdval || cdc_tvalid_hold);
assign fifo_wren = gbx_axis_tvalid && gbx_axis_tready;

// In Pkt Counter
always_ff @(posedge in_clk) begin
  if (in_rst) begin
    in_pkt_done          <= '0;
    gbx_axis_tready      <= '0;
  end
  else begin
    in_pkt_done     <= fifo_wren && fifo_din[$left(fifo_din)];
    gbx_axis_tready <= 1'b1; //!fifo_afull;
  end
end


//------------------------------------------------------------------------------------------------//
// Packet Logic
//------------------------------------------------------------------------------------------------//

logic  [$clog2(BUF_DEPTH/2)-1:0]      pkt_cnt;
logic                                 clear_to_send;
logic                                 out_pkt_done;
logic                                 in_pkt_done_sync;
logic                                 prefetch;

//Keep track of how many packets are in the buffer
always_ff @(posedge out_clk) begin
  if (out_rst) begin
    pkt_cnt                 <= 'b0;
    cdc_tvalid_hold         <= '0;
    clear_to_send           <= '0;
  end
  else begin
    pkt_cnt              <= pkt_cnt + in_pkt_done_sync - out_pkt_done;
    cdc_tvalid_hold      <= cdc_axis_tvalid && !cdc_axis_tready; // Hold tvalid on next cycle
    clear_to_send        <= !WAIT2SEND      ? 1'b1                                :
                            (pkt_cnt == '0) ? (in_pkt_done_sync && !out_pkt_done) :
                            (pkt_cnt =='d1) ? !(!in_pkt_done_sync && out_pkt_done):
                                             1'b1;
  end
end

pulse_sync u_pkt_cnt_sync (
  .src_clk        ( in_clk                      ),
  .src_rst        ( in_rst                      ),
  .dst_clk        ( out_clk                     ),
  .dst_rst        ( out_rst                     ),
  .i_src_pulse    ( in_pkt_done                 ),
  .o_dst_pulse    ( in_pkt_done_sync            )
);

//If WAIT2SEND is set, then a full packet must be in the buffer before it can be read out.
//I put this here in case we are talking to some interface that doesn't like the valid signal
//going low in the middle of a packet
assign out_pkt_done    = cdc_axis_tvalid && cdc_axis_tlast && cdc_axis_tready;
assign fifo_rden       = !fifo_empty && clear_to_send && prefetch && ((WAIT2SEND) ? !out_pkt_done : '1);
assign prefetch        = (!cdc_tvalid_hold && !cdc_axis_tvalid) || cdc_axis_tready;

//------------------------------------------------------------------------------------------------//
// SKID Buffer - to fix AXIS protocol and help with DC FIFO timing
//------------------------------------------------------------------------------------------------//

logic                                 reg_axis_tlast;
logic   [CENTER_WIDTH-1:0]            reg_axis_tdata;
logic   [(CENTER_WIDTH/8)-1:0]        reg_axis_tkeep;
logic   [CENTER_KEEP_CNT_WIDTH-1:0]   reg_axis_tkcnt;
logic                                 reg_axis_tvalid;
logic   [W_USER-1:0]                  reg_axis_tuser;
logic                                 reg_axis_tready;

localparam SKID = (DUAL_CLOCK);

axis_reg # (
  .DWIDTH             ( CENTER_WIDTH + CENTER_KEEP_CNT_WIDTH + W_USER + 1                ),
  .SKID               ( 1                                                                )
) u_axis_reg (
  .clk                ( out_clk                                                          ),
  .rst                ( out_rst                                                          ),
  .i_axis_rx_tvalid   ( cdc_axis_tvalid                                                  ),
  .i_axis_rx_tdata    ( {cdc_axis_tdata,cdc_axis_tlast,cdc_axis_tuser,cdc_axis_tkcnt}    ),
  .o_axis_rx_tready   ( cdc_axis_tready                                                  ),
  .o_axis_tx_tvalid   ( reg_axis_tvalid                                                  ),
  .o_axis_tx_tdata    ( {reg_axis_tdata,reg_axis_tlast,reg_axis_tuser,reg_axis_tkcnt}    ),
  .i_axis_tx_tready   ( reg_axis_tready                                                  )
);

//------------------------------------------------------------------------------------------------//
// Gear Box
//------------------------------------------------------------------------------------------------//

logic   [OUT_DWIDTH-1:0]            w_axis_tx_tdata;
logic                               w_axis_tx_tlast;
logic   [W_USER-1:0]                w_axis_tx_tuser;
logic   [(OUT_DWIDTH/8)-1:0]        w_axis_tx_tkeep;

axis_gearbox #(
  .DIN_WIDTH  ( CENTER_WIDTH ),
  .DOUT_WIDTH ( OUT_DWIDTH   ),
  .W_USER     ( W_USER       )
) u_axis_out_gearbox (
  .clk                ( out_clk           ),
  .rst                ( out_rst           ),
  .i_axis_rx_tvalid   ( reg_axis_tvalid   ),
  .i_axis_rx_tdata    ( reg_axis_tdata    ),
  .i_axis_rx_tlast    ( reg_axis_tlast    ),
  .i_axis_rx_tuser    ( reg_axis_tuser    ),
  .i_axis_rx_tkeep    ( reg_axis_tkeep    ),
  .o_axis_rx_tready   ( reg_axis_tready   ),
  .o_axis_tx_tvalid   ( o_axis_tx_tvalid  ),
  .o_axis_tx_tdata    ( w_axis_tx_tdata   ),
  .o_axis_tx_tlast    ( w_axis_tx_tlast   ),
  .o_axis_tx_tuser    ( w_axis_tx_tuser   ),
  .o_axis_tx_tkeep    ( w_axis_tx_tkeep   ),
  .i_axis_tx_tready   ( i_axis_tx_tready  )
);

integer j;
always_comb begin
  reg_axis_tkeep = '0;
  for (j=0;j<(CENTER_WIDTH/8);j=j+1) begin
    if (j<=reg_axis_tkcnt) begin
      reg_axis_tkeep[j] = 1'b1;
    end
  end
end

// Remove 'x

assign o_axis_tx_tdata = o_axis_tx_tvalid ? w_axis_tx_tdata : '0;
assign o_axis_tx_tlast = o_axis_tx_tvalid ? w_axis_tx_tlast : '0;
assign o_axis_tx_tuser = o_axis_tx_tvalid ? w_axis_tx_tuser : '0;
assign o_axis_tx_tkeep = o_axis_tx_tvalid ? w_axis_tx_tkeep : '0;


endmodule
