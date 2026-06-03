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

// NOTE (DUAL_CLOCK == 1):
// - wrrst and rdrst are *synchronous* resets to wrclk/rdclk (not async).
// - BOTH sides MUST be reset (assert wrrst AND rdrst) to properly reset the DC FIFO.
//   If the write/input side is not reset, the reset-handshake never completes and the read/output
//   side is intentionally held/gated idle (no reads/dval) to prevent unsafe operation.

module axis_buffer #(
                    parameter  IN_DWIDTH          = 64,
                    parameter  OUT_DWIDTH         = 8,
                    parameter  BUF_DEPTH          = 256,
                    parameter  WAIT2SEND          = 0,
                    parameter  LATENCY_CNT        = 0,
                    parameter  DUAL_CLOCK         = 0,
                    parameter  W_USER             = 1,
                    parameter  ALMOST_FULL_DEPTH  = BUF_DEPTH * 7 / 8,
                    parameter  ALMOST_EMPTY_DEPTH = BUF_DEPTH / 8,
                    parameter  OUTPUT_SKID        = 1,
                    parameter  NO_BACKPRESSURE    = 0,
                    parameter  OUT_W_USER         = W_USER,
                    localparam IN_W_USER          = W_USER
)
(
  input   logic                               in_clk,
  input   logic                               in_rst,
  input   logic                               out_clk,
  input   logic                               out_rst,

  input   logic                               i_axis_rx_tvalid,
  input   logic   [IN_DWIDTH-1:0]             i_axis_rx_tdata,
  input   logic                               i_axis_rx_tlast,
  input   logic   [IN_W_USER-1:0]             i_axis_rx_tuser,
  input   logic   [(IN_DWIDTH/8)-1:0]         i_axis_rx_tkeep,
  output  logic                               o_axis_rx_tready,

  output  logic                               o_fifo_aempty,
  output  logic                               o_fifo_afull,
  output  logic                               o_fifo_empty,
  output  logic                               o_fifo_full,

  output  logic                               o_axis_tx_tvalid,
  output  logic   [OUT_DWIDTH-1:0]            o_axis_tx_tdata,
  output  logic                               o_axis_tx_tlast,
  output  logic   [OUT_W_USER-1:0]            o_axis_tx_tuser,
  output  logic   [(OUT_DWIDTH/8)-1:0]        o_axis_tx_tkeep,
  input   logic                               i_axis_tx_tready
);

localparam CENTER_WIDTH = (IN_DWIDTH <= OUT_DWIDTH) ? OUT_DWIDTH : IN_DWIDTH;
localparam CENTER_WUSER = (IN_W_USER <= OUT_W_USER) ? OUT_W_USER : IN_W_USER;


localparam FIFO_ADDR_W     = $clog2(BUF_DEPTH) + 1;

// Store num tkeep instead of tkeep to reduce fifo width
localparam CENTER_KEEP_CNT_WIDTH = (CENTER_WIDTH == 8) ? 1 : $clog2(CENTER_WIDTH/8);

localparam FIFO_DWIDTH       = CENTER_WIDTH + CENTER_KEEP_CNT_WIDTH + 1 + CENTER_WUSER;

localparam LATENCY_CNT_FACTOR = (IN_DWIDTH > OUT_DWIDTH) ? IN_DWIDTH/OUT_DWIDTH : 1;

//------------------------------------------------------------------------------------------------//
// Gear Box
//------------------------------------------------------------------------------------------------//

logic                                 gbx_axis_tlast;
logic   [CENTER_WIDTH-1:0]            gbx_axis_tdata;
logic   [(CENTER_WIDTH/8)-1:0]        gbx_axis_tkeep;
logic                                 gbx_axis_tvalid;
logic   [CENTER_WUSER-1:0]            gbx_axis_tuser;
logic                                 gbx_axis_tready;
logic   [CENTER_KEEP_CNT_WIDTH-1:0]   gbx_axis_tkcnt;

axis_gearbox #(
  .DIN_WIDTH  ( IN_DWIDTH    ),
  .DOUT_WIDTH ( CENTER_WIDTH ),
  .W_USER     ( IN_W_USER    )
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
logic                                 fifo_rden;
logic                                 fifo_afull;
logic                                 fifo_full;

logic                                 fifo_rdval;
logic                                 fifo_wren;
logic   [FIFO_ADDR_W-1:0]             fifo_count;
logic                                 fifo_aempty;
logic                                 cdc_axis_tlast;
logic                                 w_cdc_axis_tlast;
logic   [CENTER_WIDTH-1:0]            cdc_axis_tdata;
logic                                 cdc_axis_tvalid;
logic   [CENTER_WUSER-1:0]            cdc_axis_tuser;
logic                                 cdc_axis_tready;
logic   [CENTER_KEEP_CNT_WIDTH-1:0]   cdc_axis_tkcnt;
logic                                 cdc_tvalid_hold;

logic                                 in_pkt_done;

generate
  if (DUAL_CLOCK == 1) begin
    dc_fifo_stub
    #(
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
      .over         (                       ),
      .wrcount      ( fifo_count            ),
      // rd domain signals
      .rdclk        ( out_clk               ),
      .rden         ( fifo_rden             ),
      .rddout       ( fifo_dout             ),
      .rdval        ( fifo_rdval            ),
      .empty        ( fifo_empty            ),
      .aempty       ( fifo_aempty           ),
      .under        (                       ),
      .rdcount      (                       )
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
      .over         (                       ),
      .count        ( fifo_count            ),
      // rd signals
      .rd           ( fifo_rden             ),
      .dout         ( fifo_dout             ),
      .dval         ( fifo_rdval            ),
      .empty        ( fifo_empty            ),
      .aempty       ( fifo_aempty           ),
      .under        (                       )
    );
  end
endgenerate

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
    gbx_axis_tready <= !fifo_afull || NO_BACKPRESSURE;
  end
end

assign o_fifo_aempty = fifo_aempty;
assign o_fifo_afull = fifo_afull;
assign o_fifo_empty = fifo_empty;
assign o_fifo_full = fifo_full;

//------------------------------------------------------------------------------------------------//
// Packet Logic
//------------------------------------------------------------------------------------------------//

logic  [$clog2(BUF_DEPTH):0]          pkt_cnt;
logic                                 clear_to_send;
logic                                 out_pkt_done;
logic                                 in_pkt_done_sync;
logic                                 prefetch;

logic wait_latency;

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
assign fifo_rden       = !fifo_empty && clear_to_send && !wait_latency && prefetch && ((WAIT2SEND) ? !out_pkt_done : '1);
assign prefetch        = (!cdc_tvalid_hold && !cdc_axis_tvalid) || cdc_axis_tready;


//------------------------------------------------------------------------------------------------//
// pkt latency control logic
//------------------------------------------------------------------------------------------------//

localparam LATENCY_CNT_WIDTH = 10;

logic [LATENCY_CNT_WIDTH-1:0] latency_cnt, latency_cnt_sync;
logic [LATENCY_CNT_WIDTH-1:0] latency_offset_in, latency_offset_out;

logic wren_latency;
logic rden_latency;
logic rdval_latency;
logic empty_latency;
logic wait_sop;

generate
  if (LATENCY_CNT > 0) begin

    assign wren_latency = gbx_axis_tvalid && gbx_axis_tready && wait_sop && (fifo_count != '0); // skip 1st pkt latency calc
    assign rden_latency = out_pkt_done && !empty_latency;
    assign latency_offset_in = LATENCY_CNT - latency_cnt_sync + fifo_count*LATENCY_CNT_FACTOR;

    always_ff @(posedge in_clk) begin
      if (in_rst) begin
        wait_sop    <= 1'b1;
      end
      else begin
        if (wait_sop && gbx_axis_tvalid && gbx_axis_tready) begin
          wait_sop <= 1'b0;
        end
        else if (gbx_axis_tvalid && gbx_axis_tready && gbx_axis_tlast) begin
          wait_sop <= 1'b1;
        end
      end
    end

    always_ff @(posedge out_clk) begin
      if (out_rst) begin
        wait_latency   <= 1'b0;
        latency_cnt    <= '0;
      end
      else begin
        if (fifo_empty) begin
          wait_latency <= 1'b1;
        end
        else if (out_pkt_done) begin
          wait_latency <= 1'b1;
        end
        else if (!rdval_latency && (latency_cnt >= LATENCY_CNT)) begin
          wait_latency <= 1'b0;
        end

        latency_cnt  <= fifo_empty                   ? '0 :                 // 1st pkt
                        rdval_latency                ? latency_offset_out : // after 1st pkt
                        (latency_cnt >= LATENCY_CNT) ? LATENCY_CNT :
                                                       latency_cnt + 8'h1;
      end
    end
    if (DUAL_CLOCK == 1) begin
      dc_fifo_stub #(
        .DATA_WIDTH    ( LATENCY_CNT_WIDTH    ),
        .FIFO_DEPTH    ( 8                    )
      ) latency_buffer (
        .wrrst        ( in_rst                ),
        .rdrst        ( out_rst               ),
        // wr domain signals
        .wrclk        ( in_clk                ),
        .wren         ( wren_latency          ),
        .wrdin        ( latency_offset_in     ),
        .full         (                       ),
        .afull        (                       ),
        .over         (                       ),
        .wrcount      (                       ),
        // rd domain signals
        .rdclk        ( out_clk               ),
        .rden         ( rden_latency          ),
        .rddout       ( latency_offset_out    ),
        .rdval        ( rdval_latency         ),
        .empty        ( empty_latency         ),
        .aempty       (                       ),
        .under        (                       ),
        .rdcount      (                       )
      );

      reg_cdc # (
        .NBITS         ( 8      ),
        .REG_RST_VALUE ( '0     )
      ) u_latency_cdc (
        .i_a_clk ( out_clk      ),
        .i_a_rst ( out_rst      ),
        .i_a_val ( 1'b1         ),
        .i_a_reg ( latency_cnt  ),
        .i_b_clk ( in_clk       ),
        .i_b_rst ( in_rst       ),
        .o_b_val (              ),
        .o_b_reg ( latency_cnt_sync )
      );
    end
    else if (DUAL_CLOCK == 0) begin
      sc_fifo #(
        .DATA_WIDTH     ( LATENCY_CNT_WIDTH    ),
        .FIFO_DEPTH     ( 8                    )
      ) latency_buffer (
        .clk      ( in_clk                   ),
        .rst      ( in_rst                   ),
        .wr       ( wren_latency             ),
        .din      ( latency_offset_in        ),
        .full     (                          ),
        .afull    (                          ),
        .over     (                          ),
        .rd       ( rden_latency             ),
        .dout     ( latency_offset_out       ),
        .dval     ( rdval_latency            ),
        .empty    ( empty_latency            ),
        .aempty   (                          ),
        .under    (                          ),
        .count    (                          )
      );
      assign latency_cnt_sync = latency_cnt;
    end
  end
  else begin
    assign wait_latency = 1'b0;
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// SKID Buffer - to fix AXIS protocol and help with DC FIFO timing
//------------------------------------------------------------------------------------------------//

logic                                 reg_axis_tlast;
logic   [CENTER_WIDTH-1:0]            reg_axis_tdata;
logic   [(CENTER_WIDTH/8)-1:0]        reg_axis_tkeep;
logic   [CENTER_KEEP_CNT_WIDTH-1:0]   reg_axis_tkcnt;
logic                                 reg_axis_tvalid;
logic   [CENTER_WUSER-1:0]            reg_axis_tuser;
logic                                 reg_axis_tready;


axis_reg # (
  .DWIDTH             ( CENTER_WIDTH + CENTER_KEEP_CNT_WIDTH + CENTER_WUSER + 1          ),
  .SKID               ( OUTPUT_SKID                                                      )
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
logic   [OUT_W_USER-1:0]            w_axis_tx_tuser;
logic   [(OUT_DWIDTH/8)-1:0]        w_axis_tx_tkeep;

axis_gearbox #(
  .DIN_WIDTH  ( CENTER_WIDTH ),
  .DOUT_WIDTH ( OUT_DWIDTH   ),
  .W_USER     ( CENTER_WUSER )
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
