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

module dp_pkt
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter  W_DATA     = 64,
  parameter  W_KEEP     = 8,
  parameter  N_INPT     = 4,
  parameter  W_INPT     = $clog2(N_INPT) + 1,
  parameter  MTU        = 1500,
  parameter  W_CNT      = $clog2(MTU)+1
)(
  // clock and reset
  input                             i_pclk,
  input                             i_prst,
  // Control
  input     [W_CNT-1:0]             i_pkt_len,
  input                             i_pkt_len_en,
  input     [W_INPT-1:0]            i_gnt_idx,
  output    [N_INPT-1:0]            o_tlast,
  input                             i_tlast_done,
  output                            o_unsync,
  output    [W_CNT-1:0]             o_unsync_cnt,
  // CRC Handshake
  input     [31:0]                  i_crc,
  output    [31:0]                  o_crc,
  output    [$clog2(N_INPT):0]      o_crc_idx,
  output                            o_crc_valid,
  // Input Data
  input     [N_INPT-1:0]            i_axis_tvalid,
  input     [N_INPT-1:0]            i_axis_tlast,
  input     [W_DATA-1:0]            i_axis_tdata [N_INPT-1:0],
  input     [W_KEEP-1:0]            i_axis_tkeep [N_INPT-1:0],
  input     [N_INPT-1:0]            i_axis_tuser,
  output    [N_INPT-1:0]            o_axis_tready,
  //DP Packet AXIS
  output                            o_axis_tvalid,
  output                            o_axis_tlast,
  output    [W_DATA-1:0]            o_axis_tdata,
  output    [W_KEEP-1:0]            o_axis_tkeep,
  output                            o_axis_tuser,
  input                             i_axis_tready
);

//------------------------------------------------------------------------------------------------//
// AXIS ARB
//------------------------------------------------------------------------------------------------//
localparam NUM_PAD_BITS    = 7; //128B alignment
localparam CYCLES_TO_PAGES = NUM_PAD_BITS-($clog2(W_KEEP));
localparam WIDE_DATA       = (W_DATA > 512) ? 1 : 0;

logic                       arb_axis_tvalid;
logic                       arb_axis_tlast ;
logic     [W_DATA-1:0]      arb_axis_tdata ;
logic     [W_KEEP-1:0]      arb_axis_tkeep ;
logic                       arb_axis_tuser ;

logic                       pkt_active;
logic                       pkt_pad;
logic                       pkt_last;
logic                       pkt_idle;

logic [W_INPT-1:0]          gnt_idx;
logic [W_CNT-1:0]           pkt_len;
logic [W_CNT-1:0]           pkt_cnt;
logic [W_CNT-1:0]           pkt_pad_cnt;
logic [N_INPT-1:0]          sif_tlast;
logic                       unsync;

logic [W_CNT-1:0]           unsync_cnt;
logic [CYCLES_TO_PAGES-1:0] last_cycle_pad;



always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    pkt_active  <= '0;
    gnt_idx     <= '0;
    pkt_pad     <= '0;
    pkt_cnt     <= 'd1;
    pkt_pad_cnt <= '0;
    sif_tlast   <= '0;
    unsync_cnt  <= '0;
    unsync      <= '0;
    pkt_len     <= '0;
  end
  else begin
    unsync_cnt     <= pkt_pad_cnt + last_cycle_pad;
    if (pkt_active) begin
      if (i_axis_tready) begin
        if (i_axis_tlast [gnt_idx]) begin
          sif_tlast[gnt_idx] <= '1;
        end
        if (pkt_cnt == pkt_len) begin
          pkt_active <= '0;
          pkt_cnt    <= 'd1;
        end
        else begin
          pkt_cnt <= pkt_cnt + 1;
          unsync  <= (i_axis_tlast [gnt_idx]) ? '1 : unsync;
        end
        pkt_pad     <= (i_axis_tlast [gnt_idx]) ? '1 : pkt_pad;
        pkt_pad_cnt <= (pkt_pad) ? pkt_pad_cnt + 1 : pkt_pad_cnt;
      end
    end
    else begin
      pkt_cnt <= 'd1;
      if (i_tlast_done) begin
        sif_tlast  <= '0;
      end
      if (i_pkt_len_en) begin
        pkt_active  <= '1;
        pkt_pad_cnt <= 'd0;
        gnt_idx     <= i_gnt_idx;
        pkt_len     <= i_pkt_len;
        pkt_pad     <= '0;
        unsync      <= '0;
      end
    end
  end
end

generate
  if (WIDE_DATA) begin
    assign last_cycle_pad  = '0;
  end else begin
    assign last_cycle_pad  = ~pkt_len[0+:CYCLES_TO_PAGES] + 1;
  end
endgenerate

assign pkt_last        = (pkt_cnt == pkt_len) && pkt_active;
assign arb_axis_tvalid = pkt_active;
assign arb_axis_tlast  = pkt_last;
assign arb_axis_tdata  = (pkt_pad || !pkt_active) ? '0 : i_axis_tdata  [gnt_idx];
assign arb_axis_tkeep  = '1;
assign arb_axis_tuser  = '0;

assign o_tlast      = sif_tlast;
assign o_unsync_cnt = unsync_cnt;
assign o_unsync     = unsync;

//------------------------------------------------------------------------------------------------//
// Frame CRC
//------------------------------------------------------------------------------------------------//

logic [31:0]       crc_in;
logic [31:0]       crc_out;
logic [W_DATA-1:0] crc_axis_tdata;
logic              crc_val;
logic              crc_valid;

logic [1:0] crc_ld;

lfsr_hc #(
  .DATA_WIDTH(W_DATA)
) frame_crc_inst (
  .data_in  (crc_axis_tdata ),
  .state_in (crc_in         ),
  .state_out(crc_out        )
);

always_ff @(posedge i_pclk) begin
  crc_axis_tdata    <= i_axis_tdata  [gnt_idx];
end

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    crc_in    <= '1;
    crc_val   <= '0;
    crc_ld    <= '0;
    crc_valid <= '0;
  end
  else begin
    crc_valid <= (i_pkt_len_en) ? '1 : crc_valid;
    crc_ld    <= {crc_ld[0], i_pkt_len_en};  // Takes 2 cycles to latch crc_idx, then 1 cycle to load crc_in
    crc_in    <= crc_val                ? crc_out: // Advance
                 (crc_ld[1:0] == 2'b10) ? i_crc  : // Load
                                           crc_in ; // Hold
    crc_val        <=  (arb_axis_tvalid && i_axis_tready && pkt_active && !pkt_pad);
  end
end

assign o_crc       = crc_in;
assign o_crc_idx   = gnt_idx;
assign o_crc_valid = crc_valid;
//------------------------------------------------------------------------------------------------//
// Output
//------------------------------------------------------------------------------------------------//

assign o_axis_tvalid   = arb_axis_tvalid;
assign o_axis_tdata    = arb_axis_tdata;
assign o_axis_tlast    = arb_axis_tlast;
assign o_axis_tuser    = arb_axis_tuser;
assign o_axis_tkeep    = arb_axis_tkeep;

logic [N_INPT-1:0] tready;
always_comb begin
  tready = '0;
  for (int i = 0; i < N_INPT; i++) begin
    if (i == gnt_idx) begin
      tready[i] = i_axis_tready & pkt_active & !pkt_pad;
    end
  end
end

assign o_axis_tready = tready;

endmodule
