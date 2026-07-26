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
  input                             i_axis_tready,
  // Early Packet Detection
  output                            o_hif_not_busy_soon
);

//------------------------------------------------------------------------------------------------//
// AXIS ARB
//------------------------------------------------------------------------------------------------//
localparam NUM_PAD_BITS    = 7; //128B alignment
localparam CYCLES_TO_PAGES = NUM_PAD_BITS-($clog2(W_KEEP));
localparam WIDE_DATA       = (W_DATA > 512) ? 1 : 0;
localparam CYCLES_AHEAD       = 10; // assert when (pkt_len - pkt_cnt) < CYCLES_AHEAD
localparam MIN_OVERLAP_CYCLES = 8;  // power of 2; pkt_len >= this <=> |pkt_len[W_CNT-1:MIN_OVERLAP_IDX]
localparam MIN_OVERLAP_IDX    = $clog2(MIN_OVERLAP_CYCLES);

logic                       arb_axis_tvalid;
logic                       arb_axis_tlast ;
logic     [W_DATA-1:0]      arb_axis_tdata ;
logic     [W_KEEP-1:0]      arb_axis_tkeep ;

logic                       pkt_active;
logic                       pkt_tready;
logic                       pkt_pad;
logic                       pkt_last;

logic [W_INPT-1:0]          gnt_idx;
logic [N_INPT-1:0]          gnt_onehot;
logic [W_CNT-1:0]           pkt_len;
logic [W_CNT-1:0]           pkt_cnt;
logic [W_CNT-1:0]           pkt_pad_cnt;
logic [N_INPT-1:0]          sif_tlast;
logic                       unsync;

logic [W_CNT-1:0]           unsync_cnt;
logic [CYCLES_TO_PAGES-1:0] last_cycle_pad;
logic                       hif_not_busy_soon;
logic                       near_pkt_end;


always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    near_pkt_end <= 1'b0;
  end
  else if (!pkt_active || pkt_pad) begin
    near_pkt_end <= 1'b0;
  end
  else begin
    near_pkt_end <= |pkt_len[W_CNT-1:MIN_OVERLAP_IDX] &&
                    ((pkt_len - pkt_cnt) < W_CNT'(CYCLES_AHEAD));
  end
end

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    hif_not_busy_soon <= 1'b0;
  end
  else if (!pkt_active || pkt_pad) begin
    hif_not_busy_soon <= 1'b0;
  end
  else begin
    hif_not_busy_soon <= near_pkt_end;
  end
end


always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    gnt_onehot <= '0;
  end
  else begin
    if (!pkt_active && i_pkt_len_en) begin
      gnt_onehot <= (1 << i_gnt_idx);
    end
    else begin
      gnt_onehot <= gnt_onehot;
    end
  end
end

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    pkt_active        <= '0;
    pkt_tready        <= '0;
    gnt_idx           <= '0;
    pkt_pad           <= '0;
    pkt_cnt           <= 'd1;
    pkt_pad_cnt       <= '0;
    sif_tlast         <= '0;
    unsync_cnt        <= '0;
    unsync            <= '0;
    pkt_len           <= '0;
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
          pkt_tready <= '0;
          pkt_cnt    <= 'd1;
        end
        else begin
          pkt_cnt <= pkt_cnt + 1;
          unsync  <= (i_axis_tlast [gnt_idx]) ? '1 : unsync;
        end
        pkt_pad     <= (i_axis_tlast [gnt_idx]) ? '1 : pkt_pad;
        pkt_tready  <= ((pkt_cnt == pkt_len) || (i_axis_tlast [gnt_idx])) ? '0 : pkt_tready;
        pkt_pad_cnt <= (pkt_pad) ? pkt_pad_cnt + 1 : pkt_pad_cnt;
      end
    end
    else begin
      pkt_cnt           <= 'd1;
      if (i_tlast_done) begin
        sif_tlast <= '0;
      end
      if (i_pkt_len_en) begin
        pkt_active  <= '1;
        pkt_tready  <= '1;
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

assign o_hif_not_busy_soon = hif_not_busy_soon;

assign pkt_last        = (pkt_cnt == pkt_len) && pkt_active;
assign arb_axis_tvalid = pkt_active;
assign arb_axis_tlast  = pkt_last;
assign arb_axis_tdata  = (pkt_pad || !pkt_active) ? '0 : i_axis_tdata  [gnt_idx];
assign arb_axis_tkeep  = '1;

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
assign o_axis_tkeep    = arb_axis_tkeep;

assign o_axis_tready = gnt_onehot & {N_INPT{i_axis_tready & pkt_tready}};

endmodule
