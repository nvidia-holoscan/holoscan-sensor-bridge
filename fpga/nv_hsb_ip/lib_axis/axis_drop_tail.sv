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

module axis_drop_tail #(
  parameter DROP_WIDTH = 32,
  parameter DWIDTH     = 64,
  parameter W_USER     = 1,
  parameter W_KEEP     = DWIDTH/8
)
(
  input   logic                               i_clk,
  input   logic                               i_rst,

  input   logic                               i_axis_rx_tvalid,
  input   logic   [DWIDTH-1:0]                i_axis_rx_tdata,
  input   logic                               i_axis_rx_tlast,
  input   logic   [W_USER-1:0]                i_axis_rx_tuser,
  input   logic   [W_KEEP-1:0]                i_axis_rx_tkeep,

  output  logic                               o_axis_tx_tvalid,
  output  logic   [DWIDTH-1:0]                o_axis_tx_tdata,
  output  logic                               o_axis_tx_tlast,
  output  logic   [W_USER-1:0]                o_axis_tx_tuser,
  output  logic   [W_KEEP-1:0]                o_axis_tx_tkeep
);

localparam           DROP_CYCLES = ((DROP_WIDTH-1)/DWIDTH) + 2;
localparam           DROP_MOD    = (DROP_WIDTH % DWIDTH) / 8;

//------------------------------------------------------------------------------------------------//
// Lookback Buffer
//------------------------------------------------------------------------------------------------//

logic [DROP_CYCLES-1:0] reg_axis_tvalid;
logic [DWIDTH-1:0]      reg_axis_tdata  [DROP_CYCLES];
logic [DROP_CYCLES-1:0] reg_axis_tlast;
logic [W_USER-1:0]      reg_axis_tuser  [DROP_CYCLES];
logic [W_KEEP-1:0]      reg_axis_tkeep  [DROP_CYCLES];


always_ff @(posedge i_clk) begin
  if (i_rst) begin
    reg_axis_tvalid    <= '{default:'0};
    reg_axis_tdata     <= '{default:'0};
    reg_axis_tlast     <= '{default:'0};
    reg_axis_tuser     <= '{default:'0};
    reg_axis_tkeep     <= '{default:'0};
  end
  else begin
    for (int i=0; i<DROP_CYCLES; i++) begin
      if (i==0) begin
        reg_axis_tvalid[0]      <= i_axis_rx_tvalid;
        reg_axis_tdata [0]      <= i_axis_rx_tdata;
        reg_axis_tlast [0]      <= i_axis_rx_tlast;
        reg_axis_tuser [0]      <= i_axis_rx_tuser;
        reg_axis_tkeep [0]      <= i_axis_rx_tkeep;
      end
      else begin
        reg_axis_tvalid[i]      <= reg_axis_tvalid[i-1];
        reg_axis_tdata [i]      <= reg_axis_tdata [i-1];
        reg_axis_tlast [i]      <= reg_axis_tlast [i-1];
        reg_axis_tuser [i]      <= reg_axis_tuser [i-1];
        reg_axis_tkeep [i]      <= reg_axis_tkeep [i-1];
      end
    end
  end
end

//------------------------------------------------------------------------------------------------//
// Drop Logic
//------------------------------------------------------------------------------------------------//

logic [2:0]             tlast_seen;

logic [W_KEEP-1:0]      tkeep_calc;
logic [W_KEEP-1:0]      tkeep_remaining;
logic                   drop_full_cycle;
logic                   cycle_is_valid;

// Pending latch: captures a new packet's tlast/tkeep when tlast_seen[0] is
// already set but the previous packet hasn't exited the module yet. Used for
// back to back packet scenarios.
logic                   tlast_pending;
logic [W_KEEP-1:0]      tkeep_pending;

always_ff @(posedge i_clk) begin
  if (i_rst) begin
    tlast_seen      <= '0;
    tkeep_calc      <= '0;
    tkeep_remaining <= '0;
    tlast_pending   <= '0;
    tkeep_pending   <= '0;
  end
  else begin
    tlast_seen[2:1] <= tlast_seen[1:0];
    if (tlast_seen[0]) begin
      if (reg_axis_tlast[DROP_CYCLES-1]) begin
        // Previous packet's tlast is exiting the pipeline this cycle.
        if (tlast_pending) begin
          // Service the pending tlast first (it entered the pipeline earlier).
          // Its data is already at reg[DROP_CYCLES-1], so advance tlast_seen
          // to 3'b011 to match on this cycle (for non-drop_full_cycle) or
          // to allow the pattern to progress correctly.
          tlast_seen <= 3'b011;
          {tkeep_calc,tkeep_remaining}  <= ({tkeep_pending,{W_KEEP{1'b1}}} >> (DROP_MOD));
          // If a new input tlast also arrives, it becomes the new pending.
          if (i_axis_rx_tvalid & i_axis_rx_tlast) begin
            tkeep_pending <= i_axis_rx_tkeep;
            // tlast_pending stays 1
          end
          else begin
            tlast_pending <= '0;
          end
        end
        else if (i_axis_rx_tvalid & i_axis_rx_tlast) begin
          // New packet tlast arriving, no pending – re-arm immediately
          tlast_seen <= 1'b1;
          {tkeep_calc,tkeep_remaining}  <= ({i_axis_rx_tkeep,{W_KEEP{1'b1}}} >> (DROP_MOD));
        end
        else begin
          tlast_seen <= '0;
          tkeep_calc <= '0;
        end
      end
      else begin
        // Previous packet's tlast hasn't exited the pipeline yet.
        // If a new packet's tlast arrives now, capture it as pending.
        if (i_axis_rx_tvalid & i_axis_rx_tlast & !tlast_pending) begin
          tlast_pending <= 1'b1;
          tkeep_pending <= i_axis_rx_tkeep;
        end
      end
    end
    else begin
      if (i_axis_rx_tvalid & i_axis_rx_tlast) begin
        tlast_seen[0]                 <= '1;
        {tkeep_calc,tkeep_remaining}  <= ({i_axis_rx_tkeep,{W_KEEP{1'b1}}} >> (DROP_MOD));
      end
    end
  end
end

assign drop_full_cycle = (!tkeep_calc[0]) || (DROP_MOD == 0);
assign cycle_is_valid  = ((tlast_seen[0] && drop_full_cycle) ? o_axis_tx_tlast : '1);

//------------------------------------------------------------------------------------------------//
// Output assignments
//------------------------------------------------------------------------------------------------//


assign o_axis_tx_tvalid = reg_axis_tvalid [DROP_CYCLES-1] & cycle_is_valid;
assign o_axis_tx_tdata  = reg_axis_tdata  [DROP_CYCLES-1];
assign o_axis_tx_tlast  = (drop_full_cycle ? (tlast_seen[2:0] == 'b001) : (tlast_seen[2:0] == 'b011));
assign o_axis_tx_tuser  = reg_axis_tuser  [DROP_CYCLES-1];
assign o_axis_tx_tkeep  = (!o_axis_tx_tlast) ? '1              :             // if not last, keep all bits
                          (drop_full_cycle)  ? tkeep_remaining : tkeep_calc; // if drop full cycle, use remaining bits

endmodule
