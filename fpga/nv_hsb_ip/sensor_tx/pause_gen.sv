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

module pause_gen
#(
    parameter   AXI_DWIDTH = 64,
    localparam  AXI_KWIDTH = AXI_DWIDTH/8
)(
    input                   pclk,
    input                   rst,
    input                   heartbeat,

    input                   i_start,
    input  [  15:0]         i_pause_time,
    input  [  47:0]         i_dev_mac_addr,
    output                  o_busy,
    // AXIS Interface
    output                  o_axis_tvalid,
    output [AXI_DWIDTH-1:0] o_axis_tdata,
    output                  o_axis_tlast,
    output                  o_axis_tuser,
    output [AXI_KWIDTH-1:0] o_axis_tkeep,
    input                   i_axis_tready
);

localparam [15:0] NUM_CLK_PER_QUANTA = ((512/AXI_DWIDTH)==0) ? 0 : (512/AXI_DWIDTH)-1;

localparam PAUSE_WIDTH = 144;

logic [PAUSE_WIDTH-1:0] pause_pkt;
logic [PAUSE_WIDTH-1:0] pause_pkt_be;

logic [47:0] dest_mac;
logic [47:0] src_mac;
logic [15:0] mac_ctrl;
logic [15:0] mac_opcode;
logic [15:0] mac_param;

assign dest_mac = 48'h0180_C200_0001;
assign src_mac  = i_dev_mac_addr;
assign mac_ctrl = 16'h8808;
assign mac_opcode = 16'h0001;
assign mac_param = i_start ? i_pause_time : '0;

assign pause_pkt = {dest_mac,src_mac,mac_ctrl,mac_opcode,mac_param};

genvar j;
generate
for (j=0; j<PAUSE_WIDTH/8; j++) begin
  assign pause_pkt_be[j*8+:8] = pause_pkt[(PAUSE_WIDTH/8-1-j)*8+:8];
end
endgenerate

typedef enum logic [2:0] {
  IDLE,
  LOAD,
  CNT_QUANTA,
  RESUME_TRAFFIC,
  DONE
} fsm;
fsm               state;

logic en_pause;
logic start_d1;

logic [15:0] clk_cnt_quanta;
logic [15:0] cnt_quanta;
logic        is_quanta_zero;

always_ff @(posedge pclk) begin
  if (rst) begin
    state           <= IDLE;
    en_pause        <= 1'b0;
  end
  else begin
    if ({start_d1, i_start} == 2'b10) begin
      state    <= RESUME_TRAFFIC;
      en_pause <= 1'b0;
    end
    else begin
      case(state)
        IDLE: begin
          state       <= i_start && ~o_busy ? LOAD : IDLE;
          en_pause    <= i_start && ~o_busy ? 1'b1 : 1'b0;
        end
        LOAD: begin
          state       <= |cnt_quanta ? CNT_QUANTA : LOAD;
          en_pause    <= 1'b1;
        end
        CNT_QUANTA: begin
          state       <= is_quanta_zero ? DONE : CNT_QUANTA;
          en_pause    <= 1'b1;
        end
        RESUME_TRAFFIC: begin
          state       <= DONE;
          en_pause    <= 1'b1;
        end
        DONE: begin
          state      <= IDLE;
          en_pause   <= 1'b0;
        end
        default: begin
          state      <= IDLE;
          en_pause   <= 1'b0;
        end
      endcase
    end
  end
end

always_ff @(posedge pclk) begin
  if (rst) begin
    start_d1 <= 1'b0;
  end
  else begin
    start_d1 <= i_start;
  end
end

always_ff @(posedge pclk) begin
  if (rst) begin
    cnt_quanta     <= '0;
    clk_cnt_quanta <= '0;
  end
  else begin
    if ({start_d1, i_start} == 2'b10) begin
      cnt_quanta <= '0;
    end
    else begin
      if (state == LOAD) begin
        cnt_quanta     <= i_pause_time;
        clk_cnt_quanta <= NUM_CLK_PER_QUANTA;
      end
      else if (state == CNT_QUANTA) begin
        if (clk_cnt_quanta == 'd0) begin
          clk_cnt_quanta <= NUM_CLK_PER_QUANTA;
          if (|cnt_quanta) begin
            cnt_quanta <= cnt_quanta - 16'h1;
          end
          else begin
            cnt_quanta <= '0;
          end
        end
        else begin
          clk_cnt_quanta <= clk_cnt_quanta - 1'b1;
        end
      end
    end
  end
end

assign is_quanta_zero = ~(|cnt_quanta);


vec_to_axis #(
  .AXI_DWIDTH       ( AXI_DWIDTH           ),
  .DATA_WIDTH       ( PAUSE_WIDTH          ),
  .PADDED_WIDTH     ( 60*8                 )
) pause_to_axis (
  .clk              ( pclk                 ),
  .rst              ( rst                  ),
  .trigger          ( en_pause             ),
  .data             ( pause_pkt_be         ),
  .is_busy          ( o_busy               ),
  .o_axis_tx_tvalid ( o_axis_tvalid        ),
  .o_axis_tx_tdata  ( o_axis_tdata         ),
  .o_axis_tx_tlast  ( o_axis_tlast         ),
  .o_axis_tx_tuser  ( o_axis_tuser         ),
  .o_axis_tx_tkeep  ( o_axis_tkeep         ),
  .i_axis_tx_tready ( i_axis_tready        )
);


//Debug logic to count number of pauses per second
logic [15:0] pause_cnt            /* synthesis noprune */;
logic [15:0] pause_per_sec_cnt    /* synthesis noprune */;
logic [2:0]  heartbeat_sync       /* synthesis noprune */;
logic        idle_reg             /* synthesis noprune */;

always_ff @(posedge pclk) begin
  if (rst) begin
    heartbeat_sync                <= '0;
    pause_cnt                     <= '0;
    pause_per_sec_cnt             <= '0;
    idle_reg                      <= 0;
  end
  else begin
    heartbeat_sync                <= {heartbeat_sync[1:0], heartbeat};
    idle_reg                      <= (state == IDLE);
    if (heartbeat_sync[1] && !heartbeat_sync[2]) begin
      pause_cnt                   <= '0;
      pause_per_sec_cnt           <= pause_cnt;
    end
    else if (state != IDLE && idle_reg) begin           //Count pauses at falling edge of IDLE state.
      pause_cnt                   <= pause_cnt + 1'b1;
    end
  end
end

//Debug logic to measure how low pause quanta gets.
logic [15:0]    lowest_pause_quanta /* synthesis noprune */;

always_ff @(posedge pclk) begin
  if (rst) begin
    lowest_pause_quanta           <= '1;
  end
  else begin
    if (state == CNT_QUANTA) begin
      lowest_pause_quanta         <= (cnt_quanta < lowest_pause_quanta) ? cnt_quanta : lowest_pause_quanta;
    end
  end
end


endmodule


