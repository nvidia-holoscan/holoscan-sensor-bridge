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

module axis_to_vec
#(
    parameter                               AXI_DWIDTH   = 64,
    parameter                               DATA_WIDTH   = 288,
    localparam                              W_KEEP       = (AXI_DWIDTH/8),
    localparam                              CNT_WIDTH    = $clog2(DATA_WIDTH/8)
)(
    input                                   clk,
    input                                   rst,
  //AXIS Interface
    input  logic                            i_axis_rx_tvalid,
    input  logic   [AXI_DWIDTH-1:0]         i_axis_rx_tdata,
    input  logic                            i_axis_rx_tlast,
    input  logic                            i_axis_rx_tuser,
    input  logic   [W_KEEP-1:0]             i_axis_rx_tkeep,
    output                                  o_axis_rx_tready,
    //Control
    input                                   i_done,
    output   [DATA_WIDTH-1:0]               o_data,
    output                                  o_valid,
    output                                  o_busy,
    output   [CNT_WIDTH-1:0]                o_byte_cnt
);

localparam D_CYCLES                     = (DATA_WIDTH-1)/AXI_DWIDTH + 1;

logic [CNT_WIDTH-1:0]         byte_cnt;
logic [ $clog2(D_CYCLES):0]   cnt;
logic                         pipe_incr;
logic                         early_pkt;
logic                         early_pkt_r;
logic [ $clog2(W_KEEP)-1:0]   byte_incr;
logic                         busy_r;

typedef enum logic [3:0] {
  A2V_IDLE = 4'b0001,
  A2V_DONE = 4'b0010,
  A2V_WAIT = 4'b0100,
  A2V_HOLD = 4'b1000
} a2v_states;
a2v_states a2v_state, a2v_state_nxt;

//------------------------------------------------------------------------------------------------//
// Buffer
//------------------------------------------------------------------------------------------------//

logic [AXI_DWIDTH-1:0]            data_buffer [D_CYCLES];
logic [(D_CYCLES*AXI_DWIDTH)-1:0] w_data;

genvar i;
generate
for (i=0;i<D_CYCLES;i=i+1) begin
  always_ff @(posedge clk) begin
    if (rst) begin
      data_buffer[i] <= '0;
    end
    else begin
      if (((a2v_state_nxt == A2V_IDLE) && (a2v_state != A2V_IDLE)) || (early_pkt_r && (i != 0))) begin // Reset pipe
        data_buffer[i] <= '0;
      end
      else if (pipe_incr) begin
        data_buffer[i] <= (cnt == i) ? i_axis_rx_tdata : data_buffer[i];
      end
    end
  end
end
endgenerate

assign pipe_incr = (i_axis_rx_tvalid && (a2v_state == A2V_IDLE));

//------------------------------------------------------------------------------------------------//
// FSM
//------------------------------------------------------------------------------------------------//

always_comb begin
  a2v_state_nxt = a2v_state;
  early_pkt     = '0;
  case(a2v_state)
    A2V_IDLE:   begin
      if (i_axis_rx_tvalid) begin
        early_pkt     = (i_axis_rx_tlast && (i_done));
        a2v_state_nxt = (i_axis_rx_tlast && (i_done))  ? A2V_IDLE:
                        (i_axis_rx_tlast && (!i_done)) ? A2V_HOLD:
                        (cnt == D_CYCLES-1)            ? A2V_WAIT:
                                                         A2V_IDLE;
      end
    end
    A2V_DONE: begin
      a2v_state_nxt = (i_axis_rx_tlast && i_axis_rx_tvalid) ? A2V_IDLE : A2V_DONE;
    end
    A2V_WAIT: begin
      a2v_state_nxt = (i_axis_rx_tlast && i_axis_rx_tvalid && i_done)  ? A2V_IDLE :
                      (i_axis_rx_tlast && i_axis_rx_tvalid && !i_done) ? A2V_HOLD :
                      (i_done)                                         ? A2V_DONE :
                                                                         A2V_WAIT ;
    end
    A2V_HOLD: begin
      a2v_state_nxt = (i_done) ? A2V_IDLE : A2V_HOLD;
    end
    default: begin
      a2v_state_nxt = A2V_IDLE;
    end
  endcase
end

always_ff @(posedge clk) begin
  if (rst) begin
    a2v_state   <= A2V_IDLE;
    byte_cnt    <= '0;
    cnt         <= '0;
    early_pkt_r <= '0;
    busy_r      <= '0;
  end
  else begin
    a2v_state   <= a2v_state_nxt;
    early_pkt_r <= early_pkt;
    busy_r      <= (a2v_state_nxt != A2V_IDLE);
    if (pipe_incr) begin
      cnt      <= (early_pkt) ? '0 : cnt + 1'b1;
      byte_cnt <= (early_pkt_r) ? byte_incr : byte_incr + byte_cnt;
    end
    else if (early_pkt_r) begin
      byte_cnt <= '0;
      cnt      <= '0;
    end
    else if ((a2v_state_nxt == A2V_IDLE) && (a2v_state != A2V_IDLE)) begin
      byte_cnt <= '0;
      cnt      <= '0;
    end
  end
end



//------------------------------------------------------------------------------------------------//
// Output
//------------------------------------------------------------------------------------------------//


genvar j;
generate
  if (DATA_WIDTH < AXI_DWIDTH) begin
    assign w_data = data_buffer[0][0+:DATA_WIDTH];
    assign byte_incr = $countones(i_axis_rx_tkeep[0+:(DATA_WIDTH/8)]);
  end
  else begin
    for (j=0; j<D_CYCLES; j=j+1) begin
      assign w_data[j*AXI_DWIDTH+:AXI_DWIDTH] = data_buffer[j];
    end
    assign byte_incr = $countones(i_axis_rx_tkeep);
  end
endgenerate

assign o_axis_rx_tready = (a2v_state_nxt == A2V_IDLE) || (a2v_state_nxt == A2V_DONE) || (a2v_state_nxt == A2V_WAIT);
assign o_valid = ((a2v_state == A2V_WAIT) || (a2v_state == A2V_HOLD) || (early_pkt_r));
assign o_busy = busy_r;
assign o_byte_cnt = byte_cnt;
assign o_data = w_data[DATA_WIDTH-1:0];

endmodule
