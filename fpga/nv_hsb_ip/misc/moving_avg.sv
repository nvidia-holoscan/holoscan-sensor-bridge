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

module moving_avg #(
  parameter W_DATA         = 32,
  parameter MAX_AVG_FACTOR = 3
) (
  input               i_clk,
  input               i_rst,
  input  [       1:0] i_cfg_avg_fact,

  input  [W_DATA-1:0] i_data,
  input               i_data_vld,

  output [W_DATA-1:0] o_data,
  output              o_data_vld
);

parameter AVG_DEPTH      = 2**MAX_AVG_FACTOR;
parameter AVG_DEPTH_BITS = $clog2(AVG_DEPTH);

logic [W_DATA+AVG_DEPTH_BITS-1:0] data_sum;
logic [W_DATA+AVG_DEPTH_BITS-1:0] data_sum_r;
logic [         AVG_DEPTH_BITS:0] data_cnt;
logic [               W_DATA-1:0] avg_data;
logic                             avg_data_vld;
logic [                      1:0] avg_fact;
logic [         AVG_DEPTH_BITS:0] avg_cnt;
logic [W_DATA+AVG_DEPTH_BITS-1:0] data_reg;
logic                             is_cnt;

logic                             fifo_wren;
logic                             fifo_rden;
logic                             fifo_rdval;
logic                             fifo_rst;
logic [               W_DATA-1:0] fifo_rddata;

enum logic [2:0] {IDLE, SUM, SUB, AVG} state;

assign is_cnt = (avg_cnt == data_cnt) ? 1'b1 : 1'b0;
assign data_sum = data_sum_r + data_reg;

always_ff @(posedge i_clk) begin
  if (i_rst) begin
    data_sum_r   <= 'd0;
    data_cnt     <= 'd0;
    data_reg     <= 'd0;
    avg_data     <= 'd0;
    avg_data_vld <= 1'b0;
    avg_fact     <= 'd0;
    avg_cnt      <= 'd0;
    fifo_wren    <= 1'b0;
    fifo_rst     <= 1'b0;
    state        <= IDLE;
  end
  else begin
    data_sum_r <= data_sum;
    data_reg   <= 'd0;

    case (state)
    IDLE: begin
      fifo_wren    <= 1'b0;
      fifo_rst     <= 1'b0;
      avg_data_vld <= 1'b0;
      avg_cnt      <= 1'b1 << i_cfg_avg_fact;
      avg_fact     <= i_cfg_avg_fact;

      if (i_cfg_avg_fact != avg_fact) begin
        data_sum_r <= 'd0;
        fifo_rst   <= 1'b1;
        data_cnt   <= 'd0;
      end

      if (i_data_vld && !is_cnt) begin
        data_cnt <= data_cnt + 1'b1;
      end

      if (i_cfg_avg_fact == 'd0) begin
        avg_data     <= i_data;
        avg_data_vld <= i_data_vld;
        state        <= IDLE;
      end
      else begin
        if (i_data_vld) begin
          data_reg  <= i_data;
          fifo_wren <= 1'b1;
          state     <= SUM;
        end
      end
    end
    SUM: begin
      fifo_wren <= 1'b0;
      fifo_rst  <= 1'b0;
      if (is_cnt) begin
        state <= AVG;
      end
      else begin
        state <= IDLE;
      end
    end
    AVG: begin
      data_reg     <= -signed'({1'b0,fifo_rddata});
      avg_data     <= data_sum_r >> avg_fact;
      avg_data_vld <= 1'b1;
      state        <= SUB;
    end
    SUB: begin
      avg_data_vld <= 1'b0;
      state        <= IDLE;
    end
    endcase
  end
end

assign fifo_rden = is_cnt && (state == SUM);

sc_fifo #(
  .DATA_WIDTH     ( W_DATA              ),    //tdata + tkeep + tlast + tuser
  .FIFO_DEPTH     ( AVG_DEPTH           ),
  .ALMOST_FULL    ( AVG_DEPTH - 2       ),
  .ALMOST_EMPTY   ( 2                   ),
  .MEM_STYLE      ( "AUTO"              )
) dly_buffer (
  .clk            ( i_clk               ),
  .rst            ( i_rst || fifo_rst   ),
  .wr             ( fifo_wren           ),
  .din            ( data_reg[W_DATA-1:0]),
  .full           (                     ),
  .afull          (                     ),
  .over           (                     ),
  .rd             ( fifo_rden           ),
  .dout           ( fifo_rddata         ),
  .dval           ( fifo_rdval          ),
  .empty          (                     ),
  .aempty         (                     ),
  .under          (                     ),
  .count          (                     )
);

assign o_data     = avg_data;
assign o_data_vld = avg_data_vld;

endmodule
