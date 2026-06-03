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


module virtual_port # (
  parameter          VP_SIZE     = 64,
  parameter          DIN_WIDTH   = 64,
  parameter          DOUT_WIDTH  = 64,
  parameter          DSP_SHIFT   = 1,
  parameter          RESOLUTION  = 2,
  parameter          DYNAMIC_VP  = 0,
  parameter          CTRL_WIDTH  = 1,
  parameter          MTU         = 1500,
  localparam integer AE_FLAG     = (((MTU-'d79) / (DOUT_WIDTH/8))+1),
  localparam integer WR_VP_DEPTH = AE_FLAG*2,
  localparam integer RD_VP_DEPTH = AE_FLAG*2,
  localparam         DOUT_KWIDTH = DOUT_WIDTH/8,
  localparam integer AF_FLAG     = (((WR_VP_DEPTH+1) / 8) * 7)
) (
  // Sensor
    input                           i_sclk,
    input                           i_srst,
  // Ethernet
    input                           i_pclk,
    input                           i_prst,
  // Data Input
    input  [DIN_WIDTH-1:0]          din,
    input                           din_valid,
    input                           din_tlast,
  // Configuration
    input  [CTRL_WIDTH-1:0]         ctrl,
    output                          is_empty,
    output                          almost_full,
  // Data Output
    output                          o_axis_tvalid,
    output                          o_axis_tlast,
    output [DOUT_WIDTH-1:0]         o_axis_tdata,
    output [DOUT_KWIDTH-1:0]        o_axis_tkeep,
    output                          o_axis_tuser,
    input                           i_axis_tready
);

logic [VP_SIZE-1   :0] vp_wr_data;
logic                  wr_en;
logic                  din_tlast_r;

generate
  if (DYNAMIC_VP==1) begin
//------------------------------------------------------------------------------------------------//
// Dynamic VP - Committing to VP can be of any size up to VP Size, defined by resolution
//------------------------------------------------------------------------------------------------//
    logic  [$clog2(DIN_WIDTH)-1:0]  start;
    logic  [$clog2(VP_SIZE)-1:0]    len  ;

    assign {len,start} = ctrl;

//------------------------------------------------------------------------------------------------//
// Left Shift
//------------------------------------------------------------------------------------------------//

    logic [$clog2(VP_SIZE):0] cnt, cnt_r;


    logic [VP_SIZE*2-1:0]                 shifted_data;
    logic [$clog2(DIN_WIDTH+VP_SIZE)-1:0] shift;
    logic [7:0]                           select;
    logic [VP_SIZE*2-1:0]                 select_data;
    logic [VP_SIZE*3-1:0]                 tmp_shifted_data;

    if (DSP_SHIFT == 1) begin
      logic [DIN_WIDTH+VP_SIZE-1:0]  tmp_shifted_data;
      logic [15:0]                   mult_const;
      logic [DIN_WIDTH+16-1:0]       dsp_result;
      assign dsp_result = din*mult_const;                   // DSP
      assign tmp_shifted_data = dsp_result << ({shift[$clog2(DIN_WIDTH+VP_SIZE)-1:4],4'h0});
      assign shifted_data = tmp_shifted_data[DIN_WIDTH+VP_SIZE-1-:VP_SIZE*2];
      assign mult_const = 2**(shift[3:0]);
    end
    else begin
      assign select = (start<<($clog2(RESOLUTION))) >> ($clog2(VP_SIZE));
      assign select_data = din[select*VP_SIZE+:VP_SIZE*2];
      assign tmp_shifted_data = select_data << shift[0+:$clog2(VP_SIZE)+1];
      assign shifted_data = tmp_shifted_data[VP_SIZE*3-1-:VP_SIZE*2];
    end


    assign shift = DIN_WIDTH+VP_SIZE - cnt_r - (len<<($clog2(RESOLUTION))) - (start<<($clog2(RESOLUTION)));

//------------------------------------------------------------------------------------------------//
// Data In
//------------------------------------------------------------------------------------------------//

    logic [VP_SIZE:0]       shifted_valid;
    logic [VP_SIZE-1:0]     vp_buffer, vp_buffer_r;
    logic [(VP_SIZE*2)-1:0] vp_output;
    logic                   din_valid_prev;

    always_ff @(posedge i_sclk) begin
      if (i_srst) begin
        vp_buffer_r     <= '0;
        vp_wr_data      <= '0;
        wr_en           <= '0;
        din_valid_prev  <= '0;
        din_tlast_r     <= '0;
        cnt_r           <= '0;
      end
      else begin
        vp_buffer_r     <= vp_buffer;
        vp_wr_data      <= vp_output[VP_SIZE+:VP_SIZE];
        wr_en           <= (cnt >= VP_SIZE) & din_valid;
        din_valid_prev  <= din_valid;
        din_tlast_r     <= din_tlast;
        cnt_r           <= (din_valid) ? {1'b0,cnt[$clog2(VP_SIZE)-1:0]} : cnt_r;
      end
    end

    integer i;

    always_comb begin
      // Default
      vp_buffer = vp_buffer_r;
      cnt       = cnt_r;
      // Shift Data

      // Load Data
      shifted_valid = {{VP_SIZE{1'b1}},{VP_SIZE{1'b0}}} >> cnt[$clog2(VP_SIZE)-1:0];
      for (i=VP_SIZE;i<(VP_SIZE*2);i=i+1) begin
        vp_output[i] = shifted_valid[i-VP_SIZE] ? vp_buffer[i-VP_SIZE] : shifted_data[i];
      end
      vp_output[VP_SIZE-1:0] = shifted_data[0+:VP_SIZE];
      cnt            = cnt + (len<<($clog2(RESOLUTION)));
      // Shift Data Out. Overflow buffer is loaded if data is written, or current buffer is kept
      vp_buffer = (cnt >= VP_SIZE) ? vp_output[0+:VP_SIZE] : vp_output[VP_SIZE+:VP_SIZE];
    end
  end
  else begin

//------------------------------------------------------------------------------------------------//
// Static VP - Committing to FIFO has to be equal to Size of VP
//------------------------------------------------------------------------------------------------//
  logic [($clog2(DIN_WIDTH/VP_SIZE))-1:0] select;
  logic                                   commit;


  always_ff @(posedge i_sclk) begin
    if (i_srst) begin
      vp_wr_data      <= '0;
      wr_en           <= '0;
      din_tlast_r     <= '0;
    end
    else begin
      vp_wr_data      <= din[select*VP_SIZE+:VP_SIZE];
      wr_en           <= commit && din_valid;
      din_tlast_r     <= din_tlast;
    end
  end

  assign {select,commit} = ctrl;

  end
endgenerate
//------------------------------------------------------------------------------------------------//
// FIFO
//------------------------------------------------------------------------------------------------//

logic [1:0] tlast_buffered; // Tlast has been buffered, allow tlast to be sent even if not full Eth packet of data is buffered
logic       in_is_last;
logic       in_is_last_sync;
logic       out_is_last;
logic       w_axis_tvalid;

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    out_is_last    <= '0;
    tlast_buffered <= '0;
  end
  else begin
    out_is_last    <= (w_axis_tvalid && o_axis_tlast && i_axis_tready);
    tlast_buffered <= tlast_buffered + in_is_last_sync - out_is_last;
  end
end

assign in_is_last = (din_tlast_r && wr_en);

pulse_sync u_in_is_last_sync (
  .src_clk     ( i_sclk          ),
  .src_rst     ( i_srst          ),
  .dst_clk     ( i_pclk          ),
  .dst_rst     ( i_prst          ),
  .i_src_pulse ( in_is_last      ),
  .o_dst_pulse ( in_is_last_sync )
);

logic fifo_aempty;
logic fifo_full;

axis_buffer #(
  .IN_DWIDTH         ( VP_SIZE         ), // bit width of input data
  .OUT_DWIDTH        ( DOUT_WIDTH      ), // bit width of output data
  .BUF_DEPTH         ( WR_VP_DEPTH     ),
  .WAIT2SEND         ( 0               ),
  .DUAL_CLOCK        ( 1               ),
  .ALMOST_FULL_DEPTH ( AF_FLAG         ),
  .OUTPUT_SKID       ( 0               ),
  .ALMOST_EMPTY_DEPTH( AE_FLAG         ),
  .NO_BACKPRESSURE   ( 1               )
) axis_buffer (
  .in_clk           ( i_sclk          ),
  .in_rst           ( i_srst          ),
  .out_clk          ( i_pclk          ),
  .out_rst          ( i_prst          ),
  .i_axis_rx_tvalid ( wr_en           ),
  .i_axis_rx_tdata  ( vp_wr_data      ),
  .i_axis_rx_tlast  ( din_tlast_r     ),
  .i_axis_rx_tkeep  ( '1              ),
  .i_axis_rx_tuser  ( '0              ),
  .o_axis_rx_tready (                 ),
  .o_fifo_aempty    ( fifo_aempty     ),
  .o_fifo_afull     ( almost_full     ),
  .o_fifo_empty     ( is_empty        ),
  .o_fifo_full      ( fifo_full       ),
  .o_axis_tx_tvalid ( w_axis_tvalid   ),
  .o_axis_tx_tdata  ( o_axis_tdata    ),
  .o_axis_tx_tlast  ( o_axis_tlast    ),
  .o_axis_tx_tkeep  (                 ),
  .o_axis_tx_tuser  (                 ),
  .i_axis_tx_tready ( i_axis_tready   )
);

assign  o_axis_tkeep  = '1;
assign  o_axis_tuser  = '0;
assign  o_axis_tvalid = (!fifo_aempty || i_axis_tready || ((|tlast_buffered) && !out_is_last)) ? w_axis_tvalid : '0;

endmodule