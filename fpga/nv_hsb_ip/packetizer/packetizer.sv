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


module packetizer # (
    parameter DIN_WIDTH = 64,
    parameter SORT_ENABLE = 1,
    parameter SORT_RESOLUTION = 2,
    parameter PIPE_DEPTH = 3,
    parameter VP_COUNT = 1,
    parameter VP_SIZE = 32,
    parameter DOUT_WIDTH = 64,
    parameter LUT_RAM = 1,
    parameter NUM_CYCLES = 3,
    parameter DYNAMIC_VP  = 1,
    parameter MIXED_VP_SIZE = 1,
    parameter MTU = 1500,
    parameter W_USER = 1,
    localparam RAM_D_WIDTH = 32,
    localparam DOUT_KWIDTH = DOUT_WIDTH/8
) (
    // System
    input                                            i_sclk,
    input                                            i_srst,
    // System
    input                                            i_pclk,
    input                                            i_prst,
    // RAM Access
    input                                            ram_wr,
    input  [15:0]                                    ram_addr,
    input  [RAM_D_WIDTH-1:0]                         ram_wr_data,
    output  [VP_COUNT-1:0]                           vp_is_empty,
    output  [VP_COUNT-1:0]                           vp_almost_full,
    // Data In
    input  [DIN_WIDTH-1:0]                           din,
    input                                            din_valid,
    input   [W_USER-1:0]                             din_tuser,
    input                                            din_wait,
    input                                            din_stall,
    input                                            din_tlast,
    input                                            duplicate,
    // Data Output
    output  [VP_COUNT-1:0]                           o_axis_tvalid,
    output  [VP_COUNT-1:0]                           o_axis_tlast,
    output  [DOUT_WIDTH-1:0]                         o_axis_tdata [VP_COUNT-1:0],
    output  [DOUT_KWIDTH-1:0]                        o_axis_tkeep [VP_COUNT-1:0],
    output  [VP_COUNT-1:0]                           o_axis_tuser,
    input   [VP_COUNT-1:0]                           i_axis_tready
    );


//------------------------------------------------------------------------------------------------//
// Local Params
//------------------------------------------------------------------------------------------------//

// RAM
localparam RAM_DEPTH                        = 16;
localparam RAM_ADDR_WIDTH                   = $clog2(RAM_DEPTH);
localparam [12:0] [31:0] ODD_EVEN_SIZE_VEC  = {32'd139263,32'd58367,32'd24063,32'd9727,32'd3839,32'd1471,32'd543,32'd191,32'd63,32'd19,32'd5,32'd1,32'd1}; // Number of Swaps in Odd Even for 2^n
localparam [12:0] [31:0] ODD_EVEN_DEPTH_VEC = {32'd78,32'd66,32'd55,32'd45,32'd36,32'd28,32'd21,32'd15,32'd10,32'd6,32'd3,32'd1,32'd1};                    // Number of Stages in Odd Even for 2^n
localparam CTRL_WIDTH_SORT                  = ODD_EVEN_SIZE_VEC[$clog2(DIN_WIDTH)-$clog2(SORT_RESOLUTION)];
localparam SORT_DEPTH                       = ODD_EVEN_DEPTH_VEC[$clog2(DIN_WIDTH)-$clog2(SORT_RESOLUTION)];
localparam SORT_PIPE_DEPTH                  = 3;
localparam PACK_PIPE_COUNT                  = (SORT_DEPTH < SORT_PIPE_DEPTH) ? 1 : (SORT_DEPTH-1)/SORT_PIPE_DEPTH;                              // Sorter Latency
localparam CTRL_WIDTH_CLEAR                 = (((DIN_WIDTH/SORT_RESOLUTION))*$clog2(NUM_CYCLES+1));   // Clear and Cycle Select
localparam CYCLE_SEL_WIDTH                  = $clog2(NUM_CYCLES+1);
localparam CTRL_WIDTH_VP                    = (DYNAMIC_VP) ? $clog2(DIN_WIDTH) + $clog2(VP_SIZE) + 1 : (DIN_WIDTH==VP_SIZE) ? 2 : ($clog2(DIN_WIDTH/VP_SIZE)) + 1;
localparam [3:0] [31:0] VP_SIZE_VEC         = (VP_COUNT==1) ? {DIN_WIDTH} : MIXED_VP_SIZE ? {VP_SIZE,VP_SIZE/2,VP_SIZE/4,VP_SIZE/4} : {VP_SIZE,VP_SIZE,VP_SIZE,VP_SIZE};
localparam [2:0] [31:0] RAM_WIDTH_VEC       = {(CTRL_WIDTH_VP*VP_COUNT),CTRL_WIDTH_SORT*NUM_CYCLES,CTRL_WIDTH_CLEAR};
localparam [2:0] [31:0] RAM_COUNT_VEC       = {((RAM_WIDTH_VEC[2]-1)/RAM_D_WIDTH) + 1,((RAM_WIDTH_VEC[1]-1)/RAM_D_WIDTH) + 1,((RAM_WIDTH_VEC[0]-1)/RAM_D_WIDTH) + 1};
localparam [2:0] [31:0] RAM_OFFSET_VEC      = {RAM_COUNT_VEC[1]+RAM_COUNT_VEC[0],RAM_COUNT_VEC[0],32'd0};
localparam [31:0] NUM_RAM                   = RAM_COUNT_VEC[2] + RAM_COUNT_VEC[1] + RAM_COUNT_VEC[0];
localparam        RAM_SELECT_WIDTH          = $clog2(NUM_RAM);


//------------------------------------------------------------------------------------------------//
// RAM Modules
//------------------------------------------------------------------------------------------------//

logic [NUM_RAM*RAM_D_WIDTH-1:0]                 ram_dout;
logic [NUM_RAM*RAM_D_WIDTH-1:0]                 w_ram_dout;
logic [NUM_RAM-1:0]                             ram_we;
logic [RAM_SELECT_WIDTH+RAM_ADDR_WIDTH-1:0]     ram_addr_mux  [2:0];
logic [RAM_SELECT_WIDTH+RAM_ADDR_WIDTH-1:0]     ram_addr_pipe [PACK_PIPE_COUNT:0];
logic [PACK_PIPE_COUNT:0]                       din_valid_pipe;
logic [PACK_PIPE_COUNT:0]                       din_tlast_pipe;

genvar k;
for (k=0;k<NUM_RAM;k=k+1) begin: PACK_CONFIG
  integer m;
  assign m = (k<RAM_OFFSET_VEC[1]) ? 0 :    // CLEAR
              (k<RAM_OFFSET_VEC[2]) ? 1 :    // SORT
                                      2 ;    // VP

  (* ram_style = "distributed" *) logic [31:0] cfg_mem [16] /*synthesis syn_ramstyle = "distributed"*/;
  always @ (posedge i_sclk) begin
    if (i_srst) begin
      cfg_mem                                <= '{default:'0};
      w_ram_dout[k*RAM_D_WIDTH+:RAM_D_WIDTH] <= '0;
    end
    else begin
      if (ram_we[k]) begin
        cfg_mem[ram_addr_mux[m][0+:RAM_ADDR_WIDTH]] <= ram_wr_data;
      end
      w_ram_dout[k*RAM_D_WIDTH+:RAM_D_WIDTH] <= cfg_mem[ram_addr_mux[m][0+:RAM_ADDR_WIDTH]];
    end
  end
  assign ram_we[k] = (ram_addr[RAM_ADDR_WIDTH+:RAM_SELECT_WIDTH] == k) && ram_wr;
end
always @ (posedge i_sclk) begin
  ram_dout <= w_ram_dout;
end

assign ram_addr_mux[0] = (ram_wr) ? ram_addr : ram_addr_pipe[PACK_PIPE_COUNT-1];    // Clear
assign ram_addr_mux[1] = (ram_wr) ? ram_addr : ram_addr;    // Sort
assign ram_addr_mux[2] = (ram_wr) ? ram_addr : ram_addr_pipe[PACK_PIPE_COUNT-1];    // VP, delayed address



//------------------------------------------------------------------------------------------------//
// Pipeline Address X number of Cycles
//------------------------------------------------------------------------------------------------//

logic [DIN_WIDTH-1:0] din_pipe [NUM_CYCLES-1:0];
logic [DIN_WIDTH-1:0] sort_din [NUM_CYCLES-1:0];
logic [W_USER-1:0]    tuser_pipe [PACK_PIPE_COUNT+1];
logic                 adv_pipe;

integer p,q;
always @(posedge i_sclk) begin
  if (i_srst) begin
    for (p=0;p<=PACK_PIPE_COUNT;p=p+1) begin
      ram_addr_pipe[p]   <= '0;
      din_valid_pipe [p] <= '0;
      din_tlast_pipe[p]  <= '0;
      tuser_pipe[p]      <= '0;
    end
    for (q=0;q<NUM_CYCLES;q=q+1) begin
      din_pipe[q] <= '0;
    end
  end
  else begin
    ram_addr_pipe[0]   <= ram_addr;
    din_valid_pipe [0] <= din_valid && !din_wait;
    tuser_pipe[0]      <= din_tuser;
    din_tlast_pipe[0]  <= din_tlast;
    for (p=1;p<=PACK_PIPE_COUNT;p=p+1) begin
      ram_addr_pipe[p]   <= ram_addr_pipe[p-1];
      din_valid_pipe [p] <= din_valid_pipe [p-1];
      tuser_pipe[p]      <= tuser_pipe[p-1];
      din_tlast_pipe[p]  <= din_tlast_pipe[p-1];
    end
    for (q=1;q<NUM_CYCLES;q=q+1) begin
      din_pipe[q] <= adv_pipe ? din_pipe[q-1] : din_pipe[q];
    end
    din_pipe[0] <= adv_pipe ? din : din_pipe[0];
  end
end
assign sort_din = din_pipe;
assign adv_pipe = (din_valid&&(!din_stall||din_wait));  // Only stall the pipe when not in wait state (latency)


//------------------------------------------------------------------------------------------------//
// Sorter
//------------------------------------------------------------------------------------------------//

logic [DIN_WIDTH-1:0]                    sort_dout_tmp  [NUM_CYCLES-1:0];
logic [DIN_WIDTH-1:0]                    sort_dout;
logic [(CTRL_WIDTH_SORT*NUM_CYCLES)-1:0] sort_sel;
logic [CTRL_WIDTH_CLEAR-1:0]             cycle_sel;

genvar k0,k1;

generate
  if (SORT_ENABLE == 1) begin
    for (k1=0;k1<NUM_CYCLES;k1=k1+1) begin : sort_inst_gen
      odd_even_gen #(
        .D_WIDTH    (DIN_WIDTH                                        ),
        .PIPE_DEPTH (SORT_PIPE_DEPTH                                  ),
        .SORT_DEPTH (SORT_DEPTH                                       ),
        .SEL_WIDTH  (CTRL_WIDTH_SORT                                  ),
        .RESOLUTION (SORT_RESOLUTION                                  )
      )
      sort_inst (
        .clk   ( i_sclk                                               ),
        .din   ( sort_din       [k1]                                  ),
        .sel   ( sort_sel       [k1*CTRL_WIDTH_SORT+:CTRL_WIDTH_SORT] ),
        .dout  ( sort_dout_tmp  [k1]                                  ),
        .bypass( tuser_pipe     [0][0]                                )
      ); /* synthesis syn_keep=1 */
    end
    assign sort_sel = ram_dout[RAM_OFFSET_VEC[1]*RAM_D_WIDTH+:RAM_WIDTH_VEC[1]];

    for (k0=0;k0<(DIN_WIDTH/SORT_RESOLUTION);k0=k0+1) begin
      assign sort_dout[k0*SORT_RESOLUTION+:SORT_RESOLUTION] = tuser_pipe[PACK_PIPE_COUNT][0]                         ? sort_dout_tmp[0][k0*SORT_RESOLUTION+:SORT_RESOLUTION]: // Bypass
                                                              (cycle_sel[k0*CYCLE_SEL_WIDTH+:CYCLE_SEL_WIDTH] == '0) ? '0                                                   : // Clear the bit region,
                                                              sort_dout_tmp[cycle_sel[k0*CYCLE_SEL_WIDTH+:CYCLE_SEL_WIDTH] - 1][k0*SORT_RESOLUTION+:SORT_RESOLUTION]        ; // or select which cycle
    end
  end
  else begin
    always_ff @ (posedge i_sclk) begin
      sort_dout <= sort_din[0];
    end
  end
  assign cycle_sel = ram_dout[0+:CTRL_WIDTH_CLEAR];

endgenerate

//------------------------------------------------------------------------------------------------//
// Virtual Ports
//------------------------------------------------------------------------------------------------//

logic [RAM_WIDTH_VEC[2]-1:0] vp_sel;

logic bp_wr_en;

genvar i;
for (i=0;i<VP_COUNT;i=i+1) begin: VP
  logic [CTRL_WIDTH_VP-1:0]   vp_ctrl;
  if ((CTRL_WIDTH_VP == 2) && (W_USER > 2)) begin
    assign vp_ctrl[0] = (tuser_pipe[PACK_PIPE_COUNT][3:2] == i) || ((i == 0) && bp_wr_en); // Virtual Port select via tuser
    assign vp_ctrl[1] = '0;
  end
  else begin
    assign vp_ctrl = (duplicate)            ? 'b1 :
                      ((i == 0) && bp_wr_en) ? 'b1 :
                                              vp_sel [(i*CTRL_WIDTH_VP)+:CTRL_WIDTH_VP];
  end

  virtual_port # (
    .DIN_WIDTH        ( DIN_WIDTH                           ),
    .VP_SIZE          ( VP_SIZE_VEC[i[1:0]]                 ),
    .DOUT_WIDTH       ( DOUT_WIDTH                          ),
    .RESOLUTION       ( SORT_RESOLUTION                     ),
    .CTRL_WIDTH       ( CTRL_WIDTH_VP                       ),
    .DYNAMIC_VP       ( DYNAMIC_VP                          ),
    .MTU              ( MTU                                 )
  ) vp_inst (
    .i_sclk           ( i_sclk                              ),
    .i_srst           ( i_srst                              ),
    .i_pclk           ( i_pclk                              ),
    .i_prst           ( i_prst                              ),
    .din              ( sort_dout         [0+:DIN_WIDTH]    ),
    .din_valid        ( din_valid_pipe    [PACK_PIPE_COUNT] ),
    .ctrl             ( vp_ctrl                             ),
    .is_empty         ( vp_is_empty       [i]               ),
    .din_tlast        ( din_tlast_pipe    [PACK_PIPE_COUNT] ),
    .almost_full      ( vp_almost_full    [i]               ),
    .o_axis_tvalid    ( o_axis_tvalid     [i]               ),
    .o_axis_tlast     ( o_axis_tlast      [i]               ),
    .o_axis_tdata     ( o_axis_tdata      [i]               ),
    .o_axis_tkeep     ( o_axis_tkeep      [i]               ),
    .o_axis_tuser     ( o_axis_tuser      [i]               ),
    .i_axis_tready    ( i_axis_tready     [i]               )
  ); /* synthesis syn_keep=1 */
end
assign vp_sel = ram_dout[RAM_OFFSET_VEC[2]*RAM_D_WIDTH+:RAM_WIDTH_VEC[2]];
assign bp_wr_en = (tuser_pipe[PACK_PIPE_COUNT][0]);

endmodule
