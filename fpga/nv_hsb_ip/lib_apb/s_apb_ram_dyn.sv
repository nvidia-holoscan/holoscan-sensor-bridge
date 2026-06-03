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

module s_apb_ram_dyn
  import apb_pkg::*;
#(
  parameter   DEPTH    = 512,
  parameter   W_DATA   = 32,
  localparam  W_ADDR   = $clog2(DEPTH),
  localparam  NUM_RAM  = (((W_DATA-1)/32)+1),
  localparam  W_RAM    = $clog2(NUM_RAM)
)(
  // APB Interface
  input                     i_aclk, // Slow Clock
  input                     i_arst,
  input  apb_m2s            i_apb_m2s,
  output apb_s2m            o_apb_s2m,
  // User Control Signals
  input                     i_pclk, // Fast Clock
  input                     i_prst,
  input  [W_ADDR-1:0]       i_addr,
  output [W_DATA-1:0]       o_rd_data,
  output                    o_rd_data_valid,
  input  [W_DATA-1:0]       i_wr_data,
  input                     i_wr_en,
  input                     i_rd_en
);

//------------------------------------------------------------------------------------------------//
// RAM Switch
//------------------------------------------------------------------------------------------------//

apb_m2s s_apb_m2s [NUM_RAM];
apb_s2m s_apb_s2m [NUM_RAM];

apb_m2s m_apb_m2s [1];
apb_s2m m_apb_s2m [1];

localparam W_OFSET = W_ADDR+2;
localparam W_SW    = 32-W_OFSET;

apb_switch #(
  .N_MPORT             ( 1              ),
  .N_SPORT             ( NUM_RAM        ),
  .W_OFSET             ( W_OFSET        ),
  .W_SW                ( W_SW           ),
  .MERGE_COMPLETER_SIG ( 1              )
) u_apb_ram_switch (
  .i_apb_clk           ( i_aclk          ),
  .i_apb_reset         ( i_arst          ),
  .i_apb_m2s           ( m_apb_m2s [0:0] ),
  .o_apb_s2m           ( m_apb_s2m [0:0] ),
  .o_apb_m2s           ( s_apb_m2s       ),
  .i_apb_s2m           ( s_apb_s2m       ),
  .i_apb_timeout       ( 1'b0            )
);

assign m_apb_m2s[0] = i_apb_m2s;
assign o_apb_s2m    = m_apb_s2m[0];

//------------------------------------------------------------------------------------------------//
// RAM Instances
//------------------------------------------------------------------------------------------------//

logic [NUM_RAM-1:0] rd_data_valid;

logic [31:0] wr_data [NUM_RAM-1:0];
logic [31:0] rd_data [NUM_RAM-1:0];
logic [(32*NUM_RAM)-1:0] w_rd_data;

logic [(NUM_RAM*32)-1:0] w_wr_data;
assign w_wr_data = {'0,i_wr_data};

genvar i;
generate
  for (i = 0; i < NUM_RAM; i = i + 1) begin   
    s_apb_ram #(
      .R_CTRL           ( DEPTH                               ),
      .R_TOTL           ( DEPTH*4                             )
    ) u_apb_ram_inst (
      .i_aclk           ( i_aclk                              ),
      .i_arst           ( i_arst                              ),
      .i_apb_m2s        ( s_apb_m2s[i]                        ),
      .o_apb_s2m        ( s_apb_s2m[i]                        ),
      .i_pclk           ( i_pclk                              ),
      .i_prst           ( i_prst                              ),
      .i_addr           ( i_addr                              ),
      .o_rd_data        ( rd_data[i]                          ),
      .o_rd_data_valid  ( rd_data_valid[i]                    ),
      .i_wr_data        ( wr_data[i]                          ),
      .i_wr_en          ( i_wr_en                             ),
      .i_rd_en          ( i_rd_en                             )
    );
    // zero extend the write data to 32 bits
    assign wr_data[i] = w_wr_data[i*32+:32];
    assign w_rd_data[i*32+:32] = rd_data[i][0+:32];
  end
endgenerate

assign o_rd_data_valid = |rd_data_valid;
// Truncate the read data to the width of the write data
assign o_rd_data       = w_rd_data;
endmodule