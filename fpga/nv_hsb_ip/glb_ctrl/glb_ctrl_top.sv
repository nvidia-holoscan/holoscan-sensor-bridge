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

module glb_ctrl_top
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter              N_GPIO       = 16,
  parameter [N_GPIO-1:0] GPIO_RST_VAL = '0,
  parameter              SYNC_CLK     = 0

)(
  // clock and reset
  input               i_pclk,
  input               i_prst,
  // Register Map, abp clk domain
  input               i_aclk,
  input               i_arst,
  input  apb_m2s      i_apb_m2s,
  output apb_s2m      o_apb_s2m,
  // HOLOLINK GPIO control and status
  input  [31      :0] i_hsb_ver,
  input  [31      :0] i_hsb_date,
  input  [31      :0] i_hsb_stat,
  output [63      :0] o_hsb_ctrl,
  // User GPIO
  input  [N_GPIO-1:0] i_gpio,
  output [N_GPIO-1:0] o_gpio,
  output [N_GPIO-1:0] o_gpio_dir
);

localparam  [(glb_nctrl*32)-1:0] RST_VAL = {'0,GPIO_RST_VAL,{(gpio_o_0*32){1'b0}}};

//------------------------------------------------------------------------------------------------//
// Register Map
//------------------------------------------------------------------------------------------------//

logic [31:0] ctrl_reg [glb_nctrl];
logic [31:0] stat_reg [glb_nstat];

logic [255:0] gpio_i;

assign gpio_i[N_GPIO-1:0] = i_gpio;
assign gpio_i[255:N_GPIO] = 0;

assign stat_reg[hsb_ver -stat_ofst] = i_hsb_ver;
assign stat_reg[hsb_date-stat_ofst] = i_hsb_date;
assign stat_reg[hsb_stat-stat_ofst] = i_hsb_stat;

// GPIO Input status
assign stat_reg[gpio_i_0-stat_ofst] = gpio_i[ 31:  0];
assign stat_reg[gpio_i_1-stat_ofst] = gpio_i[ 63: 32];
assign stat_reg[gpio_i_2-stat_ofst] = gpio_i[ 95: 64];
assign stat_reg[gpio_i_3-stat_ofst] = gpio_i[127: 96];
assign stat_reg[gpio_i_4-stat_ofst] = gpio_i[159:128];
assign stat_reg[gpio_i_5-stat_ofst] = gpio_i[191:160];
assign stat_reg[gpio_i_6-stat_ofst] = gpio_i[223:192];
assign stat_reg[gpio_i_7-stat_ofst] = gpio_i[255:224];

s_apb_reg #(
  .N_CTRL     ( glb_nctrl     ),
  .N_STAT     ( glb_nstat     ),
  .W_OFST     ( w_ofst        ),
  .RST_VAL    ( RST_VAL       ),
  .SYNC_CLK   ( SYNC_CLK      )
) u_reg_map   (
  // APB Interface
  .i_aclk     ( i_aclk        ),
  .i_arst     ( i_arst        ),
  .i_apb_m2s  ( i_apb_m2s     ),
  .o_apb_s2m  ( o_apb_s2m     ),
  // User Control Signals
  .i_pclk     ( i_pclk        ),
  .i_prst     ( i_prst        ),
  .o_ctrl     ( ctrl_reg      ),
  .i_stat     ( stat_reg      )
);

assign o_hsb_ctrl = {ctrl_reg[hsb_ctrl_1], ctrl_reg[hsb_ctrl_0]};

logic [255:0] gpio_o;
assign gpio_o    = {ctrl_reg[gpio_o_7], ctrl_reg[gpio_o_6],
                    ctrl_reg[gpio_o_5], ctrl_reg[gpio_o_4],
                    ctrl_reg[gpio_o_3], ctrl_reg[gpio_o_2],
                    ctrl_reg[gpio_o_1], ctrl_reg[gpio_o_0]};

logic [255:0] gpio_t;

assign gpio_t    = {ctrl_reg[gpio_t_7], ctrl_reg[gpio_t_6],
                    ctrl_reg[gpio_t_5], ctrl_reg[gpio_t_4],
                    ctrl_reg[gpio_t_3], ctrl_reg[gpio_t_2],
                    ctrl_reg[gpio_t_1], ctrl_reg[gpio_t_0]};

genvar i;
generate
  for (i=0; i<N_GPIO; i++) begin
    assign o_gpio[i]     = gpio_o[i];
    assign o_gpio_dir[i] = gpio_t[i];
  end
endgenerate

endmodule
