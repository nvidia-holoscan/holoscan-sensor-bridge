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

module general_purpose_regs 
  import apb_pkg::*;
#(
  parameter                     N_CTRL_REGS = 6,
  parameter                     N_STAT_REGS = 4
) (
  input                         i_apb_clk,
  input                         i_apb_rst,
  
  input logic                   i_apb_psel,
  input logic                   i_apb_penable,         
  input logic   [31:0]          i_apb_paddr,           
  input logic   [31:0]          i_apb_pwdata,          
  input logic                   i_apb_pwrite,          
  output logic                  o_apb_pready,
  output logic  [31:0]          o_apb_prdata,
  output logic                  o_apb_pserr,
  //Control Regs
  output logic                  o_sw_fpga_reconfig,
  output logic                  o_sw_ad9545_rst,
  output logic                  o_clk_chk_en,
  output logic  [31:0]          o_clk_chk_target,
  output logic  [31:0]          o_clk_chk_tolerance,
  ////Status Regs
  input logic                   i_clk_chk_in_tolerance,
  input logic   [31:0]          i_clk_chk_count,
  input logic   [6:0]           i_ad9545_mode_pins

);

apb_m2s     gen_purpose_apb_m2s;
apb_s2m     gen_purpose_apb_s2m;

logic [31:0]            ctrl_reg  [N_CTRL_REGS];
logic [31:0]            stat_reg  [N_STAT_REGS];

//Control Register Mappings
assign o_sw_fpga_reconfig             = ctrl_reg[0][31];
assign o_sw_ad9545_rst                = ctrl_reg[0][0];
assign o_clk_chk_en                   = ctrl_reg[1][0];
assign o_clk_chk_target               = ctrl_reg[2];
assign o_clk_chk_tolerance            = ctrl_reg[3];
assign stat_reg[0]                    = {31'b0, i_clk_chk_in_tolerance};
assign stat_reg[1]                    = i_clk_chk_count;
assign stat_reg[2]                    = {25'b0, i_ad9545_mode_pins};

//APB mapping
assign gen_purpose_apb_m2s.psel       = i_apb_psel;
assign gen_purpose_apb_m2s.penable    = i_apb_penable;
assign gen_purpose_apb_m2s.paddr      = i_apb_paddr;
assign gen_purpose_apb_m2s.pwdata     = i_apb_pwdata;
assign gen_purpose_apb_m2s.pwrite     = i_apb_pwrite;
assign o_apb_pready                   = gen_purpose_apb_s2m.pready; 
assign o_apb_prdata                   = gen_purpose_apb_s2m.prdata; 
assign o_apb_pserr                    = gen_purpose_apb_s2m.pserr; 

s_apb_reg #(
  .N_CTRL                       (N_CTRL_REGS),
  .N_STAT                       (N_STAT_REGS)
) u_gen_purpose_apb_reg (
  .i_aclk                       (i_apb_clk), 
  .i_arst                       (i_apb_rst),
  .i_apb_m2s                    (gen_purpose_apb_m2s),
  .o_apb_s2m                    (gen_purpose_apb_s2m),
  .i_pclk                       (i_apb_clk), 
  .i_prst                       (i_apb_rst),
  .o_ctrl                       (ctrl_reg),
  .i_stat                       (stat_reg)
);

endmodule
