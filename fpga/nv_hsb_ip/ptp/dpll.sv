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

module dpll #(
  // constraint to 36 for efficient dsp utilization, for 100ppm clock, need 18b for ns
  parameter W_FA    = 32
)(
  input              i_clk,
  input              i_rst,
  // Gain Configuration
  input              cfg_gain_1_en,
  input              cfg_gain_2_en,
  input  [31     :0] cfg_gain_1,
  input  [31     :0] cfg_gain_2,
  // Phase Detector Input
  input              i_pd_vld,
  input  [W_FA-1:0]  i_pd,
  // Phase Adjustment Output
  output             o_fa_vld,
  output [W_FA-1:0]  o_fa
);

//------------------------------------------------------------------------------------------------//
// Phase Detector pd Gain - assuming PD gain is 1
//------------------------------------------------------------------------------------------------//

logic                   pd_vld;
logic signed [W_FA-1:0] pd;

assign pd_vld = i_pd_vld;
assign pd     = i_pd;

//------------------------------------------------------------------------------------------------//
// Loop Filter -  G1 = 1/2, G2 = 1/4096
//------------------------------------------------------------------------------------------------//

// Gain
logic                   g12_pd_vld;
logic signed [W_FA-1:0] g1_pd;
logic signed [W_FA-1:0] g2_pd;
logic        [2     :0] g1_gain;
logic        [2     :0] g2_gain;
logic                   g1_en;
logic                   g2_en;

assign g1_gain = cfg_gain_1[2:0];
assign g2_gain = cfg_gain_2[2:0];
assign g1_en   = cfg_gain_1_en;
assign g2_en   = cfg_gain_2_en;

always_ff @(posedge i_clk) begin
  if (i_rst) begin
    g12_pd_vld <= '0;
    g1_pd      <= '0;
    g2_pd      <= '0;
  end
  else begin
    g12_pd_vld <= pd_vld;
    if (pd_vld) begin
      // Gain (shift or multiply
      g1_pd <= signed'(pd >>> unsigned'(g1_gain));
      g2_pd <= signed'(pd >>> unsigned'(g2_gain));
    end
  end
end

// Accumulator
logic                   g12_pd_vld_q;
logic signed [W_FA-1:0] acc_g2_pd;
logic        [W_FA-1:0] g1_pd_q;
logic                   fa_vld;
logic signed [W_FA-1:0] fa;

always_ff @(posedge i_clk) begin
  if (i_rst) begin
    g12_pd_vld_q <= '0;
    g1_pd_q      <= '0;
    acc_g2_pd    <= '0;
    fa_vld       <= '0;
    fa           <= '0;
  end
  else begin
    g12_pd_vld_q <= g12_pd_vld;
    acc_g2_pd    <= acc_g2_pd;
    if (g12_pd_vld) begin
      g1_pd_q    <= g1_en ? g1_pd                      : '0;
      acc_g2_pd  <= g2_en ? signed'(acc_g2_pd + g2_pd) : '0;
    end

    fa_vld       <= g12_pd_vld_q;
    if (g12_pd_vld_q) begin
      fa         <= signed'(acc_g2_pd + g1_pd_q);
    end
  end
end

assign o_fa_vld = fa_vld;
assign o_fa     = fa;

endmodule
