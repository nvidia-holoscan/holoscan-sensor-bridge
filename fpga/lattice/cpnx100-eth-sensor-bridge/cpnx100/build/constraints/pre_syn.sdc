# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# External IO pins
create_clock -name {CAM_DCLK_0} -period 2 [get_ports {CAM_DCLK[0]}]
create_clock -name {CAM_DCLK_1} -period 2 [get_ports {CAM_DCLK[1]}]
create_clock -name {ETH_REFCLK_P} -period 6.2060606 [get_ports ETH_REFCLK_P]
# Internal clock nets
create_clock -name {u_pcs_clk_hf_clk_out_o} -period 6.66666666666667 [get_nets pcs_clk]
create_clock -name {usr_clk} -period 3.1030303030303 [get_nets {usr_clk[0]}]
create_clock -name {apb_clk} -period 51.2 [get_nets apb_clk]
create_clock -name {hif_clk} -period 6.4 [get_nets hif_clk]
create_clock -name {ptp_clk} -period 9.95554401 [get_nets ptp_clk]
#create_clock -name {i2s_pll_clk} -period 27.027 [get_nets {u_clk_n_rst/i2s_pll_clk}]
create_clock -name {i2s_pll_clk} -period 27.027 [get_nets {i2s_ref_clk}]
create_generated_clock -name i2s_mclk_ext -source [get_nets {i2s_ref_clk}] -divide_by 2  [get_nets i2s_mclk_ext]
create_generated_clock -name i2s_clk_ext  -source [get_nets {i2s_ref_clk}] -divide_by 16 [get_nets i2s_clk_ext]
# create_generated_clock -name i2s_mclk_ext -source [get_nets {u_clk_n_rst/i2s_pll_clk}] -divide_by 2  [get_nets i2s_mclk_ext]
# create_generated_clock -name i2s_clk_ext  -source [get_nets {u_clk_n_rst/i2s_pll_clk}] -divide_by 16 [get_nets i2s_clk_ext]
# create_generated_clock -name i2s_clk_int  -source [get_nets {u_clk_n_rst/i2s_pll_clk}] -divide_by 16 [get_nets i2s_clk_int]
create_clock -name {lvds_rxclk_0} -period 8 [get_nets {cam_sensor_rcvr[0].u_cam_rcvr/rx_dclk}]
create_clock -name {lvds_rxclk_1} -period 8 [get_nets {cam_sensor_rcvr[1].u_cam_rcvr/rx_dclk}]
set_clock_groups -group [get_clocks {ETH_REFCLK_P usr_clk}] -group [get_clocks {CAM_DCLK_0 lvds_rxclk_0}] -group [get_clocks {CAM_DCLK_1 lvds_rxclk_1}] -group [get_clocks {apb_clk hif_clk}] -group [get_clocks {i2s_pll_clk i2s_clk_ext i2s_mclk_ext }] -group [get_clocks u_pcs_clk_hf_clk_out_o] -group [get_clocks ptp_clk] -asynchronous
#set_clock_groups -group [get_clocks {ETH_REFCLK_P usr_clk}] -group [get_clocks {CAM_DCLK_0 lvds_rxclk_0}] -group [get_clocks {CAM_DCLK_1 lvds_rxclk_1}] -group [get_clocks {hif_clk}] -group [get_clocks {i2s_pll_clk i2s_clk_ext i2s_mclk_ext i2s_clk_int}] -group [get_clocks pcs_clk] -group [get_clocks ptp_clk] -group [get_clocks apb_clk] -asynchronous
set_false_path -from [get_nets {cam_sensor_rcvr[0].u_cam_rcvr/cal_align}] -to [get_nets {cam_sensor_rcvr[0].u_cam_rcvr/u_lvds_rx/lscc_gddr_inst/RX_ECLK_CENTERED_STATIC_BYPASS.u_lscc_gddrx2_4_5_rx_eclk_centered_static_bypass/alignwd_i}]
set_false_path -from [get_nets {cam_sensor_rcvr[1].u_cam_rcvr/cal_align}] -to [get_nets {cam_sensor_rcvr[1].u_cam_rcvr/u_lvds_rx/lscc_gddr_inst/RX_ECLK_CENTERED_STATIC_BYPASS.u_lscc_gddrx2_4_5_rx_eclk_centered_static_bypass/alignwd_i}]
