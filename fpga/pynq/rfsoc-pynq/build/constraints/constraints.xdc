## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
## SPDX-License-Identifier: Apache-2.0
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

#########################################################################
# Vivado project properties
#

#########################################################################
# Pinouts
#   - PACKAGE_PIN
#   - IOSTANDARD
#   - DIFF_TERM_ADV
#
set_property PACKAGE_PIN AA34 [get_ports ETH_REFCLK_N]
set_property PACKAGE_PIN AA33 [get_ports ETH_REFCLK_P]
set_property PACKAGE_PIN AM15 [get_ports sysclk_p]
set_property PACKAGE_PIN AN15 [get_ports sysclk_n]
set_property PACKAGE_PIN AV12 [get_ports RESET]
set_property IOSTANDARD LVCMOS18 [get_ports RESET]
set_property IOSTANDARD LVDS [get_ports sysclk_p]
set_property IOSTANDARD LVDS [get_ports sysclk_n]

set_property LOC GTYE4_CHANNEL_X0Y7 [get_cells {u_100gbe/design_1_i/cmac_usplus_0/inst/design_1_cmac_usplus_0_0_gt_i/inst/gen_gtwizard_gtye4_top.design_1_cmac_usplus_0_0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[1].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[3].GTYE4_CHANNEL_PRIM_INST}]
set_property PACKAGE_PIN R33 [get_ports {gt_serial_port_0_gtx_p[3]}]
set_property PACKAGE_PIN R34 [get_ports {gt_serial_port_0_gtx_n[3]}]
set_property PACKAGE_PIN V36 [get_ports gt_serial_port_0_gtx_n[2]]
set_property PACKAGE_PIN V35 [get_ports gt_serial_port_0_gtx_p[2]]
set_property PACKAGE_PIN T36 [get_ports gt_serial_port_0_gtx_n[1]]
set_property PACKAGE_PIN T35 [get_ports gt_serial_port_0_gtx_p[1]]
set_property LOC GTYE4_CHANNEL_X0Y4 [get_cells {u_100gbe/design_1_i/cmac_usplus_0/inst/design_1_cmac_usplus_0_0_gt_i/inst/gen_gtwizard_gtye4_top.design_1_cmac_usplus_0_0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[1].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST}]
set_property PACKAGE_PIN Y35 [get_ports {gt_serial_port_0_gtx_p[0]}]
set_property PACKAGE_PIN Y36 [get_ports {gt_serial_port_0_gtx_n[0]}]

set_property PACKAGE_PIN AA39 [get_ports gt_serial_port_0_grx_n[3]]
set_property PACKAGE_PIN AA38 [get_ports gt_serial_port_0_grx_p[3]]
set_property LOC GTYE4_CHANNEL_X0Y6 [get_cells {u_100gbe/design_1_i/cmac_usplus_0/inst/design_1_cmac_usplus_0_0_gt_i/inst/gen_gtwizard_gtye4_top.design_1_cmac_usplus_0_0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[1].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[2].GTYE4_CHANNEL_PRIM_INST}]
set_property PACKAGE_PIN U38 [get_ports {gt_serial_port_0_grx_p[2]}]
set_property PACKAGE_PIN U39 [get_ports {gt_serial_port_0_grx_n[2]}]
set_property LOC GTYE4_CHANNEL_X0Y5 [get_cells {u_100gbe/design_1_i/cmac_usplus_0/inst/design_1_cmac_usplus_0_0_gt_i/inst/gen_gtwizard_gtye4_top.design_1_cmac_usplus_0_0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[1].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[1].GTYE4_CHANNEL_PRIM_INST}]
set_property PACKAGE_PIN W38 [get_ports {gt_serial_port_0_grx_p[1]}]
set_property PACKAGE_PIN W39 [get_ports {gt_serial_port_0_grx_n[1]}]
set_property PACKAGE_PIN R39 [get_ports gt_serial_port_0_grx_n[0]]
set_property PACKAGE_PIN R38 [get_ports gt_serial_port_0_grx_p[0]]



#########################################################################
# Timing Constraints
#
#

#only the P side needs a constraint
create_clock -period 6.400 -name ETH_REFCLK_P [get_ports ETH_REFCLK_P]
create_clock -period 10.000 -name sysclk_p [get_ports sysclk_p]
set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins u_100gbe/design_1_i/clk_wiz_0/inst/mmcme4_adv_inst/CLKOUT0]] -group [get_clocks -of_objects [get_pins {u_100gbe/design_1_i/cmac_usplus_0/inst/design_1_cmac_usplus_0_0_gt_i/inst/gen_gtwizard_gtye4_top.design_1_cmac_usplus_0_0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[1].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST/TXOUTCLK}]]
set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins u_100gbe/design_1_i/clk_wiz_0/inst/mmcme4_adv_inst/CLKOUT1]] -group [get_clocks -of_objects [get_pins {u_100gbe/design_1_i/cmac_usplus_0/inst/design_1_cmac_usplus_0_0_gt_i/inst/gen_gtwizard_gtye4_top.design_1_cmac_usplus_0_0_gt_gtwizard_gtye4_inst/gen_gtwizard_gtye4.gen_channel_container[1].gen_enabled_channel.gtye4_channel_wrapper_inst/channel_inst/gtye4_channel_gen.gen_gtye4_channel_inst[0].GTYE4_CHANNEL_PRIM_INST/TXOUTCLK}]]
set_clock_groups -asynchronous -group [get_clocks -of_objects [get_pins u_100gbe/design_1_i/clk_wiz_0/inst/mmcme4_adv_inst/CLKOUT1]] -group [get_clocks -of_objects [get_pins u_100gbe/design_1_i/clk_wiz_0/inst/mmcme4_adv_inst/CLKOUT0]]

#########################################################################
# Keep the shallow memories that were distributed in the 13:43 map in
# distributed RAM. Synthesis reads this before block-RAM mapping, so it
# overrides the block attribute on these instances only.
# The timestamp and SIF ILAs, the output buffer, data-gen 2/3, and
# packetizer bypass 1/3 are not in this list.
set dram_rams [get_cells -quiet -hierarchical -regexp {.*(u_ram_player|ipv4_buffer|fcs_buffer|hdr_buffer|u_axis_inp_buffer|u_tx_stream_buffer|ecb_rdwr_ctrl_inst|u_evt_int|u_ptp_ingress|i2c_ctrl_inst|spi_ctrl_inst|gen_data_gen_inst\[0\]|gen_data_gen_inst\[1\]|gen_pack_inst\[0\]|u_dp_pkt_ts|u_config_ram|u_rx_ls_parseri_1/u_axis_buffer|DISTRIBUTED_RAM\.lut_mem).*(blk_mem|lut_mem).* }]
set_property RAM_STYLE DISTRIBUTED $dram_rams

#########################################################################
# RAM-player enable, timer, window size, and window count are written
# over APB and then sit still. Let the synchronized value take two user
# clocks to reach the playback registers. Paths that start at data_cnt,
# the read address, or the RAM stay single-cycle.
set ram_player_cfg [get_cells -quiet -hierarchical -filter {NAME =~ *u_ram_player/u_s_ram_apb_reg/*u_ctrl_cdc/a2b_reg_sync_reg*}]
set ram_player_dp [get_cells -quiet -hierarchical -filter {NAME =~ *u_ram_player/* && NAME !~ *u_s_ram_apb_reg* && NAME !~ *u_s_apb_ram*}]
set_multicycle_path -setup 2 -from $ram_player_cfg -to $ram_player_dp
set_multicycle_path -hold 1 -from $ram_player_cfg -to $ram_player_dp
