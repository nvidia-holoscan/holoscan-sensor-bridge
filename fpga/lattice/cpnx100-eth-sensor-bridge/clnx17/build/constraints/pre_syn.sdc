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

#**************************************************************
# Create Clock
#**************************************************************
create_clock -name {PCLK_IN} -period 8 [get_ports PCLK_IN]
create_clock -name {u_lvds_pll/clkop_o} -period 2 [get_pins u_lvds_pll/clkop_o]
create_clock -name {u_lvds_pll/clkos_o} -period 2 [get_pins u_lvds_pll/clkos_o]
create_clock -name {u_lvds_pll/clkos2_o} -period 40 [get_pins u_lvds_pll/clkos2_o]
create_clock -name {u_lvds_pll/clkos3_o} -period 16.66667 [get_pins u_lvds_pll/clkos3_o]
create_clock -name {mipi_rx[0].mipi_bridge_inst/mipi_rx_inst/clk_byte_hs_o} -period 5.3 [get_pins {mipi_rx[0].mipi_bridge_inst/mipi_rx_inst/clk_byte_hs_o}]
create_clock -name {mipi_rx[0].mipi_bridge_inst/lvds_tx_inst/sclk_o} -period 8 [get_pins {mipi_rx[0].mipi_bridge_inst/lvds_tx_inst/sclk_o}]
create_clock -name {mipi_rx[1].mipi_bridge_inst/mipi_rx_inst/clk_byte_hs_o} -period 5.3 [get_pins {mipi_rx[1].mipi_bridge_inst/mipi_rx_inst/clk_byte_hs_o}]
create_clock -name {mipi_rx[1].mipi_bridge_inst/lvds_tx_inst/sclk_o} -period 8 [get_pins {mipi_rx[1].mipi_bridge_inst/lvds_tx_inst/sclk_o}]
create_clock -name {mipi_rx[0].mipi_bridge_inst/mipi_clk_p_io} -period 0.8 [get_pins {mipi_rx[0].mipi_bridge_inst/mipi_clk_p_io}]
create_clock -name {MIPI_CAM_CLK_P[0]} -period 0.8 [get_nets {MIPI_CAM_CLK_P[0]}]
create_clock -name {MIPI_CAM_CLK_P[1]} -period 0.8 [get_nets {MIPI_CAM_CLK_P[1]}]
create_clock -name {mipi_rx[0].mipi_bridge_inst/lvds_tx_inst/lscc_gddr_inst/TX_ECLK_CENTERED_STATIC_BYPASS.u_lscc_gddrx2_4_5_tx_eclk_centered_static_bypass/eclkout_w} -period 2 [get_nets {mipi_rx[0].mipi_bridge_inst/lvds_tx_inst/lscc_gddr_inst/TX_ECLK_CENTERED_STATIC_BYPASS.u_lscc_gddrx2_4_5_tx_eclk_centered_static_bypass/eclkout_w}]
create_clock -name {mipi_rx[1].mipi_bridge_inst/lvds_tx_inst/lscc_gddr_inst/TX_ECLK_CENTERED_STATIC_BYPASS.u_lscc_gddrx2_4_5_tx_eclk_centered_static_bypass/eclkout_w} -period 2 [get_nets {mipi_rx[1].mipi_bridge_inst/lvds_tx_inst/lscc_gddr_inst/TX_ECLK_CENTERED_STATIC_BYPASS.u_lscc_gddrx2_4_5_tx_eclk_centered_static_bypass/eclkout_w}]
create_clock -name {mipi_rx[0].mipi_bridge_inst/mipi_rx_inst/clk_byte_o} -period 5.3 [get_nets {mipi_rx[0].mipi_bridge_inst/mipi_rx_inst/clk_byte_o}]
create_clock -name {mipi_rx[1].mipi_bridge_inst/mipi_rx_inst/clk_byte_o} -period 5.3 [get_nets {mipi_rx[1].mipi_bridge_inst/mipi_rx_inst/clk_byte_o}]


set_clock_groups -group [get_clocks {u_lvds_pll/clkop_o u_lvds_pll/clkos_o}] -group [get_clocks {mipi_rx[0].mipi_bridge_inst/mipi_rx_inst/clk_byte_hs_o}] -group [get_clocks {mipi_rx[1].mipi_bridge_inst/mipi_rx_inst/clk_byte_hs_o}] -group [get_clocks {mipi_rx[0].mipi_bridge_inst/lvds_tx_inst/sclk_o}] -group [get_clocks {mipi_rx[1].mipi_bridge_inst/lvds_tx_inst/sclk_o}] -group [get_clocks u_lvds_pll/clkos3_o] -group [get_clocks u_lvds_pll/clkos2_o] -asynchronous

