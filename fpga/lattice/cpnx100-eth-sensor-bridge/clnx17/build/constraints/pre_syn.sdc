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

# The MIPI byte clocks and LVDS sclk_o clocks are hard-IP generated clocks.
# Their RTL-level names do not exist during Synplify constraint processing;
# post-synthesis constraints and IP-generated constraints handle them.
set_clock_groups -group [get_clocks {u_lvds_pll/clkop_o u_lvds_pll/clkos_o}] -group [get_clocks u_lvds_pll/clkos3_o] -group [get_clocks u_lvds_pll/clkos2_o] -asynchronous

