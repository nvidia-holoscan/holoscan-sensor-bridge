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

#ReLingo_waive_start:temp:RL01,RL02
# Block-design generator for the RFSoC PYNQ 100GbE design.
# When run directly, this writes ./bd/design_1/design_1.bd.
# build.tcl sets ::origin_dir_loc so the normal build writes to build/ip/bd.

set required_vivado "2024.1"
set part_name       "xczu48dr-ffvg1517-2-e"
set design_name     "design_1"
set bd_root         [file normalize "./bd"]

if {[info exists ::origin_dir_loc]} {
    set bd_root [file normalize $::origin_dir_loc]
}

# Keep the script tied to the Vivado version that generated the original BD.
if {[string first $required_vivado [version -short]] == -1} {
    error "This BD script expects Vivado $required_vivado, but found [version -short]"
}

# Let the script run from a standalone Vivado shell, but reuse the caller's project
# when build.tcl has already created one.
if {[get_projects -quiet] eq ""} {
    create_project project_1 myproj -part $part_name
}

# Stop early if a required IP is not available in this Vivado install.
proc require_ip_defs {ip_defs} {
    set missing {}

    foreach ip_vlnv $ip_defs {
        if {[get_ipdefs -all $ip_vlnv] eq ""} {
            lappend missing $ip_vlnv
        }
    }

    if {[llength $missing] != 0} {
        error "Missing IP definitions: $missing"
    }
}

# Create a remote block design at bd_root/design_name/design_name.bd.
proc create_remote_bd {design_name bd_root} {
    set bd_file [file normalize "$bd_root/$design_name/$design_name.bd"]

    if {[file exists $bd_file]} {
        error "BD already exists: $bd_file"
    }
    if {[get_bd_designs -quiet $design_name] ne ""} {
        error "BD already exists in memory: $design_name"
    }
    if {[get_files -quiet "*/${design_name}.bd"] ne ""} {
        error "BD already exists in this project: $design_name"
    }

    create_bd_design -dir $bd_root $design_name
    current_bd_design $design_name
}

# Build the CMAC/clock-wizard block design and wire its top-level ports.
proc create_design {parent_cell} {
    if {$parent_cell eq ""} {
        set parent_cell [get_bd_cells /]
    }

    set parent_obj [get_bd_cells $parent_cell]
    if {$parent_obj eq ""} {
        error "Could not find parent BD cell: $parent_cell"
    }
    if {[get_property TYPE $parent_obj] ne "hier"} {
        error "Parent BD cell is not hierarchical: $parent_cell"
    }

    set old_instance [current_bd_instance .]
    current_bd_instance $parent_obj

    # External interfaces exposed by the block design.
    set gt_ref_clk_0 [create_bd_intf_port \
        -mode Slave \
        -vlnv xilinx.com:interface:diff_clock_rtl:1.0 \
        gt_ref_clk_0]
    set_property CONFIG.FREQ_HZ 156250000 $gt_ref_clk_0

    set gt_serial_port_0 [create_bd_intf_port \
        -mode Master \
        -vlnv xilinx.com:interface:gt_rtl:1.0 \
        gt_serial_port_0]

    set core_drp_0 [create_bd_intf_port \
        -mode Slave \
        -vlnv xilinx.com:interface:drp_rtl:1.0 \
        core_drp_0]

    set ctl_rx_0 [create_bd_intf_port \
        -mode Slave \
        -vlnv xilinx.com:display_cmac_usplus:ctrl_ports:2.0 \
        ctl_rx_0]

    set ctl_tx_0 [create_bd_intf_port \
        -mode Slave \
        -vlnv xilinx.com:display_cmac_usplus:ctrl_ports:2.0 \
        ctl_tx_0]

    set axis_tx_0 [create_bd_intf_port \
        -mode Slave \
        -vlnv xilinx.com:interface:axis_rtl:1.0 \
        axis_tx_0]
    set_property -dict [list \
        CONFIG.HAS_TKEEP 1 \
        CONFIG.HAS_TLAST 1 \
        CONFIG.HAS_TREADY 1 \
        CONFIG.HAS_TSTRB 0 \
        CONFIG.LAYERED_METADATA undef \
        CONFIG.TDATA_NUM_BYTES 64 \
        CONFIG.TDEST_WIDTH 0 \
        CONFIG.TID_WIDTH 0 \
        CONFIG.TUSER_WIDTH 1 \
    ] $axis_tx_0

    set axis_rx_0 [create_bd_intf_port \
        -mode Master \
        -vlnv xilinx.com:interface:axis_rtl:1.0 \
        axis_rx_0]
    set_property CONFIG.FREQ_HZ 322265625 $axis_rx_0

    # External scalar ports exposed by the block design.
    set clk_out1_0        [create_bd_port -dir O -type clk clk_out1_0]
    set gt_txusrclk2_0    [create_bd_port -dir O -type clk gt_txusrclk2_0]
    set stat_rx_aligned_0 [create_bd_port -dir O stat_rx_aligned_0]
    set usr_tx_reset_0    [create_bd_port -dir O -type rst usr_tx_reset_0]
    set sys_reset_0       [create_bd_port -dir I -type rst sys_reset_0]
    set gt_loopback_in_0  [create_bd_port -dir I -from 11 -to 0 gt_loopback_in_0]
    set clk_out2_0        [create_bd_port -dir O -type clk clk_out2_0]
    set locked_0          [create_bd_port -dir O locked_0]
    set init_clk_0        [create_bd_port -dir I -type clk -freq_hz 100017438 init_clk_0]
    set gt_powergoodout_0 [create_bd_port -dir O -from 3 -to 0 gt_powergoodout_0]
    set gt_ref_clk_out_0  [create_bd_port -dir O -type clk gt_ref_clk_out_0]

    set_property -dict [list CONFIG.ASSOCIATED_BUSIF axis_tx_0] $gt_txusrclk2_0
    set_property CONFIG.ASSOCIATED_BUSIF.VALUE_SRC DEFAULT $gt_txusrclk2_0
    set_property CONFIG.POLARITY ACTIVE_HIGH $sys_reset_0

    # The CMAC instance is the 100GbE MAC/PCS with GT support.
    set cmac_usplus_0 [create_bd_cell \
        -type ip \
        -vlnv xilinx.com:ip:cmac_usplus:3.1 \
        cmac_usplus_0]
    set_property -dict [list \
        CONFIG.CMAC_CAUI4_MODE 1 \
        CONFIG.GT_REF_CLK_FREQ 156.25 \
        CONFIG.INS_LOSS_NYQ 1 \
        CONFIG.RX_EQ_MODE LPM \
        CONFIG.RX_FLOW_CONTROL 0 \
        CONFIG.TX_FLOW_CONTROL 0 \
        CONFIG.USER_INTERFACE AXIS \
    ] $cmac_usplus_0

    # The clock wizard derives the slower clocks used outside the CMAC.
    set clk_wiz_0 [create_bd_cell \
        -type ip \
        -vlnv xilinx.com:ip:clk_wiz:6.0 \
        clk_wiz_0]
    set_property -dict [list \
        CONFIG.CLKIN1_JITTER_PS 31.03 \
        CONFIG.CLKOUT1_JITTER 175.085 \
        CONFIG.CLKOUT1_PHASE_ERROR 280.757 \
        CONFIG.CLKOUT1_REQUESTED_OUT_FREQ 50.000 \
        CONFIG.CLKOUT2_JITTER 159.339 \
        CONFIG.CLKOUT2_PHASE_ERROR 280.757 \
        CONFIG.CLKOUT2_USED true \
        CONFIG.MMCM_CLKFBOUT_MULT_F 108.625 \
        CONFIG.MMCM_CLKIN1_PERIOD 3.103 \
        CONFIG.MMCM_CLKIN2_PERIOD 10.0 \
        CONFIG.MMCM_CLKOUT0_DIVIDE_F 28.000 \
        CONFIG.MMCM_CLKOUT1_DIVIDE 14 \
        CONFIG.MMCM_DIVCLK_DIVIDE 25 \
        CONFIG.NUM_OUT_CLKS 2 \
        CONFIG.PRIM_IN_FREQ 322.265625 \
        CONFIG.PRIM_SOURCE Global_buffer \
        CONFIG.USE_RESET false \
    ] $clk_wiz_0

    # Tie unused reset/DRP pins low.
    set xlconstant_0 [create_bd_cell \
        -type ip \
        -vlnv xilinx.com:ip:xlconstant:1.1 \
        xlconstant_0]
    set_property CONFIG.CONST_VAL 0 $xlconstant_0

    # Interface-level wiring.
    connect_bd_intf_net -intf_net axis_tx_0_1 \
        [get_bd_intf_ports axis_tx_0] \
        [get_bd_intf_pins cmac_usplus_0/axis_tx]
    connect_bd_intf_net -intf_net cmac_usplus_0_axis_rx \
        [get_bd_intf_ports axis_rx_0] \
        [get_bd_intf_pins cmac_usplus_0/axis_rx]
    connect_bd_intf_net -intf_net cmac_usplus_0_gt_serial_port \
        [get_bd_intf_ports gt_serial_port_0] \
        [get_bd_intf_pins cmac_usplus_0/gt_serial_port]
    connect_bd_intf_net -intf_net core_drp_0_1 \
        [get_bd_intf_ports core_drp_0] \
        [get_bd_intf_pins cmac_usplus_0/core_drp]
    connect_bd_intf_net -intf_net ctl_rx_0_1 \
        [get_bd_intf_ports ctl_rx_0] \
        [get_bd_intf_pins cmac_usplus_0/ctl_rx]
    connect_bd_intf_net -intf_net ctl_tx_0_1 \
        [get_bd_intf_ports ctl_tx_0] \
        [get_bd_intf_pins cmac_usplus_0/ctl_tx]
    connect_bd_intf_net -intf_net gt_ref_clk_0_1 \
        [get_bd_intf_ports gt_ref_clk_0] \
        [get_bd_intf_pins cmac_usplus_0/gt_ref_clk]

    # Scalar clock/status/reset wiring.
    connect_bd_net -net clk_wiz_0_clk_out1 \
        [get_bd_pins clk_wiz_0/clk_out1] \
        [get_bd_ports clk_out1_0]
    connect_bd_net -net clk_wiz_0_clk_out2 \
        [get_bd_pins clk_wiz_0/clk_out2] \
        [get_bd_ports clk_out2_0]
    connect_bd_net -net clk_wiz_0_locked \
        [get_bd_pins clk_wiz_0/locked] \
        [get_bd_ports locked_0]
    connect_bd_net -net cmac_usplus_0_gt_powergoodout \
        [get_bd_pins cmac_usplus_0/gt_powergoodout] \
        [get_bd_ports gt_powergoodout_0]
    connect_bd_net -net cmac_usplus_0_gt_ref_clk_out \
        [get_bd_pins cmac_usplus_0/gt_ref_clk_out] \
        [get_bd_ports gt_ref_clk_out_0]
    connect_bd_net -net cmac_usplus_0_gt_txusrclk2 \
        [get_bd_pins cmac_usplus_0/gt_txusrclk2] \
        [get_bd_ports gt_txusrclk2_0] \
        [get_bd_pins clk_wiz_0/clk_in1] \
        [get_bd_pins cmac_usplus_0/rx_clk]
    connect_bd_net -net cmac_usplus_0_stat_rx_aligned \
        [get_bd_pins cmac_usplus_0/stat_rx_aligned] \
        [get_bd_ports stat_rx_aligned_0]
    connect_bd_net -net cmac_usplus_0_usr_tx_reset \
        [get_bd_pins cmac_usplus_0/usr_tx_reset] \
        [get_bd_ports usr_tx_reset_0]
    connect_bd_net -net gt_loopback_in_0_1 \
        [get_bd_ports gt_loopback_in_0] \
        [get_bd_pins cmac_usplus_0/gt_loopback_in]
    connect_bd_net -net init_clk_0_1 \
        [get_bd_ports init_clk_0] \
        [get_bd_pins cmac_usplus_0/init_clk]
    connect_bd_net -net sys_reset_0_1 \
        [get_bd_ports sys_reset_0] \
        [get_bd_pins cmac_usplus_0/sys_reset]
    connect_bd_net -net xlconstant_0_dout \
        [get_bd_pins xlconstant_0/dout] \
        [get_bd_pins cmac_usplus_0/drp_clk] \
        [get_bd_pins cmac_usplus_0/gtwiz_reset_tx_datapath] \
        [get_bd_pins cmac_usplus_0/gtwiz_reset_rx_datapath] \
        [get_bd_pins cmac_usplus_0/core_rx_reset] \
        [get_bd_pins cmac_usplus_0/core_tx_reset] \
        [get_bd_pins cmac_usplus_0/core_drp_reset]

    current_bd_instance $old_instance
    validate_bd_design
    save_bd_design
}

require_ip_defs [list \
    xilinx.com:ip:cmac_usplus:3.1 \
    xilinx.com:ip:clk_wiz:6.0 \
    xilinx.com:ip:xlconstant:1.1 \
]

create_remote_bd $design_name $bd_root
create_design ""

set bd_file [file normalize "$bd_root/$design_name/$design_name.bd"]
puts "Done: $bd_file"
#ReLingo_waive_end:temp:RL01,RL02