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

#----------------------------------------------------------------------------------------------
# FPGA Build Script
#----------------------------------------------------------------------------------------------

set prj_name    fpga_cpnx
set vendor_ID   0000
set date        [clock format [clock seconds] -format "%m%d%Y"]

#----------------------------------------------------------------------------------------------
# Create Project
#----------------------------------------------------------------------------------------------

set prj_dir $date
file mkdir $prj_dir/$prj_name
file mkdir $prj_dir/bitfile
cd $prj_dir/$prj_name

prj_create -name $prj_name \
           -impl "impl_1"  \
           -dev LFCPNX-100-9LFG672C \
           -performance "9_High-Performance_1.0V" \
           -synthesis "synplify"

prj_set_strategy_value -strategy Strategy1 syn_pipelining_retiming=Pipelining and Retiming \
                                            syn_arrange_vhdl_files=False \
                                            syn_critical_path_num=100 \
                                            syn_start_end_pt_num=20 \
                                            syn_frequency=157 \
                                            par_core_number=8 \
                                            par_place_iterator=8 \
                                            par_stop_zero=True

prj_set_strategy_value -strategy Strategy1 syn_ram_rw_check=False syn_allow_dup_modules=True syn_default_enum_encode=Onehot syn_res_sharing=False

prj_set_strategy_value -strategy Strategy1 par_spd_hld_opt=9_High-Performance_1.0V par_spd_setup_opt=9_High-Performance_1.0V


#Generate BitFile even if timing fails
#prj_set_strategy_value -strategy Strategy1 tmchk_enable_check=False

#Syn Netlist Output
#prj_set_strategy_value -strategy Strategy1 syn_output_netlist_format=VHDL
#
#----------------------------------------------------------------------------------------------
#Find Files Proc
#----------------------------------------------------------------------------------------------

# addFiles
# basedir - the directory to start looking in
# pattern - A pattern, as defined by the glob command, that the files must match

proc addFiles { basedir pattern } {

    # Fix the directory name, this ensures the directory name is in the
    # native format for the platform and contains a final directory separator
    set basedir [string trimright [file join [file normalize $basedir] { }]]

    # Look in the current directory for matching files, Added pkg files first
    # for the correct compile order

    set patternToExclude "pkg"

    foreach file [glob -nocomplain -type {f r} -path $basedir $pattern]  {
        if {[string first $patternToExclude [file tail $file]] >= 0} {
            puts $file
            prj_add_source $file
        }
    }

    # Adding the rest of the non pkg files

    foreach file [glob -nocomplain -type {f r} -path $basedir $pattern]  {
        if {[string first $patternToExclude [file tail $file]] == -1} {
            puts $file
            prj_add_source $file
        }
    }
}

#----------------------------------------------------------------------------------------------
# Add Source Files
#----------------------------------------------------------------------------------------------

set src_dir "../../../"
set hsb_dir "../../../../../../"

ip_catalog_install -vlnv latticesemi.com:ip:ten_gbe_mac:1.1.0
ip_catalog_install -vlnv latticesemi.com:ip:ten_gbe_pcs:1.3.0

# IP Generation
exec ipgen -o ../../ip/eclk_pll \
           -ip $env(RADIANT_PATH)/ip/lifcl/pll \
           -name eclk_pll \
           -cfg "../../ip/eclk_pll/eclk_pll.cfg" \
           -a LFCPNX -p LFCPNX-100 -t LFG672 -sp 9_High-Performance_1.0V -op COM

exec ipgen -o ../../ip/osc_clk \
           -ip $env(RADIANT_PATH)/ip/lifcl/osc \
           -name osc_clk \
           -cfg "../../ip/osc_clk/osc_clk.cfg" \
           -a LFCPNX -p LFCPNX-100 -t LFG672 -sp 9_High-Performance_1.0V -op COM

exec ipgen -o ../../ip/i2s_pll \
           -ip $env(RADIANT_PATH)/ip/lifcl/pll \
           -name i2s_pll \
           -cfg "../../ip/i2s_pll/i2s_pll.cfg" \
           -a LFCPNX -p LFCPNX-100 -t LFG672 -sp 9_High-Performance_1.0V -op COM

exec ipgen -o ../../ip/ptp_sensor_pll \
           -ip $env(RADIANT_PATH)/ip/lifcl/pll \
           -name ptp_sensor_pll \
           -cfg "../../ip/ptp_sensor_pll/ptp_sensor_pll.cfg" \
           -a LFCPNX -p LFCPNX-100 -t LFG672 -sp 9_High-Performance_1.0V -op COM

exec ipgen -o ../../ip/eth_10gb_mac \
           -vlnv latticesemi.com:ip:ten_gbe_mac:1.1.0 \
           -name eth_10gb_mac \
           -cfg "../../ip/eth_10gb_mac/eth_10gb_mac.cfg" \
           -a LFCPNX -p LFCPNX-100 -t LFG672 -sp 9_High-Performance_1.0V -op COM

exec ipgen -o ../../ip/eth_10gb_pcs_0 \
           -vlnv latticesemi.com:ip:ten_gbe_pcs:1.3.0 \
           -name eth_10gb_pcs_0 \
           -cfg "../../ip/eth_10gb_pcs_0/eth_10gb_pcs_0.cfg" \
           -a LFCPNX -p LFCPNX-100 -t LFG672 -sp 9_High-Performance_1.0V -op COM

exec ipgen -o ../../ip/eth_10gb_pcs_1 \
           -vlnv latticesemi.com:ip:ten_gbe_pcs:1.3.0 \
           -name eth_10gb_pcs_1 \
           -cfg "../../ip/eth_10gb_pcs_1/eth_10gb_pcs_1.cfg" \
           -a LFCPNX -p LFCPNX-100 -t LFG672 -sp 9_High-Performance_1.0V -op COM

exec ipgen -o ../../ip/lvds_ddr_rx \
           -ip $env(RADIANT_PATH)/ip/lifcl/ddr \
           -name lvds_ddr_rx \
           -cfg "../../ip/lvds_ddr_rx/lvds_ddr_rx.cfg" \
           -a LFCPNX -p LFCPNX-100 -t LFG672 -sp 9_High-Performance_1.0V -op COM

# Add IP
set src_ip "build/ip/*/*ipx"
addFiles $src_dir $src_ip

# Add hololink header
set hlnk_svh "rtl/top/*h"
addFiles $src_dir $hlnk_svh

# Add hololink RTL
set hlnk_rtl_src "nv_hsb_ip/*/*v"
addFiles $hsb_dir $hlnk_rtl_src

# Add hololink RTL
set hlnk_ref_src "nv_hsb_ip/ref_design/*/*v"
addFiles $hsb_dir $hlnk_ref_src

# Add RTL
set src_cpnx "rtl/*/*v"
addFiles $src_dir $src_cpnx

# Add Design Constraints
set src_pdc "build/constraints/post_syn.pdc"
addFiles $src_dir $src_pdc

set src_sdc "build/constraints/pre_syn.sdc"
addFiles $src_dir $src_sdc

#----------------------------------------------------------------------------------------------
# Set Tool Options
#----------------------------------------------------------------------------------------------

prj_set_impl_opt -impl "impl_1" "top" "FPGA_top"
prj_set_impl_opt -impl "impl_1" "VerilogStandard" "System Verilog"
prj_set_impl_opt -impl "impl_1" "include path" $src_dir/rtl/top
prj_set_impl_opt -impl "impl_1" -append "include path" $hsb_dir/nv_hsb_ip/top

#----------------------------------------------------------------------------------------------
# Generate Build Revision ID (Version + Datecode)
#----------------------------------------------------------------------------------------------

set dateTime [clock format [clock seconds] -format "%Y%m%d%H%M%S"]
set dateTime [format %x $dateTime]

prj_set_impl_opt -impl "impl_1" "HDL_PARAM" "BUILD_REV=48'h$dateTime$vendor_ID"

#----------------------------------------------------------------------------------------------
# Run Project
#----------------------------------------------------------------------------------------------

prj_run Synthesis -impl impl_1 -task SynTrace
prj_run Map -impl impl_1 -task MapTrace
prj_run PAR -impl impl_1 -task PARTrace
prj_run Export -impl impl_1 -task Bitgen
prj_save
prj_close

#----------------------------------------------------------------------------------------------
# Copy Bitfile
#----------------------------------------------------------------------------------------------

file copy ./impl_1/fpga_cpnx_impl_1.bit ../../../fpga_cpnx_$dateTime.bit
