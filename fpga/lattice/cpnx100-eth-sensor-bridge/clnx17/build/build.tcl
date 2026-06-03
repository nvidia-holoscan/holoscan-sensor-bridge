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

set prj_name    fpga_clnx
set fpga_version 2603
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
           -dev LIFCL-17-9BG256C \
           -performance "9_High-Performance_1.0V" \
           -synthesis "synplify"

prj_set_strategy_value -strategy Strategy1 {syn_pipelining_retiming=Pipelining and Retiming \
										                        syn_arrange_vhdl_files=False \
											                      syn_critical_path_num=100 \
											                      syn_start_end_pt_num=20 \
											                      syn_frequency=157 \
                                            par_core_number=8 \
                                            par_place_iterator=8 \
											                      par_stop_zero=True}

prj_set_strategy_value -strategy Strategy1 syn_ram_rw_check=False

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
ip_catalog_install -vlnv latticesemi.com:ip:dphy_rx:2.1.0

set src_dir "../../../"
set hsb_dir "../../../../../../"

exec ipgen -o ../../ip/lvds_pll \
           -ip $env(RADIANT_PATH)/ip/lifcl/pll \
           -name lvds_pll \
           -cfg "../../ip/lvds_pll/lvds_pll.cfg" \
           -a LIFCL -p LIFCL-17 -t CABGA256 -sp 9_High-Performance_1.0V -op COM

exec ipgen -o ../../ip/mipi_csi_rx_rcfg \
           -vlnv latticesemi.com:ip:dphy_rx:2.1.0 \
           -name mipi_csi_rx_rcfg \
           -cfg "../../ip/mipi_csi_rx_rcfg/mipi_csi_rx_rcfg.cfg" \
           -a LIFCL -p LIFCL-17 -t CABGA256 -sp 9_High-Performance_1.0V -op COM

exec ipgen -o ../../ip/lvds_tx \
           -ip $env(RADIANT_PATH)/ip/lifcl/ddr \
           -name lvds_tx \
           -cfg "../../ip/lvds_tx/lvds_tx.cfg" \
           -a LIFCL -p LIFCL-17 -t CABGA256 -sp 9_High-Performance_1.0V -op COM

# Add IP
set src_ip "build/ip/*/*ipx"
addFiles $src_dir $src_ip

# Add hololink header
set hlnk_svh "../cpnx100/rtl/top/*h"
addFiles $src_dir $hlnk_svh

# Add hololink RTL
set hlnk_rst_src "nv_hsb_ip/reg_map/*v"
addFiles $hsb_dir $hlnk_rst_src

# Add hololink RTL
set hlnk_rst_src "nv_hsb_ip/lib_axis/*v"
addFiles $hsb_dir $hlnk_rst_src

# Add hololink RTL
set hlnk_rst_src "nv_hsb_ip/lib_apb/*v"
addFiles $hsb_dir $hlnk_rst_src

# Add hololink RTL
set hlnk_rst_src "nv_hsb_ip/misc/*v"
addFiles $hsb_dir $hlnk_rst_src

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

prj_set_impl_opt -impl "impl_1" "top" "FPGA_clnx_top"
prj_set_impl_opt -impl "impl_1" "VerilogStandard" "System Verilog"
prj_set_impl_opt -impl "impl_1" "include path" $src_dir/../cpnx100/rtl/top

#----------------------------------------------------------------------------------------------
# Generate Build Revision ID (Version + Datecode)
#----------------------------------------------------------------------------------------------	

set dateTime [clock format [clock seconds] -format "%Y%m%d%H%M%S"]
set dateTime [format %x $dateTime]

prj_set_impl_opt -impl "impl_1" "HDL_PARAM" "FPGA_VERSION=16'h$fpga_version" 

prj_set_strategy_value -strategy Strategy1 tmchk_enable_check=False
prj_set_strategy_value -strategy Strategy1 syn_arrange_vhdl_files=True syn_force_gsr=False syn_critical_path_num= syn_start_end_pt_num= syn_frequency=200 syn_res_sharing=False par_prioritize_hldcorrection=True

#----------------------------------------------------------------------------------------------
# Run Project
#----------------------------------------------------------------------------------------------	

prj_run Synthesis -impl impl_1
prj_run Map -impl impl_1
prj_run PAR -impl impl_1 
prj_run Export -impl impl_1 -task Bitgen
prj_save
prj_close

#----------------------------------------------------------------------------------------------
# Copy Bitfile
#----------------------------------------------------------------------------------------------

file copy ./impl_1/fpga_clnx_impl_1.bit ../../../fpga_clnx_$dateTime.bit
