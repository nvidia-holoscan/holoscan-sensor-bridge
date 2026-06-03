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


# RFSoC PYNQ Vivado build.
# build.sh runs build.tcl

# Project defaults. These can be overridden from the command line with
# -tclargs where it makes sense.
set part_name    "xczu48dr-ffvg1517-2-e"
set project_name "rfsoc-pynq"
set top_module   "FPGA_top"
set origin_dir   [file normalize "."]

# Print the short command-line help text.
proc usage {} {
    puts "Usage: build.tcl -tclargs \[--origin_dir <path>\] \[--project_name <name>\]"
}

# Normalize a path and stop immediately if the required file is missing.
proc require_file {path label} {
    set path [file normalize $path]
    if {![file exists $path]} {
        error "$label not found: $path"
    }
    return $path
}

# Add all Verilog/SystemVerilog files one directory level below root.
# Package files are added first so dependent modules compile in the right order.
proc add_sv_tree {fileset root} {
    set root [file normalize $root]

    # Make missing RTL directories fail loudly instead of producing a half-built
    # Vivado project with missing sources.
    if {![file isdirectory $root]} {
        error "RTL directory not found: $root"
    }

    # Match the original build pattern: root/*/*v catches .v and .sv files in
    # each immediate child directory.
    set matches [lsort [glob -nocomplain -type {f r} [file join $root * *v]]]
    if {[llength $matches] == 0} {
        puts "WARNING: no RTL files matched [file join $root * *v]"
        return
    }

    # Split package files from everything else so they are registered first.
    set pkgs {}
    set rest {}
    foreach path $matches {
        if {[string first "pkg" [file tail $path]] >= 0} {
            lappend pkgs $path
        } else {
            lappend rest $path
        }
    }

    # Add each RTL file to sources_1 and mark it as SystemVerilog.
    foreach path [concat $pkgs $rest] {
        puts "Adding RTL: $path"
        add_files -norecurse -fileset $fileset $path

        set file_obj [get_files -quiet -of_objects $fileset [list $path]]
        if {[llength $file_obj] == 0} {
            error "Vivado did not register added RTL file: $path"
        }
        set_property file_type SystemVerilog $file_obj
    }
}

# Vivado passes arguments through ::argv when this script is sourced in batch.
set args {}
if {[info exists ::argv]} {
    set args $::argv
}

# Handle the small set of options this build script supports.
for {set i 0} {$i < [llength $args]} {incr i} {
    set arg [lindex $args $i]
    switch -- $arg {
        --origin_dir {
            incr i
            if {$i >= [llength $args]} {
                error "--origin_dir needs a path"
            }
            set origin_dir [file normalize [lindex $args $i]]
        }
        --project_name {
            incr i
            if {$i >= [llength $args]} {
                error "--project_name needs a name"
            }
            set project_name [lindex $args $i]
        }
        --help {
            usage
            return
        }
        default {
            error "Unknown argument '$arg'. Use --help for usage."
        }
    }
}

# Resolve all source paths relative to the build directory.
set rtl_dir    [file normalize "$origin_dir/../../rtl"]
set hsb_ip_dir [file normalize "$origin_dir/../../../../nv_hsb_ip"]
set bd_root    [file normalize "$origin_dir/../ip/bd"]
set bd_file    [file normalize "$bd_root/design_1/design_1.bd"]
set bda_file   [file normalize "$bd_root/design_1/design_1.bda"]
set def_file   [file normalize "$rtl_dir/top/HOLOLINK_def.svh"]
set xdc_file   [file normalize "$origin_dir/../constraints/constraints.xdc"]

# Start a fresh Vivado project in the current directory.
create_project $project_name ./ -part $part_name -force

# Set only the project properties this build flow relies on.
set project_dir [get_property directory [current_project]]
set_property -dict [list \
    default_lib xil_defaultlib \
    enable_vhdl_2008 1 \
    ip_cache_permissions {read write} \
    ip_output_repo [file normalize "$project_dir/${project_name}.cache/ip"] \
    mem.enable_memory_map_generation 1 \
    part $part_name \
    revised_directory_structure 1 \
    sim.central_dir [file normalize "$project_dir/${project_name}.ip_user_files"] \
    simulator_language Mixed \
    xpm_libraries {XPM_CDC XPM_FIFO XPM_MEMORY} \
] [current_project]

# Get the main source fileset, creating it if Vivado did not create it for us.
if {[string equal [get_filesets -quiet sources_1] ""]} {
    create_fileset -srcset sources_1
}
set sources [get_filesets sources_1]

# Generate the block design only when the checked-in/generated BD is missing.
if {![file exists $bd_file]} {
    puts "Generating block design"

    set bd_gen [require_file "$origin_dir/../ip/bd_gen.tcl" "BD generator"]

    # bd_gen.tcl uses ::origin_dir_loc to decide where to write the BD. Preserve
    # any existing value so sourcing this script does not leak state.
    set had_origin_override [info exists ::origin_dir_loc]
    if {$had_origin_override} {
        set saved_origin_override $::origin_dir_loc
    }

    set ::origin_dir_loc $bd_root
    source $bd_gen

    if {$had_origin_override} {
        set ::origin_dir_loc $saved_origin_override
    } elseif {[info exists ::origin_dir_loc]} {
        unset ::origin_dir_loc
    }
}

# Make sure the key project inputs exist before asking Vivado to register them.
set bd_file  [require_file $bd_file "Block design"]
set bda_file [require_file $bda_file "Block design metadata"]
set def_file [require_file $def_file "RFSoC header"]

# Add the block design, BD metadata, and RFSoC header directly to sources_1.
add_files -norecurse -fileset $sources $bd_file $def_file $bda_file

# Mark the BD as managed so Vivado treats it as a real block design source.
set bd_obj [get_files -quiet -of_objects $sources [list $bd_file]]
if {[llength $bd_obj] == 0} {
    error "Vivado did not register the block design: $bd_file"
}
set_property registered_with_manager 1 $bd_obj

# The shared RFSoC defines file is a Verilog header, not a normal module.
set def_obj [get_files -quiet -of_objects $sources [list $def_file]]
if {[llength $def_obj] == 0} {
    error "Vivado did not register the RFSoC header: $def_file"
}
set_property file_type {Verilog Header} $def_obj

# Add the local HSB IP RTL and the board-level RFSoC RTL.
add_sv_tree $sources $hsb_ip_dir
add_sv_tree $sources $rtl_dir

# Tell Vivado the HDL top and let it recompute compile order.
set_property top $top_module $sources
update_compile_order -fileset sources_1

# Import the XDC constraints into constrs_1.
if {[string equal [get_filesets -quiet constrs_1] ""]} {
    create_fileset -constrset constrs_1
}
set constraints [get_filesets constrs_1]
set xdc_file [require_file $xdc_file "Constraint file"]
import_files -fileset $constraints [list $xdc_file]
set_property target_part $part_name $constraints

# Keep a sim fileset around with the same top, even though this script does not
# add dedicated simulation-only sources.
if {[string equal [get_filesets -quiet sim_1] ""]} {
    create_fileset -simset sim_1
}
set simset [get_filesets sim_1]
set_property top $top_module $simset
set_property top_lib xil_defaultlib $simset

# Create/configure the synthesis run.
if {[string equal [get_runs -quiet synth_1] ""]} {
    create_run -name synth_1 \
        -part $part_name \
        -flow {Vivado Synthesis 2024} \
        -strategy {Vivado Synthesis Defaults} \
        -constrset constrs_1
}
set_property -dict [list \
    flow {Vivado Synthesis 2024} \
    part $part_name \
    strategy {Vivado Synthesis Defaults} \
] [get_runs synth_1]
current_run -synthesis [get_runs synth_1]

# Create/configure the implementation run and enable bitstream generation.
if {[string equal [get_runs -quiet impl_1] ""]} {
    create_run -name impl_1 \
        -part $part_name \
        -flow {Vivado Implementation 2024} \
        -strategy {Vivado Implementation Defaults} \
        -constrset constrs_1 \
        -parent_run synth_1
}
set_property -dict [list \
    flow {Vivado Implementation 2024} \
    part $part_name \
    strategy {Vivado Implementation Defaults} \
    steps.write_bitstream.args.readback_file 0 \
    steps.write_bitstream.args.verbose 0 \
] [get_runs impl_1]
current_run -implementation [get_runs impl_1]

# Use more host threads and open the BD before launching the build.
set_param general.maxThreads 16
open_bd_design $bd_file

# Run synthesis and wait for it to finish before implementation starts.
puts "Starting synthesis"
launch_runs synth_1
wait_on_run synth_1

# Run implementation through write_bitstream and wait for the bit file.
puts "Starting implementation and bitstream generation"
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1

# Locate the generated bitstream in the Vivado project output directory.
set bit_src [file normalize "$project_dir/${project_name}.runs/impl_1/${top_module}.bit"]
set bit_src [require_file $bit_src "Generated bitstream"]

# Copy the bitstream to the shared bitfile directory with a timestamped name.
set bit_dir [file normalize "$origin_dir/../../bitfile"]
file mkdir $bit_dir

set stamp [clock format [clock seconds] -format "%Y%m%d%H%M%S"]
set bit_dst [file join $bit_dir "rfsoc_pynq_${stamp}.bit"]
file copy -force $bit_src $bit_dst

puts "Done: $bit_dst"
