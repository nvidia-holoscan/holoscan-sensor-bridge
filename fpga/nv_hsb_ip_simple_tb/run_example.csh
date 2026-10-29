#!/bin/csh -f
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#==============================================================================
# Example script to run the NV HSB IP Simple Testbench
# This demonstrates how to run the basic simulation
#==============================================================================

set SCRIPT_DIR = `dirname "$0"`
cd "$SCRIPT_DIR"
setenv HOLOLINK_TB_DIR "$cwd"
source ./setup_env.csh

echo "=== NV HSB IP Simple Testbench Example ==="
echo ""

set TEST_MODE_ARG = "SINGLE_SENSOR"
if ($#argv >= 1) then
    set TEST_MODE_ARG = "$argv[1]"
endif

set TARGET_HOST_ARG = "0"
if ($#argv >= 2) then
    set TARGET_HOST_ARG = "$argv[2]"
endif

echo "Selected TEST_MODE: $TEST_MODE_ARG"
echo "Selected TARGET_HOST: $TARGET_HOST_ARG"
echo "Usage examples:"
echo "  ./run_example.csh SINGLE_SENSOR"
echo "  ./run_example.csh ALL_SENSORS 0"
echo ""

echo "Using environment from setup_env.csh..."
echo "SIMULATOR: $SIMULATOR"
echo "HOLOLINK_PATH: $HOLOLINK_PATH"
echo "HOLOLINK_DEF_INCLUDE: $HOLOLINK_DEF_INCLUDE"
echo "VCS_HOME: $VCS_HOME"
echo "DESIGNWARE_HOME: $DESIGNWARE_HOME" 
echo "VERDI_HOME: $VERDI_HOME"

echo "Current directory: $PWD"
echo "Available files:"
ls -la
echo ""

echo "=== Step 1: Clean any previous builds ==="
make clean SIMULATOR=$SIMULATOR
echo ""

echo "=== Step 2: Compile the testbench ==="
make compile SIMULATOR=$SIMULATOR
if ($status != 0) then
    echo "ERROR: Compilation failed!"
    exit 1
endif
echo "Compilation successful!"
echo ""

echo "=== Step 3: Run the simulation ==="
make run SIMULATOR=$SIMULATOR TEST_MODE=$TEST_MODE_ARG TARGET_HOST=$TARGET_HOST_ARG
if ($status != 0) then
    echo "ERROR: Simulation failed!"
    exit 1
endif
echo ""

echo "=== Step 4: Check results ==="
if (-f waves.fsdb) then
    echo "Waveform file generated: waves.fsdb"
    echo "Use Verdi or DVE to view waveforms:"
    echo "  verdi -ssf waves.fsdb &"
else
    echo "No waveform file found - check simulation settings"
endif
echo ""

echo "=== Simulation Complete ==="
echo ""
echo "To run with GUI:"
echo "  make gui"
echo ""
echo "To run with custom options:"
echo "  ./nv_hsb_ip_simple_tb.simv +fsdbfile+custom.fsdb"
echo ""
echo "To run all Sensor RX interfaces into host 0:"
echo "  ./run_example.csh ALL_SENSORS 0"
echo ""
echo "To modify the test:"
echo "  1. Edit nv_hsb_ip_simple_tb.sv for test sequence changes"
echo "  2. Edit fpga_sensor_top_pkg.svh for parameter changes"
echo "  3. Add actual DUT instantiation to replace placeholder"
echo ""

exit 0
