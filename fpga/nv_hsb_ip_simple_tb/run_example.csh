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
# Example script to run the Hololink Simple Testbench
# This demonstrates how to run the basic simulation
#==============================================================================

echo "=== NV HSB IP Simple Testbench Example ==="
echo ""

# Set environment variables for VCS
echo "Setting up VCS environment..."
setenv VCS_HOME "/home/tools/vcs/2023.03-SP2-8"
setenv PATH ${PATH}:${VCS_HOME}/bin
setenv DESIGNWARE_HOME /home/tools/synopsys/designware_home/
setenv VERDI_HOME /home/tools/debussy/verdi3_2023.03-SP2-8

echo "VCS_HOME: $VCS_HOME"
echo "DESIGNWARE_HOME: $DESIGNWARE_HOME" 
echo "VERDI_HOME: $VERDI_HOME"

echo "Current directory: $PWD"
echo "Available files:"
ls -la
echo ""

echo "=== Step 1: Clean any previous builds ==="
make clean
echo ""

echo "=== Step 2: Compile the testbench ==="
make compile
if ($status != 0) then
    echo "ERROR: Compilation failed!"
    exit 1
endif
echo "Compilation successful!"
echo ""

echo "=== Step 3: Run the simulation ==="
make run
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
echo "To modify the test:"
echo "  1. Edit nv_hsb_ip_simple_tb.sv for test sequence changes"
echo "  2. Edit fpga_sensor_top_pkg.svh for parameter changes"
echo "  3. Add actual DUT instantiation to replace placeholder"
echo ""

exit 0
