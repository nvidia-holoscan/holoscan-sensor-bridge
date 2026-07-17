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
# Makefile for NV HSB IP Simple Testbench
# Supports both VCS and XCELIUM simulators
#==============================================================================

# Simulator selection (VCS or XCELIUM)
SIMULATOR ?= VCS

# Validate simulator choice
ifneq ($(SIMULATOR),VCS)
ifneq ($(SIMULATOR),XCELIUM)
$(error Invalid SIMULATOR=$(SIMULATOR). Valid options: VCS, XCELIUM)
endif
endif

# Testbench and source files
TB_TOP = nv_hsb_ip_simple_tb
SV_FILES = $(TB_TOP).sv
INCLUDE_FILES = fpga_sensor_top_pkg.svh

#==============================================================================
# Simulator-Specific Configuration
#==============================================================================

ifeq ($(SIMULATOR),VCS)
    # VCS compiler and options
    SIM_CMD = vcs
    COMP_OPTS = +v2k -sverilog -timescale=1ns/1ps -debug_access+all -l vcs.log
    COMP_OPTS += -kdb -lca -debug_region=cell+encrypt -full64
    COMP_OPTS += +fsdb+mda=on +fsdb+functions +fsdb+glitch=on

    # VCS waveform options
    WAVE_OPTS = +fsdbfile+waves.fsdb
    VCD_OPTS = +vcs+dumpvars+waves.vcd

    # VCS output binary
    SIM_EXEC = ./$(TB_TOP).simv

    # VCS-specific log files
    RUN_LOG = run.log
    COE_LOG = coe_run.log
    COMPILE_LOG = vcs.log

else ifeq ($(SIMULATOR),XCELIUM)
    # XCELIUM compiler and options (using xrun command)
    SIM_CMD = xrun
    COMP_OPTS = -64bit -sv -timescale 1ps/1ps
    COMP_OPTS += -access +rw
    COMP_OPTS += -top $(TB_TOP)

    # Relaxation options for compatibility
    COMP_OPTS += -relax_svbtis
    COMP_OPTS += -vlogcontrolrelax NOTDOT
    COMP_OPTS += -vlogcontrolrelax PMBNOB
    COMP_OPTS += -vlogcontrolrelax EXPRLC

    # Advanced SV features
    COMP_OPTS += -enable_use_before_decl_in_class
    COMP_OPTS += -enable_with_opr_stream
    COMP_OPTS += -enable_const_sformatf
    COMP_OPTS += -enable_vif_uninst_if

    # Performance optimizations
    COMP_OPTS += -delay_trigger
    COMP_OPTS += -always_trigger

    # Warning suppression
    COMP_OPTS += -nowarn UEXPSC
    COMP_OPTS += -nowarn MACRDF
    COMP_OPTS += -nowarn NOSYST

    # No UVM library needed - this is a simple SystemVerilog testbench (non-UVM)
    # Uses basic tasks/functions for stimulus and checking, not UVM methodology

    # XCELIUM waveform options (VCD for compatibility, SHM requires TCL script)
    WAVE_OPTS =
    VCD_OPTS =

    # XCELIUM runs directly (no separate executable)
    SIM_EXEC = $(SIM_CMD)

    # XCELIUM-specific log files
    RUN_LOG = xcelium_run.log
    COE_LOG = xcelium_coe.log
    COMPILE_LOG = xcelium.log

endif

# Include paths - MUST be provided by user or build system
# These paths are required and must point to the RTL source directories
ifndef HOLOLINK_PATH
$(error HOLOLINK_PATH is not set. Please set it to the path of the hololink RTL directory)
endif

ifndef HOLOLINK_DEF_INCLUDE
$(error HOLOLINK_DEF_INCLUDE is not set. Please set it to the path of the  HOLOLINK_def.svh file directory)
endif
INCLUDE_OPTS = +incdir+.
INCLUDE_OPTS += +incdir+$(HOLOLINK_DEF_INCLUDE)
# Original curated include list (kept for reference):
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/top
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/enum_int
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/common
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/roce
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/misc
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/ecb
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/spi
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/i2c
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/lib_apb
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/lib_axis
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/dp_pkt
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/eth_pkt
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/bootp
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/rx_parser
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/glb_ctrl
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/packetizer
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/sensor_tx
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/routing_table
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/reg_map
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/sys_init
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/ctrl_bus_evt
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/reset_gen
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/data_gen
# INCLUDE_OPTS += +incdir+$(HOLOLINK_PATH)/ptp

# Dynamically include all immediate subdirectories under hololink (directories only)
INCLUDE_OPTS += $(addprefix +incdir+,$(patsubst %/,%,$(wildcard $(HOLOLINK_PATH)/*/)))

# Library paths - let VCS search all immediate subdirs for modules automatically.
# Original curated library list (kept for reference):
# LIB_OPTS = -y $(HOLOLINK_PATH)/top
# LIB_OPTS += -y $(HOLOLINK_PATH)/enum_int
# LIB_OPTS += -y $(HOLOLINK_PATH)/roce
# LIB_OPTS += -y $(HOLOLINK_PATH)/misc
# LIB_OPTS += -y $(HOLOLINK_PATH)/ecb
# LIB_OPTS += -y $(HOLOLINK_PATH)/spi
# LIB_OPTS += -y $(HOLOLINK_PATH)/i2c
# LIB_OPTS += -y $(HOLOLINK_PATH)/lib_apb
# LIB_OPTS += -y $(HOLOLINK_PATH)/lib_axis
# LIB_OPTS += -y $(HOLOLINK_PATH)/dp_pkt
# LIB_OPTS += -y $(HOLOLINK_PATH)/eth_pkt
# LIB_OPTS += -y $(HOLOLINK_PATH)/bootp
# LIB_OPTS += -y $(HOLOLINK_PATH)/rx_parser
# LIB_OPTS += -y $(HOLOLINK_PATH)/glb_ctrl
# LIB_OPTS += -y $(HOLOLINK_PATH)/packetizer
# LIB_OPTS += -y $(HOLOLINK_PATH)/sensor_tx
# LIB_OPTS += -y $(HOLOLINK_PATH)/routing_table
# LIB_OPTS += -y $(HOLOLINK_PATH)/reg_map
# LIB_OPTS += -y $(HOLOLINK_PATH)/sys_init
# LIB_OPTS += -y $(HOLOLINK_PATH)/ctrl_bus_evt
# LIB_OPTS += -y $(HOLOLINK_PATH)/reset_gen
# LIB_OPTS += -y $(HOLOLINK_PATH)/data_gen
# LIB_OPTS += -y $(HOLOLINK_PATH)/ptp

LIB_DIRS := $(patsubst %/,%,$(wildcard $(HOLOLINK_PATH)/*/))
LIB_OPTS = $(addprefix -y ,$(LIB_DIRS))

# Package files - following exact order from working UVM testbench
PKG_OPTS = $(HOLOLINK_PATH)/ptp/ptp_pkg.sv
PKG_OPTS += $(HOLOLINK_PATH)/ecb/ecb_pkg.sv
PKG_OPTS += $(HOLOLINK_PATH)/lib_apb/apb_pkg.sv
PKG_OPTS += $(HOLOLINK_PATH)/rx_parser/rx_parser_pkg.sv
PKG_OPTS += $(HOLOLINK_PATH)/lib_axis/axis_pkg.sv
PKG_OPTS += $(HOLOLINK_PATH)/reg_map/regmap_pkg.sv

# Testbench library includes (included via `include directives in source files):
# - tb_lib/hololink_ecb_tasks.svh     (ECB read/write tasks)
# - tb_lib/hololink_sensor_stim.svh   (sensor data generation) 
# - tb_lib/hololink_monitors.svh      (unified host monitoring)
# - tb_lib/hololink_roce_helpers.svh  (RoCE protocol support)
# - tb_lib/hololink_coe_helpers.svh   (COE/IEEE 1722B support)

# Source files - packages first, then testbench modules
SRC_OPTS = $(PKG_OPTS)
SRC_OPTS += $(SV_FILES)
SRC_OPTS += +libext+.v+.sv+.svp+.vt+.gv+.vlib+.bvrl+

# Default target
all: env_check compile run

# Check environment setup
env_check:
	@echo "Checking $(SIMULATOR) environment setup..."
ifeq ($(SIMULATOR),VCS)
	@if [ -z "$$VCS_HOME" ]; then \
		echo "WARNING: VCS_HOME not set. Please run: source setup_env.csh"; \
		echo "Or manually set:"; \
		echo "  setenv VCS_HOME \"/home/tools/vcs/2023.03-SP2-8\""; \
		echo "  setenv PATH \$${PATH}:\$${VCS_HOME}/bin"; \
		echo "  setenv DESIGNWARE_HOME /home/tools/synopsys/designware_home/"; \
		echo "  setenv VERDI_HOME /home/tools/debussy/verdi3_2023.03-SP2-8"; \
	else \
		echo "VCS_HOME: $$VCS_HOME"; \
		echo "Environment appears to be set correctly."; \
	fi
else ifeq ($(SIMULATOR),XCELIUM)
	@echo "Checking for XCELIUM/xrun in PATH..."
	@if command -v xrun > /dev/null 2>&1; then \
		echo "xrun found: $$(which xrun)"; \
		echo "XCELIUM version: $$(xrun -version 2>&1 | grep -i xcelium | head -1)"; \
		echo "Environment appears to be set correctly."; \
	else \
		echo "WARNING: xrun not found in PATH"; \
		echo "Please ensure XCELIUM is in your PATH or set XCELIUM_HOME"; \
		echo "Example:"; \
		echo "  setenv PATH /home/tools/cadence/XCELIUM24.06.071/tools/bin:\$$PATH"; \
	fi
endif
	@echo ""

# Compile the testbench
compile:
	@echo "Compiling Hololink Simple Testbench with $(SIMULATOR)..."
ifeq ($(SIMULATOR),VCS)
	$(SIM_CMD) $(COMP_OPTS) $(INCLUDE_OPTS) $(LIB_OPTS) $(SRC_OPTS) -o $(TB_TOP).simv
	@echo "VCS compilation complete. Binary: $(TB_TOP).simv"
else ifeq ($(SIMULATOR),XCELIUM)
	@echo "XCELIUM uses single-step compile+run mode. Run 'make run' or 'make coe' to simulate."
	@echo "Compilation check will be performed during simulation."
endif

# Run the simulation (supports VERBOSITY=LOW/MEDIUM/HIGH)
run:
	@echo "Running $(SIMULATOR) simulation..."
ifeq ($(SIMULATOR),VCS)
	@$(MAKE) compile SIMULATOR=$(SIMULATOR)
	@if [ "$(VERBOSITY)" = "HIGH" ]; then \
		$(SIM_EXEC) $(WAVE_OPTS) +VERBOSITY=HIGH -l $(RUN_LOG); \
	elif [ "$(VERBOSITY)" = "MEDIUM" ]; then \
		$(SIM_EXEC) $(WAVE_OPTS) +VERBOSITY=MEDIUM -l $(RUN_LOG); \
	else \
		$(SIM_EXEC) $(WAVE_OPTS) +VERBOSITY=LOW -l $(RUN_LOG); \
	fi
	@echo "VCS simulation complete. Log: $(RUN_LOG)"
else ifeq ($(SIMULATOR),XCELIUM)
	@if [ "$(VERBOSITY)" = "HIGH" ]; then \
		$(SIM_CMD) $(COMP_OPTS) $(INCLUDE_OPTS) $(LIB_OPTS) $(SRC_OPTS) +VERBOSITY=HIGH -l $(RUN_LOG); \
	elif [ "$(VERBOSITY)" = "MEDIUM" ]; then \
		$(SIM_CMD) $(COMP_OPTS) $(INCLUDE_OPTS) $(LIB_OPTS) $(SRC_OPTS) +VERBOSITY=MEDIUM -l $(RUN_LOG); \
	else \
		$(SIM_CMD) $(COMP_OPTS) $(INCLUDE_OPTS) $(LIB_OPTS) $(SRC_OPTS) +VERBOSITY=LOW -l $(RUN_LOG); \
	fi
	@echo "XCELIUM simulation complete. Log: $(RUN_LOG)"
endif

# Run COE (Camera over Ethernet) simulation
coe:
	@echo "Running $(SIMULATOR) COE (Camera over Ethernet) simulation..."
ifeq ($(SIMULATOR),VCS)
	@$(MAKE) compile SIMULATOR=$(SIMULATOR)
	@if [ "$(VERBOSITY)" = "HIGH" ]; then \
		$(SIM_EXEC) $(WAVE_OPTS) +ENABLE_COE +VERBOSITY=HIGH -l $(COE_LOG); \
	elif [ "$(VERBOSITY)" = "MEDIUM" ]; then \
		$(SIM_EXEC) $(WAVE_OPTS) +ENABLE_COE +VERBOSITY=MEDIUM -l $(COE_LOG); \
	else \
		$(SIM_EXEC) $(WAVE_OPTS) +ENABLE_COE +VERBOSITY=LOW -l $(COE_LOG); \
	fi
	@echo "VCS COE simulation complete. Log: $(COE_LOG)"
else ifeq ($(SIMULATOR),XCELIUM)
	@if [ "$(VERBOSITY)" = "HIGH" ]; then \
		$(SIM_CMD) $(COMP_OPTS) $(INCLUDE_OPTS) $(LIB_OPTS) $(SRC_OPTS) +ENABLE_COE +VERBOSITY=HIGH -l $(COE_LOG); \
	elif [ "$(VERBOSITY)" = "MEDIUM" ]; then \
		$(SIM_CMD) $(COMP_OPTS) $(INCLUDE_OPTS) $(LIB_OPTS) $(SRC_OPTS) +ENABLE_COE +VERBOSITY=MEDIUM -l $(COE_LOG); \
	else \
		$(SIM_CMD) $(COMP_OPTS) $(INCLUDE_OPTS) $(LIB_OPTS) $(SRC_OPTS) +ENABLE_COE +VERBOSITY=LOW -l $(COE_LOG); \
	fi
	@echo "XCELIUM COE simulation complete. Log: $(COE_LOG)"
endif

# Run with GUI
gui:
	@echo "Running $(SIMULATOR) simulation with GUI..."
ifeq ($(SIMULATOR),VCS)
	@$(MAKE) compile SIMULATOR=$(SIMULATOR)
	$(SIM_EXEC) -gui -i dvr_run.tcl
	@echo "VCS DVE GUI launched."
else ifeq ($(SIMULATOR),XCELIUM)
	$(SIM_CMD) -gui $(COMP_OPTS) $(INCLUDE_OPTS) $(LIB_OPTS) $(SRC_OPTS) +VERBOSITY=HIGH
	@echo "XCELIUM SimVision GUI launched."
endif

# Run COE simulation with GUI
coe-gui:
	@echo "Running $(SIMULATOR) COE (Camera over Ethernet) simulation with GUI..."
ifeq ($(SIMULATOR),VCS)
	@$(MAKE) compile SIMULATOR=$(SIMULATOR)
	$(SIM_EXEC) -gui +ENABLE_COE +VERBOSITY=HIGH
	@echo "VCS DVE GUI launched (COE mode)."
else ifeq ($(SIMULATOR),XCELIUM)
	$(SIM_CMD) -gui $(COMP_OPTS) $(INCLUDE_OPTS) $(LIB_OPTS) $(SRC_OPTS) +ENABLE_COE +VERBOSITY=HIGH
	@echo "XCELIUM SimVision GUI launched (COE mode)."
endif

# Run with Verdi (VCS only - not applicable for XCELIUM)
verdi:
ifeq ($(SIMULATOR),VCS)
	@$(MAKE) compile SIMULATOR=$(SIMULATOR)
	@echo "Running VCS simulation with Verdi..."
	$(SIM_EXEC) $(WAVE_OPTS) -verdi +VERBOSITY=LOW
else
	@echo "ERROR: Verdi target is only available with VCS simulator."
	@echo "Use 'make gui' for XCELIUM SimVision GUI instead."
	@exit 1
endif

# Debug run with VCD dumps (most compatible)
debug:
	@echo "Running $(SIMULATOR) simulation with VCD dumps for debugging..."
ifeq ($(SIMULATOR),VCS)
	@$(MAKE) compile SIMULATOR=$(SIMULATOR)
	$(SIM_EXEC) $(VCD_OPTS) +VERBOSITY=HIGH
	@echo "VCS VCD file generated: waves.vcd"
else ifeq ($(SIMULATOR),XCELIUM)
	$(SIM_CMD) $(COMP_OPTS) $(INCLUDE_OPTS) $(LIB_OPTS) $(SRC_OPTS) +VERBOSITY=HIGH
	@echo "XCELIUM simulation complete. Check $(RUN_LOG) for results."
endif

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts for all simulators..."
	@# VCS artifacts
	rm -rf csrc $(TB_TOP).simv $(TB_TOP).simv.daidir
	rm -rf ucli.key vc_hdrs.h DVEfiles inter.vpd
	rm -rf waves.fsdb* novas.* verdiLog
	rm -rf .fsm .restart*
	rm -rf run.log coe_run.log vcs.log
	@# XCELIUM artifacts
	rm -rf INCA_libs xcelium.d
	rm -rf irun.log xcelium.log xcelium_run.log xcelium_coe.log
	rm -rf waves.shm xrun.history
	@# Common artifacts
	rm -rf waves.vcd
	@echo "Clean complete."

# Help
help:
	@echo "================================================================================"
	@echo "Hololink Simple Testbench Makefile - Dual Simulator Support"
	@echo "================================================================================"
	@echo ""
	@echo "Testbench Architecture:"
	@echo "  This is a simple SystemVerilog testbench (non-UVM)"
	@echo "  Uses basic tasks/functions for stimulus and checking"
	@echo "  Verbosity control: +VERBOSITY=LOW/MEDIUM/HIGH"
	@echo ""
	@echo "Available Targets:"
	@echo "  all       - Check environment, compile and run simulation (default)"
	@echo "  env_check - Check simulator environment setup"
	@echo "  compile   - Compile the testbench"
	@echo "  run       - Run the simulation (RoCE mode)"
	@echo "  coe       - Run COE (Camera over Ethernet) simulation"
	@echo "  gui       - Run simulation with GUI (RoCE mode)"
	@echo "  coe-gui   - Run COE simulation with GUI"
	@echo "  verdi     - Run with Verdi (VCS only)"
	@echo "  debug     - Run with VCD dumps for debugging"
	@echo "  clean     - Clean build artifacts for all simulators"
	@echo "  help      - Show this help"
	@echo ""
	@echo "Required Variables:"
	@echo "  HOLOLINK_PATH    - Path to hololink RTL directory (REQUIRED)"
	@echo "  HOLOLINK_DEF_INCLUDE  - Path to  HOLOLINK_def.svh file directory (REQUIRED)"
	@echo ""
	@echo "Optional Variables:"
	@echo "  SIMULATOR - Simulator to use: VCS (default) or XCELIUM"
	@echo "  VERBOSITY - Debug level: LOW (default), MEDIUM, HIGH"
	@echo ""
	@echo "Simulator Selection:"
	@echo "  VCS (default):"
	@echo "    - Two-step compilation: compile then run"
	@echo "    - Waveforms: FSDB format (Verdi compatible)"
	@echo "    - GUI: DVE (make gui) or Verdi (make verdi)"
	@echo "  XCELIUM:"
	@echo "    - Single-step compile+run"
	@echo "    - Waveforms: VCD format (universal)"
	@echo "    - GUI: SimVision (make gui)"
	@echo ""
	@echo "Environment Setup:"
	@echo "  VCS:"
	@echo "    source setup_env.csh"
	@echo "  XCELIUM:"
	@echo "    setenv PATH /home/tools/cadence/XCELIUM20.09_e698/tools/bin:\$$PATH"
	@echo ""
	@echo "Usage Examples:"
	@echo "  # VCS simulation (default)"
	@echo "  make run HOLOLINK_PATH=/path/to/hololink HOLOLINK_DEF_INCLUDE=/path/to/ HOLOLINK_def.svh file directory"
	@echo ""
	@echo "  # XCELIUM simulation"
	@echo "  make SIMULATOR=XCELIUM run HOLOLINK_PATH=... HOLOLINK_DEF_INCLUDE=..."
	@echo ""
	@echo "  # High verbosity with XCELIUM"
	@echo "  make SIMULATOR=XCELIUM VERBOSITY=HIGH coe HOLOLINK_PATH=... HOLOLINK_DEF_INCLUDE=..."
	@echo ""
	@echo "  # VCS with GUI"
	@echo "  make gui HOLOLINK_PATH=... HOLOLINK_DEF_INCLUDE=..."
	@echo ""
	@echo "  # XCELIUM with GUI"
	@echo "  make SIMULATOR=XCELIUM gui HOLOLINK_PATH=... HOLOLINK_DEF_INCLUDE=..."
	@echo "================================================================================"

.PHONY: all env_check compile run coe gui coe-gui verdi debug clean help
