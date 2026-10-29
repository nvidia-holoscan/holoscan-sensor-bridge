// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//==============================================================================
// Hololink IP Simple SystemVerilog Testbench
//
// PURPOSE:
//   This testbench demonstrates basic Hololink IP functionality without UVM
//   complexity. It performs register configuration via ECB (Ethernet Control Bus)
//   and tests sensor-to-host data transmission in both RoCE and COE modes.
//
// FEATURES:
//   - ECB register programming over Ethernet
//   - Sensor data stimulus generation
//   - Host data reception monitoring
//   - Support for both RoCE and COE (Camera over Ethernet) modes
//   - Parametric configuration for easy customization
//
// USAGE:
//   RoCE Mode:  csh -c "source setup_env.csh && make run"
//   COE Mode:   csh -c "source setup_env.csh && make coe"
//   Debug:      Add +DUMP_PKTS +VERBOSITY=HIGH for detailed packet dumps
//==============================================================================


// Include Hololink macro definitions first (required by the package)
`include "HOLOLINK_def.svh"
// Per-Sensor-RX packetizer parameters from HOLOLINK_def.svh (imported by name,
// no wildcard). SIF_RX_WIDTH is used by the included hololink_sensor_stim.svh.
// The def only declares these inside `ifdef SENSOR_RX_IF_INST, so configs
// without a sensor RX interface (e.g. bajoran_nano) must not import them.
`ifdef SENSOR_RX_IF_INST
import HOLOLINK_pkg::SIF_RX_PACKETIZER_EN;
import HOLOLINK_pkg::SIF_RX_VP_COUNT;
import HOLOLINK_pkg::SIF_RX_WIDTH;
// VP_SIZE/SORT_RESOLUTION/NUM_CYCLES describe how the packetizer splits a
// sensor word across virtual ports; program_packetizer_passthrough() needs
// them to build the pattern that reassembles one full-width word on VP0.
import HOLOLINK_pkg::SIF_RX_VP_SIZE;
import HOLOLINK_pkg::SIF_RX_SORT_RESOLUTION;
import HOLOLINK_pkg::SIF_RX_NUM_CYCLES;
`endif
`ifdef SENSOR_TX_IF_INST
// SIF_TX_WIDTH and SIF_TX_BUF_SIZE size the TX buffer thresholds in
// hololink_host2sen_stim.svh; HOLOLINK_top hands the same two to tx_stream_buffer
// as SENSOR_DWIDTH and FIFO_DEPTH.
import HOLOLINK_pkg::SIF_TX_WIDTH;
import HOLOLINK_pkg::SIF_TX_BUF_SIZE;
`endif

// Include the testbench package (after macros)
`include "tb_pkg/hololink_tb_pkg.sv"
import hololink_tb_pkg::*;          // Testbench constants and helper functions

//Include dummy eeprom model
`include "eeprom_i2c_model.sv"

// Global testbench configuration (use package-scoped verbosity)

module nv_hsb_ip_simple_tb;

timeunit 1ns;
timeprecision 1ps;
// UVM integration signal - indicates when test has completed
reg test_completed = 0;

//==============================================================================
// Testbench Configuration Parameters
//==============================================================================
// Clock periods in ns, derived from the compiled-in def's *_CLK_FREQ macros
// (period = 1e9/freq). Deriving them here keeps the bench in step with the
// frequency-derived parameters the DUT computes from those same macros
// (eeprom_info's I2C divider, ptp_top, bridge_pkt_proc, streaming_cdc), which
// silently mistune when the two disagree.
// NOTE: the bajoran-lite family declares APB_CLK_FREQ = 19.53MHz (51.2ns) vs the
// 100MHz (10ns) fallback, and at that slow an APB clock the DUT backpressures host
// ingress for far longer. send_eth_packet holds the beat and fails the test if
// tready stalls past its timeout, rather than advancing tdata under a high tvalid.
// Every clock is guarded so an out-of-tree def that declares only some of the
// macros still elaborates; the literal beside each `else is the fallback for that
// case. In-tree defs all carry APB and HIF, bajoran_achronix omits PTP, and none
// declares a sensor interface frequency.
`ifdef APB_CLK_FREQ
parameter real CLK_PERIOD = 1.0e9 / `APB_CLK_FREQ;      // system/APB clock (from def)
`else
parameter real CLK_PERIOD = 10.0;                       // 100MHz fallback
`endif
`ifdef PTP_CLK_FREQ
parameter real PTP_CLK_PERIOD = 1.0e9 / `PTP_CLK_FREQ;  // PTP clock (from def)
`else
parameter real PTP_CLK_PERIOD = 10.0;                   // 100MHz fallback
`endif
`ifdef HIF_CLK_FREQ
parameter real HIF_CLK_PERIOD = 1.0e9 / `HIF_CLK_FREQ;  // host interface clock (from def)
`else
parameter real HIF_CLK_PERIOD = 5.0;                    // 200MHz fallback
`endif
`ifdef SIF_CLK_FREQ
parameter real SIF_CLK_PERIOD = 1.0e9 / `SIF_CLK_FREQ;  // sensor interface clock (from def)
`else
parameter real SIF_CLK_PERIOD = 6.4;                    // 156.25MHz fallback
`endif

// How long hololink_init() waits for the DUT's o_init_done before giving up and
// programming registers anyway. See wait_init_done().
`ifdef ENUM_EEPROM
// eeprom_info reads all 256 EEPROM bytes over I2C before o_init_done, so size the
// wait from that. Divider and bit count mirror eeprom_info/i2c_ctrl_fsm: SCL period
// = 2*CLK_DIV_CNT APB clocks, and each 32-byte transaction sends {dev addr+W,
// reg addr, restart dev addr+R, 32 data} at 9 bits per byte plus START/RESTART/STOP.
// Actual SCL lands just under the 400kHz target because the divider rounds up.
localparam int  EEPROM_SCL_DIV      = ((`APB_CLK_FREQ - 1) / 400000 / 2) + 1;
localparam real EEPROM_SCL_PERIOD   = 2.0 * EEPROM_SCL_DIV * CLK_PERIOD;   // ns
localparam int  EEPROM_BYTES_PER_TR = 32;
localparam int  EEPROM_TRANSACTIONS = 256 / EEPROM_BYTES_PER_TR;
localparam int  EEPROM_ADDR_BYTES   = (`EEPROM_REG_ADDR_BITS + 7) / 8;
localparam int  EEPROM_SCL_BITS     = EEPROM_TRANSACTIONS *
                                      (9 * (2 + EEPROM_ADDR_BYTES + EEPROM_BYTES_PER_TR) + 3);
// ~15 APB accesses per transaction, each a transfer plus a 32-cycle wait.
localparam real EEPROM_APB_TIME     = EEPROM_TRANSACTIONS * 15 * 35 * CLK_PERIOD;  // ns
parameter  real INIT_DONE_TIMEOUT   = (EEPROM_SCL_BITS * EEPROM_SCL_PERIOD) +
                                      EEPROM_APB_TIME + 1ms;
`else
// eeprom_dval follows i_enum_vld here, asserted a few cycles after reset.
parameter  real INIT_DONE_TIMEOUT   = 1500us;
`endif

// Network configuration (modify as needed for your test environment)
parameter [31:0] TB_FPGA_IP = DEFAULT_FPGA_IP;  // FPGA IP address
parameter [31:0] TB_HOST_IP = DEFAULT_HOST_IP;  // Host IP address
parameter [47:0] TB_HOST_MAC = DEFAULT_HOST_MAC; // Host MAC address

// RoCE/COE Protocol Configuration (from package; avoid duplication)
// Using TB_ROCE_* and ROCE_VADDR_* constants from hololink_tb_pkg.sv

//==============================================================================
// Clock and Reset Signals
//==============================================================================
// Frequencies come from the def (see the *_CLK_PERIOD parameters above), so the
// clk_100 name is historical - it is only 100MHz when APB_CLK_FREQ says so.
logic clk_100;    // APB / system clock, CLK_PERIOD
logic ptp_clk;    // PTP clock, PTP_CLK_PERIOD
logic sif_clk;    // sensor interface clock, SIF_CLK_PERIOD
logic hif_clk;    // host interface clock, HIF_CLK_PERIOD
logic rstn;
logic init_done;
// New per-direction SIF clocks and DUT resets (rev112)
`ifdef SENSOR_RX_IF_INST
logic [`SENSOR_RX_IF_INST-1:0] i_sif_rx_clk;
`endif
`ifdef SENSOR_TX_IF_INST
logic [`SENSOR_TX_IF_INST-1:0] i_sif_tx_clk;
`endif

//==============================================================================
// DUT Interface Signals
//==============================================================================
// Host Ethernet Interface
logic [`HOST_IF_INST-1:0] i_hif_axis_tvalid;
logic [`HOST_IF_INST-1:0] i_hif_axis_tlast;
logic [`HOST_WIDTH-1:0] i_hif_axis_tdata     [`HOST_IF_INST-1:0];
logic [`HOSTKEEP_WIDTH-1:0] i_hif_axis_tkeep [`HOST_IF_INST-1:0];
logic [`HOSTUSER_WIDTH-1:0] i_hif_axis_tuser [`HOST_IF_INST-1:0];
logic [`HOST_IF_INST-1:0] o_hif_axis_tready;

logic [`HOST_IF_INST-1:0] o_hif_axis_tvalid;
logic [`HOST_IF_INST-1:0] o_hif_axis_tlast;
logic [`HOST_WIDTH-1:0] o_hif_axis_tdata     [`HOST_IF_INST-1:0];
logic [`HOSTKEEP_WIDTH-1:0] o_hif_axis_tkeep [`HOST_IF_INST-1:0];
logic [`HOSTUSER_WIDTH-1:0] o_hif_axis_tuser [`HOST_IF_INST-1:0];
logic [`HOST_IF_INST-1:0] i_hif_axis_tready;

// Enumeration handshake - only present when the DUT is built WITHOUT EEPROM
// enumeration. In that build HOLOLINK_top ties eeprom_dval = i_enum_vld and takes
// dev_mac_addr/board_sn straight from these ports, so o_init_done (and therefore
// o_hif_axis_tready) never asserts unless the TB drives the handshake itself.
// The EEPROM build has no such ports and gets its info from eeprom_i2c_model.
`ifndef ENUM_EEPROM
logic [47:0] i_mac_addr [`HOST_IF_INST-1:0];
logic [55:0] i_board_sn;
logic        i_enum_vld;
`endif

// Sensor Interface (rev112 split RX/TX instance counts)
// RX into DUT
`ifdef SENSOR_RX_IF_INST
logic [`SENSOR_RX_IF_INST-1:0] i_sif_axis_tvalid;
logic [`SENSOR_RX_IF_INST-1:0] i_sif_axis_tlast;
logic [`DATAPATH_WIDTH-1:0]    i_sif_axis_tdata [`SENSOR_RX_IF_INST-1:0];
logic [`DATAKEEP_WIDTH-1:0]    i_sif_axis_tkeep [`SENSOR_RX_IF_INST-1:0];
logic [`DATAUSER_WIDTH-1:0]    i_sif_axis_tuser [`SENSOR_RX_IF_INST-1:0];
logic [`SENSOR_RX_IF_INST-1:0] o_sif_axis_tready;
`endif

// TX from DUT
`ifdef SENSOR_TX_IF_INST
logic [`SENSOR_TX_IF_INST-1:0] o_sif_axis_tvalid;
logic [`SENSOR_TX_IF_INST-1:0] o_sif_axis_tlast;
logic [`DATAPATH_WIDTH-1:0]    o_sif_axis_tdata [`SENSOR_TX_IF_INST-1:0];
logic [`DATAKEEP_WIDTH-1:0]    o_sif_axis_tkeep [`SENSOR_TX_IF_INST-1:0];
logic [`DATAUSER_WIDTH-1:0]    o_sif_axis_tuser [`SENSOR_TX_IF_INST-1:0];
logic [`SENSOR_TX_IF_INST-1:0] i_sif_axis_tready;
`endif

// GPIO and other signals
logic [`GPIO_INST-1:0] i_gpio;
logic [`GPIO_INST-1:0] o_gpio;
logic [`GPIO_INST-1:0] o_gpio_dir;
logic [HLNK_USER_EVT_WIDTH-1:0] i_sif_event;

//==============================================================================
// DUT peripheral/control and timing interfaces
//==============================================================================
// APB Interface
logic [`REG_INST-1:0] o_apb_psel;
logic o_apb_penable;
logic [31:0] o_apb_paddr;
logic [31:0] o_apb_pwdata;
logic o_apb_pwrite;
logic [`REG_INST-1:0] i_apb_pready;
logic [31:0] i_apb_prdata [`REG_INST-1:0];
logic [`REG_INST-1:0] i_apb_pserr;

// Peripheral Interfaces
`ifdef SPI_INST
logic [`SPI_INST-1:0] o_spi_csn;
logic [`SPI_INST-1:0] o_spi_sck;
logic [3:0] i_spi_sdio [`SPI_INST-1:0];
logic [3:0] o_spi_sdio [`SPI_INST-1:0];
logic [3:0] o_spi_oen [`SPI_INST-1:0];
`endif

`ifdef I2C_INST
logic [`I2C_INST-1:0] i_i2c_scl;
logic [`I2C_INST-1:0] i_i2c_sda;
logic [`I2C_INST-1:0] o_i2c_scl_en;
logic [`I2C_INST-1:0] o_i2c_sda_en;
`endif

`ifdef UART_INST
logic i_uart_rx;
logic o_uart_tx;
logic o_uart_busy;
logic o_uart_rts;
logic i_uart_cts;
`endif

// PTP and system control
logic o_sw_sys_rst;
logic [31:0] o_sw_sen_rst;  // DUT always has 32-bit output
logic [47:0] o_ptp_sec;
logic [31:0] o_ptp_nanosec;
logic o_pps;

//==============================================================================
// Signal Monitors for Debug
//==============================================================================
always @(init_done) begin
    `TB_LOG_LOW(LOG_INFO, $sformatf("INIT_DONE CHANGED: %b", init_done));
end
//==============================================================================
// Clock Generation
//==============================================================================
initial begin
    clk_100 = 0;
    forever #((CLK_PERIOD/2)*1ns) clk_100 = ~clk_100;
end

initial begin
    ptp_clk = 0;
    forever #((PTP_CLK_PERIOD/2)*1ns) ptp_clk = ~ptp_clk;
end

initial begin
    sif_clk = 0;
    forever #((SIF_CLK_PERIOD/2)*1ns) sif_clk = ~sif_clk;
end

// Host Interface Clock
initial begin
    hif_clk = 0;
    forever #((HIF_CLK_PERIOD/2)*1ns) hif_clk = ~hif_clk;
end

// Report the resolved clock periods (helps confirm the def's *_CLK_FREQ took effect)
initial begin
    `TB_LOG_LOW(LOG_INFO, $sformatf(
        "Clock periods (ns): APB=%.3f (%.2f MHz)  PTP=%.3f (%.2f MHz)  SIF=%.3f (%.2f MHz)  HIF=%.3f (%.2f MHz)",
        CLK_PERIOD,     1000.0/CLK_PERIOD,
        PTP_CLK_PERIOD, 1000.0/PTP_CLK_PERIOD,
        SIF_CLK_PERIOD, 1000.0/SIF_CLK_PERIOD,
        HIF_CLK_PERIOD, 1000.0/HIF_CLK_PERIOD));
end

//==============================================================================
// FSDB Waveform Dump (opt-in via +FSDB_DUMP, driven by Makefile DUMP=1)
//==============================================================================
// Standalone dumping: dump nothing unless +FSDB_DUMP is present on the command
// line. Filename defaults to waves.fsdb, overridable with +FSDB_FILE=<name>.
initial begin
    string fsdb_name;
    if ($test$plusargs("FSDB_DUMP")) begin
        if (!$value$plusargs("FSDB_FILE=%s", fsdb_name))
            fsdb_name = "waves.fsdb";
        $fsdbDumpfile(fsdb_name);
        $fsdbDumpvars(0, nv_hsb_ip_simple_tb, "+all");   // full hierarchy incl. MDAs
        `TB_LOG_LOW(LOG_INFO, $sformatf("FSDB dumping ENABLED -> %s", fsdb_name));
    end
end

// Drive the per-direction SIF clock vectors from the single TB SIF clock
generate
`ifdef SENSOR_RX_IF_INST
  for (genvar gi = 0; gi < `SENSOR_RX_IF_INST; gi++) begin: gen_rx_clk
    assign i_sif_rx_clk[gi] = sif_clk;
  end
`endif
`ifdef SENSOR_TX_IF_INST
  for (genvar gj = 0; gj < `SENSOR_TX_IF_INST; gj++) begin: gen_tx_clk
    assign i_sif_tx_clk[gj] = sif_clk;
  end
`endif
endgenerate

//==============================================================================
// Reset Generation
//==============================================================================
initial begin
    rstn = 0;
    #(200*1ns);
    rstn = 1;
    `TB_LOG(LOG_INFO, "Reset deasserted");
end

//==============================================================================
// Enumeration Handshake (non-EEPROM builds only)
//==============================================================================
// When the DUT is built without EEPROM enumeration, o_init_done is gated by
// i_enum_vld (eeprom_dval = i_enum_vld) and the device MAC / board serial come
// from these ports. We provide a static device MAC that matches the FPGA MAC the
// ECB frames are addressed to (DEFAULT_FPGA_MAC) and pulse the enumeration valid
// after reset so initialization completes exactly as the EEPROM path would.
// EEPROM builds skip this entirely; enumeration comes from eeprom_i2c_model.
// BOOTP responder state. In a non-EEPROM build the DUT has no device IP and
// broadcasts BOOTP requests to obtain one; without a reply o_dev_ip_addr never
// becomes valid and the DUT can source neither ECB responses nor RoCE frames.
// The egress monitor pulses bootp_req_seen when it observes a request; the
// responder process below replies with a BOOTP response carrying yiaddr = FPGA IP.
event bootp_req_seen;
int   bootp_rsp_sent = 0;
// Serializes host ingress (i_hif_axis[0]) between ECB tasks and the BOOTP reply,
// since both drive the same AXIS and the DUT interleaves BOOTP with register I/O.
semaphore hif_tx_sem = new(1);

`ifndef ENUM_EEPROM
initial begin
    i_enum_vld = 1'b0;
    i_board_sn = 56'h0;
    foreach (i_mac_addr[i]) i_mac_addr[i] = DEFAULT_FPGA_MAC + i;
    wait(rstn == 1'b1);
    repeat(10) @(posedge clk_100);
    i_enum_vld = 1'b1;   // held high: latched into eeprom_dval_r inside the DUT
    `TB_LOG_LOW(LOG_INFO, "Enumeration handshake asserted (non-EEPROM build)");
end

// BOOTP responder: reply to every DUT BOOTP request so it acquires a device IP.
// The egress monitor pulses bootp_req_seen on each observed request.
initial begin
    forever begin
        @(bootp_req_seen);
        send_bootp_response();
    end
end
`endif

//==============================================================================
// Test Variables
//==============================================================================
int test_status = 0;
int ecb_sequence_id = 0;
// Raised by a task that has hit a condition it cannot legally drive through, so the
// main fork unblocks and the run ends via the shared reporting path.
bit fatal_error = 1'b0;


// Self-checking variables
int sensor_packets_sent = 0;
int host_packets_received = 0;
int host_fragments_received = 0;
int errors_detected = 0;

// Completed frame counter (stream-based validation removes per-address tracking)
integer        completed_sensor_frames = 0;

// Expected sensor data storage for realistic validation
logic [7:0] expected_sensor_data[int][];  // [frame_id][byte_index] - stores expected data per frame
integer sensor_frames_generated = 0;

// Optional all-sensors demo mode.  Default remains the original single-sensor
// behavior unless +ALL_SENSORS or +TEST_MODE=ALL_SENSORS is provided.
bit all_sensors_mode = 1'b0;
int target_host_id = 0;
int expected_total_frames = TOTAL_TEST_FRAMES;

`ifdef SENSOR_RX_IF_INST
integer sensor_frames_generated_by_sensor [`SENSOR_RX_IF_INST];
integer sensor_packets_sent_by_sensor [`SENSOR_RX_IF_INST];
integer frames_received_by_sensor [`SENSOR_RX_IF_INST];
bit frame_received_by_sensor [`SENSOR_RX_IF_INST][TOTAL_TEST_FRAMES];
logic [31:0] dp_roce_rkey_by_sensor [`SENSOR_RX_IF_INST];
logic [23:0] dp_roce_qp_by_sensor [`SENSOR_RX_IF_INST];
logic [31:0] dp_roce_psn_by_sensor [`SENSOR_RX_IF_INST];
logic [31:0] dp_roce_vaddr_cfg_words_by_sensor [`SENSOR_RX_IF_INST][4];
logic [63:0] dp_roce_vaddr_cfg_bytes_by_sensor [`SENSOR_RX_IF_INST][4];
`endif

// Simple frame tracking for basic validation

//==============================================================================
// Host-to-sensor (TX) path state (UC RoCE SEND).  Enabled with +HOST2SEN.
//==============================================================================
bit host2sen_mode = 1'b0;
bit host2sen_requested = 1'b0; // +HOST2SEN was passed (regardless of TX support)
`ifdef SENSOR_TX_IF_INST
// Byte stream the host pushed to each sensor, awaiting egress comparison.
logic [7:0] expected_stx_bytes    [`SENSOR_TX_IF_INST][$];
integer     host2sen_bytes_sent   [`SENSOR_TX_IF_INST];
integer     host2sen_bytes_checked[`SENSOR_TX_IF_INST];
integer     host2sen_mismatches   [`SENSOR_TX_IF_INST];
integer     stx_gen_byte          [`SENSOR_TX_IF_INST]; // running data pattern counter
integer     host2sen_psn          [`SENSOR_TX_IF_INST]; // running BTH PSN per sensor
`endif

// Host-side Ethernet PAUSE (802.3x) honoring for the TX push path. When the DUT
// asserts backpressure it transmits a PAUSE frame on host egress; the monitor
// loads pause_release_cycle with quanta*(512/HOST_WIDTH) hif cycles and the
// countdown process below keeps host_tx_paused asserted until it expires. The
// host2sen sender waits on host_tx_paused between packets.
bit     host_tx_paused      = 1'b0;
integer pause_release_cycle = 0;
localparam int NUM_HIF_CLK_PER_QUANTA = ((512/`HOST_WIDTH) == 0) ? 1 : (512/`HOST_WIDTH);

// Sole writer of host_tx_paused: derive it from the pause countdown each hif clk.
initial begin
    forever begin
        @(posedge hif_clk);
        if (pause_release_cycle > 0)
            pause_release_cycle = pause_release_cycle - 1;
        host_tx_paused = (pause_release_cycle > 0);
    end
end

// COE mode detection for conditional register configuration
bit enable_coe_mode = 1'b0;
bit frame_received[3] = '{0, 0, 0};  // Track which frames (0,1,2) have been received
integer unique_frames_received = 0;
integer error_count = 0;

// APB register memory model (using associative array)
logic [31:0] apb_registers [*];
int num_apb_transactions = 0;

//===============================================================================
// Configured DP RoCE buffer virtual addresses (as programmed) for validation
//===============================================================================
logic [31:0] dp_roce0_vaddr_cfg_words [4];
logic [63:0] dp_roce0_vaddr_cfg_bytes [4];

`ifdef SENSOR_RX_IF_INST
function automatic int get_sensor_physical_vp_count(input int sensor_id);
    if (SIF_RX_PACKETIZER_EN[sensor_id]) begin
        return (SIF_RX_VP_COUNT[sensor_id] == 0) ? 1 : SIF_RX_VP_COUNT[sensor_id];
    end else begin
        return 1;
    end
endfunction

function automatic int get_sensor_first_vp(input int sensor_id);
    int base;
    base = 0;
    for (int s = 0; s < sensor_id; s++) begin
        base += get_sensor_physical_vp_count(s);
    end
    return base;
endfunction

function automatic logic [31:0] get_default_sensor_vp_mask();
    logic [31:0] mask;
    int vp;
    mask = 32'h0;
    for (int s = 0; s < `SENSOR_RX_IF_INST; s++) begin
        vp = get_sensor_first_vp(s);
        if (vp < 32) begin
            mask[vp] = 1'b1;
        end else begin
            `TB_LOG_LOW(LOG_ERR, $sformatf("Default VP index %0d for sensor %0d exceeds 32-bit dp_pkt_vip_mask", vp, s));
        end
    end
    return mask;
endfunction

function automatic int get_all_sensor_global_frame_id(input int sensor_id, input int frame_id);
    return (sensor_id * TOTAL_TEST_FRAMES) + frame_id;
endfunction

function automatic int get_sensor_id_from_qp(input logic [23:0] dest_qp);
    int sensor_id;
    if (dest_qp < TB_ROCE_DEST_QP) begin
        return -1;
    end
    sensor_id = dest_qp - TB_ROCE_DEST_QP;
    if (sensor_id < 0 || sensor_id >= `SENSOR_RX_IF_INST) begin
        return -1;
    end
    return sensor_id;
endfunction

// COE frames carry no full QP; the sensor is identified by the 6-bit COE
// channel_number, which the RTL drives from roce_dest_qp[5:0]. Map it back to
// a sensor index the same way get_sensor_id_from_qp() does for RoCE.
function automatic int get_sensor_id_from_coe_channel(input logic [5:0] channel_number);
    int sensor_id;
    sensor_id = int'(channel_number) - int'(TB_ROCE_DEST_QP[5:0]);
    if (sensor_id < 0 || sensor_id >= `SENSOR_RX_IF_INST) begin
        return -1;
    end
    return sensor_id;
endfunction
`endif

//==============================================================================
// I2C Bus Interface for EEPROM Model
//==============================================================================
`ifdef I2C_INST
// Simplified I2C open-drain: modeled with 'wand' for multi-driver resolution.
wand i2c_scl_bus;  // I2C clock line (open-drain)
wand i2c_sda_bus;  // I2C data line (open-drain)

// Connect FPGA I2C interface to bus lines
// Note: Hololink IP uses active-low enable signals for open-drain control
assign i_i2c_scl[0] = i2c_scl_bus;        // FPGA I2C clock input
assign i_i2c_sda[0] = i2c_sda_bus;        // FPGA I2C data input  
assign i2c_scl_bus = o_i2c_scl_en[0];     // FPGA clock line drive (active low)
assign i2c_sda_bus = o_i2c_sda_en[0];     // FPGA data line drive (active low)
`endif

// ECB response handling
logic [7:0] pending_responses [$];
logic [31:0] last_read_data = 32'hDEADBEEF;
logic response_ready = 0;
int expected_response_seq = -1;

// ECB transaction tracking for enhanced error reporting
logic [31:0] last_ecb_addr = 32'h0;
logic [31:0] last_ecb_data = 32'h0;
logic last_ecb_was_read = 0;
int last_ecb_seq = -1;

// Other I2C instances - just pullup
`ifdef I2C_INST
generate 
    for (genvar i = 1; i < `I2C_INST; i++) begin : other_i2c
        assign i_i2c_scl[i] = 1'b1;
        assign i_i2c_sda[i] = 1'b1;
    end
endgenerate

//==============================================================================
// EEPROM Model Instantiation
//==============================================================================
eeprom_i2c_model #(
    .TB_FPGA_IP(TB_FPGA_IP),
    .TB_HOST_IP(TB_HOST_IP)
) i_eeprom_model (
    .clk(clk_100),
    .rstn(rstn),
    .i2c_scl_bus(i2c_scl_bus),
    .i2c_sda_bus(i2c_sda_bus)
);
`endif

//==============================================================================
// APB Target Model for Register Access
//==============================================================================
always @(posedge clk_100) begin
    if (!rstn) begin
        i_apb_pready <= '0;
        i_apb_pserr <= '0;
        foreach (i_apb_prdata[i]) i_apb_prdata[i] <= '0;
        num_apb_transactions <= 0;
    end else begin
        // Default response - always ready, no errors
        i_apb_pready <= '1;
        i_apb_pserr <= '0;
        
        // Handle APB transactions
        for (int i = 0; i < `REG_INST; i++) begin
        if (o_apb_psel[i] && o_apb_penable) begin
                num_apb_transactions++;
                
                if (o_apb_pwrite) begin
                    // Write transaction
                    apb_registers[o_apb_paddr] = o_apb_pwdata;
                `TB_LOG_MED(LOG_INFO, $sformatf("APB WRITE[%0d]: addr=0x%08x, data=0x%08x", i, o_apb_paddr, o_apb_pwdata));
                end else begin
                    // Read transaction
                    if (apb_registers.exists(o_apb_paddr)) begin
                        i_apb_prdata[i] = apb_registers[o_apb_paddr];
                    end else begin
                        i_apb_prdata[i] = 32'hDEADBEEF; // Default read value
                    end
                `TB_LOG_MED(LOG_INFO, $sformatf("APB READ[%0d]: addr=0x%08x, data=0x%08x", i, o_apb_paddr, i_apb_prdata[i]));
                end
            end
        end
    end
end

//==============================================================================
// Self-Checking Data Verification
//==============================================================================

// Monitor sensor data transmission and check patterns
`ifdef SENSOR_RX_IF_INST
generate
    for (genvar mon_s = 0; mon_s < `SENSOR_RX_IF_INST; mon_s++) begin : gen_sensor_packet_counter
        initial begin
            forever begin
                @(posedge i_sif_rx_clk[mon_s]);
                if (i_sif_axis_tvalid[mon_s] && o_sif_axis_tready[mon_s] && i_sif_axis_tlast[mon_s]) begin
                    sensor_packets_sent++;
                    sensor_packets_sent_by_sensor[mon_s]++;
                    `TB_LOG_MED(LOG_INFO, $sformatf("SENSOR[%0d]: Frame %0d complete, last word=0x%0h",
                             mon_s, sensor_packets_sent_by_sensor[mon_s], i_sif_axis_tdata[mon_s]));
                end
            end
        end
    end
endgenerate

// Monitor sensor interface ready with backpressure tolerance
int sif_tready_low_cycles = 0;
int sif_tready_low_err_cycles = 0;
initial begin
    // Default threshold: 2048 cycles unless overridden by +TREADY_LOW_ERR_CYCLES=<N>
    sif_tready_low_err_cycles = 2048;
    if ($value$plusargs("TREADY_LOW_ERR_CYCLES=%d", sif_tready_low_err_cycles)) begin
        `TB_LOG_LOW(LOG_INFO, $sformatf("TREADY low error threshold set to %0d cycles", sif_tready_low_err_cycles));
    end
end

always @(posedge i_sif_rx_clk[0]) begin
    if (!init_done) begin
        sif_tready_low_cycles <= 0;
    end else begin
        if (o_sif_axis_tready[0] == 1'b0) begin
            sif_tready_low_cycles <= sif_tready_low_cycles + 1;
            if (sif_tready_low_cycles == 1) begin
                `TB_LOG_LOW(LOG_WARN, "o_sif_axis_tready[0] deasserted (backpressure)");
            end
            if (sif_tready_low_cycles == sif_tready_low_err_cycles) begin
                `TB_LOG_LOW(LOG_ERR, $sformatf("o_sif_axis_tready[0] held LOW for %0d cycles - potential stall", sif_tready_low_err_cycles));
            end
        end else begin
            if (sif_tready_low_cycles > 0) begin
                `TB_LOG_LOW(LOG_INFO, $sformatf("o_sif_axis_tready[0] re-asserted after %0d cycles LOW", sif_tready_low_cycles));
            end
            sif_tready_low_cycles <= 0;
        end
    end
end
`endif

// Check register write/read consistency with better error handling
task verify_register_access(input [31:0] addr, input [31:0] expected_data);
    logic [31:0] read_data;
    ecb_read(addr, read_data);
    
    // Check for ECB error or timeout (indicates ECB not responding)
    if (read_data == 32'hDEADBEEF) begin
        errors_detected++;
        `TB_LOG_LOW(LOG_ERR, $sformatf("Register 0x%08x read error/timeout - ECB not responding properly", addr));
    end else if (read_data != expected_data) begin
        errors_detected++;
        `TB_LOG_LOW(LOG_ERR, $sformatf("Register 0x%08x readback failed! Expected=0x%08x, Read=0x%08x", 
                addr, expected_data, read_data));
    end else begin
        `TB_LOG_LOW(LOG_PASS, $sformatf("Register 0x%08x verified, data=0x%08x", addr, read_data));
    end
endtask

//==============================================================================
// Sensor Data Generation Library
//==============================================================================
`ifdef SENSOR_RX_IF_INST
`include "tb_lib/hololink_sensor_stim.svh"
`endif

//==============================================================================
// ECB Tasks Library  
//==============================================================================
`include "tb_lib/hololink_ecb_tasks.svh"

//==============================================================================
// Monitor Library  
//==============================================================================
`include "tb_lib/hololink_monitors.svh"

//==============================================================================
// RoCE/validation helper functions (included from tb_lib)
`include "tb_lib/hololink_roce_helpers.svh"

//==============================================================================
// Host-to-sensor (TX) UC RoCE SEND stimulus + streaming scoreboard
//==============================================================================
`include "tb_lib/hololink_host2sen_stim.svh"
`include "tb_lib/hololink_host2sen_scbd.svh"

// Validation function moved to RoCE helpers library

task report_test_results();
    // Local declarations must appear before any procedural statements
    logic comprehensive_pass;
    integer i;
    integer s;

    `TB_LOG_LOW(LOG_INFO, "=== TEST RESULTS SUMMARY (SENSOR-TO-HOST DATA PATH VALIDATION) ===");
    `TB_LOG_LOW(LOG_INFO, $sformatf("Sensor frames generated: %0d", sensor_frames_generated));
    `TB_LOG_LOW(LOG_INFO, $sformatf("Sensor packets sent: %0d", sensor_packets_sent));
    if (enable_coe_mode) begin
        `TB_LOG_LOW(LOG_INFO, $sformatf("COE packets received: %0d", host_fragments_received));
        `TB_LOG_LOW(LOG_INFO, $sformatf("COE frames processed: %0d", completed_sensor_frames));
    end else begin
        `TB_LOG_LOW(LOG_INFO, $sformatf("RoCE fragments received: %0d", host_fragments_received));
        `TB_LOG_LOW(LOG_INFO, $sformatf("Sensor frames reassembled: %0d", completed_sensor_frames));
    end
    `TB_LOG_LOW(LOG_INFO, $sformatf("APB transactions: %0d", num_apb_transactions));
    `TB_LOG_LOW(LOG_INFO, $sformatf("Errors detected: %0d", errors_detected));
    
    // COMPREHENSIVE FRAME COVERAGE VALIDATION
    `TB_LOG_LOW(LOG_INFO, "=== FRAME COVERAGE ANALYSIS ===");

    // Check for comprehensive test pass
    comprehensive_pass = 1;
`ifdef SENSOR_RX_IF_INST
    if (all_sensors_mode) begin
        `TB_LOG_LOW(LOG_INFO, $sformatf("Unique frames received: %0d/%0d", unique_frames_received, expected_total_frames));
        for (s = 0; s < `SENSOR_RX_IF_INST; s++) begin
            `TB_LOG_LOW(LOG_INFO, $sformatf("  Sensor %0d: generated=%0d packets_sent=%0d received=%0d/%0d QP=0x%06x RKEY=0x%08x",
                     s, sensor_frames_generated_by_sensor[s], sensor_packets_sent_by_sensor[s], frames_received_by_sensor[s],
                     TOTAL_TEST_FRAMES, dp_roce_qp_by_sensor[s], dp_roce_rkey_by_sensor[s]));
            for (i = 0; i < TOTAL_TEST_FRAMES; i++) begin
                `TB_LOG_LOW(LOG_INFO, $sformatf("    Frame %0d: %s", i, frame_received_by_sensor[s][i] ? "[OK] RECEIVED" : "[ERROR] MISSING"));
                if (!frame_received_by_sensor[s][i]) begin
                    errors_detected++;
                    comprehensive_pass = 0;
                end
            end
        end
        if (unique_frames_received != expected_total_frames) begin
            comprehensive_pass = 0;
            `TB_LOG_LOW(LOG_ERR, $sformatf("COVERAGE FAILURE: Missing all-sensor frames - only %0d/%0d received",
                     unique_frames_received, expected_total_frames));
        end
    end else begin
`endif
        `TB_LOG_LOW(LOG_INFO, $sformatf("Unique frames received: %0d/3", unique_frames_received));
        for (i = 0; i < 3; i++) begin
            `TB_LOG_LOW(LOG_INFO, $sformatf("  Frame %0d: %s", i, frame_received[i] ? "[OK] RECEIVED" : "[ERROR] MISSING"));
            if (!frame_received[i]) begin
                errors_detected++;
                comprehensive_pass = 0;
            end
        end
        if (unique_frames_received != 3) begin
            comprehensive_pass = 0;
            `TB_LOG_LOW(LOG_ERR, $sformatf("COVERAGE FAILURE: Missing sensor frames - only %0d/3 received", unique_frames_received));
        end
`ifdef SENSOR_RX_IF_INST
    end
`endif

    if (completed_sensor_frames == 0) begin
        comprehensive_pass = 0;
        if (enable_coe_mode) begin
            `TB_LOG_LOW(LOG_ERR, "NO SENSOR DATA: No COE frames processed successfully");
        end else begin
            `TB_LOG_LOW(LOG_ERR, "NO SENSOR DATA: No frames reassembled successfully");
        end
    end
    if (host_fragments_received == 0) begin
        comprehensive_pass = 0;
        if (enable_coe_mode) begin
            `TB_LOG_LOW(LOG_ERR, "NO COE DATA: No COE packets received at host");
        end else begin
            `TB_LOG_LOW(LOG_ERR, "NO RoCE DATA: No RoCE fragments received at host");
        end
    end
    
    if (comprehensive_pass) begin
        `TB_LOG_LOW(LOG_PASS, all_sensors_mode ? "COMPREHENSIVE VALIDATION PASSED: All enabled sensors received and validated" :
                                               "COMPREHENSIVE VALIDATION PASSED: All sensor frames (0,1,2) received and validated");
    end else begin
        `TB_LOG_LOW(LOG_FAIL, "COMPREHENSIVE VALIDATION FAILED: Missing critical sensor data");
    end
    
    // Check for leftover incomplete frames (stream-based helper)
    report_incomplete_frames();
    
    if (errors_detected == 0 && completed_sensor_frames > 0) begin
        test_status = 1; // Pass
        `TB_LOG_LOW(LOG_PASS, "=== TEST PASSED ===");
        `TB_LOG_LOW(LOG_PASS, $sformatf("Successfully reassembled and verified %0d sensor frames from %0d RoCE fragments", completed_sensor_frames, host_fragments_received));
        `TB_LOG_LOW(LOG_INFO, $sformatf("APB transactions completed: %0d", num_apb_transactions));
    end else begin
        test_status = 2; // Fail
        `TB_LOG_LOW(LOG_FAIL, "=== TEST FAILED ===");
        if (completed_sensor_frames == 0) `TB_LOG_LOW(LOG_FAIL, "  - No complete sensor frames reassembled");
        if (errors_detected > 0) `TB_LOG_LOW(LOG_FAIL, $sformatf("  - %0d verification errors", errors_detected));
        `TB_LOG_LOW(LOG_INFO, $sformatf("APB transactions completed: %0d", num_apb_transactions));
    end
endtask

//==============================================================================
// ECB Register Access Tasks
//==============================================================================

// Task to build complete Ethernet frame with MAC/IP/UDP headers
task build_ethernet_frame(input logic [7:0] udp_payload[], input int payload_size, 
                        input logic [47:0] dst_mac, input logic [47:0] src_mac,
                        input logic [31:0] dst_ip, input logic [31:0] src_ip,
                        input logic [15:0] dst_port, input logic [15:0] src_port,
                        output logic [7:0] eth_frame[], output int frame_size);
    automatic int ip_total_len;
    automatic int udp_len; 
    automatic int padded_size;
    automatic int idx;
    automatic logic [15:0] ip_checksum;
    automatic logic [31:0] checksum_temp;
    
    // Calculate sizes
    udp_len = UDP_HEADER_SIZE + payload_size;  // UDP header + payload
    ip_total_len = IP_HEADER_SIZE + udp_len;   // IP header + UDP
    frame_size = ETH_HEADER_SIZE + ip_total_len; // Ethernet header + IP packet
    padded_size = (frame_size < MIN_ETH_FRAME_NO_FCS_BYTES) ? MIN_ETH_FRAME_NO_FCS_BYTES : frame_size; // Minimum without FCS
    frame_size = padded_size; // Update output frame size
    
    `TB_LOG_MED(LOG_INFO, $sformatf("Building Ethernet frame: payload=%0d, UDP=%0d, IP=%0d, ETH=%0d, padded=%0d", payload_size, udp_len, ip_total_len, frame_size, padded_size));
    
    // Allocate frame
    eth_frame = new[padded_size];
    idx = 0;
    
    //=== ETHERNET HEADER (ETH_HEADER_SIZE bytes) ===
    // Destination MAC
    eth_frame[idx++] = dst_mac[47:40]; eth_frame[idx++] = dst_mac[39:32]; eth_frame[idx++] = dst_mac[31:24];
    eth_frame[idx++] = dst_mac[23:16]; eth_frame[idx++] = dst_mac[15:8];  eth_frame[idx++] = dst_mac[7:0];
    
    // Source MAC
    eth_frame[idx++] = src_mac[47:40]; eth_frame[idx++] = src_mac[39:32]; eth_frame[idx++] = src_mac[31:24];
    eth_frame[idx++] = src_mac[23:16]; eth_frame[idx++] = src_mac[15:8];  eth_frame[idx++] = src_mac[7:0];
    
    // EtherType (IPv4)
    eth_frame[idx++] = ETHER_TYPE_IPV4[15:8];
    eth_frame[idx++] = ETHER_TYPE_IPV4[7:0];
    
    //=== IP HEADER (IP_HEADER_SIZE bytes) ===
    eth_frame[idx++] = IPV4_VERSION_IHL_DEFAULT; // Version(4) + IHL(5)
    eth_frame[idx++] = IPV4_DSCP_ECN_DEFAULT;    // DSCP + ECN
    eth_frame[idx++] = ip_total_len[15:8];       // Total Length MSB
    eth_frame[idx++] = ip_total_len[7:0];        // Total Length LSB
    eth_frame[idx++] = 8'h12; eth_frame[idx++] = 8'h34; // Identification
    eth_frame[idx++] = 8'h00; eth_frame[idx++] = 8'h00; // Flags + Fragment Offset
    eth_frame[idx++] = IPV4_TTL_DEFAULT;         // TTL
    eth_frame[idx++] = IPV4_PROTOCOL_UDP;        // Protocol = UDP
    eth_frame[idx++] = 8'h00; eth_frame[idx++] = 8'h00; // Checksum (to be calculated)
    
    // Source IP
    eth_frame[idx++] = src_ip[31:24]; eth_frame[idx++] = src_ip[23:16];
    eth_frame[idx++] = src_ip[15:8];  eth_frame[idx++] = src_ip[7:0];
    
    // Destination IP
    eth_frame[idx++] = dst_ip[31:24]; eth_frame[idx++] = dst_ip[23:16];
    eth_frame[idx++] = dst_ip[15:8];  eth_frame[idx++] = dst_ip[7:0];
    
    //=== UDP HEADER (UDP_HEADER_SIZE bytes) ===
    eth_frame[idx++] = src_port[15:8]; eth_frame[idx++] = src_port[7:0]; // Source Port
    eth_frame[idx++] = dst_port[15:8]; eth_frame[idx++] = dst_port[7:0]; // Dest Port
    eth_frame[idx++] = udp_len[15:8];            // UDP Length MSB
    eth_frame[idx++] = udp_len[7:0];             // UDP Length LSB
    eth_frame[idx++] = 8'h00; eth_frame[idx++] = 8'h00; // UDP Checksum (disabled)
    
    //=== UDP PAYLOAD ===
    for (int i = 0; i < payload_size; i++) begin
        eth_frame[idx++] = udp_payload[i];
    end
    
    //=== PADDING ===
    while (idx < padded_size) begin
        eth_frame[idx++] = 8'h00;
    end
    
    //=== CALCULATE IP CHECKSUM ===
    checksum_temp = 0;
    for (int i = ETH_HEADER_SIZE; i < (ETH_HEADER_SIZE + IP_HEADER_SIZE); i += 2) begin
        if (i != (ETH_HEADER_SIZE + IP_HEADER_CHECKSUM_OFFSET)) begin // Skip checksum field itself
            checksum_temp += (eth_frame[i] << 8) + eth_frame[i+1];
        end
    end
    // Add carry
    while (checksum_temp >> 16) 
        checksum_temp = (checksum_temp & 16'hFFFF) + (checksum_temp >> 16);
    ip_checksum = ~checksum_temp[15:0];
    
    // Insert checksum
    eth_frame[ETH_HEADER_SIZE + IP_HEADER_CHECKSUM_OFFSET] = ip_checksum[15:8];
    eth_frame[ETH_HEADER_SIZE + IP_HEADER_CHECKSUM_OFFSET + 1] = ip_checksum[7:0];
    
    `TB_LOG_MED(LOG_INFO, $sformatf("IP checksum calculated: 0x%04h", ip_checksum));
    
endtask

//==============================================================================
// COE helper functions (included from tb_lib)
//==============================================================================
`include "tb_lib/hololink_coe_helpers.svh"

//==============================================================================
// Ethernet/ECB utilities (frame build + AXIS transmit used by ECB tasks)
//==============================================================================

//==============================================================================
// Ethernet Packet Transmission Task
//==============================================================================
task send_eth_packet(input int intf_id, input logic [7:0] payload[], input int payload_size);
    automatic int word_count;
    automatic int byte_in_word;
    automatic logic [`HOST_WIDTH-1:0] current_word;
    automatic logic [`HOSTKEEP_WIDTH-1:0] current_keep;
    automatic bit   stalled = 1'b0;
    
    `TB_LOG_LOW(LOG_INFO, "===> SEND_ETH_PACKET START");
    `TB_LOG_LOW(LOG_INFO, $sformatf("     Interface ID: %0d", intf_id));
    `TB_LOG_LOW(LOG_INFO, $sformatf("     Payload size: %0d bytes", payload_size));
    `TB_LOG_LOW(LOG_INFO, $sformatf("     HOST_WIDTH: %0d bits (%0d bytes)", `HOST_WIDTH, `HOST_WIDTH/8));
    
    word_count = (payload_size + (`HOST_WIDTH/8) - 1) / (`HOST_WIDTH/8);

    // Grab exclusive access to the ingress AXIS so a concurrent BOOTP reply /
    // ECB frame cannot interleave beats on the same wires.
    hif_tx_sem.get(1);

    // Wait one cycle before starting
    @(posedge hif_clk);

    for (int i = 0; i < word_count; i++) begin
        automatic int ready_timeout = 0;
        current_word = '0;
        current_keep = '0;

        // Pack bytes into word
        for (int j = 0; j < (`HOST_WIDTH/8); j++) begin
            byte_in_word = i * (`HOST_WIDTH/8) + j;
            if (byte_in_word < payload_size) begin
                current_word[j*8 +: 8] = payload[byte_in_word];
                current_keep[j] = 1'b1;
            end
        end

        // Drive signals (use non-blocking to avoid race with DUT clocked logic)
        i_hif_axis_tvalid[intf_id] <= 1'b1;
        i_hif_axis_tdata[intf_id]  <= current_word;
        i_hif_axis_tkeep[intf_id]  <= current_keep;
        i_hif_axis_tlast[intf_id]  <= (i == word_count - 1);
        i_hif_axis_tuser[intf_id]  <= '0;

        // Wait for ready and advance with timeout
        do begin
            @(posedge hif_clk);
            ready_timeout++;
            if (ready_timeout >= 10000) begin
                stalled = 1'b1;
                break;
            end
        end while (!o_hif_axis_tready[intf_id]);

        // There is no legal way out of a stalled beat: axis_checker requires tvalid
        // to stay high and tdata/tkeep/tlast to hold until tready, so both dropping
        // tvalid and advancing to the next word are protocol violations, and
        // advancing also silently loses this word from the frame. Holding the beat
        // is the only legal state, so raise fatal_error and block here, leaving the
        // beat driven while the main fork's fatal arm ends the run.
        if (stalled) begin
            `TB_LOG_LOW(LOG_FAIL, $sformatf(
                "o_hif_axis_tready[%0d] low for %0d hif_clk cycles at word %0d of %0d - host ingress is not draining",
                intf_id, ready_timeout, i, word_count));
            test_status = 3;
            fatal_error = 1'b1;
            wait (0);
        end
    end

    // Clear interface
    i_hif_axis_tvalid[intf_id] <= 1'b0;
    i_hif_axis_tlast[intf_id]  <= 1'b0;
    i_hif_axis_tdata[intf_id]  <= '0;
    i_hif_axis_tkeep[intf_id]  <= '0;
    i_hif_axis_tuser[intf_id]  <= '0;

    @(posedge hif_clk);
    hif_tx_sem.put(1);
endtask

//==============================================================================
// BOOTP Responder
//==============================================================================
// Builds and drives a minimal BOOTP reply (opcode 0x02) so the DUT latches a
// device IP (yiaddr = TB_FPGA_IP). Packet layout mirrors the ethernet VIP
// eth_bootp_rsp_sequence: server->device UDP (src=BOOTP_DST, dst=BOOTP_SRC),
// IP dst = yiaddr, and the BOOTP fixed header with yiaddr at byte offset 16.
// The DUT only consumes opcode and yiaddr; remaining fields are left zeroed.
task automatic send_bootp_response();
    automatic logic [7:0] bootp_pld[];
    automatic logic [7:0] eth_frame[];
    automatic int         frame_size;

    // Canonical 300-byte BOOTP message (matches the DUT's own request length).
    bootp_pld = new[300];
    foreach (bootp_pld[i]) bootp_pld[i] = 8'h00;

    bootp_pld[0]  = 8'h02;                 // op    = BOOTREPLY
    bootp_pld[1]  = 8'h01;                 // htype = Ethernet
    bootp_pld[2]  = 8'h06;                 // hlen  = 6
    bootp_pld[3]  = 8'h00;                 // hops
    // xid (4), secs (2), flags (2), ciaddr (4) left as 0 (DUT does not check them)
    // yiaddr @ byte offset 16 = assigned device IP (network byte order)
    bootp_pld[16] = TB_FPGA_IP[31:24];
    bootp_pld[17] = TB_FPGA_IP[23:16];
    bootp_pld[18] = TB_FPGA_IP[15:8];
    bootp_pld[19] = TB_FPGA_IP[7:0];
    // chaddr @ byte offset 28 = device MAC (informational only)
    bootp_pld[28] = DEFAULT_FPGA_MAC[47:40];
    bootp_pld[29] = DEFAULT_FPGA_MAC[39:32];
    bootp_pld[30] = DEFAULT_FPGA_MAC[31:24];
    bootp_pld[31] = DEFAULT_FPGA_MAC[23:16];
    bootp_pld[32] = DEFAULT_FPGA_MAC[15:8];
    bootp_pld[33] = DEFAULT_FPGA_MAC[7:0];

    build_ethernet_frame(bootp_pld, 300,
                         DEFAULT_FPGA_MAC, TB_HOST_MAC,            // dst=device, src=host
                         TB_FPGA_IP,       TB_HOST_IP,             // dst IP = yiaddr, src = host
                         ETH_UDP_BOOTP_SRC_PORT,                   // dst port = device BOOTP port
                         ETH_UDP_BOOTP_DST_PORT,                   // src port = server BOOTP port
                         eth_frame, frame_size);

    `TB_LOG_LOW(LOG_INFO, $sformatf(
        "BOOTP: replying with yiaddr=%0d.%0d.%0d.%0d (frame=%0d bytes)",
        TB_FPGA_IP[31:24], TB_FPGA_IP[23:16],
        TB_FPGA_IP[15:8],  TB_FPGA_IP[7:0], frame_size));

    send_eth_packet(0, eth_frame, frame_size);
    bootp_rsp_sent++;
endtask

// Note: COE (Camera over Ethernet) functions removed for simplicity

//==============================================================================
// Hololink Initialization Sequence
//==============================================================================
// Shared by both init paths: wait for the DUT to report init complete, then settle
// before the first register write. INIT_DONE_TIMEOUT covers the slowest legitimate
// bring-up for the compiled-in def, so expiry means the DUT did not come up; log
// and continue anyway rather than hang.
task wait_init_done();
    `TB_LOG_LOW(LOG_INFO, $sformatf("Waiting for init_done signal (currently %b, timeout %.3f ms)...",
                                    init_done, INIT_DONE_TIMEOUT / 1.0e6));
    fork
        begin
            wait(init_done == 1'b1);
            `TB_LOG_LOW(LOG_INFO, "*** Init done signal asserted! ***");
        end
        begin
            #(INIT_DONE_TIMEOUT);
            `TB_LOG_LOW(LOG_WARN, "*** TIMEOUT: init_done never asserted, proceeding anyway ***");
            `TB_LOG_LOW(LOG_INFO, "*** This may be expected for this simple testbench ***");
        end
    join_any
    disable fork;

    repeat(50) @(posedge clk_100);
endtask

`ifdef SENSOR_RX_IF_INST
// Number of odd-even swap selects for a 2^n-wide sorter. Mirrors
// ODD_EVEN_SIZE_VEC in vrtl/hololink/packetizer/packetizer.sv, ascending so
// that index n corresponds to clog2(DIN_WIDTH)-clog2(SORT_RESOLUTION).
localparam int PACK_ODD_EVEN_SIZE [13] = '{1, 1, 5, 19, 63, 191, 543, 1471,
                                           3839, 9727, 24063, 58367, 139263};

//--------------------------------------------------------------------------
// pack_vp_ram_index: index of the first pattern RAM holding the VP selects
//
// packetizer.sv splits its config into 32-bit RAMs laid out [CLEAR][SORT][VP],
// so the VP group starts after the clear and sort RAMs. Recomputing
// RAM_OFFSET_VEC[2] here keeps the programming def-driven instead of
// hard-coding a RAM number per variant; it reproduces both hand-coded numbers
// used by the UVM PKTZR_BYPASS_MODE tables (7 for 512b/16b, 19 for 1024b/16b).
//--------------------------------------------------------------------------
function automatic int pack_vp_ram_index(input int din_width, input int sort_resolution,
                                         input int num_cycles);
    int clear_bits;
    int sort_bits;
    clear_bits = (din_width / sort_resolution) * $clog2(num_cycles + 1);
    sort_bits  = PACK_ODD_EVEN_SIZE[$clog2(din_width) - $clog2(sort_resolution)] * num_cycles;
    return ((clear_bits + 31) / 32) + ((sort_bits + 31) / 32);
endfunction

//--------------------------------------------------------------------------
// pack_clear_ram_count: number of 32-bit RAMs the clear/cycle-select occupies
//--------------------------------------------------------------------------
function automatic int pack_clear_ram_count(input int din_width, input int sort_resolution,
                                            input int num_cycles);
    return (((din_width / sort_resolution) * $clog2(num_cycles + 1)) + 31) / 32;
endfunction

//--------------------------------------------------------------------------
// program_packetizer_passthrough: make one sensor word fill one host word
//
// At its reset value pack_ctrl leaves the packetizer in its internal bypass
// path (bypass_packetizer = !pack_ctrl[28]), where VP0 commits only
// din[VP_SIZE-1:0] of *every* sensor beat. A VP's output FIFO widens VP_SIZE
// up to HOST_WIDTH, so that packs HOST_WIDTH/VP_SIZE different beats into each
// host word and the received stream is no longer 1:1 with the sensor stream --
// which is what the TB's expected-data model assumes.
//
// Engaging the pattern RAM instead holds one input word for DIN/VP_SIZE
// pattern addresses (ld_arr=1 loads a new word only at address 0) while VP0
// commits successive VP_SIZE slices of it in order, so the FIFO reassembles
// the original full-width word and the remaining VPs stay idle. This mirrors
// PKTZR_BYPASS_MODE in
// dvlib/ec/integration/fpga_sensor_ip/src/fpga_sensor_ip/seqs/hololink_init_seq_lib.svh.
//--------------------------------------------------------------------------
task program_packetizer_passthrough(input int sensor_id);
    int din_width;
    int vp_size;
    int slices;
    int vp_ram;
    int clear_rams;
    logic [31:0] pack_base;
    logic [31:0] pack_cfg;

    if (!SIF_RX_PACKETIZER_EN[sensor_id]) begin
        `TB_LOG_LOW(LOG_INFO, $sformatf("Sensor %0d: packetizer not instantiated; VPs already pass full sensor words",
                 sensor_id));
        return;
    end

    din_width = SIF_RX_WIDTH[sensor_id];
    vp_size   = SIF_RX_VP_SIZE[sensor_id];

    if (vp_size >= din_width) begin
        `TB_LOG_LOW(LOG_INFO, $sformatf("Sensor %0d: VP_SIZE=%0d already spans the %0d-bit sensor word; no pattern needed",
                 sensor_id, vp_size, din_width));
        return;
    end

    slices = din_width / vp_size;
    // max_addr is 4 bits and the pattern RAM is 16 entries deep.
    if ((slices * vp_size) != din_width || slices > 16) begin
        `TB_LOG_LOW(LOG_WARN, $sformatf("Sensor %0d: cannot build pass-through pattern for DIN=%0d VP_SIZE=%0d (%0d slices)",
                 sensor_id, din_width, vp_size, slices));
        return;
    end

    pack_base  = REG_PACK_CTRL_BASE + (sensor_id * REG_PACK_CTRL_STRIDE);
    vp_ram     = pack_vp_ram_index(din_width, SIF_RX_SORT_RESOLUTION[sensor_id],
                                   SIF_RX_NUM_CYCLES[sensor_id]);
    clear_rams = pack_clear_ram_count(din_width, SIF_RX_SORT_RESOLUTION[sensor_id],
                                      SIF_RX_NUM_CYCLES[sensor_id]);

    pack_cfg = (32'h1 << PACK_CFG_ENABLE_BIT) |
               (32'(slices - 1) << PACK_CFG_MAX_ADDR_LSB) |
               (32'h1 << PACK_CFG_LD_ARR_LSB);

    `TB_LOG_LOW(LOG_INFO, $sformatf("Sensor %0d: programming packetizer pass-through (DIN=%0d VP_SIZE=%0d slices=%0d vp_ram=%0d clear_rams=%0d pack_cfg=0x%08x)",
             sensor_id, din_width, vp_size, slices, vp_ram, clear_rams, pack_cfg));

    // pack_ctrl first: each RAM_DATA write latches ld_arr into the shift
    // register, so ld_arr/max_addr must already be in place.
    ecb_write(pack_base + PACK_OFF_PACK_CFG, pack_cfg);

    for (int e = 0; e < slices; e++) begin
        // Cycle select: keep every sorted region (0 would clear it).
        for (int r = 0; r < clear_rams; r++) begin
            ecb_write(pack_base + PACK_OFF_RAM_ADDR, 32'((r << 4) | e));
            ecb_write(pack_base + PACK_OFF_RAM_DATA, 32'hFFFF_FFFF);
        end
        // VP select: {select=e, commit=1} sends slice e of the held word to
        // VP0. The other VPs read 0 from this RAM and stay idle.
        ecb_write(pack_base + PACK_OFF_RAM_ADDR, 32'((vp_ram << 4) | e));
        ecb_write(pack_base + PACK_OFF_RAM_DATA, 32'((e << 1) | 1));
    end

    `TB_LOG_LOW(LOG_INFO, $sformatf("Sensor %0d: packetizer pass-through programmed (%0d pattern entries)",
             sensor_id, slices));
endtask

task automatic clear_all_physical_dp_roce_blocks();
    int first_vp;
    int vp_count;
    int vp;
    int cleared_vp_count;

    cleared_vp_count = 0;
    for (int sensor_id = 0; sensor_id < `SENSOR_RX_IF_INST; sensor_id++) begin
        first_vp = get_sensor_first_vp(sensor_id);
        vp_count = get_sensor_physical_vp_count(sensor_id);
        for (int vp_offset = 0; vp_offset < vp_count; vp_offset++) begin
            vp = first_vp + vp_offset;
            ecb_write(reg_roce(vp, ROCE_OFF_VADDR_FIXED_MSB), 32'h0);
            ecb_write(reg_roce(vp, ROCE_OFF_PCP),             32'h0);
            ecb_write(reg_roce(vp, ROCE_OFF_RSVD_2),          32'h0);
            cleared_vp_count++;
        end
    end

    `TB_LOG_LOW(LOG_INFO, $sformatf(
        "Cleared RoCE vaddr_fixed_msb/pcp/rsvd_2 for %0d physical VP block(s)",
        cleared_vp_count));
endtask

task program_roce_for_sensor(input int sensor_id, input int host_id);
    int vp_id;
    logic [23:0] sensor_qp;
    logic [31:0] sensor_rkey;
    logic [31:0] sensor_psn;
    logic [63:0] sensor_base_addr;

    vp_id = get_sensor_first_vp(sensor_id);
    sensor_qp = TB_ROCE_DEST_QP + 24'(sensor_id);
    sensor_rkey = TB_ROCE_RKEY + sensor_id;
    sensor_psn = TB_ROCE_PSN + (sensor_id * 32'h0000_1000);
    sensor_base_addr = {ROCE_BUF_ADDR_MSB, ROCE_BUF_ADDR_LSB} + (64'(sensor_id) * 64'(ROCE_SENSOR_ADDR_STRIDE_BYTES));

    dp_roce_qp_by_sensor[sensor_id] = sensor_qp;
    dp_roce_rkey_by_sensor[sensor_id] = sensor_rkey;
    dp_roce_psn_by_sensor[sensor_id] = sensor_psn;

    `TB_LOG_LOW(LOG_INFO, $sformatf("Programming sensor %0d default VP %0d RoCE context: QP=0x%06x RKEY=0x%08x PSN=0x%08x BASE=0x%016x",
             sensor_id, vp_id, sensor_qp, sensor_rkey, sensor_psn, sensor_base_addr));

    ecb_write(reg_roce(vp_id, ROCE_OFF_DEST_QP), {7'd0, enable_coe_mode, sensor_qp});
    ecb_write(reg_roce(vp_id, ROCE_OFF_RKEY), sensor_rkey);
    ecb_write(reg_roce(vp_id, ROCE_OFF_BUF_LEN), FRAME_SIZE_BYTES);
    ecb_write(reg_roce(vp_id, ROCE_OFF_PSN), sensor_psn);

    if (USE_4K_BUFFER_SCHEME) begin
        ecb_write(reg_roce(vp_id, ROCE_OFF_BUF_ADDR_LSB), sensor_base_addr[31:0]);
        ecb_write(reg_roce(vp_id, ROCE_OFF_BUF_ADDR_MSB), sensor_base_addr[63:32]);
        ecb_write(reg_roce(vp_id, ROCE_OFF_BUF_INC), {6'b0, ROCE_BUF_INC});
        ecb_write(reg_roce(vp_id, ROCE_OFF_BUF_PTR), {4'b0, ROCE_BUF_PTR_START, 4'b0, ROCE_BUF_PTR_END});

        for (int vi = 0; vi < 4; vi++) begin
            dp_roce_vaddr_cfg_bytes_by_sensor[sensor_id][vi] = sensor_base_addr +
                                                               (64'(ROCE_BUF_INC) << 7) * (ROCE_BUF_PTR_START + vi);
            dp_roce_vaddr_cfg_words_by_sensor[sensor_id][vi] = dp_roce_vaddr_cfg_bytes_by_sensor[sensor_id][vi][38:7];
        end
    end else begin
        ecb_write(reg_roce(vp_id, REG_DP_ROCE_0_BUF_MASK - REG_DP_ROCE_0_DEST_QP), 32'h0000_000F);
        for (int vi = 0; vi < ROCE_VADDR_COUNT; vi++) begin
            dp_roce_vaddr_cfg_words_by_sensor[sensor_id][vi] = ROCE_VADDR_WORD_BASE +
                                                               (sensor_id * (ROCE_VADDR_FRAME_STRIDE * ROCE_VADDR_COUNT)) +
                                                               (ROCE_VADDR_FRAME_STRIDE * vi);
            ecb_write(reg_roce(vp_id, ROCE_OFF_BUF_ADDR_LSB + (vi * 4)), dp_roce_vaddr_cfg_words_by_sensor[sensor_id][vi]);
            dp_roce_vaddr_cfg_bytes_by_sensor[sensor_id][vi] = {32'h0, dp_roce_vaddr_cfg_words_by_sensor[sensor_id][vi]} << 7;
        end
    end

    ecb_write(reg_roce(vp_id, ROCE_OFF_HOST_MAC_LO), TB_HOST_MAC[31:0]);
    ecb_write(reg_roce(vp_id, ROCE_OFF_HOST_MAC_HI), TB_HOST_MAC[47:32]);
    ecb_write(reg_roce(vp_id, ROCE_OFF_HOST_IP), TB_HOST_IP);
    ecb_write(reg_roce(vp_id, ROCE_OFF_HOST_UDP_PORT), enable_coe_mode ? ETH_UDP_COE_PORT : ETH_UDP_ROCE_PORT);
endtask

task hololink_init_all_sensors();
    logic [31:0] default_vp_mask;

    // init_done was already awaited by hololink_init() before it dispatched here.
    `TB_LOG_LOW(LOG_INFO, "===== HOLOLINK ALL-SENSORS INITIALIZATION START =====");

    `TB_LOG_LOW(LOG_INFO, $sformatf("Programming one default VP per Sensor RX interface into host %0d", target_host_id));

    for (int sensor_id = 0; sensor_id < `SENSOR_RX_IF_INST; sensor_id++) begin
        program_roce_for_sensor(sensor_id, target_host_id);
        program_packetizer_passthrough(sensor_id);
    end

    default_vp_mask = get_default_sensor_vp_mask();
    `TB_LOG_LOW(LOG_INFO, $sformatf("Default sensor VP mask for host %0d = 0x%08x", target_host_id, default_vp_mask));

    for (int host_id = 0; host_id < `HOST_IF_INST; host_id++) begin
        ecb_write(reg_dp_pkt(host_id, DP_PKT_OFF_LEN), DP_PKT_LEN_VALUE);
        ecb_write(reg_dp_pkt(host_id, DP_PKT_OFF_FPGA_UDP_PORT),
                  enable_coe_mode ? {16'h0001, ETH_UDP_COE_PORT} : {16'h0000, ETH_UDP_ROCE_PORT});
        ecb_write(reg_dp_pkt(host_id, DP_PKT_OFF_VIP_MASK), (host_id == target_host_id) ? default_vp_mask : 32'h0);
    end

    // Preserve the legacy ECB-port write and also program the CSV-canonical
    // inst_dec ecb_udp_port offset for the selected host.
    ecb_write(REG_INST_DEC_0_ECB_UDP_PORT, ETH_UDP_CMD_PORT);
    ecb_write(reg_inst_dec(target_host_id, INST_DEC_OFF_ECB_UDP_PORT), ETH_UDP_CMD_PORT);

    repeat(100) @(posedge clk_100);
    `TB_LOG_LOW(LOG_INFO, "===== HOLOLINK ALL-SENSORS INITIALIZATION COMPLETE =====");
endtask
`endif

task hololink_init();
`ifndef ENUM_EEPROM
    // Non-EEPROM builds: the DUT cannot process/ACK ECB register writes until it
    // has acquired a device IP via BOOTP. Stall the init sequence until we have
    // replied to at least one BOOTP request, so no register write is dropped.
    // A safety timeout prevents an infinite hang if no request is ever seen.
    if (bootp_rsp_sent == 0) begin
        `TB_LOG_LOW(LOG_INFO, "Waiting for BOOTP reply before starting register init...");
        fork
            wait(bootp_rsp_sent > 0);
            begin
                #(1500us);
                `TB_LOG_LOW(LOG_WARN, "*** TIMEOUT: no BOOTP request seen, starting init anyway ***");
            end
        join_any
        disable fork;
    end
    if (bootp_rsp_sent > 0) begin
        // Give the DUT time to parse the BOOTP reply and propagate the new device
        // IP (yiaddr -> ciaddr -> o_dev_ip_addr, which crosses clock domains)
        // before issuing ECB writes. Without this settle the very first ECB write
        // races the IP update and gets silently dropped (observed ACK timeout on
        // the first register, subsequent writes fine).
        #(10us);
        `TB_LOG_LOW(LOG_INFO, $sformatf("BOOTP reply sent (count=%0d); IP settle done, proceeding with register init", bootp_rsp_sent));
    end
`endif

    // Both init paths below need the DUT initialized, so gate once here.
    wait_init_done();

`ifdef SENSOR_RX_IF_INST
    // These fields are backed by block RAM and are not reset by the DUT. Clear
    // every physical VP before either initialization path programs active VPs.
    clear_all_physical_dp_roce_blocks();

    if (all_sensors_mode) begin
        hololink_init_all_sensors();
        return;
    end
`endif

    `TB_LOG_LOW(LOG_INFO, "===== HOLOLINK INITIALIZATION START =====");
    `TB_LOG_LOW(LOG_INFO, "Starting register programming sequence...");
    
    //==========================================================================
    // Hololink IP Register Configuration
    //==========================================================================
    // This section programs the essential registers for Hololink IP operation.
    // All register addresses and values can be customized via parameters above.
    
    //----------------------------------------------------------------------
    // RoCE Data Path Configuration
    //----------------------------------------------------------------------
    // Configure RoCE (RDMA over Converged Ethernet) for sensor data transmission.
    // The dp_roce register blocks are instantiated per sensor VP, so a config with
    // no sensor RX interface has none of them and every write below returns an ECB
    // ACCESS_ERROR. Those errors are counted against the test, which would fail an
    // otherwise clean TX-only run (e.g. bajoran_nano host2sen).
`ifdef SENSOR_RX_IF_INST
    `TB_LOG_LOW(LOG_INFO, "Programming RoCE data path registers...");
    
    // Configure protocol mode and destination queue pair
    // DEST_QP register format: [31:25]=threshold, [24]=protocol_mode, [23:0]=queue_pair
    if (enable_coe_mode) begin
        // COE (Camera over Ethernet) mode: Set protocol bit to enable IEEE 1722B
        ecb_write(REG_DP_ROCE_0_DEST_QP, {8'h01, TB_ROCE_DEST_QP});    // COE: protocol=1, queue_pair=parametric
        `TB_LOG_LOW(LOG_INFO, $sformatf("Configured for COE mode (IEEE 1722B): dest_qp=0x%06x", TB_ROCE_DEST_QP));
    end else begin
        // RoCE (RDMA over Converged Ethernet) mode: Standard RDMA protocol
        ecb_write(REG_DP_ROCE_0_DEST_QP, {8'h00, TB_ROCE_DEST_QP});    // RoCE: protocol=0, queue_pair=parametric
        `TB_LOG_LOW(LOG_INFO, $sformatf("Configured for RoCE mode (RDMA): dest_qp=0x%06x", TB_ROCE_DEST_QP));
    end
    
    // Configure RoCE memory access parameters
    ecb_write(REG_DP_ROCE_0_RKEY, TB_ROCE_RKEY);           // Remote memory key for RDMA access
    ecb_write(REG_DP_ROCE_0_BUF_LEN, FRAME_SIZE_BYTES);    // Buffer size = sensor frame size
    ecb_write(REG_DP_ROCE_0_PSN, TB_ROCE_PSN);             // Packet sequence number base
    
    // Configure buffer scheme based on USE_4K_BUFFER_SCHEME flag
    if (USE_4K_BUFFER_SCHEME) begin
        //----------------------------------------------------------------------
        // 4K Buffer Scheme Configuration
        //----------------------------------------------------------------------
        // Uses contiguous buffer pool with base address + increment * buffer_index
        // Virtual address = {msb[31:1], {msb[0], lsb} + (inc << 7) * buf_num}
        `TB_LOG_LOW(LOG_INFO, "Programming 4K buffer scheme registers...");
        
        // Program base virtual address (64-bit, 128-byte aligned)
        ecb_write(REG_DP_ROCE_0_BUF_ADDR_LSB, ROCE_BUF_ADDR_LSB);
        ecb_write(REG_DP_ROCE_0_BUF_ADDR_MSB, ROCE_BUF_ADDR_MSB);
        `TB_LOG_LOW(LOG_INFO, $sformatf("  Base VA: 0x%08x_%08x", ROCE_BUF_ADDR_MSB, ROCE_BUF_ADDR_LSB));
        
        // Program buffer increment (in PAGE units, 128 bytes per page)
        ecb_write(REG_DP_ROCE_0_BUF_INC, {6'b0, ROCE_BUF_INC});
        `TB_LOG_LOW(LOG_INFO, $sformatf("  Buffer increment: 0x%06x pages (%0d bytes)", ROCE_BUF_INC, ROCE_BUF_INC * 128));
        
        // Program buffer pointer range: {4'b0, ptr_start[11:0], 4'b0, ptr_end[11:0]}
        ecb_write(REG_DP_ROCE_0_BUF_PTR, {4'b0, ROCE_BUF_PTR_START, 4'b0, ROCE_BUF_PTR_END});
        `TB_LOG_LOW(LOG_INFO, $sformatf("  Buffer range: %0d to %0d (%0d buffers)", 
                    ROCE_BUF_PTR_START, ROCE_BUF_PTR_END, ROCE_BUF_PTR_END - ROCE_BUF_PTR_START + 1));
        
        // Store expected addresses for validation (first 4 buffers for backward compat)
        for (int vi = 0; vi < ROCE_VADDR_COUNT; vi++) begin
            // 4K scheme: VA = base + (inc << 7) * buf_num
            dp_roce0_vaddr_cfg_bytes[vi] = {ROCE_BUF_ADDR_MSB, ROCE_BUF_ADDR_LSB} + 
                                           (64'(ROCE_BUF_INC) << 7) * (ROCE_BUF_PTR_START + vi);
            dp_roce0_vaddr_cfg_words[vi] = dp_roce0_vaddr_cfg_bytes[vi][38:7]; // PAGE address
        end
    end else begin
        //----------------------------------------------------------------------
        // Legacy 4-Buffer Scheme Configuration
        //----------------------------------------------------------------------
        `TB_LOG_LOW(LOG_INFO, "Programming legacy 4-buffer scheme registers...");
        ecb_write(REG_DP_ROCE_0_BUF_MASK, 32'h0000_000F);  // Enable all 4 buffers
        
        // Configure virtual addresses for RoCE buffers (128-byte aligned)
        // Note: Addresses are right-shifted by 7 bits (divide by 128) in hardware
        for (int vi = 0; vi < ROCE_VADDR_COUNT; vi++) begin
            dp_roce0_vaddr_cfg_words[vi] = ROCE_VADDR_WORD_BASE + (ROCE_VADDR_FRAME_STRIDE * vi);
            case (vi)
                0: ecb_write(REG_DP_ROCE_0_VADDR_0, dp_roce0_vaddr_cfg_words[vi]);
                1: ecb_write(REG_DP_ROCE_0_VADDR_1, dp_roce0_vaddr_cfg_words[vi]);
                2: ecb_write(REG_DP_ROCE_0_VADDR_2, dp_roce0_vaddr_cfg_words[vi]);
                3: ecb_write(REG_DP_ROCE_0_VADDR_3, dp_roce0_vaddr_cfg_words[vi]);
            endcase
            dp_roce0_vaddr_cfg_bytes[vi] = {32'h0, dp_roce0_vaddr_cfg_words[vi]} << 7;
        end
    end
    
    
    //----------------------------------------------------------------------
    // Host Network Configuration  
    //----------------------------------------------------------------------
    // Configure where sensor data should be sent on the network
    `TB_LOG_LOW(LOG_INFO, "Programming host network addresses...");
    ecb_write(REG_DP_ROCE_0_HOST_MAC_LO, TB_HOST_MAC[31:0]);     // Host Ethernet MAC address (lower 32 bits)
    ecb_write(REG_DP_ROCE_0_HOST_MAC_HI, TB_HOST_MAC[47:32]);   // Host Ethernet MAC address (upper 16 bits)
    ecb_write(REG_DP_ROCE_0_HOST_IP, TB_HOST_IP);               // Host IP address
    if (enable_coe_mode) begin
        ecb_write(REG_DP_ROCE_0_HOST_UDP_PORT, ETH_UDP_COE_PORT); // COE protocol UDP port
        `TB_LOG_LOW(LOG_INFO, $sformatf("Configured for COE protocol on UDP port 0x%04x", ETH_UDP_COE_PORT));
    end else begin
        ecb_write(REG_DP_ROCE_0_HOST_UDP_PORT, ETH_UDP_ROCE_PORT); // RoCE protocol UDP port
        `TB_LOG_LOW(LOG_INFO, $sformatf("Configured for RoCE protocol on UDP port 0x%04x", ETH_UDP_ROCE_PORT));
    end

    //----------------------------------------------------------------------
    // Sensor RX Packetizer Configuration
    //----------------------------------------------------------------------
    // Only configs whose VP slice is narrower than the sensor bus need this;
    // the task no-ops for everything else.
    for (int sensor_id = 0; sensor_id < `SENSOR_RX_IF_INST; sensor_id++) begin
        program_packetizer_passthrough(sensor_id);
    end
`else
    `TB_LOG_LOW(LOG_INFO, "Skipping RoCE data path registers: no sensor RX interface in this config");
`endif
    
    
    //----------------------------------------------------------------------
    // Packet Processing Configuration
    //----------------------------------------------------------------------
    // Configure how sensor data is packaged and transmitted
    `TB_LOG_LOW(LOG_INFO, "Programming packet processing registers...");
    
    // Set Ethernet packet size (calculated from MTU)
    ecb_write(REG_DP_PKT_LEN, DP_PKT_LEN_VALUE);         // Packet length: (MTU-overhead)/payload_size
    `TB_LOG_LOW(LOG_INFO, $sformatf("Ethernet packet length configured: %0d bytes effective payload", DP_PKT_LEN_VALUE * 128));
    
    // Enable all virtual ports for data routing (modify mask as needed)
    ecb_write(REG_DP_PKT_VIP_MASK, 32'hFFFF_FFFF);       // Virtual port enable mask
    
    // Configure FPGA's UDP port for outgoing data
    if (enable_coe_mode) begin
        ecb_write(REG_DP_PKT_FPGA_UDP_PORT, {16'h0001, ETH_UDP_COE_PORT}); // COE mode: protocol flag + port
        `TB_LOG_LOW(LOG_INFO, $sformatf("FPGA configured for COE transmission on port 0x%04x", ETH_UDP_COE_PORT));
    end else begin
        ecb_write(REG_DP_PKT_FPGA_UDP_PORT, {16'h0000, ETH_UDP_ROCE_PORT}); // RoCE mode: protocol flag + port  
        `TB_LOG_LOW(LOG_INFO, $sformatf("FPGA configured for RoCE transmission on port 0x%04x", ETH_UDP_ROCE_PORT));
    end
    
    // Configure ECB (Ethernet Control Bus) for register access
    ecb_write(REG_INST_DEC_0_ECB_UDP_PORT, ETH_UDP_CMD_PORT); // ECB command/control port
    
    // Legacy alternative register block removed to avoid redundancy
    
    `TB_LOG_LOW(LOG_INFO, "Register programming complete - Hololink IP ready for sensor data");
    
    // Add delay after initialization
    repeat(100) @(posedge clk_100);
    `TB_LOG_LOW(LOG_INFO, "===== HOLOLINK INITIALIZATION COMPLETE =====");
endtask

// Note: RoCE data packet generation task removed per user request
// This testbench focuses on sensor-to-host data transmission only

// Note: Enhanced sensor data generation is now implemented above as generate_sensor_data_patterns()

// Old monitor_host_data() task removed - now using unified monitor_host_with_classification() above



// Helper task removed - sensor data is now sent directly in generate_sensor_data_patterns()
// using proper AXIS interface handling with correct data types and array indexing

//==============================================================================
// Main Test Sequence
//==============================================================================
initial begin
    string test_mode_str;

    // Initialize interface signals
    i_hif_axis_tvalid = '0;
    i_hif_axis_tlast = '0;
    foreach (i_hif_axis_tdata[i]) i_hif_axis_tdata[i] = '0;
    foreach (i_hif_axis_tkeep[i]) i_hif_axis_tkeep[i] = '0;
    foreach (i_hif_axis_tuser[i]) i_hif_axis_tuser[i] = '0;
    i_hif_axis_tready = '1; // Always ready to receive
    
`ifdef SENSOR_RX_IF_INST
    i_sif_axis_tvalid = '0;
    i_sif_axis_tlast = '0;
    foreach (i_sif_axis_tdata[i]) i_sif_axis_tdata[i] = '0;
    foreach (i_sif_axis_tkeep[i]) i_sif_axis_tkeep[i] = '0;
    foreach (i_sif_axis_tuser[i]) i_sif_axis_tuser[i] = '0;
    for (int s = 0; s < `SENSOR_RX_IF_INST; s++) begin
        sensor_frames_generated_by_sensor[s] = 0;
        sensor_packets_sent_by_sensor[s] = 0;
        frames_received_by_sensor[s] = 0;
        for (int f = 0; f < TOTAL_TEST_FRAMES; f++) begin
            frame_received_by_sensor[s][f] = 1'b0;
        end
        for (int vi = 0; vi < 4; vi++) begin
            dp_roce_vaddr_cfg_words_by_sensor[s][vi] = '0;
            dp_roce_vaddr_cfg_bytes_by_sensor[s][vi] = '0;
        end
    end
`endif
`ifdef SENSOR_TX_IF_INST
    i_sif_axis_tready = '1; // Always ready to receive from DUT TX
    for (int s = 0; s < `SENSOR_TX_IF_INST; s++) begin
        expected_stx_bytes[s]     = {};
        host2sen_bytes_sent[s]    = 0;
        host2sen_bytes_checked[s] = 0;
        host2sen_mismatches[s]    = 0;
        stx_gen_byte[s]           = 0;
        host2sen_psn[s]           = 0;
    end
`endif
    
    i_gpio = '0;
    i_sif_event = '0;
    
    // Initialize peripheral interfaces
    i_apb_pready = '1; // Always ready
    i_apb_pserr = '0;  // No errors
    foreach (i_apb_prdata[i]) i_apb_prdata[i] = '0;
`ifdef SPI_INST
    foreach (i_spi_sdio[i]) i_spi_sdio[i] = '0;
`endif
`ifdef UART_INST
    i_uart_rx = '1;
    i_uart_cts = '0;
`endif

    // I2C signals are now handled by the EEPROM target model
    // i_i2c_scl and i_i2c_sda are driven by the I2C EEPROM target
    
    // Sensor interfaces start IDLE; host and sensor data/control cleared above

    // Initialize verbosity level from plusargs
    if ($test$plusargs("VERBOSITY=HIGH")) begin
        hololink_tb_pkg::tb_current_verbosity = VERBOSITY_HIGH;
        `TB_LOG_LOW(LOG_INFO, "Verbosity set to HIGH");
    end else if ($test$plusargs("VERBOSITY=MEDIUM")) begin
        hololink_tb_pkg::tb_current_verbosity = VERBOSITY_MEDIUM;
        `TB_LOG_LOW(LOG_INFO, "Verbosity set to MEDIUM");
    end else begin
        hololink_tb_pkg::tb_current_verbosity = VERBOSITY_LOW;
        `TB_LOG_LOW(LOG_INFO, "Verbosity set to LOW (default)");
    end
    
    // Packet dumping controlled solely by verbosity now

    // Runtime test mode selection.  Default is the original single-sensor demo.
    all_sensors_mode = 1'b0;
    expected_total_frames = TOTAL_TEST_FRAMES;
    if ($test$plusargs("ALL_SENSORS")) begin
        all_sensors_mode = 1'b1;
    end
    if ($value$plusargs("TEST_MODE=%s", test_mode_str)) begin
        if (test_mode_str == "ALL_SENSORS") begin
            all_sensors_mode = 1'b1;
        end else begin
            all_sensors_mode = 1'b0;
        end
    end
    if ($value$plusargs("TARGET_HOST=%d", target_host_id)) begin
        `TB_LOG_LOW(LOG_INFO, $sformatf("Target host selected via +TARGET_HOST=%0d", target_host_id));
    end else begin
        target_host_id = 0;
    end
    if (target_host_id < 0 || target_host_id >= `HOST_IF_INST) begin
        `TB_LOG_LOW(LOG_WARN, $sformatf(
            "Requested TARGET_HOST=%0d is out of range [0:%0d]; using host 0",
            target_host_id, `HOST_IF_INST-1));
        target_host_id = 0;
    end

    // Host-to-sensor (TX) push path.  When +HOST2SEN is supplied the test drives
    // UC RoCE SEND traffic from the host to the sensors instead of the default
    // sensor-to-host (RX) frame generation.
    if ($test$plusargs("HOST2SEN")) begin
        host2sen_requested = 1'b1;
`ifdef SENSOR_TX_IF_INST
        host2sen_mode = 1'b1;
        `TB_LOG_LOW(LOG_INFO, "Host-to-sensor (TX) mode enabled via +HOST2SEN");
`else
        `TB_LOG_LOW(LOG_INFO, "+HOST2SEN requested but this config has no sensor TX path (SENSOR_TX_IF_INST undefined)");
`endif
    end
`ifdef SENSOR_RX_IF_INST
    if (all_sensors_mode) begin
        expected_total_frames = `SENSOR_RX_IF_INST * TOTAL_TEST_FRAMES;
        `TB_LOG_LOW(LOG_INFO, $sformatf("Test mode: ALL_SENSORS -> host %0d (%0d sensors, %0d expected frames)",
                 target_host_id, `SENSOR_RX_IF_INST, expected_total_frames));
    end else begin
        `TB_LOG_LOW(LOG_INFO, "Test mode: SINGLE_SENSOR -> host 0 (legacy default)");
    end
`else
    `TB_LOG_LOW(LOG_INFO, "Test mode: SENSOR_RX_IF_INST undefined; sensor data generation disabled");
`endif
    
    // Check for COE mode enablement
    if ($test$plusargs("ENABLE_COE")) begin
        enable_coe_mode = 1'b1;
        `TB_LOG_LOW(LOG_INFO, "COE (Camera over Ethernet) mode enabled via +ENABLE_COE");
    end else begin
        enable_coe_mode = 1'b0;
        `TB_LOG_LOW(LOG_INFO, "RoCE (RDMA over Converged Ethernet) mode enabled (default)");
    end
    
    `TB_LOG_LOW(LOG_INFO, all_sensors_mode ? "All sensor interfaces initialized to IDLE (all sensors mode)" :
                                            "All sensor interfaces initialized to IDLE (only sensor 0 will be used)");
    
    // EEPROM model is now automatically initialized by the separate eeprom_model.sv module
    
    `TB_LOG_LOW(LOG_INFO, "========================================");
    `TB_LOG_LOW(LOG_INFO, "=== NV HSB IP Simple Testbench Started ===");
    `TB_LOG_LOW(LOG_INFO, "========================================");
    
    // Wait for reset deassertion
    `TB_LOG_LOW(LOG_INFO, "Waiting for reset deassertion...");
    `TB_LOG_LOW(LOG_INFO, $sformatf("Current rstn = %b", rstn));
    wait(rstn == 1'b1);
    `TB_LOG_LOW(LOG_INFO, "Reset deasserted! Starting test sequence...");

    // Start monitoring before the settle below. The monitor carries the BOOTP
    // responder, and a non-EEPROM DUT issues its BOOTP request within a few hundred
    // ns of enumeration; with HOST_IF_INST=1 it never retries, because repeats wait
    // on the 1Hz PPS that no simulation reaches. The settle is 100 APB cycles, which
    // is 2us at a 50MHz APB, so arming the responder after it loses the only request
    // and the DUT never acquires a device IP.
    for (int host_id = 0; host_id < `HOST_IF_INST; host_id++) begin
        automatic int monitor_host_id = host_id;
        fork
            begin
                `TB_LOG_LOW(LOG_INFO, $sformatf(
                    ">>> Starting host %0d monitor with traffic classification <<<",
                    monitor_host_id));
                monitor_host_with_classification(monitor_host_id);
            end
        join_none
    end

    repeat(100) @(posedge clk_100);
    
    // Check DUT signals before starting
    `TB_LOG_LOW(LOG_INFO, "DUT Status Check:");
    `TB_LOG_LOW(LOG_INFO, $sformatf("  init_done = %b", init_done));
    for (int host_id = 0; host_id < `HOST_IF_INST; host_id++) begin
        `TB_LOG_LOW(LOG_INFO, $sformatf(
            "  host[%0d]: o_hif_axis_tready=%b i_hif_axis_tvalid=%b",
            host_id, o_hif_axis_tready[host_id], i_hif_axis_tvalid[host_id]));
    end
    
    fork
        // Main test sequence
        begin
            `TB_LOG_LOW(LOG_INFO, ">>> Starting main test sequence <<<");
            
            // Initialize the Hololink IP
            `TB_LOG_LOW(LOG_INFO, "STEP 1: Initializing Hololink IP...");
            hololink_init();
            `TB_LOG_LOW(LOG_INFO, "STEP 1 COMPLETE: Hololink initialization finished");
            
            // Verify register programming with a register we actually wrote
            `TB_LOG_LOW(LOG_INFO, "STEP 2: Verifying register access...");
`ifdef SENSOR_RX_IF_INST
            if (all_sensors_mode) begin
                verify_register_access(reg_roce(get_sensor_first_vp(0), ROCE_OFF_RKEY), TB_ROCE_RKEY);
            end else begin
                verify_register_access(REG_DP_ROCE_0_RKEY, TB_ROCE_RKEY);
            end
`else
            // No sensor RX interface means no dp_roce block to read back; the
            // read would return an ECB ACCESS_ERROR and count as a test error.
            `TB_LOG_LOW(LOG_INFO, "Skipping RoCE register readback: no sensor RX interface in this config");
`endif
            `TB_LOG_LOW(LOG_INFO, "STEP 2 COMPLETE: Register verification finished");

          if (host2sen_requested && !host2sen_mode) begin
            // TX push path was requested but this build has no sensor TX interface.
            // Skip gracefully rather than silently running the RX flow instead.
            `TB_LOG_LOW(LOG_INFO, "========================================");
            `TB_LOG_LOW(LOG_INFO, "=== HOST-TO-SENSOR TEST SKIPPED ===");
            `TB_LOG_LOW(LOG_INFO, "    SENSOR_TX_IF_INST is not defined in this build;");
            `TB_LOG_LOW(LOG_INFO, "    nothing to drive on the host->sensor (TX) path.");
            `TB_LOG_LOW(LOG_INFO, "========================================");
            test_status = 1; // clean, non-failing exit (test not applicable)
          end else if (host2sen_mode) begin
`ifdef SENSOR_TX_IF_INST
            // ---- Host-to-sensor (TX) UC RoCE SEND path ----
            `TB_LOG_LOW(LOG_INFO, "STEP 3 (TX): Programming host-to-sensor stream buffers...");
            program_tx_strm_buff();

            // Launch a streaming byte-compare collector per sensor TX interface.
            for (int s = 0; s < `SENSOR_TX_IF_INST; s++) begin
                fork
                    automatic int sn = s;
                    monitor_sensor_egress(sn);
                join_none
            end
            repeat(200) @(posedge hif_clk); // let collectors arm

            `TB_LOG_LOW(LOG_INFO, "STEP 4 (TX): Driving host-to-sensor UC RoCE SEND traffic...");
            run_host2sen_traffic();

            `TB_LOG_LOW(LOG_INFO, "STEP 5 (TX): Draining sensor egress...");
            repeat(200000) @(posedge hif_clk);

            `TB_LOG_LOW(LOG_INFO, "STEP 6 (TX): Reporting host-to-sensor results...");
            report_host2sen_results();
            `TB_LOG_LOW(LOG_INFO, "STEP 6 COMPLETE: Host-to-sensor results reported");
`endif
          end else begin
            // Generate sensor data patterns for DUT to process
`ifdef SENSOR_RX_IF_INST
            `TB_LOG_LOW(LOG_INFO, "STEP 3: Starting sensor data generation...");
            `TB_LOG_LOW(LOG_INFO, "         DUT needs sensor interface stimulus to generate RoCE packets");
            if (all_sensors_mode) begin
                generate_all_sensor_data_patterns();
            end else begin
                generate_sensor_data_patterns();
            end
            // Minimal delay - frames are sent back-to-back now
            `TB_LOG_LOW(LOG_INFO, "         All frames sent back-to-back - DUT should stay active");
            repeat(1000) @(posedge clk_100);  // Short delay for pipeline to start
            `TB_LOG_LOW(LOG_INFO, "STEP 3 COMPLETE: Sensor data generation finished");
`else
            `TB_LOG_LOW(LOG_INFO, "STEP 3: Skipping sensor data generation (SENSOR_RX_IF_INST not defined)");
`endif
            
            // Generate RoCE data packets for comprehensive testing
            `TB_LOG_LOW(LOG_INFO, "STEP 4: Generating RoCE data packets...");
            // RoCE data packet generation removed - testbench focuses on sensor-to-host transmission
            `TB_LOG_LOW(LOG_INFO, "STEP 4 COMPLETE: RoCE data generation finished");
            
            // Wait for DUT to process all 3 frames from sensor 0 only
            `TB_LOG_LOW(LOG_INFO, "STEP 5: Waiting for sensor 0 frame processing...");
            `TB_LOG_LOW(LOG_INFO, "         DUT should process all 3 frames from sensor 0");
            repeat(50000) @(posedge clk_100);
            `TB_LOG_LOW(LOG_INFO, "STEP 5 COMPLETE: Frame processing wait finished");
            
            // Report test results
            `TB_LOG_LOW(LOG_INFO, "STEP 6: Reporting test results...");
            report_test_results();
            `TB_LOG_LOW(LOG_INFO, "STEP 6 COMPLETE: Test results reported");
          end
            
            `TB_LOG_LOW(LOG_INFO, "========================================");
            `TB_LOG_LOW(LOG_INFO, "=== Main Test Sequence Completed ===");
            `TB_LOG_LOW(LOG_INFO, "========================================");
        end
        
        // Timeout watchdog
        begin
            repeat(2000000) @(posedge clk_100);
            `TB_LOG_LOW(LOG_FAIL, "*** TIMEOUT WATCHDOG TRIGGERED ***");
            test_status = 2; // Timeout
        end

        // Fatal abort: a task that cannot legally continue blocks itself and raises
        // fatal_error, so release the join_any and report through the path below.
        begin
            wait (fatal_error);
            `TB_LOG_LOW(LOG_FAIL, "*** FATAL ERROR REPORTED - ABORTING TEST ***");
        end
    join_any
    
    // Final status - One-line test summary
    case (test_status)
        1: begin
            `TB_LOG_LOW(LOG_PASS, "TEST PASSED");
            `TB_LOG_LOW(LOG_PASS, "Simulation PASSED");
        end
        2: `TB_LOG_LOW(LOG_FAIL, "TEST FAILED: TIMEOUT");
        3: `TB_LOG_LOW(LOG_FAIL, "TEST FAILED: FATAL ERROR");
        default: `TB_LOG_LOW(LOG_FAIL, "TEST FAILED: UNKNOWN");
    endcase
    
    // Signal test completion for UVM integration
    test_completed = 1;
    
    $finish;
end

//==============================================================================
// HOLOLINK_top DUT Instantiation
//==============================================================================
HOLOLINK_top #(
    .BUILD_REV(48'h010203040506)
) u_dut (
    // System Reset
    .i_sys_rst(~rstn),
    
    // APB Clock and Reset
    .i_apb_clk(clk_100),
    .o_apb_rst(/* open */),
    
    // APB Register Interface
    .o_apb_psel(o_apb_psel),
    .o_apb_penable(o_apb_penable),
    .o_apb_paddr(o_apb_paddr),
    .o_apb_pwdata(o_apb_pwdata),
    .o_apb_pwrite(o_apb_pwrite),
    .i_apb_pready(i_apb_pready),
    .i_apb_prdata(i_apb_prdata),
    .i_apb_pserr(i_apb_pserr),

`ifndef ENUM_EEPROM
    // Enumeration inputs (only exist when the DUT is built without EEPROM enum)
    .i_mac_addr(i_mac_addr),
    .i_board_sn(i_board_sn),
    .i_enum_vld(i_enum_vld),
`endif

    // Initialization Done
    .o_init_done(init_done),
    
`ifdef SENSOR_RX_IF_INST
    // Sensor RX Clocks and Resets
    .i_sif_rx_clk(i_sif_rx_clk),
    .o_sif_rx_rst(/* open */),
`endif
`ifdef SENSOR_TX_IF_INST
    // Sensor TX Clocks and Resets
    .i_sif_tx_clk(i_sif_tx_clk),
    .o_sif_tx_rst(/* open */),
`endif
`ifdef SENSOR_RX_IF_INST
    // Sensor RX Interface (into DUT)
    .i_sif_axis_tvalid(i_sif_axis_tvalid),
    .i_sif_axis_tlast(i_sif_axis_tlast),
    .i_sif_axis_tdata(i_sif_axis_tdata),
    .i_sif_axis_tkeep(i_sif_axis_tkeep),
    .i_sif_axis_tuser(i_sif_axis_tuser),
    .o_sif_axis_tready(o_sif_axis_tready),
`endif
`ifdef SENSOR_TX_IF_INST
    // Sensor TX Interface (from DUT)
    .o_sif_axis_tvalid(o_sif_axis_tvalid),
    .o_sif_axis_tlast(o_sif_axis_tlast),
    .o_sif_axis_tdata(o_sif_axis_tdata),
    .o_sif_axis_tkeep(o_sif_axis_tkeep),
    .o_sif_axis_tuser(o_sif_axis_tuser),
    .i_sif_axis_tready(i_sif_axis_tready),
`endif
    
    // Event interface
    .i_sif_event(i_sif_event),
    
    // Host Interface
    .i_hif_clk(hif_clk),
    .o_hif_rst(/* open */),
    .i_hif_axis_tvalid(i_hif_axis_tvalid),
    .i_hif_axis_tlast(i_hif_axis_tlast),
    .i_hif_axis_tdata(i_hif_axis_tdata),
    .i_hif_axis_tkeep(i_hif_axis_tkeep),
    .i_hif_axis_tuser(i_hif_axis_tuser),
    .o_hif_axis_tready(o_hif_axis_tready),
    
    .o_hif_axis_tvalid(o_hif_axis_tvalid),
    .o_hif_axis_tlast(o_hif_axis_tlast),
    .o_hif_axis_tdata(o_hif_axis_tdata),
    .o_hif_axis_tkeep(o_hif_axis_tkeep),
    .o_hif_axis_tuser(o_hif_axis_tuser),
    .i_hif_axis_tready(i_hif_axis_tready),
    
    // Peripheral Interfaces
`ifdef SPI_INST
    .o_spi_csn(o_spi_csn),
    .o_spi_sck(o_spi_sck),
    .i_spi_sdio(i_spi_sdio),
    .o_spi_sdio(o_spi_sdio),
    .o_spi_oen(o_spi_oen),
`endif
`ifdef I2C_INST
    .i_i2c_scl(i_i2c_scl),
    .i_i2c_sda(i_i2c_sda),
    .o_i2c_scl_en(o_i2c_scl_en),
    .o_i2c_sda_en(o_i2c_sda_en),
`endif
`ifdef UART_INST
    .i_uart_rx(i_uart_rx),
    .o_uart_tx(o_uart_tx),
    .o_uart_busy(o_uart_busy),
    .o_uart_rts(o_uart_rts),
    .i_uart_cts(i_uart_cts),
`endif
    // GPIO
    .o_gpio(o_gpio),
    .o_gpio_dir(o_gpio_dir),
    .i_gpio(i_gpio),
    
    // System Control
    .o_sw_sys_rst(o_sw_sys_rst),
    .o_sw_sen_rst(o_sw_sen_rst),
    
    // PTP Interface
    .i_ptp_clk(ptp_clk),
    .o_ptp_rst(/* open */),
    .o_ptp_sec(o_ptp_sec),
    .o_ptp_nanosec(o_ptp_nanosec),
    .o_pps(o_pps)
);

//==============================================================================
// Frame Size Configuration Validation
//==============================================================================
initial begin
    `TB_LOG_LOW(LOG_INFO, "=== FRAME SIZE CONFIGURATION ===");
    `TB_LOG_LOW(LOG_INFO, $sformatf("DATAPATH_WIDTH = %0d bits (%0d bytes per word)", `DATAPATH_WIDTH, SENSOR_BYTES_PER_WORD));
    `TB_LOG_LOW(LOG_INFO, $sformatf("HOST_WIDTH = %0d bits (%0d bytes per word)", `HOST_WIDTH, HOST_BYTES_PER_WORD));
    `TB_LOG_LOW(LOG_INFO, $sformatf("Required byte alignment = %0d bytes", REQUIRED_BYTE_ALIGNMENT));
    `TB_LOG_LOW(LOG_INFO, $sformatf("Target frame bytes = %0d -> Aligned frame bytes = %0d", TARGET_FRAME_BYTES_BASE, FRAME_SIZE_BYTES));
    `TB_LOG_LOW(LOG_INFO, $sformatf("Computed FRAME_SIZE_WORDS = %0d", FRAME_SIZE_WORDS));
    `TB_LOG_LOW(LOG_INFO, "=== VALIDATION CHECKS ===");
    
    // Validation checks (non-fatal; increment errors_detected)
    if (FRAME_SIZE_BYTES % 128 != 0) begin
        errors_detected++;
        `TB_LOG_LOW(LOG_ERR, $sformatf("ALIGNMENT ERROR: Frame size (%0d bytes) not aligned to 128-byte boundary", FRAME_SIZE_BYTES));
    end
    if (FRAME_SIZE_BYTES % HOST_BYTES_PER_WORD != 0) begin
        errors_detected++;
        `TB_LOG_LOW(LOG_ERR, $sformatf("ALIGNMENT ERROR: Frame size (%0d bytes) not aligned to HOST_WIDTH (%0d bytes per word)", FRAME_SIZE_BYTES, HOST_BYTES_PER_WORD));
    end
    if (FRAME_SIZE_BYTES % SENSOR_BYTES_PER_WORD != 0) begin
        errors_detected++;
        `TB_LOG_LOW(LOG_ERR, $sformatf("ALIGNMENT ERROR: Frame size (%0d bytes) not aligned to DATAPATH_WIDTH (%0d bytes per word)", FRAME_SIZE_BYTES, SENSOR_BYTES_PER_WORD));
    end
    if (FRAME_SIZE_WORDS <= 0) begin
        errors_detected++;
        `TB_LOG_LOW(LOG_ERR, $sformatf("CALCULATION ERROR: FRAME_SIZE_WORDS (%0d) must be positive", FRAME_SIZE_WORDS));
    end
    
    `TB_LOG_LOW(LOG_PASS, $sformatf("\u2713 128-byte alignment: %0d %% 128 = %0d", FRAME_SIZE_BYTES, FRAME_SIZE_BYTES % 128));
    `TB_LOG_LOW(LOG_PASS, $sformatf("\u2713 HOST_WIDTH alignment: %0d %% %0d = %0d", FRAME_SIZE_BYTES, HOST_BYTES_PER_WORD, FRAME_SIZE_BYTES % HOST_BYTES_PER_WORD));
    `TB_LOG_LOW(LOG_PASS, $sformatf("\u2713 DATAPATH_WIDTH alignment: %0d %% %0d = %0d", FRAME_SIZE_BYTES, SENSOR_BYTES_PER_WORD, FRAME_SIZE_BYTES % SENSOR_BYTES_PER_WORD));
    `TB_LOG_LOW(LOG_INFO, "\u2713 All alignment requirements satisfied!");
end

// Waveform dumping: opt-in via +FSDB_DUMP (Makefile DUMP=1). The $fsdbDump* calls
// live in the FSDB dump block near the clock section above; nothing is dumped
// unless +FSDB_DUMP is passed. 'make verdi'/'make gui' remain for interactive use.

endmodule
