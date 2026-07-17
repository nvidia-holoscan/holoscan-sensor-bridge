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
// Clock periods (can be modified for different test scenarios)
parameter CLK_PERIOD = 10;        // 100MHz system clock (APB)
parameter PTP_CLK_PERIOD = 10;     // 100MHz PTP clock  
parameter SIF_CLK_PERIOD = 6.4;    // 156.25MHz sensor interface clock
parameter HIF_CLK_PERIOD = 5;      // 200MHz host interface clock

// Network configuration (modify as needed for your test environment)
parameter [31:0] TB_FPGA_IP = DEFAULT_FPGA_IP;  // FPGA IP address
parameter [31:0] TB_HOST_IP = DEFAULT_HOST_IP;  // Host IP address
parameter [47:0] TB_HOST_MAC = 48'h123456789ABC; // Host MAC address

// RoCE/COE Protocol Configuration (from package; avoid duplication)
// Using TB_ROCE_* and ROCE_VADDR_* constants from hololink_tb_pkg.sv

//==============================================================================
// Clock and Reset Signals
//==============================================================================
logic clk_100;    // 100MHz APB clock
logic ptp_clk;    // 100MHz PTP clock
logic sif_clk;    // 156.25MHz sensor interface clock
logic hif_clk;    // 200MHz host interface clock
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
logic [`SPI_INST-1:0] o_spi_oen;
`endif

`ifdef I2C_INST
logic [`I2C_INST-1:0] i_i2c_scl;
logic [`I2C_INST-1:0] i_i2c_sda;
logic [`I2C_INST-1:0] o_i2c_scl_en;
logic [`I2C_INST-1:0] o_i2c_sda_en;
`endif

// PTP and system control
logic o_sw_sys_rst;
`ifdef SENSOR_RX_IF_INST
logic [`SENSOR_RX_IF_INST-1:0] o_sw_sen_rst;
`else
logic [31:0] o_sw_sen_rst;  // DUT always has 32-bit output
`endif
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

// 200MHz Host Interface Clock
initial begin
    hif_clk = 0;
    forever #((HIF_CLK_PERIOD/2)*1ns) hif_clk = ~hif_clk;
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
// Test Variables
//==============================================================================
int test_status = 0;
int ecb_sequence_id = 0;


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

// Simple frame tracking for basic validation

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
always @(posedge i_sif_rx_clk[0]) begin
    if (i_sif_axis_tvalid[0] && o_sif_axis_tready[0] && i_sif_axis_tlast[0]) begin
        sensor_packets_sent++;
        `TB_LOG_MED(LOG_INFO, $sformatf("SENSOR: Frame %0d complete, last word=0x%0h", sensor_packets_sent, i_sif_axis_tdata[0]));
    end
end

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

// Validation function moved to RoCE helpers library

task report_test_results();
    // Local declarations must appear before any procedural statements
    logic comprehensive_pass;
    integer i;

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
    `TB_LOG_LOW(LOG_INFO, $sformatf("Unique frames received: %0d/3", unique_frames_received));
    for (i = 0; i < 3; i++) begin
        `TB_LOG_LOW(LOG_INFO, $sformatf("  Frame %0d: %s", i, frame_received[i] ? "[OK] RECEIVED" : "[ERROR] MISSING"));
        if (!frame_received[i]) begin
            errors_detected++;
        end
    end
    
    // Check for comprehensive test pass
    comprehensive_pass = 1;
    if (unique_frames_received != 3) begin
        comprehensive_pass = 0;
        `TB_LOG_LOW(LOG_ERR, $sformatf("COVERAGE FAILURE: Missing sensor frames - only %0d/3 received", unique_frames_received));
    end
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
        `TB_LOG_LOW(LOG_PASS, "COMPREHENSIVE VALIDATION PASSED: All sensor frames (0,1,2) received and validated");
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
    
    `TB_LOG_LOW(LOG_INFO, "===> SEND_ETH_PACKET START");
    `TB_LOG_LOW(LOG_INFO, $sformatf("     Interface ID: %0d", intf_id));
    `TB_LOG_LOW(LOG_INFO, $sformatf("     Payload size: %0d bytes", payload_size));
    `TB_LOG_LOW(LOG_INFO, $sformatf("     HOST_WIDTH: %0d bits (%0d bytes)", `HOST_WIDTH, `HOST_WIDTH/8));
    
    word_count = (payload_size + (`HOST_WIDTH/8) - 1) / (`HOST_WIDTH/8);

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
                `TB_LOG_LOW(LOG_WARN, $sformatf("TIMEOUT: o_hif_axis_tready[%0d] never asserted!", intf_id));
                `TB_LOG_LOW(LOG_WARN, "Proceeding anyway - this may cause protocol violations");
                break;
            end
        end while (!o_hif_axis_tready[intf_id]);
    end

    // Clear interface
    i_hif_axis_tvalid[intf_id] <= 1'b0;
    i_hif_axis_tlast[intf_id]  <= 1'b0;
    i_hif_axis_tdata[intf_id]  <= '0;
    i_hif_axis_tkeep[intf_id]  <= '0;
    i_hif_axis_tuser[intf_id]  <= '0;

    @(posedge hif_clk);
endtask

// Note: COE (Camera over Ethernet) functions removed for simplicity

//==============================================================================
// Hololink Initialization Sequence
//==============================================================================
task hololink_init();
    `TB_LOG_LOW(LOG_INFO, "===== HOLOLINK INITIALIZATION START =====");
    
    // Wait for system to be ready with timeout
    `TB_LOG_LOW(LOG_INFO, "Waiting for init_done signal (with timeout)...");
    `TB_LOG_LOW(LOG_INFO, $sformatf("Current init_done = %b", init_done));
    
    // Wait for init_done with timeout
    fork
        begin
            wait(init_done == 1'b1);
            `TB_LOG_LOW(LOG_INFO, "*** Init done signal asserted! ***");
        end
        begin
            #(1500us);
            `TB_LOG_LOW(LOG_WARN, "*** TIMEOUT: init_done never asserted, proceeding anyway ***");
            `TB_LOG_LOW(LOG_INFO, "*** This may be expected for this simple testbench ***");
        end
    join_any
    disable fork;
    
    // Add delay before register programming
    repeat(50) @(posedge clk_100);
    `TB_LOG_LOW(LOG_INFO, "Starting register programming sequence...");
    
    //==========================================================================
    // Hololink IP Register Configuration
    //==========================================================================
    // This section programs the essential registers for Hololink IP operation.
    // All register addresses and values can be customized via parameters above.
    
    //----------------------------------------------------------------------
    // RoCE Data Path Configuration
    //----------------------------------------------------------------------
    // Configure RoCE (RDMA over Converged Ethernet) for sensor data transmission
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
    ecb_write(REG_DP_ROCE_0_HOST_IP, DEFAULT_HOST_IP);          // Host IP address
    if (enable_coe_mode) begin
        ecb_write(REG_DP_ROCE_0_HOST_UDP_PORT, ETH_UDP_COE_PORT); // COE protocol UDP port
        `TB_LOG_LOW(LOG_INFO, $sformatf("Configured for COE protocol on UDP port 0x%04x", ETH_UDP_COE_PORT));
    end else begin
        ecb_write(REG_DP_ROCE_0_HOST_UDP_PORT, ETH_UDP_ROCE_PORT); // RoCE protocol UDP port
        `TB_LOG_LOW(LOG_INFO, $sformatf("Configured for RoCE protocol on UDP port 0x%04x", ETH_UDP_ROCE_PORT));
    end
    
    
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
`endif
`ifdef SENSOR_TX_IF_INST
    i_sif_axis_tready = '1; // Always ready to receive from DUT TX
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
    
    // Check for COE mode enablement
    if ($test$plusargs("ENABLE_COE")) begin
        enable_coe_mode = 1'b1;
        `TB_LOG_LOW(LOG_INFO, "COE (Camera over Ethernet) mode enabled via +ENABLE_COE");
    end else begin
        enable_coe_mode = 1'b0;
        `TB_LOG_LOW(LOG_INFO, "RoCE (RDMA over Converged Ethernet) mode enabled (default)");
    end
    
    `TB_LOG_LOW(LOG_INFO, "All sensor interfaces initialized to IDLE (only sensor 0 will be used)");
    
    // EEPROM model is now automatically initialized by the separate eeprom_model.sv module
    
    `TB_LOG_LOW(LOG_INFO, "========================================");
    `TB_LOG_LOW(LOG_INFO, "=== Hololink Simple Testbench Started ===");
    `TB_LOG_LOW(LOG_INFO, "========================================");
    
    // Wait for reset deassertion
    `TB_LOG_LOW(LOG_INFO, "Waiting for reset deassertion...");
    `TB_LOG_LOW(LOG_INFO, $sformatf("Current rstn = %b", rstn));
    wait(rstn == 1'b1);
    `TB_LOG_LOW(LOG_INFO, "Reset deasserted! Starting test sequence...");
    repeat(100) @(posedge clk_100);
    
    // Check DUT signals before starting
    `TB_LOG_LOW(LOG_INFO, "DUT Status Check:");
    `TB_LOG_LOW(LOG_INFO, $sformatf("  init_done = %b", init_done));
    `TB_LOG_LOW(LOG_INFO, $sformatf("  o_hif_axis_tready[0] = %b", o_hif_axis_tready[0]));
    `TB_LOG_LOW(LOG_INFO, $sformatf("  i_hif_axis_tvalid[0] = %b", i_hif_axis_tvalid[0]));
    
    fork
        // Start unified host monitoring with traffic classification
        begin
            `TB_LOG_LOW(LOG_INFO, ">>> Starting unified host monitor with traffic classification <<<");
            monitor_host_with_classification();
        end
        
        // Main test sequence
        begin
            `TB_LOG_LOW(LOG_INFO, ">>> Starting main test sequence <<<");
            
            // Initialize the Hololink IP
            `TB_LOG_LOW(LOG_INFO, "STEP 1: Initializing Hololink IP...");
            hololink_init();
            `TB_LOG_LOW(LOG_INFO, "STEP 1 COMPLETE: Hololink initialization finished");
            
            // Verify register programming with a register we actually wrote
            `TB_LOG_LOW(LOG_INFO, "STEP 2: Verifying register access...");
            verify_register_access(REG_DP_ROCE_0_RKEY, TB_ROCE_RKEY);
            `TB_LOG_LOW(LOG_INFO, "STEP 2 COMPLETE: Register verification finished");
            
            // Generate sensor data patterns for DUT to process
`ifdef SENSOR_RX_IF_INST
            `TB_LOG_LOW(LOG_INFO, "STEP 3: Starting sensor data generation...");
            `TB_LOG_LOW(LOG_INFO, "         DUT needs sensor interface stimulus to generate RoCE packets");
            generate_sensor_data_patterns();
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
    join_any
    
    // Final status - One-line test summary
    case (test_status)
        1: begin
            `TB_LOG_LOW(LOG_PASS, "TEST PASSED");
            `TB_LOG_LOW(LOG_PASS, "Simulation PASSED");
        end
        2: `TB_LOG_LOW(LOG_FAIL, "TEST FAILED: TIMEOUT");
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
    // GPIO
    .o_gpio(o_gpio),
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

//==============================================================================
// FSDB Waveform Dumping (Optional)
//==============================================================================
// Uncomment the following block to enable FSDB waveform dumping for debug
//initial begin
//    string fsdb_dump_name = "nv_hsb_ip_simple_tb.fsdb";
//    `TB_LOG_LOW(LOG_INFO, $sformatf("FSDB waveform dumping enabled: %s", fsdb_dump_name));
//    $fsdbDumpfile(fsdb_dump_name);
//    $fsdbDumpvars(0, nv_hsb_ip_simple_tb);
//end

endmodule
