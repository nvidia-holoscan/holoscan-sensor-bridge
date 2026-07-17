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
// Hololink Testbench Support Package
//
// DESCRIPTION:
//   This package contains all constants, data types, and helper functions used
//   by the Hololink IP simple testbench. It provides a centralized location for
//   configuration parameters that customers can modify for their specific test
//   requirements.
//
// USAGE:
//   Import this package in your testbench: import hololink_tb_pkg::*;
//   Modify constants below to match your network configuration and test setup.
//
// KEY COMPONENTS:
//   - Network protocol constants (RoCE, COE, ECB, BOOTP)
//   - Register address definitions for Hololink IP
//   - Frame size and timing parameters
//   - Helper functions for packet analysis
//   - Verbosity control for debug output
//==============================================================================

package hololink_tb_pkg;

    //==========================================================================
    // Verbosity and Debug Control (Modify for Different Debug Levels)
    //==========================================================================
    typedef enum int {
        VERBOSITY_LOW    = 0,    // Minimal output - test results only
        VERBOSITY_MEDIUM = 1,    // Moderate output - key transactions and status
        VERBOSITY_HIGH   = 2     // Detailed output - all packets and register operations
    } verbosity_level_t;

    // Global testbench verbosity control accessible without local declarations
    // Can be overridden at runtime: hololink_tb_pkg::tb_current_verbosity = VERBOSITY_HIGH;
    int tb_current_verbosity = VERBOSITY_LOW;

    // Data structure for sensor frame validation and reassembly
    typedef struct {
        logic [7:0] fragments[];  // Byte array containing reassembled sensor frame data
    } sensor_frame_t;

    //==========================================================================
    // ECB (Ethernet Control Bus) Protocol Constants
    //==========================================================================
    // ECB provides register access to Hololink IP over Ethernet packets
    // Command format: 8-bit opcode where bit[7] indicates response
    localparam logic [7:0] ECB_CMD_WR_DWORD      = 8'h04;  // Write 32-bit register command
    localparam logic [7:0] ECB_CMD_WR_DWORD_RESP = 8'h84;  // Write response (0x04 | 0x80)
    localparam logic [7:0] ECB_CMD_RD_DWORD      = 8'h14;  // Read 32-bit register command  
    localparam logic [7:0] ECB_CMD_RD_DWORD_RESP = 8'h94;  // Read response (0x14 | 0x80)

    //==========================================================================
    // Network Protocol Port Definitions (Customize for Your Environment)
    //==========================================================================
    // These UDP ports define how different protocols communicate with Hololink IP
    // Modify these values to match your network configuration requirements
    
    localparam logic [15:0] ETH_UDP_CMD_PORT      = 16'h2000; // ECB register access commands
    localparam logic [15:0] ETH_UDP_ROCE_PORT     = 16'h12B7; // RoCE sensor data transmission  
    localparam logic [15:0] ETH_UDP_COE_PORT      = 16'h22F0; // COE (IEEE 1722B) sensor data
    localparam logic [15:0] ETH_UDP_BOOTP_SRC_PORT = 16'h2FEC; // BOOTP protocol source
    localparam logic [15:0] ETH_UDP_BOOTP_DST_PORT = 16'h2FEB; // BOOTP protocol destination

    //=========================================================================
    // RoCE (RDMA over Converged Ethernet) Opcode Constants (avoid magic numbers)
    //=========================================================================
    localparam logic [7:0] ROCE_OPCODE_SEND_FIRST            = 8'h06;
    localparam logic [7:0] ROCE_OPCODE_SEND_MIDDLE           = 8'h07;
    localparam logic [7:0] ROCE_OPCODE_SEND_LAST             = 8'h08;
    localparam logic [7:0] ROCE_OPCODE_RDMA_WRITE_ONLY       = 8'h0A; // Regular sensor data
    localparam logic [7:0] ROCE_OPCODE_RDMA_WRITE_FIRST      = 8'h2A; // First segment of multi-packet write
    localparam logic [7:0] ROCE_OPCODE_RDMA_WRITE_ONLY_IMMDT = 8'h2B; // Meta packet after BPW

    //==========================================================================
    // COE (Camera over Ethernet) Protocol Constants - IEEE 1722B Standard
    //==========================================================================
    // COE enables camera data transmission over standard Ethernet infrastructure
    // These constants define the IEEE 1722B packet structure and identification
    
    localparam logic [7:0]  COE_FLAG_MASK        = 8'hF0;     // Protocol identification mask
    localparam logic [7:0]  COE_FLAG_VALUE       = 8'h10;     // COE protocol identifier (0001xxxx)
    localparam logic [7:0]  COE_VERSION_MASK     = 8'h07;     // IEEE 1722B version field mask
    // IEEE 1722B COE header size (bytes) at L2 payload start
    localparam int          COE_HDR_SIZE          = 32;
    // COE header byte offsets (relative to start of COE L2 payload)
    localparam int          COE_FLAGS_BYTE_OFFSET = 27;       // Flags nibble in low 4 bits
    localparam int          COE_LAST_WORD_OFFSET  = 28;       // 4-byte word with frame_number and byte_offset
    // COE header bitfield positions within the last 32-bit header word
    localparam int          COE_FRAME_NUM_MSB     = 29;
    localparam int          COE_FRAME_NUM_LSB     = 28;
    localparam int          COE_BYTE_OFFSET_MSB   = 27;
    localparam int          COE_BYTE_OFFSET_LSB   = 0;
    // COE flags nibble slice
    localparam int          COE_FLAGS_NIBBLE_MSB  = 3;
    localparam int          COE_FLAGS_NIBBLE_LSB  = 0;
    localparam logic [15:0] IEEE_1722B_ETHERTYPE = 16'h22F0;  // Standard IEEE 1722B EtherType
    localparam int          COE_STREAM_ID_SIZE    = 8;         // Stream identifier field size

    // Move COE flag enumeration near top so it is available to all code below
    typedef enum logic [3:0] {
        COE_FRAME_START = 4'h1,
        COE_MIDDLE      = 4'h0, 
        COE_END_OF_LINE = 4'h4,
        COE_FRAME_END   = 4'h2
    } coe_flags_e;



    //==========================================================================
    // Default Network Addresses (Modify for Your Test Environment)
    //==========================================================================
    // These addresses define the network endpoints for sensor data transmission
    // Change these values to match your actual host and FPGA network configuration
    
    localparam logic [47:0] DEFAULT_HOST_MAC = 48'h123456789ABC; // Host Ethernet MAC address
    localparam logic [47:0] DEFAULT_FPGA_MAC = 48'hCAFEC0FFEE00; // FPGA Ethernet MAC address  
    localparam logic [31:0] DEFAULT_HOST_IP  = 32'hC0A8010A;    // Host IP: 192.168.1.10
    localparam logic [31:0] DEFAULT_FPGA_IP  = 32'hC0A8010C;    // FPGA IP: 192.168.1.12

    //==========================================================================
    // Sensor Frame Configuration - Intelligently Calculated for All Alignments
    //==========================================================================
    // ALGORITHM: Calculate FRAME_SIZE_WORDS such that actual frame bytes satisfy:
    // 1. actual_frame_bytes = FRAME_SIZE_WORDS * (DATAPATH_WIDTH/8) 
    // 2. actual_frame_bytes % 128 == 0                    (128-byte alignment)
    // 3. actual_frame_bytes % (HOST_WIDTH/8) == 0         (Host interface alignment)
    // 4. FRAME_SIZE_WORDS is a reasonable integer         (Not too small/large)
    
    // Step 1: Calculate LCM of alignment requirements
    function automatic int gcd(int a, int b);
        while (b != 0) begin
            int temp = b;
            b = a % b;
            a = temp;
        end
        return a;
    endfunction
    
    function automatic int lcm(int a, int b);
        return (a * b) / gcd(a, b);
    endfunction
    
    // Step 2: Compute required byte alignment
    localparam int SENSOR_BYTES_PER_WORD = `DATAPATH_WIDTH / 8;
    localparam int HOST_BYTES_PER_WORD   = `HOST_WIDTH / 8;
    localparam int REQUIRED_BYTE_ALIGNMENT = lcm(128, lcm(SENSOR_BYTES_PER_WORD, HOST_BYTES_PER_WORD));
    
    // Step 3: Choose target frame size and compute FRAME_SIZE_WORDS
    localparam int TARGET_FRAME_BYTES_BASE = 6144;  // Base target: 6KB (reasonable size)
    localparam int FRAME_SIZE_BYTES = ((TARGET_FRAME_BYTES_BASE + REQUIRED_BYTE_ALIGNMENT - 1) / REQUIRED_BYTE_ALIGNMENT) * REQUIRED_BYTE_ALIGNMENT;
    localparam int FRAME_SIZE_WORDS = FRAME_SIZE_BYTES / SENSOR_BYTES_PER_WORD;
    
    localparam int TOTAL_TEST_FRAMES     = 3;                                // Number of test frames to generate
    
    // Note: Validation moved to testbench module since initial blocks are not allowed in packages
    
    // Ethernet Fragmentation Configuration (Usually no need to change)
    localparam int ETHERNET_PKT_SIZE     = 1500;                             // Standard Ethernet MTU size
    localparam int DP_PKT_LEN_VALUE      = (ETHERNET_PKT_SIZE - 78) / 128;   // Hardware packet length calculation
    localparam int EXPECTED_FRAGMENTS    = (FRAME_SIZE_BYTES + ETHERNET_PKT_SIZE - 1) / ETHERNET_PKT_SIZE; // Expected packet fragments

    //=========================================================================
    // Testbench Default Parameters Centralization (to avoid duplication)
    //=========================================================================
    // RoCE/COE protocol configuration for testbench
    localparam logic [23:0] TB_ROCE_DEST_QP        = 24'h001234;
    localparam logic [31:0] TB_ROCE_RKEY           = 32'hFACE_F00D;
    localparam logic [31:0] TB_ROCE_PSN            = 32'h0000_0001;

    // =========================================================================
    // 4K Buffer Scheme Configuration (New Mode)
    // =========================================================================
    // The 4K buffer scheme uses a contiguous buffer pool with pointer-based selection
    // instead of the legacy 4 individual buffer addresses
    localparam bit USE_4K_BUFFER_SCHEME = 1'b1;  // Set to 1 for 4K mode, 0 for legacy
    
    // 4K Buffer Scheme Parameters
    localparam logic [31:0] ROCE_BUF_ADDR_LSB    = 32'h0010_0000;   // Base VA lower 32 bits (128-byte aligned)
    localparam logic [31:0] ROCE_BUF_ADDR_MSB    = 32'h0000_0000;   // Base VA upper 32 bits
    localparam logic [25:0] ROCE_BUF_INC         = 26'h000_0300;    // Buffer increment in PAGE units (128B each)
                                                                    // 0x300 pages = 0x18000 bytes = 96KB per buffer
    localparam logic [11:0] ROCE_BUF_PTR_START   = 12'd0;           // Starting buffer index
    localparam logic [11:0] ROCE_BUF_PTR_END     = 12'd9;           // Ending buffer index (10 buffers total)
    
    // Legacy 4-buffer scheme configuration (for backward compatibility)
    localparam logic [31:0] ROCE_VADDR_WORD_BASE    = 32'h0000_2000; // 0x0010_0000 bytes
    localparam logic [31:0] ROCE_VADDR_FRAME_STRIDE = 32'h0000_0200; // 0x0001_0000 bytes
    localparam int          ROCE_VADDR_COUNT        = 4;

    //==========================================================================
    // Hololink IP Register Address Map (Based on Hardware Specification)
    //==========================================================================
    // These addresses access Hololink IP configuration registers via ECB protocol
    // Addresses are fixed by hardware design - do not modify unless hardware changes
    
    // System Control Registers (Address Range: 0x0000-0x00FF)
    localparam logic [31:0] REG_SYS_CTRL    = 32'h00000000; // System enable/control
    localparam logic [31:0] REG_SYS_STATUS  = 32'h00000004; // System status readback
    localparam logic [31:0] REG_SYS_VERSION = 32'h00000008; // Hardware version info

    // Primary RoCE Configuration Registers (Address Range: 0x1000-0x1FFF)
    // These registers configure the main RoCE data path for sensor transmission
    localparam logic [31:0] REG_DP_ROCE_0_DEST_QP        = 32'h00001000; // Destination queue pair + protocol mode
    localparam logic [31:0] REG_DP_ROCE_0_RKEY           = 32'h00001004; // Remote memory access key
    
    // 4K Buffer Scheme Registers (new mode)
    localparam logic [31:0] REG_DP_ROCE_0_BUF_ADDR_LSB   = 32'h00001008; // Base VA lower 32 bits
    localparam logic [31:0] REG_DP_ROCE_0_BUF_ADDR_MSB   = 32'h0000100C; // Base VA upper 32 bits
    localparam logic [31:0] REG_DP_ROCE_0_BUF_INC        = 32'h00001010; // Buffer increment (PAGE units)
    localparam logic [31:0] REG_DP_ROCE_0_BUF_PTR        = 32'h00001014; // {ptr_start[27:16], ptr_end[11:0]}
    localparam logic [31:0] REG_DP_ROCE_0_BUF_LEN        = 32'h00001018; // Buffer size (must = frame size)
    
    // Legacy 4-Buffer Scheme Registers (for backward compatibility)
    localparam logic [31:0] REG_DP_ROCE_0_VADDR_0        = 32'h00001008; // Virtual address buffer 0 (>>7 shifted)
    localparam logic [31:0] REG_DP_ROCE_0_VADDR_1        = 32'h0000100C; // Virtual address buffer 1 (>>7 shifted)
    localparam logic [31:0] REG_DP_ROCE_0_VADDR_2        = 32'h00001010; // Virtual address buffer 2 (>>7 shifted)
    localparam logic [31:0] REG_DP_ROCE_0_VADDR_3        = 32'h00001014; // Virtual address buffer 3 (>>7 shifted)
    localparam logic [31:0] REG_DP_ROCE_0_BUF_MASK       = 32'h0000101C; // Buffer enable mask (bit per buffer)
    
    // Common Registers (both modes)
    localparam logic [31:0] REG_DP_ROCE_0_HOST_MAC_LO    = 32'h00001020; // Destination MAC address [31:0]
    localparam logic [31:0] REG_DP_ROCE_0_HOST_MAC_HI    = 32'h00001024; // Destination MAC address [47:32]
    localparam logic [31:0] REG_DP_ROCE_0_HOST_IP        = 32'h00001028; // Destination IP address
    localparam logic [31:0] REG_DP_ROCE_0_HOST_UDP_PORT  = 32'h0000102C; // Destination UDP port
    localparam logic [31:0] REG_DP_ROCE_0_PSN            = 32'h0000103C; // RoCE packet sequence number
    
    // Note: Register addresses overlap because 4K and legacy modes use same hardware addresses
    // but interpret the data differently

    // Packet Processing Configuration Registers (Address Range: 0x2000000+)
    // These registers control how sensor data is packaged into Ethernet frames
    localparam logic [31:0] REG_DP_PKT_SCRATCH       = 32'h02000300; // Scratch register for testing
    localparam logic [31:0] REG_DP_PKT_LEN           = 32'h02000304; // Ethernet packet payload length
    localparam logic [31:0] REG_DP_PKT_FPGA_UDP_PORT = 32'h02000308; // FPGA source UDP port + protocol flag
    localparam logic [31:0] REG_DP_PKT_VIP_MASK      = 32'h0200030C; // Virtual port enable mask

    // Instance Decoder Registers (Address Range: 0x2000000+)
    // Configure instance-specific settings
    localparam logic [31:0] REG_INST_DEC_0_ECB_UDP_PORT = 32'h02000000; // ECB command protocol port

    // Sensor Control Registers (Address Range: 0x1000000+) - Optional Use
    // Note: These registers may conflict with direct sensor interface stimulus
    localparam logic [31:0] REG_PACK_CTRL_BASE    = 32'h01000000; // Packetizer control base
    localparam logic [31:0] REG_SENSOR_ENABLE     = 32'h01000000; // Sensor enable control
    localparam logic [31:0] REG_SENSOR_FRAME_RATE = 32'h01000004; // Sensor frame rate setting
    localparam logic [31:0] REG_SENSOR_RESOLUTION = 32'h01000008; // Sensor resolution setting

    //==========================================================================
    // Utility Constants  
    //==========================================================================
    localparam int HLNK_USER_EVT_WIDTH = 16; // User event width definition

    //==========================================================================
    // Ethernet/IP/UDP Header Sizes and Byte Offsets (avoid magic numbers)
    //==========================================================================
    // Ethernet II header (bytes)
    localparam int ETH_HEADER_SIZE           = 14;
    localparam int ETH_DST_MAC_OFFSET        = 0;   // 6 bytes
    localparam int ETH_SRC_MAC_OFFSET        = 6;   // 6 bytes
    localparam int ETH_ETHERTYPE_OFFSET      = 12;  // 2 bytes

    // EtherType values
    localparam logic [15:0] ETHER_TYPE_IPV4  = 16'h0800;
    localparam logic [15:0] ETHER_TYPE_1722B = IEEE_1722B_ETHERTYPE;

    // IPv4 header (bytes)
    localparam int IP_HEADER_SIZE                = 20; // without options
    localparam int IP_OFFSET_FROM_ETH            = ETH_HEADER_SIZE;
    localparam int IP_VERSION_IHL_OFFSET         = 0;  // 1 byte
    localparam int IP_DSCP_ECN_OFFSET            = 1;  // 1 byte
    localparam int IP_TOTAL_LENGTH_OFFSET        = 2;  // 2 bytes
    localparam int IP_IDENTIFICATION_OFFSET      = 4;  // 2 bytes
    localparam int IP_FLAGS_FRAG_OFFSET          = 6;  // 2 bytes
    localparam int IP_TTL_OFFSET                 = 8;  // 1 byte
    localparam int IP_PROTOCOL_OFFSET            = 9;  // 1 byte
    localparam int IP_HEADER_CHECKSUM_OFFSET     = 10; // 2 bytes
    localparam int IP_SRC_ADDR_OFFSET            = 12; // 4 bytes
    localparam int IP_DST_ADDR_OFFSET            = 16; // 4 bytes

    // Common IPv4 field defaults used by TB
    localparam logic [7:0] IPV4_VERSION_IHL_DEFAULT = 8'h45; // v4, IHL=5 words
    localparam logic [7:0] IPV4_DSCP_ECN_DEFAULT    = 8'h00;
    localparam logic [7:0] IPV4_TTL_DEFAULT         = 8'h40; // 64
    localparam logic [7:0] IPV4_PROTOCOL_UDP        = 8'h11; // UDP

    // ECB payload sizes and offsets within UDP payload (bytes)
    localparam int ECB_RD_PAYLOAD_LEN       = 10; // CMD(1)+FLAGS(1)+SEQ(2)+RSVD(2)+ADDR(4)
    localparam int ECB_WR_PAYLOAD_LEN       = 14; // RD payload + DATA(4)
    localparam int ECB_RD_RESP_LEN          = 14; // RD response includes DATA(4)
    localparam int ECB_CMD_OFFSET           = 0;
    localparam int ECB_FLAGS_OFFSET         = 1;
    localparam int ECB_SEQ_OFFSET           = 2;  // 2 bytes (MSB first)
    localparam int ECB_RSVD_OFFSET          = 4;  // 2 bytes
    localparam int ECB_ADDR_OFFSET          = 6;  // 4 bytes
    localparam int ECB_DATA_OFFSET          = 10; // 4 bytes (present in WR payload and RD response)

    // UDP header (bytes)
    localparam int UDP_HEADER_SIZE              = 8;
    localparam int UDP_OFFSET_FROM_IP           = IP_HEADER_SIZE;
    localparam int UDP_OFFSET_FROM_ETH          = ETH_HEADER_SIZE + IP_HEADER_SIZE;
    localparam int UDP_SRC_PORT_OFFSET          = 0; // 2 bytes
    localparam int UDP_DST_PORT_OFFSET          = 2; // 2 bytes
    localparam int UDP_LENGTH_OFFSET            = 4; // 2 bytes
    localparam int UDP_CHECKSUM_OFFSET          = 6; // 2 bytes

    // Payload offset from Ethernet start
    localparam int UDP_PAYLOAD_OFFSET_FROM_ETH  = ETH_HEADER_SIZE + IP_HEADER_SIZE + UDP_HEADER_SIZE;

    // Minimum Ethernet frame size (without FCS)
    localparam int MIN_ETH_FRAME_NO_FCS_BYTES   = 60; // 64B incl. FCS -> 60B without

    //==========================================================================
    // Protocol Detection Helper Functions  
    //==========================================================================
    // These functions help identify different protocol types in network traffic
    // Useful for packet monitoring and traffic classification
    
    //--------------------------------------------------------------------------
    // is_ecb_response: Determine if ECB command is a response packet
    //
    // PARAMETERS:
    //   cmd_code - 8-bit ECB command code from packet header
    //
    // RETURNS:
    //   1 if this is a response packet (bit 7 set), 0 if command packet
    //
    // USAGE:
    //   if (is_ecb_response(packet_cmd)) $display("Got ECB response");
    //--------------------------------------------------------------------------
    function automatic logic is_ecb_response(logic [7:0] cmd_code);
        return cmd_code[7]; // Response commands have bit 7 set
    endfunction

    //--------------------------------------------------------------------------
    // is_valid_ecb_cmd: Validate ECB command codes
    //
    // PARAMETERS: 
    //   cmd_code - 8-bit command code to validate
    //
    // RETURNS:
    //   1 if valid ECB command/response, 0 if invalid
    //--------------------------------------------------------------------------
    function automatic logic is_valid_ecb_cmd(logic [7:0] cmd_code);
        return ((cmd_code == ECB_CMD_WR_DWORD) || (cmd_code == ECB_CMD_WR_DWORD_RESP) ||
                (cmd_code == ECB_CMD_RD_DWORD) || (cmd_code == ECB_CMD_RD_DWORD_RESP));
    endfunction
    
    //--------------------------------------------------------------------------
    // is_bootp_ports: Identify BOOTP protocol traffic
    //
    // PARAMETERS:
    //   src_port, dst_port - UDP source and destination ports
    //
    // RETURNS: 
    //   1 if packet uses BOOTP protocol ports, 0 otherwise
    //--------------------------------------------------------------------------
    function automatic logic is_bootp_ports(logic [15:0] src_port, logic [15:0] dst_port);
        return ((src_port == ETH_UDP_BOOTP_SRC_PORT && dst_port == ETH_UDP_BOOTP_DST_PORT) ||
                (src_port == ETH_UDP_BOOTP_DST_PORT && dst_port == ETH_UDP_BOOTP_SRC_PORT));
    endfunction

    //--------------------------------------------------------------------------
    // is_ecb_ports: Identify ECB (register access) traffic
    //
    // PARAMETERS:
    //   src_port, dst_port - UDP source and destination ports  
    //
    // RETURNS:
    //   1 if packet uses ECB command port, 0 otherwise
    //--------------------------------------------------------------------------
    function automatic logic is_ecb_ports(logic [15:0] src_port, logic [15:0] dst_port);
        return (dst_port == ETH_UDP_CMD_PORT || src_port == ETH_UDP_CMD_PORT);
    endfunction

    //==========================================================================
    // Standardized Logging Functions
    //==========================================================================
    
    // Log levels for standardized output
    typedef enum {
        LOG_INFO,
        LOG_WARN, 
        LOG_ERR,
        LOG_PASS,
        LOG_FAIL
    } log_level_t;
    
    // Helper function to get log prefix string
    function automatic string get_log_prefix(log_level_t level);
        case (level)
            LOG_INFO: return "[INFO]";
            LOG_WARN: return "[WARN]";
            LOG_ERR:  return "[ERROR]";
            LOG_PASS: return "[PASS]";
            LOG_FAIL: return "[FAIL]";
            default:  return "[INFO]";
        endcase
    endfunction
    
    // Standardized logging macro (to be used as display statements)
    // Pass a single pre-formatted message (use $sformatf at call sites for arguments)
    `define TB_LOG(level, msg) $display("%s [%0t] %s", get_log_prefix(level), $time, msg)

    // Verbosity-aware logging helpers (lightweight gating around displays)
    // Gating uses package-scoped tb_current_verbosity; no local symbol needed
    `define TB_LOGV(min_v, level, msg) \
        if (hololink_tb_pkg::tb_current_verbosity >= min_v) $display("%s [%0t] %s", get_log_prefix(level), $time, msg)
    // Note: Use $sformatf(...) at call sites for formatted messages

    // Gate any statement under a verbosity threshold
    `define TB_LOG_WHEN(min_v, stmt) \
        if (hololink_tb_pkg::tb_current_verbosity >= min_v) stmt

    // Convenience wrappers for common thresholds
    `define TB_LOG_LOW(level, msg)     `TB_LOGV(VERBOSITY_LOW, level, msg)
    `define TB_LOG_MED(level, msg)     `TB_LOGV(VERBOSITY_MEDIUM, level, msg)
    `define TB_LOG_HIGH(level, msg)    `TB_LOGV(VERBOSITY_HIGH, level, msg)

    // Stream-style write macros removed; use $sformatf with `TB_LOG* macros instead.

endpackage : hololink_tb_pkg
