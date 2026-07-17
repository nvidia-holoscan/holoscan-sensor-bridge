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

`ifndef HOLOLINK_ECB_TASKS_SVH
`define HOLOLINK_ECB_TASKS_SVH

//==============================================================================
// ECB (Ethernet Control Bus) Protocol Task Library
//
// DESCRIPTION:
//   This file provides tasks for communicating with Hololink IP registers over
//   Ethernet using the ECB protocol. ECB enables register read/write operations
//   through standard UDP packets, allowing remote configuration and monitoring.
//
// KEY TASKS:
//   - ecb_write()     : Write 32-bit values to Hololink registers
//   - ecb_read()      : Read 32-bit values from Hololink registers  
//   - ECB response parsing and validation
//
// PROTOCOL OVERVIEW:
//   ECB packets contain: Command(1B) + Flags(1B) + Sequence(2B) + Address(4B) + Data(4B)
//   Responses have bit[7] set in command field and include status codes
//
// USAGE:
//   ecb_write(REG_SENSOR_CTRL, 32'h0000_0001);  // Enable sensor
//   ecb_read(REG_SENSOR_STATUS, read_value);    // Read sensor status
//==============================================================================

//==============================================================================  
// ECB Response Parser Task
//==============================================================================
//--------------------------------------------------------------------------
// parse_ecb_response: Parse and validate ECB response packets
//
// PARAMETERS:
//   frame_q - Byte queue containing complete Ethernet frame with ECB response
//
// DESCRIPTION:
//   Extracts ECB command, sequence ID, and status from response packets.
//   Provides detailed error reporting for failed register operations.
//   Updates global response tracking variables for use by ecb_read/write tasks.
//
// USAGE:
//   Called automatically by host traffic monitor when ECB responses are detected
//--------------------------------------------------------------------------
task parse_ecb_response(input logic [7:0] frame_q[$]);
    automatic int udp_payload_start;
    automatic logic [7:0] cmd_code;
    automatic logic [15:0] seq_id;
    automatic logic [7:0] resp_code;
    automatic logic [31:0] read_data;
    automatic string cmd_name;
    automatic string resp_status;
    
    // UDP payload start offset from Ethernet
    udp_payload_start = UDP_PAYLOAD_OFFSET_FROM_ETH;
    
    if (frame_q.size() > udp_payload_start + ECB_DATA_OFFSET) begin
        cmd_code = frame_q[udp_payload_start];
        seq_id = {frame_q[udp_payload_start + ECB_SEQ_OFFSET], frame_q[udp_payload_start + ECB_SEQ_OFFSET + 1]};
        resp_code = frame_q[udp_payload_start + ECB_RSVD_OFFSET];
        
        // Decode command name for readability
        if (cmd_code == ECB_CMD_WR_DWORD)            cmd_name = "WRITE";
        else if (cmd_code == ECB_CMD_WR_DWORD_RESP)  cmd_name = "WRITE_RESP";
        else if (cmd_code == ECB_CMD_RD_DWORD)       cmd_name = "READ";
        else if (cmd_code == ECB_CMD_RD_DWORD_RESP)  cmd_name = "READ_RESP";
        else                                          cmd_name = "UNKNOWN";

        // Decode response status for readability
        if (resp_code == 8'h00)      resp_status = "SUCCESS";
        else if (resp_code == 8'h01) resp_status = "INVALID_CMD";
        else if (resp_code == 8'h02) resp_status = "INVALID_ADDR";  
        else if (resp_code == 8'h03) resp_status = "ACCESS_ERROR";
        else if (resp_code == 8'h04) resp_status = "INVALID_DATA";
        else if (resp_code == 8'h05) resp_status = "TIMEOUT";
        else                         resp_status = $sformatf("UNKNOWN_0x%02x", resp_code);

        // Enhanced logging with original transaction details for errors
        if (resp_code != 8'h00) begin
            if (last_ecb_was_read) begin
                `TB_LOG_LOW(LOG_ERR, $sformatf("ECB Response: cmd=%s (0x%02x), seq=0x%04x, status=%s (0x%02x)", 
                         cmd_name, cmd_code, seq_id, resp_status, resp_code));
                `TB_LOG_LOW(LOG_ERR, $sformatf("  *** ERROR DETAILS: Original READ transaction addr=0x%08x ***", 
                         last_ecb_addr));
            end else begin
                `TB_LOG_LOW(LOG_ERR, $sformatf("ECB Response: cmd=%s (0x%02x), seq=0x%04x, status=%s (0x%02x)", 
                         cmd_name, cmd_code, seq_id, resp_status, resp_code));
                `TB_LOG_LOW(LOG_ERR, $sformatf("  *** ERROR DETAILS: Original write transaction addr=0x%08x, data=0x%08x ***", 
                         last_ecb_addr, last_ecb_data));
            end
        end else begin
            `TB_LOG_LOW(LOG_INFO, $sformatf("ECB Response: cmd=%s (0x%02x), seq=0x%04x, status=%s (0x%02x)", 
                     cmd_name, cmd_code, seq_id, resp_status, resp_code));
        end
        
        // Handle both read and write responses
        if (cmd_code == ECB_CMD_RD_DWORD_RESP && frame_q.size() >= udp_payload_start + ECB_RD_RESP_LEN) begin
            // Extract 32-bit read data (big-endian)
            read_data = {frame_q[udp_payload_start + ECB_DATA_OFFSET], 
                        frame_q[udp_payload_start + ECB_DATA_OFFSET + 1],
                        frame_q[udp_payload_start + ECB_DATA_OFFSET + 2], 
                        frame_q[udp_payload_start + ECB_DATA_OFFSET + 3]};
            
            `TB_LOG_LOW(LOG_INFO, $sformatf("ECB Read Response: addr=0x%08x, data=0x%08x", 
                     {frame_q[udp_payload_start + ECB_ADDR_OFFSET], frame_q[udp_payload_start + ECB_ADDR_OFFSET + 1],
                            frame_q[udp_payload_start + ECB_ADDR_OFFSET + 2], frame_q[udp_payload_start + ECB_ADDR_OFFSET + 3]}, read_data));
            
            // Store response if sequence ID matches expected
            if (expected_response_seq == seq_id) begin
                last_read_data = read_data;
                response_ready = 1;
                expected_response_seq = -1;
                `TB_LOG_LOW(LOG_INFO, $sformatf("Stored read response data: 0x%08x", read_data));
            end
        end else if (cmd_code == ECB_CMD_WR_DWORD_RESP) begin
            // Write ACK response
            `TB_LOG_LOW(LOG_INFO, $sformatf("ECB Write ACK: addr=0x%08x, status=%s", 
                     {frame_q[udp_payload_start + ECB_ADDR_OFFSET], frame_q[udp_payload_start + ECB_ADDR_OFFSET + 1],
                            frame_q[udp_payload_start + ECB_ADDR_OFFSET + 2], frame_q[udp_payload_start + ECB_ADDR_OFFSET + 3]}, resp_status));
            
            // Signal response received for write operations
            if (expected_response_seq == seq_id) begin
                response_ready = 1;
                expected_response_seq = -1;
                `TB_LOG_LOW(LOG_INFO, $sformatf("Write ACK processed for seq=0x%04x", seq_id));
            end
        end
    end
endtask

//==============================================================================
// ECB Write Task - Send register write command via ethernet
//==============================================================================
task ecb_write(input [31:0] addr, input [31:0] data);
    logic [7:0] ecb_payload[];
    logic [7:0] eth_frame[];
    int eth_frame_size;
    string frame_hex;
    
    `TB_LOG_LOW(LOG_INFO, $sformatf("==> ECB WRITE START: addr=0x%08x, data=0x%08x", addr, data));
    `TB_LOG_LOW(LOG_INFO, $sformatf("    Current sequence ID: %0d", ecb_sequence_id));
    
    // Build ECB payload according to ECB specification
    // ECB Write Format: CMD[1] + FLAGS[1] + SEQ[2] + RSVD[2] + ADDR[4] + DATA[4] = 14 bytes
    ecb_payload = new[14];
    ecb_payload[0] = ECB_CMD_WR_DWORD;          // Command: Write DWORD
    ecb_payload[1] = 8'h01;                      // Flags: ACK required
    ecb_payload[2] = ecb_sequence_id[15:8];      // Sequence MSB
    ecb_payload[3] = ecb_sequence_id[7:0];       // Sequence LSB  
    ecb_payload[4] = 8'h00; ecb_payload[5] = 8'h00; // Reserved
    ecb_payload[6] = addr[31:24];                // Address (big-endian)
    ecb_payload[7] = addr[23:16];
    ecb_payload[8] = addr[15:8];
    ecb_payload[9] = addr[7:0];
    ecb_payload[10] = data[31:24];               // Data (big-endian)
    ecb_payload[11] = data[23:16];
    ecb_payload[12] = data[15:8];
    ecb_payload[13] = data[7:0];
    
    `TB_LOG_LOW(LOG_INFO, $sformatf("    ECB payload (14 bytes): CMD=0x%02x FLAGS=0x%02x SEQ=0x%04x ADDR=0x%08x DATA=0x%08x", 
             ecb_payload[0], ecb_payload[1], 
             {ecb_payload[2], ecb_payload[3]}, 
             {ecb_payload[6], ecb_payload[7], ecb_payload[8], ecb_payload[9]},
             {ecb_payload[10], ecb_payload[11], ecb_payload[12], ecb_payload[13]}));
    
    // Build complete Ethernet frame with correct addresses and ports
    build_ethernet_frame(ecb_payload, 14, 
                        DEFAULT_FPGA_MAC,      // dst_mac (FPGA)
                        DEFAULT_HOST_MAC,      // src_mac (Host)
                        DEFAULT_FPGA_IP,       // dst_ip (FPGA) 
                        DEFAULT_HOST_IP,       // src_ip (Host)
                        ETH_UDP_CMD_PORT,      // dst_port (ECB command port)
                        ETH_UDP_CMD_PORT,      // src_port
                        eth_frame, eth_frame_size);
    
    `TB_LOG_LOW(LOG_INFO, $sformatf("    Complete Ethernet frame size: %0d bytes", eth_frame_size));
    
    // Print first 64 bytes of frame for debugging (single consolidated line)
    frame_hex = "";
    for (int i = 0; i < (eth_frame_size < 64 ? eth_frame_size : 64); i++) begin
        if (i % 16 == 0 && i > 0) frame_hex = {frame_hex, "\n               "};
        frame_hex = {frame_hex, $sformatf("%02h ", eth_frame[i])};
    end
    `TB_LOG_MED(LOG_INFO, $sformatf("    Frame:\n               %s", frame_hex));
    
    // Send complete Ethernet frame
    send_eth_packet(0, eth_frame, eth_frame_size);
    
    ecb_sequence_id++;
    `TB_LOG_LOW(LOG_INFO, $sformatf("    Incremented sequence ID to: %0d", ecb_sequence_id));
    
    // Track transaction for enhanced error reporting
    last_ecb_addr = addr;
    last_ecb_data = data;
    last_ecb_was_read = 0;
    last_ecb_seq = ecb_sequence_id - 1;
    
    // Wait for ACK response with timeout (proper ECB protocol)
    expected_response_seq = ecb_sequence_id - 1; // We already incremented it
    response_ready = 0;
    
    fork
        begin
            // Wait for ACK response
            while (!response_ready && expected_response_seq != -1) begin
                @(posedge hif_clk);
            end
            if (response_ready) begin
                `TB_LOG_LOW(LOG_INFO, $sformatf("ECB write ACK received for addr=0x%08x", addr));
            end else begin
                `TB_LOG_LOW(LOG_WARN, $sformatf("ECB write ACK timeout for addr=0x%08x", addr));
            end
        end
        begin
            // Timeout after 1000 cycles
            repeat(1000) @(posedge hif_clk);
            if (expected_response_seq != -1) begin
                `TB_LOG_LOW(LOG_WARN, $sformatf("ECB write ACK timeout for addr=0x%08x!", addr));
                expected_response_seq = -1;
            end
        end
    join_any
    disable fork;
    
    `TB_LOG_LOW(LOG_INFO, $sformatf("<== ECB WRITE COMPLETE: addr=0x%08x", addr));
    
endtask

//==============================================================================
// ECB Read Task - Send register read command via ethernet
//==============================================================================
task ecb_read(input [31:0] addr, output [31:0] data);
    // Variable declarations at beginning
    logic [7:0] ecb_cmd_data[];
    logic [7:0] eth_frame[];
    int eth_frame_size;
    
    `TB_LOG_LOW(LOG_INFO, $sformatf("ECB READ: addr=0x%08x", addr));
    
    // Build ECB read payload according to ECB specification
    // ECB Read Format: CMD[1] + FLAGS[1] + SEQ[2] + RSVD[2] + ADDR[4] = 10 bytes
    ecb_cmd_data = new[10];
    ecb_cmd_data[0] = ECB_CMD_RD_DWORD;          // Command: Read DWORD
    ecb_cmd_data[1] = 8'h01;                      // Flags: ACK required
    ecb_cmd_data[2] = ecb_sequence_id[15:8];      // Sequence MSB
    ecb_cmd_data[3] = ecb_sequence_id[7:0];       // Sequence LSB
    ecb_cmd_data[4] = 8'h00; ecb_cmd_data[5] = 8'h00; // Reserved
    ecb_cmd_data[6] = addr[31:24];                // Address (big-endian)
    ecb_cmd_data[7] = addr[23:16];
    ecb_cmd_data[8] = addr[15:8];
    ecb_cmd_data[9] = addr[7:0];
    
    `TB_LOG_LOW(LOG_INFO, $sformatf("    ECB payload (10 bytes): CMD=0x%02x FLAGS=0x%02x SEQ=0x%04x ADDR=0x%08x", 
             ecb_cmd_data[0], ecb_cmd_data[1], 
             {ecb_cmd_data[2], ecb_cmd_data[3]}, 
             {ecb_cmd_data[6], ecb_cmd_data[7], ecb_cmd_data[8], ecb_cmd_data[9]}));
    
    // Build complete Ethernet frame and send it  
    build_ethernet_frame(ecb_cmd_data, 10,
                        DEFAULT_FPGA_MAC,      // dst_mac (FPGA)
                        DEFAULT_HOST_MAC,      // src_mac (Host)
                        DEFAULT_FPGA_IP,       // dst_ip (FPGA)
                        DEFAULT_HOST_IP,       // src_ip (Host) 
                        ETH_UDP_CMD_PORT,      // dst_port (ECB command port)
                        ETH_UDP_CMD_PORT,      // src_port
                        eth_frame, eth_frame_size);
    
    `TB_LOG_LOW(LOG_INFO, $sformatf("    Complete Ethernet frame size: %0d bytes", eth_frame_size));
    send_eth_packet(0, eth_frame, eth_frame_size);
    
    ecb_sequence_id++;
    `TB_LOG_LOW(LOG_INFO, $sformatf("    Incremented sequence ID to: %0d", ecb_sequence_id));
    
    // Wait for response with timeout
    expected_response_seq = ecb_sequence_id - 1; // We already incremented it
    response_ready = 0;
    
    // Wait for response with timeout
    fork
        begin
            // Wait for response
            while (!response_ready && expected_response_seq != -1) begin
                @(posedge hif_clk);
            end
            if (response_ready) begin
                data = last_read_data;
                `TB_LOG_LOW(LOG_INFO, $sformatf("ECB read got response: 0x%08x", data));
            end else begin
                data = 32'hDEADBEEF;
                `TB_LOG_LOW(LOG_WARN, "ECB read timeout - returning dummy data");
            end
        end
        begin
            // Timeout after 1000 cycles
            repeat(1000) @(posedge hif_clk);
            if (expected_response_seq != -1) begin
                `TB_LOG_LOW(LOG_WARN, "ECB read timeout!");
                expected_response_seq = -1;
                data = 32'hDEADBEEF;
            end
        end
    join_any
    disable fork;
    
    `TB_LOG_LOW(LOG_INFO, $sformatf("<== ECB READ COMPLETE: addr=0x%08x, data=0x%08x", addr, data));
    
endtask

`endif // HOLOLINK_ECB_TASKS_SVH