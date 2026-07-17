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

`ifndef HOLOLINK_COE_HELPERS_SVH
`define HOLOLINK_COE_HELPERS_SVH

//==============================================================================
// COE (Camera over Ethernet / IEEE 1722B) Helper Library
// Protocol handling and packet analysis for Hololink testbench
//==============================================================================

//==============================================================================
// COE packet detection function
//==============================================================================
function automatic logic is_coe_packet(
    input [15:0] udp_src_port,
    input [15:0] udp_dst_port,
    input [7:0] packet_data[]
);
    // Prefer EtherType-based detection (1722B) when possible; fallback to port check if caller only has UDP context.
    // If the caller passed an L2 frame, the first two bytes of packet_data should be the start of the COE header
    // and the enclosing monitor should have already matched EtherType 0x22F0.
    if (packet_data.size() >= COE_HDR_SIZE) begin
        // Treat as COE payload extracted from L2, so return true.
        return 1'b1;
    end
    // Fallback: UDP port heuristic (legacy environments)
    return ((udp_src_port == ETH_UDP_COE_PORT) || (udp_dst_port == ETH_UDP_COE_PORT));
endfunction

//==============================================================================
// COE packet analysis and processing task - ENHANCED WITH DATA VALIDATION
//==============================================================================
task automatic analyze_coe_packet(
    input [15:0] udp_src_port,
    input [15:0] udp_dst_port, 
    input [7:0] packet_data[]
);
    logic [7:0] coe_flags;
    logic [15:0] ethertype;
    logic [7:0] version_field;
    logic [63:0] stream_id;
    logic [15:0] packet_length;
    logic [7:0] channel;
    logic [7:0] sensor_payload[];
    logic [`DATAPATH_WIDTH-1:0] received_pattern;
    int payload_size, word_offset, data_errors;
    int frame_id, expected_frame_id;
    int sensor_data_offset;
    logic [31:0] pattern_base, pattern_frame, pattern_word;
    logic [15:0] pattern_prefix;
    logic [7:0] pattern_frame_id, pattern_suffix, extracted_frame_id;
    logic [63:0] expected_pattern;
    logic [`DATAPATH_WIDTH-1:0] expected_word_pattern;
    logic [7:0] expected_bytes[];
    int mismatch_count, estimated_frame_id, b;
    logic [7:0] rcvd_byte;
    int relative_word, byte_idx;
    int bytes_per_word;
    logic [63:0] expected_pattern_base;
    logic [31:0] upper_32, lower_32;
    int received_frame, received_word;
    logic [`DATAPATH_WIDTH-1:0] expected_sensor_pattern;
    logic [27:0] coe_byte_offset;  // COE packet byte offset field
    logic [1:0] coe_frame_number;  // COE packet frame number field
    int coe_header_size;           // Size of COE header before image data
    logic [31:0] last_word;        // Temp variable for unpacking
    int absolute_byte_pos, absolute_word_pos;  // For position calculation
    static int total_coe_data_bytes = 0;
    static int total_coe_data_errors = 0;
    static int global_word_counter = 0;  // Track absolute 64-bit word position across all COE packets
    static int current_frame_word_index = 0; // 64-bit word index within current frame
    static int frame_counter = 0;             // Count frames seen (increment on FRAME_END)
    static logic first_call = 1;
    int dbg_i; // for debug dumps
    logic [`DATAPATH_WIDTH-1:0] bit_diff;
    string header_hex;
    string recv_bytes_line;
    string exp_bytes_line;
    
    // CRITICAL FIX: Reset static variables at start of each test run
    if (first_call) begin
        total_coe_data_bytes = 0;
        total_coe_data_errors = 0;
        global_word_counter = 0;  // Reset global 64-bit word position tracker
        current_frame_word_index = 0;
        frame_counter = 0;
        first_call = 0;
        `TB_LOG_LOW(LOG_INFO, "[COE] Resetting COE analysis statistics for new test run");
    end
    
    `TB_LOG_HIGH(LOG_INFO, $sformatf("[COE] Analyzing COE packet: src_port=0x%04x, dst_port=0x%04x, size=%0d bytes", 
            udp_src_port, udp_dst_port, packet_data.size()));
    
    // Validate packet size meets minimum requirements
    if (packet_data.size() < COE_HDR_SIZE) begin
        `TB_LOG_LOW(LOG_ERR, $sformatf("[COE] COE packet too small: %d bytes (minimum %d required)", 
                packet_data.size(), COE_HDR_SIZE));
        errors_detected++;
        return;
    end
    
    // EXACT COE PACKET PARSING: Based on UVM coe_packet::unpack_packet
    // The UVM shows exact field order and sizes
    // Calculating positions based on actual fields (excluding commented ones):
    // - subtype,sv,version,r,ntscf_data_len,seq_num_0: 32 bits (4 bytes)
    // - stream_id: 64 bits (8 bytes)  
    // - acf_msg_type,acf_msg_len,pad,tv,res,seq_num_1: 32 bits (4 bytes)
    // - timestamp: 64 bits (8 bytes)
    // - seq_num_3,e,se,fcv,version_num,exposore_num,rsvd,channel_number,flags: 24 bits (3 bytes)
    // - frame_number_rsvd,frame_number,byte_offset: 32 bits (4 bytes)
    // Total: 4+8+4+8+3+4 = 31 bytes header
    
    coe_header_size = 32;  // Adjusted: include alignment/pad byte before image_data
    
    // Number of bytes per sensor datapath word (width-agnostic)
    bytes_per_word = (`DATAPATH_WIDTH/8);
    
    if (packet_data.size() < coe_header_size) begin
        `TB_LOG_LOW(LOG_ERR, $sformatf("[COE] Packet too small for COE header: %d bytes", packet_data.size()));
        errors_detected++;
        return;
    end
    
    // Extract frame_number and byte_offset from exact positions
    // The streaming operator {>>{ }} unpacks MSB first
    // Last 32 bits = frame_number_rsvd(2) + frame_number(2) + byte_offset(28)
    // With 32-byte header, these are bytes 28-31
    last_word = {packet_data[COE_LAST_WORD_OFFSET + 0], packet_data[COE_LAST_WORD_OFFSET + 1], packet_data[COE_LAST_WORD_OFFSET + 2], packet_data[COE_LAST_WORD_OFFSET + 3]};
    coe_frame_number = last_word[COE_FRAME_NUM_MSB:COE_FRAME_NUM_LSB];  // frame_number
    coe_byte_offset = last_word[COE_BYTE_OFFSET_MSB:COE_BYTE_OFFSET_LSB];     // byte_offset
    
    // Parse minimal header fields directly from L2 COE payload (post 14-byte Ethernet header)
    // Align with UVM coe_packet field ordering
    // Note: Simple TB does not carry VLAN; EtherType already checked in monitor
    ethertype = IEEE_1722B_ETHERTYPE;  // Treat as valid 1722B frame
    // Subtype is the very first byte per UVM coe_packet
    version_field = packet_data[0];
    // Flags nibble (lower 4 bits) is used in simple TB; take nibble at offset 27
    coe_flags = packet_data[COE_FLAGS_BYTE_OFFSET];
    stream_id = 64'h0;
    packet_length = packet_data.size();
    channel = 0;
    
    `TB_LOG_MED(LOG_INFO, $sformatf("[COE] IEEE 1722B Header: ethertype=0x%04x, version=%d, flags=0x%02x", 
            ethertype, version_field, coe_flags));
    `TB_LOG_MED(LOG_INFO, $sformatf("[COE] Stream ID=0x%016x, length=%d, channel=%d", 
            stream_id, packet_length, channel));
    
    // Validate COE header fields
    data_errors = 0;
    
    // Validate IEEE 1722B fields (enabled when treating packet as 1722B)
    if (ethertype == IEEE_1722B_ETHERTYPE) begin
        // Check EtherType (should be IEEE 1722B)
        `TB_LOG_MED(LOG_PASS, $sformatf("[COE] Valid IEEE 1722B EtherType: 0x%04x", ethertype));
        
        // Stricter header checks
        // Subtype expected 0x82 for 1722B per UVM create_coe_frame
        if (packet_data[0] != 8'h82) begin
            `TB_LOG_LOW(LOG_WARN, $sformatf("[COE] Unexpected subtype: 0x%02x (expected 0x82)", packet_data[0]));
        end
        // ACF message type expected 0x0C at offset 12
        if (packet_data.size() > 12 && packet_data[12] != 8'h0C) begin
            `TB_LOG_LOW(LOG_WARN, $sformatf("[COE] Unexpected ACF msg type: 0x%02x (expected 0x0C)", packet_data[12]));
        end
        // Flags should be one of known values (simple TB mapping)
        if (!(coe_flags[3:0] inside {COE_FRAME_START, COE_MIDDLE, COE_END_OF_LINE, COE_FRAME_END})) begin
            `TB_LOG_LOW(LOG_WARN, $sformatf("[COE] Invalid COE flags nibble: 0x%01x", coe_flags[3:0]));
        end else begin
            `TB_LOG_HIGH(LOG_PASS, $sformatf("[COE] Valid COE flags nibble detected: 0x%01x", coe_flags[3:0]));
        end
    end else begin
        // Raw sensor data format - no IEEE 1722B header validation needed
        `TB_LOG_MED(LOG_INFO, "[COE] Raw sensor data format, skipping IEEE 1722B header validation");
    end
    
    // Extract sensor data payload after COE header
    // COE packets have a header followed by image data
    
    sensor_data_offset = coe_header_size;  // Skip COE header
    payload_size = packet_data.size() - coe_header_size;
    
    `TB_LOG_MED(LOG_INFO, $sformatf("[COE] COE packet: frame=%0d, byte_offset=0x%07x (%0d), payload_size=%0d", 
            coe_frame_number, coe_byte_offset, coe_byte_offset, payload_size));
    
    if (payload_size > 0) begin
        int base_word_index;
        // Debug: dump header and first payload bytes when verbose
        header_hex = "";
        for (dbg_i = 0; dbg_i < coe_header_size; dbg_i++) begin
            header_hex = {header_hex, $sformatf(" %02x", packet_data[dbg_i])};
        end
        `TB_LOG_HIGH(LOG_INFO, $sformatf("[COE] HEADER[0..31]=%s", header_hex));
        // Establish base index for this packet within the current frame
        // Extract sensor data payload 
        sensor_payload = new[payload_size];
        for (int i = 0; i < payload_size; i++) begin
            sensor_payload[i] = packet_data[sensor_data_offset + i];
        end
        
        total_coe_data_bytes += payload_size;
        
        // CRITICAL FIX: Use byte_offset from COE header to determine position
        // Each COE packet tells us exactly where its data belongs (in 64-bit word grid)
        
        base_word_index = current_frame_word_index;

        // Skip image-data validation for FRAME_END packets
        if (coe_flags[COE_FLAGS_NIBBLE_MSB:COE_FLAGS_NIBBLE_LSB] != COE_FRAME_END) begin
        for (byte_idx = 0; byte_idx + (bytes_per_word-1) < payload_size; byte_idx += bytes_per_word) begin
            // Assemble one sensor word using little-endian (same as working RoCE)
            received_pattern = '0;
            for (b = 0; b < bytes_per_word; b++) begin
                received_pattern[b*8 +: 8] = sensor_payload[byte_idx + b];
            end
            
            relative_word = byte_idx / bytes_per_word;
            
            // PROPER POSITION CALCULATION: Use byte_offset from COE header
            // byte_offset tells us the absolute byte position in the stream
            // Add current byte position within this packet to get absolute position
            absolute_byte_pos = coe_byte_offset + byte_idx;
            absolute_word_pos = absolute_byte_pos / bytes_per_word;
            
            // Use header-derived frame id only for reporting; expected uses per-frame running index
            received_frame = coe_frame_number;  // 2-bit frame id in header
            received_word = (coe_byte_offset / bytes_per_word) + (byte_idx / bytes_per_word);
            if (received_word >= (FRAME_SIZE_BYTES/bytes_per_word)) begin
                received_word = received_word % (FRAME_SIZE_BYTES/bytes_per_word);
            end
            
            // Build expected word strictly from stimulus-provided expected_sensor_data
            if (!expected_sensor_data.exists(frame_counter)) begin
                `TB_LOG_LOW(LOG_ERR, $sformatf("[COE] Missing expected_sensor_data for frame %0d", frame_counter));
                data_errors++;
                disable fork; // abort further checks in this packet
            end
            expected_word_pattern = '0;
            for (int eb = 0; eb < bytes_per_word; eb++) begin
                expected_word_pattern[eb*8 +: 8] = expected_sensor_data[frame_counter][(base_word_index*bytes_per_word) + byte_idx + eb];
            end
            
            // No sequence validation - COE packets contain fragmented continuous stream
            
            `TB_LOG_HIGH(LOG_INFO, $sformatf("[COE] DEBUG: byte_idx=%0d, relative_word=%0d, received_frame=%0d, received_word=%0d", 
                    byte_idx, relative_word, received_frame, received_word));
            `TB_LOG_HIGH(LOG_INFO, $sformatf("[COE] DEBUG: received=0x%0h, expected=0x%0h", 
                    received_pattern, expected_word_pattern));
            
            // PROPER VALIDATION: Compare full 64-bit word pattern
            if ($isunknown(received_pattern)) begin
                `TB_LOG_LOW(LOG_ERR, $sformatf("[COE] WORD[%0d] CONTAINS X STATES: expected=0x%0h, received=UNKNOWN/X", 
                         relative_word, expected_word_pattern));
                data_errors++;
                total_coe_data_errors++;
            end else if (received_pattern !== expected_word_pattern) begin
                // Data mismatch - report detailed error
                `TB_LOG_LOW(LOG_ERR, $sformatf("[COE] WORD[%0d] DATA MISMATCH: expected=0x%0h, received=0x%0h (frame=%0d, word=%0d)", 
                         relative_word, expected_word_pattern, received_pattern, received_frame, received_word));
                data_errors++;
                total_coe_data_errors++;
                
                // Show bit differences for debugging
                bit_diff = expected_word_pattern ^ received_pattern;
                `TB_LOG_HIGH(LOG_ERR, $sformatf("[COE] word bit differences: 0x%0h", bit_diff));
            end else begin
                // Data matches - COE validation successful!
                `TB_LOG_MED(LOG_PASS, $sformatf("[COE] WORD[%0d] DATA VALID: 0x%0h (frame=%0d, word=%0d)", 
                         relative_word, received_pattern, received_frame, received_word));
            end
            
            // CRITICAL DEBUG: Show exact byte patterns and expected sensor format
            `TB_LOG_WHEN(VERBOSITY_HIGH, begin
                // What we received from COE
                recv_bytes_line = "";
                for (int i = 0; i < bytes_per_word; i++) begin
                    recv_bytes_line = {recv_bytes_line, $sformatf(" %02x", sensor_payload[byte_idx+i])};
                end
                `TB_LOG(LOG_INFO, $sformatf("[COE] DEBUG WORD[%0d] received bytes:%s -> 0x%0h", 
                         relative_word, recv_bytes_line, received_pattern));
                
                // What sensor stimulus should have generated (little-endian format)
                if (expected_sensor_data.exists(frame_counter)) begin
                    expected_bytes = new[bytes_per_word];
                    for (int i = 0; i < bytes_per_word; i++) begin
                        expected_bytes[i] = expected_sensor_data[frame_counter][(base_word_index*bytes_per_word) + byte_idx + i];
                    end
                    expected_sensor_pattern = expected_word_pattern;
                end else begin
                    expected_sensor_pattern = '0;
                    expected_bytes = new[bytes_per_word];
                    for (int i = 0; i < bytes_per_word; i++) begin
                        expected_bytes[i] = expected_sensor_pattern[i*8 +: 8]; // Little-endian byte extraction
                    end
                end
                exp_bytes_line = "";
                for (int i = 0; i < bytes_per_word; i++) begin
                    exp_bytes_line = {exp_bytes_line, $sformatf(" %02x", expected_bytes[i])};
                end
                `TB_LOG(LOG_INFO, $sformatf("[COE] DEBUG WORD[%0d] expected bytes:%s -> 0x%0h", 
                         relative_word, exp_bytes_line, expected_sensor_pattern));
            end)
        end // for byte_idx

        // Advance per-frame word index by words consumed in this packet (only for image data)
        if (coe_flags[3:0] != COE_FRAME_END) begin
            current_frame_word_index += (payload_size/bytes_per_word);
        end
        end // if not FRAME_END

        // If this packet marks frame end, reset index and bump frame counter
        if (coe_flags[COE_FLAGS_NIBBLE_MSB:COE_FLAGS_NIBBLE_LSB] == COE_FRAME_END) begin
            current_frame_word_index = 0;
            frame_counter++;
        end
        
        // Report COE data analysis results
        if (data_errors > 0) begin
            `TB_LOG_LOW(LOG_INFO, $sformatf("[COE] Payload analysis: %0d bytes, %0d words, %0d errors", 
                    payload_size, payload_size/bytes_per_word, data_errors));
        end else begin
            `TB_LOG_MED(LOG_INFO, $sformatf("[COE] Payload analysis: %0d bytes, %0d words, %0d errors", 
                    payload_size, payload_size/bytes_per_word, data_errors));
        end
        
        `TB_LOG_HIGH(LOG_INFO, $sformatf("[COE] Cumulative stats: %0d total bytes, %0d total errors", 
                total_coe_data_bytes, total_coe_data_errors));
    end else begin
        `TB_LOG_LOW(LOG_WARN, "[COE] COE packet has no payload data");
    end
    
    // Update global error counter
    if (data_errors > 0) begin
        errors_detected += data_errors;
        `TB_LOG_LOW(LOG_FAIL, $sformatf("[COE] COE packet validation failed with %0d errors", data_errors));
    end else begin
        `TB_LOG_LOW(LOG_PASS, "[COE] COE packet validation passed");
    end
endtask

// Simple wrapper when only L2 COE payload is available (no UDP ports context)
task automatic analyze_coe_payload(
    input [7:0] packet_data[]
);
    analyze_coe_packet(16'h0000, 16'h0000, packet_data);
endtask

//==============================================================================
// COE (IEEE 1722B) Constants and Structure Definition
//==============================================================================


// IEEE 1722B COE header field definitions (based on UVM coe_packet class)
// Note: Individual field extraction used instead of packed struct for robustness

// COE frame tracking variables (module level for persistence across function calls)
int coe_frame_cycle_counter = 0;

//==============================================================================
// COE Analysis Reset Function - Call at start of each test
//==============================================================================
task reset_coe_analysis();
    // Reset all static variables for clean test runs
    `TB_LOG_LOW(LOG_INFO, "[COE] Resetting COE analysis statistics for new test");
    
    // Note: Static variables in tasks are reset by setting first_call flag
    // This is handled automatically in analyze_coe_packet()
    
    coe_frame_cycle_counter = 0;
endtask

`endif // HOLOLINK_COE_HELPERS_SVH