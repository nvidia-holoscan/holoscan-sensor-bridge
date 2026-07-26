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

`ifndef HOLOLINK_MONITORS_SVH
`define HOLOLINK_MONITORS_SVH

//==============================================================================
// Hololink Traffic Monitoring Library  
//
// DESCRIPTION:
//   Monitoring tasks for analyzing network traffic and data transmission
//   in Hololink IP testbenches. Includes packet classification, protocol
//   detection, and data validation.
//
// NOTES:
//   Output verbosity is controlled via testbench verbosity settings and +DUMP_PKTS.
//==============================================================================

//==============================================================================
// BOOTP Traffic Classification (port-first)
//==============================================================================
function automatic logic is_bootp_packet(logic [7:0] packet_bytes[$]);
    logic [15:0] udp_src_port, udp_dst_port;
    // Basic IPv4 UDP validation using package params
    if (packet_bytes.size() <= UDP_PAYLOAD_OFFSET_FROM_ETH ||
        {packet_bytes[ETH_ETHERTYPE_OFFSET], packet_bytes[ETH_ETHERTYPE_OFFSET+1]} != ETHER_TYPE_IPV4 ||
        packet_bytes[ETH_HEADER_SIZE + IP_PROTOCOL_OFFSET] != IPV4_PROTOCOL_UDP) begin
        return 0; // Not IPv4 UDP
    end
    udp_src_port = {packet_bytes[UDP_OFFSET_FROM_ETH + UDP_SRC_PORT_OFFSET],
                    packet_bytes[UDP_OFFSET_FROM_ETH + UDP_SRC_PORT_OFFSET + 1]};
    udp_dst_port = {packet_bytes[UDP_OFFSET_FROM_ETH + UDP_DST_PORT_OFFSET],
                    packet_bytes[UDP_OFFSET_FROM_ETH + UDP_DST_PORT_OFFSET + 1]};
    return is_bootp_ports(udp_src_port, udp_dst_port);
endfunction

//==============================================================================
// ECB Traffic Classification (port-first then command validation)
//==============================================================================
function automatic logic is_ecb_packet(logic [7:0] packet_bytes[$]);
    logic [15:0] udp_src_port, udp_dst_port;
    logic [7:0] cmd_code;
    // Basic IPv4 UDP validation using package params
    if (packet_bytes.size() <= UDP_PAYLOAD_OFFSET_FROM_ETH ||
        {packet_bytes[ETH_ETHERTYPE_OFFSET], packet_bytes[ETH_ETHERTYPE_OFFSET+1]} != ETHER_TYPE_IPV4 ||
        packet_bytes[ETH_HEADER_SIZE + IP_PROTOCOL_OFFSET] != IPV4_PROTOCOL_UDP) begin
        return 0; // Not IPv4 UDP
    end
    udp_src_port = {packet_bytes[UDP_OFFSET_FROM_ETH + UDP_SRC_PORT_OFFSET],
                    packet_bytes[UDP_OFFSET_FROM_ETH + UDP_SRC_PORT_OFFSET + 1]};
    udp_dst_port = {packet_bytes[UDP_OFFSET_FROM_ETH + UDP_DST_PORT_OFFSET],
                    packet_bytes[UDP_OFFSET_FROM_ETH + UDP_DST_PORT_OFFSET + 1]};
    // Port-first screening
    if (!is_ecb_ports(udp_src_port, udp_dst_port)) begin
        return 0;
    end
    // Then command validity (only when ports match)
    cmd_code = packet_bytes[UDP_PAYLOAD_OFFSET_FROM_ETH + ECB_CMD_OFFSET];
    return is_valid_ecb_cmd(cmd_code);
endfunction

//==============================================================================
// IPv4 UDP Basic Validation
//==============================================================================
function automatic logic is_ipv4_udp_packet(logic [7:0] packet_bytes[$]);
    if (packet_bytes.size() <= UDP_PAYLOAD_OFFSET_FROM_ETH) begin
        return 0;
    end
    if ({packet_bytes[ETH_ETHERTYPE_OFFSET], packet_bytes[ETH_ETHERTYPE_OFFSET+1]} != ETHER_TYPE_IPV4) begin
        return 0;
    end
    if (packet_bytes[ETH_HEADER_SIZE + IP_PROTOCOL_OFFSET] != IPV4_PROTOCOL_UDP) begin
        return 0;
    end
    return 1;
endfunction

//==============================================================================
// RoCE Traffic Classification (by UDP destination port)
//==============================================================================
function automatic logic is_roce_packet(logic [7:0] packet_bytes[$]);
    logic [15:0] udp_dst_port;
    if (packet_bytes.size() <= UDP_PAYLOAD_OFFSET_FROM_ETH ||
        {packet_bytes[ETH_ETHERTYPE_OFFSET], packet_bytes[ETH_ETHERTYPE_OFFSET+1]} != ETHER_TYPE_IPV4 ||
        packet_bytes[ETH_HEADER_SIZE + IP_PROTOCOL_OFFSET] != IPV4_PROTOCOL_UDP) begin
        return 0;
    end
    udp_dst_port = {packet_bytes[UDP_OFFSET_FROM_ETH + UDP_DST_PORT_OFFSET],
                    packet_bytes[UDP_OFFSET_FROM_ETH + UDP_DST_PORT_OFFSET + 1]};
    return (udp_dst_port == ETH_UDP_ROCE_PORT);
endfunction

//==============================================================================
// Unified Host Interface Monitor with Traffic Classification
//==============================================================================
task monitor_host_with_classification();
    automatic logic [7:0] frame_buffer[$];
    automatic int frame_length = 0;
    automatic logic frame_active = 0;
    automatic logic [7:0] packet_array[];
    automatic logic frame_completed = 0;
    automatic logic [7:0] cmd_code;
    automatic logic [15:0] udp_src_port, udp_dst_port;
    
    forever begin
        @(posedge hif_clk);
        
        if (o_hif_axis_tvalid[0] && i_hif_axis_tready[0]) begin
            // Start of new frame
            if (!frame_active) begin
                frame_buffer = {};
                frame_length = 0;
                frame_active = 1;
            end
            
            // Extract bytes from current word
            for (int byte_idx = 0; byte_idx < (`HOST_WIDTH/8); byte_idx++) begin
                if (o_hif_axis_tkeep[0][byte_idx]) begin
                    frame_buffer.push_back(o_hif_axis_tdata[0][byte_idx*8 +: 8]);
                    frame_length++;
                end
            end
            
            // End of frame processing
            if (o_hif_axis_tlast[0]) begin
                frame_active = 0;
                
                if (frame_length > ETH_HEADER_SIZE) begin // At least Ethernet header
                    // Port-first classification: BOOTP, then ECB
                    if (is_bootp_packet(frame_buffer)) begin
                        `TB_LOG_HIGH(LOG_INFO, "BOOTP packet skipped");
                    end else if (is_ecb_packet(frame_buffer)) begin
                        if (frame_length > UDP_PAYLOAD_OFFSET_FROM_ETH) begin
                            cmd_code = frame_buffer[UDP_PAYLOAD_OFFSET_FROM_ETH + ECB_CMD_OFFSET];
                            if (cmd_code[7] == 1'b1) begin // ECB response
                                `TB_LOG_MED(LOG_INFO, $sformatf("ECB Response: cmd_code=0x%02x", cmd_code));
                                parse_ecb_response(frame_buffer);
                            end else begin
                                `TB_LOG_LOW(LOG_ERR, "ECB packet without response bit set");
                            end
                        end
                    end else begin
                        // Sensor data (RoCE/COE) - invoke appropriate handlers
                        if (enable_coe_mode && frame_length > 16 && 
                            {frame_buffer[ETH_ETHERTYPE_OFFSET], frame_buffer[ETH_ETHERTYPE_OFFSET+1]} == IEEE_1722B_ETHERTYPE) begin
                            // COE packet - invoke COE handler
                            process_coe_packet(frame_buffer, frame_length);
                        end else if (is_roce_packet(frame_buffer)) begin
                            // RoCE packet - invoke RoCE handler (verified by UDP dest port)
                            packet_array = new[frame_buffer.size()];
                            foreach(frame_buffer[i]) packet_array[i] = frame_buffer[i];
                            frame_completed = process_roce_fragment(packet_array, UDP_PAYLOAD_OFFSET_FROM_ETH, frame_length);
                            
                            if (frame_completed) begin
                                `TB_LOG_LOW(LOG_PASS, "Sensor frame reassembly completed");
                            end
                            
                            // Conditional packet dumping - based on verbosity
                            `TB_LOG_HIGH(LOG_INFO, $sformatf("RoCE packet dump (%0d bytes)", frame_length));
                            // Note: Full hex dump implementation would go here
                        end else begin
                            if (is_ipv4_udp_packet(frame_buffer)) begin
                                `TB_LOG_LOW(LOG_ERR, $sformatf("Unrecognized UDP packet: dst_port=0x%02x%02x",
                                         frame_buffer[UDP_OFFSET_FROM_ETH + UDP_DST_PORT_OFFSET],
                                         frame_buffer[UDP_OFFSET_FROM_ETH + UDP_DST_PORT_OFFSET + 1]));
                            end else begin
                                `TB_LOG_LOW(LOG_ERR, $sformatf("Non-IPv4 or non-UDP packet: ethertype=0x%02x%02x",
                                         frame_buffer[ETH_ETHERTYPE_OFFSET],
                                         frame_buffer[ETH_ETHERTYPE_OFFSET+1]));
                            end
                            errors_detected++;
                        end
                    end
                end else begin
                        `TB_LOG_LOW(LOG_ERR, $sformatf("Truncated Ethernet frame: length=%0d (< %0d)", frame_length, ETH_HEADER_SIZE));
                        errors_detected++;
                end
            end
        end
    end
endtask

//==============================================================================
// COE Packet Handler for Unified Monitor
//==============================================================================
task process_coe_packet(logic [7:0] coe_frame_data[$], int frame_length);
    logic [7:0] coe_payload_array[];
    logic coe_frame_completed;
    int frame_id;
    int header_size;
    logic [31:0] last_word;
    logic [1:0] coe_frame_number;
    logic [27:0] coe_byte_offset;
    int payload_size;
    logic [7:0] coe_flags;
    
    host_packets_received++;
    host_fragments_received++;
    
    `TB_LOG_MED(LOG_INFO, $sformatf("Host monitor: COE packet %0d detected, ethertype=0x22F0, len=%0d", 
            host_packets_received, frame_length));
    
    // Extract COE payload as L2 payload (immediately after 14-byte Ethernet header)
    if (frame_length > ETH_HEADER_SIZE) begin
        coe_payload_array = new[frame_length - ETH_HEADER_SIZE];
        for (int i = 0; i < (frame_length - ETH_HEADER_SIZE); i++) begin
            coe_payload_array[i] = coe_frame_data[ETH_HEADER_SIZE + i];
        end

        // Validate COE data content (L2 COE payload)
        analyze_coe_payload(coe_payload_array);

        // Header-driven frame completion based on flags and byte_offset/payload size
        coe_frame_completed = 0;
        header_size = COE_HDR_SIZE; // IEEE 1722B header size
        if (coe_payload_array.size() >= header_size) begin
            // Extract fields using correct COE header layout
            last_word = {coe_payload_array[COE_LAST_WORD_OFFSET + 0], coe_payload_array[COE_LAST_WORD_OFFSET + 1], coe_payload_array[COE_LAST_WORD_OFFSET + 2], coe_payload_array[COE_LAST_WORD_OFFSET + 3]};
            coe_frame_number = last_word[COE_FRAME_NUM_MSB:COE_FRAME_NUM_LSB];
            coe_byte_offset = last_word[COE_BYTE_OFFSET_MSB:COE_BYTE_OFFSET_LSB];
            coe_flags = coe_payload_array[COE_FLAGS_BYTE_OFFSET]; // Flags nibble in lower 4 bits
            payload_size = coe_payload_array.size() - header_size;

            frame_id = coe_frame_number;

            // Frame start when byte_offset == 0
            if (coe_byte_offset == 0) begin
                `TB_LOG_MED(LOG_INFO, $sformatf("[COE] Frame %0d start detected (byte_offset=0)", frame_id));
            end

            // End-of-frame when explicit FRAME_END flag is set (meta packet),
            // or when this packet covers/exceeds total frame size
            if ((coe_flags[COE_FLAGS_NIBBLE_MSB:COE_FLAGS_NIBBLE_LSB] == COE_FRAME_END) ||
                ((coe_byte_offset + payload_size) >= FRAME_SIZE_BYTES)) begin
                coe_frame_completed = 1;
                if (frame_id >= 0 && frame_id < 3) begin
                    if (!frame_received[frame_id]) begin
                        frame_received[frame_id] = 1;
                        unique_frames_received++;
                        `TB_LOG_LOW(LOG_PASS, $sformatf("[OK] COE Frame %0d received and ANALYZED (total unique frames: %0d)", 
                                 frame_id, unique_frames_received));
                    end
                end
                completed_sensor_frames++;
                `TB_LOG_LOW(LOG_PASS, $sformatf("[COE] Complete sensor frame %0d reassembled and ANALYZED via header-driven logic", frame_id));
            end
        end
    end
endtask

`endif // HOLOLINK_MONITORS_SVH