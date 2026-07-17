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

`ifndef HOLOLINK_ROCE_HELPERS_SVH
`define HOLOLINK_ROCE_HELPERS_SVH

//==============================================================================
// RoCE (RDMA over Converged Ethernet) Helper Library
// Streaming validation and frame reassembly utilities for Hololink testbench
//==============================================================================

//==============================================================================
// Simplified streaming validation state (no associative arrays)
//==============================================================================
int stream_bytes_received = 0;
int stream_fragment_count = 0;
int stream_validation_errors = 0;
bit stream_reported_missing_expected = 0;
bit stream_in_progress = 0;
int  current_frame_index = 0;

//==============================================================================
// RoCE Header parsing function (based on UVM ethernet package)
//==============================================================================
function automatic void parse_roce_headers(
    input  logic [7:0] packet_data[], 
    input  integer     udp_start,
    output logic [7:0]  opcode,
    output logic [23:0] dest_qp,
    output logic [31:0] psn,
    output logic [63:0] vaddr,
    output logic [31:0] rkey,
    output logic [31:0] dma_len,
    output integer      data_start
);
    integer bth_start, reth_start;
    
    // BTH (Base Transport Header) starts immediately at UDP payload
    bth_start = udp_start;  // udp_start should be UDP_PAYLOAD_OFFSET_FROM_ETH
    
    // Extract BTH fields (12 bytes total) - Standard RoCE BTH format
    opcode = packet_data[bth_start+0];
    
    // Optional debug dump of BTH bytes
    `TB_LOG_WHEN(VERBOSITY_HIGH, begin
        `TB_LOG(LOG_INFO, $sformatf("RoCE Header Debug: OpCode=0x%02x, BTH bytes[0:11] = %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
                 opcode,
                 packet_data[bth_start+0], packet_data[bth_start+1], packet_data[bth_start+2], packet_data[bth_start+3],
                 packet_data[bth_start+4], packet_data[bth_start+5], packet_data[bth_start+6], packet_data[bth_start+7],
                 packet_data[bth_start+8], packet_data[bth_start+9], packet_data[bth_start+10], packet_data[bth_start+11]));
    end)
    
    // Bytes 4-6: Destination QP (24 bits, network byte order)
    dest_qp = {packet_data[bth_start+4], packet_data[bth_start+5], packet_data[bth_start+6]};
    
    // Bytes 8-10: PSN (24 bits, network byte order)  
    psn = {packet_data[bth_start+8], packet_data[bth_start+9], packet_data[bth_start+10]};
    
    // Display RoCE opcode information  
    case (opcode)
      ROCE_OPCODE_SEND_FIRST:            `TB_LOG_LOW(LOG_INFO, "RoCE OpCode: SEND_FIRST (0x06)");
      ROCE_OPCODE_SEND_MIDDLE:           `TB_LOG_LOW(LOG_INFO, "RoCE OpCode: SEND_MIDDLE (0x07)");  
      ROCE_OPCODE_SEND_LAST:             `TB_LOG_LOW(LOG_INFO, "RoCE OpCode: SEND_LAST (0x08)");
      ROCE_OPCODE_RDMA_WRITE_ONLY:       `TB_LOG_LOW(LOG_INFO, "RoCE OpCode: RDMA_WRITE_ONLY (0x0A) - Regular sensor data");
      ROCE_OPCODE_RDMA_WRITE_FIRST:      `TB_LOG_LOW(LOG_INFO, "RoCE OpCode: RDMA_WRITE_FIRST (0x2A) - First segment of multi-packet RDMA write");
      ROCE_OPCODE_RDMA_WRITE_ONLY_IMMDT: `TB_LOG_LOW(LOG_INFO, "RoCE OpCode: RDMA_WRITE_ONLY_IMMDT (0x2B) - Meta packet after BPW");
      default: `TB_LOG_LOW(LOG_WARN, $sformatf("Unknown RoCE opcode 0x%02x", opcode));
    endcase
    
    // RETH (RDMA Extended Transport Header) follows BTH (16 bytes total)
    reth_start = bth_start + 12;
    
    // Bounds check for RETH header
    if (reth_start + 16 > packet_data.size()) begin
        `TB_LOG_LOW(LOG_ERR, $sformatf("Packet too small for RETH header (need %0d bytes, have %0d)", 
                 reth_start + 16, packet_data.size()));
        vaddr = 64'hDEADBEEF;
        rkey = 32'hDEADBEEF;  
        dma_len = 0;
        data_start = packet_data.size();
        return;
    end
    
    // RETH Byte 0-7: Virtual Address (64 bits, network byte order)
    vaddr = {packet_data[reth_start+0], packet_data[reth_start+1], packet_data[reth_start+2], packet_data[reth_start+3],
             packet_data[reth_start+4], packet_data[reth_start+5], packet_data[reth_start+6], packet_data[reth_start+7]};
    
    // RETH Byte 8-11: Remote Key (32 bits, network byte order)
    rkey = {packet_data[reth_start+8], packet_data[reth_start+9], packet_data[reth_start+10], packet_data[reth_start+11]};
    
    // RETH Byte 12-15: DMA Length (32 bits, network byte order)
    dma_len = {packet_data[reth_start+12], packet_data[reth_start+13], packet_data[reth_start+14], packet_data[reth_start+15]};
    
    // Debug RETH fields
    `TB_LOG_LOW(LOG_INFO, $sformatf("RETH Fields: VA=0x%016x, RKEY=0x%08x, DMA_LEN=%0d", vaddr, rkey, dma_len));
    
    // Data starts after RETH header
    data_start = reth_start + 16;
endfunction

//==============================================================================
// Fragment reassembly function (based on UVM roce_data_scbd logic)
//==============================================================================
function automatic logic process_roce_fragment(
    input logic [7:0] packet_data[],
    input integer     udp_start,
    input integer     packet_len
);
    logic [7:0]  opcode;
    logic [23:0] dest_qp;
    logic [31:0] psn;
    logic [63:0] vaddr;
    logic [31:0] rkey, dma_len;
    integer      data_start, data_len;
    int          frame_id;
    logic [63:0] expected_vaddr;
    // Streaming compare state (declare at top per SV rules)
    int          fragment_mismatches;
    int          fragment_base_offset;
    int          bytes_available;
    int          bytes_to_compare;
    int          words_to_compare;
    int          bytes_per_word;
    logic [`DATAPATH_WIDTH-1:0] expected_word;
    logic [`DATAPATH_WIDTH-1:0] received_word;

    // Stream validation: compare bytes directly against expected_sensor_data (data opcodes only)
    fragment_mismatches = 0;
    fragment_base_offset = stream_bytes_received;
    
    // Parse RoCE headers
    parse_roce_headers(packet_data, udp_start, opcode, dest_qp, psn, vaddr, rkey, dma_len, data_start);
    
    // Only handle RDMA WRITE opcodes; flag unexpected opcodes as errors
    if (!(opcode == ROCE_OPCODE_RDMA_WRITE_ONLY || opcode == ROCE_OPCODE_RDMA_WRITE_FIRST || opcode == ROCE_OPCODE_RDMA_WRITE_ONLY_IMMDT)) begin
        errors_detected++;
        `TB_LOG_LOW(LOG_ERR, $sformatf("Unexpected RoCE opcode 0x%02x for sensor data path", opcode));
        return 0;
    end

    // Determine actual data length within the packet bounds
    data_len = dma_len;
    if ((data_start + data_len) > packet_len) begin
        `TB_LOG_LOW(LOG_ERR, $sformatf("RoCE fragment length clamp: dma_len=%0d exceeds remaining payload=%0d (packet_len=%0d, data_start=%0d)",
                 dma_len, (packet_len > data_start) ? (packet_len - data_start) : 0, packet_len, data_start));
        data_len = packet_len - data_start;
        if (data_len < 0) data_len = 0;
    end
    
    host_fragments_received++;
    frame_id = completed_sensor_frames; // default; may capture to current_frame_index below

    // On first data fragment of a frame, capture frame index and mark in-progress
    if ((opcode == ROCE_OPCODE_RDMA_WRITE_ONLY || opcode == ROCE_OPCODE_RDMA_WRITE_FIRST) && (stream_bytes_received == 0) && !stream_in_progress) begin
        current_frame_index = completed_sensor_frames;
        stream_in_progress = 1;
    end

    // Address sanity check: expected address derives from frame counter → buffer index
    // Use dp_roce0_vaddr_cfg_bytes[buffer_index] captured when registers were written
    // Only check VA on first data fragment of a frame
    if ((opcode == ROCE_OPCODE_RDMA_WRITE_ONLY || opcode == ROCE_OPCODE_RDMA_WRITE_FIRST) && (stream_bytes_received == 0)) begin
        expected_vaddr = dp_roce0_vaddr_cfg_bytes[current_frame_index % 4];
        if (vaddr !== expected_vaddr) begin
            stream_validation_errors++;
            `TB_LOG_LOW(LOG_ERR, $sformatf("VA mismatch: expected(frame %0d buf %0d)=0x%016x, received=0x%016x",
                     current_frame_index, (current_frame_index % 4), expected_vaddr, vaddr));
        end else begin
            `TB_LOG_LOW(LOG_PASS, $sformatf("VA matched: frame %0d buf %0d address=0x%016x",
                     current_frame_index, (current_frame_index % 4), vaddr));
        end
    end
    
    fragment_mismatches = 0;
    fragment_base_offset = stream_bytes_received;
    if (opcode == ROCE_OPCODE_RDMA_WRITE_ONLY || opcode == ROCE_OPCODE_RDMA_WRITE_FIRST) begin
        // Width-agnostic bytes per word for comparisons
        bytes_per_word = (`DATAPATH_WIDTH/8);
        // Guard: report once if expected hasn't been populated yet
        if (!expected_sensor_data.exists(current_frame_index)) begin
            if (!stream_reported_missing_expected) begin
                `TB_LOG_LOW(LOG_ERR, $sformatf("expected_sensor_data for frame %0d not populated yet", current_frame_index));
            errors_detected++;
                stream_reported_missing_expected = 1;
            end
        end else begin
            // Determine how many bytes we can compare this fragment
            bytes_available  = expected_sensor_data[current_frame_index].size() - stream_bytes_received;
            if (bytes_available < 0) bytes_available = 0;
            bytes_to_compare = (data_len < bytes_available) ? data_len : bytes_available;
            words_to_compare = bytes_to_compare / bytes_per_word; // number of full datapath words

            // Compare full datapath-width words
            for (int w = 0; w < words_to_compare; w++) begin
                // Assemble received word from packet bytes (little-endian per existing code style)
                received_word = '0;
                for (int b = 0; b < bytes_per_word; b++) begin
                    received_word[b*8 +: 8] = packet_data[data_start + (w*bytes_per_word) + b];
                end
                // Assemble expected word from expected_sensor_data stream
                expected_word = '0;
                for (int eb = 0; eb < bytes_per_word; eb++) begin
                    expected_word[eb*8 +: 8] = expected_sensor_data[current_frame_index][stream_bytes_received + (w*bytes_per_word) + eb];
                end
                if (received_word !== expected_word) begin
                    stream_validation_errors++;
                    fragment_mismatches++;
                    `TB_LOG_LOW(LOG_ERR, $sformatf("Data word mismatch: frame %0d bytes %0d..%0d", 
                             current_frame_index,
                             fragment_base_offset + (w*bytes_per_word), fragment_base_offset + (w*bytes_per_word) + (bytes_per_word-1)));
                    `TB_LOG_LOW(LOG_ERR, $sformatf("  Expected: 0x%0h", expected_word));
                    `TB_LOG_LOW(LOG_ERR, $sformatf("  Received: 0x%0h", received_word));
                    `TB_LOG_LOW(LOG_ERR, $sformatf("  XOR Diff: 0x%0h", expected_word ^ received_word));
                end else begin
                    `TB_LOG_MED(LOG_PASS, $sformatf("Data word matched: frame %0d bytes %0d..%0d value=0x%0h",
                             current_frame_index,
                             fragment_base_offset + (w*bytes_per_word), fragment_base_offset + (w*bytes_per_word) + (bytes_per_word-1),
                             received_word));
                end
            end
            
            // Compare remaining residual bytes (if any)
            for (int r = words_to_compare*bytes_per_word; r < bytes_to_compare; r++) begin
                int abs_byte = stream_bytes_received + r;
                if (packet_data[data_start + r] !== expected_sensor_data[current_frame_index][abs_byte]) begin
                    stream_validation_errors++;
                    fragment_mismatches++;
                    `TB_LOG_LOW(LOG_ERR, $sformatf("Data byte mismatch: frame %0d offset %0d", current_frame_index, abs_byte));
                    `TB_LOG_LOW(LOG_ERR, $sformatf("  Expected: 0x%02x", expected_sensor_data[current_frame_index][abs_byte]));
                    `TB_LOG_LOW(LOG_ERR, $sformatf("  Received: 0x%02x", packet_data[data_start + r]));
                end else begin
                    `TB_LOG_HIGH(LOG_PASS, $sformatf("Data byte matched: frame %0d offset %0d value=0x%02x",
                             current_frame_index, abs_byte, packet_data[data_start + r]));
                end
            end
        end
    end

    if (opcode == ROCE_OPCODE_RDMA_WRITE_ONLY || opcode == ROCE_OPCODE_RDMA_WRITE_FIRST) begin
        stream_bytes_received += data_len;
        stream_fragment_count++;
        if (fragment_mismatches == 0 && data_len > 0) begin
            `TB_LOG_MED(LOG_PASS, $sformatf("Data matched: frame %0d bytes %0d..%0d (len=%0d)",
                     current_frame_index, fragment_base_offset, fragment_base_offset + data_len - 1, data_len));
        end else begin
            `TB_LOG_LOW(LOG_ERR, $sformatf("Data mismatch: frame %0d bytes %0d..%0d (len=%0d)",
                     current_frame_index, fragment_base_offset, fragment_base_offset + data_len - 1, data_len));
        end
    end

    // Frame completion conditions: size reached OR meta packet (only if in-progress)
    if ((opcode == ROCE_OPCODE_RDMA_WRITE_ONLY_IMMDT && stream_in_progress) || stream_bytes_received >= FRAME_SIZE_BYTES) begin
        // Mark coverage for frames 0..2
        if (current_frame_index >= 0 && current_frame_index < 3) begin
            if (!frame_received[current_frame_index]) begin
                frame_received[current_frame_index] = 1;
                unique_frames_received++;
            end
        end

        if (stream_validation_errors == 0) begin
            `TB_LOG_LOW(LOG_PASS, $sformatf("Frame %0d validation PASSED (%0d bytes, %0d fragments)",
                     current_frame_index, stream_bytes_received, stream_fragment_count));
        end else begin
            errors_detected += stream_validation_errors;
            `TB_LOG_LOW(LOG_FAIL, $sformatf("Frame %0d validation FAILED (%0d errors, %0d bytes, %0d fragments)",
                     current_frame_index, stream_validation_errors, stream_bytes_received, stream_fragment_count));
        end

        completed_sensor_frames++;
        stream_reported_missing_expected = 0; // reset for next frame
        stream_in_progress = 0;

        // Reset stream state for next frame
        stream_bytes_received = 0;
        stream_fragment_count = 0;
        stream_validation_errors = 0;
        return 1'b1;
    end

    return 1'b0;
endfunction

// (validate_reassembled_sensor_frame removed; streaming validation is used now)

`endif // HOLOLINK_ROCE_HELPERS_SVH
`ifndef HOLOLINK_ROCE_HELPERS_SVH_EXTRA
`define HOLOLINK_ROCE_HELPERS_SVH_EXTRA

//==============================================================================
// Simple reporting utility for leftover incomplete frames
//==============================================================================
function automatic void report_incomplete_frames();
    if (stream_bytes_received > 0) begin
            `TB_LOG_LOW(LOG_ERR, $sformatf("Incomplete sensor frame in progress: %0d bytes, %0d fragments",
                 stream_bytes_received, stream_fragment_count));
    end
endfunction

`endif // HOLOLINK_ROCE_HELPERS_SVH_EXTRA