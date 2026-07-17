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

`ifndef HOLOLINK_SENSOR_STIM_SVH
`define HOLOLINK_SENSOR_STIM_SVH

//==============================================================================
// Hololink Sensor Data Stimulus Library
//
// DESCRIPTION:
//   This file provides tasks for generating realistic sensor data patterns to
//   test Hololink IP sensor-to-host data transmission functionality. It creates
//   structured data frames that are transmitted through the sensor interface.
//
// KEY TASKS:
//   - generate_sensor_data_patterns() : Creates test frames with known patterns
//   - Frame structure and timing generation
//   - Support for both single and multi-frame scenarios
//
// CUSTOMIZATION:
//   Modify FRAME_SIZE_WORDS and data patterns in hololink_tb_pkg.sv
//   to match your specific sensor data requirements and test scenarios.
//==============================================================================

//==============================================================================
// Sensor Data Generation Task - Parametric Pattern Generation
//==============================================================================
//--------------------------------------------------------------------------
// generate_sensor_data_patterns: Create test sensor data frames
//
// DESCRIPTION:
//   Generates structured sensor data frames with incrementing patterns for
//   easy validation. Creates TOTAL_TEST_FRAMES frames, each containing
//   FRAME_SIZE_WORDS of DATAPATH_WIDTH-bit data words. Only uses sensor interface 0.
//
// DATA PATTERN:
//   Pattern splits DATAPATH_WIDTH equally: {frame_field[half_width], word_field[half_width]}
//   where half_width = DATAPATH_WIDTH / 2
//   Examples:
//     16-bit: {frame[7:0], word[7:0]}
//     32-bit: {frame[15:0], word[15:0]}  
//     64-bit: {frame[31:0], word[31:0]}
//     128-bit: {frame[63:0], word[63:0]}
//     1024-bit: {frame[511:0], word[511:0]}
//
// TIMING:
//   Frames are sent back-to-back with proper AXI4-Stream handshaking.
//   Waits for o_sif_axis_tready before sending each DATAPATH_WIDTH-bit word.
//
// ALIGNMENT:
//   Works with ANY power-of-2 DATAPATH_WIDTH (16, 32, 64, 128, 256, 512, 1024, etc.)
//   tkeep is always set to all '1's regardless of DATAPATH_WIDTH
//--------------------------------------------------------------------------
task generate_sensor_data_patterns();
    // Declare ALL automatic variables at the beginning of task
    automatic int num_frames = TOTAL_TEST_FRAMES;
    automatic int frame_size_words = FRAME_SIZE_WORDS;
    automatic logic [`DATAPATH_WIDTH-1:0] data_pattern;
    automatic int bytes_per_word = `DATAPATH_WIDTH / 8;
    automatic int frame_size_bytes = frame_size_words * bytes_per_word;
    
    `TB_LOG_LOW(LOG_INFO, "Starting sensor data generation...");
    `TB_LOG_LOW(LOG_INFO, $sformatf("Frame configuration: %0d frames × %0d words (%0d-bit) = %0d bytes per frame", 
             num_frames, frame_size_words, `DATAPATH_WIDTH, frame_size_bytes));
    `TB_LOG_LOW(LOG_INFO, "Data pattern: Incrementing values for easy verification");
    `TB_LOG_LOW(LOG_INFO, "Using sensor interface 0 only (others remain idle)");
    
    // Validate frame size alignment to design requirements (based on actual DATAPATH_WIDTH)
    
    `TB_LOG_LOW(LOG_INFO, $sformatf("Configuration: DATAPATH_WIDTH=%0d bits, %0d bytes per word", `DATAPATH_WIDTH, bytes_per_word));
    `TB_LOG_LOW(LOG_INFO, $sformatf("Pattern strategy: Split equally {frame[%0d:0], word[%0d:0]} = %0d bits total",
             (`DATAPATH_WIDTH/2)-1, (`DATAPATH_WIDTH/2)-1, `DATAPATH_WIDTH));
    
    if (frame_size_bytes % 128 != 0) begin
        `TB_LOG_LOW(LOG_WARN, $sformatf("WARNING: Frame size %0d bytes not aligned to 128-byte boundary", frame_size_bytes));
        `TB_LOG_LOW(LOG_WARN, "This may cause issues with some design configurations");
    end
    `TB_LOG_LOW(LOG_INFO, $sformatf("Frame size validation: %0d bytes (%s aligned to 128-byte boundary)", 
             frame_size_bytes, (frame_size_bytes % 128 == 0) ? "properly" : "NOT"));
    
    // PRE-POPULATE all expected_sensor_data arrays BEFORE sending any data to DUT
    // This prevents race condition where DUT processes data faster than testbench populates expected arrays
    `TB_LOG_LOW(LOG_INFO, "Pre-populating expected sensor data arrays to prevent race conditions...");
    for (int frame = 0; frame < num_frames; frame++) begin
        // Allocate array for this frame (size depends on actual DATAPATH_WIDTH)
        expected_sensor_data[frame] = new[frame_size_bytes];  // Total bytes for this frame
        
        // Pre-populate the entire expected data array for this frame
        for (int word = 0; word < frame_size_words; word++) begin
            // Create deterministic pattern by splitting DATAPATH_WIDTH equally between frame and word
            // This approach works elegantly for ANY DATAPATH_WIDTH
            
            // Split the width equally: {frame_field[half_width], word_field[half_width]}
            // Use proper bit width casting
            data_pattern = {(`DATAPATH_WIDTH/2)'(frame), (`DATAPATH_WIDTH/2)'(word)};
            
            // Store expected data (convert word to bytes based on actual width)
            for (int byte_idx = 0; byte_idx < bytes_per_word; byte_idx++) begin
                expected_sensor_data[frame][word*bytes_per_word + byte_idx] = data_pattern[byte_idx*8 +: 8];
            end
        end
        `TB_LOG_LOW(LOG_INFO, $sformatf("Pre-populated expected data for frame %0d (%0d bytes)", frame, frame_size_bytes));
    end
    `TB_LOG_LOW(LOG_INFO, "All expected sensor data arrays pre-populated successfully");
    
    // NOW send the actual sensor data to DUT - expected data is already ready for validation
    for (int frame = 0; frame < num_frames; frame++) begin
        `TB_LOG_LOW(LOG_INFO, $sformatf("===== Transmitting FRAME %0d to DUT =====", frame));
        
        for (int word = 0; word < frame_size_words; word++) begin
            // Create the same deterministic pattern that was used for expected data
            // MUST match the exact same logic used in pre-population above
            // Split the width equally: {frame_field[half_width], word_field[half_width]}
            // Use proper bit width casting (same as pre-population)
            data_pattern = {(`DATAPATH_WIDTH/2)'(frame), (`DATAPATH_WIDTH/2)'(word)};
            
            // Send ONLY to sensor interface 0 - avoid multi-sensor conflicts
            i_sif_axis_tdata[0] <= data_pattern;
            i_sif_axis_tkeep[0] <= {`DATAKEEP_WIDTH{1'b1}};    // CRITICAL: All tkeep bits MUST be '1' per design requirement (width depends on DATAPATH_WIDTH)
            i_sif_axis_tuser[0] <= 2'h0;     // Standard user signals
            i_sif_axis_tvalid[0] <= 1'b1;   // Only enable sensor 0
            i_sif_axis_tlast[0] <= (word == (frame_size_words - 1)) ? 1'b1 : 1'b0;
            
            // Debug: Log detailed transfer information for first and last words
            if (word == 0 || word == (frame_size_words - 1)) begin
                `TB_LOG_LOW(LOG_INFO, $sformatf("SENSOR[0] Transfer: frame=%0d word=%0d/%0d data=0x%0h tkeep=0x%0h tlast=%b", 
                         frame, word, frame_size_words-1, data_pattern, {`DATAKEEP_WIDTH{1'b1}}, (word == (frame_size_words - 1))));
            end
            
            // Keep other sensor interfaces IDLE
            for (int inst = 1; inst < `SENSOR_RX_IF_INST; inst++) begin
                i_sif_axis_tdata[inst] <= {`DATAPATH_WIDTH{1'b0}};
                i_sif_axis_tkeep[inst] <= {`DATAKEEP_WIDTH{1'b0}};
                i_sif_axis_tuser[inst] <= 2'h0;
                i_sif_axis_tvalid[inst] <= 1'b0;
                i_sif_axis_tlast[inst] <= 1'b0;
            end
            
            // AXI-Stream handshake: wait for tready HIGH at clock edge
            // Transfer is valid ONLY when both tvalid AND tready are high at posedge
            // Hold data stable until handshake completes to prevent data loss
            @(posedge sif_clk);
            while (!o_sif_axis_tready[0]) @(posedge sif_clk);
            // Handshake complete - data was accepted, safe to advance to next word
        end
        
        // Frame complete - IMMEDIATELY deassert tvalid to prevent duplicate transfers
        // This must happen at the SAME simulation time as the for loop exit (before next @posedge)
        // so that when the next clock edge arrives, tvalid is already 0
        i_sif_axis_tvalid[0] <= 1'b0;  // Schedule tvalid deassert NOW
        i_sif_axis_tlast[0] <= 1'b0;   // Schedule tlast deassert NOW
        // Keep tdata stable (assertion requires stability while tvalid transitions)
        
        // Wait for NBA to take effect, then clear remaining signals
        @(posedge sif_clk);
        // Now tvalid=0, safe to clear tdata without causing transfers or assertion failures
        i_sif_axis_tdata[0] <= {`DATAPATH_WIDTH{1'b0}};
        i_sif_axis_tkeep[0] <= {`DATAKEEP_WIDTH{1'b0}};
        i_sif_axis_tuser[0] <= 2'h0;
        
        // Keep other sensors permanently idle (already done above, but ensure consistency)
        for (int inst = 1; inst < `SENSOR_RX_IF_INST; inst++) begin
            i_sif_axis_tdata[inst] <= {`DATAPATH_WIDTH{1'b0}};
            i_sif_axis_tkeep[inst] <= {`DATAKEEP_WIDTH{1'b0}};
            i_sif_axis_tuser[inst] <= 2'h0;
            i_sif_axis_tvalid[inst] <= 1'b0;
            i_sif_axis_tlast[inst] <= 1'b0;
        end
        
        sensor_frames_generated++;
        `TB_LOG_LOW(LOG_INFO, $sformatf("===== FRAME %0d TRANSMITTED to DUT =====", frame));
        `TB_LOG_LOW(LOG_INFO, $sformatf("Expected data was pre-populated - no race condition"));
        `TB_LOG_LOW(LOG_INFO, $sformatf("Sensor interface states after frame %0d:", frame));
        `TB_LOG_LOW(LOG_INFO, $sformatf("  Sensor 0: tvalid=%b, tlast=%b, tready=%b", 
                 i_sif_axis_tvalid[0], i_sif_axis_tlast[0], o_sif_axis_tready[0]));
        if (`SENSOR_RX_IF_INST > 1) begin
            `TB_LOG_LOW(LOG_INFO, $sformatf("  Sensor 1: tvalid=%b, tlast=%b, tready=%b", 
                     i_sif_axis_tvalid[1], i_sif_axis_tlast[1], o_sif_axis_tready[1]));
        end
        
        // Add inter-frame delay to help DUT pipeline processing
        if (frame < (num_frames - 1)) begin  // Don't delay after last frame
            `TB_LOG_LOW(LOG_INFO, $sformatf("Inter-frame delay: allowing DUT to process frame %0d...", frame));
            repeat(1000) @(posedge sif_clk);  // 1000 clock inter-frame delay
        end
    end
    
    `TB_LOG_LOW(LOG_INFO, "===== SIMPLE SENSOR DATA GENERATION COMPLETE =====");
    `TB_LOG_LOW(LOG_INFO, $sformatf("Successfully generated ALL %0d frames:", num_frames));
    for (int i = 0; i < num_frames; i++) begin
        `TB_LOG_LOW(LOG_PASS, $sformatf("  [OK] Frame %0d: Pattern base {frame=%0d, word=idx}", i, i));
    end
    `TB_LOG_LOW(LOG_INFO, $sformatf("Total sensor data: %0d bytes (%0d %0d-bit words) sent to DUT", 
             num_frames * frame_size_bytes, num_frames * frame_size_words, `DATAPATH_WIDTH));
endtask

`endif // HOLOLINK_SENSOR_STIM_SVH