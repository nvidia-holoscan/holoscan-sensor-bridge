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
// EEPROM I2C Model Module
// Extracted from nv_hsb_ip_simple_tb.sv for modularity and reuse
// Uses the original working embedded EEPROM logic with wand bus connection
//==============================================================================

import hololink_tb_pkg::*;

module eeprom_i2c_model #(
    parameter [31:0] TB_FPGA_IP = 32'hC0_A8_01_0C,  // 192.168.1.12 (FPGA IP) 
    parameter [31:0] TB_HOST_IP = 32'hC0_A8_01_0A    // 192.168.1.10 (HOST IP)
) (
    input  logic clk,
    input  logic rstn,
    
    // I2C wand bus connections - following original working approach
    input  wand i2c_scl_bus,
    output wand i2c_sda_bus
);
//==============================================================================
// EEPROM Memory Model (256 bytes)
//==============================================================================
logic [7:0] eeprom_memory [256];
logic eeprom_initialized;

// EEPROM target drives wand bus directly (like original working approach)
logic eeprom_sda_drive;
assign i2c_sda_bus = eeprom_sda_drive;  // EEPROM drives wand bus directly

// I2C protocol detection
logic i2c_scl_prev, i2c_sda_prev;
logic i2c_start_condition, i2c_stop_condition;

//==============================================================================
// EEPROM Initialization Model
//==============================================================================
task initialize_eeprom_model();
    `TB_LOG_LOW(LOG_INFO, "Initializing EEPROM model with correct layout from eeprom_mem_model.svh");
    
    // Clear EEPROM
    for (int i = 0; i < 256; i++)
        eeprom_memory[i] = 8'h00;
    
    // Version Information (bytes 0-3) - matches eeprom_mem_model constraints
    eeprom_memory[0] = 8'hFF;  // major_ver_num 
    eeprom_memory[1] = 8'h00;  // minor_ver_num
    eeprom_memory[2] = 8'h02;  // eeprom_len[15:8] - 0x0254
    eeprom_memory[3] = 8'h54;  // eeprom_len[7:0]
    
    // MAC Address Count (byte 19) - from eeprom_mem_model constraint 
    eeprom_memory[19] = 8'h01;  // num_mac_addr = 1
    
    // Board Version (bytes 20-39) - from eeprom_mem_model constraint  
    eeprom_memory[20] = 8'h0B; eeprom_memory[21] = 8'h20; eeprom_memory[22] = 8'h00; eeprom_memory[23] = 8'h00;
    eeprom_memory[24] = 8'h01; eeprom_memory[25] = 8'h2D; eeprom_memory[26] = 8'h00; eeprom_memory[27] = 8'h00;
    eeprom_memory[28] = 8'h00; eeprom_memory[29] = 8'h00; eeprom_memory[30] = 8'h2D; eeprom_memory[31] = 8'h08;
    eeprom_memory[32] = 8'h00; eeprom_memory[33] = 8'h08; eeprom_memory[34] = 8'h03; eeprom_memory[35] = 8'h01;
    eeprom_memory[36] = 8'h2D; eeprom_memory[37] = 8'h09; eeprom_memory[38] = 8'h09; eeprom_memory[39] = 8'h06;
    
    // FPGA IP Address (bytes 60-63) - matching original EEPROM layout
    eeprom_memory[60] = TB_FPGA_IP[31:24];  // FPGA IP byte 3
    eeprom_memory[61] = TB_FPGA_IP[23:16];  // FPGA IP byte 2 
    eeprom_memory[62] = TB_FPGA_IP[15:8];   // FPGA IP byte 1
    eeprom_memory[63] = TB_FPGA_IP[7:0];    // FPGA IP byte 0
    eeprom_memory[64] = 8'h17;              // IP validity marker
    
    // MAC Address (bytes 68-73) - from eeprom_mem_model constraint 48'hCA_FE_C0_FF_EE_00
    eeprom_memory[68] = 8'hCA;  // MAC byte 0 (MSB)
    eeprom_memory[69] = 8'hFE;  // MAC byte 1  
    eeprom_memory[70] = 8'hC0;  // MAC byte 2
    eeprom_memory[71] = 8'hFF;  // MAC byte 3
    eeprom_memory[72] = 8'hEE;  // MAC byte 4
    eeprom_memory[73] = 8'h00;  // MAC byte 5 (LSB)
    
    // Board Serial Number (bytes 74-80) - 56'h56341278563412 (7 bytes)
    eeprom_memory[74] = 8'h56; eeprom_memory[75] = 8'h34; eeprom_memory[76] = 8'h12; 
    eeprom_memory[77] = 8'h78; eeprom_memory[78] = 8'h56; eeprom_memory[79] = 8'h34; 
    eeprom_memory[80] = 8'h12;
    
    // CPRO FPGA Firmware Version (bytes 90-91) - 16'h0010
    eeprom_memory[90] = 8'h00;  // MSB
    eeprom_memory[91] = 8'h10;  // LSB
    
    // CLNX FPGA Firmware Version (bytes 92-93) - 16'h00FF
    eeprom_memory[92] = 8'h00;  // MSB
    eeprom_memory[93] = 8'hFF;  // LSB
    
    // CPRO FPGA Image 0 CRC (bytes 94-95) - 16'h00FF
    eeprom_memory[94] = 8'h00;  // MSB
    eeprom_memory[95] = 8'hFF;  // LSB
    
    // CLNX FPGA Image 0 CRC (bytes 102-103) - 16'h00FF  
    eeprom_memory[102] = 8'h00; // MSB
    eeprom_memory[103] = 8'hFF; // LSB
    
    `TB_LOG_LOW(LOG_INFO, "EEPROM Model: Memory initialized with proper layout");
    `TB_LOG_LOW(LOG_INFO, $sformatf("EEPROM Model: MAC@68-73 = %02x:%02x:%02x:%02x:%02x:%02x",
            eeprom_memory[68], eeprom_memory[69], eeprom_memory[70],
            eeprom_memory[71], eeprom_memory[72], eeprom_memory[73]));
    `TB_LOG_LOW(LOG_INFO, $sformatf("EEPROM Model: IP@60-63  = %0d.%0d.%0d.%0d",
            eeprom_memory[60], eeprom_memory[61], eeprom_memory[62], eeprom_memory[63]));
    `TB_LOG_LOW(LOG_INFO, "EEPROM Model: Board Ver@20-39, Serial@74-80, FW@90-95,102-103");
endtask

// I2C state machine (read-only EEPROM)
typedef enum logic [2:0] {
    I2C_IDLE,
    I2C_ADDR_RX,
    I2C_ADDR_ACK, 
    I2C_REG_ADDR_RX,
    I2C_REG_ADDR_ACK,
    I2C_DATA_TX,
    I2C_WAIT_ACK
} i2c_state_t;

i2c_state_t i2c_state;
logic [7:0] i2c_rx_shift, i2c_tx_shift;
logic [2:0] i2c_bit_cnt;
logic [6:0] i2c_addr_received;
logic i2c_rw_bit, i2c_addr_match;
logic [7:0] i2c_reg_addr;
logic ack_phase;
logic [2:0] data_bit_idx;
logic prime_pending;
logic ack_sample_pending;
logic ack_sda_low_seen;
logic after_bit0_fall;

// EEPROM configuration
parameter I2C_EEPROM_ADDR = 7'h50;  // Standard EEPROM I2C address
parameter DEBUG_ACCEPT_ANY_ADDR = 0;  // Set to 1 only for debugging

//==============================================================================
// I2C Start/Stop Condition Detection
//==============================================================================
always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        i2c_scl_prev <= 1'b1;
        i2c_sda_prev <= 1'b1;
        i2c_start_condition <= 1'b0;
        i2c_stop_condition <= 1'b0;
    end else begin
        i2c_scl_prev <= i2c_scl_bus;
        i2c_sda_prev <= i2c_sda_bus;
        
        // Start: SDA falling edge while SCL is high
        i2c_start_condition <= (i2c_scl_bus && i2c_sda_prev && !i2c_sda_bus);
        
        // Stop: SDA rising edge while SCL is high  
        i2c_stop_condition <= (i2c_scl_bus && !i2c_sda_prev && i2c_sda_bus);
        
        // Debug output for START/STOP detection
        if (i2c_scl_bus && i2c_sda_prev && !i2c_sda_bus) begin
            `TB_LOG_MED(LOG_INFO, "I2C: START condition detected (SCL=1, SDA: 1->0)");
        end
        if (i2c_scl_bus && !i2c_sda_prev && i2c_sda_bus) begin
            `TB_LOG_MED(LOG_INFO, "I2C: STOP condition detected (SCL=1, SDA: 0->1)");
        end
    end
end

// I2C EEPROM Target Protocol Handler (original working logic)
always_ff @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        i2c_state <= I2C_IDLE;
        i2c_rx_shift <= 8'h0;
        i2c_tx_shift <= 8'h0;
        i2c_bit_cnt <= 3'h0;
        i2c_reg_addr <= 8'h0;
        i2c_addr_received <= 7'h0;
        i2c_rw_bit <= 1'b0;
        i2c_addr_match <= 1'b0;
        eeprom_sda_drive <= 1'b1;    // released (pull-up)
        ack_phase <= 1'b0;
        data_bit_idx <= 3'h0;
        prime_pending <= 1'b0;
        ack_sample_pending <= 1'b0;
        ack_sda_low_seen <= 1'b0;
        after_bit0_fall <= 1'b0;
    end else begin
        // Default: do not change SDA unless state dictates
        case (i2c_state)
            I2C_IDLE: begin
                eeprom_sda_drive <= 1'b1;
                i2c_bit_cnt <= 3'h0;
                if (i2c_start_condition) begin
                    i2c_state <= I2C_ADDR_RX;
                    i2c_rx_shift <= 8'h0;
                    i2c_bit_cnt <= 3'h0;
                end
            end

            I2C_ADDR_RX: begin
                // sample address+R/W on SCL rising
                if (i2c_scl_bus && !i2c_scl_prev) begin
                    i2c_rx_shift <= {i2c_rx_shift[6:0], i2c_sda_bus};
                    if (i2c_bit_cnt == 3'h7) begin
                        // full byte received
                        i2c_addr_received <= {i2c_rx_shift[6:0], i2c_sda_bus}[7:1];
                        i2c_rw_bit <= {i2c_rx_shift[6:0], i2c_sda_bus}[0];
                        i2c_addr_match <= ({i2c_rx_shift[6:0], i2c_sda_bus}[7:1] == I2C_EEPROM_ADDR) || DEBUG_ACCEPT_ANY_ADDR;
                        i2c_state <= I2C_ADDR_ACK;
                        ack_phase <= 1'b0;
                        i2c_bit_cnt <= 3'h0;
                    end else begin
                        i2c_bit_cnt <= i2c_bit_cnt + 1;
                    end
                end
            end

            I2C_ADDR_ACK: begin
                if (i2c_addr_match) begin
                    // Hold ACK low for the 9th bit: from one falling to the next falling
                    if (!i2c_scl_bus && i2c_scl_prev) begin
                        if (!ack_phase) begin
                            eeprom_sda_drive <= 1'b0; // start ACK
                            ack_phase <= 1'b1;
                        end else begin
                            // end ACK bit -> move to next phase
                            eeprom_sda_drive <= 1'b1; // release
                            ack_phase <= 1'b0;
                            if (i2c_rw_bit) begin
                                // READ: present first data bit immediately (SCL is low here)
                                i2c_tx_shift <= eeprom_memory[i2c_reg_addr];
                                data_bit_idx <= 3'd6; // we drive bit7 below, next will be 6
                                eeprom_sda_drive <= eeprom_memory[i2c_reg_addr][7];
                                i2c_state <= I2C_DATA_TX;
                            end else begin
                                // WRITE: receive register address
                                i2c_rx_shift <= 8'h0;
                                i2c_bit_cnt <= 3'h0;
                                i2c_state <= I2C_REG_ADDR_RX;
                            end
                        end
                    end
                end else begin
                    // NACK: keep released during 9th bit and return to IDLE after it ends
                    eeprom_sda_drive <= 1'b1;
                    if (!i2c_scl_bus && i2c_scl_prev) begin
                        if (!ack_phase) begin
                            ack_phase <= 1'b1; // wait one ACK bit
                        end else begin
                            ack_phase <= 1'b0;
                            i2c_state <= I2C_IDLE;
                        end
                    end
                end
            end

            I2C_REG_ADDR_RX: begin
                // receive 8-bit register address on SCL rising
                if (i2c_scl_bus && !i2c_scl_prev) begin
                    i2c_rx_shift <= {i2c_rx_shift[6:0], i2c_sda_bus};
                    if (i2c_bit_cnt == 3'h7) begin
                        i2c_reg_addr <= {i2c_rx_shift[6:0], i2c_sda_bus};
                        i2c_state <= I2C_REG_ADDR_ACK;
                        ack_phase <= 1'b0;
                        i2c_bit_cnt <= 3'h0;
                    end else begin
                        i2c_bit_cnt <= i2c_bit_cnt + 1;
                    end
                end
            end

            I2C_REG_ADDR_ACK: begin
                // ACK the register address then finish (read-only device)
                if (!i2c_scl_bus && i2c_scl_prev) begin
                    if (!ack_phase) begin
                        eeprom_sda_drive <= 1'b0;
                        ack_phase <= 1'b1;
                    end else begin
                        eeprom_sda_drive <= 1'b1;
                        ack_phase <= 1'b0;
                        i2c_state <= I2C_IDLE;
                    end
                end
            end

            I2C_DATA_TX: begin
                // Drive bits on SCL falling; controller samples on rising
                if (!i2c_scl_bus && i2c_scl_prev) begin
                    if (after_bit0_fall) begin
                        // One falling after bit0 was presented: release for ACK and go wait
                        after_bit0_fall <= 1'b0;
                        eeprom_sda_drive <= 1'b1; // release SDA for controller's ACK/NACK
                        i2c_state <= I2C_WAIT_ACK;
                        ack_sample_pending <= 1'b1;
                        ack_sda_low_seen <= 1'b0;
                        data_bit_idx <= 3'd7; // reset index for next byte (if any)
                    end else begin
                        // Drive current bit
                        eeprom_sda_drive <= i2c_tx_shift[data_bit_idx];
                        if (data_bit_idx == 3'd0) begin
                            // Keep bit0 held through next rising; mark to release at next falling
                            after_bit0_fall <= 1'b1;
                        end else begin
                            data_bit_idx <= data_bit_idx - 3'd1;
                        end
                    end
                end
            end

            I2C_WAIT_ACK: begin
                // Release SDA so controller can ACK/NACK
                eeprom_sda_drive <= 1'b1;
                // Latch if SDA low while SCL high during the ACK bit
                if (i2c_scl_bus && ack_sample_pending)
                    ack_sda_low_seen <= ack_sda_low_seen | (i2c_sda_bus == 1'b0);
                // Decide at the end of the 9th clock (falling)
                if (!i2c_scl_bus && i2c_scl_prev && ack_sample_pending) begin
                    ack_sample_pending <= 1'b0;
                    if (ack_sda_low_seen) begin
                        // Controller ACK: continue with next byte
                        i2c_reg_addr <= i2c_reg_addr + 1;
                        i2c_tx_shift <= eeprom_memory[i2c_reg_addr + 1];
                        eeprom_sda_drive <= eeprom_memory[i2c_reg_addr + 1][7];
                        data_bit_idx <= 3'd6;
                        i2c_state <= I2C_DATA_TX;
                    end else begin
                        // Controller NACK: end read
                        i2c_state <= I2C_IDLE;
                    end
                end
            end
        endcase

        // STOP condition at any time brings us back to IDLE and releases SDA
        if (i2c_stop_condition) begin
            // Allow STOP after ACK decision; ignore during ACK sampling window
            if (!ack_sample_pending) begin
                i2c_state <= I2C_IDLE;
                eeprom_sda_drive <= 1'b1;
                ack_phase <= 1'b0;
                after_bit0_fall <= 1'b0;
            end
        end
    end
end

//==============================================================================
// EEPROM Initialization
//==============================================================================
initial begin
    eeprom_initialized = 1'b0;   // Initialize flag
    @(posedge rstn);             // Wait for reset release
    #(100*1ns);                      // Additional delay 
    initialize_eeprom_model();
    eeprom_initialized = 1'b1;   // Set flag after initialization complete
end

endmodule
