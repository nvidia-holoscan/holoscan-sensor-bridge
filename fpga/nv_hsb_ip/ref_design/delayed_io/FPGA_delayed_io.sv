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

module FPGA_delayed_io
  import apb_pkg::*;
#(
  parameter CLK_FREQ     = 19541250,
  parameter NUM_OUTPUTS  = 1,    // Configurable number of outputs
  parameter MAX_OUTPUTS  = 32,   // Maximum supported outputs
  parameter W_OFSET      = 8     // APB address offset width
)(
  // Clock and Reset
  input  logic                  cmd_clk,      // APB clock domain
  input  logic                  cmd_rst_n,    // APB reset
  input  logic                  proc_clk,     // Processing clock domain
  input  logic                  proc_rst_n,   // Processing reset
  
  // APB interface
  input  apb_m2s                i_apb_m2s,
  output apb_s2m                o_apb_s2m,
  
  // I/O Interface (scalable)
  output logic [NUM_OUTPUTS-1:0] o_io_pins,
  output logic [NUM_OUTPUTS-1:0] o_io_busy
);

//------------------------------------------------------------------------------
// Register Map
//------------------------------------------------------------------------------
// 0x04: Delay value in microseconds
// 0x08: Immediate Set  
// 0x0C: Immediate Clear
// 0x14: Delayed Set
// 0x18: Delayed Clear
// 0x20: Cancel Delayed (cancel pending delayed operations)
// 0x80: Output states (read)
// 0x84: Countdown active (read)
// 0xF0: Global Control
// 0xF8: Cycles per microsecond (read)

// Calculate clock cycles per microsecond (assuming 20MHz APB clock)
localparam CYCLES_PER_US = (CLK_FREQ+999999)/1000000;

//------------------------------------------------------------------------------
// APB Register Interface (Direct Decode)
//------------------------------------------------------------------------------

// APB address decoding
logic [7:0] reg_addr;
logic       apb_write;
logic       apb_read;

assign reg_addr  = i_apb_m2s.paddr[7:0];
assign apb_write = i_apb_m2s.psel && i_apb_m2s.penable && i_apb_m2s.pwrite;
assign apb_read  = i_apb_m2s.psel && i_apb_m2s.penable && !i_apb_m2s.pwrite;

// Configuration registers
logic [31:0] global_ctrl;
logic [31:0] delay_value_us;
logic        global_enable;

assign global_enable = global_ctrl[0];

// Status registers
logic [31:0] output_states;
logic [31:0] countdown_active;

// Immediate operation decode (single cycle pulses)
logic [31:0] immediate_set_pulse;
logic [31:0] immediate_clear_pulse;
logic [31:0] cancel_delayed_pulse;
logic [31:0] delayed_set_pulse;
logic [31:0] delayed_clear_pulse;

assign immediate_set_pulse   = (apb_write && (reg_addr == 8'h08)) ? i_apb_m2s.pwdata : 32'h0;
assign immediate_clear_pulse = (apb_write && (reg_addr == 8'h0C)) ? i_apb_m2s.pwdata : 32'h0;
assign delayed_set_pulse     = (apb_write && (reg_addr == 8'h14)) ? i_apb_m2s.pwdata : 32'h0;
assign delayed_clear_pulse   = (apb_write && (reg_addr == 8'h18)) ? i_apb_m2s.pwdata : 32'h0;
assign cancel_delayed_pulse  = (apb_write && (reg_addr == 8'h20)) ? i_apb_m2s.pwdata : 32'h0;

//------------------------------------------------------------------------------
// APB Register Write Logic
//------------------------------------------------------------------------------

always_ff @(posedge cmd_clk or negedge cmd_rst_n) begin
  if (!cmd_rst_n) begin
    global_ctrl <= 32'h0;
    delay_value_us <= 32'd1000;  // Default 1000 microseconds
  end else begin
    if (apb_write) begin
      case (reg_addr)
        8'h04: delay_value_us <= i_apb_m2s.pwdata;  // Delay value in microseconds
        8'hF0: global_ctrl <= i_apb_m2s.pwdata;     // Global control
        default: ;
      endcase
    end
  end
end

//------------------------------------------------------------------------------
// APB Register Read Logic
//------------------------------------------------------------------------------

logic [31:0] read_data;
logic        read_valid;

always_ff @(posedge cmd_clk or negedge cmd_rst_n) begin
  if (!cmd_rst_n) begin
    read_data  <= 32'h0;
    read_valid <= 1'b0;
  end else begin
    read_valid <= apb_read;
    if (apb_read) begin
      case (reg_addr)
        8'h04: read_data <= delay_value_us;
        8'h80: read_data <= output_states;
        8'h84: read_data <= countdown_active;
        8'hF0: read_data <= global_ctrl;
        8'hF8: read_data <= CYCLES_PER_US;  // Debug: cycles per microsecond
        default: read_data <= 32'hBADADD12;
      endcase
    end
  end
end

// APB Interface outputs
// For writes: ready immediately (1 cycle)
// For reads: ready when read_data is valid (2 cycles total)
assign o_apb_s2m.pready = apb_write || read_valid;
assign o_apb_s2m.prdata = read_data;
assign o_apb_s2m.pserr  = 1'b0;

//------------------------------------------------------------------------------
// Internal signals and state
//------------------------------------------------------------------------------

// Output control signals
logic [NUM_OUTPUTS-1:0] output_states_bits;
logic [NUM_OUTPUTS-1:0] countdown_active_bits;

// Track delayed operation type for each output
typedef enum logic [1:0] {
  DELAYED_NONE  = 2'b00,
  DELAYED_SET   = 2'b01,
  DELAYED_CLEAR = 2'b10
} delayed_op_t;

delayed_op_t delayed_op_type [NUM_OUTPUTS-1:0];

// FSM control signals per output
logic [NUM_OUTPUTS-1:0] fsm_start;
logic [31:0] fsm_delay_cycles;
logic [NUM_OUTPUTS-1:0] fsm_busy;
logic [NUM_OUTPUTS-1:0] fsm_done;
logic [NUM_OUTPUTS-1:0] fsm_io_busy;

//------------------------------------------------------------------------------
// Cross Clock Domain (APB to Processing)
//------------------------------------------------------------------------------

// Synchronize control signals to processing clock domain
logic        global_enable_proc;
logic [31:0] delay_value_us_proc;

// Simple 2FF synchronizers for CDC
always_ff @(posedge proc_clk or negedge proc_rst_n) begin
  if (!proc_rst_n) begin
    global_enable_proc <= 1'b0;
    delay_value_us_proc <= 32'd1000;
  end else begin
    global_enable_proc <= global_enable;
    delay_value_us_proc <= delay_value_us;
  end
end

// Synchronize immediate operation pulses (single cycle in APB domain)
logic [31:0] immediate_set_sync;
logic [31:0] immediate_clear_sync;
logic [31:0] cancel_delayed_sync;
logic [31:0] delayed_set_sync;
logic [31:0] delayed_clear_sync;

always_ff @(posedge proc_clk or negedge proc_rst_n) begin
  if (!proc_rst_n) begin
    immediate_set_sync <= 32'h0;
    immediate_clear_sync <= 32'h0;
    cancel_delayed_sync <= 32'h0;
    delayed_set_sync <= 32'h0;
    delayed_clear_sync <= 32'h0;
  end else begin
    immediate_set_sync <= immediate_set_pulse;
    immediate_clear_sync <= immediate_clear_pulse;
    cancel_delayed_sync <= cancel_delayed_pulse;
    delayed_set_sync <= delayed_set_pulse;
    delayed_clear_sync <= delayed_clear_pulse;
  end
end

//------------------------------------------------------------------------------
// Calculate FSM timing parameters
//------------------------------------------------------------------------------

// Convert microseconds to clock cycles
assign fsm_delay_cycles = delay_value_us_proc * CYCLES_PER_US;

//------------------------------------------------------------------------------
// Generate logic for each output using FSM
//------------------------------------------------------------------------------

genvar i;
generate
  for (i = 0; i < NUM_OUTPUTS; i++) begin : gen_outputs
    
    // Per-output control logic
    always_ff @(posedge proc_clk or negedge proc_rst_n) begin
      if (!proc_rst_n) begin
        output_states_bits[i] <= 1'b0;
        delayed_op_type[i] <= DELAYED_NONE;
        fsm_start[i] <= 1'b0;
      end else begin
        // Handle immediate operations (highest priority - cancels delayed operations)
        if (immediate_set_sync[i]) begin
          fsm_start[i] <= 1'b0;  // Cancel any FSM operation
          delayed_op_type[i] <= DELAYED_NONE;
          output_states_bits[i] <= 1'b1;
        end
        else if (immediate_clear_sync[i]) begin
          fsm_start[i] <= 1'b0;  // Cancel any FSM operation
          delayed_op_type[i] <= DELAYED_NONE;
          output_states_bits[i] <= 1'b0;
        end
        // Handle cancel delayed operation (cancel without changing output state)
        else if (cancel_delayed_sync[i]) begin
          fsm_start[i] <= 1'b0;  // Cancel any FSM operation
          delayed_op_type[i] <= DELAYED_NONE;
          // Output state remains unchanged
        end
        // Handle delayed set operations
        else if (delayed_set_sync[i] && global_enable_proc) begin
          fsm_start[i] <= 1'b1;  // Start FSM
          delayed_op_type[i] <= DELAYED_SET;
        end
        // Handle delayed clear operations  
        else if (delayed_clear_sync[i] && global_enable_proc) begin
          fsm_start[i] <= 1'b1;  // Start FSM
          delayed_op_type[i] <= DELAYED_CLEAR;
        end
        // Clear start signal after one cycle
        else begin
          fsm_start[i] <= 1'b0;
        end
        
        // Execute delayed operation when FSM completes
        if (fsm_done[i]) begin
          case (delayed_op_type[i])
            DELAYED_SET:   output_states_bits[i] <= 1'b1;
            DELAYED_CLEAR: output_states_bits[i] <= 1'b0;
            default:       ; // No operation
          endcase
          delayed_op_type[i] <= DELAYED_NONE;
        end
        
        // Global disable - stop all operations but preserve output states
        if (!global_enable_proc) begin
          fsm_start[i] <= 1'b0;
          delayed_op_type[i] <= DELAYED_NONE;
        end
      end
    end
    
    // Instantiate FSM for each output
    FPGA_delayed_io_ctrl_fsm #(
      .NUM_OUTPUTS ( 1 )
    ) u_fsm (
      .clk              ( proc_clk           ),
      .rst_n            ( proc_rst_n         ),
      .i_start          ( fsm_start[i]       ),
      .i_delay_cycles   ( fsm_delay_cycles   ),
      .o_busy           ( fsm_busy[i]        ),
      .o_done           ( fsm_done[i]        ),
      .o_io_pins        (                    ), // Not used
      .o_io_busy        ( fsm_io_busy[i]     )
    );
    
    // Combine immediate and delayed outputs
    logic final_output_state;
    assign final_output_state = output_states_bits[i];
    
    // Apply global enable (no polarity control in original spec)
    assign o_io_pins[i] = global_enable_proc ? final_output_state : 1'b0;
    assign o_io_busy[i] = fsm_busy[i] || (delayed_op_type[i] != DELAYED_NONE);
    
    // Update countdown active status
    assign countdown_active_bits[i] = fsm_busy[i];
    
  end
endgenerate

//------------------------------------------------------------------------------
// Status register updates (sync back to APB clock domain)
//------------------------------------------------------------------------------

always_ff @(posedge cmd_clk or negedge cmd_rst_n) begin
  if (!cmd_rst_n) begin
    output_states <= 32'h0;
    countdown_active <= 32'h0;
  end else begin
    // Synchronize status from processing domain
    output_states[NUM_OUTPUTS-1:0] <= output_states_bits;
    countdown_active[NUM_OUTPUTS-1:0] <= countdown_active_bits;
    
    // Clear unused bits
    if (NUM_OUTPUTS < 32) begin
      output_states[31:NUM_OUTPUTS] <= '0;
      countdown_active[31:NUM_OUTPUTS] <= '0;
    end
  end
end

endmodule
