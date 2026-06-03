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
module FPGA_delayed_io_ctrl_fsm #(
    parameter NUM_OUTPUTS = 1
)(
    input  logic        clk,
    input  logic        rst_n,
    
    // Control inputs
    input  logic        i_start,
    input  logic [31:0] i_delay_cycles,
    
    // Status outputs
    output logic        o_busy,
    output logic        o_done,
    
    // Output control
    output logic [NUM_OUTPUTS-1:0] o_io_pins,
    output logic [NUM_OUTPUTS-1:0] o_io_busy
);

// State machine states
typedef enum logic [1:0] {
    IDLE  = 2'b00,
    DELAY = 2'b01,
    DONE  = 2'b10
} state_t;

state_t state, state_nxt;
logic [31:0] counter, counter_nxt;

// Edge detection for start signal
logic start_prev;
logic start_edge;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        start_prev <= 1'b0;
    end else begin
        start_prev <= i_start;
    end
end

assign start_edge = i_start && !start_prev;

// State machine
always_comb begin
    state_nxt = state;
    counter_nxt = counter;
    
    case (state)
        IDLE: begin
            counter_nxt = 32'h0;
            if (start_edge) begin
                if (i_delay_cycles == 32'h0) begin
                    // No delay, go directly to done
                    state_nxt = DONE;
                end else begin
                    state_nxt = DELAY;
                    counter_nxt = i_delay_cycles;
                end
            end
        end
        
        DELAY: begin
            if (counter == 32'h1) begin
                // Delay finished
                state_nxt = DONE;
                counter_nxt = 32'h0;
            end else begin
                counter_nxt = counter - 1'b1;
            end
        end
        
        DONE: begin
            if (!i_start) begin
                state_nxt = IDLE;
            end
        end
        
        default: begin
            state_nxt = IDLE;
            counter_nxt = 32'h0;
        end
    endcase
end

// Sequential logic
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        counter <= 32'h0;
    end else begin
        state <= state_nxt;
        counter <= counter_nxt;
    end
end

// Output logic
assign o_busy = (state == DELAY);
assign o_done = (state == DONE);

genvar i;
generate
    for (i = 0; i < NUM_OUTPUTS; i++) begin : gen_outputs
        assign o_io_pins[i] = 1'b0;  // FSM doesn't control pins directly
        assign o_io_busy[i] = (state == DELAY);
    end
endgenerate

endmodule
