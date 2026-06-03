
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

// convert a rising or falling edge to a pulse

module edge_to_pulse #(
  parameter WIDTH = 1,
  parameter PULSE_WIDTH = 1,
  parameter EDGE_TYPE = "RISING" // "RISING" | "FALLING" | "BOTH"
) (
  input              clk,
  input              rst,
  input  [WIDTH-1:0] i_edge,
  output [WIDTH-1:0] o_pulse
);

logic [WIDTH-1:0] edge_reg;
logic [WIDTH-1:0] pulse_reg;

always_ff @(posedge clk) begin
  if (!rst) begin
    edge_reg <= '0;
  end
  else begin
    edge_reg <= i_edge;
  end
end

logic [WIDTH-1:0] posedge_detect;
logic [WIDTH-1:0] negedge_detect;
logic [WIDTH-1:0] edge_detect;
 assign edge_detect = (EDGE_TYPE == "RISING")  ? posedge_detect :
                      (EDGE_TYPE == "FALLING") ? negedge_detect :
                      (posedge_detect | negedge_detect);

generate
  for (genvar i = 0; i < WIDTH; i++) begin
    assign posedge_detect[i] = i_edge[i] & ~edge_reg[i];
    assign negedge_detect[i] = ~i_edge[i] & edge_reg[i];

    logic [$clog2(PULSE_WIDTH)-1:0] pulse_counter;
    always_ff @(posedge clk) begin
      if (!rst) begin
        pulse_reg[i]  <= '0;
        pulse_counter <= '0;
      end
      else begin
        if (edge_detect[i]) begin
            pulse_reg[i]  <= (PULSE_WIDTH > 1) ? '1 : '0;
            pulse_counter <= (PULSE_WIDTH > 1) ? PULSE_WIDTH - 2 : 'b0;
        end
        else if (pulse_counter > 0) begin
            pulse_reg[i]  <= '1;
            pulse_counter <= pulse_counter - 1;
        end
        else begin
            pulse_reg[i] <= '0;
        end
      end
    end
  end
endgenerate

assign o_pulse = edge_detect | pulse_reg;


endmodule
