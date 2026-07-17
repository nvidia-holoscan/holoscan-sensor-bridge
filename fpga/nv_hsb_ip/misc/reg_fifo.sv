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

module reg_fifo #(
  parameter DATA_WIDTH = 8,
  parameter DEPTH      = 2
) (
  input                   clk,
  input                   rst,
  input                   wr,
  input  [DATA_WIDTH-1:0] din,
  output                  full,
  input                   rd,
  output                  dval,
  output [DATA_WIDTH-1:0] dout,
  output                  over,
  output                  under,
  output                  empty
);

logic [DATA_WIDTH-1:0] reg_fifo [DEPTH:0];

logic [DEPTH+1:0] reg_valid;
logic [DEPTH  :0] reg_en   ;


logic wr_en;
logic rd_en;


// First Stage
assign reg_fifo[0] = din;
assign reg_en[0] = wr_en;
assign reg_valid[0] = wr_en;

// Middle Stages
genvar i;
generate
  for (i=1; i<DEPTH+1; i++) begin : gen_fifo_data
    // Data
    always_ff @(posedge clk) begin
      if (reg_en[i]) begin
        reg_fifo[i] <= reg_fifo[i-1];
      end
    end
    // Control
    always_ff @(posedge clk) begin
      if (rst) begin
        reg_valid[i] <= '0;
      end
      else begin
        if (reg_en[i]) begin
          reg_valid[i] <= reg_valid[i-1];
        end
      end
    end
    assign reg_en[i] = !(&reg_valid[DEPTH+1:i]); // Enable when there is space ahead
  end
endgenerate

assign reg_valid[DEPTH+1] = !rd_en;

// Outputs
assign dout = reg_fifo[DEPTH];
assign dval = reg_valid[DEPTH];

assign full = &reg_valid[DEPTH+1:1];
assign empty = !(reg_valid[DEPTH]);

assign wr_en = wr & ~full;
assign rd_en = rd & ~empty;

assign over = wr & full;
assign under = rd & empty;


`ifdef ASSERT_ON
//Assertions
assert_reg_fifo_overflow: assert property ( @(posedge clk) disable iff (rst)
    (!over));

assert_reg_fifo_underflow: assert property ( @(posedge clk) disable iff (rst)
    (!under));
`endif

endmodule
