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

module debouncer  #(
  parameter                     DEBOUNCE_CNT = 2000000 )
(
    input   logic                 clk,
    input   logic                 rst,

    input   logic                 debounce_in,
    output  logic                 debounce_out
);

localparam COUNT_MSB = $clog2(DEBOUNCE_CNT);

logic [1:0]           debounce_reg;
logic                 start_count;
logic [COUNT_MSB-1:0] debounce_counter;

always_ff @(posedge clk) begin
  if (rst) begin
    debounce_reg                <= 0;
  end
  else begin
    debounce_reg                <= {debounce_reg[0], debounce_in};
  end
end

assign start_count              = debounce_reg[1] ^ debounce_reg[0];


always_ff @(posedge clk) begin
  if (rst) begin
    debounce_counter            <= 0;
    debounce_out                <= 0;
  end
  else begin
    if (start_count) begin
      debounce_counter          <= 0;
    end
    else if (debounce_counter < DEBOUNCE_CNT) begin
      debounce_counter          <= debounce_counter + 1'b1;
    end
    else begin
      debounce_out              <= debounce_reg[1];
    end
  end
end

endmodule
