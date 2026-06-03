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

module rst_pb_debounce  #(
  parameter                     DEBOUNCE_CNT = 2000000 )
(
    input   logic                 clk,
    input   logic                 pb_rst_in,
    output  logic                 pb_rst_out
);

localparam COUNT_MSB = $clog2(DEBOUNCE_CNT);

logic [2:0]           debounce_sync;
logic [COUNT_MSB-1:0] debounce_counter;
logic [1:0]           pb_synced;

always_ff @(posedge clk) begin
  pb_synced                     <= {pb_synced[0], pb_rst_in};
end

always_ff @(posedge clk or posedge pb_synced[1]) begin
  if (pb_synced[1]) begin
    debounce_counter            <= 0;
  end
  else if (debounce_counter == DEBOUNCE_CNT) begin
    debounce_counter            <= debounce_counter;
  end
  else begin
    debounce_counter            <= debounce_counter + 1'b1;
  end
end

always_ff @(posedge clk) begin
  debounce_sync                 <= {debounce_sync[1:0], (debounce_counter == DEBOUNCE_CNT)};
end

assign pb_rst_out = debounce_sync[2];

endmodule
