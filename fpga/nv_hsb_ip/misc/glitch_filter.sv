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

// synchronizer and configurable glitch filter

module glitch_filter #(
  parameter DATA_WIDTH    = 1,
  parameter RESET_VALUE   = 1'b0,
  parameter FILTER_DEPTH  = 4
) (
  input                         clk,
  input                         rst_n,
  input      [DATA_WIDTH-1:0]   sync_in,
  output reg [DATA_WIDTH-1:0]   sync_out
);

genvar s_idx;
reg [FILTER_DEPTH-1:0] filter_pipe [DATA_WIDTH-1:0];

generate
  for(s_idx=0; s_idx < DATA_WIDTH; s_idx++) begin: FILTER_SYNC
    always @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        filter_pipe[s_idx] <= {DATA_WIDTH{RESET_VALUE}};
        sync_out   [s_idx] <= RESET_VALUE;
      end
      else begin
        if (&filter_pipe[s_idx]) begin
          sync_out[s_idx] <= 1'b1;
        end
        else if (~|filter_pipe[s_idx]) begin
          sync_out[s_idx] <= 1'b0;
        end
        filter_pipe[s_idx] <= {filter_pipe[s_idx], sync_in[s_idx]};
      end
    end
  end
endgenerate

endmodule
