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

// synchronizer

module data_sync #(
  parameter DATA_WIDTH    = 1,
  parameter RESET_VALUE   = 1'b0,
  parameter SYNC_DEPTH    = 2,
  parameter RST_SYNC      = "FALSE" // "TRUE"|"FALSE"
) (
  input                         clk,
  input                         rst_n,
  input      [DATA_WIDTH-1:0]   sync_in,
  output     [DATA_WIDTH-1:0]   sync_out
);

genvar s_idx;
(* DONT_TOUCH = "TRUE" *) (* ASYNC_REG = "TRUE" *) reg [SYNC_DEPTH-1:0] filter_pipe [DATA_WIDTH-1:0] = '{default: {SYNC_DEPTH{RESET_VALUE}}} /* synthesis syn_preserve=1 CDC_Register=2 syn_maxfan=1 syn_replicate=0 syn_allow_retiming=0 */;

generate
  if (RST_SYNC == "TRUE") begin
    for(s_idx=0; s_idx < DATA_WIDTH; s_idx++) begin: FILTER_SYNC
      always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          filter_pipe[s_idx] <= {SYNC_DEPTH{RESET_VALUE}};
        end
        else begin
          filter_pipe[s_idx] <= {filter_pipe[s_idx][SYNC_DEPTH-2:0], sync_in[s_idx]};
        end
      end
      assign sync_out[s_idx] = filter_pipe[s_idx][SYNC_DEPTH-1];
    end
  end
  else begin
    for(s_idx=0; s_idx < DATA_WIDTH; s_idx++) begin: FILTER_SYNC
      always @(posedge clk) begin
        if (!rst_n) begin
          filter_pipe[s_idx] <= {SYNC_DEPTH{RESET_VALUE}};
        end
        else begin
          filter_pipe[s_idx] <= {filter_pipe[s_idx][SYNC_DEPTH-2:0], sync_in[s_idx]};
        end
      end
      assign sync_out[s_idx] = filter_pipe[s_idx][SYNC_DEPTH-1];
    end
  end
endgenerate

endmodule