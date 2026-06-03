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

//This module will synchronize a one-cycle pulse in the source clock domain and create a one cycle pulse
//in the destination clock domain.

module pulse_sync (
  input logic             src_clk,
  input logic             src_rst,
  input logic             dst_clk,
  input logic             dst_rst,

  input logic             i_src_pulse,
  output logic            o_dst_pulse
);

logic   src_init_done;
logic   dst_init_done;

logic   src_pulse_reg;
logic   dst_pulse_sync;
logic   dst_pulse_sync_reg;

//------------------------------------------------------------------------------------------------//
// CDC Reset Handshake
//------------------------------------------------------------------------------------------------//

cdc_reset_handshake u_cdc_reset_handshake (
  .src_clk                          (src_clk           ),
  .src_rst                          (src_rst           ),
  .dst_clk                          (dst_clk           ),
  .dst_rst                          (dst_rst           ),
  .o_src_init_done                  (src_init_done     ),
  .o_dst_init_done                  (dst_init_done     )
);

//------------------------------------------------------------------------------------------------//
// Source clock domain
//------------------------------------------------------------------------------------------------//

always_ff @(posedge src_clk) begin
  if (src_rst) begin
    src_pulse_reg             <= 1'b0;
  end
  else begin
    if (!src_init_done) begin
      src_pulse_reg           <= 1'b0;
    end
    else begin
      src_pulse_reg           <= i_src_pulse ^ src_pulse_reg;
    end
  end
end

//------------------------------------------------------------------------------------------------//
// Destination clock domain
//------------------------------------------------------------------------------------------------//

// Sync pulse toggle into the destination domain
data_sync #(
  .DATA_WIDTH                       (1                      ),
  .RESET_VALUE                      (1'b0                   ),
  .SYNC_DEPTH                       (2                      )
) src_pulse_to_dst_sync (
  .clk                              (dst_clk                ),
  .rst_n                            (!dst_rst               ),
  .sync_in                          (src_pulse_reg          ),
  .sync_out                         (dst_pulse_sync         )
);

always_ff @(posedge dst_clk) begin
  if (dst_rst) begin
    dst_pulse_sync_reg        <= 1'b0;
    o_dst_pulse               <= 1'b0;
  end
  else begin
    dst_pulse_sync_reg        <= dst_pulse_sync;
    if (dst_init_done) begin
      o_dst_pulse             <= dst_pulse_sync_reg ^ dst_pulse_sync;
    end
    else begin
      o_dst_pulse             <= 1'b0;
    end
  end
end

endmodule
