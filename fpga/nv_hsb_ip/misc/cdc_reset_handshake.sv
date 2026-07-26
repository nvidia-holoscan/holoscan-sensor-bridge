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


// CDC Reset Handshake
//
// Ensures dst does not complete handshake until src has reset.
// This prevents false "ready" when dst resets but src does not.
//
// Handshake Flow:
//   1. dst comes out of reset, sees src_ready_sync=0, asserts dest_ack
//   2. src sees dest_ack_sync, but ONLY responds if src_gate=0 (cleared by src_rst)
//   3. src asserts src_ready (one-shot pulse), sets src_gate=1
//   4. dst sees src_ready_sync=1, asserts dest_ready (handshake complete)
//   5. src sees dest_ready_sync=1, clears src_ready
//
// Key Guarantee:
//   - src_gate only clears on src_rst
//   - If dst resets without src resetting, src_gate=1 blocks step 3
//   - Handshake stays incomplete until src also resets
//


module cdc_reset_handshake (
  input  logic            src_clk,
  input  logic            src_rst,
  input  logic            dst_clk,
  input  logic            dst_rst,

  output logic            o_src_init_done,
  output logic            o_dst_init_done
);

logic dest_ready;
logic dest_ready_sync;

logic dest_ack;
logic dest_ack_sync;

logic src_ready;
logic src_ready_sync;

//------------------------------------------------------------------------------------------------//
// Source clock domain
//------------------------------------------------------------------------------------------------//

// Prevent src from responding to an ACK until src has been reset
logic src_gate;

data_sync #(
  .DATA_WIDTH                       (1                 ),
  .RESET_VALUE                      (1'b0              ),
  .SYNC_DEPTH                       (2                 )
) dst_ack_to_src_sync (
  .clk                              (src_clk           ),
  .rst_n                            (!src_rst          ),
  .sync_in                          (dest_ack          ),
  .sync_out                         (dest_ack_sync     )
);

data_sync #(
  .DATA_WIDTH                       (1                 ),
  .RESET_VALUE                      (1'b0              ),
  .SYNC_DEPTH                       (2                 )
) dst_ready_to_src_sync (
  .clk                              (src_clk           ),
  .rst_n                            (!src_rst          ),
  .sync_in                          (dest_ready        ),
  .sync_out                         (dest_ready_sync   )
);

always_ff @(posedge src_clk) begin
  if (src_rst) begin
    src_gate  <= '0;
    src_ready <= '0;
  end
  else begin
    if (dest_ack_sync && !src_gate) begin
      src_ready <= '1;
      src_gate  <= '1;
    end
    if (dest_ready_sync) begin
      src_ready <= '0;
    end
  end
end


assign o_src_init_done = src_gate & dest_ready_sync;

//------------------------------------------------------------------------------------------------//
// Destination clock domain
//------------------------------------------------------------------------------------------------//

logic dest_gate;

data_sync #(
  .DATA_WIDTH                       (1                      ),
  .RESET_VALUE                      (1'b1                   ),
  .SYNC_DEPTH                       (2                      )
) src_ready_to_dst_sync (
  .clk                              (dst_clk                ),
  .rst_n                            (!dst_rst               ),
  .sync_in                          (src_ready              ),
  .sync_out                         (src_ready_sync         )
);

always_ff @(posedge dst_clk) begin
  if (dst_rst) begin
    dest_ready <= '0;
    dest_ack   <= '0;
    dest_gate  <= '0;
  end
  else begin
    if (!src_ready_sync && !dest_gate) begin
      dest_ack  <= '1;
      dest_gate <= '1;
    end

    if (src_ready_sync && dest_gate) begin
      dest_ready <= '1;
      dest_ack   <= '0;
    end
  end
end

assign o_dst_init_done = dest_ready;

endmodule
