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

module ptp_sync (
  input           i_pclk,
  input           i_prst,

  input  [47:0]   i_sec,
  input  [31:0]   i_nano_sec,

  input           i_sync_msg_vld,
  input           i_follow_up_msg_vld,
  input  [15:0]   i_flag,
  input  [47:0]   i_cf_ns,
  input  [79:0]   i_timestamp,
  input  [15:0]   i_seq_id,
  input  [15:0]   i_src_portid,
  input  [63:0]   i_src_clkid,

  output          o_dly_req,
  output [79:0]   o_sync_t1_ts,
  output [79:0]   o_sync_t2_ts,
  output [47:0]   o_sync_cf_ns,
  output          o_sync_ld
);

//------------------------------------------------------------------------------------------------//
// SYNC STATE MACHINE
//------------------------------------------------------------------------------------------------//

enum logic [1:0] {IDLE, SYNC, FOLLOW_UP} state;

logic [79:0] sync_t1_ts;
logic [79:0] sync_t2_ts;
logic [47:0] sync_cf_ns;
logic [15:0] sync_srcportid;
logic [63:0] sync_clkid;
logic [15:0] sync_seq_id;
logic        sync_ld;
logic        dly_req;
logic        two_step;
assign two_step = i_flag[9];

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    dly_req          <= 1'b0;
    sync_seq_id      <= '0;
    sync_clkid       <= '0;
    sync_srcportid   <= '0;
    sync_t1_ts       <= '0;
    sync_t2_ts       <= '0;
    sync_cf_ns       <= '0;
    sync_ld          <= 1'b0;
    state            <= IDLE;
  end
  else begin
    case(state)
    IDLE: begin
      dly_req <= 1'b0;
      sync_ld <= 1'b0;
      if (i_sync_msg_vld) begin
        sync_seq_id    <= i_seq_id;
        sync_srcportid <= i_src_portid;
        sync_clkid     <= i_src_clkid;
        sync_t2_ts     <= {i_sec, i_nano_sec};
        state          <= SYNC;
        if (!two_step) begin
          sync_ld       <= 1'b1;
          dly_req       <= 1'b1;
          sync_cf_ns    <= i_cf_ns;
          sync_t1_ts    <= i_timestamp;
          state         <= IDLE;
        end
      end
    end
    SYNC: begin
      dly_req <= 1'b0;
      sync_ld <= 1'b0;
      if (i_sync_msg_vld) begin
        sync_seq_id    <= i_seq_id;
        sync_srcportid <= i_src_portid;
        sync_clkid     <= i_src_clkid;
        sync_t2_ts     <= {i_sec, i_nano_sec};
        state          <= SYNC;
        if (!two_step) begin
          sync_ld       <= 1'b1;
          dly_req       <= 1'b1;
          sync_cf_ns    <= i_cf_ns;
          sync_t1_ts    <= i_timestamp;
          state         <= IDLE;
        end
      end
        else if (i_follow_up_msg_vld) begin
        if (i_seq_id == sync_seq_id && i_src_portid == sync_srcportid && i_src_clkid == sync_clkid) begin
          dly_req       <= 1'b1;
          sync_ld       <= 1'b1;
          sync_cf_ns    <= i_cf_ns;
          sync_t1_ts    <= i_timestamp;
          state         <= IDLE;
        end
      end
    end
    endcase
  end
end

assign o_sync_t1_ts    = sync_t1_ts;
assign o_sync_t2_ts    = sync_t2_ts;
assign o_sync_cf_ns    = sync_cf_ns;
assign o_sync_ld       = sync_ld;
assign o_dly_req       = dly_req;

endmodule
