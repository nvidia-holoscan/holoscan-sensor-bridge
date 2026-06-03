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

module ptp_calc #(
  parameter W_DLY = 32,
  parameter W_OFM = 32
)(
  input                     i_pclk,
  input                     i_prst,

  //Config
  input         [1:0]       i_cfg_avg_fact,
  input         [31:0]      i_cfg_dly_asymm,
  input         [W_DLY-1:0] i_ip_dly_asymm,
  input                     i_P2P,

  //Sync Timestamps
  input         [79:0]      i_sync_t1_ts,
  input         [79:0]      i_sync_t2_ts,
  input         [47:0]      i_sync_cf_ns,
  input                     i_sync_ld,
  //Offset Measurement
  output signed [W_OFM-1:0] o_ofm,
  output                    o_ofm_vld,

  //Delay Timestamps
  input         [79:0]      i_dly_t1_ts,
  input         [79:0]      i_dly_t2_ts,
  input         [79:0]      i_dly_t3_ts,
  input         [79:0]      i_dly_t4_ts,
  input         [47:0]      i_dly_cf_ns,
  input                     i_dly_ld,
  output        [W_DLY-1:0] o_mean_dly,
  output                    o_mean_dly_vld

);

localparam [31:0] BILLION = 10**9;

enum logic [3:0] {
  SYNC_IDLE,
  SYNC_TS1_MULTI,
  SYNC_TS2_MULTI,
  SYNC_SUB,
  SYNC_CF,
  SYNC_CORR,
  SYNC_ADD_DLY
} sync_state;

enum logic [3:0] {
  DLY_IDLE,
  DLY_TS1_MULTI,
  DLY_TS2_MULTI,
  DLY_TS3_MULTI,
  DLY_TS4_MULTI,
  DLY_SUB_SM,
  DLY_CF_SM,
  DLY_SUB_MS,
  DLY_ADD_SM_MS,
  DLY_ABS,
  DLY_MEAN
} dly_state;

logic [2:0] sync_timer;
logic       sync_timer_done;
assign sync_timer_done = sync_timer[2];

logic [2:0] dly_timer;
logic       dly_timer_done;
assign dly_timer_done = dly_timer[2];

logic [79:0] sync_ts;
logic [79:0] sync_ts_r;
logic [79:0] dly_ts;
logic [79:0] dly_ts_r;

// DSP INPUT PIPELINE
always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    sync_ts_r <= 0;
    dly_ts_r  <= 0;
  end
  else begin
    sync_ts_r <= sync_ts;
    dly_ts_r  <= dly_ts;
  end
end

logic [63:0] sync_mult;
logic [63:0] dly_mult;
logic [63:0] sync_mult_ts_f;
logic [63:0] dly_mult_ts_f;

assign sync_mult = sync_ts_r[63:32] * BILLION;
assign dly_mult  = dly_ts_r[63:32]  * BILLION;

// DSP MULT PIPELINE
always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    sync_mult_ts_f  <= 0;
    dly_mult_ts_f   <= 0;
  end
  else begin
    sync_mult_ts_f  <= sync_mult;
    dly_mult_ts_f   <= dly_mult;
  end
end

logic          [63:0] sync_ts_add;
logic          [63:0] dly_ts_add;

logic unsigned [63:0] sync_ts_fp;
logic unsigned [63:0] dly_ts_fp;

assign sync_ts_add = sync_mult_ts_f + sync_ts_r[29:0];
assign dly_ts_add  = dly_mult_ts_f  + dly_ts_r[29:0];

// DSP OUTPUT PIPELINE
always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    sync_ts_fp <= 0;
    dly_ts_fp  <= 0;
  end
  else begin
    sync_ts_fp <= sync_ts_add;
    dly_ts_fp  <= dly_ts_add;
  end
end

logic        [63:0]      sync_ts1_fp;
logic        [63:0]      sync_ts2_fp;
logic signed [W_OFM-1:0] sync_diff;
logic signed [W_OFM-1:0] corr_sync;
logic signed [W_OFM-1:0] ofm_no_dly;
logic signed [W_OFM-1:0] ofm;
logic                    ofm_vld;

logic        [63:0]      dly_ts1_fp;
logic        [63:0]      dly_ts2_fp;
logic        [63:0]      dly_ts3_fp;
logic        [63:0]      dly_ts4_fp;
logic        [W_DLY-1:0] mean_dly;
logic                    mean_dly_vld;
logic        [W_DLY-1:0] mean_dly_avg;
logic                    mean_dly_avg_vld;

logic signed [W_OFM-1:0] dly_sm;
logic signed [W_OFM-1:0] dly_ms;
logic signed [W_OFM-1:0] dly_diff;
logic signed [W_DLY-1:0] dly;
logic        [W_DLY-1:0] dly_asymm;

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    dly_asymm <= '0;
  end
  else begin
    dly_asymm <= i_cfg_dly_asymm[W_DLY-1:0] + i_ip_dly_asymm[W_DLY-1:0];
  end
end
always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    sync_timer   <= 1'b0;
    sync_ts      <= '0;
    sync_ts1_fp  <= '0;
    sync_ts2_fp  <= '0;
    sync_diff    <= '0;
    corr_sync    <= '0;
    ofm_no_dly   <= '0;
    ofm          <= '0;
    ofm_vld      <= 1'b0;
    sync_state   <= SYNC_IDLE;
  end
  else begin
    case (sync_state)
    SYNC_IDLE: begin
      ofm_vld      <= 1'b0;
      if (i_sync_ld) begin
        sync_ts    <= i_sync_t1_ts;
        sync_state <= SYNC_TS1_MULTI;
      end
    end
    SYNC_TS1_MULTI: begin
      sync_timer <= sync_timer + 1'b1;
      if (sync_timer_done) begin
        sync_timer  <= 0;
        sync_ts1_fp <= sync_ts_fp;
        sync_ts     <= i_sync_t2_ts;
        sync_state  <= SYNC_TS2_MULTI;
      end
    end
      SYNC_TS2_MULTI: begin
      sync_timer <= sync_timer + 1'b1;
      if (sync_timer_done) begin
        sync_timer  <= 0;
        sync_ts2_fp <= sync_ts_fp;
        sync_state  <= SYNC_SUB;
      end
    end
    SYNC_SUB: begin
      sync_diff    <= sync_ts1_fp[W_OFM-1:0] - sync_ts2_fp[W_OFM-1:0];
      sync_state   <= SYNC_CF;
    end
    SYNC_CF: begin
      ofm_no_dly  <= signed'(sync_diff) + signed'({1'b0,i_sync_cf_ns[W_DLY-1:0]});
      sync_state  <= SYNC_CORR;
    end
    SYNC_CORR: begin
      corr_sync   <= signed'(ofm_no_dly) + signed'({1'b0,dly_asymm[W_DLY-1:0]});
      sync_state  <= SYNC_ADD_DLY;
    end
    SYNC_ADD_DLY: begin
      ofm         <= signed'(corr_sync) + signed'(mean_dly_avg);
      ofm_vld     <= 1'b1;
      sync_state  <= SYNC_IDLE;
    end
    endcase
  end
end

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    dly_timer    <= 1'b0;
    dly_ts       <= '0;
    dly_ts1_fp   <= '0;
    dly_ts2_fp   <= '0;
    dly_ts3_fp   <= '0;
    dly_ts4_fp   <= '0;
    dly_diff     <= '0;
    dly_sm       <= '0;
    dly_ms       <= '0;
    dly          <= '0;
    mean_dly     <= '0;
    mean_dly_vld <= 1'b0;
    dly_state    <= DLY_IDLE;
  end
  else begin
    case (dly_state)
      DLY_IDLE: begin
        mean_dly_vld <= 1'b0;
        dly_ms       <= ofm_no_dly;
        if (i_dly_ld) begin
          dly_ts    <= i_dly_t1_ts;
          dly_state <= DLY_TS1_MULTI;
        end
      end
      DLY_TS1_MULTI: begin
        dly_timer <= dly_timer + 1'b1;;
        if (dly_timer_done) begin
          dly_timer  <= 0;
          dly_ts     <= i_dly_t2_ts;
          dly_ts1_fp <= dly_ts_fp;
          dly_state  <= DLY_TS2_MULTI;
        end
      end
      DLY_TS2_MULTI: begin
        dly_timer <= dly_timer + 1'b1;;
        if (dly_timer_done) begin
          dly_timer  <= 0;
          dly_ts2_fp <= dly_ts_fp;
          dly_state  <= DLY_SUB_SM;
        end
      end
      DLY_SUB_SM: begin
        dly_diff  <= dly_ts1_fp[W_OFM-1:0] - dly_ts2_fp[W_OFM-1:0];
        dly_state <= DLY_CF_SM;
      end
      DLY_CF_SM: begin
        dly_sm    <= signed'(dly_diff) + signed'({1'b0,i_dly_cf_ns[W_DLY-1:0]});
        if (i_P2P) begin
          dly_ts    <= i_dly_t3_ts;
          dly_state <= DLY_TS3_MULTI;
        end
        else begin
          dly_state <= DLY_ADD_SM_MS;
        end
      end
      DLY_TS3_MULTI: begin
        dly_timer <= dly_timer + 1'b1;;
        if (dly_timer_done) begin
          dly_timer  <= 0;
          dly_ts     <= i_dly_t4_ts;
          dly_ts3_fp <= dly_ts_fp;
          dly_state  <= DLY_TS4_MULTI;
        end
      end
      DLY_TS4_MULTI: begin
        dly_timer <= dly_timer + 1'b1;;
        if (dly_timer_done) begin
          dly_timer  <= 0;
          dly_ts4_fp <= dly_ts_fp;
          dly_state  <= DLY_SUB_MS;
        end
      end
      DLY_SUB_MS: begin
        dly_ms    <= dly_ts3_fp[W_OFM-1:0] - dly_ts4_fp[W_OFM-1:0];
        dly_state <= DLY_ADD_SM_MS;
      end
      DLY_ADD_SM_MS: begin
        dly       <= signed'(dly_sm) + signed'(dly_ms);
        dly_state <= DLY_ABS;
      end
      DLY_ABS: begin
        if (dly[W_DLY-1]) begin
          dly <= -signed'(dly);
        end
        dly_state <= DLY_MEAN;
      end
      DLY_MEAN: begin
        mean_dly     <= dly >> 1;
        mean_dly_vld <= 1'b1;
        dly_state    <= DLY_IDLE;
      end
    endcase
  end
end

moving_avg #(
  .W_DATA         ( W_DLY ),
  .MAX_AVG_FACTOR ( 3     )
) u_moving_avg (
  .i_clk          ( i_pclk              ),
  .i_rst          ( i_prst              ),
  .i_cfg_avg_fact ( i_cfg_avg_fact[1:0] ),
  .i_data         ( mean_dly            ),
  .i_data_vld     ( mean_dly_vld        ),
  .o_data         ( mean_dly_avg        ),
  .o_data_vld     ( mean_dly_avg_vld    )
);

assign o_mean_dly     = mean_dly_avg;
assign o_mean_dly_vld = mean_dly_avg_vld;

assign o_ofm          = ofm;
assign o_ofm_vld      = ofm_vld;

endmodule
