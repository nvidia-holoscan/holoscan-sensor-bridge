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

module apb_switch
  import apb_pkg::*;
#(
  parameter      N_MPORT             = 1,
  parameter      N_SPORT             = 2,
  parameter      W_OFSET             = 8, // offset address width
  parameter      W_SW                = 4,
  parameter      MERGE_COMPLETER_SIG = 1 //Merge Completer (pserr, prdata, pready) signals to reduce LUT
)(
  input          i_apb_clk,
  input          i_apb_reset, // synchronous active high reset
  // Connect to Decoder
  input  apb_m2s i_apb_m2s [N_MPORT],
  output apb_s2m o_apb_s2m [N_MPORT],
  // Connect to Modules
  input  apb_s2m i_apb_s2m [N_SPORT],
  output apb_m2s o_apb_m2s [N_SPORT],
  //APB Timeout
  input          i_apb_timeout
);

localparam W_MPORT = N_MPORT <= 1 ? 1 : $clog2(N_MPORT);
localparam W_SPORT = N_SPORT <= 1 ? 1 : $clog2(N_SPORT);

//------------------------------------------------------------------------------------------------//
// Switch Arbitration
//------------------------------------------------------------------------------------------------//

logic                 apb_pready;
logic [W_MPORT - 1:0] apb_midx;
logic [W_SPORT - 1:0] apb_sidx;
logic                 apb_pserr;
logic [         31:0] apb_prdata;

enum logic [1:0] {IDLE, REQUESTED, DELAY} state, state_prev;

//------------------------------------------------------------------------------------------------//
// Requester to Completer signals
//------------------------------------------------------------------------------------------------//
//If there is only one Controller port, then the Controller port index is 0
generate
if (N_MPORT == 1) begin
  assign apb_midx = 0;

  always_comb begin
    o_apb_m2s = '{default:0};
    for (int j=0;j<N_SPORT;j++) begin
      o_apb_m2s[j].penable  = i_apb_m2s[apb_midx].penable;
      o_apb_m2s[j].pwrite   = i_apb_m2s[apb_midx].pwrite;
      o_apb_m2s[j].paddr    = i_apb_m2s[apb_midx].paddr[W_OFSET-1:0];
      o_apb_m2s[j].pwdata   = i_apb_m2s[apb_midx].pwdata;
    end

    o_apb_m2s[apb_sidx].psel = N_SPORT > i_apb_m2s[apb_midx].paddr[W_OFSET+W_SW-1:W_OFSET] ?
                              i_apb_m2s[apb_midx].psel                                    :
                              0;
  end

end
else begin
  //If there are multiple controller ports, then arbitrate
  logic               psel_active;
  logic [W_MPORT-1:0] apb_midx_grant;
  logic [N_MPORT-1:0] apb_midx_req_onehot;
  logic [N_MPORT-1:0] apb_midx_grant_onehot;
  logic [3:0]         delay_cnt;

  //Round Robin
  always_comb begin
    apb_midx_req_onehot = '0;
    for (int i=0; i<N_MPORT; i++) begin
      if (i_apb_m2s[i].psel) begin
        apb_midx_req_onehot[i] = 1'b1;
      end
    end
  end

  rrarb #(
    .WIDTH(N_MPORT),
    .STICKY_EN(0)
  ) rrarb_inst (
    .clk  ( i_apb_clk             ),
    .rst_n( !i_apb_reset          ),
    .rst  ( 1'b0                  ),
    .idle ( (state == IDLE)       ),
    .req  ( apb_midx_req_onehot   ),
    .gnt  ( apb_midx_grant_onehot )
  );

  always_comb begin
    apb_midx = '0;
    for (int i=0; i<N_MPORT; i++) begin
      if (apb_midx_grant_onehot[i]) begin
        apb_midx = i[W_MPORT-1:0];
      end
    end
  end

  assign psel_active = |apb_midx_grant_onehot;

  always @(posedge i_apb_clk) begin
    if (i_apb_reset) begin
      state      <= IDLE;
      state_prev <= IDLE;
      o_apb_m2s  <= '{default:0};
      delay_cnt  <= 0;
    end
    else begin
      state_prev <= state;
        case (state)
          IDLE: begin
          o_apb_m2s <= '{default:0};
          if (psel_active) begin
            state                    <= REQUESTED;
            o_apb_m2s[apb_sidx].psel <= N_SPORT > i_apb_m2s[apb_midx].paddr[W_OFSET+W_SW-1:W_OFSET] ?
                                        i_apb_m2s[apb_midx].psel                                    :
                                        0;
            for (int j=0;j<N_SPORT;j++) begin
              o_apb_m2s[j].pwrite   <= i_apb_m2s[apb_midx].pwrite;
              o_apb_m2s[j].paddr    <= i_apb_m2s[apb_midx].paddr[W_OFSET-1:0];
              o_apb_m2s[j].pwdata   <= i_apb_m2s[apb_midx].pwdata;
            end
          end
          delay_cnt  <= 0;
        end
        REQUESTED: begin
          if (i_apb_timeout || !i_apb_m2s[apb_midx].psel) begin
            o_apb_m2s[apb_sidx].psel    <= 1'b0;
              for (int j=0;j<N_SPORT;j++) begin
                o_apb_m2s[j].penable  <= 1'b0;
                o_apb_m2s[j].pwrite   <= 1'b0;
                o_apb_m2s[j].paddr    <= 0;
                o_apb_m2s[j].pwdata   <= 0;
              end
            state <= DELAY;
        end
        else if (o_apb_m2s[apb_sidx].penable) begin
          if (apb_pready) begin
            o_apb_m2s[apb_sidx].psel    <= 1'b0;
            for (int j=0;j<N_SPORT;j++) begin
              o_apb_m2s[j].penable  <= 1'b0;
              o_apb_m2s[j].pwrite   <= 1'b0;
              o_apb_m2s[j].paddr    <= 0;
              o_apb_m2s[j].pwdata   <= 0;
            end
            state <= DELAY;
          end
          else begin
            for (int j=0;j<N_SPORT;j++) begin
              o_apb_m2s[j].penable  <= 1'b1;
            end
          end
        end
        else begin
          for (int j=0;j<N_SPORT;j++) begin
            o_apb_m2s[j].penable  <= 1'b1;
          end
        end
          delay_cnt <= '0;
        end
        DELAY: begin  // Add delay so that the requested packet can queue up another command to keep arbitration
          delay_cnt  <= delay_cnt + 1;
          state      <= (delay_cnt == 4'd10) ? IDLE : DELAY;
        end
        endcase
    end
  end
end
endgenerate

// Peri index calculation (common for both single and multiple controller cases)
always_comb begin
  apb_sidx = N_SPORT > i_apb_m2s[apb_midx].paddr[W_OFSET+W_SW-1:W_OFSET] ?
              i_apb_m2s[apb_midx].paddr[W_OFSET+W_SW-1:W_OFSET]           :
              0;
end

//------------------------------------------------------------------------------------------------//
// Completer to Requester signals
//------------------------------------------------------------------------------------------------//

if (MERGE_COMPLETER_SIG) begin //Merge Completer signals to reduce LUT
  always_comb begin
    apb_pserr = 0;
    apb_pready = 0;
    apb_prdata = 0;
    for (int k=0;k<N_SPORT;k++) begin
      apb_pserr  = apb_pserr  | i_apb_s2m[k].pserr;
      apb_pready = apb_pready | i_apb_s2m[k].pready;
      apb_prdata = apb_prdata | i_apb_s2m[k].prdata;
    end

    apb_pserr  = i_apb_timeout ? 1'b1         : apb_pserr;
    apb_pready = !i_apb_m2s[apb_midx].penable ? '0 : i_apb_timeout ? 1'b1         : apb_pready;
    apb_prdata = i_apb_timeout ? 32'hBADADD12 : apb_prdata;
  end
end
else if (N_MPORT != 1) begin
  always_comb begin
    apb_pserr  = i_apb_timeout ? 1'b1         : (state == REQUESTED) ? i_apb_s2m[apb_sidx].pserr : 1'b0;
    apb_pready = !i_apb_m2s[apb_midx].penable ? '0 : i_apb_timeout ? 1'b1         :
                  (((state_prev == IDLE) || (state_prev == DELAY)) ? 1'b0 : i_apb_s2m[apb_sidx].pready);
    apb_prdata = i_apb_timeout ? 32'hBADADD12 : i_apb_s2m[apb_sidx].prdata;
  end
end
else begin
  always_comb begin
    apb_pserr  = i_apb_timeout ? 1'b1         : i_apb_s2m[apb_sidx].pserr;
    apb_pready = !i_apb_m2s[apb_midx].penable ? '0 : i_apb_timeout ? 1'b1 : i_apb_s2m[apb_sidx].pready;
    apb_prdata = i_apb_timeout  ? 32'hBADADD12 : i_apb_s2m[apb_sidx].prdata;
  end
end

always_comb begin
  o_apb_s2m = '{default:0};
  o_apb_s2m[apb_midx].pserr  = apb_pserr;
  o_apb_s2m[apb_midx].pready = apb_pready;
  o_apb_s2m[apb_midx].prdata = apb_prdata;
end

endmodule
