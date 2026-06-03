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

// Fixed priority arbiter with sticky grant. Request 0 has the highest priority.
// The !idle signal freezes the arbiter, and no new grant can be issued when not idle (but an already given grant can stay or fall).

module priority_arb
#(
  parameter WIDTH     = 1,
  parameter PORT_EN   = {WIDTH{1'b1}}
)
(
  input  logic             clk,    // Clock
  input  logic             rst_n,  // Asynchronous reset active low
  input  logic             rst,    // Synchronous reset active high
  input  logic             idle,   // Only allow new grants when idle. Tie to 1 to grant new req at any time.
  input  logic [WIDTH-1:0] req,    // vector of requests
  output logic [WIDTH-1:0] gnt     // onehot0 vector of grants
);

logic [WIDTH-1:0] gnt_d1;   // grant vector from last cycle.

logic [WIDTH-1:0] in;
logic [WIDTH-1:0] out;

always_ff @(posedge clk) begin
  if (!rst_n) begin
    gnt_d1   <= '0;
  end
  else if (rst) begin
    gnt_d1   <= '0;
  end
  else begin
    gnt_d1   <= gnt;
  end
end

logic             lock;   // set if the granted request from last cycle is still asserted

always_comb begin : comb_gnt
  lock = |(gnt_d1 & req);
  gnt  = lock          ? gnt_d1                  : // Is still asserted or no new grant for Idle
         !idle         ? '0                      : // Is not still asserted and is not idle
                         out                     ; // Find first index
end

genvar i;
generate
  // return onehot0 vector with lowest set bit from input set
  for (i = 1; i < WIDTH; i++) begin
    assign out[i] = (in[i] && (in[i-1:0] == '0));
  end
  assign out[0] = in[0];
  assign in = (req);

endgenerate

`ifdef ASSERT_ON
  asrt_grant_onehot0 : assert property ( @(posedge clk) disable iff (!rst_n)
      $onehot0(gnt) && $onehot0(gnt_d1)
  ); //else $error("Arbiter gnt vector should be zero/one hot");
  asrt_no_new_gnt_if_not_idle : assert property ( @(posedge clk) disable iff (!rst_n)
       !idle |-> ((gnt == gnt_d1) | (gnt == 'b0))
  ); //else $error("If not idle, grant either falls or doesn't change");
  for (genvar r = 0; r < WIDTH; r++) begin : gen_asrt
    if (PORT_EN[r]) begin: port_r
      `ifndef SIMULATION
      asrt_req_fall_liveness : assert property ( @(posedge clk) disable iff (!rst_n)
          $rose(req[r]) |-> strong ( ##[1:$] $fell(req[r]))
      ); //else $error("request should eventually deassert");
      `endif
      asrt_gnt_lock : assert property ( @(posedge clk) disable iff (!rst_n)
          gnt[r]|-> gnt[r] || !req[r]
      ); //else $error("If request won grant and request is held, it should continue to win grant");
      asrt_gnt_implies_req : assert property ( @(posedge clk) disable iff (!rst_n)
          gnt[r] |-> req[r]
      ); //else $error("Grant should not be asserted without request");
    end
  end
`endif // ASSERT_ON
endmodule
