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

// Combinational round-robin arbiter with sticky grant.  If a request was granted last cycle and request is still being
// held, that request is given priority, else there is round-robin priority. No new grant is sent if not idle.

module rrarb #(
  parameter WIDTH     = 1,
  parameter PORT_EN   = {WIDTH{1'b1}},
  parameter STICKY_EN = 0               // Enable sticky grant, will only send new grant on idle cycle
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
logic [WIDTH-1:0] mask;     // mask indicating which requests are considered first for circular find-first

logic [WIDTH-1:0] in1;
logic [WIDTH-1:0] in2;
logic [WIDTH-1:0] in3;
logic [WIDTH-1:0] out1;
logic [WIDTH-1:0] out2;
logic [WIDTH-1:0] out3;

logic             lock;   // set if the granted request from last cycle is still asserted


always_ff @(posedge clk) begin
  if (!rst_n) begin
    gnt_d1   <= '0;
    mask     <= '0;
  end
  else if (rst) begin
    gnt_d1   <= '0;
    mask     <= '0;
  end
  else begin
    gnt_d1   <= gnt;
    mask     <= (req=='0) ? mask :
                (!idle)   ? mask : // Update mask if idle
                            out3 ;
  end
end

always_comb begin : comb_gnt
  lock = ((|(gnt_d1 & req)) || (STICKY_EN && !idle));
  gnt  = lock          ? gnt_d1                  : // Is still asserted or no new grant for Idle
          !idle         ? '0                      : // Is not still asserted and is not idle
          |(req & mask) ? out1                    : // Find first above index of previous grant
                          out2                    ; // Find first below index of previous grant (circular)
end

genvar i;
generate
  // return onehot0 vector with lowest set bit from input set
  for (i = 1; i < WIDTH; i++) begin
    assign out1[i] = (in1[i] && (in1[i-1:0] == '0));
    assign out2[i] = (in2[i] && (in2[i-1:0] == '0));
  end
  assign out1[0] = in1[0];
  assign out2[0] = in2[0];
  assign in1 = (req & mask);
  assign in2 = (req & ~mask);

  // return vector with bits set which are above the first set bit of input vector
  assign out3[0] = 1'b0;
  for (i = 1; i < WIDTH; i++) begin
    assign out3[i] = (|in3[i-1:0]);
  end
  assign in3 = gnt;

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
          gnt[r] ##1 req[r] |-> gnt[r]
      ); //else $error("If request won grant and request is held, it should continue to win grant");
      if (!STICKY_EN) begin
          asrt_gnt_implies_req : assert property ( @(posedge clk) disable iff (!rst_n)
              gnt[r] |-> req[r]
          ); //else $error("Grant should not be asserted without request");
      end
    end
  end
`endif // ASSERT_ON
endmodule
