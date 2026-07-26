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
  parameter WIDTH      = 1,
  parameter PORT_EN    = {WIDTH{1'b1}},
  parameter STICKY_EN  = 0,              // Enable sticky grant, will only send new grant on idle cycle
  localparam PTR_W = (WIDTH <= 1) ? 1 : $clog2(WIDTH)
)
(
  input  logic             clk,    // Clock
  input  logic             rst_n,  // Asynchronous reset active low
  input  logic             rst,    // Synchronous reset active high
  input  logic             idle,   // Only allow new grants when idle. Tie to 1 to grant new req at any time.
  input  logic [WIDTH-1:0] req,    // vector of requests
  output logic [WIDTH-1:0] gnt,    // onehot0 vector of grants
  output logic [PTR_W-1:0] gnt_idx // index of grant
);

localparam GROUP_SIZE = 8;
localparam LOCAL_W    = 3;
localparam ARB_WIDTH  = (WIDTH <= 1) ? 1 : (1 << $clog2(WIDTH));
localparam N_GROUP    = (ARB_WIDTH + GROUP_SIZE - 1) / GROUP_SIZE;
localparam GROUP_W    = (N_GROUP <= 1) ? 1 : $clog2(N_GROUP);
localparam PAD_WIDTH  = N_GROUP * GROUP_SIZE;

logic [WIDTH-1:0]      gnt_d1;   // grant vector from last cycle.
logic [PTR_W-1:0]      gnt_idx_d1;
logic [PTR_W-1:0]      ptr;      // first request index to check for round-robin priority.
logic [PTR_W-1:0]      next_ptr;
logic [GROUP_W-1:0]    ptr_group;
logic [GROUP_W-1:0]    next_ptr_group;
logic [GROUP_SIZE-1:0] ptr_local_mask;
logic [GROUP_SIZE-1:0] next_ptr_local_mask;

logic [PAD_WIDTH-1:0]  req_pad;
logic [PAD_WIDTH-1:0]  gnt_pad;
logic [GROUP_SIZE-1:0] group_req     [N_GROUP];
logic [GROUP_SIZE-1:0] group_gnt     [N_GROUP];
logic [LOCAL_W-1:0]    group_idx     [N_GROUP];
logic [N_GROUP-1:0]    group_valid;
logic [N_GROUP-1:0]    group_after_ptr;
logic [N_GROUP-1:0]    group_before_ptr;

logic [GROUP_SIZE-1:0] ptr_hi_req;
logic [GROUP_SIZE-1:0] ptr_lo_req;
logic [GROUP_SIZE-1:0] ptr_hi_gnt;
logic [GROUP_SIZE-1:0] ptr_lo_gnt;
logic [LOCAL_W-1:0]    ptr_hi_idx;
logic [LOCAL_W-1:0]    ptr_lo_idx;
logic                  ptr_hi_valid;
logic                  ptr_lo_valid;

logic [GROUP_W-1:0]    after_group_idx;
logic [GROUP_W-1:0]    before_group_idx;
logic                  after_group_valid;
logic                  before_group_valid;

logic [GROUP_SIZE-1:0] win_local_gnt;
logic [LOCAL_W-1:0]    win_local_idx;
logic [GROUP_W-1:0]    win_group_idx;
logic [PTR_W-1:0]      win_idx;
logic                  win_valid;
logic [PTR_W-1:0]      update_idx;
logic                  update_valid;

logic                  lock;   // set if the granted request from last cycle is still asserted

function automatic [GROUP_SIZE+LOCAL_W:0] enc_local(input logic [GROUP_SIZE-1:0] din);
  logic [GROUP_SIZE-1:0] onehot;
  logic [LOCAL_W-1:0]    idx;
  logic                  valid;
  begin
    onehot = '0;
    idx    = '0;
    valid  = |din;
    for (int i=GROUP_SIZE-1;i>=0;i=i-1) begin
      if (din[i]) begin
        onehot    = '0;
        onehot[i] = 1'b1;
        idx       = LOCAL_W'(i);
      end
    end
    enc_local = {valid,idx,onehot};
  end
endfunction

function automatic [GROUP_W:0] enc_group(input logic [N_GROUP-1:0] din);
  logic [GROUP_W-1:0] idx;
  logic               valid;
  begin
    idx   = '0;
    valid = |din;
    for (int i=N_GROUP-1;i>=0;i=i-1) begin
      if (din[i]) begin
        idx = GROUP_W'(i);
      end
    end
    enc_group = {valid,idx};
  end
endfunction


always_ff @(posedge clk) begin
  if (!rst_n) begin
    gnt_d1         <= '0;
    gnt_idx_d1     <= '0;
    ptr            <= '0;
    ptr_group      <= '0;
    ptr_local_mask <= '1;
  end
  else if (rst) begin
    gnt_d1         <= '0;
    gnt_idx_d1     <= '0;
    ptr            <= '0;
    ptr_group      <= '0;
    ptr_local_mask <= '1;
  end
  else begin
    gnt_d1         <= gnt;
    gnt_idx_d1     <= gnt_idx;
    if (idle && update_valid) begin
      ptr            <= next_ptr;
      ptr_group      <= next_ptr_group;
      ptr_local_mask <= next_ptr_local_mask;
    end
  end
end

always_comb begin : comb_gnt
  lock = ((|(gnt_d1 & req)) || (STICKY_EN && !idle));
  gnt  = lock          ? gnt_d1                  : // Is still asserted or no new grant for Idle
         !idle         ? '0                      : // Is not still asserted and is not idle
                         gnt_pad[0+:WIDTH]       ; // Flat-equivalent grouped round-robin grant
  gnt_idx = lock       ? gnt_idx_d1              :
            !idle      ? '0                      :
                         win_idx                 ;
end

always_comb begin
  req_pad          = '0;
  req_pad[0+:WIDTH] = req;

  ptr_hi_req       = '0;
  ptr_lo_req       = '0;
  ptr_hi_gnt       = '0;
  ptr_lo_gnt       = '0;
  ptr_hi_idx       = '0;
  ptr_lo_idx       = '0;
  ptr_hi_valid     = '0;
  ptr_lo_valid     = '0;
  group_after_ptr  = '0;
  group_before_ptr = '0;
  after_group_idx  = '0;
  before_group_idx = '0;
  after_group_valid  = '0;
  before_group_valid = '0;
  win_local_gnt    = '0;
  win_local_idx    = '0;
  win_group_idx    = '0;
  win_valid        = '0;
  gnt_pad          = '0;

  for (int g=0;g<N_GROUP;g=g+1) begin
    group_req[g]   = req_pad[g*GROUP_SIZE+:GROUP_SIZE];
    {group_valid[g],group_idx[g],group_gnt[g]} = enc_local(group_req[g]);
    group_after_ptr[g]  = group_valid[g] && (g > ptr_group);
    group_before_ptr[g] = group_valid[g] && (g < ptr_group);
  end

  ptr_hi_req = group_req[ptr_group] & ptr_local_mask;
  ptr_lo_req = group_req[ptr_group] & ~ptr_local_mask;
  {ptr_hi_valid,ptr_hi_idx,ptr_hi_gnt} = enc_local(ptr_hi_req);
  {ptr_lo_valid,ptr_lo_idx,ptr_lo_gnt} = enc_local(ptr_lo_req);
  {after_group_valid,after_group_idx} = enc_group(group_after_ptr);
  {before_group_valid,before_group_idx} = enc_group(group_before_ptr);

  if (ptr_hi_valid) begin
    win_valid     = 1'b1;
    win_group_idx = ptr_group;
    win_local_idx = ptr_hi_idx;
    win_local_gnt = ptr_hi_gnt;
  end
  else if (after_group_valid) begin
    win_valid     = 1'b1;
    win_group_idx = after_group_idx;
    win_local_idx = group_idx[after_group_idx];
    win_local_gnt = group_gnt[after_group_idx];
  end
  else if (before_group_valid) begin
    win_valid     = 1'b1;
    win_group_idx = before_group_idx;
    win_local_idx = group_idx[before_group_idx];
    win_local_gnt = group_gnt[before_group_idx];
  end
  else if (ptr_lo_valid) begin
    win_valid     = 1'b1;
    win_group_idx = ptr_group;
    win_local_idx = ptr_lo_idx;
    win_local_gnt = ptr_lo_gnt;
  end

  if (win_valid) begin
    for (int l=0;l<GROUP_SIZE;l=l+1) begin
      gnt_pad[win_group_idx*GROUP_SIZE+l] = win_local_gnt[l];
    end
  end
  win_idx = PTR_W'({win_group_idx,win_local_idx});
end

always_comb begin
  update_valid        = lock ? |(gnt_d1 & req) : win_valid;
  update_idx          = lock ? gnt_idx_d1      : win_idx;
  next_ptr            = update_idx + 1'b1;
  next_ptr_group      = GROUP_W'(next_ptr >> LOCAL_W);
  next_ptr_local_mask = {GROUP_SIZE{1'b1}} << LOCAL_W'(next_ptr);
end

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
