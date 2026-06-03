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

module comparator  #(
  parameter WIDTH    = 32
)
(
  input                 iClk,
  input                 iRstn,
  input                 en,
  input  [WIDTH-1:0]    iD1,
  input  [WIDTH-1:0]    iD2,
  output                oValid,
  output                oLess, // iD1 < iD2
  output                oEqual, // iD1 = iD2
  output                oGreater   // iD1 > iD2
);

localparam N_BYTES = (WIDTH+7)/8;

typedef enum logic [1:0] {
  S_IDLE = 2'b00,
  S_LOAD = 2'b01,
  S_COMP = 2'b10,
  S_DONE = 2'b11
} comp_state_t;

comp_state_t cur_state, nxt_state;
logic [$clog2(N_BYTES)-1:0]  const_1         = 1;
logic [$clog2(N_BYTES)-1:0]  const_n_minus_1 = N_BYTES-1;

logic rValid, rLess, rEqual, rGreater;

logic [$clog2(N_BYTES)-1:0] rvCnt;
logic                       found;
logic [$clog2(N_BYTES)-1:0] rvPos;

logic [N_BYTES-1:0] rvFlagLe;
logic [N_BYTES-1:0] rvFlagEq;
logic [N_BYTES-1:0] rvFlagGe;

genvar k;

assign oValid   = rValid   ;
assign oLess    = rLess    ;
assign oEqual   = rEqual   ;
assign oGreater = rGreater ;

always_ff @(posedge iClk) begin
  if (~iRstn) begin
    cur_state <= S_IDLE;
  end
  else begin
    cur_state <= nxt_state;
  end
end

always_comb begin
  case(cur_state)
    S_IDLE: begin
      if (en) begin
        nxt_state = S_LOAD;
      end
      else begin
        nxt_state = S_IDLE;
      end
    end
    S_LOAD: begin
      nxt_state = S_COMP;
    end
    S_COMP: begin
      nxt_state = found ? S_DONE : S_COMP;
    end
    S_DONE: begin
      nxt_state = S_IDLE;
    end
    default: begin
      nxt_state = S_IDLE;
    end
  endcase
end

generate
  for (k=0; k<N_BYTES; k=k+1) begin
    always @(posedge iClk or negedge iRstn) begin
      if (~iRstn) begin
        rvFlagLe[k]   <= 1'b0;
        rvFlagEq[k]   <= 1'b0;
        rvFlagGe[k]   <= 1'b0;
      end
      else begin
        case(nxt_state)
          S_IDLE: begin
            rvFlagLe[k]   <= 1'b0;
            rvFlagEq[k]   <= 1'b0;
            rvFlagGe[k]   <= 1'b0;
          end
          S_LOAD: begin
            if (iD1[8*k+:8] < iD2[8*k+:8]) begin
              rvFlagLe[k] <= 1'b1;
            end
            else if (iD1[8*k+:8] == iD2[8*k+:8]) begin
              rvFlagEq[k] <= 1'b1;
            end
            else if (iD1[8*k+:8] > iD2[8*k+:8]) begin
              rvFlagGe[k] <= 1'b1;
            end
          end
          default: begin
            rvFlagLe[k]   <= rvFlagLe[k];
            rvFlagEq[k]   <= rvFlagEq[k];
            rvFlagGe[k]   <= rvFlagGe[k];
          end
        endcase
      end
    end
  end
endgenerate

always@(posedge iClk or negedge iRstn) begin
  if (~iRstn) begin
    found   <= 1'b0;
    rvPos   <= '0;
    rvCnt   <= const_n_minus_1;
  end
  else begin
    case(cur_state)
      S_COMP: begin
        if (|rvCnt == 1'b0) begin
          rvCnt <= '0;
        end
        else if (!found && rvFlagEq[rvCnt] == 1'b1) begin
          rvCnt <= rvCnt-1'b1;
        end
        if (&rvFlagEq == 1'b1) begin
          found <= 1'b1;
        end
        else if (rvFlagEq[rvCnt] == 1'b0 && !found) begin
          rvPos <= rvCnt;
          found <= 1'b1;
        end
      end
      default: begin
        found   <= 1'b0;
        rvPos   <= '0;
        rvCnt   <= const_n_minus_1;
      end
    endcase
  end
end

always@(posedge iClk or negedge iRstn) begin
  if (~iRstn) begin
    rValid     <= 1'b0;
    rLess      <= 1'b0;
    rEqual     <= 1'b0;
    rGreater   <= 1'b0;
  end
  else begin
    case(nxt_state)
      S_DONE: begin
        rValid   <= 1'b1;
        if (&rvFlagEq) begin
          rLess      <= 1'b0;
          rEqual     <= 1'b1;
          rGreater   <= 1'b0;
        end
        else begin
          rEqual     <= 1'b0;
          if (rvFlagLe[rvPos]) begin
            rLess    <= 1'b1;
          end
          else if (rvFlagGe[rvPos]) begin
            rGreater <= 1'b1;
          end
        end
      end
      default: begin
        rValid     <= 1'b0;
        rLess      <= 1'b0;
        rEqual     <= 1'b0;
        rGreater   <= 1'b0;
      end
    endcase
  end
end

`ifdef ASSERT_ON
  asrt_onehot_le_eq_ge : assert property (@ (posedge iClk) disable iff (!iRstn) oValid |-> $onehot({oGreater, oEqual, oLess}));
`endif

endmodule
