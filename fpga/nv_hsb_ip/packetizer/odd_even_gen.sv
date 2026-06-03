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

module odd_even_gen # (
    parameter D_WIDTH    = 64,
    parameter PIPE_DEPTH = 3,
    parameter SORT_DEPTH = 21,
    parameter RESOLUTION = 1,
    parameter SEL_WIDTH  = 543
) (
    input                  clk,
    input  [D_WIDTH-1:0]   din,
    input  [SEL_WIDTH-1:0] sel,
    output [D_WIDTH-1:0]   dout,
    input                  bypass
);

localparam SORT_WIDTH = D_WIDTH/RESOLUTION;

integer p,k,j,i,m,n,r;
integer x,y;
logic [D_WIDTH-1:0]      d [SORT_DEPTH-1:0];
logic [D_WIDTH-1:0]      dq [SORT_DEPTH-1+1:0];
logic [SEL_WIDTH-1:0]    selq [SORT_DEPTH+1:0];
logic [SORT_DEPTH-1+1:0] bq;

always_comb begin
  n = 0;
  m = 0;  // Stage
  for (p=1;p<SORT_WIDTH;p=p+p) begin
    for (k=p;k>0;k=k>>1) begin
      d[m] = dq[m];
      for (j=0;j<k;j=j+1) begin
        for (i=k%p;(i+k)<SORT_WIDTH;i=i+k+k) begin
          if ((i + j)/(p + p) == (i + j + k)/(p + p)) begin
            // Swap, increment select index
            for (r=0;r<RESOLUTION;r=r+1) begin
              x = (i+j+k)*(RESOLUTION)+r;
              y = (i+j)*(RESOLUTION)+r;
              {d[m][y],d[m][x]} = (selq[m][n]&!bq[m]) ? {dq[m][x],dq[m][y]} : {dq[m][y],dq[m][x]};
            end
            n = n+1;
          end
        end
      end
      m=m+1;  // Increment stage count
    end
  end
end


genvar q;
generate
for (q=0;q<SORT_DEPTH;q=q+1) begin
  if ((q%PIPE_DEPTH)==(PIPE_DEPTH-1) || (SORT_DEPTH==1)) begin // Pipeline every 3 stages
    always_ff @(posedge clk) begin
      dq[q+1]   <= d[q];
      bq[q+1]   <= bq[q];
      selq[q+1] <= selq[q];
    end
  end
  else begin
    assign dq[q+1]   = d[q];
    assign bq[q+1]   = bq[q];
    assign selq[q+1] = selq[q];
  end
end
endgenerate


assign selq[0]  = sel;
assign dq[0]    = din;
assign bq[0]    = bypass;
assign dout     = (SORT_DEPTH == 1) ? dq[SORT_DEPTH] : d[SORT_DEPTH-1];

endmodule


