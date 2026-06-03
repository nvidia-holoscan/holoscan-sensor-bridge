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

module dp_pkt_ts # (
    parameter  N_INPT = 4,
    localparam W_INPT = $clog2(N_INPT)+1
) (
  input                    i_pclk,
  input                    i_prst,
  // Request
  input  [N_INPT-1:0]      i_req,
  // PTP
  input  [79:0]            i_ptp,
  // Handshake
  input  [W_INPT-1:0]      i_ptp_sel,
  input                    i_ram_rd,
  // Output
  output [79:0]            o_ptp
);

localparam W_PTP  = 64;
localparam W_FIFO = W_PTP+N_INPT;

//------------------------------------------------------------------------------------------------//
// Request FIFO
//------------------------------------------------------------------------------------------------//

logic              fifo_rd;
logic              fifo_dval;
logic [W_FIFO-1:0] fifo_dout;
logic              fifo_empty;
logic [N_INPT-1:0] req;
logic [W_PTP-1:0]  ptp;
logic              fifo_active;


reg_fifo #(
  .DATA_WIDTH ( W_FIFO                   ),
  .DEPTH      ( N_INPT                   )
) u_reg_fifo (
  .clk        ( i_pclk                   ),
  .rst        ( i_prst                   ),
  .wr         ( (|i_req)                 ),
  .din        ( {i_req,i_ptp[W_PTP-1:0]} ),
  .full       (                          ),
  .rd         ( fifo_rd                  ),
  .dval       ( fifo_dval                ),
  .dout       ( fifo_dout                ),
  .over       (                          ),
  .under      (                          ),
  .empty      ( fifo_empty               )
);

assign {req,ptp} = fifo_dout;

//------------------------------------------------------------------------------------------------//
// Serving RRARB
//------------------------------------------------------------------------------------------------//

logic [W_INPT-1:0] ram_wr_addr;
logic              ram_wr;

logic [N_INPT-1:0] mask;
logic [N_INPT-1:0] gnt;
logic [N_INPT-1:0] gnt_r;
logic [W_INPT-1:0] gnt_idx;
logic [1:0]        lsb_wr;
logic [1:0]        lsb_rd;
logic              fifo_empty_r;

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    mask         <= '0;
    fifo_rd      <= '0;
    fifo_active  <= '0;
    fifo_empty_r <= '1;
    gnt_r        <= '0;
  end
  else begin
    gnt_r <= gnt;
    if (!fifo_empty && !fifo_rd) begin
      if (!fifo_active) begin
        mask        <= '0;
        fifo_rd     <= '0;
        fifo_active <= '1;
      end
      else if ((req&gnt) == '0) begin
        mask        <= '0;
        fifo_rd     <= '1; // Pop fifo if all requests are served
        fifo_active <= '0;
      end
      else begin
        mask        <= (mask|gnt); // Mask out all served requests
        fifo_rd     <= '0;
        fifo_active <= '1;
      end
    end
    else begin
      mask        <= '0;
      fifo_rd     <= '0;
      fifo_active <= '0;
    end
  end
end

rrarb #(
  .WIDTH      ( N_INPT                        )
) u_rrarb (
  .clk        ( i_pclk                        ),
  .rst_n      ( '1                            ),
  .rst        ( i_prst                        ),
  .idle       ( '1                            ),
  .req        ( fifo_active ? (req^mask) : '0 ),
  .gnt        ( gnt                           )
);

integer i;
always_comb begin
  gnt_idx = '0;
  for (i=0;i<N_INPT;i=i+1) begin
    if (gnt_r[i]) begin
      gnt_idx = i;
    end
  end
end


//------------------------------------------------------------------------------------------------//
// Pointer to the timestamp
//------------------------------------------------------------------------------------------------//

logic [1:0] rd_ptr [N_INPT-1:0];
logic [1:0] wr_ptr [N_INPT-1:0];

genvar j;
generate
  for (j=0;j<N_INPT;j=j+1) begin
    always_ff @(posedge i_pclk) begin
      if (i_prst) begin
        rd_ptr[j] <= '0;
        wr_ptr[j] <= '0;
      end
      else begin
        if (ram_wr && (j==gnt_idx)) begin
          wr_ptr[j] <= wr_ptr[j] + 1;
        end
        if (i_ram_rd && (j==i_ptp_sel)) begin
          rd_ptr[j] <= rd_ptr[j] + 1;
        end
      end
    end
  end
endgenerate


assign lsb_wr = wr_ptr[gnt_idx];
assign lsb_rd = rd_ptr[i_ptp_sel];

//------------------------------------------------------------------------------------------------//
// Timestamp RAM
//------------------------------------------------------------------------------------------------//

logic [W_PTP-1:0] ram_rd_data;

dp_ram #(
  .DATA_WIDTH ( W_PTP    ),
  .RAM_DEPTH  ( N_INPT*4 ),
  .ADDR_WIDTH ( W_INPT+2 ),
  .RAM_TYPE   ( "SIMPLE" ),
  .MEM_STYLE  ( "BLOCK"  )
) u_dp_ram (
  .clk_a      ( i_pclk               ),
  .en_a       ( '1                   ),
  .we_a       ( ram_wr               ),
  .din_a      ( ptp                  ),
  .addr_a     ( {ram_wr_addr,lsb_wr} ),
  .dout_a     (                      ),
  .clk_b      ( i_pclk               ),
  .en_b       ( '1                   ),
  .we_b       ( '0                   ),
  .din_b      ( '0                   ),
  .addr_b     ( {i_ptp_sel,lsb_rd}   ),
  .dout_b     ( ram_rd_data          )
);

assign o_ptp       = {16'h0,ram_rd_data};
assign ram_wr_addr = gnt_idx;
assign ram_wr      = |gnt_r;

endmodule
