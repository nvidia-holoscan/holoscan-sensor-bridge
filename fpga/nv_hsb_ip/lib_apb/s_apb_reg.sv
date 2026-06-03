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

module s_apb_reg
  import apb_pkg::*;
#(
  parameter                   N_CTRL  = 8,
  parameter                   N_STAT  = 4,
  parameter                   W_OFST  = 8,
  parameter                   R_TOTL  = (2**W_OFST)/4, // Total Reg Range
  parameter                   R_CTRL  = R_TOTL/2,
  parameter                   SYNC_CLK = 0,
  parameter                   SAME_CLK = 0,
  parameter [(N_CTRL*32)-1:0] RST_VAL = '0
)(
  // APB Interface
  input               i_aclk, // Slow Clock
  input               i_arst,
  input  apb_m2s      i_apb_m2s,
  output apb_s2m      o_apb_s2m,
  // User Control Signals
  input               i_pclk, // Fast Clock
  input               i_prst,
  output [W_DATA-1:0] o_ctrl [N_CTRL],
  input  [W_DATA-1:0] i_stat [N_STAT]
);

//------------------------------------------------------------------------------------------------//
// Register Read and Write
//------------------------------------------------------------------------------------------------//

localparam N_REG = N_CTRL + N_STAT;

logic [W_DATA-1:0] reg_ctrl [N_CTRL];
logic [W_DATA-1:0] reg_ctrl_q [N_CTRL];
logic [W_DATA-1:0] reg_stat [N_STAT];
logic [W_OFST-3:0] reg_addr;
logic [W_OFST-3:0] reg_addr_new;
logic [W_DATA-1:0] w_rdata;
logic              reg_wren;
logic [W_DATA-1:0] reg_wdata;
logic              reg_rden;
logic [W_DATA-1:0] reg_rdata;
logic              reg_ready;
logic              reg_err;
logic              reg_addrerr;
logic              reg_wr2ro;

assign reg_addr_new = (reg_addr >= R_CTRL) ? (reg_addr - R_CTRL) : reg_addr;

// Setup phase
always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    reg_addr  <= 0;
    reg_wren  <= 0;
    reg_wdata <= 0;
    reg_rden  <= 0;
  end
  else begin
    reg_wren    <= i_apb_m2s.psel & i_apb_m2s.penable &  i_apb_m2s.pwrite;
    reg_rden    <= i_apb_m2s.psel & i_apb_m2s.penable & !i_apb_m2s.pwrite;
    if (i_apb_m2s.psel) begin
      reg_addr  <= i_apb_m2s.paddr[W_OFST-1:2];
      reg_wdata <= i_apb_m2s.pwdata;
    end
  end
end

logic reg_wdval;
// Write Read Phase
always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    reg_ready  <= '0;
    reg_wdval  <= '0;
    for (int j = 0; j < N_CTRL; j++) begin
      reg_ctrl[j] <= RST_VAL[j*32+:32];
    end
    //reg_rdata  <= '0;
  end
  else begin
    reg_ready             <= reg_wren | reg_rden;
    reg_wdval             <= reg_wren & !reg_addrerr;
    if (reg_wren && !reg_addrerr) begin
      reg_ctrl[reg_addr]  <=  reg_wdata;
    end
  end
end


(* ram_style = "distributed" *) logic [31:0] lut_mem [N_CTRL]/*synthesis syn_ramstyle = "distributed"*/;

always @ (posedge i_aclk) begin
  if (i_arst) begin
    for (int j = 0; j < N_CTRL; j++) begin
      lut_mem[j] <= RST_VAL[j*32+:32];
    end
    w_rdata <= 'd0;
  end
  else begin
    if ((reg_wren && !reg_addrerr)) begin
      lut_mem[reg_addr] <= reg_wdata;
    end
    if (reg_rden) begin
      w_rdata     <= lut_mem[reg_addr];
    end
  end
end

assign reg_rdata = (!reg_rden         ) ? '0                     :
                   (reg_addrerr       ) ? 32'hBADADD12           :
                   (reg_addr >= R_CTRL) ? reg_stat[reg_addr_new] :
                                          w_rdata                ;

//------------------------------------------------------------------------------------------------//
// Error Condition
//------------------------------------------------------------------------------------------------//

logic reg_addr_ro;
assign reg_addr_ro = (reg_addr inside{[R_CTRL:N_STAT+R_CTRL-1]});

logic reg_addr_inside;
assign reg_addr_inside = (reg_addr inside{[0:N_CTRL-1]}) || reg_addr_ro;

assign reg_addrerr = (reg_wren || reg_rden) && !reg_addr_inside;
assign reg_wr2ro   = reg_wren &&  reg_addr_ro;

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    reg_err <= 0;
  end
  else begin
    reg_err <= reg_addrerr;
  end
end

assign o_apb_s2m.prdata  = reg_rdata;
assign o_apb_s2m.pready  = reg_ready & i_apb_m2s.penable;
assign o_apb_s2m.pserr   = reg_err   & i_apb_m2s.penable;

//------------------------------------------------------------------------------------------------//
// CDC
//------------------------------------------------------------------------------------------------//

genvar i;
generate
  for (i=0; i<N_CTRL; i++) begin
    if (SAME_CLK) begin
      assign o_ctrl[i] = reg_ctrl[i];
    end else if (SYNC_CLK) begin
      always_ff @(posedge i_pclk) begin
        if (i_prst) begin
          reg_ctrl_q[i] <= RST_VAL[i*32+:32];
        end else begin
          reg_ctrl_q[i] <= reg_ctrl[i];
        end
      end

      assign o_ctrl[i] = reg_ctrl_q[i];

    end else begin
      reg_cdc # (
        .REG_RST_VALUE ( RST_VAL[i*32+:32] )
      ) u_ctrl_cdc (
        .i_a_clk ( i_aclk       ),
        .i_a_rst ( i_arst       ),
        .i_a_val ( reg_wdval    ),
        .i_a_reg ( reg_ctrl [i] ),
        .i_b_clk ( i_pclk       ),
        .i_b_rst ( i_prst       ),
        .o_b_val (              ),
        .o_b_reg ( o_ctrl   [i] )
      );
    end
  end
endgenerate

generate
  for (i=0; i<N_STAT; i++) begin

    if (SAME_CLK) begin
      assign reg_stat[i] = i_stat[i];
    end else if (SYNC_CLK) begin
      always_ff @(posedge i_pclk) begin
        if (i_prst) begin
          reg_stat[i] <= '0;
        end else begin
          reg_stat[i] <= i_stat[i];
        end
      end
    end else begin
      reg_cdc u_ctrl_cdc (
        // Processing Clock Domain, Slow
        .i_a_clk ( i_pclk       ),
        .i_a_rst ( i_prst       ),
        .i_a_val ( 1'b1         ),
        .i_a_reg ( i_stat   [i] ),
        // APB Clock Domain, Fast
        .i_b_clk ( i_aclk       ),
        .i_b_rst ( i_arst       ),
        .o_b_val (              ),
        .o_b_reg ( reg_stat [i] )
      );
    end
  end
endgenerate

endmodule
