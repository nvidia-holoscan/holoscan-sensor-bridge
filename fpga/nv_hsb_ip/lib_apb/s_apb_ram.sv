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

module s_apb_ram
  import apb_pkg::*;
#(
  parameter  R_CTRL    = 512,
  parameter  R_TOTL    = 512,
  parameter  R_WIDTH   = 32,
  localparam W_COFST   = $clog2(R_CTRL),
  localparam W_OFST    = $clog2(R_TOTL)
)(
  // APB Interface
  input                     i_aclk, // Slow Clock
  input                     i_arst,
  input  apb_m2s            i_apb_m2s,
  output apb_s2m            o_apb_s2m,
  // User Control Signals
  input                     i_pclk, // Fast Clock
  input                     i_prst,
  input  [W_COFST-1:0]      i_addr,
  output [R_WIDTH-1:0]      o_rd_data,
  output                    o_rd_data_valid,
  input  [R_WIDTH-1:0]      i_wr_data,
  input                     i_wr_en,
  input                     i_rd_en
);

//------------------------------------------------------------------------------------------------//
// Register Read and Write
//------------------------------------------------------------------------------------------------//

logic [W_OFST-1:0] reg_addr;
logic [W_OFST-1:0] reg_raddr;
logic              reg_wren;
logic [W_DATA-1:0] reg_wdata;
logic              reg_rden;
logic [W_DATA-1:0] reg_rdata;
logic [W_DATA-1:0] reg_rdata_q;
logic              reg_ready;
logic              reg_err;
logic              reg_addrerr;
logic              reg_rdy_prev_sync;
logic [W_DATA-1:0] reg_wrdata;
logic              reg_wr_en;
logic              reg_rd_en;

always_ff @(posedge i_aclk or posedge i_arst) begin
  if (i_arst) begin
    reg_ready <= 0;
    reg_addr  <= 0;
    reg_wren  <= 0;
    reg_wdata <= 0;
    reg_rden  <= 0;
  end
  else begin
    reg_ready     <= reg_wren | reg_rden;
    reg_wren      <= i_apb_m2s.psel & i_apb_m2s.penable &  i_apb_m2s.pwrite;
    reg_rden      <= i_apb_m2s.psel & i_apb_m2s.penable & !i_apb_m2s.pwrite;
    if (i_apb_m2s.psel) begin
      reg_addr  <= i_apb_m2s.paddr[W_OFST-1:2];
      reg_wdata <= i_apb_m2s.pwdata;
    end
  end
end


//------------------------------------------------------------------------------------------------//
// Error Condition
//------------------------------------------------------------------------------------------------//

assign reg_addrerr = (reg_wren || reg_rden) && !(reg_addr inside{[0:R_CTRL-1]});

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    reg_err <= 0;
  end
  else begin
    reg_err <= reg_addrerr;
  end
end

assign o_apb_s2m.prdata  = reg_rdata_q;
assign o_apb_s2m.pready  = reg_rdy_prev_sync & i_apb_m2s.penable;
assign o_apb_s2m.pserr   = reg_err           & i_apb_m2s.penable;

//------------------------------------------------------------------------------------------------//
// CDC
//------------------------------------------------------------------------------------------------//

logic reg_wren_sync;
logic reg_rden_sync;


data_sync #(
  .DATA_WIDTH                       (1                      ),
  .RESET_VALUE                      (1'b0                   ),
  .SYNC_DEPTH                       (2                      )
) u_wr_en_sync (
  .clk                              (i_pclk                 ),
  .rst_n                            (!i_prst                ),
  .sync_in                          (reg_wren               ),
  .sync_out                         (reg_wren_sync          )
);

data_sync #(
  .DATA_WIDTH                       (1                      ),
  .RESET_VALUE                      (1'b0                   ),
  .SYNC_DEPTH                       (2                      )
) u_rd_en_sync (
  .clk                              (i_pclk                 ),
  .rst_n                            (!i_prst                ),
  .sync_in                          (reg_rden               ),
  .sync_out                         (reg_rden_sync          )
);

logic reg_rdy_prev  = '1;
logic reg_wren_prev = '1;


always @ (posedge i_pclk) begin
  if (i_prst) begin
    reg_rdy_prev  <= '0;
    reg_rdata_q   <= '0;
    reg_wren_prev <= '0;
  end
  else begin
    reg_rdy_prev  <= (reg_wren_sync || reg_rden_sync);
    reg_rdata_q   <= reg_rden_sync ? ((reg_addrerr) ? 32'hBADADD12 : reg_rdata) : '0;
    reg_wren_prev <= i_wr_en;
  end
end

data_sync #(
  .DATA_WIDTH                       (1                      ),
  .RESET_VALUE                      (1'b0                   ),
  .SYNC_DEPTH                       (3                      )
) u_rdy_sync (
  .clk                              (i_aclk                 ),
  .rst_n                            (!i_arst                ),
  .sync_in                          (reg_rdy_prev           ),
  .sync_out                         (reg_rdy_prev_sync      )
);

//------------------------------------------------------------------------------------------------//
// RAM
//------------------------------------------------------------------------------------------------//

(* ram_style = "block" *) logic [R_WIDTH-1:0] blk_mem [R_CTRL] = '{default:'0}/*synthesis syn_ramstyle = "block_ram"*/;
logic [W_COFST-1:0] addr_prev = '0;

always @ (posedge i_pclk) begin
  if ((reg_wr_en)) begin
    blk_mem[reg_raddr[W_COFST-1:0]] <= reg_wrdata;
  end
  if (reg_rd_en) begin
    reg_rdata     <= blk_mem[reg_raddr[W_COFST-1:0]];
  end
end

always @ (posedge i_pclk) begin
  addr_prev <= (o_rd_data_valid) ? i_addr : addr_prev;
end

assign reg_wrdata      = (reg_rden_sync || reg_wren_sync) ? reg_wdata            : i_wr_data;
assign reg_raddr       = (reg_rden_sync || reg_wren_sync) ? reg_addr[0+:W_COFST] : (reg_rdy_prev) ? addr_prev : i_addr;
assign reg_wr_en       = (reg_rden_sync || reg_wren_sync) ? reg_wren_sync        : i_wr_en && !reg_rdy_prev;
assign reg_rd_en       = (reg_rden_sync || reg_wren_sync) ? reg_rden_sync        : i_rd_en;
assign o_rd_data       = reg_rdata;
assign o_rd_data_valid = !(reg_rdy_prev || reg_rden_sync || reg_wren_sync);

endmodule
