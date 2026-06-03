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

module apb_adapter
  import apb_pkg::*;
#(
  parameter NUM_INST = 2,
  parameter W_OFSET  = 9
)(
  input                                               i_aclk,
  input                                               i_arst_n,
  //Peripheral Data Bus address interface
  output                                              o_wren,
  output  [31:0]                                      o_addr,
  output  [31:0]                                      o_dout,
  output                                              o_sel,
  input                                               i_rdval,
  input   [31:0]                                      i_din,
  input                                               i_addr_chk,
  input                                               i_err,
  // APB interface
  input  apb_m2s                                      i_apb_m2s,
  output apb_s2m                                      o_apb_s2m
);

reg        reg_ready;
reg        reg_error;
reg        reg_rdChk;
reg [31:0] reg_rdata;

wire wren;
wire rden;

wire reg_invaddrChk;
assign reg_invaddrChk = i_apb_m2s.psel && i_apb_m2s.penable && |(i_apb_m2s.paddr[W_ADDR-1:W_OFSET+$clog2(NUM_INST)]);

assign wren = i_apb_m2s.psel && i_apb_m2s.penable &&  i_apb_m2s.pwrite && !reg_invaddrChk;
assign rden = i_apb_m2s.psel && i_apb_m2s.penable && !i_apb_m2s.pwrite && !reg_invaddrChk;

assign o_wren    = wren             ;
assign o_addr    = i_apb_m2s.paddr  ;
assign o_dout    = i_apb_m2s.pwdata ;
assign o_sel     = i_apb_m2s.psel && i_apb_m2s.penable ;

assign o_apb_s2m.pready   = reg_ready;
assign o_apb_s2m.pserr    = i_apb_m2s.psel && i_apb_m2s.penable && reg_ready ? reg_error : 1'b0;
assign o_apb_s2m.prdata   = reg_rdata;


always @ (posedge i_aclk or negedge i_arst_n) begin
  if (!i_arst_n) begin
    reg_ready <= 1'b0 ;
    reg_error <= 1'b0 ;
    reg_rdChk <= 1'b0 ;
    reg_rdata <= 32'h0;
  end
  else begin
    reg_rdata   <= i_din;
    reg_error   <= i_err;
    if (reg_ready) begin
      reg_rdChk <= 1'b0;
    end
    else if (rden && i_addr_chk && !i_err) begin
      reg_rdChk <= 1'b1;
    end
    if (reg_ready) begin
      reg_ready <= 1'b0;
    end
    else if (reg_invaddrChk) begin
      reg_ready <= 1'b1;
      reg_error <= 1'b1;
    end
    else if (wren && i_addr_chk) begin
      reg_ready <= 1'b1;
    end
    else if (rden) begin
      if (i_addr_chk && i_err) begin
        reg_ready <= 1'b1;
      end
      else if (reg_rdChk && i_rdval) begin
        reg_ready <= 1'b1;
      end
    end
  end
end

endmodule
