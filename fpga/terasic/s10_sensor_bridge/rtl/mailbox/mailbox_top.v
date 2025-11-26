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

module mailbox_top (
  input logic                 apb_clk,
  input logic                 apb_rst,
  input logic                 i_mailbox_apb_psel,   
  input logic                 i_mailbox_apb_penable,
  input logic   [31:0]        i_mailbox_apb_paddr,  
  input logic   [31:0]        i_mailbox_apb_pwdata, 
  input logic                 i_mailbox_apb_pwrite, 
  output logic                o_mailbox_apb_pready,           
  output logic  [31:0]        o_mailbox_apb_prdata,           
  output logic                o_mailbox_apb_pserr
);

logic   [31:0]            mailbox_avmm_addr;
logic                     mailbox_avmm_write;
logic   [31:0]            mailbox_avmm_wrdata;
logic                     mailbox_avmm_read;
logic   [31:0]            mailbox_avmm_rddata;
logic                     mailbox_avmm_rddataval;
logic                     mailbox_avmm_waitreq;

avmm_to_apb #(
  .AVMM_ADDR_WIDTH                  (32),
  .AVMM_DATA_WIDTH                  (32),
  .USE_AVMM_READDATAVALID           (1),
  .AVMM_RD_LATENCY                  (0)         
) u_mailbox_avmm_to_apb (
  .clk                              (apb_clk),
  .rst                              (apb_rst),
  .psel                             (i_mailbox_apb_psel),
  .penable                          (i_mailbox_apb_penable),
  .paddr                            (i_mailbox_apb_paddr),
  .pwrite                           (i_mailbox_apb_pwrite),
  .pwdata                           (i_mailbox_apb_pwdata),
  .pready                           (o_mailbox_apb_pready),
  .prdata                           (o_mailbox_apb_prdata),
  .pserr                            (o_mailbox_apb_pserr),
  .avmm_address                     (mailbox_avmm_addr),
  .avmm_write                       (mailbox_avmm_write),
  .avmm_writedata                   (mailbox_avmm_wrdata),
  .avmm_read                        (mailbox_avmm_read),
  .avmm_readdata                    (mailbox_avmm_rddata),
  .avmm_readdatavalid               (mailbox_avmm_rddataval),
  .avmm_waitrequest                 (mailbox_avmm_waitreq)
);


`ifdef QUARTUS_BUILD    
  mailbox_client u_mailbox_client(
   .in_clk_clk                             (apb_clk                     ),
   .in_reset_reset                         (apb_rst                     ),
   .avmm_address                           (mailbox_avmm_addr[5:2]      ),
   .avmm_write                             (mailbox_avmm_write          ),
   .avmm_writedata                         (mailbox_avmm_wrdata         ),
   .avmm_read                              (mailbox_avmm_read           ),
   .avmm_readdata                          (mailbox_avmm_rddata         ),
   .avmm_readdatavalid                     (mailbox_avmm_rddataval      ),
   .irq_irq                                (                            ) 
  );
  
  assign mailbox_avmm_waitreq = 1'b0; //(mailbox_avmm_write || mailbox_avmm_read) ? 1'b0 : 1'b1;
  
`else
  assign         mailbox_avmm_rddata    = 32'h1bad_add1;
  assign         mailbox_avmm_rddataval = 1'b1;
  assign         mailbox_avmm_waitreq   = 1'b0;
`endif

endmodule
