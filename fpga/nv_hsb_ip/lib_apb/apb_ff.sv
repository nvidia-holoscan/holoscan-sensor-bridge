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

module apb_ff
  import apb_pkg::*;
(
  // APB Interface
  input               i_aclk,
  input               i_arst,
  input  apb_m2s      i_apb_m2s,
  output apb_m2s      o_apb_m2s,
  input  apb_s2m      i_apb_s2m,
  output apb_s2m      o_apb_s2m
);

apb_s2m apb_s2m_q;
apb_m2s apb_m2s_q;


always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    apb_s2m_q <= '{default:0};
    apb_m2s_q <= '{default:0};
  end
  else begin
    apb_s2m_q <= i_apb_s2m;
    apb_m2s_q <= i_apb_m2s;
  end
end


assign o_apb_m2s.psel    = apb_m2s_q.psel   ;
assign o_apb_m2s.penable = apb_m2s_q.penable;
assign o_apb_m2s.paddr   = apb_m2s_q.paddr  ;
assign o_apb_m2s.pwdata  = apb_m2s_q.pwdata ;
assign o_apb_m2s.pwrite  = apb_m2s_q.pwrite ;

assign o_apb_s2m.pready = !i_apb_m2s.penable ? 0 : apb_s2m_q.pready; // Delay on assert
assign o_apb_s2m.pserr  = apb_s2m_q.pserr;
assign o_apb_s2m.prdata = apb_s2m_q.prdata;

endmodule
