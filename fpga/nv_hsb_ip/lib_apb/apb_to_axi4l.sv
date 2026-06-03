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

module apb_to_axi4l #(
  parameter AWIDTH = 32,
  parameter DWIDTH = 32
) (
  // Clock and Reset
  input  logic                      i_aclk,
  input  logic                      i_arst,

  // APB Interface
  input  logic                      i_apb_psel,
  input  logic                      i_apb_penable,
  input  logic  [AWIDTH-1:0]        i_apb_paddr,
  input  logic  [DWIDTH-1:0]        i_apb_pwdata,
  input  logic                      i_apb_pwrite,
  output logic                      o_apb_pready,
  output logic  [DWIDTH-1:0]        o_apb_prdata,
  output logic                      o_apb_pserr,

  //AXI4-Lite Interface
  output logic [AWIDTH-1:0]         o_axi4l_awaddr,
  output logic                      o_axi4l_awvalid,
  input  logic                      i_axi4l_awready,
  output logic [DWIDTH-1:0]         o_axi4l_wdata,
  output logic [DWIDTH/8-1:0]       o_axi4l_wstrb,
  output logic                      o_axi4l_wvalid,
  input  logic                      i_axi4l_wready,
  input  logic [1:0]                i_axi4l_bresp,
  input  logic                      i_axi4l_bvalid,
  output logic                      o_axi4l_bready,
  output logic [AWIDTH-1:0]         o_axi4l_araddr,
  output logic                      o_axi4l_arvalid,
  input  logic                      i_axi4l_arready,
  input  logic [DWIDTH-1:0]         i_axi4l_rdata,
  input  logic [1:0]                i_axi4l_rresp,
  input  logic                      i_axi4l_rvalid,
  output logic                      o_axi4l_rready
);

typedef enum logic [2:0] {
  APB_IDLE,
  APB_W,
  APB_W_RESP,
  APB_R,
  APB_R_DATA,
  APB_DONE
} apb_state_e;

apb_state_e apb_state_q;

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    o_axi4l_awaddr  <= '0;
    o_axi4l_awvalid <= '0;
    o_axi4l_wdata   <= '0;
    o_axi4l_wstrb   <= '0;
    o_axi4l_wvalid  <= '0;
    o_axi4l_araddr  <= '0;
    o_axi4l_arvalid <= '0;
    o_apb_pready    <= '0;
    o_apb_prdata    <= '0;
    o_apb_pserr     <= '0;
    apb_state_q     <= APB_IDLE;
  end
  else begin
    // default deassert until operation completes
    o_apb_pready <= 1'b0;

    unique case (apb_state_q)
      APB_IDLE: begin
        if (i_apb_psel && i_apb_penable) begin
          if (i_apb_pwrite) begin
            // Launch AXI4-Lite write address + data
            o_axi4l_awaddr  <= i_apb_paddr;
            o_axi4l_awvalid <= 1'b1;
            o_axi4l_wdata   <= i_apb_pwdata;
            o_axi4l_wstrb   <= '1;
            o_axi4l_wvalid  <= 1'b1;
            apb_state_q     <= APB_W;
          end
          else begin
            // Launch AXI4-Lite read address
            o_axi4l_araddr  <= i_apb_paddr;
            o_axi4l_arvalid <= 1'b1;
            apb_state_q     <= APB_R;
          end
        end
      end

      APB_W: begin
        if (o_axi4l_awvalid && i_axi4l_awready) o_axi4l_awvalid                                     <= 1'b0;
        if (o_axi4l_wvalid  && i_axi4l_wready)  o_axi4l_wvalid                                      <= 1'b0;
        if ((!o_axi4l_awvalid | i_axi4l_awready) && (!o_axi4l_wvalid | i_axi4l_wready)) apb_state_q <= APB_W_RESP;
      end

      APB_W_RESP: begin
        if (i_axi4l_bvalid) begin
          o_apb_pready <= 1'b1; // complete APB write
          o_apb_pserr  <= (i_axi4l_bresp != 2'b00);
          apb_state_q  <= APB_DONE;
        end
      end

      APB_R: begin
        if (o_axi4l_arvalid && i_axi4l_arready) begin
          o_axi4l_arvalid <= 1'b0;
          apb_state_q     <= APB_R_DATA;
        end
      end

      APB_R_DATA: begin
        if (i_axi4l_rvalid) begin
          o_apb_prdata <= i_axi4l_rdata;
          o_apb_pserr  <= (i_axi4l_rresp != 2'b00);
          o_apb_pready <= 1'b1; // complete APB read
          apb_state_q  <= APB_DONE;
        end
      end
      APB_DONE: begin
        if (!i_apb_psel && !i_apb_penable) begin
          apb_state_q <= APB_IDLE;
        end
      end
      default: apb_state_q <= APB_IDLE;
    endcase
  end
end

assign o_axi4l_bready = 1'b1;
assign o_axi4l_rready = 1'b1;


endmodule
