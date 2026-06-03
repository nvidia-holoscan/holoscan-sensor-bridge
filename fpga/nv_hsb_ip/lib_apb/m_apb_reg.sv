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

module m_apb_reg
  import apb_pkg::*;
#(
  parameter N_REG   = 1, // Number of registers in this sequence
  parameter N_DELAY = 0  // Number of clock cycle delay in between sequence, 1-255. 0: i_init triggered delay only
)(
  // Clock and Reset
  input               i_aclk,
  input               i_arst,
  // Register Sequence
  input  [W_DATA-1:0] i_data[N_REG],
  input  [W_ADDR-1:0] i_addr[N_REG],
  input  [ N_REG-1:0] i_wren,
  input               i_init,
  input               i_init_en,
  output              o_done,
  output              o_ready,
  output [W_DATA-1:0] o_data[N_REG],
  output [W_ADDR-1:0] o_addr[N_REG],
  output              o_err,
  //
  output apb_m2s      o_apb_m2s,
  input  apb_s2m      i_apb_s2m
);

logic [7:0] init_delay;
logic       init_q;
logic       init_en;

typedef enum logic [1:0] {IDLE, SEQ_START, SEQ_WAIT} states;
states state;

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    init_delay <= N_DELAY;
    init_q     <= 1'b0;
    init_en    <= 1'b0;
  end
  else begin
    init_q     <= i_init;
    init_delay <= (|init_delay) ? init_delay - 1'b1 : init_delay;
    init_en    <= (!i_init_en & (init_delay == 1)) | (i_init & !init_q);
  end
end

localparam W_CNT = $clog2(N_REG+1);

logic [W_DATA-1:0] reg_rdbk [N_REG];
logic [W_ADDR-1:0] reg_addr [N_REG];
logic [ N_REG-1:0] reg_err;
logic [ W_CNT-1:0] reg_seq;
logic [       7:0] cntr;
logic              done;
logic              ready;

logic              psel;
logic              penable;
logic [W_ADDR-1:0] paddr;
logic [W_DATA-1:0] pwdata;
logic              pwrite;
logic              pserr;

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    state    <= IDLE;
    reg_addr <= '{default:0};
    reg_rdbk <= '{default:0};
    reg_seq  <= 0;
    cntr     <= 0;
    done     <= 0;
    ready    <= 0;
    psel     <= 0;
    penable  <= 0;
    paddr    <= 0;
    pwdata   <= 0;
    pwrite   <= 0;
    pserr    <= 0;
  end
  else begin
    pserr    <= i_apb_s2m.pserr & i_apb_s2m.pready;
    case (state)
      IDLE: begin
        reg_addr  <= '{default:0};
        reg_rdbk  <= '{default:0};
        reg_seq   <= 0;
        cntr      <= 0;
        done      <= 0;
        ready     <= 1;
        psel      <= 0;
        penable   <= 0;
        pwrite    <= 0;
        paddr     <= 0;
        pwdata    <= 0;
        if (init_en) begin
          state   <= SEQ_START;
          done    <= 0;
          ready   <= 0;
          psel    <= 1;
          penable <= 0;
          pwrite  <= i_wren[reg_seq];
          paddr   <= i_addr[reg_seq];
          pwdata  <= i_data[reg_seq];
        end
      end
      SEQ_START: begin
        penable   <= 1;
        if (penable & i_apb_s2m.pready) begin
          reg_addr[reg_seq] <= paddr;
          reg_rdbk[reg_seq] <= i_apb_s2m.prdata;
          reg_seq           <= reg_seq + 1'b1;
          if (reg_seq == N_REG-1) begin
            state   <= IDLE;
            reg_seq <= 0;
            done    <= 1;
          end
          else begin
            state   <= SEQ_WAIT;
          end
          psel    <= 0;
          penable <= 0;
          pwrite  <= 0;
          paddr   <= 0;
          pwdata  <= 0;
          cntr    <= N_DELAY; // wait 8 cycles before next sequence
        end
      end
      SEQ_WAIT: begin
        if (|cntr) begin
          cntr    <= cntr - 1'b1;
        end
        if (~|cntr) begin
          state   <= SEQ_START;
          psel    <= 1'b1;
          pwrite  <= i_wren[reg_seq];
          paddr   <= i_addr[reg_seq];
          pwdata  <= i_data[reg_seq];
        end
      end
      default: state <= IDLE;
    endcase
  end
end

assign o_done            = done;
assign o_data            = reg_rdbk;
assign o_addr            = reg_addr;
assign o_ready           = ready;
assign o_err             = pserr;
assign o_apb_m2s.psel    = psel;
assign o_apb_m2s.penable = penable;
assign o_apb_m2s.paddr   = paddr;
assign o_apb_m2s.pwdata  = pwdata;
assign o_apb_m2s.pwrite  = pwrite;

endmodule
