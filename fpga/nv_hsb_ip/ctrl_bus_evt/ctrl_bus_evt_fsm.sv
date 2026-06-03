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

module ctrl_bus_evt_fsm
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter         W_EVENT     = 32,
  localparam        W_EVENT_IDX = 5
)(
  input                    i_aclk,
  input                    i_arst,
  input   apb_m2s          i_apb_m2s,
  output  apb_s2m          o_apb_s2m,

  output  apb_m2s          o_apb_m2s,
  input   apb_s2m          i_apb_s2m,

  input  [W_EVENT-1:0]     i_evt_vec,
  input  [23:0]            i_timeout,
  output                   o_timeout,
  output [W_EVENT_IDX-1:0] o_evt_vec,
  output                   o_error
);

typedef enum logic [3:0] {
  IDLE,
  SET_ADDR,
  LD_PTR,
  JUMP,
  LD_CMD,
  LD_ADDR,
  LD_DATA,
  LD_MASK,
  LATCH_DATA,
  ADV_DATA,
  APB_WAIT,
  APB_COMPARE,
  APB_DONE
} evt_state_t;

evt_state_t        evt_state;

typedef enum logic [1:0] {
  OPCODE_NOP  = 2'b11,
  OPCODE_POLL = 2'b00,
  OPCODE_WRITE = 2'b01,
  OPCODE_READ = 2'b10
} opcode_t;


//------------------------------------------------------------------------------------------------//
// Register Map
//------------------------------------------------------------------------------------------------//

logic [31:0] ram_addr;
logic [31:0] ram_data;
logic        ram_data_valid;
logic        ram_wr_en;
logic [31:0] ram_wr_data;
logic        ram_rd_en;

s_apb_ram #(
  .R_CTRL           ( 1024              ),
  .R_WIDTH          ( 32                ),
  .R_TOTL           ( 2**12             )
) u_config_ram  (
  // APB Interface
  .i_aclk           ( i_aclk            ),
  .i_arst           ( i_arst            ),
  .i_apb_m2s        ( i_apb_m2s         ),
  .o_apb_s2m        ( o_apb_s2m         ),
  // User Control Signals
  .i_pclk           ( i_aclk            ),
  .i_prst           ( i_arst            ),
  .i_addr           ( ram_addr[9:0]     ),
  .o_rd_data        ( ram_data          ),
  .o_rd_data_valid  ( ram_data_valid    ),
  .i_wr_data        ( ram_wr_data       ),
  .i_wr_en          ( ram_wr_en         ),
  .i_rd_en          ( ram_rd_en         )
);

//------------------------------------------------------------------------------------------------//
// ARB - Arbitration of Events
//------------------------------------------------------------------------------------------------//

logic        fifo_rd;
logic        fifo_dval;
logic [31:0] fifo_dout;
logic        fifo_empty;


reg_fifo #(
  .DATA_WIDTH ( W_EVENT              ),
  .DEPTH      ( 8                    )
) u_reg_fifo (
  .clk        ( i_aclk               ),
  .rst        ( i_arst               ),
  .wr         ( (|i_evt_vec)         ),
  .din        ( i_evt_vec            ),
  .full       (                      ),
  .rd         ( fifo_rd              ),
  .dval       ( fifo_dval            ),
  .dout       ( fifo_dout            ),
  .over       (                      ),
  .under      (                      ),
  .empty      ( fifo_empty           )
);

logic [W_EVENT-1:0]     mask;
logic                   fifo_empty_r;
logic                   fifo_active;
logic [W_EVENT-1:0]     req;
logic [W_EVENT-1:0]     evt_arb_gnt;
logic [W_EVENT_IDX-1:0] evt_arb_gnt_idx;
logic [W_EVENT_IDX-1:0] r_evt_idx;


always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    mask         <= '0;
    fifo_rd      <= '0;
    fifo_active  <= '0;
    fifo_empty_r <= '1;
  end
  else begin
    if (!fifo_empty && !fifo_rd) begin
      if (!fifo_active) begin
        mask        <= '0;
        fifo_rd     <= '0;
        fifo_active <= '1;
      end
      else if (((req&evt_arb_gnt) == '0) && (evt_state == IDLE)) begin
        mask        <= '0;
        fifo_rd     <= '1; // Pop fifo if all requests are served
        fifo_active <= '0;
      end
      else begin
        mask        <= (mask|evt_arb_gnt); // Mask out all served requests
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

priority_arb #(
  .WIDTH      ( W_EVENT                       )
) u_priority_arb (
  .clk        ( i_aclk                        ),
  .rst_n      ( '1                            ),
  .rst        ( i_arst                        ),
  .idle       ( (evt_state == IDLE)           ),
  .req        ( fifo_active ? (req^mask) : '0 ),
  .gnt        ( evt_arb_gnt                   )
);

always_comb begin
  evt_arb_gnt_idx = '0;
  for (int j=0;j<W_EVENT;j=j+1) begin
    if (evt_arb_gnt[j]) begin
      evt_arb_gnt_idx = j;
    end
  end
end

assign req = fifo_dout;


//------------------------------------------------------------------------------------------------//
// Timer
//------------------------------------------------------------------------------------------------//


logic [23:0] timer_cnt;
logic        fsm_timeout;
logic [4:0]  evt_vec_timeout;
logic        fsm_error;

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    timer_cnt       <= '0;
    fsm_timeout     <= 1'b0;
    evt_vec_timeout <= '0;
  end
  else begin
    if (timer_cnt == i_timeout) begin
      timer_cnt   <= '0;
      fsm_timeout <= (i_timeout != '0);
    end
    else if (evt_state == IDLE) begin
      timer_cnt   <= '0;
      fsm_timeout <= 1'b0;
    end
    else begin
      timer_cnt   <= timer_cnt + 1;
      fsm_timeout <= 1'b0;
    end
    if (fsm_timeout) begin
      evt_vec_timeout <= evt_arb_gnt_idx;
    end
  end
end


//------------------------------------------------------------------------------------------------//
// FSM
//------------------------------------------------------------------------------------------------//

logic [31:0] cmd;
logic [3:0]  seq;
logic [3:0]  cnt;
logic        psel;
logic        penable;
logic        pwrite;
logic [31:0] paddr;
logic [31:0] pwdata;
logic [31:0] rd_data;
logic [31:0] val_mask;
logic [4:0]  seq_addr;

opcode_t opcode;

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    evt_state <= IDLE;
    psel      <= 1'b0;
    penable   <= 1'b0;
    pwrite    <= 1'b0;
    paddr     <= '0;
    pwdata    <= '0;
    cnt       <= '1;
    val_mask  <= '0;
    ram_addr  <= '0;
    r_evt_idx <= '0;
    opcode    <= OPCODE_NOP;
    fsm_error <= 1'b0;
    cmd       <= '0;
    seq       <= '0;
    rd_data   <= '0;
  end
  else begin
    if (fsm_timeout || fsm_error) begin
      evt_state <= IDLE;
      fsm_error <= 1'b0;
      psel      <= 1'b0;
      penable   <= 1'b0;
      pwrite    <= 1'b0;
      paddr     <= '0;
    end
    else begin
      if (ram_data_valid || (evt_state == IDLE)) begin
        case(evt_state)
          IDLE: begin
            psel      <= 1'b0;
            penable   <= 1'b0;
            pwrite    <= 1'b0;
            paddr     <= '0;
            if (|evt_arb_gnt) begin
              evt_state <= SET_ADDR;
              r_evt_idx <= evt_arb_gnt_idx;
              ram_addr  <= evt_arb_gnt_idx;
            end
          end
          SET_ADDR: begin
            psel      <= 1'b0;
            penable   <= 1'b0;
            pwrite    <= 1'b0;
            seq       <= '0;
            cnt       <= '1;
            evt_state <= LD_PTR;
          end
          LD_PTR: begin
            ram_addr  <= {2'h0,ram_data[31:2]};
            evt_state <= JUMP;
          end
          JUMP: begin
            evt_state <= LD_CMD;
            ram_addr  <= ram_addr + 1;
          end
          LD_CMD: begin
            cmd       <= ram_data[31:0];
            evt_state <= LD_ADDR;
            ram_addr  <= ram_addr + 1;
            seq       <= '0;
          end
          LD_ADDR: begin
            paddr     <= ram_data[31:0];
            evt_state <= (ram_data[31:0] == '1) ? APB_DONE : LD_DATA;
            ram_addr  <= (cmd[seq_addr+1]) ? ram_addr : ram_addr + 1;
            opcode    <= opcode_t'({cmd[seq_addr+1],cmd[seq_addr]});
          end
          LD_DATA: begin
            pwdata    <= ram_data[31:0];
            evt_state <= LD_MASK;
            pwrite    <= (opcode == OPCODE_WRITE);
            ram_addr  <= (opcode == OPCODE_POLL) ? ram_addr + 1 : ram_addr;
            psel      <= 1'b1;
          end
          LD_MASK: begin
            val_mask  <= ram_data[31:0];
            evt_state <= APB_WAIT;
            penable   <= 1'b1;
          end
          APB_WAIT: begin
            if (i_apb_s2m.pready) begin
              rd_data   <= i_apb_s2m.prdata;
              fsm_error <= i_apb_s2m.pserr;
              psel      <= 1'b0;
              penable   <= 1'b0;
              pwrite    <= 1'b0;
              evt_state <= (opcode == OPCODE_READ) ? LATCH_DATA : APB_COMPARE;
            end
            else begin
              penable <= 1'b1;
            end
          end
          LATCH_DATA: begin
            evt_state <= ADV_DATA;
            ram_addr  <= ram_addr + 1;
          end
          ADV_DATA: begin
            evt_state <= (seq == '1) ? LD_CMD : LD_ADDR;
            ram_addr  <= ram_addr + 1;
            seq       <= seq + 1;
          end
          APB_COMPARE: begin
            if ((opcode == OPCODE_WRITE)) begin
              evt_state <= (seq == '1) ? LD_CMD : LD_ADDR;
              ram_addr  <= ram_addr + 1;
              seq       <= seq + 1;
            end
            else begin
              if (cnt == '1) begin
                if ((rd_data&val_mask) == pwdata) begin
                  evt_state <= (seq == '1) ? LD_CMD : LD_ADDR;
                  ram_addr  <= ram_addr + 1;
                  seq       <= seq + 1;
                end
                else begin
                  evt_state <= APB_WAIT;   // Repeat Instruction
                  psel      <= 1'b1;
                end
                cnt       <= '0;
              end
              else begin
                cnt       <= cnt + 1;
              end
            end
          end
          APB_DONE: begin
            evt_state <= IDLE;
          end
        endcase
      end
    end
  end
end

assign seq_addr = {seq,1'b0};
assign o_apb_m2s.psel    = psel;
assign o_apb_m2s.penable = penable;
assign o_apb_m2s.paddr   = paddr;
assign o_apb_m2s.pwdata  = pwdata;
assign o_apb_m2s.pwrite  = pwrite;

assign o_timeout = fsm_timeout;
assign o_evt_vec = evt_vec_timeout;
assign ram_wr_en = (evt_state == LATCH_DATA);
assign ram_rd_en = (evt_state != IDLE);
assign ram_wr_data = rd_data;
assign o_error = fsm_error;

endmodule
