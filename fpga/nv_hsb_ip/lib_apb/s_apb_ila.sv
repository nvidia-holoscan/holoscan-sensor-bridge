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

module s_apb_ila
  import apb_pkg::*;
#(
  parameter   DEPTH      = 512,
  parameter   W_DATA     = 32,
  parameter   EN_RST_VAL = 0,
  localparam  W_ADDR     = $clog2(DEPTH),
  localparam  NUM_RAM    = (((W_DATA-1)/32)+1),
  localparam  W_RAM      = $clog2(NUM_RAM)
)(
  // APB Interface
  input                     i_aclk, // Slow Clock
  input                     i_arst,
  input  apb_m2s            i_apb_m2s,
  output apb_s2m            o_apb_s2m,
  // User Control Signals
  input                     i_pclk, // Fast Clock
  input                     i_prst,
  input                     i_trigger,
  input                     i_enable,
  input  [W_DATA-1:0]       i_wr_data,
  input                     i_wr_en,
  output [31:0]             o_ctrl_reg
);


//------------------------------------------------------------------------------------------------//
// APB Switch
//------------------------------------------------------------------------------------------------//

localparam W_OFSET = W_ADDR+W_RAM+2;
localparam W_SW    = 32-W_OFSET;

apb_m2s m_apb_m2s [1];
apb_s2m m_apb_s2m [1];

apb_m2s sw_apb_m2s [2];
apb_s2m sw_apb_s2m [2];

apb_switch #(
  .N_MPORT             ( 1                 ),
  .N_SPORT             ( 2                 ),
  .W_OFSET             ( W_OFSET           ),
  .W_SW                ( W_SW              ),
  .MERGE_COMPLETER_SIG ( 0                 )
) u_apb_ila_switch     (
  .i_apb_clk           ( i_aclk           ),
  .i_apb_reset         ( i_arst           ),
  .i_apb_m2s           ( m_apb_m2s [0:0]  ),
  .o_apb_s2m           ( m_apb_s2m [0:0]  ),
  .o_apb_m2s           ( sw_apb_m2s       ),
  .i_apb_s2m           ( sw_apb_s2m       ),
  .i_apb_timeout       ( 1'b0             )
);

assign m_apb_m2s[0] = i_apb_m2s;
assign o_apb_s2m    = m_apb_s2m[0];

//------------------------------------------------------------------------------------------------//
// Register Interface
//------------------------------------------------------------------------------------------------//

logic [31:0]            ctrl_reg  [2];
logic [31:0]            stat_reg  [2];

logic trigger_en;
logic ila_rst;
logic fsm_busy;
logic fsm_done;
logic fsm_busy_sync;

logic [W_ADDR-1:0] sample_addr;

s_apb_reg #(
  .N_CTRL     ( 2                    ),
  .N_STAT     ( 2                    )
) u_gen_purpose_apb_reg (
  .i_aclk     ( i_aclk               ), 
  .i_arst     ( i_arst               ),
  .i_apb_m2s  ( sw_apb_m2s[0]        ),
  .o_apb_s2m  ( sw_apb_s2m[0]        ),
  .i_pclk     ( i_pclk               ), 
  .i_prst     ( i_prst               ),
  .o_ctrl     ( ctrl_reg             ),
  .i_stat     ( stat_reg             )
);

assign trigger_en = ctrl_reg[0][0];
assign ila_rst    = ctrl_reg[0][1];
assign o_ctrl_reg = ctrl_reg[1];

assign stat_reg[0] = {'0,i_trigger,fsm_done,fsm_busy};
assign stat_reg[1] = {'0,sample_addr};

//------------------------------------------------------------------------------------------------//
// RAM Interface
//------------------------------------------------------------------------------------------------//

logic [W_DATA-1:0] wr_data;
logic wr_en;

apb_m2s sww_apb_m2s;

s_apb_ram_dyn #(
  .DEPTH            ( DEPTH                               ),
  .W_DATA           ( W_DATA                              )
) u_apb_ram_inst (
  .i_aclk           ( i_aclk                              ),
  .i_arst           ( i_arst                              ),
  .i_apb_m2s        ( sww_apb_m2s                         ),
  .o_apb_s2m        ( sw_apb_s2m[1]                       ),
  .i_pclk           ( i_pclk                              ),
  .i_prst           ( i_prst                              ),
  .i_addr           ( sample_addr                         ),
  .o_rd_data        (                                     ),
  .o_rd_data_valid  (                                     ),
  .i_wr_data        ( wr_data                             ),
  .i_wr_en          ( wr_en                               ),
  .i_rd_en          ( 1'b0                                )
);

data_sync #(
  .DATA_WIDTH                       (1                      ),
  .RESET_VALUE                      (1'b0                   ),
  .SYNC_DEPTH                       (2                      )
) u_fsm_busy_sync (
  .clk                              (i_aclk                 ),
  .rst_n                            (!i_arst                ),
  .sync_in                          (fsm_busy               ),
  .sync_out                         (fsm_busy_sync          )
);

assign sww_apb_m2s.psel    = sw_apb_m2s[1].psel && !fsm_busy_sync;
assign sww_apb_m2s.penable = sw_apb_m2s[1].penable && !fsm_busy_sync;
assign sww_apb_m2s.paddr   = sw_apb_m2s[1].paddr;
assign sww_apb_m2s.pwdata  = sw_apb_m2s[1].pwdata;
assign sww_apb_m2s.pwrite  = sw_apb_m2s[1].pwrite;

//------------------------------------------------------------------------------------------------//
// FSM
//------------------------------------------------------------------------------------------------//

typedef enum logic [3:0] {
  ILA_IDLE,
  ILA_CAPTURE,
  ILA_DONE,
  ILA_RESET
} ila_fsm_t;

ila_fsm_t ila_state;

logic wr_state;


always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    ila_state   <= ILA_IDLE;
    sample_addr <= '0;
    wr_state    <= '0;
  end
  else begin
    case (ila_state)
      ILA_IDLE: begin
        if (ila_rst) begin
          ila_state <= ILA_RESET;
          wr_state  <= '0;
        end 
        else if (trigger_en && i_enable) begin
          ila_state <= ILA_CAPTURE;
          wr_state  <= '1;
        end
        else begin
          ila_state <= ILA_IDLE;
          wr_state  <= '0;
        end
        sample_addr <= '0;
      end
      ILA_CAPTURE: begin
        if (!trigger_en) begin
          ila_state <= ILA_IDLE;
          wr_state  <= '0;
        end
        else if (i_trigger) begin
          sample_addr <= sample_addr + i_wr_en;
          if ((sample_addr == (DEPTH-1)) && i_wr_en) begin
            ila_state <= ILA_DONE;
            wr_state  <= '0;
          end
        end
        else begin
          ila_state <= ILA_CAPTURE;
          wr_state  <= '1;
        end
      end
      ILA_DONE: begin
        ila_state <= (!trigger_en && !ila_rst) ? ILA_IDLE : ILA_DONE;
        wr_state  <= '0;
      end
      ILA_RESET: begin
        sample_addr <= sample_addr + 1'b1;
        ila_state   <= (sample_addr == (DEPTH-1)) ? ILA_DONE : ILA_RESET;
        wr_state    <= '0;
      end
    endcase
  end
end


assign fsm_busy = (ila_state != ILA_IDLE);
assign fsm_done = (ila_state == ILA_DONE);

assign wr_data = (ila_rst) ? '0 : i_wr_data;
assign wr_en   = (ila_rst) ? '1 : (i_wr_en && trigger_en && wr_state);

endmodule
