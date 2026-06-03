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

module data_gen
  import apb_pkg::*;
  import regmap_pkg::*;
  import axis_pkg::*;
#(
  parameter  W_DATA      = 512,
  parameter  W_KEEP      = W_DATA/8,
  localparam GEN_DWIDTH  = (W_DATA == 8) ? 8 : 16
) (
  input  logic                i_apb_clk,
  input  logic                i_apb_rst,
  input  logic                i_sif_clk,
  input  logic                i_sif_rst,

  input  apb_m2s              i_apb_m2s,
  output apb_s2m              o_apb_s2m,

  output logic                o_data_gen_axis_mux,

  output logic                o_axis_tvalid,
  output logic [W_DATA-1:0]   o_axis_tdata,
  output logic [W_KEEP-1:0]   o_axis_tkeep,
  output logic                o_axis_tuser,
  output logic                o_axis_tlast,
  input  logic                i_axis_tready
);

localparam NUM_REPEATS    = W_DATA/GEN_DWIDTH;

logic [31:0]            ctrl_reg  [sen_rx_data_gen_nctrl];
logic [31:0]            stat_reg  [sen_rx_data_gen_nstat];

assign stat_reg = '{default:0};

logic                   axis_tvalid;
logic [W_DATA-1:0]      axis_tdata;
logic [W_KEEP-1:0]      axis_tkeep;
logic                   axis_tuser;
logic                   axis_tlast;
logic                   axis_tready;
logic                   output_ena;
logic                   data_gen_ena;
logic                   cont_mode;
logic                   data_gen_ena_q;
logic                   data_gen_ena_posedge;
logic                   data_gen_ena_negedge;
logic                   prbs_mode;
logic                   cnt_mode;
logic [GEN_DWIDTH-1:0]  prbs_dout;
logic [GEN_DWIDTH-1:0]  prbs_dout_reg /* synthesis noprune */;
logic [    W_DATA-1:0]  prbs_dout_vec;
logic [          31:0]  prbs_seed;
logic [GEN_DWIDTH-1:0]  counter_gen;
logic [    W_DATA-1:0]  counter_vec;
logic                   div_msb;
logic                   div_msb_r;
`ifdef SIMULATION
logic [           3:0]  output_div;
logic [           4:0]  output_div_cnt;
assign div_msb = output_div_cnt[4];
`else
logic [          15:0]  output_div;
logic [          16:0]  output_div_cnt;
assign div_msb = output_div_cnt[16];
`endif
logic [          31:0]  data_gen_size;
logic [          31:0]  data_gen_counter;
logic                   axis_mux;


assign data_gen_ena  = ctrl_reg[sen_rx_data_gen_ena][0];
assign cont_mode     = ctrl_reg[sen_rx_data_gen_ena][1];
assign prbs_mode     = ctrl_reg[sen_rx_data_gen_mode][1:0] == 2'd0;
assign cnt_mode      = ctrl_reg[sen_rx_data_gen_mode][1:0] == 2'd1;
assign data_gen_size = ctrl_reg[sen_rx_data_gen_size][31:0];
assign output_div    = ctrl_reg[sen_rx_data_gen_output_rate][15:0];

assign prbs_seed = ctrl_reg[sen_rx_prbs_seed];

s_apb_reg #(
  .N_CTRL                   ( sen_rx_data_gen_nctrl ),
  .N_STAT                   ( sen_rx_data_gen_nstat ),
  .W_OFST                   ( w_ofst                ),
  .R_CTRL                   ( 48                    )
) u_s_data_gen_apb_reg (
  // APB Interface
  .i_aclk                   ( i_apb_clk             ),
  .i_arst                   ( i_apb_rst             ),
  .i_apb_m2s                ( i_apb_m2s             ),
  .o_apb_s2m                ( o_apb_s2m             ),
  // User Control Signals
  .i_pclk                   ( i_sif_clk             ),
  .i_prst                   ( i_sif_rst             ),
  .o_ctrl                   ( ctrl_reg              ),
  .i_stat                   ( stat_reg              )
);

always_ff @(posedge i_sif_clk) begin
  if (i_sif_rst) begin
    data_gen_ena_q <= 1'b0;
  end
  else begin
    data_gen_ena_q <= data_gen_ena;
  end
end

assign data_gen_ena_posedge = data_gen_ena  && !data_gen_ena_q;
assign data_gen_ena_negedge = !data_gen_ena && data_gen_ena_q;

//------------------------------------------------------------------------------------------------//
// State Machine
//------------------------------------------------------------------------------------------------//
typedef enum logic [1:0] {
  IDLE,
  GEN_DATA,
  DONE
} states;
states state;

always_ff @(posedge i_sif_clk) begin
  if (i_sif_rst) begin
    data_gen_counter <= '0;
    axis_tlast       <= 1'b0;
    output_ena       <= 1'b0;
    output_div_cnt   <= '0;
    div_msb_r        <= 1'b0;
    state            <= IDLE;
  end
  else begin
    div_msb_r <= div_msb;
    case (state)
      IDLE: begin
        data_gen_counter <= '0;
        output_div_cnt   <= '0;
        axis_tlast       <= 1'b0;
        if (data_gen_ena_posedge) begin
          state    <= GEN_DATA;
        end
        else begin
          state <= IDLE;
        end
      end
      GEN_DATA: begin
        if (data_gen_ena_negedge) begin
          axis_tlast     <= 1'b1;
          output_ena     <= 1'b1;
          state          <= DONE;
        end
        else begin
          output_div_cnt     <= output_div_cnt + output_div;
          if (axis_tready) begin
            if (div_msb != div_msb_r) begin
              output_ena         <= 1'b1;
              data_gen_counter   <= data_gen_counter + W_KEEP;
              if ((data_gen_counter + W_KEEP >= data_gen_size) & !cont_mode) begin
                axis_tlast     <= 1'b1;
                state          <= DONE;
              end
              else begin
                axis_tlast     <= 1'b0;
                state          <= GEN_DATA;
              end
            end
            else begin
              output_ena       <= 1'b0;
            end
          end
          else begin
            output_ena       <= output_ena;
          end
        end
      end
      DONE: begin
        if (axis_tready) begin
          data_gen_counter <= '0;
          axis_tlast       <= 1'b0;
          output_ena       <= 1'b0;
          state            <= IDLE;
        end
      end
      default: begin
        state <= IDLE;
      end
    endcase
  end
end

//------------------------------------------------------------------------------------------------//
// PRBS Mode
//------------------------------------------------------------------------------------------------//


logic [30:0]           crc;
logic [30:0]           crc_next;

always @(posedge i_sif_clk) begin
  if (i_sif_rst) begin
    crc <= '0;
  end
  else begin
    if (data_gen_ena_posedge) begin
      crc <= prbs_seed[30:0];
    end
    else if (prbs_mode && output_ena && axis_tready) begin
      crc <= crc_next;
    end
  end
end

// Next-state from unrolled PRBS LFSR (direct mapping)
prbs_lfsr_hc #(
  .DATA_WIDTH               (GEN_DWIDTH)
) u_prbs_lfsr_hc (
  .data_in                  ('0       ),
  .state_in                 (crc      ),
  .state_out                (crc_next )
);

assign prbs_dout = ~crc_next[GEN_DWIDTH-1:0];


//------------------------------------------------------------------------------------------------//
// Counter Mode
//------------------------------------------------------------------------------------------------//
always_ff @(posedge i_sif_clk) begin
  if (i_sif_rst) begin
    counter_gen              <= '0;
  end
  else begin
    if (cnt_mode) begin
      if (output_ena && axis_tready) begin
        counter_gen          <= counter_gen + NUM_REPEATS;
      end
    end
    else begin
      counter_gen            <= '0;
    end
  end
end

genvar k;
generate
for (k=0;k<NUM_REPEATS;k++) begin
  always_comb begin
    counter_vec  [(GEN_DWIDTH*(k+1))-1:GEN_DWIDTH*k] = counter_gen + k;
    // Offset replication: each segment gets prbs_dout + offset
    prbs_dout_vec[(GEN_DWIDTH*(k+1))-1:GEN_DWIDTH*k] = prbs_dout + k;
  end
end
endgenerate

assign  axis_tdata        = output_ena ? (cnt_mode ? counter_vec : prbs_mode ? prbs_dout_vec : 0) : 0;
assign  axis_tvalid       = output_ena && (prbs_mode || cnt_mode);
assign  axis_tkeep        = output_ena ? '1 : '0;
assign  axis_tuser        = 1'b0;

assign  o_data_gen_axis_mux = data_gen_ena;

//------------------------------------------------------------------------------------------------//
// Output Buffer
//------------------------------------------------------------------------------------------------//

axis_buffer # (
  .IN_DWIDTH   ( W_DATA         ),
  .OUT_DWIDTH  ( W_DATA         ),
  .WAIT2SEND   ( 0              ),
  .BUF_DEPTH   ( 16             ),
  .DUAL_CLOCK  ( 0              ),
  .OUTPUT_SKID ( '0             )
) u_axis_buffer (
  .in_clk            ( i_sif_clk             ),
  .in_rst            ( i_sif_rst             ),
  .out_clk           ( i_sif_clk             ),
  .out_rst           ( i_sif_rst             ),
  .i_axis_rx_tvalid  ( axis_tvalid           ),
  .i_axis_rx_tdata   ( axis_tdata            ),
  .i_axis_rx_tlast   ( axis_tlast            ),
  .i_axis_rx_tuser   ( axis_tuser            ),
  .i_axis_rx_tkeep   ( axis_tkeep            ),
  .o_axis_rx_tready  ( axis_tready           ),
  .o_fifo_aempty     (                       ),
  .o_fifo_afull      (                       ),
  .o_fifo_empty      (                       ),
  .o_fifo_full       (                       ),
  .o_axis_tx_tvalid  ( o_axis_tvalid         ),
  .o_axis_tx_tdata   ( o_axis_tdata          ),
  .o_axis_tx_tlast   ( o_axis_tlast          ),
  .o_axis_tx_tuser   ( o_axis_tuser          ),
  .o_axis_tx_tkeep   ( o_axis_tkeep          ),
  .i_axis_tx_tready  ( i_axis_tready         )
);

endmodule
