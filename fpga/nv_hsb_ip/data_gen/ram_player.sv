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

module ram_player
  import apb_pkg::*;
  import regmap_pkg::*;
  import axis_pkg::*;
#(
  parameter  W_DATA      = 512,
  parameter  W_KEEP      = W_DATA/8,
  parameter  RAM_DEPTH   = 512
) (
  input  logic                i_apb_clk,
  input  logic                i_apb_rst,
  input  logic                i_sif_clk,
  input  logic                i_sif_rst,

  input  apb_m2s              i_apb_m2s,
  output apb_s2m              o_apb_s2m,

  output logic                o_ram_axis_mux,

  input  logic  [79:0]        i_ptp,

  output logic                o_axis_tvalid,
  output logic [W_DATA-1:0]   o_axis_tdata,
  output logic [W_KEEP-1:0]   o_axis_tkeep,
  output logic                o_axis_tuser,
  output logic                o_axis_tlast,
  input  logic                i_axis_tready
);
localparam sen_rx_ram_nctrl = 5;
localparam sen_rx_ram_nstat = 1;

localparam sen_rx_ram_scratch = 0;  // 0x0000_0000
localparam sen_rx_ram_ena = 1;  // 0x0000_0004
localparam sen_rx_ram_timer = 2;  // 0x0000_0008  
localparam sen_rx_ram_window_size = 3;  // 0x0000_000C
localparam sen_rx_ram_window_num = 4;  // 0x0000_0010

logic [31:0]            ctrl_reg  [sen_rx_ram_nctrl];
logic [31:0]            stat_reg  [sen_rx_ram_nstat];

assign stat_reg = '{default:0};

logic                   ram_ena, ram_ena_q;
logic [          31:0]  timer;
logic [          31:0]  data_cnt;
logic                   axis_mux;
logic [          31:0]  window_size;
logic [          31:0]  wait_time;
logic [          31:0]  window_num, window_cnt; 
logic [W_DATA-1:0]      rd_data, rd_data_lat;
logic                   rd_en;
logic [           8:0]  rd_addr;
logic                   axis_tready_q, axis_tready_lat;
logic                   send_fst_cyc;
logic                   ptp_ena;
logic                   ptp_bram_ena;
logic                   loop_dis;

assign ram_ena        = ctrl_reg[sen_rx_ram_ena][0];
assign ptp_ena        = ctrl_reg[sen_rx_ram_ena][1];
assign loop_dis       = ctrl_reg[sen_rx_ram_ena][2];
assign ptp_bram_ena   = ctrl_reg[sen_rx_ram_ena][3];
assign timer          = ctrl_reg[sen_rx_ram_timer];

assign window_size    = ctrl_reg[sen_rx_ram_window_size];
assign window_num     = ctrl_reg[sen_rx_ram_window_num];

apb_m2s reg_m2s;
apb_s2m reg_s2m;
apb_m2s ram_m2s;
apb_s2m ram_s2m;

localparam logic [31:0] RAM_BASE_ADDR = 32'h10_0000;

//------------------------------------------------------------------------------------------------//
// register: 0x0000_0000 - 0x000_1000
//ram_ena: 0x0000_0004
//timer: 0x0000_0008
//window_size: 0x0000_000C
//window_num: 0x0000_0010

// ram: 0x0000_2000 - 
//------------------------------------------------------------------------------------------------//

always_comb begin
  reg_m2s = '0;
  ram_m2s = '0;
  o_apb_s2m = '0;
  if (i_apb_m2s.paddr[21:20] == 2'b00) begin
    reg_m2s   = i_apb_m2s;
    o_apb_s2m = reg_s2m;
  end
  else if (i_apb_m2s.paddr[21:20] == 2'b01) begin
    ram_m2s   = i_apb_m2s;
    // Strip RAM base so u_s_apb_ram_dyn sees bank index in paddr[31:10].
    ram_m2s.paddr = i_apb_m2s.paddr - RAM_BASE_ADDR;
    o_apb_s2m     = ram_s2m;
  end
end

//------------------------------------------------------------------------------------------------//
// APB Register
//------------------------------------------------------------------------------------------------//

s_apb_reg #(
  .N_CTRL                   ( sen_rx_ram_nctrl ),
  .N_STAT                   ( sen_rx_ram_nstat ),
  .W_OFST                   ( w_ofst           ),
  .R_CTRL                   ( 48               )
) u_s_ram_apb_reg (
  // APB Interface
  .i_aclk                   ( i_apb_clk        ),
  .i_arst                   ( i_apb_rst        ),
  .i_apb_m2s                ( reg_m2s          ),
  .o_apb_s2m                ( reg_s2m          ),
  // User Control Signals
  .i_pclk                   ( i_sif_clk        ),
  .i_prst                   ( i_sif_rst        ),
  .o_ctrl                   ( ctrl_reg         ),
  .i_stat                   ( stat_reg         )
);

//------------------------------------------------------------------------------------------------//
// RAM Player
//------------------------------------------------------------------------------------------------//  

s_apb_ram_dyn #(
  .DEPTH                   ( RAM_DEPTH ),
  .W_DATA                  ( W_DATA    )
) u_s_apb_ram_dyn (
  .i_aclk                  ( i_apb_clk ),
  .i_arst                  ( i_apb_rst ),
  .i_apb_m2s               ( ram_m2s   ),
  .o_apb_s2m               ( ram_s2m   ),
  .i_pclk                  ( i_sif_clk ),
  .i_prst                  ( i_sif_rst ),
  .i_addr                  ( rd_addr   ),
  .o_rd_data               ( rd_data   ),
  .o_rd_data_valid         (           ), // for this case, no ECB wr/rd when read data
  .i_wr_data               (           ),
  .i_wr_en                 (           ),
  .i_rd_en                 ( rd_en     )
);


logic tvalid_q;
logic tready_q;
logic [79:0] ptp_q;
logic [79:0] ptp;

always_ff @(posedge i_sif_clk) begin
  if (i_sif_rst) begin
    tvalid_q <= '0;
    tready_q <= '0;
    ptp_q    <= '0;
  end
  else begin
    tvalid_q <= o_axis_tvalid;
    tready_q <= i_axis_tready;
    ptp_q    <= (o_axis_tvalid && !tvalid_q) || (tready_q) ? i_ptp : ptp_q; // Hold value on tvalid
  end
end

assign ptp = ((o_axis_tvalid && !tvalid_q) || (tready_q)) ? i_ptp : ptp_q;

//------------------------------------------------------------------------------------------------//
// State Machine
//------------------------------------------------------------------------------------------------//
typedef enum logic [1:0] {
  IDLE,
  SEND_DATA,
  WAIT_TIME,
  DONE
} states;
states state;

logic [W_DATA-1:0]      axis_tdata;
logic                   sof;

always_ff @(posedge i_sif_clk) begin
  if (i_sif_rst) begin
    data_cnt        <= '0;
    o_axis_tlast    <= 1'b0;
    o_axis_tvalid   <= 1'b0;
    axis_tdata      <= '0;
    ram_ena_q       <= 1'b0;
    rd_addr         <= '0;
    rd_en           <= 1'b0;
    wait_time       <= '0;
    window_cnt      <= '0;
    rd_data_lat     <= '0;
    axis_tready_q   <= 1'b0;
    axis_tready_lat <= 1'b0;
    send_fst_cyc    <= 1'b0;  
    state           <= IDLE;
    sof             <= 1'b0;
  end
  else begin
    axis_tready_q <= i_axis_tready;
    case (state)
      IDLE: begin
        ram_ena_q       <= ram_ena;
        data_cnt        <= '0;
        wait_time       <= '0;
        window_cnt      <= '0;
        rd_en           <= 1'b1;
        rd_data_lat     <= '0;
        axis_tready_lat <= 1'b0;
        send_fst_cyc    <= 1'b0;
        if (ram_ena) begin
          if (ram_ena_q) begin
            state         <= SEND_DATA;
            o_axis_tvalid <= 1'b1;
            sof           <= 1'b1;
            o_axis_tlast  <= (data_cnt >= window_size - W_KEEP) ? 1'b1 : 1'b0;
            axis_tdata    <= rd_data;
            data_cnt      <= (data_cnt >= window_size - W_KEEP) ? '0 : data_cnt + W_KEEP;
            rd_addr       <= (data_cnt >= window_size - W_KEEP) ? rd_addr : rd_addr + 1'b1;
          end
          else begin
            rd_addr <= rd_addr + 1'b1;
          end
        end
        else begin
          rd_addr       <= '0;
          o_axis_tlast  <= 1'b0;
          o_axis_tvalid <= 1'b0;
        end
      end

      SEND_DATA: begin
        send_fst_cyc   <= 1'b1;
        if (!ram_ena) begin
          o_axis_tlast  <= 1'b1;
          o_axis_tvalid <= 1'b1;
          state         <= i_axis_tready ? IDLE : DONE;
        end
        else begin
          if (i_axis_tready) begin
            sof           <= 1'b0;
            axis_tdata <= (i_axis_tready && !axis_tready_q && axis_tready_lat) ? rd_data_lat : rd_data;
            if (o_axis_tlast) begin
              o_axis_tlast  <= 1'b0;
              o_axis_tvalid <= 1'b0;
              window_cnt    <= window_cnt + 1'b1;
              data_cnt      <= '0;
              if (window_cnt >= window_num - 1) begin
                state      <= ((timer == '0) || loop_dis) ? DONE : WAIT_TIME;
                rd_addr    <= '0;
                window_cnt <= '0;
              end
              else begin
                state      <= (timer == '0) ? SEND_DATA : WAIT_TIME;
                rd_addr    <= ram_ena_q ? rd_addr : rd_addr + 1'b1;
              end
            end
            else if (data_cnt >= window_size - W_KEEP) begin
              o_axis_tlast  <= 1'b1;
              o_axis_tvalid <= 1'b1;
            end
            else begin
              rd_addr       <= rd_addr + 1'b1;
              o_axis_tlast  <= 1'b0; 
              o_axis_tvalid <= 1'b1;
              data_cnt      <= data_cnt + W_KEEP;
            end
          end
          else if (!i_axis_tready && axis_tready_q) begin
            rd_data_lat <= rd_data;
            axis_tready_lat <= 1'b1;
          end
          else if(!i_axis_tready && !send_fst_cyc) begin
            rd_data_lat <= rd_data;
            axis_tready_lat <= 1'b1;
          end
        end
      end

      WAIT_TIME: begin
        send_fst_cyc    <= 1'b0;
        axis_tready_lat <= 1'b0;
        if (i_axis_tready && o_axis_tlast) begin
          o_axis_tvalid <= 1'b0;
          o_axis_tlast  <= 1'b0;
        end
        else if (!ram_ena) begin
          state <= IDLE;
        end
        else if (wait_time == timer - 1) begin
          rd_addr   <= rd_addr + 1'b1;
          wait_time <= wait_time + 1'b1;
        end
        else if (wait_time == timer) begin
          state         <= SEND_DATA;
          o_axis_tlast  <= (data_cnt >= window_size - W_KEEP) ? 1'b1 : 1'b0;
          sof           <= 1'b1;
          o_axis_tvalid <= 1'b1;
          axis_tdata    <= rd_data;
          rd_addr       <= (data_cnt >= window_size - W_KEEP) ? rd_addr : rd_addr + 1'b1;
          data_cnt      <= (data_cnt >= window_size - W_KEEP) ? '0 : data_cnt + W_KEEP;
          wait_time     <= '0;
        end
        else begin
          wait_time <= wait_time + 1'b1;
        end
      end

      DONE: begin
        if (!ram_ena) begin
          state <= IDLE;
        end
      end

      default: begin
        state <= IDLE;
      end
    endcase
  end
end

assign  o_axis_tkeep        = o_axis_tvalid ? '1 : '0;
assign  o_axis_tuser        = 1'b0;
assign  o_ram_axis_mux      = ram_ena;


assign o_axis_tdata = (ptp_bram_ena && sof) ? {axis_tdata[511:192],ptp[63:0],axis_tdata[127:0]} : 
                      ptp_ena               ? {'0,window_cnt[11:0],ptp}                         :
                                               axis_tdata;

endmodule
