// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


module eth_100gb_top
   (axis_rx_0_tdata,
    axis_rx_0_tkeep,
    axis_rx_0_tlast,
    axis_rx_0_tuser,
    axis_rx_0_tvalid,
    axis_tx_0_tdata,
    axis_tx_0_tkeep,
    axis_tx_0_tlast,
    axis_tx_0_tready,
    axis_tx_0_tuser,
    axis_tx_0_tvalid,
    o_ptp_clk,
    o_aclk,
    i_refclk_n,
    i_refclk_p,
    gt_serial_port_0_grx_n,
    gt_serial_port_0_grx_p,
    gt_serial_port_0_gtx_n,
    gt_serial_port_0_gtx_p,
    o_pll_locked,
    i_cmac_rst,
    o_usr_rst,
    o_usr_clk,
    init_clk,
    gt_powergoodout,
    gt_ref_clk_out,
    o_aligned
    );
  output [511:0]axis_rx_0_tdata;
  output [63:0]axis_rx_0_tkeep;
  output axis_rx_0_tlast;
  output axis_rx_0_tuser;
  output axis_rx_0_tvalid;
  input [511:0]axis_tx_0_tdata;
  input [63:0]axis_tx_0_tkeep;
  input axis_tx_0_tlast;
  output axis_tx_0_tready;
  input axis_tx_0_tuser;
  input axis_tx_0_tvalid;
  output o_ptp_clk;
  output o_aclk;
  input i_refclk_n;
  input i_refclk_p;
  input [3:0]gt_serial_port_0_grx_n;
  input [3:0]gt_serial_port_0_grx_p;
  output [3:0]gt_serial_port_0_gtx_n;
  output [3:0]gt_serial_port_0_gtx_p;
  output o_pll_locked;
  input i_cmac_rst;
  output o_usr_clk;
  output o_usr_rst;
  output o_aligned;
  input init_clk;
  output [3:0] gt_powergoodout;
  output gt_ref_clk_out;

  wire [511:0]axis_rx_0_tdata;
  wire [63:0]axis_rx_0_tkeep;
  wire axis_rx_0_tlast;
  wire axis_rx_0_tuser;
  wire axis_rx_0_tvalid;
  wire [511:0]axis_tx_0_tdata;
  wire [63:0]axis_tx_0_tkeep;
  wire axis_tx_0_tlast;
  wire axis_tx_0_tready;
  wire axis_tx_0_tuser;
  wire axis_tx_0_tvalid;
  wire i_refclk_n;
  wire i_refclk_p;
  wire [3:0]gt_serial_port_0_grx_n;
  wire [3:0]gt_serial_port_0_grx_p;
  wire [3:0]gt_serial_port_0_gtx_n;
  wire [3:0]gt_serial_port_0_gtx_p;
  wire o_pll_locked;
  wire i_pll_rst;
  wire i_cmac_rst;
  wire o_usr_rst;
  logic ctl_rx_0_ctl_enable;
  logic ctl_rx_0_ctl_rx_force_resync;
  logic ctl_rx_0_ctl_test_pattern;
  logic ctl_tx_0_ctl_enable;
  logic ctl_tx_0_ctl_test_pattern;
  logic ctl_tx_0_ctl_tx_send_idle;
  logic ctl_tx_0_ctl_tx_send_lfi;
  logic ctl_tx_0_ctl_tx_send_rfi;  
  logic stat_rx_aligned;
  logic o_aligned;
  
  

typedef enum logic [1:0] {
  IDLE = 2'b00,
  WAIT = 2'b01,
  RX_ALIGNED = 2'b10
} state_rx_init;

state_rx_init rx_init;


always_ff@(posedge o_usr_clk) begin
  if(o_usr_rst) begin
    ctl_rx_0_ctl_enable <= 1'b0;
    ctl_rx_0_ctl_rx_force_resync <= 1'b0;
    ctl_rx_0_ctl_test_pattern <= 1'b0;
    ctl_tx_0_ctl_enable <= 1'b0;
    ctl_tx_0_ctl_test_pattern <= 1'b0;
    ctl_tx_0_ctl_tx_send_idle <= 1'b0;
    ctl_tx_0_ctl_tx_send_lfi <= 1'b0;
    ctl_tx_0_ctl_tx_send_rfi <= 1'b0;
    rx_init <= IDLE;
  end
  else begin
    case(rx_init)
      IDLE: begin
        ctl_rx_0_ctl_enable <= 1'b1;
        ctl_tx_0_ctl_tx_send_rfi <= 1'b1;
        rx_init <= WAIT;
      end
      WAIT: begin
        if(stat_rx_aligned) begin
          ctl_tx_0_ctl_tx_send_rfi <= 1'b0;
          ctl_tx_0_ctl_enable <= 1'b1;
          rx_init <= RX_ALIGNED;
        end
      end
      RX_ALIGNED: begin
       // do nothing, data transmission and reception can be performed.
      end
      default: begin
        rx_init <= IDLE;
      end
    endcase
  end
end

  design_1 design_1_i
       (.axis_rx_0_tdata(axis_rx_0_tdata),
        .axis_rx_0_tkeep(axis_rx_0_tkeep),
        .axis_rx_0_tlast(axis_rx_0_tlast),
        .axis_rx_0_tuser(axis_rx_0_tuser),
        .axis_rx_0_tvalid(axis_rx_0_tvalid),
        .axis_tx_0_tdata(axis_tx_0_tdata),
        .axis_tx_0_tkeep(axis_tx_0_tkeep),
        .axis_tx_0_tlast(axis_tx_0_tlast),
        .axis_tx_0_tready(axis_tx_0_tready),
        .axis_tx_0_tuser(axis_tx_0_tuser),
        .axis_tx_0_tvalid(axis_tx_0_tvalid),
        .clk_out1_0(o_aclk), // 50MHz
        .clk_out2_0(o_ptp_clk), // 100MHz
        .core_drp_0_daddr('0),
        .core_drp_0_den('0),
        .core_drp_0_di('0),
        .core_drp_0_do(),
        .core_drp_0_drdy(),
        .core_drp_0_dwe('0),
        .ctl_rx_0_ctl_enable(ctl_rx_0_ctl_enable),
        .ctl_rx_0_ctl_rx_force_resync(ctl_rx_0_ctl_rx_force_resync),
        .ctl_rx_0_ctl_test_pattern(ctl_rx_0_ctl_test_pattern),
        .ctl_tx_0_ctl_enable(ctl_tx_0_ctl_enable),
        .ctl_tx_0_ctl_tx_send_idle(ctl_tx_0_ctl_tx_send_idle),
        .ctl_tx_0_ctl_tx_send_lfi(ctl_tx_0_ctl_tx_send_lfi),
        .ctl_tx_0_ctl_tx_send_rfi(ctl_tx_0_ctl_tx_send_rfi),
        .gt_ref_clk_0_clk_n(i_refclk_n), // 156.25MHz external ref clock
        .gt_ref_clk_0_clk_p(i_refclk_p), // 156.25MHz external ref clock
        .gt_loopback_in_0(12'b0),
        .gt_powergoodout_0(gt_powergoodout),
        .gt_ref_clk_out_0(gt_ref_clk_out),
        .gt_serial_port_0_grx_n(gt_serial_port_0_grx_n),
        .gt_serial_port_0_grx_p(gt_serial_port_0_grx_p),
        .gt_serial_port_0_gtx_n(gt_serial_port_0_gtx_n),
        .gt_serial_port_0_gtx_p(gt_serial_port_0_gtx_p),
        .locked_0(o_pll_locked),
        .init_clk_0(init_clk),// free running clock
        .sys_reset_0(i_cmac_rst),  // cmac system reset
        .gt_txusrclk2_0(o_usr_clk),
        .stat_rx_aligned_0(stat_rx_aligned),
        .usr_tx_reset_0(o_usr_rst));
endmodule
