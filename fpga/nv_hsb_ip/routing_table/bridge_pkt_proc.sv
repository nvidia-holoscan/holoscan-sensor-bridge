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


module bridge_pkt_proc
  import ptp_pkg::*;
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter         AXI_DWIDTH                 = 64,
  parameter         NUM_HOST                   = 2,
  parameter         NUM_SENSOR                 = 8,
  parameter         SIF_DWIDTH                 = 64,
  parameter         FREQ                       = 156250000, // clock period in Hz
  parameter integer W_SIF_TX  [NUM_SENSOR-1:0] = '{default:AXI_DWIDTH},
  parameter         HOST_MTU                   = 4000,
  localparam        TOTAL_TX                   = NUM_HOST + NUM_SENSOR,
  localparam        KEEP_WIDTH                 = (AXI_DWIDTH / 8),
  localparam        W_SIF_KEEP                 = (SIF_DWIDTH / 8),
  parameter         SYNC_CLK                   = 0
)(
  input   logic                     i_pclk,
  input   logic                     i_prst,
  input   logic  [NUM_SENSOR-1:0]   i_sif_clk,
  input   logic  [NUM_SENSOR-1:0]   i_sif_rst,
  // Register Map, abp clk domain
  input                             i_aclk,
  input                             i_arst,
  input  apb_m2s                    i_apb_m2s,
  output apb_s2m                    o_apb_s2m,

  input   logic   [  NUM_HOST-1:0]  i_hif_axis_tvalid,
  input   logic   [AXI_DWIDTH-1:0]  i_hif_axis_tdata [NUM_HOST-1:0],
  input   logic   [  NUM_HOST-1:0]  i_hif_axis_tlast,
  input   logic   [           1:0]  i_hif_axis_tuser [NUM_HOST-1:0],
  input   logic   [KEEP_WIDTH-1:0]  i_hif_axis_tkeep [NUM_HOST-1:0],
  output  logic   [  NUM_HOST-1:0]  o_hif_axis_tready,

  input   logic   [NUM_SENSOR-1:0]  i_sif_axis_tvalid,
  input   logic   [AXI_DWIDTH-1:0]  i_sif_axis_tdata [NUM_SENSOR-1:0],
  input   logic   [NUM_SENSOR-1:0]  i_sif_axis_tlast,
  input   logic   [NUM_SENSOR-1:0]  i_sif_axis_tuser,
  input   logic   [KEEP_WIDTH-1:0]  i_sif_axis_tkeep [NUM_SENSOR-1:0],
  output  logic   [NUM_SENSOR-1:0]  o_sif_axis_tready,

  input           [          47:0]  i_dest_info      [NUM_HOST],

  input                             i_ptp_sync_msg,

  input           [NUM_SENSOR-1:0]  i_chk_del_req,
  input           [NUM_SENSOR-1:0]  i_is_del_req,

  input           [  NUM_HOST-1:0]  i_del_req_val,

  input           [ TOTAL_TX-1:0]   i_dest_port,
  input           [ NUM_HOST-1:0]   i_dest_val,
  output          [         47:0]   o_mac_addr       [NUM_HOST],
  output          [ NUM_HOST-1:0]   o_mac_req,

  output  logic   [  NUM_HOST-1:0]  o_hif_axis_tvalid,
  output  logic   [AXI_DWIDTH-1:0]  o_hif_axis_tdata [NUM_HOST-1:0],
  output  logic   [  NUM_HOST-1:0]  o_hif_axis_tlast,
  output  logic   [  NUM_HOST-1:0]  o_hif_axis_tuser,
  output  logic   [KEEP_WIDTH-1:0]  o_hif_axis_tkeep [NUM_HOST-1:0],
  input   logic   [NUM_HOST  -1:0]  i_hif_axis_tready,

  output  logic   [NUM_SENSOR-1:0]  o_sif_axis_tvalid,
  output  logic   [SIF_DWIDTH-1:0]  o_sif_axis_tdata [NUM_SENSOR-1:0],
  output  logic   [NUM_SENSOR-1:0]  o_sif_axis_tlast,
  output  logic   [NUM_SENSOR-1:0]  o_sif_axis_tuser,
  output  logic   [W_SIF_KEEP-1:0]  o_sif_axis_tkeep [NUM_SENSOR-1:0],
  input   logic   [NUM_SENSOR-1:0]  i_sif_axis_tready
);

localparam HIF_MUX_WIDTH = $clog2(NUM_HOST + NUM_SENSOR);
localparam SIF_MUX_WIDTH = $clog2(NUM_HOST + NUM_SENSOR);

logic [($clog2(NUM_HOST)):0] ptp_port;
assign ptp_port = '0;

//------------------------------------------------------------------------------------------------//
// Register Map
//------------------------------------------------------------------------------------------------//

localparam bridge_nctrl = NUM_HOST*2 + 1;
localparam bridge_nstat = 1;
localparam bridge_stat  = 32; // 0x0000_0080

logic   [ TOTAL_TX-1:0]  clear_to_send;
logic                    is_gnt;


logic [31:0] ctrl_reg [bridge_nctrl];
logic [31:0] stat_reg [bridge_nstat];

assign stat_reg[bridge_stat-stat_ofst] = '{default:0};

s_apb_reg #(
  .N_CTRL    ( bridge_nctrl   ),
  .N_STAT    ( bridge_nstat   ),
  .W_OFST    ( w_ofst         ),
  .SYNC_CLK  ( SYNC_CLK       )
) u_reg_map  (
  // APB Interface
  .i_aclk    ( i_aclk         ),
  .i_arst    ( i_arst         ),
  .i_apb_m2s ( i_apb_m2s      ),
  .o_apb_s2m ( o_apb_s2m      ),
  // User Control Signals
  .i_pclk    ( i_pclk         ),
  .i_prst    ( i_prst         ),
  .o_ctrl    ( ctrl_reg       ),
  .i_stat    ( stat_reg       )
);

//------------------------------------------------------------------------------------------------//
// PTP Parameters
//------------------------------------------------------------------------------------------------//

localparam [W_NS:0]   default_inc = 10**9 * (2**W_FRAC_NS) / FREQ ;
localparam            W_RES       = 16;   // Max width of cycles of res time
localparam            IS_PTP_DEL  = 2;    // Routing table takes 2 cycles to parse is_ptp
localparam            W_NS_RES    = 24;   // Max width of NS of res time

//------------------------------------------------------------------------------------------------//
// HIF RX
//------------------------------------------------------------------------------------------------//

logic   [  NUM_HOST-1:0] hif_buf_axis_tvalid;
logic   [AXI_DWIDTH-1:0] hif_buf_axis_tdata [NUM_HOST-1:0];
logic   [  NUM_HOST-1:0] hif_buf_axis_tlast;
logic   [           1:0] hif_buf_axis_tuser [NUM_HOST-1:0];
logic   [KEEP_WIDTH-1:0] hif_buf_axis_tkeep [NUM_HOST-1:0];
logic   [  NUM_HOST-1:0] hif_buf_axis_tready;

logic   [           1:0] sif_mux_axis_tuser [NUM_SENSOR-1:0];
logic   [NUM_SENSOR-1:0] sif_mux_axis_tvalid;

logic   [  NUM_HOST-1:0]  axis_sop;
logic   [  NUM_HOST-1:0]  mac_req;
logic   [  TOTAL_TX-1:0]  dest_port [NUM_HOST];
logic   [  TOTAL_TX-1:0]  pkt_active;
logic   [          47:0]  dest_info [NUM_HOST];
logic   [  NUM_HOST-1:0]  dest_buf_rd;
logic   [  NUM_HOST-1:0]  dest_empty;
logic   [  NUM_HOST-1:0]  inp_pkt_active;
logic   [  NUM_HOST-1:0]  hif_axis_tready;
logic   [  NUM_HOST-1:0]  dest_val_reg;

logic [W_RES-1:0] hif_res_time;
logic             hif_res_latch;
logic [63:0]      hif_correction_val;
logic [47:0]      hif_cf;
logic [47:0]      hif_cf_be;  //big endian

logic [HIF_MUX_WIDTH:0]         gnt_idx;
logic [$clog2(NUM_SENSOR)-1:0]  sif_gnt_idx;
logic [$clog2(NUM_SENSOR)-1:0]  r_sif_gnt_idx [IS_PTP_DEL:0];

logic [NUM_SENSOR-1:0] is_ptp;
logic [NUM_SENSOR-1:0] is_ptp_mac;
logic [NUM_SENSOR-1:0] is_ptp_pkt;
logic [NUM_HOST-1:0]   dest_buf_rd_active;
logic [NUM_HOST-1:0]   hif_tx_buffer_afull;
logic [NUM_SENSOR-1:0] sif_tx_buffer_afull;

genvar i,j;

generate
  for (i=0;i<NUM_HOST;i=i+1) begin : gen_hif_rx
    axis_buffer # (
      .IN_DWIDTH         ( AXI_DWIDTH                                     ),
      .OUT_DWIDTH        ( AXI_DWIDTH                                     ),
      .WAIT2SEND         ( 1                                              ),
      .W_USER            ( 2                                              )
    ) u_axis_hif_rx_buffer (
      .in_clk            ( i_pclk                                         ),
      .in_rst            ( i_prst                                         ),
      .out_clk           ( i_pclk                                         ),
      .out_rst           ( i_prst                                         ),
      .i_axis_rx_tvalid  ( (i_hif_axis_tvalid [i] && hif_axis_tready [i]) ),
      .i_axis_rx_tdata   ( i_hif_axis_tdata    [i]                        ),
      .i_axis_rx_tlast   ( i_hif_axis_tlast    [i]                        ),
      .i_axis_rx_tuser   ( i_hif_axis_tuser    [i]                        ),
      .i_axis_rx_tkeep   ( i_hif_axis_tkeep    [i]                        ),
      .o_axis_rx_tready  (                                                ),
      .o_axis_tx_tvalid  ( hif_buf_axis_tvalid [i]                        ),
      .o_axis_tx_tdata   ( hif_buf_axis_tdata  [i]                        ),
      .o_axis_tx_tlast   ( hif_buf_axis_tlast  [i]                        ),
      .o_axis_tx_tuser   ( hif_buf_axis_tuser  [i]                        ),
      .o_axis_tx_tkeep   ( hif_buf_axis_tkeep  [i]                        ),
      .i_axis_tx_tready  ( hif_buf_axis_tready [i]                        )
    );

    sc_fifo #(
      .DATA_WIDTH ( TOTAL_TX                          ),
      .FIFO_DEPTH ( 31                                )
    ) dest_buffer (
      .clk        ( i_pclk                            ),
      .rst        ( i_prst                            ),
      .wr         ( i_dest_val[i] && !dest_val_reg[i] ),
      .din        ( i_dest_port                       ),
      .full       (                                   ),
      .afull      (                                   ),
      .over       (                                   ),
      .rd         ( dest_buf_rd[i]                    ),
      .dout       ( dest_port[i]                      ),
      .dval       (                                   ),
      .empty      ( dest_empty[i]                     ),
      .aempty     (                                   ),
      .under      (                                   ),
      .count      (                                   )
    );

    always_ff @(posedge i_pclk) begin
      if (i_prst  ) begin
        mac_req   [i]     <= '0;
        inp_pkt_active[i] <= '0;
        dest_info [i]     <= '0;
        dest_val_reg[i]   <= '0;
        hif_axis_tready[i]<= '1;
      end
      else begin
        dest_val_reg[i]   <= i_dest_val[i];
        if (inp_pkt_active[i]) begin
          if ((i_hif_axis_tvalid[i] && i_hif_axis_tlast[i]) || (!hif_axis_tready[i])) begin // End of Packet, or backpressure is enabled (packet already ended)
            if (({dest_val_reg[i],i_dest_val[i]} == 2'b10) || (!mac_req[i] && !i_dest_val[i])) begin   // has CAM response
              inp_pkt_active[i]  <= '0;
              hif_axis_tready[i] <= '1;
            end
            else begin  // Doesn't have response yet
              inp_pkt_active[i]  <= '1;
              hif_axis_tready[i] <= '0; // Enable backpressure
            end
          end
          if ({dest_val_reg[i],i_dest_val[i]} == 2'b01) begin  // Cam response
            mac_req[i] <= '0;       // Disable request
          end
        end
        else begin
          mac_req    [i]  <= i_hif_axis_tvalid[i];
          dest_info  [i]  <= (!i_dest_info[i][0]) ? i_dest_info[i] : // Send packet if not broadcast packet
                              (ctrl_reg[0][i]    ) ? i_dest_info[i] : // Send packet if broadcast && port is broadcast-able
                                                    '0             ; // Drop packet if broadcast && port is non-broadcast-able
          inp_pkt_active[i]  <= i_hif_axis_tvalid[i];
          hif_axis_tready[i] <= !(i_hif_axis_tlast[i] && i_hif_axis_tvalid[i]); // Backpressure if single beat packet
        end
      end
    end

    assign o_hif_axis_tready[i] = hif_axis_tready[i];  // backpressure until current packet has dest info responded to

    always_ff @(posedge i_pclk) begin
      if (i_prst  ) begin
        pkt_active[i+NUM_SENSOR] <= '0;
        dest_buf_rd_active[i]    <= '0;
      end
      else begin
        if (pkt_active[i+NUM_SENSOR]) begin
          pkt_active[i+NUM_SENSOR] <= !(hif_buf_axis_tvalid[i] && hif_buf_axis_tlast[i]);
          dest_buf_rd_active[i]    <= !(hif_buf_axis_tvalid[i] && hif_buf_axis_tlast[i]);
        end
        else begin
          pkt_active [i+NUM_SENSOR] <= hif_buf_axis_tvalid[i] && (gnt_idx == (i+NUM_SENSOR)) && is_gnt;
          dest_buf_rd_active[i]     <= (dest_buf_rd_active[i]) ? '1 : !dest_empty[i];
        end
      end
    end
    assign clear_to_send[i+NUM_SENSOR] = hif_buf_axis_tvalid[i] && !pkt_active[i+NUM_SENSOR] && dest_buf_rd_active[i];
    assign dest_buf_rd[i] = (!dest_buf_rd_active[i] && !dest_empty[i]);
    assign hif_buf_axis_tready[i] = pkt_active[i+NUM_SENSOR];
  end
endgenerate

assign o_mac_req  = mac_req;
assign o_mac_addr = dest_info;


always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    hif_res_time       <= '0;
    hif_res_latch      <= '1;
    hif_correction_val <= '0;
  end
  else begin
    hif_res_time <= (i_ptp_sync_msg && hif_res_latch)                                                      ? '0                 : // Reset Timer
                      hif_res_latch                                                                        ? hif_res_time       : // Stop Timer
                                                                                                              hif_res_time + 1'b1;
    hif_res_latch <= (sif_mux_axis_tuser[0] == 'd1) && (sif_mux_axis_tvalid[0])                            ? '1                 : // Stop Timer, all SIF receive PTP at same time
                      i_ptp_sync_msg                                                                        ? '0                 : // Start Timer
                                                                                                              hif_res_latch      ;
    hif_correction_val <= hif_res_time * default_inc;
  end
end

assign hif_cf = {'0,hif_correction_val[W_FRAC_NS+:W_NS_RES]};

generate
  for (i=0; i<6; i++) begin
    assign hif_cf_be[i*8+:8] = hif_cf[(6-1-i)*8+:8];
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// SIF RX
//------------------------------------------------------------------------------------------------//

logic   [NUM_SENSOR-1:0] sif_buf_axis_tvalid;
logic   [AXI_DWIDTH-1:0] sif_buf_axis_tdata [NUM_SENSOR-1:0];
logic   [NUM_SENSOR-1:0] sif_buf_axis_tlast;
logic   [           1:0] sif_buf_axis_tuser [NUM_SENSOR-1:0];
logic   [KEEP_WIDTH-1:0] sif_buf_axis_tkeep [NUM_SENSOR-1:0];
logic   [NUM_SENSOR-1:0] sif_buf_axis_tready;

generate
  for (i=0;i<NUM_SENSOR;i=i+1) begin : gen_sif_rx
    always_ff @(posedge i_pclk) begin
      if (i_prst  ) begin
        pkt_active[i]          <= '0;
        clear_to_send[i]       <= '0;
      end
      else begin
        if (pkt_active[i]) begin
          pkt_active[i]          <= !(sif_buf_axis_tvalid[i] && sif_buf_axis_tlast[i]);
          clear_to_send[i]       <= '0;
        end
        else begin
          pkt_active [i]         <= sif_buf_axis_tvalid[i] && (gnt_idx == i) && is_gnt;
          clear_to_send[i]       <= sif_buf_axis_tvalid[i];
        end
      end
    end
    assign sif_buf_axis_tready[i] = pkt_active[i];
  end
endgenerate

assign sif_buf_axis_tvalid  = i_sif_axis_tvalid;
assign sif_buf_axis_tdata   = i_sif_axis_tdata ;
assign sif_buf_axis_tlast   = i_sif_axis_tlast ;
assign sif_buf_axis_tuser   = '{default:0}     ;
assign sif_buf_axis_tkeep   = i_sif_axis_tkeep ;
assign o_sif_axis_tready    = sif_buf_axis_tready;

//------------------------------------------------------------------------------------------------//
// Arbitration - Sets ready signal + MUX Select
//------------------------------------------------------------------------------------------------//
logic   [  NUM_HOST-1:0] hif_mux_axis_tlast;
logic   [NUM_SENSOR-1:0] sif_mux_axis_tlast;

logic   [  NUM_HOST-1:0] hif_mux_axis_tvalid;
logic   [AXI_DWIDTH-1:0] hif_mux_axis_tdata [NUM_HOST];
logic   [           1:0] hif_mux_axis_tuser [NUM_HOST];
logic   [KEEP_WIDTH-1:0] hif_mux_axis_tkeep [NUM_HOST];
logic   [  NUM_HOST-1:0] hif_mux_axis_tready;

logic   [AXI_DWIDTH-1:0] sif_mux_axis_tdata [NUM_SENSOR];
logic   [KEEP_WIDTH-1:0] sif_mux_axis_tkeep [NUM_SENSOR];
logic   [NUM_SENSOR-1:0] sif_mux_axis_tready;

logic   [NUM_SENSOR-1:0] sif_tx_axis_tvalid;
logic   [AXI_DWIDTH-1:0] sif_tx_axis_tdata [NUM_SENSOR];
logic   [           1:0] sif_tx_axis_tuser [NUM_SENSOR];
logic   [KEEP_WIDTH-1:0] sif_tx_axis_tkeep [NUM_SENSOR];
logic   [NUM_SENSOR-1:0] sif_tx_axis_tlast;
logic   [NUM_SENSOR-1:0] sif_tx_axis_tready;

logic   [SIF_DWIDTH-1:0] w_sif_tx_axis_tdata [NUM_SENSOR];
logic   [W_SIF_KEEP-1:0] w_sif_tx_axis_tkeep [NUM_SENSOR];

logic [TOTAL_TX-1:0] will_rec [TOTAL_TX-1:0];

logic [  NUM_HOST-1:0]  hif_req;
logic [NUM_SENSOR-1:0]  sif_req;

logic [  NUM_HOST-1:0]  hif_gnt;
logic [NUM_SENSOR-1:0]  sif_gnt;
logic [NUM_SENSOR-1:0]  r_sif_gnt [IS_PTP_DEL:0];


logic [   NUM_HOST-1:0] hif_tx_busy;
logic [ NUM_SENSOR-1:0] sif_tx_busy;
logic [   TOTAL_TX-1:0] tx_eop;
logic [  NUM_HOST-1:0]  self_mask [NUM_HOST];

rrarb #(
  .WIDTH(NUM_HOST)
) u_rrarb_host (
  .clk    ( i_pclk       ),    // Clock
  .rst_n  ( !i_prst      ),    // Asynchronous reset active low
  .rst    ( 1'b0         ),    // Synchronous reset active high
  .idle   ( '1           ),    // Only allow new grants when idle. Tie to 1 to grant new req at any time.
  .req    ( hif_req      ),    // vector of requests
  .gnt    ( hif_gnt      )     // onehot0 vector of grants
);

generate
  for (i=0;i<NUM_HOST;i=i+1) begin : gen_hif_req
    assign will_rec[i+NUM_SENSOR][0+:NUM_SENSOR]        = dest_port[i][0+:NUM_SENSOR];
    assign self_mask[i]                                 = '0 | (1'b1 << i); // Mask to not loopback to self
    assign will_rec[i+NUM_SENSOR][NUM_SENSOR+:NUM_HOST] = (dest_port[i][NUM_SENSOR+:NUM_HOST] & ctrl_reg[(i*2)+1][0+:NUM_HOST] & ~self_mask[i]);               // TX path will receive if the dest port is enabled and it is subscribed, and not itself
    assign hif_req[i]                                   = clear_to_send[i+NUM_SENSOR] && (&(~(will_rec[i+NUM_SENSOR] & ({hif_tx_busy,sif_tx_busy} | {hif_tx_buffer_afull,sif_tx_buffer_afull}))));     // RX path can send if every path that will be sent is not busy
  end
endgenerate

rrarb #(
  .WIDTH(NUM_SENSOR)
) u_rrarb_sensor (
  .clk    ( i_pclk    ),    // Clock
  .rst_n  ( !i_prst   ),    // Asynchronous reset active low
  .rst    ( 1'b0      ),    // Synchronous reset active high
  .idle   ( '1        ),    // Only allow new grants when idle. Tie to 1 to grant new req at any time.
  .req    ( sif_req   ),    // vector of requests
  .gnt    ( sif_gnt   )     // onehot0 vector of grants
);

always_comb begin
  gnt_idx = '0;
  for (int i = 0; i < NUM_SENSOR; i++) begin
    if (sif_gnt[i]) begin
      gnt_idx = i;
      sif_gnt_idx = i;
    end
  end
  for (int i = 0; i < NUM_HOST; i++) begin
    if (hif_gnt[i]) begin
      gnt_idx = i+NUM_SENSOR;
    end
  end
end

generate
  for (i=0;i<NUM_SENSOR;i=i+1) begin : gen_sif_chk_ptp
    assign is_ptp[i] = is_ptp_pkt[i];
    assign is_ptp_mac[i] = i_sif_axis_tvalid[i] && (i_sif_axis_tdata[i][47:0] == PTP_FW_MULTI_ADDR);
    always_ff @(posedge i_pclk) begin
      if (i_prst) begin
        is_ptp_pkt[i] <= 1'b0;
      end
      else begin
        if (i_sif_axis_tvalid[i] && i_sif_axis_tlast[i] && o_sif_axis_tready[i]) begin
          is_ptp_pkt[i] <= 1'b0;
        end
        else if (is_ptp_mac[i]) begin
          is_ptp_pkt[i] <= 1'b1;
        end
      end
    end
  end
endgenerate

generate
  for (i=0;i<NUM_SENSOR;i=i+1) begin : gen_sif_req
    assign will_rec[i][0+:NUM_SENSOR]  = '0;
    for (j=0;j<NUM_HOST;j=j+1) begin                                  // No sensor RX to sensor TX path
      assign will_rec[i][NUM_SENSOR+j] = (ctrl_reg[0][j] && is_ptp[i]) ||            // TX path will receive if it is ptp and host is broadcast port
                                         (ctrl_reg[(j*2)+2][i] && ~is_ptp[i]);       // TX path will receive if is subscribed and not ptp
    end
    // RX path can send if every path either will not receive, or it is not busy
    assign sif_req[i]                  = clear_to_send[i] && (&(~(will_rec[i] & ({hif_tx_busy,sif_tx_busy} | {hif_tx_buffer_afull,sif_tx_buffer_afull}))));
  end
endgenerate

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    sif_tx_busy   <= '0;
    hif_tx_busy   <= '0;
    r_sif_gnt_idx <= '{default:0};
    r_sif_gnt     <= '{default:0};
  end
  else begin
    if (is_gnt) begin
      {hif_tx_busy,sif_tx_busy} <= ((({hif_tx_busy,sif_tx_busy}) & ~tx_eop) | will_rec[gnt_idx]); // tx_eop should never collide with will_rec
    end
    else begin
      {hif_tx_busy,sif_tx_busy} <= (({hif_tx_busy,sif_tx_busy}) & ~tx_eop);
    end
    r_sif_gnt_idx     <= {r_sif_gnt_idx[IS_PTP_DEL-1:0],sif_gnt_idx};
    r_sif_gnt         <= {r_sif_gnt[IS_PTP_DEL-1:0],sif_gnt};
  end
end

assign tx_eop = {hif_mux_axis_tlast,sif_mux_axis_tlast} & {hif_mux_axis_tvalid,sif_mux_axis_tvalid};
assign is_gnt = (hif_gnt || sif_gnt);


//------------------------------------------------------------------------------------------------//
// DEMUX: HIF and SIF
//------------------------------------------------------------------------------------------------//

logic [HIF_MUX_WIDTH:0] hif_mux_sel [NUM_HOST];
logic [SIF_MUX_WIDTH:0] sif_mux_sel [NUM_SENSOR];

logic   [  TOTAL_TX-1:0] hif_demux_axis_tvalid;
logic   [AXI_DWIDTH-1:0] hif_demux_axis_tdata [TOTAL_TX-1:0];
logic   [  TOTAL_TX-1:0] hif_demux_axis_tlast;
logic   [           1:0] hif_demux_axis_tuser [TOTAL_TX-1:0];
logic   [KEEP_WIDTH-1:0] hif_demux_axis_tkeep [TOTAL_TX-1:0];
logic   [  TOTAL_TX-1:0] hif_demux_axis_tready;

logic   [  TOTAL_TX-1:0] hsif_demux_axis_tvalid;
logic   [AXI_DWIDTH-1:0] hsif_demux_axis_tdata [TOTAL_TX-1:0];
logic   [  TOTAL_TX-1:0] hsif_demux_axis_tlast;
logic   [           1:0] hsif_demux_axis_tuser [TOTAL_TX-1:0];
logic   [KEEP_WIDTH-1:0] hsif_demux_axis_tkeep [TOTAL_TX-1:0];
logic   [  TOTAL_TX-1:0] hsif_demux_axis_tready;


generate
  for (i=0;i<NUM_SENSOR;i=i+1) begin : gen_sif_mux_sel
    always_ff @(posedge i_pclk) begin
      if (i_prst) begin
        sif_mux_sel[i] <= '0;
      end
      else begin
        if (is_gnt && will_rec[gnt_idx][i]) begin
          sif_mux_sel[i] <= gnt_idx;
        end
      end
    end
  end
endgenerate

generate
  for (i=0;i<NUM_HOST;i=i+1) begin : gen_hif_mux_sel
    always_ff @(posedge i_pclk) begin
      if (i_prst) begin
        hif_mux_sel[i] <= '0;
      end
      else begin
        if (is_gnt && will_rec[gnt_idx][i+NUM_SENSOR]) begin
          hif_mux_sel[i] <= gnt_idx;
        end
      end
    end
  end
endgenerate

assign hsif_demux_axis_tvalid = {hif_buf_axis_tvalid,sif_buf_axis_tvalid};
assign hsif_demux_axis_tdata  = {hif_buf_axis_tdata ,sif_buf_axis_tdata };
assign hsif_demux_axis_tlast  = {hif_buf_axis_tlast ,sif_buf_axis_tlast };
assign hsif_demux_axis_tuser  = {hif_buf_axis_tuser ,sif_buf_axis_tuser };
assign hsif_demux_axis_tkeep  = {hif_buf_axis_tkeep ,sif_buf_axis_tkeep };

assign hif_demux_axis_tvalid [NUM_SENSOR+:NUM_HOST] = hif_buf_axis_tvalid;
assign hif_demux_axis_tdata  [NUM_SENSOR+:NUM_HOST] = hif_buf_axis_tdata ;
assign hif_demux_axis_tlast  [NUM_SENSOR+:NUM_HOST] = hif_buf_axis_tlast ;
assign hif_demux_axis_tuser  [NUM_SENSOR+:NUM_HOST] = hif_buf_axis_tuser ;
assign hif_demux_axis_tkeep  [NUM_SENSOR+:NUM_HOST] = hif_buf_axis_tkeep ;

assign hif_demux_axis_tvalid [0+:NUM_SENSOR]        = '{default:0};
assign hif_demux_axis_tdata  [0+:NUM_SENSOR]        = '{default:0};
assign hif_demux_axis_tlast  [0+:NUM_SENSOR]        = '{default:0};
assign hif_demux_axis_tuser  [0+:NUM_SENSOR]        = '{default:0};
assign hif_demux_axis_tkeep  [0+:NUM_SENSOR]        = '{default:0};

//------------------------------------------------------------------------------------------------//
// HIF TX
//------------------------------------------------------------------------------------------------//

localparam HOST_BUF_DEPTH = HOST_MTU * 2 / (AXI_DWIDTH / 8);


generate
  for (i=0;i<NUM_HOST;i=i+1) begin : gen_hif_mux
    assign hif_mux_axis_tvalid[i] = hsif_demux_axis_tvalid[hif_mux_sel[i]] && hif_tx_busy[i];
    assign hif_mux_axis_tdata [i] = hsif_demux_axis_tdata [hif_mux_sel[i]];
    assign hif_mux_axis_tlast [i] = hsif_demux_axis_tlast [hif_mux_sel[i]];
    assign hif_mux_axis_tuser [i] = '0;
    assign hif_mux_axis_tkeep [i] = hsif_demux_axis_tkeep [hif_mux_sel[i]];

    axis_buffer # (
      .IN_DWIDTH         ( AXI_DWIDTH         ),
      .OUT_DWIDTH        ( AXI_DWIDTH         ),
      .WAIT2SEND         ( 1                  ),
      .BUF_DEPTH         ( HOST_BUF_DEPTH     ),
      .ALMOST_FULL_DEPTH ( HOST_BUF_DEPTH / 2 ), // Can fit at least 1 full packet
      .NO_BACKPRESSURE   ( 1                  )
    ) u_axis_hif_tx_buffer (
      .in_clk            ( i_pclk                  ),
      .in_rst            ( i_prst                  ),
      .out_clk           ( i_pclk                  ),
      .out_rst           ( i_prst                  ),
      .i_axis_rx_tvalid  ( hif_mux_axis_tvalid [i] ),
      .i_axis_rx_tdata   ( hif_mux_axis_tdata  [i] ),
      .i_axis_rx_tlast   ( hif_mux_axis_tlast  [i] ),
      .i_axis_rx_tuser   ( hif_mux_axis_tuser  [i] ),
      .i_axis_rx_tkeep   ( hif_mux_axis_tkeep  [i] ),
      .o_axis_rx_tready  ( hif_mux_axis_tready [i] ),
      .o_fifo_aempty     (                         ),
      .o_fifo_afull      ( hif_tx_buffer_afull [i] ),
      .o_fifo_empty      (                         ),
      .o_fifo_full       (                         ),
      .o_axis_tx_tvalid  ( o_hif_axis_tvalid   [i] ),
      .o_axis_tx_tdata   ( o_hif_axis_tdata    [i] ),
      .o_axis_tx_tlast   ( o_hif_axis_tlast    [i] ),
      .o_axis_tx_tuser   ( o_hif_axis_tuser    [i] ),
      .o_axis_tx_tkeep   ( o_hif_axis_tkeep    [i] ),
      .i_axis_tx_tready  ( i_hif_axis_tready   [i] )
    );

  end
endgenerate



//------------------------------------------------------------------------------------------------//
// SIF TX
//------------------------------------------------------------------------------------------------//

// No backpressure FIFO, i_sif_axis_tready cannot go low

logic [223:0]          ptp_mask [NUM_SENSOR];
logic [63:0]           sif_correction_val [NUM_SENSOR];
logic [47:0]           sif_cf             [NUM_SENSOR];
logic [47:0]           sif_cf_be          [NUM_SENSOR];
logic [W_RES-1:0]      sif_res_time [NUM_SENSOR];
logic [NUM_SENSOR-1:0] sif_res_latch;
logic [NUM_SENSOR-1:0] del_req_val;
logic                  del_req_dval;
logic                  wr_ptp_fifo;
logic                  rd_ptp_fifo;
logic [NUM_SENSOR-1:0] drop_sif_pkt;
logic [NUM_SENSOR-1:0] w_sif_axis_tready;

`ifdef SIMULATION
  logic [11:0] sif_tready_cnt [NUM_SENSOR];
`else
  logic [15:0] sif_tready_cnt [NUM_SENSOR];
`endif


generate
  for (i=0;i<NUM_SENSOR;i=i+1) begin : gen_sif_mux
    assign sif_mux_axis_tvalid[i] = hif_demux_axis_tvalid[sif_mux_sel[i]] && sif_tx_busy[i];
    assign sif_mux_axis_tdata [i] = hif_demux_axis_tdata [sif_mux_sel[i]];
    assign sif_mux_axis_tlast [i] = hif_demux_axis_tlast [sif_mux_sel[i]] && sif_tx_busy[i];
    assign sif_mux_axis_tuser [i] = hif_demux_axis_tuser [sif_mux_sel[i]];
    assign sif_mux_axis_tkeep [i] = hif_demux_axis_tkeep [sif_mux_sel[i]];
  // Direct connection, no FIFO

    axis_mask # (
      .MASK_WIDTH         ( 224                    ),
      .DWIDTH             ( AXI_DWIDTH             ),
      .OP                 ( "OR"                   )
    ) axis_ptp_mask (
      .clk                ( i_pclk                 ),
      .rst                ( i_prst                 ),
      .i_mask             ( ptp_mask           [i] ),
      .i_mask_en          ( |sif_mux_axis_tuser[i] ),
      .i_axis_rx_tvalid   ( sif_mux_axis_tvalid[i] ),
      .i_axis_rx_tdata    ( sif_mux_axis_tdata [i] ),
      .i_axis_rx_tlast    ( sif_mux_axis_tlast [i] ),
      .i_axis_rx_tuser    ( '0                     ),
      .i_axis_rx_tkeep    ( sif_mux_axis_tkeep [i] ),
      .o_axis_rx_tready   (                        ),
      .o_axis_tx_tvalid   ( sif_tx_axis_tvalid [i] ),
      .o_axis_tx_tdata    ( sif_tx_axis_tdata  [i] ),
      .o_axis_tx_tlast    ( sif_tx_axis_tlast  [i] ),
      .o_axis_tx_tuser    ( sif_tx_axis_tuser  [i] ),
      .o_axis_tx_tkeep    ( sif_tx_axis_tkeep  [i] ),
      .i_axis_tx_tready   ( '1                     )
    );

    axis_buffer # (
      .IN_DWIDTH         ( AXI_DWIDTH                                ),
      .OUT_DWIDTH        ( W_SIF_TX[i]                               ),
      .WAIT2SEND         ( 1                                         ),
      .BUF_DEPTH         ( HOST_BUF_DEPTH                            ),
      .ALMOST_FULL_DEPTH ( HOST_BUF_DEPTH / 2                        ), // Can fit at least 1 full packet
      .NO_BACKPRESSURE   ( 1                                         ),
      .DUAL_CLOCK        ( 1                                         )
    ) u_axis_sif_tx_buffer (
      .in_clk            ( i_pclk                                    ),
      .in_rst            ( i_prst                                    ),
      .out_clk           ( i_sif_clk           [i]                   ),
      .out_rst           ( i_sif_rst           [i]                   ),
      .i_axis_rx_tvalid  ( sif_tx_axis_tvalid  [i]                   ),
      .i_axis_rx_tdata   ( sif_tx_axis_tdata   [i]                   ),
      .i_axis_rx_tlast   ( sif_tx_axis_tlast   [i]                   ),
      .i_axis_rx_tuser   ( sif_tx_axis_tuser   [i]                   ),
      .i_axis_rx_tkeep   ( sif_tx_axis_tkeep   [i]                   ),
      .o_axis_rx_tready  ( sif_tx_axis_tready  [i]                   ),
      .o_fifo_aempty     (                                           ),
      .o_fifo_afull      ( sif_tx_buffer_afull  [i]                  ),
      .o_fifo_empty      (                                           ),
      .o_fifo_full       (                                           ),
      .o_axis_tx_tvalid  ( o_sif_axis_tvalid   [i]                   ),
      .o_axis_tx_tdata   ( w_sif_tx_axis_tdata [i][0+:W_SIF_TX[i]]   ),
      .o_axis_tx_tlast   ( o_sif_axis_tlast    [i]                   ),
      .o_axis_tx_tuser   ( o_sif_axis_tuser    [i]                   ),
      .o_axis_tx_tkeep   ( w_sif_tx_axis_tkeep [i][0+:W_SIF_TX[i]/8] ),
      .i_axis_tx_tready  ( w_sif_axis_tready   [i]                    )
    );

    assign o_sif_axis_tdata [i] = {'0,w_sif_tx_axis_tdata [i][0+:W_SIF_TX[i]]};
    assign o_sif_axis_tkeep [i] = {'0,w_sif_tx_axis_tkeep [i][0+:W_SIF_TX[i]/8]};
    assign w_sif_axis_tready[i] = drop_sif_pkt[i] ? '1 : i_sif_axis_tready[i];

    always_ff @(posedge i_sif_clk[i]) begin
      if (i_sif_rst[i]) begin
        sif_tready_cnt[i] <= '1;
        drop_sif_pkt[i]   <= '1;
      end
      else begin
        drop_sif_pkt[i] <= (sif_tready_cnt[i] == '1);
        if (!i_sif_axis_tready[i] && o_sif_axis_tvalid[i]) begin
          if (sif_tready_cnt[i] == '1) begin
            sif_tready_cnt[i] <= '1;
          end
          else begin
            sif_tready_cnt[i] <= sif_tready_cnt[i] + 1;
          end
        end
        else begin
          sif_tready_cnt[i] <= '0;
        end
      end
    end

    always_ff @(posedge i_pclk) begin
      if (i_prst) begin
        sif_res_time[i]       <= '0;
        sif_res_latch[i]      <= '1;
        sif_correction_val[i] <= '0;
      end
      else begin
        sif_res_time[i] <= i_chk_del_req[i] && !i_is_del_req[i] ? '0                    :
                            is_ptp[i] && sif_res_latch[i]        ? '0                    : // Reset timer
                            sif_res_latch[i]                     ? sif_res_time[i]       : // Stop timer
                                                                  sif_res_time[i] + 1'b1;

        sif_res_latch[i] <= i_chk_del_req[i] && !i_is_del_req[i] ? 1'b1         :
                            del_req_val [i] && del_req_dval      ? 1'b1         : // Stop Timer
                            is_ptp[i]                            ? 1'b0         : // Start Timer
                                                                sif_res_latch[i];
        sif_correction_val[i] <= sif_res_time[i] * default_inc; // Value is 1 cycle delayed
      end
    end

    assign sif_cf[i] = {'0,sif_correction_val[i][W_FRAC_NS+:W_NS_RES]};

    for (j=0; j<6; j++) begin
      assign sif_cf_be[i][j*8+:8] = sif_cf[i][(6-1-j)*8+:8];
    end

    assign ptp_mask[i][175:0]   = '0; // PTP Header
    assign ptp_mask[i][223:176] = (sif_mux_axis_tuser[i] == 'd2) ? hif_cf_be    :  // Is Follow Up
                                  (sif_mux_axis_tuser[i] == 'd3) ? sif_cf_be[i] :  // Is Delay Response
                                                                   '0;

  end
endgenerate


sc_fifo #(
  .DATA_WIDTH ( NUM_SENSOR    ),
  .FIFO_DEPTH ( NUM_SENSOR*2  )
) ptp_del_req_buffer (
  .clk      ( i_pclk                ),
  .rst      ( i_prst                ),
  .wr       ( wr_ptp_fifo           ),
  .din      ( r_sif_gnt[IS_PTP_DEL] ),
  .full     (                       ),
  .afull    (                       ),
  .over     (                       ),
  .rd       ( rd_ptp_fifo           ),
  .dout     ( del_req_val           ),
  .dval     ( del_req_dval          ),
  .empty    (                       ),
  .aempty   (                       ),
  .under    (                       ),
  .count    (                       )
);

assign wr_ptp_fifo = (|r_sif_gnt[IS_PTP_DEL] && i_is_del_req[r_sif_gnt_idx[IS_PTP_DEL]]);
assign rd_ptp_fifo = i_del_req_val[ptp_port];


endmodule
