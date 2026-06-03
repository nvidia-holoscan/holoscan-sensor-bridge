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

module packetizer_top
  import regmap_pkg::*;
  import apb_pkg::*;
# (
// Bajoran Lite
    parameter  DIN_WIDTH         = 64,
    parameter  SORT_ENABLE       = 1,
    parameter  SORT_RESOLUTION   = 2,
    parameter  PIPE_DEPTH        = 3,
    parameter  VP_COUNT          = 1,
    parameter  VP_SIZE           = 32,
    parameter  DOUT_WIDTH        = 64,
    parameter  NUM_CYCLES        = 3,
    parameter  DYNAMIC_VP        = 1,
    parameter  MIXED_VP_SIZE     = 1,
    parameter  MTU               = 1500,
    parameter  PACKETIZER_ENABLE = 0,
    parameter  W_USER            = 1,
    localparam DKEEP_WIDTH       = DIN_WIDTH/8,
    localparam DOUT_KWIDTH       = DOUT_WIDTH/8
    ) (
    input                       i_pclk,
    input                       i_prst,
    // Sensor Clock
    input                       i_sclk,
    input                       i_srst,
    output  [VP_COUNT-1:0]      o_sof,
        // Register Map, abp clk domain
    input                       i_aclk,
    input                       i_arst,
    input  apb_m2s              i_apb_m2s,
    output apb_s2m              o_apb_s2m,
    // Incoming Sensor Data
    input                       i_axis_tvalid,
    input                       i_axis_tlast,
    input   [DIN_WIDTH-1:0]     i_axis_tdata,
    input   [DKEEP_WIDTH-1:0]   i_axis_tkeep,
    input   [W_USER-1:0]        i_axis_tuser,
    output                      o_axis_tready,
    // Data Output
    output  [VP_COUNT-1:0]      o_axis_tvalid,
    output  [VP_COUNT-1:0]      o_axis_tlast,
    output  [DOUT_WIDTH-1:0]    o_axis_tdata [VP_COUNT-1:0],
    output  [DOUT_KWIDTH-1:0]   o_axis_tkeep [VP_COUNT-1:0],
    output  [VP_COUNT-1:0]      o_axis_tuser,
    input   [VP_COUNT-1:0]      i_axis_tready
    );

localparam PACK_LATENCY = 4'hF;
localparam PAD_CYCLES   = (DKEEP_WIDTH >= 64) ? 1 : (64 / DKEEP_WIDTH);
localparam NO_PAD       = (PAD_CYCLES == 1);
localparam PAD_WIDTH    = (PAD_CYCLES > 1) ? ($clog2(PAD_CYCLES)-1) : 0;
localparam CNT_WIDTH    = (PAD_WIDTH > 3) ? PAD_WIDTH : 3;

//------------------------------------------------------------------------------------------------//
// Register Map
//------------------------------------------------------------------------------------------------//

logic [15:0] sif_tvalid_cnt;
logic [15:0] sif_psn;

logic [31:0] ctrl_reg [pack_nctrl];
logic [31:0] stat_reg [pack_nstat];

assign stat_reg[pack_tvalid_cnt-stat_ofst] = {'0,sif_tvalid_cnt};
assign stat_reg[pack_psn_cnt-stat_ofst] = {'0,sif_psn};

s_apb_reg #(
  .N_CTRL    ( pack_nctrl     ),
  .N_STAT    ( pack_nstat     ),
  .W_OFST    ( w_ofst         )
) u_reg_map  (
  // APB Interface
  .i_aclk    ( i_aclk         ), // Slow Clock
  .i_arst    ( i_arst         ),
  .i_apb_m2s ( i_apb_m2s      ),
  .o_apb_s2m ( o_apb_s2m      ),
  // User Control Signals
  .i_pclk    ( i_sclk         ), // Fast Clock
  .i_prst    ( i_srst         ),
  .o_ctrl    ( ctrl_reg       ),
  .i_stat    ( stat_reg       )
);

logic [15:0] wr_addr;
logic [15:0] wr_addr_q;
logic [31:0] wr_data;
logic [15:0] ld_arr;
logic [3:0]  max_addr;
logic        pad_enable;
logic [3:0]  latency;

logic [CNT_WIDTH:0]  cnt;
logic [CNT_WIDTH:0]  cnt_nxt;
logic [2:0]          latency_cnt;

logic        bypass_packetizer;
logic        wr_en;
logic        wr_en_sync;
logic        wr_en_sync_q;
logic        wr_en_fall;
logic        pack_tready;
logic        duplicate;


assign wr_addr           = ctrl_reg[pack_ram_addr][15:0];
assign wr_data           = ctrl_reg[pack_ram_data][31:0];
assign ld_arr            = ctrl_reg[pack_ctrl][15:0];
assign pad_enable        = ctrl_reg[pack_ctrl][16];
assign duplicate         = ctrl_reg[pack_ctrl][17];
assign max_addr          = ctrl_reg[pack_ctrl][23:20];
assign latency           = ctrl_reg[pack_ctrl][27:24];
assign bypass_packetizer = !ctrl_reg[pack_ctrl][28];
// Successful Write to wr_data
assign wr_en   = i_apb_m2s.psel & i_apb_m2s.penable & i_apb_m2s.pwrite & (i_apb_m2s.paddr[7:0] == 8'h08) && (o_apb_s2m.pready);

data_sync #(
  .DATA_WIDTH     (1             ),
  .RESET_VALUE    (1'b0          ),
  .SYNC_DEPTH     (3             )
) u_wr_en_sync (
  .clk            ( i_sclk       ),
  .rst_n          ( !i_srst      ),
  .sync_in        ( wr_en        ),
  .sync_out       ( wr_en_sync   )
);

always_ff @(posedge i_sclk) begin
  if (i_srst) begin
    wr_en_sync_q <= '0;
    wr_addr_q    <= '0;
  end
  else begin
    wr_en_sync_q <= wr_en_sync;
    wr_addr_q    <= wr_addr;
  end
end

assign wr_en_fall = ({wr_en_sync,wr_en_sync_q} == 2'b01);

typedef enum logic [6:0] {
  PACK_IDLE     = 7'b0000001,
  PACK_WAIT     = 7'b0000010,
  PACK_RUN      = 7'b0000100,
  PACK_PATTERN  = 7'b0001000,
  PACK_PAD      = 7'b0010000,
  PACK_WAIT2    = 7'b0100000,
  PACK_DELAY    = 7'b1000000
} pack_fsm;

pack_fsm state, state_nxt, state_prev;

//------------------------------------------------------------------------------------------------//
// Timestamp
//------------------------------------------------------------------------------------------------//

logic        pkt_active;
logic        sif_sof;
logic [1:0]  last_buf; // Only 2 tlast can be buffered at a time
logic        last_tready;
logic        in_is_last;
logic        sif_sof_sync;
logic        out_is_last;
logic        out_is_last_sync;

always_ff @(posedge i_sclk) begin
  if (i_srst) begin
    pkt_active     <= '0;
    sif_tvalid_cnt <= '0;
    sif_psn        <= '0;
    last_buf       <= '0;
    last_tready    <= '0;
  end
  else begin
    last_buf       <= last_buf + in_is_last - out_is_last_sync;
    pkt_active     <= (i_axis_tvalid && !pkt_active && o_axis_tready) ? !(i_axis_tlast && o_axis_tready) :
                      (i_axis_tvalid && i_axis_tlast && o_axis_tready) ? '0:
                      pkt_active;
    sif_tvalid_cnt <= (i_axis_tvalid) ? sif_tvalid_cnt + 1'b1 : sif_tvalid_cnt;
    sif_psn        <= (in_is_last) ? sif_psn + 1'b1 : sif_psn;
    last_tready    <= !((last_buf == 2'b10) || ((last_buf == 2'b01) && in_is_last));
  end
end

assign in_is_last = (i_axis_tvalid && i_axis_tlast && o_axis_tready);
assign out_is_last = (o_axis_tvalid[0] && o_axis_tlast[0] && i_axis_tready[0]);

pulse_sync u_sof_sync (
  .src_clk     ( i_sclk          ),
  .src_rst     ( i_srst          ),
  .dst_clk     ( i_pclk          ),
  .dst_rst     ( i_prst          ),
  .i_src_pulse ( sif_sof         ),
  .o_dst_pulse ( sif_sof_sync    )
);

pulse_sync u_eof_sync (
  .src_clk     ( i_pclk          ),
  .src_rst     ( i_prst          ),
  .dst_clk     ( i_sclk          ),
  .dst_rst     ( i_srst          ),
  .i_src_pulse ( out_is_last     ),
  .o_dst_pulse ( out_is_last_sync)
);

assign sif_sof = (i_axis_tvalid && !pkt_active && o_axis_tready);

genvar i;
generate
  for (i=0;i<VP_COUNT;i=i+1) begin: gen_sof
    assign o_sof[i] = sif_sof_sync;
  end
endgenerate

localparam W_PCK_USER = (W_USER <= 1) ? 2 : W_USER;

logic [W_PCK_USER-1:0] buf_tuser;
logic                  buf_bypass;
logic [W_PCK_USER-1:0] inp_tuser;


assign buf_tuser = (inp_tuser | {'0,bypass_packetizer});
assign buf_bypass = buf_tuser[0];
assign inp_tuser = (W_USER<2) ? {1'b0,i_axis_tuser} : i_axis_tuser;

//------------------------------------------------------------------------------------------------//
// Packetizer Pattern Address
//------------------------------------------------------------------------------------------------//

logic                       buf_rd_val;
logic                       buf_rd_req;
logic [DIN_WIDTH-1:0]       pack_din;
logic                       pack_tlast;
logic                       pack_dvalid;
logic [W_USER-1:0]          pack_tuser;
logic                       pack_wait;
logic [3:0]                 ram_rd_addr;
logic [3:0]                 ram_rd_addr_nxt;
logic                       pack_stall;
logic [VP_COUNT-1:0]        w_vp_almost_full;
logic [VP_COUNT-1:0]        w_vp_is_empty;
logic [VP_COUNT-1:0]        vp_is_empty;
logic [VP_COUNT-1:0]        vp_is_empty_sync;
logic                       pad_state;
logic                       ld_nxt;
logic                       r_pack_tready;

always @ (posedge i_sclk) begin
  if (i_srst) begin
    pack_din        <= '0;
    pack_tlast      <= '0;
    pack_stall      <= '0;
    pack_tuser      <= '0;
    state           <= PACK_IDLE;
    state_prev      <= PACK_IDLE;
    cnt             <= '0;
    pack_dvalid     <= '0;
    latency_cnt     <= '0;
    ram_rd_addr     <= '0;
    ld_nxt          <= '0;
    r_pack_tready   <= '0;
  end
  else begin
    pack_din      <= (buf_rd_val) ? i_axis_tdata : '0;
    pack_tuser    <= (buf_rd_val) ? buf_tuser : pack_tuser;
    pack_tlast    <= (buf_rd_val) ? i_axis_tlast : pack_tlast;
    pack_stall    <= (!ld_nxt && (state != PACK_IDLE));
    state         <= state_nxt;
    state_prev    <= state;
    cnt           <= cnt_nxt;
    pack_dvalid   <= (state == PACK_IDLE)      ? buf_rd_val                               :
                      (pad_state)               ? '1                                       :
                      (state == PACK_DELAY)     ? '0                                       :
                                                  (buf_rd_val && !buf_bypass) || (!ld_nxt) ;
    latency_cnt   <= ((state == PACK_WAIT) || (state == PACK_WAIT2)) ? latency_cnt + 1'b1 : '0;
    ram_rd_addr   <= ram_rd_addr_nxt;
    ld_nxt        <= ld_arr[ram_rd_addr_nxt];
    r_pack_tready <= pack_tready;
  end
end

assign buf_rd_val = (i_axis_tvalid && o_axis_tready);


always_comb begin
  state_nxt = state;
  case(state)
    PACK_IDLE: begin
      state_nxt = (!buf_bypass && buf_rd_val) ?
                        (latency == '0) ? (i_axis_tlast && (max_addr == '0)) ? PACK_IDLE : PACK_RUN : // Check for End of Frame
                              PACK_WAIT     :  // Add Latency if needed
                  (buf_rd_val && buf_tuser[1] && !bypass_packetizer && pad_enable) ? // Only Pad to 64 Bytes if packetizer is enabled
                    ((cnt[PAD_WIDTH:0] == (PAD_CYCLES-1)) && !NO_PAD && (!pack_dvalid) ||
                     (cnt[PAD_WIDTH:0] == (PAD_CYCLES-2)) && !NO_PAD && (pack_dvalid)    ) ? PACK_IDLE : PACK_PAD : // Pad if needed
                  (buf_rd_val && i_axis_tlast) ? PACK_IDLE :
                                 PACK_IDLE;
    end
    PACK_WAIT: begin
      state_nxt = (latency_cnt == (latency-1)) ? PACK_RUN : PACK_WAIT;
    end
    PACK_RUN: begin
      state_nxt = (buf_rd_val && (i_axis_tlast || buf_tuser[1]) || ((ram_rd_addr == max_addr) && pack_tlast)) ? // End of Frame or Line
                      (latency != '0)          ? PACK_WAIT2                     :
                      (ram_rd_addr < max_addr) ? PACK_PATTERN                   :
                      (cnt[PAD_WIDTH:0] == (PAD_CYCLES-2) && !NO_PAD) || !pad_enable ? PACK_IDLE  :
                                                                            PACK_PAD    :
                                                  PACK_RUN   ;
    end
    PACK_WAIT2: begin
      state_nxt =  (latency_cnt == (latency-1))     ?
                         (ram_rd_addr < max_addr)                    ? PACK_PATTERN:
                         (cnt[PAD_WIDTH:0] == (PAD_CYCLES-2) && !NO_PAD) || !pad_enable ? PACK_IDLE  :
                                                                                PACK_PAD
                            : PACK_WAIT2;
    end
    PACK_PATTERN: begin
      state_nxt = (ram_rd_addr < max_addr)                    ? PACK_PATTERN:
                  (cnt[PAD_WIDTH:0] == (PAD_CYCLES-2) && !NO_PAD) || !pad_enable ? PACK_IDLE  :
                                                                            PACK_PAD    ;
    end
    PACK_PAD: begin
      state_nxt = (cnt[PAD_WIDTH:0] == (PAD_CYCLES-2) && !NO_PAD) || !pad_enable ? PACK_IDLE : PACK_PAD;
    end
    PACK_DELAY: begin
      state_nxt = ((cnt >= PACK_LATENCY) && (state_prev == PACK_DELAY)) ? PACK_IDLE : PACK_DELAY;
    end
    default: begin
      state_nxt = PACK_IDLE;
    end
  endcase
end

assign pad_state   = ((state == PACK_PATTERN) || (state == PACK_PAD) || (state == PACK_WAIT2));
assign ram_rd_addr_nxt = (state == PACK_IDLE) && (!buf_rd_val || (latency != '0)) ? '0             :
                         (buf_rd_val || !ld_nxt || pad_state)                     ? (ram_rd_addr >= max_addr) ? '0 : ram_rd_addr + 1 :
                                                                                    ram_rd_addr    ;

assign buf_rd_req = (state == PACK_DELAY  ) ? 1'b0              :
                    (pad_state            ) ? 1'b0              :
                    (!r_pack_tready       ) ? 1'b0              :
                    (state == PACK_IDLE   ) ? 1'b1              :
                                             ld_nxt && i_axis_tvalid ;

assign pack_wait = (state == PACK_WAIT);
assign cnt_nxt   =  ((state == PACK_IDLE) && (state_prev != PACK_IDLE))   ? '0:
                    ((state == PACK_DELAY) && (state_prev != PACK_DELAY)) ? '0:
                    (state == PACK_WAIT)                                  ? cnt :
                    ((state == PACK_IDLE) || (state == PACK_RUN) || (state == PACK_PATTERN)) ?
                                    (pack_dvalid) ? (cnt + 1) : cnt :
                                                    (cnt + 1)       ;

//------------------------------------------------------------------------------------------------//
// Packetizer
//------------------------------------------------------------------------------------------------//

logic [15:0]           ram_addr;
genvar j;

localparam SORT_ENABLE_INT = (SORT_RESOLUTION == DIN_WIDTH) ? 0 : SORT_ENABLE;

generate
  if (PACKETIZER_ENABLE) begin: gen_pack_inst
    packetizer # (
      .DIN_WIDTH        ( DIN_WIDTH       ),
      .SORT_ENABLE      ( SORT_ENABLE_INT ),
      .SORT_RESOLUTION  ( SORT_RESOLUTION ),
      .PIPE_DEPTH       ( PIPE_DEPTH      ),
      .VP_COUNT         ( VP_COUNT        ),
      .VP_SIZE          ( VP_SIZE         ),
      .DOUT_WIDTH       ( DOUT_WIDTH      ),
      .NUM_CYCLES       ( NUM_CYCLES      ),
      .DYNAMIC_VP       ( DYNAMIC_VP      ),
      .MIXED_VP_SIZE    ( MIXED_VP_SIZE   ),
      .W_USER           ( W_USER          ),
      .MTU              ( MTU             )
    ) packetizer_inst(
      .i_sclk           ( i_sclk                                 ),
      .i_srst           ( i_srst                                 ),
      .i_pclk           ( i_pclk                                 ),
      .i_prst           ( i_prst                                 ),
      .ram_wr           ( wr_en_fall                             ),
      .ram_addr         ( ram_addr                               ),
      .ram_wr_data      ( wr_data                                ),
      .din              ( pack_din                               ),
      .din_valid        ( pack_dvalid                            ),
      .din_tuser        ( pack_tuser                             ),
      .duplicate        ( duplicate                              ),
      .din_wait         ( pack_wait                              ),
      .din_stall        ( pack_stall                             ),
      .din_tlast        ( pack_tlast  && (state == PACK_IDLE)    ),
      .vp_is_empty      ( w_vp_is_empty                          ),
      .vp_almost_full   ( w_vp_almost_full                       ),
      .o_axis_tvalid    ( o_axis_tvalid                          ),
      .o_axis_tlast     ( o_axis_tlast                           ),
      .o_axis_tdata     ( o_axis_tdata                           ),
      .o_axis_tkeep     ( o_axis_tkeep                           ),
      .o_axis_tuser     ( o_axis_tuser                           ),
      .i_axis_tready    ( i_axis_tready                          )
    );
  end
  else begin: gen_bypass_vp
    logic [VP_COUNT-1:0] dvalid_vp;
    for (j=0;j<VP_COUNT;j=j+1) begin
      virtual_port # (
        .DIN_WIDTH  ( DIN_WIDTH             ),
        .VP_SIZE    ( DIN_WIDTH             ),
        .DOUT_WIDTH ( DOUT_WIDTH            ),
        .CTRL_WIDTH ( 1                     ),
        .DYNAMIC_VP ( 0                     ),
        .MTU        ( MTU                   )
      ) bypass_vp (
        .i_sclk             ( i_sclk                              ),
        .i_srst             ( i_srst                              ),
        .i_pclk             ( i_pclk                              ),
        .i_prst             ( i_prst                              ),
        .din                ( pack_din                            ),
        .din_valid          ( dvalid_vp[j]                        ),
        .din_tlast          ( pack_tlast                          ),
        .ctrl               ( pack_dvalid                         ),
        .is_empty           ( w_vp_is_empty     [j]               ),
        .almost_full        ( w_vp_almost_full  [j]               ),
        .o_axis_tvalid      ( o_axis_tvalid     [j]               ),
        .o_axis_tlast       ( o_axis_tlast      [j]               ),
        .o_axis_tdata       ( o_axis_tdata      [j]               ),
        .o_axis_tkeep       ( o_axis_tkeep      [j]               ),
        .o_axis_tuser       ( o_axis_tuser      [j]               ),
        .i_axis_tready      ( i_axis_tready     [j]               )
      ); /* synthesis syn_keep=1 */
      assign dvalid_vp[j] = pack_dvalid && (j==0 ? '1 : duplicate);
    end
  end
endgenerate

assign ram_addr   = wr_en_fall                    ? wr_addr_q      :
                                                   {'0,ram_rd_addr};
//------------------------------------------------------------------------------------------------//
// Outgoing Data
//------------------------------------------------------------------------------------------------//

assign pack_tready   = !(|w_vp_almost_full);
assign o_axis_tready = last_tready & buf_rd_req;

endmodule
