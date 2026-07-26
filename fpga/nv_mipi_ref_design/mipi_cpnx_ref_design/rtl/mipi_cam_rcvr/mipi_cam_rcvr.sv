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

module mipi_cam_rcvr
  import apb_pkg::*;
#(
  parameter N_CSI_VC = 1,
  parameter W_DATA = 64,
  localparam W_KEEP = W_DATA/8,
  localparam W_CSI_VC = (N_CSI_VC <= 2) ? 1 : $clog2(N_CSI_VC)
)(
    // clock and reset
    input                     i_mipi_sync_clk, //150Mhz
    input                     i_rst_n,
    input                     i_sclk,   //156.25Mhz
    input                     i_srst,

    input                     i_pll_locked,

  // APB Interface
    input                     i_apb_clk,
    input                     i_apb_rst,
    input                     i_apb_sel,
    input                     i_apb_enable,
    input      [31:0]         i_apb_addr,
    input      [31:0]         i_apb_wdata,
    input                     i_apb_write,
    output reg                o_apb_ready,
    output reg [31:0]         o_apb_rdata,
    output reg                o_apb_serr,
    
    // MIPI RX DPHY payload output
    output                    o_axis_tvalid,
    output                    o_axis_tlast,
    output     [W_DATA-1:0]   o_axis_tdata,
    output     [W_KEEP-1:0]   o_axis_tkeep,
    output     [ 1:0]         o_axis_tuser,
    output     [W_CSI_VC-1:0] o_axis_tidx,
    input                     i_axis_tready,
    
    // MIPI CAM Data Input
    inout                     mipi_cam_clk_p_io,
    inout                     mipi_cam_clk_n_io,
    inout  [3:0]              mipi_cam_data_p_io,
    inout  [3:0]              mipi_cam_data_n_io
);

localparam PAD_IDX  = 3;

localparam W_AXIS_USER = W_CSI_VC + 2; // {vc, line_end, is_embedded}


//------------------------------------------------------------------------------
// APB
//------------------------------------------------------------------------------

  apb_m2s s_apb_m2s;
  apb_s2m s_apb_s2m;

  assign s_apb_m2s.psel    = i_apb_sel;
  assign s_apb_m2s.penable = i_apb_enable;
  assign s_apb_m2s.paddr   = i_apb_addr;
  assign s_apb_m2s.pwdata  = i_apb_wdata;
  assign s_apb_m2s.pwrite  = i_apb_write;
  assign o_apb_ready       = s_apb_s2m.pready;
  assign o_apb_rdata       = {24'h0,s_apb_s2m.prdata[7:0]};
  assign o_apb_serr        = s_apb_s2m.pserr; 

//------------------------------------------------------------------------------
// 
//------------------------------------------------------------------------------

  logic [63:0] payload_o;
  logic [63:0] r_payload;
  logic        payload_en_o;
  logic [1:0]  r_payload_en;
  logic [5:0]  dt;
  logic [1:0]  vc;
  logic [1:0]  vcx;
  logic [3:0]  vc_ext;
  logic [W_CSI_VC-1:0] vc_idx;
  logic [15:0] wc;
  logic [7:0]  ecc;
  logic [7:0]  payload_bytevld_o;
  logic [15:0] payload_crc_o;
  logic [15:0] r_payload_crc;
  logic        r_payload_crcvld;
  logic [7:0]  r_payload_byte_vld;
  logic [7:0]  rr_payload_byte_vld;
  logic        sp_en_o;

  logic [63:0] w_data;

  logic [63:0] mipi_axis_tdata;
  logic [7:0]  mipi_axis_tkeep;
  logic [W_AXIS_USER-1:0] mipi_axis_tuser;
  logic        mipi_axis_tvalid;
  logic        mipi_axis_tlast;

  logic [3:0]  line_pad;
  logic [15:0] wc_padded;
  logic [15:0] byte_cnt;
  logic [15:0] pad_bytes;
  logic [3:0]  byte_incr;
  logic [3:0]  byte_incr_r;
  logic [15:0] wc_r;
  logic [5:0]  dt_r;
  logic [W_CSI_VC-1:0] vc_r;
  logic        last_cycle;
  logic        one_pad_cycle;

  logic        is_embedded;
  logic [N_CSI_VC-1:0] is_frame_vc;

  logic        crc_check_w;
  logic        crc_error_w;

assign vc_ext = {vcx,vc};
assign vc_idx = vc_ext[W_CSI_VC-1:0];

//------------------------------------------------------------------------------
// Resets
//------------------------------------------------------------------------------

  logic clk_byte_o;
  logic clk_byte_hs_o;

  logic clk_byte_hs_rst;
  logic mipi_sync_rst_n;
  logic mipi_sync_rst;

  logic payload_crcvld_o;
  logic mipi_axis_tready;
  logic pck_axis_tready;
  logic reg_axis_tready;

  logic [7:0] tkeep_rounded;

reset_sync mipi_u_rst (
    .i_clk     ( i_mipi_sync_clk  ),
    .i_arst_n  ( !i_srst          ),
    .i_srst    ( 1'b0             ),
    .i_locked  ( i_pll_locked     ),
    .o_arst    (                  ),
    .o_arst_n  (                  ),
    .o_srst    ( mipi_sync_rst    ), 
    .o_srst_n  ( mipi_sync_rst_n  ) 
);

reset_sync mipi_clkhs_rst (
    .i_clk     ( clk_byte_hs_o    ),
    .i_arst_n  ( !i_srst          ),
    .i_srst    ( 1'b0             ),
    .i_locked  ( i_pll_locked     ),
    .o_arst    (                  ),
    .o_arst_n  (                  ),
    .o_srst    ( clk_byte_hs_rst  ), 
    .o_srst_n  (                  ) 
);

//------------------------------------------------------------------------------
// soft MIPI CSI2 RX DPHY 4 lanes, 1250Mbps datarate
//------------------------------------------------------------------------------

  logic lmmi_ready;
  logic lmmi_rdata_valid;
  logic lmmi_request;

  assign s_apb_s2m.pready = lmmi_ready && (s_apb_m2s.pwrite ? 1'b1 : lmmi_rdata_valid);
  assign lmmi_request = (s_apb_m2s.psel && s_apb_m2s.penable);
  assign s_apb_s2m.pserr = 1'b0;

    mipi_rx_ip mipi_csi2_rx_inst (
        .lmmi_clk_i            ( i_apb_clk               ),
        .lmmi_resetn_i         ( !i_apb_rst              ),
        .lmmi_wdata_i          ( s_apb_m2s.pwdata[7:0]   ),
        .lmmi_rdata_o          ( s_apb_s2m.prdata[7:0]   ),
        .lmmi_rdata_valid_o    ( lmmi_rdata_valid        ),
        .lmmi_wr_rdn_i         ( s_apb_m2s.pwrite        ),
        .lmmi_offset_i         ( s_apb_m2s.paddr[9:2]    ),
        .lmmi_request_i        ( lmmi_request            ),
        .lmmi_ready_o          ( lmmi_ready              ),
        .pll_lock_i            ( i_pll_locked            ),  
        .sync_clk_i            ( i_mipi_sync_clk         ),    // 60MHz
        .sync_rst_i            ( mipi_sync_rst           ),   
        .clk_byte_o            (                         ),    // Byte Clock
        .clk_byte_hs_o         ( clk_byte_hs_o           ),    // HS Byte Clock
        .clk_byte_fr_i         ( clk_byte_hs_o           ),    // Clk
        .reset_n_i             ( mipi_sync_rst_n         ),    // RESET
        .reset_byte_fr_n_i     ( mipi_sync_rst_n         ),    // RESET
        .clk_n_io              ( mipi_cam_clk_n_io       ),    // MIPI D-PHY CLK Lane
        .clk_p_io              ( mipi_cam_clk_p_io       ),    // MIPI D-PHY CLK Lane
        .d_p_io                ( mipi_cam_data_p_io      ),    // MIPI D-PHY DATA Lane (2)
        .d_n_io                ( mipi_cam_data_n_io      ),    // MIPI D-PHY DATA Lane (2)
        .payload_en_o          ( payload_en_o            ),    // Enable signal of payload
        .payload_o             ( payload_o [31:0]        ),    // Payload
        .dt_o                  ( dt                      ),    // Packet Header
        .vc_o                  ( vc                      ),    // Packet Header
        .wc_o                  ( wc                      ),    // Packet Header
        .ecc_o                 ( ecc                     ),    // Packet Header   
        .tx_rdy_i              ( 1'b1                    ),    // Receiver ready input
        .sp_en_o               ( sp_en_o                 ),    // Packet Parser Output
        .lp_en_o               ( lp_en_o                 ),    // Packet Parser Output
        .lp_av_en_o            (                         ),    // Packet Parser Output
        .lp_d_rx_p_o           (                         ),
        .lp_d_rx_n_o           (                         ),
        .bd_o                  (                         ),
        .hs_sync_o             (                         ),
        .vcx_o                 ( vcx                     ), 
        .payload_bytevld_o     ( payload_bytevld_o[3:0]  ),
        .payload_crc_o         ( payload_crc_o           ),
        .payload_crcvld_o      ( payload_crcvld_o        ),
        .crc_check_o           ( crc_check_w             ),
        .crc_error_o           ( crc_error_w             ),
        .ecc_check_o           ( ecc_check_w             ),
        .ecc_byte_error_o      (                         ),
        .ecc_1bit_error_o      (                         ),
        .ecc_2bit_error_o      ( ecc_2bit_error_w        ),
        .dphy_rxdatawidth_hs_o (                         ),
        .dphy_cfg_num_lanes_o  (                         ),
        .ready_o               ( o_ready                 ) 
    );
  
    assign payload_bytevld_o[7:4] = '0;
    assign payload_o[63:32]       = '0;

  always @(posedge clk_byte_hs_o or posedge clk_byte_hs_rst) begin
    if (clk_byte_hs_rst) begin
      r_payload_en      <= '0;
      r_payload         <= '0;
      r_payload_crc     <= '0;
      r_payload_crcvld  <= '0;
      mipi_axis_tdata   <= '0; 
      mipi_axis_tkeep   <= '0; 
      mipi_axis_tvalid  <= '0;
      mipi_axis_tlast   <= '0; 
      mipi_axis_tuser   <= '0; 
      wc_padded         <= '0;
      byte_cnt          <= '0;
      wc_r              <= '0;
      dt_r              <= '0;
      vc_r              <= '0;
      byte_incr_r       <= '0;
      r_payload_byte_vld  <= '0;
      rr_payload_byte_vld <= '0;
      one_pad_cycle       <= '0;
      is_frame_vc         <= '0;
    end
    else begin
      r_payload_en       <= {r_payload_en[0],payload_en_o};
      r_payload          <= payload_o;
      r_payload_crc      <= (payload_crcvld_o) ? payload_crc_o : r_payload_crc;
      r_payload_crcvld   <= payload_crcvld_o;
      rr_payload_byte_vld <= payload_bytevld_o;
      if (sp_en_o && (dt == 6'h00)) begin
        is_frame_vc[vc_idx] <= 1'b1;
      end
      else if (sp_en_o && (dt == 6'h01)) begin
        is_frame_vc[vc_idx] <= 1'b0;
      end
      if (({payload_en_o,r_payload_en}==3'b100) || (sp_en_o)) begin
        mipi_axis_tvalid <= '0;
        mipi_axis_tlast  <= '0;
        byte_cnt         <= '0;
        wc_r             <= wc;
        dt_r             <= dt;
        vc_r             <= vc_idx;
        byte_incr_r      <= '0;
        r_payload_byte_vld <= payload_bytevld_o;
        //wc_padded                <= wc;
        wc_padded[15:PAD_IDX]    <= wc[15:PAD_IDX] + (|wc[PAD_IDX-1:0]);
        wc_padded[PAD_IDX-1:0]   <= '0;
      end
      else if (r_payload_en[0]) begin
        mipi_axis_tdata    <= w_data;
        mipi_axis_tkeep    <= tkeep_rounded;
        mipi_axis_tvalid   <= is_frame_vc[vc_r];
        mipi_axis_tlast    <= last_cycle;
        mipi_axis_tuser[0] <= (sp_en_o) ? mipi_axis_tuser[0] : is_embedded; // Only latch data type on LP
        byte_cnt           <= byte_cnt + byte_incr_r;
        byte_incr_r        <= (byte_incr_r == '0) ? byte_incr : byte_incr_r;
        mipi_axis_tuser[1] <= last_cycle;
        mipi_axis_tuser[W_AXIS_USER-1:2] <= vc_r;
        one_pad_cycle      <= ((wc_padded - wc) < (byte_incr_r));
      end
      else if (((byte_cnt+byte_incr_r) >= wc) && (byte_incr_r != '0) && (!last_cycle||!one_pad_cycle)) begin // Pad line to 64 Bytes
        mipi_axis_tdata    <= '0;
        mipi_axis_tvalid   <= is_frame_vc[vc_r]; // Line Padding
        mipi_axis_tlast    <= last_cycle;
        byte_cnt           <= byte_cnt + byte_incr_r;
        mipi_axis_tuser[1] <= last_cycle;
        mipi_axis_tuser[W_AXIS_USER-1:2] <= vc_r;
        if (last_cycle) begin
          byte_incr_r      <= '0;
        end
      end
      else begin 
        mipi_axis_tvalid   <= '0;
        byte_cnt           <= '0;
        byte_incr_r        <= '0;
        mipi_axis_tuser[1] <= '0;
      end
    end
  end

assign byte_incr = r_payload_byte_vld[7] ? 8'd8:
                   r_payload_byte_vld[6] ? 8'd8:
                   r_payload_byte_vld[5] ? 8'd8:
                   r_payload_byte_vld[4] ? 8'd8:
                   r_payload_byte_vld[3] ? 8'd4:
                   r_payload_byte_vld[2] ? 8'd4:
                   r_payload_byte_vld[1] ? 8'd2:
                   r_payload_byte_vld[0] ? 8'd1:
                                           8'd0;

assign tkeep_rounded =  r_payload_byte_vld[7] ? 8'hFF:
                        r_payload_byte_vld[6] ? 8'hFF:
                        r_payload_byte_vld[5] ? 8'hFF:
                        r_payload_byte_vld[4] ? 8'hFF:
                        r_payload_byte_vld[3] ? 8'h0F:
                        r_payload_byte_vld[2] ? 8'h0F:
                        r_payload_byte_vld[1] ? 8'h03:
                        r_payload_byte_vld[0] ? 8'h01:
                                                8'h00;
                                                
assign last_cycle = !((byte_cnt+(byte_incr_r<<1)) < wc_padded);
assign is_embedded = (dt_r == 6'h12);

// Clear out invalid bytes in the mipi payload
genvar j;
generate
    for (j=0;j<8;j=j+1) begin
        assign w_data[j*8+:8] = (rr_payload_byte_vld[j]) ? r_payload[j*8+:8] : '0;
    end
endgenerate

//------------------------------------------------------------------------------
// AXIS Pack
//------------------------------------------------------------------------------

  logic [31:0] pck_axis_tdata;
  logic [3:0]  pck_axis_tkeep;
  logic [W_AXIS_USER-1:0] pck_axis_tuser;
  logic        pck_axis_tvalid;
  logic        pck_axis_tlast;
  
  logic [31:0] reg_axis_tdata;
  logic [3:0]  reg_axis_tkeep;
  logic [W_AXIS_USER-1:0] reg_axis_tuser;
  logic        reg_axis_tvalid;
  logic        reg_axis_tlast;

  axis_packer # (
    .DWIDTH           ( 32                     ),
    .W_USER           ( W_AXIS_USER            )
  ) u_axis_pack  (
    .clk              ( clk_byte_hs_o          ),
    .rst              ( clk_byte_hs_rst        ),
    .i_axis_tvalid    ( mipi_axis_tvalid       ),
    .i_axis_tdata     ( mipi_axis_tdata [31:0] ),
    .i_axis_tlast     ( mipi_axis_tlast        ),
    .i_axis_tuser     ( mipi_axis_tuser        ),
    .i_axis_tkeep     ( mipi_axis_tkeep [3:0]  ),
    .o_axis_tready    ( mipi_axis_tready       ),
    .o_axis_tvalid    ( pck_axis_tvalid        ),
    .o_axis_tdata     ( pck_axis_tdata         ),
    .o_axis_tlast     ( pck_axis_tlast         ),
    .o_axis_tuser     ( pck_axis_tuser         ),
    .o_axis_tkeep     ( pck_axis_tkeep         ),
    .i_axis_tready    ( pck_axis_tready        )
);


  axis_reg # (
    .DWIDTH             ( 32 + 4 + 1 + W_AXIS_USER                                        )
  ) u_axis_pck_reg (
    .clk                ( clk_byte_hs_o                                                   ),
    .rst                ( clk_byte_hs_rst                                                 ),
    .i_axis_rx_tvalid   ( pck_axis_tvalid                                                 ),
    .i_axis_rx_tdata    ( {pck_axis_tdata,pck_axis_tlast,pck_axis_tuser,pck_axis_tkeep}   ),
    .o_axis_rx_tready   ( pck_axis_tready                                                 ),
    .o_axis_tx_tvalid   ( reg_axis_tvalid                                                 ),
    .o_axis_tx_tdata    ( {reg_axis_tdata,reg_axis_tlast,reg_axis_tuser,reg_axis_tkeep}   ),
    .i_axis_tx_tready   ( reg_axis_tready                                                 )
  );

//------------------------------------------------------------------------------
// Per-VC line-end hold
//------------------------------------------------------------------------------

  logic [31:0] hld_axis_tdata;
  logic [3:0]  hld_axis_tkeep;
  logic [W_AXIS_USER-1:0] hld_axis_tuser;
  logic        hld_axis_tvalid;
  logic        hld_axis_tlast;
  logic        hld_axis_tready;

  logic [31:0] hold_axis_tdata  [N_CSI_VC];
  logic [3:0]  hold_axis_tkeep  [N_CSI_VC];
  logic [W_AXIS_USER-1:0] hold_axis_tuser  [N_CSI_VC];
  logic [N_CSI_VC-1:0] hold_axis_tvalid;

  logic        hld_can_load;
  logic        hld_release;
  logic        hld_release_tlast;

  assign hld_can_load     = !hld_axis_tvalid || hld_axis_tready;
  assign hld_release      = hold_axis_tvalid[vc_idx] &&
                            (({payload_en_o,r_payload_en} == 3'b100) ||
                             (sp_en_o && (dt == 6'h01)));
  assign hld_release_tlast= sp_en_o && (dt == 6'h01);
  assign reg_axis_tready  = hld_can_load && !hld_release;

  always_ff @(posedge clk_byte_hs_o) begin
    if (clk_byte_hs_rst) begin
      hld_axis_tdata  <= '0;
      hld_axis_tkeep  <= '0;
      hld_axis_tuser  <= '0;
      hld_axis_tvalid <= '0;
      hld_axis_tlast  <= '0;
      hold_axis_tvalid <= '0;
      for (int i = 0; i < N_CSI_VC; i++) begin
        hold_axis_tdata[i] <= '0;
        hold_axis_tkeep[i] <= '0;
        hold_axis_tuser[i] <= '0;
      end
    end
    else if (hld_can_load) begin
      hld_axis_tvalid <= '0;
      hld_axis_tlast  <= '0;

      if (hld_release) begin
        hld_axis_tdata      <= hold_axis_tdata[vc_idx];
        hld_axis_tkeep      <= hold_axis_tkeep[vc_idx];
        hld_axis_tuser      <= hold_axis_tuser[vc_idx];
        hld_axis_tvalid     <= '1;
        hld_axis_tlast      <= hld_release_tlast;
        hold_axis_tvalid[vc_idx]<= '0;
      end
      else if (reg_axis_tvalid) begin
        if (reg_axis_tuser[1]) begin
          hold_axis_tdata[reg_axis_tuser[W_AXIS_USER-1:2]]  <= reg_axis_tdata;
          hold_axis_tkeep[reg_axis_tuser[W_AXIS_USER-1:2]]  <= reg_axis_tkeep;
          hold_axis_tuser[reg_axis_tuser[W_AXIS_USER-1:2]]  <= reg_axis_tuser;
          hold_axis_tvalid[reg_axis_tuser[W_AXIS_USER-1:2]] <= '1;
        end
        else begin
          hld_axis_tdata  <= reg_axis_tdata;
          hld_axis_tkeep  <= reg_axis_tkeep;
          hld_axis_tuser  <= reg_axis_tuser;
          hld_axis_tvalid <= '1;
          hld_axis_tlast  <= '0;
        end
      end
    end
  end

//------------------------------------------------------------------------------
// AXIS buffer - DC FIFO 32-bit WR 156Mhz -> 64-bit RD 156.25Mhz
//------------------------------------------------------------------------------

logic [W_AXIS_USER-1:0] w_axis_tuser;

  axis_buffer # (
    .IN_DWIDTH       ( 32     ),
    .OUT_DWIDTH      ( W_DATA ),
    .WAIT2SEND       ( 0      ),
    .W_USER          ( W_AXIS_USER ),
    .USER_LAST_BEAT_MASK ( {{(W_AXIS_USER-2){1'b0}}, 2'b10} ),
    .DUAL_CLOCK      ( 1      ),
    .NO_BACKPRESSURE ( 1      ),
    .BUF_DEPTH       ( 512    )
) u_axis_buffer (
    .in_clk            ( clk_byte_hs_o         ),
    .in_rst            ( clk_byte_hs_rst       ),
    .out_clk           ( i_sclk                ),
    .out_rst           ( i_srst                ),
    .i_axis_rx_tvalid  ( hld_axis_tvalid       ),
    .i_axis_rx_tdata   ( hld_axis_tdata        ),
    .i_axis_rx_tlast   ( hld_axis_tlast        ),
    .i_axis_rx_tuser   ( hld_axis_tuser        ),
    .i_axis_rx_tkeep   ( hld_axis_tkeep        ),
    .o_axis_rx_tready  ( hld_axis_tready       ),
    .o_axis_tx_tvalid  ( o_axis_tvalid         ),
    .o_axis_tx_tdata   ( o_axis_tdata          ),
    .o_axis_tx_tlast   ( o_axis_tlast          ),
    .o_axis_tx_tuser   ( w_axis_tuser          ),
    .o_axis_tx_tkeep   ( o_axis_tkeep          ),
    .i_axis_tx_tready  ( i_axis_tready         )
);

assign o_axis_tuser = w_axis_tuser[1:0];
assign o_axis_tidx  = w_axis_tuser[W_AXIS_USER-1:2];

endmodule
