
module mipi_bridge (
    // LVDS Clocks
    input         eclk_i         , // 500Mhz LVDS Clock
    input         eclk90_i       , // 500Mhz LVDS clock 90 degree
    input         clk_25         , // 25MHz clk
    // MIPI Clocks
    input         mipi_sync_clk  , // MIPI Sync clk
    input         lmmi_clk       , // 19MHz

    input         pll_locked     ,
    input         rst_n          ,
    input         soft_rstn      , 

    input         mipi_en        , // Enables MIPI core
    // LMMI Ctrl Interface
    output        lmmi_ready      ,
    output [7:0]  lmmi_rdata      ,
    output        lmmi_rdata_valid,
    input  [7:0]  lmmi_wdata      ,
    input         lmmi_wr_rdn     ,
    input  [7:0]  lmmi_offset     ,
    input         lmmi_request    ,
    // RX - MIPI Interfaces
    inout         mipi_clk_p_io  , 
    inout         mipi_clk_n_io  ,
    inout  [3:0]  mipi_data_p_io ,
    inout  [3:0]  mipi_data_n_io ,

    // TX - LVDS Interface
    input         lvds_rdy_i     ,
    output        lvds_clk_o     ,
    output [10:0] lvds_data_o    

);


localparam PAD_IDX = 3;

//------------------------------------------------------------------------------
// Resets
//------------------------------------------------------------------------------

  logic clk_byte_o;
  logic clk_byte_hs_o;
  logic dphy_rstn;
  logic dphy_rstn_sync;
  logic mipi_en_sync;

assign dphy_rstn = soft_rstn & mipi_en_sync & rst_n;


reset_sync mipi_u_rst (
    .i_clk     ( clk_byte_hs_o ),
    .i_arst_n  ( dphy_rstn     ),
    .i_srst    ( 1'b0          ),
    .i_locked  ( 1'b1          ),
    .o_arst    ( mipi_rst      ),
    .o_arst_n  (               ),
    .o_srst    (               ), 
    .o_srst_n  ( mipi_rst_n    ) 
);

reset_sync mipi_sft_rst (
    .i_clk     ( mipi_sync_clk  ),
    .i_arst_n  ( dphy_rstn      ),
    .i_srst    ( 1'b0           ),
    .i_locked  ( pll_locked     ),
    .o_arst    (                ),
    .o_arst_n  ( dphy_rstn_sync ),
    .o_srst    (                ), 
    .o_srst_n  (                ) 
);

reset_sync lvds_u_rst (
    .i_clk     ( clk_25             ),
    .i_arst_n  ( (rst_n & soft_rstn)),
    .i_srst    ( 1'b0               ),
    .i_locked  ( pll_locked         ),
    .o_arst    (                    ),
    .o_arst_n  (                    ),
    .o_srst    ( lvds_rst           ), 
    .o_srst_n  (                    ) 
);

reset_sync lvds_sclk_u_rst (
    .i_clk     ( sclk_o             ),
    .i_arst_n  ( rst_n              ),
    .i_srst    ( 1'b0               ),
    .i_locked  ( lvds_ready         ),
    .o_arst    ( lvds_sclk_rst      ),
    .o_arst_n  (                    ),
    .o_srst    (                    ), 
    .o_srst_n  ( lvds_sclk_rst_n    ) 
);

data_sync #(
    .DATA_WIDTH ( 1              )  
) mipi_en_sync_inst (
    .clk        ( mipi_sync_clk  ),
    .rst_n      ( rst_n          ),
    .sync_in    ( mipi_en        ),
    .sync_out   ( mipi_en_sync   )
);

data_sync #(
    .DATA_WIDTH ( 1              )  
) mipi_en_hs_sync_inst (
    .clk        ( clk_byte_hs_o  ),
    .rst_n      ( rst_n          ),
    .sync_in    ( mipi_en        ),
    .sync_out   ( mipi_en_hs_sync)
);


data_sync #(
    .DATA_WIDTH ( 2                )  
) lvds_sync (
    .clk        ( sclk_o           ),
    .rst_n      ( lvds_sclk_rst_n  ),
    .sync_in    ( {lvds_rdy_i   , lvds_ready}    ),
    .sync_out   ( {lvds_rdy_sync, lvds_ready_sync}    )
);

//------------------------------------------------------------------------------
// RX - MIPI Interface
//------------------------------------------------------------------------------


  logic [63:0] payload_o;
  logic [63:0] r_payload;
  logic        payload_en_o;
  logic [1:0]  r_payload_en;
  logic [5:0]  dt;
  logic [1:0]  vc;
  logic [15:0] wc;
  logic [7:0]  ecc;
  logic [7:0]  payload_bytevld_o;
  logic [15:0] payload_crc_o;
  logic        payload_crcvld_o;
  logic [15:0] r_payload_crc;
  logic        r_payload_crcvld;
  logic [7:0]  r_payload_byte_vld;
  logic [7:0]  rr_payload_byte_vld;
  logic        sp_en_o;
  logic        lp_en_o;
  logic        is_frame;

  logic [63:0] w_data;

  logic [63:0] mipi_axis_tdata;
  logic [7:0]  mipi_axis_tkeep;
  logic [3:0]  mipi_axis_tuser;
  logic        mipi_axis_tvalid;
  logic        mipi_axis_tlast;

  logic       delay_valid;

  logic [79:0] lvds_axis_tdata;
  logic [9:0]  lvds_axis_tkeep;
  logic [4:0]  lvds_axis_tuser;
  logic        lvds_axis_tvalid;
  logic        lvds_axis_tlast;

  logic [3:0]  line_pad;
  logic [15:0] wc_padded;
  logic [15:0] byte_cnt;
  logic [15:0] pad_bytes;
  logic [3:0]  byte_incr;
  logic [3:0]  byte_incr_r;
  logic [15:0] wc_r;
  logic [5:0]  dt_r;
  logic        last_cycle;
  logic        one_pad_cycle;

  logic        drop_metadata;
  logic        pad_lines;
  logic        is_embedded;
  
    mipi_csi_rx_rcfg mipi_rx_inst(
       .lmmi_clk_i          ( lmmi_clk            ),    // LMMI Interface
       .lmmi_resetn_i       ( (pll_locked&rst_n)  ),    // LMMI Interface
       .lmmi_wdata_i        ( lmmi_wdata          ),    // LMMI Interface
       .lmmi_rdata_o        ( lmmi_rdata          ),    // LMMI Interface
       .lmmi_rdata_valid_o  ( lmmi_rdata_valid    ),    // LMMI Interface
       .lmmi_wr_rdn_i       ( lmmi_wr_rdn         ),    // LMMI Interface
       .lmmi_offset_i       ( lmmi_offset         ),    // LMMI Interface
       .lmmi_request_i      ( lmmi_request        ),    // LMMI Interface
       .lmmi_ready_o        ( lmmi_ready          ),    // LMMI Interface
       .sync_clk_i          ( mipi_sync_clk       ),    // 60 MHz
       .sync_rst_i          ( !dphy_rstn_sync     ),    // 
       .clk_byte_o          ( clk_byte_o          ),    // Byte Clock
       .clk_byte_hs_o       ( clk_byte_hs_o       ),    // HS Byte Clock
       .clk_byte_fr_i       ( clk_byte_hs_o       ),    // Clk
       .reset_byte_fr_n_i   ( dphy_rstn_sync      ),    // RESET
       .clk_n_io            ( mipi_clk_n_io       ),    // MIPI D-PHY CLK Lane
       .clk_p_io            ( mipi_clk_p_io       ),    // MIPI D-PHY CLK Lane
       .d_p_io              ( mipi_data_p_io      ),    // MIPI D-PHY DATA Lane (4)
       .d_n_io              ( mipi_data_n_io      ),    // MIPI D-PHY DATA Lane (4)
       .payload_en_o        ( payload_en_o        ),    // Enable signal of payload
       .payload_o           ( payload_o           ),    // Payload
       .dt_o                ( dt                  ),    // Packet Header
       .vc_o                ( vc                  ),    // Packet Header
       .vcx_o               (                     ),
       .wc_o                ( wc                  ),    // Packet Header
       .ecc_o               ( ecc                 ),    // Packet Header
       .payload_crc_o       ( payload_crc_o       ),    // Payload valid size
       .payload_crcvld_o    ( payload_crcvld_o    ),
       .crc_check_o         (                     ),    // CRC Error Flag
       .crc_error_o         (                     ),    // CRC Error Flag
       .ecc_check_o         (                     ),    // ECC Error Flag
       .ecc_byte_error_o    (                     ),    // ECC Error Flags
       .ecc_1bit_error_o    (                     ),    // ECC Error Flags
       .ecc_2bit_error_o    (                     ),    // ECC Error Flags
       .dphy_rxdatawidth_hs_o(                    ),
       .dphy_cfg_num_lanes_o (                    ),
       .pd_dphy_i           ( !dphy_rstn_sync     ),    // Power Down
       .sp_en_o             ( sp_en_o             ),    // Packet Parser Output 
       .lp_en_o             ( lp_en_o             ),       
       .payload_bytevld_o   ( payload_bytevld_o   ),    // Valid bytes of payload
       .lp_av_en_o          (                     ),
       .skewcal_det_o       (                     ),
       .skewcal_done_o      (                     ));


  always @(posedge clk_byte_hs_o or posedge mipi_rst) begin
    if (mipi_rst) begin
      r_payload_en      <= '0;
      r_payload         <= '0;
      r_payload_crc     <= '0;
      r_payload_crcvld  <= '0;
      mipi_axis_tdata   <= '0; 
      mipi_axis_tkeep   <= '0; 
      mipi_axis_tvalid  <= '0;
      mipi_axis_tlast   <= '0; 
      mipi_axis_tuser   <= '0; 
      is_frame          <= '0;
      wc_padded         <= '0;
      byte_cnt          <= '0;
      wc_r              <= '0;
      dt_r              <= '0;
      byte_incr_r       <= '0;
      delay_valid       <= '0;
      r_payload_byte_vld  <= '0;
      rr_payload_byte_vld <= '0;
      one_pad_cycle       <= '0;
    end
    else begin
      r_payload_en       <= {r_payload_en[0],payload_en_o};
      r_payload          <= payload_o;
      r_payload_crc      <= (payload_crcvld_o) ? payload_crc_o : r_payload_crc;
      r_payload_crcvld   <= payload_crcvld_o;
      is_frame           <= (sp_en_o && (dt == 6'h00)) ? 1'b1   :
                            (sp_en_o && (dt == 6'h01)) ? 1'b0   : 
                                                        is_frame;
      rr_payload_byte_vld <= payload_bytevld_o;
      if (({payload_en_o,r_payload_en}==3'b100) || (sp_en_o)) begin
        if (!sp_en_o || (dt == 6'h01)) begin  // Only delay valid on next long packet start, or frame end
          mipi_axis_tvalid <= (is_frame && delay_valid);
          delay_valid      <= '0;
        end
        mipi_axis_tlast  <= (dt == 6'h01);
        mipi_axis_tuser[3:1] <= {vc,delay_valid}; //vc,line_end
        byte_cnt         <= '0;
        wc_r             <= wc;
        dt_r             <= dt;
        byte_incr_r      <= '0;
        r_payload_byte_vld <= payload_bytevld_o;
        //wc_padded                <= wc;
        wc_padded[15:PAD_IDX]    <= wc[15:PAD_IDX] + (|wc[PAD_IDX-1:0]);
        wc_padded[PAD_IDX-1:0]   <= '0;
      end
      else if (r_payload_en[0]) begin
        mipi_axis_tdata    <= w_data;
        mipi_axis_tkeep    <= r_payload_byte_vld;
        mipi_axis_tvalid   <= is_frame && !last_cycle;  // Don't assert high on last cycle of line
        mipi_axis_tlast    <= '0;
        mipi_axis_tuser[0] <= (sp_en_o) ? mipi_axis_tuser[0] : is_embedded; // Only latch data type on LP
        delay_valid        <= '1;
        is_frame           <= is_frame;
        byte_cnt           <= byte_cnt + byte_incr_r;
        byte_incr_r        <= (byte_incr_r == '0) ? byte_incr : byte_incr_r;
        mipi_axis_tuser[1] <= last_cycle;
        one_pad_cycle      <= ((wc_padded - wc) < (byte_incr_r));
      end
      else if (((byte_cnt+byte_incr_r) >= wc) && (byte_incr_r != '0) && (!last_cycle||!one_pad_cycle)) begin // Pad line to 64 Bytes
        mipi_axis_tdata    <= '0;
        mipi_axis_tvalid   <= !last_cycle; // Line Padding
        mipi_axis_tlast    <= '0;
        byte_cnt           <= byte_cnt + byte_incr_r;
        mipi_axis_tuser[1] <= '0;
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
                   r_payload_byte_vld[6] ? 8'd7:
                   r_payload_byte_vld[5] ? 8'd6:
                   r_payload_byte_vld[4] ? 8'd5:
                   r_payload_byte_vld[3] ? 8'd4:
                   r_payload_byte_vld[2] ? 8'd3:
                   r_payload_byte_vld[1] ? 8'd2:
                   r_payload_byte_vld[0] ? 8'd1:
                                           8'd0;
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

  logic [63:0] pck_axis_tdata;
  logic [7:0]  pck_axis_tkeep;
  logic [3:0]  pck_axis_tuser;
  logic        pck_axis_tvalid;
  logic        pck_axis_tlast;
  logic        pck_axis_tready;

  logic [63:0] reg_axis_tdata;
  logic [7:0]  reg_axis_tkeep;
  logic [3:0]  reg_axis_tuser;
  logic        reg_axis_tvalid;
  logic        reg_axis_tlast;
  logic        reg_axis_tready;

  axis_packer # (
    .DWIDTH           ( 64                     ),
    .W_USER           ( 4                      )
  ) u_axis_pack  (
    .clk              ( clk_byte_hs_o          ),
    .rst              ( mipi_rst               ),
    .i_axis_tvalid    ( mipi_axis_tvalid       ),
    .i_axis_tdata     ( mipi_axis_tdata        ),
    .i_axis_tlast     ( mipi_axis_tlast        ),
    .i_axis_tuser     ( mipi_axis_tuser        ),
    .i_axis_tkeep     ( mipi_axis_tkeep        ),
    .o_axis_tready    ( mipi_axis_tready       ),
    .o_axis_tvalid    ( pck_axis_tvalid        ),
    .o_axis_tdata     ( pck_axis_tdata         ),
    .o_axis_tlast     ( pck_axis_tlast         ),
    .o_axis_tuser     ( pck_axis_tuser         ),
    .o_axis_tkeep     ( pck_axis_tkeep         ),
    .i_axis_tready    ( pck_axis_tready        )
);


  axis_reg # (
    .DWIDTH             ( 64 + 8 + 1 + 4                                                  )
  ) u_axis_pck_reg (
    .clk                ( clk_byte_hs_o                                                   ),
    .rst                ( mipi_rst                                                        ),
    .i_axis_rx_tvalid   ( pck_axis_tvalid                                                 ),
    .i_axis_rx_tdata    ( {pck_axis_tdata,pck_axis_tlast,pck_axis_tuser,pck_axis_tkeep}   ),
    .o_axis_rx_tready   ( pck_axis_tready                                                 ),
    .o_axis_tx_tvalid   ( reg_axis_tvalid                                                 ),
    .o_axis_tx_tdata    ( {reg_axis_tdata,reg_axis_tlast,reg_axis_tuser,reg_axis_tkeep}   ),
    .i_axis_tx_tready   ( reg_axis_tready                                                 )
  );

//------------------------------------------------------------------------------
// AXIS buffer
//------------------------------------------------------------------------------

  logic [9:0] buf_tuser;
  logic [4:0] lvds_tuser;

  axis_buffer # (
      .IN_DWIDTH     ( 64     ),
      .OUT_DWIDTH    ( 80     ),
      .WAIT2SEND     ( 0      ),
      .W_USER        ( 4      ),
      .DUAL_CLOCK    ( 1      ),
      .OUT_W_USER    ( 5      )
    ) u_axis_buffer (
      .in_clk            ( clk_byte_hs_o         ),
      .in_rst            ( mipi_rst              ),
      .out_clk           ( sclk_o                ),
      .out_rst           ( !lvds_ready_sync      ),
      .i_axis_rx_tvalid  ( reg_axis_tvalid       ),
      .i_axis_rx_tdata   ( reg_axis_tdata        ),
      .i_axis_rx_tlast   ( reg_axis_tlast        ),
      .i_axis_rx_tuser   ( reg_axis_tuser        ),
      .i_axis_rx_tkeep   ( reg_axis_tkeep        ),
      .o_axis_rx_tready  ( reg_axis_tready       ),
      .o_axis_tx_tvalid  ( lvds_axis_tvalid      ),
      .o_axis_tx_tdata   ( lvds_axis_tdata       ),
      .o_axis_tx_tlast   ( lvds_axis_tlast       ),
      .o_axis_tx_tuser   ( lvds_axis_tuser       ),
      .o_axis_tx_tkeep   ( lvds_axis_tkeep       ),
      .i_axis_tx_tready  ( '1                    )
    );

//------------------------------------------------------------------------------
// TX - LVDS Interface
//------------------------------------------------------------------------------

logic sync_start;
logic [87:0] lvds_din;

logic [3:0] lvds_tkeep_cnt; // Send count instead of tkeep to reduce number of bits

always_ff @ (posedge clk_25 or posedge lvds_rst) begin
  if (lvds_rst) begin
    sync_start <= 1'b0;
  end
  else begin
    if (!sync_start) begin  // Activates 1 slow clk cycle after reset
      sync_start <= 1'b1;
    end
  end
end

logic [2:0] lvds_axis_metadata;


// Since data is padded to 8 bytes, only tkeep value of 16, 32, 48, 64, 80 is possible

assign lvds_axis_metadata =  (!lvds_axis_tvalid ) ? 'd0 : // !valid, !tlast, tkeep=x
                             (!lvds_axis_tlast  ) ? 'd1 : //  valid, !tlast, tkeep=10 bytes
                             (lvds_axis_tkeep[9]) ? 'd2 : //  valid,  tlast, tkeep=10 bytes
                             (lvds_axis_tkeep[7]) ? 'd3 : //  valid,  tlast, tkeep= 8 bytes
                             (lvds_axis_tkeep[5]) ? 'd4 : //  valid,  tlast, tkeep= 6 bytes
                             (lvds_axis_tkeep[3]) ? 'd5 : //  valid,  tlast, tkeep= 4 bytes
                             (lvds_axis_tkeep[1]) ? 'd6 : //  valid,  tlast, tkeep= 2 bytes
                                                    'd7 ; //  valid,  tlast, tkeep= 0 bytes


lvds_tx lvds_tx_inst(
    .sync_clk_i   ( clk_25        ),  // 25 MHz sync clk - TB uses 25MHz, other docs say "should be slowest clock in design"
    .sync_rst_i   ( lvds_rst      ),  // Active high main reset
    .sync_start_i ( sync_start    ),  // high 1 clock after !rst_n, stays high
    .eclk_i       ( eclk_i        ),  // 500Mhz
    .eclk90_i     ( eclk90_i      ),  // 500Mhz 90deg
    .data_i       ( lvds_din      ),  // [87:0] data in - gear 2 of 10 bit data, 1 valid bit by 8 packets
    .data_o       ( lvds_data_o   ),  // LVDS - [10:0] interface - 10 bits of data, 1 valid bit
    .clk_o        ( lvds_clk_o    ),  // LVDS - Output clk
    .sclk_o       ( sclk_o        ),  // Clk output for data_i
    .ready_o      ( lvds_ready    )   // Ready from LVDS
);

wire [87:0] cal_data = 88'h005A55FEDCBA9876543210;

always_ff @ (posedge sclk_o) begin 
  if (!lvds_ready_sync) begin 
    lvds_din   <= 88'b0;
  end else begin 
    lvds_din   <= lvds_rdy_sync ? {lvds_axis_tuser,lvds_axis_metadata,lvds_axis_tdata} : cal_data;
  end 
end

endmodule

