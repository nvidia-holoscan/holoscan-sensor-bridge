module eth_10gb_top
#(
  parameter ID = 0
)(
  // clock and reset
  input          i_refclk_p, // 161.18MHz external ref clock
  input          i_refclk_n, // 161.18MHz external ref clock
  // SERDES IO
  input          i_pad_rx_n,
  input          i_pad_rx_p,
  output         o_pad_tx_n,
  output         o_pad_tx_p,
  // PCS Serdes clock
  input          i_pcs_clk,  // 100MHz - 300MHz calib clock
  input          i_pcs_rst_n,
  // PCS output user clock
  output         o_usr_clk,  // pcs clock 322.2656 MHz
  output         o_usr_clk_rdy,
  // MAC APB Interface, abp clk domain
  input          i_aclk,
  input          i_arst_n,
  input          i_mac_apb_psel,
  input          i_mac_apb_penable,
  input   [31:0] i_mac_apb_paddr,
  input   [31:0] i_mac_apb_pwdata,
  input          i_mac_apb_pwrite,
  output         o_mac_apb_pready,
  output  [31:0] o_mac_apb_prdata,
  output         o_mac_apb_pserr,
  // PCS APB Interface, abp clk domain
  input          i_pcs_apb_psel,
  input          i_pcs_apb_penable,
  input   [31:0] i_pcs_apb_paddr,
  input   [31:0] i_pcs_apb_pwdata,
  input          i_pcs_apb_pwrite,
  output         o_pcs_apb_pready,
  output  [31:0] o_pcs_apb_prdata,
  output         o_pcs_apb_pserr,
  // Ethernet XGMII MAC, pclk domain
  // XGMII processing clock
  input          i_pclk,     // 156.25MHz processing clock
  input          i_prst_n,   // active low reset
  // AXIS Tx
  input          i_axis_tx_tvalid,
  input          i_axis_tx_tlast,
  input   [ 7:0] i_axis_tx_tkeep,
  input   [63:0] i_axis_tx_tdata,
  input          i_axis_tx_tuser,
  output         o_axis_tx_tready,
  // AXIS Rx
  output         o_axis_rx_tvalid,
  output         o_axis_rx_tlast,
  output  [ 7:0] o_axis_rx_tkeep,
  output  [63:0] o_axis_rx_tdata,
  output         o_axis_rx_tuser,
  input          i_axis_rx_tready,
  // Debug Status
  output         o_mac_interrupt,
  output         o_mac_tx_staten,
  output  [25:0] o_mac_tx_statvec,
  output  [25:0] o_mac_rx_statvec,
  output         o_mac_rx_staten,
  output         o_mac_crc_err,
  output         o_pcs_rxval,
  output         o_pcs_txrdy
);

//------------------------------------------------------------------------------
// MAC Layer
//------------------------------------------------------------------------------

  // 10GbE MAC signals
  logic        xgmii_rxval;
  logic [63:0] xgmii_rxd;
  logic [ 7:0] xgmii_rxc;
  logic [63:0] xgmii_txd;
  logic [ 7:0] xgmii_txc;
  logic        xgmii_txrdy;
  logic        xgmii_rx_hi_ber;
  logic        xgmii_rx_blk_lock;

  // 10GbE MAC
  eth_10gb_mac u_10gbe_mac (
    .reset_n_i        ( i_pcs_rst_n       ), // XGMII reset needs to be free from pclk or doesn't work
    .rxmac_clk_i      ( i_pclk            ),
    .txmac_clk_i      ( i_pclk            ),
     // PCS interface
    .xgmii_rxd_i      ( xgmii_rxd         ),
    .xgmii_rxc_i      ( xgmii_rxc         ),
    .xgmii_txd_o      ( xgmii_txd         ),
    .xgmii_txc_o      ( xgmii_txc         ),
     // axis user interface
    .axis_tx_tvalid_i ( i_axis_tx_tvalid  ),
    .axis_tx_tlast_i  ( i_axis_tx_tlast   ),
    .axis_tx_tkeep_i  ( i_axis_tx_tkeep   ),
    .axis_tx_tdata_i  ( i_axis_tx_tdata   ),
    .axis_tx_tuser_i  ( i_axis_tx_tuser   ),
    .axis_tx_tready_o ( o_axis_tx_tready  ),
    .axis_rx_tvalid_o ( o_axis_rx_tvalid  ),
    .axis_rx_tlast_o  ( o_axis_rx_tlast   ),
    .axis_rx_tkeep_o  ( o_axis_rx_tkeep   ),
    .axis_rx_tdata_o  ( o_axis_rx_tdata   ),
    .axis_rx_tuser_o  ( o_axis_rx_tuser   ),
    // status
    .tx_statvec_o     ( o_mac_tx_statvec  ),
    .tx_staten_o      ( o_mac_tx_staten   ),
    .rx_statvec_o     ( o_mac_rx_statvec  ),
    .rx_staten_o      ( o_mac_rx_staten   ),
    // apb register interface
    .apb_clk_i        ( i_aclk            ),
    .apb_psel_i       ( i_mac_apb_psel    ),
    .apb_paddr_i      ( i_mac_apb_paddr   ),
    .apb_pwdata_i     ( i_mac_apb_pwdata  ),
    .apb_pwrite_i     ( i_mac_apb_pwrite  ),
    .apb_penable_i    ( i_mac_apb_penable ),
    .apb_pready_o     ( o_mac_apb_pready  ),
    .apb_prdata_o     ( o_mac_apb_prdata  ),
    .apb_pslverr_o    ( o_mac_apb_pserr   ), // ReLingo_waive_line:vendor_lscc:RL02
    // interrupt
    .int_o            ( o_mac_interrupt   )
  );
  
  logic curr_mac_crc_err;
  logic curr_mac_crc_known;
  logic next_mac_crc_err;
  logic next_mac_crc_known;
  logic rx_staten_reg;

  // The below code keeps track of a packet's FCS status at the output of the MAC. The indication of the FCS
  // error via the staten and statvec signals can take place before a packet egresses or while a packet is
  // egressing. In back-to-back packet scenarios, the indication for the second packet can take place prior
  // to the first packet starting egress or while the first packet is egressing. So this code keeps track of
  // the current packet's FCS status as well as the next packet's status. In this way, back-to-back packets
  // can be handled. Otherwise, the FCS error condition could be applied to the wrong packet going into the 
  // rx_parser.
  // 11/16/2023 - This code is not needed because the tuser signal indicates the FCS error (among others) at 
  // tlast, which is what is required at the rx_parser module. Leaving the code in for future ref, if needed.   
  always_ff @(posedge i_pclk) begin
    if (!i_prst_n) begin
      curr_mac_crc_err          <= 1'b0;
      curr_mac_crc_known        <= 1'b0;
      next_mac_crc_err          <= 1'b0;
      next_mac_crc_known        <= 1'b0;
      rx_staten_reg             <= 1'b0;
    end else begin
      rx_staten_reg             <= o_mac_rx_staten;
      if (o_mac_rx_staten && !rx_staten_reg) begin
        if (curr_mac_crc_known) begin
          next_mac_crc_known    <= 1'b1;
          next_mac_crc_err      <= o_mac_rx_statvec[17];
        end else begin
          curr_mac_crc_known    <= 1'b1;
          curr_mac_crc_err      <= o_mac_rx_statvec[17];
        end
      end else if (o_axis_rx_tvalid && o_axis_rx_tlast) begin
        if (next_mac_crc_known) begin
          curr_mac_crc_err      <= next_mac_crc_err;
          next_mac_crc_err      <= 1'b0;
          next_mac_crc_known    <= 1'b0;
        end else begin
          curr_mac_crc_known    <= 1'b0;
        end
      end
    end
  end

  // 11/16/2023 - o_mac_crc_err commented out because it is not needed (see above comments).
  //assign o_mac_crc_err = curr_mac_crc_known ? curr_mac_crc_err : 1'b0;
  assign o_mac_crc_err = 1'b0;

//------------------------------------------------------------------------------
// PCS PHY Layer
//------------------------------------------------------------------------------

  logic [3:0] xg_rx_fifo_st;
  logic [3:0] xg_tx_fifo_st;

  generate
    if (ID == 0) begin : PCS_0// Ethernet PCS Tile 2

      eth_10gb_pcs_0 u_10gbe_pcs (
        // Reference clock select. Use external PAD refclk
        .pad_refclkn_i    ( i_refclk_n              ),
        .pad_refclkp_i    ( i_refclk_p              ),
        .refclkp0_ext_i   ( 1'b0                    ),
        .refclkn0_ext_i   ( 1'b1                    ),
        .refclkp1_ext_i   ( 1'b0                    ),
        .refclkn1_ext_i   ( 1'b1                    ),
        .pll_0_refclk_i   ( 1'b0                    ),
        .pll_1_refclk_i   ( 1'b0                    ),
        .sd_pll_refclk_i  ( 1'b0                    ),
        .use_refmux_i     ( 1'b0                    ),
        .diffioclksel_i   ( 1'b0                    ),
        .clksel_i         ( 2'b0                    ),
        // // PAD SERDES
        .pad_rxn_i        ( i_pad_rx_n              ),
        .pad_rxp_i        ( i_pad_rx_p              ),
        .pad_txn_o        ( o_pad_tx_n              ),
        .pad_txp_o        ( o_pad_tx_p              ),
        // // XGMII Interface
        .xg_tx_clk_i      ( i_pclk                  ),
        .xg_tx_rst_n_i    ( i_pcs_rst_n             ),
        .xg_rx_clk_i      ( i_pclk                  ),
        .xg_rx_rst_n_i    ( i_pcs_rst_n             ),
        .xg_pcs_clkin_i   ( i_pcs_clk               ),
        .xg_tx_clk_o      ( o_usr_clk               ),
        .xg_rx_clk_o      (                         ),
        .xg_txc_i         ( xgmii_txc               ),
        .xg_txd_i         ( xgmii_txd               ),
        .xg_rxc_o         ( xgmii_rxc               ),
        .xg_rxd_o         ( xgmii_rxd               ),
        .xg_rxval_o       ( xgmii_rxval             ),
        .xg_txval_i       ( 1'b1                    ),
        .xg_txrdy_o       ( xgmii_txrdy             ),
        .xg_rx_hi_ber_o   ( xgmii_rx_hi_ber         ),
        .xg_rx_blk_lock_o ( xgmii_rx_blk_lock       ),
        // apb register interface
        .apb_pclk_i       ( i_aclk                  ),
        .apb_preset_n_i   ( i_arst_n                ),
        .apb_psel_i       ( i_pcs_apb_psel          ),
        .apb_penable_i    ( i_pcs_apb_penable       ),
        .apb_paddr_i      ( i_pcs_apb_paddr  [15:0] ),
        .apb_pwdata_i     ( i_pcs_apb_pwdata [15:0] ),
        .apb_pwrite_i     ( i_pcs_apb_pwrite        ),
        .apb_prdata_o     ( o_pcs_apb_prdata [15:0] ),
        .apb_pready_o     ( o_pcs_apb_pready        )
      );

    end else begin : PCS_1// Ethernet PCS Tile 3

      eth_10gb_pcs_1 u_10gbe_pcs (
        // Reference clock select. Use external PAD refclk
        .pad_refclkn_i    ( i_refclk_n              ),
        .pad_refclkp_i    ( i_refclk_p              ),
        .refclkp0_ext_i   ( 1'b0                    ),
        .refclkn0_ext_i   ( 1'b1                    ),
        .refclkp1_ext_i   ( 1'b0                    ),
        .refclkn1_ext_i   ( 1'b1                    ),
        .pll_0_refclk_i   ( 1'b0                    ),
        .pll_1_refclk_i   ( 1'b0                    ),
        .sd_pll_refclk_i  ( 1'b0                    ),
        .use_refmux_i     ( 1'b0                    ),
        .diffioclksel_i   ( 1'b0                    ),
        .clksel_i         ( 2'b0                    ),
        // // PAD SERDES
        .pad_rxn_i        ( i_pad_rx_n              ),
        .pad_rxp_i        ( i_pad_rx_p              ),
        .pad_txn_o        ( o_pad_tx_n              ),
        .pad_txp_o        ( o_pad_tx_p              ),
        // // XGMII Interface
        .xg_tx_clk_i      ( i_pclk                  ),
        .xg_tx_rst_n_i    ( i_pcs_rst_n             ),
        .xg_rx_clk_i      ( i_pclk                  ),
        .xg_rx_rst_n_i    ( i_pcs_rst_n             ),
        .xg_pcs_clkin_i   ( i_pcs_clk               ),
        .xg_tx_clk_o      ( o_usr_clk               ),
        .xg_rx_clk_o      (                         ),
        .xg_txc_i         ( xgmii_txc               ),
        .xg_txd_i         ( xgmii_txd               ),
        .xg_rxc_o         ( xgmii_rxc               ),
        .xg_rxd_o         ( xgmii_rxd               ),
        .xg_rxval_o       ( xgmii_rxval             ),
        .xg_txval_i       ( 1'b1                    ),
        .xg_txrdy_o       ( xgmii_txrdy             ),
        .xg_rx_hi_ber_o   ( xgmii_rx_hi_ber         ),
        .xg_rx_blk_lock_o ( xgmii_rx_blk_lock       ),
        // apb register interface
        .apb_pclk_i       ( i_aclk                  ),
        .apb_preset_n_i   ( i_arst_n                ),
        .apb_psel_i       ( i_pcs_apb_psel          ),
        .apb_penable_i    ( i_pcs_apb_penable       ),
        .apb_paddr_i      ( i_pcs_apb_paddr  [15:0] ),
        .apb_pwdata_i     ( i_pcs_apb_pwdata [15:0] ),
        .apb_pwrite_i     ( i_pcs_apb_pwrite        ),
        .apb_prdata_o     ( o_pcs_apb_prdata [15:0] ),
        .apb_pready_o     ( o_pcs_apb_pready        )
      );

    end
  endgenerate

  assign o_pcs_apb_pserr     = 1'b0;
  assign o_pcs_rxval         = xgmii_rxval;
  assign o_pcs_txrdy         = xgmii_txrdy;
  assign o_usr_clk_rdy       = xgmii_txrdy;

endmodule
