// Lattice Sensor Board Example Design
`include "HOLOLINK_def.svh"

module FPGA_top
  import HOLOLINK_pkg::*;
#(
  parameter BUILD_REV = 48'h0
)(
  input         RESET_N,  // board reset push button
  output        SWRST_N,  // SW reset
  // 10GbE SFP
  input         ETH_REFCLK_P,
  input         ETH_REFCLK_N,
  input  [ 1:0] ETH_RXD_P,
  input  [ 1:0] ETH_RXD_N,
  output [ 1:0] ETH_TXD_P,
  output [ 1:0] ETH_TXD_N,
  // SFP Fiber Optics Disable
  output [ 1:0] SFP_TX_DISABLE,
  // LVDS Pixel Data Input
  input  [ 1:0] CAM_DCLK,
  input  [10:0] CAM_DATA [2],
  output        CAM_DRDY,
  // I2C Interfaces
  inout         CTRL_I2C_SCL,
  inout         CTRL_I2C_SDA,

  inout  [ 2:0] CAM_I2C_SCL,
  inout  [ 2:0] CAM_I2C_SDA,

  inout  [14:0] CAM_GPIO,
  // SPI Interfaces
  output        CTRL_SPI_MCSN,
  output        CTRL_SPI_MSCK,
  output        CTRL_SPI_MOSI,
  input         CTRL_SPI_MISO,

  output        FLASH_SPI_MCSN,
  output        FLASH_SPI_MSCK,
  inout  [ 3:0] FLASH_SPI_SDIO,
  // GPIO Control
  output [ 1:0] CAM_RST_N,
  output        CAM_1V8_EN,
  output        CAM_2V8_EN,
  output        CAM_3V3_EN,
  output        CLK_SYNTH_EN,
  output        CLK_BUFF_OEN,
  // GPIO Status
  input         CLK_SYNTH_LOCKED,
  // GPIO
  inout  [15:0] GPIO,
  // PPS
  output        PPS,
  output        PTP_CAM_CLK
);

//------------------------------------------------------------------------------
// FPGA Board Control
//------------------------------------------------------------------------------

  logic [`SENSOR_IF_INST-1:0] sw_sen_rst;
  logic                       sw_sys_rst;

  assign CAM_1V8_EN       = 1'b1;
  assign CAM_2V8_EN       = 1'b1;
  assign CAM_3V3_EN       = 1'b1;
  assign CLK_SYNTH_EN     = 1'b1;
  assign CLK_BUFF_OEN     = 1'b0;
  // Sensor Reset
  assign CAM_RST_N[0]     = ~sw_sen_rst[0];
  assign CAM_RST_N[1]     = ~sw_sen_rst[1];
  // Fiber Optics Disable Input Jumper
  assign SFP_TX_DISABLE   = '0;

//------------------------------------------------------------------------------
// Clock and Reset
//------------------------------------------------------------------------------

  logic [`HOST_IF_INST-1:0] usr_clk;     // pcs user clock out
  logic [`HOST_IF_INST-1:0] usr_clk_rdy; // pcs user clock out ready
  logic                     usr_clk_locked;
  /* synthesis syn_keep=1 nomerge=""*/
  logic                     adc_clk;     // 50MHz ADC clock
  /* synthesis syn_keep=1 nomerge=""*/
  logic                     pcs_clk;     // 100-300 MHz PCS calibration clock
  /* synthesis syn_keep=1 nomerge=""*/
  logic                     apb_clk;     // ctrl plane clock
  /* synthesis syn_keep=1 nomerge=""*/
  logic                     hif_clk;     // data plane clock
  /* synthesis syn_keep=1 nomerge=""*/
  logic                     ptp_clk;     // ptp clock
  /* synthesis syn_keep=1 nomerge=""*/
  logic                     sys_rst;     // system active high reset
  logic                     apb_rst;     // apb active high reset
  logic                     hif_rst;     // host interface active high reset
  logic                     sif_rst;     // sensor interface active high reset
  logic                     ptp_rst;     // ptp active high reset
  logic                     pcs_rst_n;   // ethernet pcs active low reset
  logic                     ptp_cam_24m_clk;
  /* synthesis syn_keep=1 nomerge=""*/
  logic [31:0]              ptp_nsec;

  logic                     i2s_mclk_ext;
  /* synthesis syn_keep=1 nomerge=""*/
  logic                     i2s_clk_ext;
  /* synthesis syn_keep=1 nomerge=""*/
  logic                     i2s_clk_int;
  /* synthesis syn_keep=1 nomerge=""*/

  assign usr_clk_locked = &usr_clk_rdy;

  clk_n_rst u_clk_n_rst (
    .i_refclk        ( usr_clk [0]    ), // pcs user clock output
    .i_locked        ( usr_clk_locked ), // pcs user clock locked

    .o_adc_clk       ( adc_clk        ), // 50MHz clock for ADC Temp
    .o_pcs_clk       ( pcs_clk        ), // pcs calibration clock
    .o_hif_clk       ( hif_clk        ), // host interface clock
    .o_apb_clk       ( apb_clk        ), // apb interface clock
    .o_ptp_clk       ( ptp_clk        ), // ptp clock

    .i_ptp_nsec      ( ptp_nsec       ),
    .o_ptp_sensor_pll_lock ( ptp_sensor_pll_locked ),
    .o_ptp_cam_clk   ( ptp_cam_24m_clk), //ptp 24MHz clock

    .o_i2s_clk_int   ( i2s_clk_int    ),
    .o_i2s_clk_ext   ( i2s_clk_ext    ),
    .o_i2s_mclk_ext  ( i2s_mclk_ext   ),

    .i_pb_rst_n      ( RESET_N        ), // asynchronous active low board reset
    .i_sw_rst        ( sw_sys_rst     ), // software controlled active high reset

    .o_sys_rst       ( sys_rst        ), // system active high reset
    .o_pcs_rst_n     ( pcs_rst_n      )  // ethernet pcs active low reset
  );

  assign SWRST_N = !sys_rst;
  assign PTP_CAM_CLK = ptp_cam_24m_clk;


//------------------------------------------------------------------------------
// PPS
//------------------------------------------------------------------------------
  logic        sys_pps;
  logic        sys_pps_stretch;
  logic [17:0] timer_cnt;
  logic        timer_done;

`ifdef SIMULATION
  assign timer_done = timer_cnt[3];
`else
  assign timer_done = timer_cnt[17];
`endif

  always_ff @ (posedge ptp_clk) begin
    if (ptp_rst) begin 
      sys_pps_stretch <= 1'b0;
      timer_cnt   <= '0;
    end else begin
      if (sys_pps) begin
        sys_pps_stretch <= 1'b1;
      end else if (timer_done) begin
        timer_cnt   <= '0;
        sys_pps_stretch <= 1'b0;
      end else if (sys_pps_stretch) begin
        timer_cnt   <= timer_cnt + 1'b1;
      end
    end
  end

  logic        init_done;

//------------------------------------------------------------------------------
// APB Interface
//------------------------------------------------------------------------------

  // User Drops
  logic [`REG_INST-1:0] apb_psel;
  logic                 apb_penable;
  logic [31         :0] apb_paddr;
  logic [31         :0] apb_pwdata;
  logic                 apb_pwrite;
  logic [`REG_INST-1:0] apb_pready;
  logic [31         :0] apb_prdata [`REG_INST];
  logic [`REG_INST-1:0] apb_pserr;


  //Tie off unused REG_INST APB signals
  genvar i;

/*
  for(i=7; i<`REG_INST; i++) begin
    assign apb_pready[i] = '0;
    assign apb_prdata[i] = '0;
    assign apb_pserr [i] = '0;
  end
*/

//------------------------------------------------------------------------------
// Lattice 10GbE Host Interface
//------------------------------------------------------------------------------

  logic [`HOST_IF_INST-1  :0] hif_tx_axis_tvalid;
  logic [`HOST_IF_INST-1  :0] hif_tx_axis_tlast;
  logic [`HOST_WIDTH-1    :0] hif_tx_axis_tdata [`HOST_IF_INST];
  logic [`HOSTKEEP_WIDTH-1:0] hif_tx_axis_tkeep [`HOST_IF_INST];
  logic [`HOSTUSER_WIDTH-1:0] hif_tx_axis_tuser [`HOST_IF_INST];
  logic [`HOST_IF_INST-1  :0] hif_tx_axis_tready;

  logic [`HOST_IF_INST-1  :0] hif_rx_axis_tvalid;
  logic [`HOST_IF_INST-1  :0] hif_rx_axis_tlast;
  logic [`HOST_WIDTH-1    :0] hif_rx_axis_tdata [`HOST_IF_INST];
  logic [`HOSTKEEP_WIDTH-1:0] hif_rx_axis_tkeep [`HOST_IF_INST];
  logic [`HOSTUSER_WIDTH-1:0] hif_rx_axis_tuser [`HOST_IF_INST];
  logic [`HOST_IF_INST-1  :0] hif_rx_axis_tready;

  generate
    for (i=0; i<`HOST_IF_INST; i++) begin: ethernet_10gb

      eth_10gb_top #(
        .ID               ( i                         )
      ) u_10gbe (
        // clock and reset
        .i_refclk_p       ( ETH_REFCLK_P              ),
        .i_refclk_n       ( ETH_REFCLK_N              ),
        // SERDES IO
        .i_pad_rx_p       ( ETH_RXD_P             [i] ),
        .i_pad_rx_n       ( ETH_RXD_N             [i] ),
        .o_pad_tx_p       ( ETH_TXD_P             [i] ),
        .o_pad_tx_n       ( ETH_TXD_N             [i] ),
        // PCS clock
        .i_pcs_clk        ( pcs_clk                   ),
        .i_pcs_rst_n      ( pcs_rst_n                 ),
        .o_usr_clk        ( usr_clk               [i] ),
        .o_usr_clk_rdy    ( usr_clk_rdy           [i] ),
        // APB Interface, abp clk domain
        .i_aclk           ( apb_clk                   ),
        .i_arst_n         (~apb_rst                   ),
        // PCS APB Interface, abp clk domain
        .i_pcs_apb_psel   ( apb_psel          [0+i*2] ),
        .i_pcs_apb_penable( apb_penable               ),
        .i_pcs_apb_paddr  ( apb_paddr                 ),
        .i_pcs_apb_pwdata ( apb_pwdata                ),
        .i_pcs_apb_pwrite ( apb_pwrite                ),
        .o_pcs_apb_pready ( apb_pready        [0+i*2] ),
        .o_pcs_apb_prdata ( apb_prdata        [0+i*2] ),
        .o_pcs_apb_pserr  ( apb_pserr         [0+i*2] ),
        // MAC APB Interface, abp clk domain
        .i_mac_apb_psel   ( apb_psel          [1+i*2] ),
        .i_mac_apb_penable( apb_penable               ),
        .i_mac_apb_paddr  ( apb_paddr                 ),
        .i_mac_apb_pwdata ( apb_pwdata                ),
        .i_mac_apb_pwrite ( apb_pwrite                ),
        .o_mac_apb_pready ( apb_pready        [1+i*2] ),
        .o_mac_apb_prdata ( apb_prdata        [1+i*2] ),
        .o_mac_apb_pserr  ( apb_pserr         [1+i*2] ),
        // Ethernet XGMII MAC, hif_clkdomain
        .i_pclk           ( hif_clk                   ),
        .i_prst_n         (~hif_rst                   ),

        .i_axis_tx_tvalid ( hif_tx_axis_tvalid    [i] ),
        .i_axis_tx_tlast  ( hif_tx_axis_tlast     [i] ),
        .i_axis_tx_tkeep  ( hif_tx_axis_tkeep     [i] ),
        .i_axis_tx_tdata  ( hif_tx_axis_tdata     [i] ),
        .i_axis_tx_tuser  ( hif_tx_axis_tuser     [i] ),
        .o_axis_tx_tready ( hif_tx_axis_tready    [i] ),

        .o_axis_rx_tvalid ( hif_rx_axis_tvalid    [i] ),
        .o_axis_rx_tlast  ( hif_rx_axis_tlast     [i] ),
        .o_axis_rx_tkeep  ( hif_rx_axis_tkeep     [i] ),
        .o_axis_rx_tdata  ( hif_rx_axis_tdata     [i] ),
        .o_axis_rx_tuser  ( hif_rx_axis_tuser     [i] ),
        .i_axis_rx_tready ( hif_rx_axis_tready    [i] ),
        // Debug Status
        .o_mac_interrupt  (                           ),
        .o_mac_tx_staten  (                           ),
        .o_mac_tx_statvec (                           ),
        .o_mac_rx_statvec (                           ),
        .o_mac_rx_staten  (                           ),
        .o_mac_crc_err    (                           ),
        .o_pcs_rxval      (                           ),
        .o_pcs_txrdy      (                           )
      );

    end
  endgenerate

//------------------------------------------------------------------------------
// SPI Interface
//------------------------------------------------------------------------------

  // Adding a glitch filter, but can be removed if IO pads itself provides glitch filtering
  logic [3:0] flsh_spi_sdio_sync;
  logic       ctrl_spi_miso_sync;

  data_sync    #(
    .DATA_WIDTH ( 5                                      )
  ) spi_glitch_filter (
    .clk        ( hif_clk                                ),
    .rst_n      (~hif_rst                                ),
    .sync_in    ({FLASH_SPI_SDIO    , CTRL_SPI_MISO}     ),
    .sync_out   ({flsh_spi_sdio_sync, ctrl_spi_miso_sync})
  );

  // SPI Interface, QSPI compatible
  logic [`SPI_INST-1:0] spi_csn;
  logic [`SPI_INST-1:0] spi_sck;
  logic [3          :0] spi_sdio_i [`SPI_INST];
  logic [3          :0] spi_sdio_o [`SPI_INST];
  logic [`SPI_INST-1:0] spi_oen;

  // Regular SPI
  assign CTRL_SPI_MSCK = spi_sck   [0];
  assign CTRL_SPI_MCSN = spi_csn   [0];
  assign CTRL_SPI_MOSI = spi_sdio_o[0];
  assign spi_sdio_i[0] = {2'b0, ctrl_spi_miso_sync, 1'b0};
  // QSPI Flash
  assign FLASH_SPI_MSCK = spi_sck  [1];
  assign FLASH_SPI_MCSN = spi_csn  [1];
  assign FLASH_SPI_SDIO = spi_oen  [1] ? spi_sdio_o[1] : 4'hz;
  assign spi_sdio_i[1]  = flsh_spi_sdio_sync;

//------------------------------------------------------------------------------
// I2C Interface
//------------------------------------------------------------------------------

  // Adding a glitch filter, but can be removed if IO pads itself provides glitch filtering
  logic                 ctrl_i2c_scl_sync;
  logic                 ctrl_i2c_sda_sync;
  logic [`I2C_INST-2:0] cam_i2c_scl_sync;
  logic [`I2C_INST-2:0] cam_i2c_sda_sync;

  glitch_filter  #(
    .DATA_WIDTH   ( `I2C_INST*2                           ),
    .RESET_VALUE  ( 1'b1                                  ),
    .FILTER_DEPTH ( 8                                     )
  ) i2c_glitch_filter (
    .clk          ( hif_clk                               ),
    .rst_n        (~hif_rst                               ),
    .sync_in      ({CTRL_I2C_SDA, CTRL_I2C_SCL,
                    CAM_I2C_SDA , CAM_I2C_SCL            }),
    .sync_out     ({ctrl_i2c_sda_sync, ctrl_i2c_scl_sync,
                    cam_i2c_sda_sync , cam_i2c_scl_sync  })
  );

  logic [`I2C_INST-1:0] i2c_scl;
  logic [`I2C_INST-1:0] i2c_sda;
  logic [`I2C_INST-1:0] i2c_scl_en;
  logic [`I2C_INST-1:0] i2c_sda_en;

  assign i2c_scl[0]   = i2c_scl_en[0] ? ctrl_i2c_scl_sync : 1'b0;
  assign i2c_sda[0]   = i2c_sda_en[0] ? ctrl_i2c_sda_sync : 1'b0;
  assign CTRL_I2C_SCL = i2c_scl_en[0] ? 1'bz : 1'b0;
  assign CTRL_I2C_SDA = i2c_sda_en[0] ? 1'bz : 1'b0;

  generate
    for (i=0; i<`I2C_INST-1; i++) begin
      assign i2c_scl[i+1]   = i2c_scl_en[i+1] ? cam_i2c_scl_sync[i] : 1'b0;
      assign i2c_sda[i+1]   = i2c_sda_en[i+1] ? cam_i2c_sda_sync[i] : 1'b0;
      assign CAM_I2C_SCL[i] = i2c_scl_en[i+1] ? 1'bz : 1'b0;
      assign CAM_I2C_SDA[i] = i2c_sda_en[i+1] ? 1'bz : 1'b0;
    end
  endgenerate

//------------------------------------------------------------------------------
// I2S IF
//------------------------------------------------------------------------------
  logic [`SENSOR_IF_INST-1:0] sif_tx_axis_tvalid;
  logic [`SENSOR_IF_INST-1:0] sif_tx_axis_tlast;
  logic [`DATAPATH_WIDTH-1:0] sif_tx_axis_tdata [`SENSOR_IF_INST];
  logic [`DATAKEEP_WIDTH-1:0] sif_tx_axis_tkeep [`SENSOR_IF_INST];
  logic [`DATAUSER_WIDTH-1:0] sif_tx_axis_tuser [`SENSOR_IF_INST];
  logic [`SENSOR_IF_INST-1:0] sif_tx_axis_tready;

  assign sif_tx_axis_tready[1] = 1'b0; //Tie Sensor TX[1]

  logic                       i2s_tx_axis_tvalid;
  logic                       i2s_tx_axis_tlast;
  logic [31:0]                i2s_tx_axis_tdata;
  logic [3:0]                 i2s_tx_axis_tkeep;
  logic                       i2s_tx_axis_tuser;
  logic                       i2s_tx_axis_tready;

  axis_buffer #(
    .IN_DWIDTH           ( `HOST_WIDTH            ),
    .OUT_DWIDTH          ( 32                     ),
    .BUF_DEPTH           ( 64                     ),
    .WAIT2SEND           ( 0                      ),
    .DUAL_CLOCK          ( 1                      ),
    .W_USER              ( 1                      )
  ) u_i2s_tx_axis_buffer (

    .in_clk              ( hif_clk                ),
    .in_rst              ( hif_rst                ),
    .out_clk             ( apb_clk                ),
    .out_rst             ( apb_rst                ),

    //Sensor TX
    .i_axis_rx_tvalid    ( sif_tx_axis_tvalid[0]  ),
    .i_axis_rx_tdata     ( sif_tx_axis_tdata[0]   ),
    .i_axis_rx_tlast     ( sif_tx_axis_tlast[0]   ),
    .i_axis_rx_tuser     ( sif_tx_axis_tuser[0]   ),
    .i_axis_rx_tkeep     ( sif_tx_axis_tkeep[0]   ),
    .o_axis_rx_tready    ( sif_tx_axis_tready[0]  ),
    //I2S IP
    .o_axis_tx_tvalid    ( i2s_tx_axis_tvalid     ),
    .o_axis_tx_tdata     ( i2s_tx_axis_tdata      ),
    .o_axis_tx_tlast     ( i2s_tx_axis_tlast      ),
    .o_axis_tx_tuser     ( i2s_tx_axis_tuser      ),
    .o_axis_tx_tkeep     ( i2s_tx_axis_tkeep      ),
    .i_axis_tx_tready    ( i2s_tx_axis_tready     )
  );

//------------------------------------------------------------------------------
// I2S Instantiation IP NOT available to the public
//------------------------------------------------------------------------------
/*  
  logic I2S_LRCLK;
  logic I2S_SDATA;

  NV_i2s_fpga u_i2s (
    .ahc2client_paddr   ( apb_paddr          ), //|< i
    .ahc2client_penable ( apb_penable        ), //|< i
    .ahc2client_pprot   ( 3'b0               ), //|< i
    .ahc2client_psel    ( apb_psel[7]        ), //|< i
    .ahc2client_pwdata  ( apb_pwdata         ), //|< i
    .ahc2client_pwrite  ( apb_pwrite         ), //|< i
    .ahub_clk           ( apb_clk            ), //|< i
    .ape_aperstn        ( !apb_rst           ), //|< i
    .i2s_clk            ( i2s_clk_int        ), //|< i
    .i2s_lrck_in        ( 1'b0               ), //|< i  UNUSED IN MMODE
    .i2s_rx_axis_tready ( 1'b0               ), //|< i  UNUSED IN TX
    .i2s_sdata_in       ( 1'b0               ), //|< i  UNUSED IN TX
    .i2s_tx_axis_tdata  ( i2s_tx_axis_tdata  ), //|< i
    .i2s_tx_axis_tvalid ( i2s_tx_axis_tvalid ), //|< i
    .client2ahc_prdata  ( apb_prdata[7]      ), //|> o
    .client2ahc_pready  ( apb_pready[7]      ), //|> o
    .client2ahc_pslverr ( apb_pserr[7]       ), // ReLingo_waive_line:temp:RL02
    .i2s_lrck_oen       (                    ), //|> o
    .i2s_lrck_out       ( I2S_LRCLK          ), //|> o
    .i2s_rx_axis_tdata  (                    ), //|> o  UNUSED IN TX
    .i2s_rx_axis_tvalid (                    ), //|> o  UNUSED IN TX
    .i2s_sdata_oen      (                    ), //|> o
    .i2s_sdata_out      ( I2S_SDATA          ), //|> o
    .i2s_tx_axis_tready ( i2s_tx_axis_tready )  //|> o
  );
*/
//------------------------------------------------------------------------------
// Camera LVDS Data
//------------------------------------------------------------------------------

  logic                       sif_clk;
  logic [`SENSOR_IF_INST-1:0] sif_rx_axis_tvalid;
  logic [`SENSOR_IF_INST-1:0] sif_rx_axis_tlast;
  logic [`DATAPATH_WIDTH-1:0] sif_rx_axis_tdata [`SENSOR_IF_INST];
  logic [`DATAKEEP_WIDTH-1:0] sif_rx_axis_tkeep [`SENSOR_IF_INST];
  logic [`DATAUSER_WIDTH-1:0] sif_rx_axis_tuser [`SENSOR_IF_INST];
  logic [`SENSOR_IF_INST-1:0] sif_rx_axis_tready;

  assign sif_clk = hif_clk;

  logic [`SENSOR_IF_INST-1:0] cam_drdy_i;

  assign CAM_DRDY = &cam_drdy_i;

  generate
    for (i=0; i<`SENSOR_IF_INST; i++) begin: cam_sensor_rcvr

      cam_rcvr u_cam_rcvr (
        // LVDS PAD IO
        .i_rx_sclk     ( apb_clk                ),
        .i_rx_dclk     ( CAM_DCLK           [i] ), // 500MHz lvds data clock
        .i_rx_data     ( CAM_DATA           [i] ), // 500MHz ddr lvds data (1000 Mbps)
        .o_rx_drdy     ( cam_drdy_i         [i] ),
        // clock and reset
        .i_pclk        ( sif_clk                ),
        .i_prst        ( sif_rst                ),
        // Double ECC Detected
        .o_phy_err_det (                        ),
        // Frame header info
        // User AXIS Interface
        .o_axis_tvalid ( sif_rx_axis_tvalid [i] ),
        .o_axis_tlast  ( sif_rx_axis_tlast  [i] ),
        .o_axis_tdata  ( sif_rx_axis_tdata  [i] ),
        .o_axis_tkeep  ( sif_rx_axis_tkeep  [i] ),
        .o_axis_tuser  ( sif_rx_axis_tuser  [i] ),
        .i_axis_tready ( sif_rx_axis_tready [i] ),

        .i_apb_clk     ( apb_clk                ),
        .i_apb_rst     ( apb_rst                ),
        .i_apb_sel     ( apb_psel         [i+4] ),
        .i_apb_enable  ( apb_penable            ),
        .i_apb_addr    ( apb_paddr              ),
        .i_apb_wdata   ( apb_pwdata             ),
        .i_apb_write   ( apb_pwrite             ),
        .o_apb_ready   ( apb_pready       [i+4] ),
        .o_apb_rdata   ( apb_prdata       [i+4] ),
        .o_apb_serr    ( apb_pserr        [i+4] )
     );

    end
  endgenerate


//-------------------------------------------------------------------------
// VSYNC
//-------------------------------------------------------------------------
  logic [7:0]  gpio_mux_en;
  logic        vsync;

  vsync_gen u_vsync_gen (
    .i_clk           ( ptp_clk           ),
    .i_rst           ( ptp_rst           ),

    .i_apb_clk       ( apb_clk           ),
    .i_apb_rst       ( apb_rst           ),
    .i_apb_sel       ( apb_psel      [6] ),
    .i_apb_enable    ( apb_penable       ),
    .i_apb_addr      ( apb_paddr         ),
    .i_apb_wdata     ( apb_pwdata        ),
    .i_apb_write     ( apb_pwrite        ),
    .o_apb_ready     ( apb_pready    [6] ),
    .o_apb_rdata     ( apb_prdata    [6] ),
    .o_apb_serr      ( apb_pserr     [6] ),

    .i_pps           ( sys_pps           ),
    .i_ptp_nanosec   ( ptp_nsec          ),

    .o_vsync_strb    ( vsync             ),
    .o_gpio_mux_en   ( gpio_mux_en       )
  );

  logic [1:0] synth_lock_sync;
  always_ff @ (posedge hif_clk) begin
    if (hif_rst) begin
      synth_lock_sync <= 'd0;
    end else begin
      synth_lock_sync[0] <= CLK_SYNTH_LOCKED;
      synth_lock_sync[1] <= synth_lock_sync[0];
    end
  end

  ///////////////////////////////////////
  //MUX DEBUG SIGNALS INTO GPIO
  ///////////////////////////////////////
  logic [15:0] gpio_out;
  logic [15:0] gpio_tri;

  for (i=0; i<3; i++) begin
    assign gpio_tri[i] = gpio_out[i];
  end
  for (i=8; i<12; i++) begin
    assign gpio_tri[i] = gpio_out[i];
  end

  assign gpio_tri[3]  = gpio_mux_en[3] ? 1'b1               : gpio_out[3]; //VCC
  assign gpio_tri[4]  = gpio_mux_en[3] ? i2s_mclk_ext       : gpio_out[4];
  assign gpio_tri[5]  = gpio_mux_en[3] ? I2S_LRCLK          : gpio_out[5];
  assign gpio_tri[6]  = gpio_mux_en[3] ? i2s_clk_ext        : gpio_out[6];
  assign gpio_tri[7]  = gpio_mux_en[3] ? I2S_SDATA          : gpio_out[7];

  assign gpio_tri[12] = gpio_mux_en[2] ? ptp_sensor_pll_locked : gpio_out[12];
  assign gpio_tri[13] = gpio_mux_en[2] ? synth_lock_sync[1]    : gpio_out[13];
  assign gpio_tri[14] = gpio_mux_en[1] ? vsync                 : gpio_out[14];
  assign gpio_tri[15] = gpio_mux_en[0] ? sys_pps_stretch       : gpio_out[15];

  assign GPIO = gpio_tri;

  ///////////////////////////////////////
  //CAM GPIO
  ///////////////////////////////////////
  logic [14:0] cam_gpio_out;
  logic [14:0] cam_gpio_tri;

  for (i=0; i<3; i++) begin
    assign cam_gpio_tri[i] = cam_gpio_out[i];
  end
  for (i=4; i<6; i++) begin
    assign cam_gpio_tri[i] = cam_gpio_out[i];
  end
  for (i=9; i<15; i++) begin
    assign cam_gpio_tri[i] = cam_gpio_out[i];
  end

  assign cam_gpio_tri[3] = gpio_mux_en[4] ? vsync          : cam_gpio_out[3];
  assign cam_gpio_tri[6] = gpio_mux_en[5] ? vsync          : cam_gpio_out[6];
  assign cam_gpio_tri[7] = gpio_mux_en[6] ? vsync          : cam_gpio_out[7];
  assign cam_gpio_tri[8] = gpio_mux_en[7] ? vsync          : cam_gpio_out[8];

  assign CAM_GPIO = cam_gpio_tri[14:0];

  //FIXME wjohn : Temporarily commenting out due to timing violations
  //temp_tlm u_temp_tlm_inst (
  //  .pll_clk_in   ( adc_clk  ),
  //  .aclk         ( apb_clk  ),
  //  .rst          ( sys_rst  ),
  //  .dtr_out_code ( temp_tlm )
  //);
  assign temp_tlm = '0;

  logic [15:0] sif_event;
  assign sif_event = {14'h0, sif_rx_axis_tlast[1:0]};

//------------------------------------------------------------------------------
// HOLOLINK Top Instantiation
//------------------------------------------------------------------------------

  HOLOLINK_top #(
    .BUILD_REV         ( BUILD_REV          )
  ) u_hololink_top (
    .i_sys_rst         ( sys_rst            ),
  //------------------------------------------------------------------------------
  // User Reg Interface
  //------------------------------------------------------------------------------
  // Control Plane
    .i_apb_clk         ( apb_clk            ),
    .o_apb_rst         ( apb_rst            ),
    // APB Register Interface
    .o_apb_psel        ( apb_psel           ),
    .o_apb_penable     ( apb_penable        ),
    .o_apb_paddr       ( apb_paddr          ),
    .o_apb_pwdata      ( apb_pwdata         ),
    .o_apb_pwrite      ( apb_pwrite         ),
    .i_apb_pready      ( apb_pready         ),
    .i_apb_prdata      ( apb_prdata         ),
    .i_apb_pserr       ( apb_pserr          ),
  //------------------------------------------------------------------------------
  // User Auto Initialization Interface
  //------------------------------------------------------------------------------
    .o_init_done       ( init_done          ),
  //------------------------------------------------------------------------------
  // Sensor IF
  //------------------------------------------------------------------------------
  // Sensor Interface Clock and Reset
    .i_sif_clk         ( sif_clk            ),
    .o_sif_rst         ( sif_rst            ),
    // Sensor Rx Streaming Interface
    .i_sif_axis_tvalid ( sif_rx_axis_tvalid ),
    .i_sif_axis_tlast  ( sif_rx_axis_tlast  ),
    .i_sif_axis_tdata  ( sif_rx_axis_tdata  ),
    .i_sif_axis_tkeep  ( sif_rx_axis_tkeep  ),
    .i_sif_axis_tuser  ( sif_rx_axis_tuser  ),
    .o_sif_axis_tready ( sif_rx_axis_tready ),
    // Sensor Tx Streaming Interface (Unimplemented)
    .o_sif_axis_tvalid ( sif_tx_axis_tvalid ),
    .o_sif_axis_tlast  ( sif_tx_axis_tlast  ),
    .o_sif_axis_tdata  ( sif_tx_axis_tdata  ),
    .o_sif_axis_tkeep  ( sif_tx_axis_tkeep  ),
    .o_sif_axis_tuser  ( sif_tx_axis_tuser  ),
    .i_sif_axis_tready ( sif_tx_axis_tready ),
    // Sensor Event
    .i_sif_event       ( sif_event          ),
  //------------------------------------------------------------------------------
  // Host IF
  //------------------------------------------------------------------------------
  // Host Interface Clock and Reset
    .i_hif_clk         ( hif_clk            ),
    .o_hif_rst         ( hif_rst            ),
    // Host Rx Interface
    .i_hif_axis_tvalid ( hif_rx_axis_tvalid ),
    .i_hif_axis_tlast  ( hif_rx_axis_tlast  ),
    .i_hif_axis_tdata  ( hif_rx_axis_tdata  ),
    .i_hif_axis_tkeep  ( hif_rx_axis_tkeep  ),
    .i_hif_axis_tuser  ( hif_rx_axis_tuser  ),
    .o_hif_axis_tready ( hif_rx_axis_tready ),
    // Host Tx Interface
    .o_hif_axis_tvalid ( hif_tx_axis_tvalid ),
    .o_hif_axis_tlast  ( hif_tx_axis_tlast  ),
    .o_hif_axis_tdata  ( hif_tx_axis_tdata  ),
    .o_hif_axis_tkeep  ( hif_tx_axis_tkeep  ),
    .o_hif_axis_tuser  ( hif_tx_axis_tuser  ),
    .i_hif_axis_tready ( hif_tx_axis_tready ),
  //------------------------------------------------------------------------------
  // Peripheral IF
  //------------------------------------------------------------------------------
    // SPI Interface, QSPI compatible
    .o_spi_csn         ( spi_csn            ),
    .o_spi_sck         ( spi_sck            ),
    .i_spi_sdio        ( spi_sdio_i         ),
    .o_spi_sdio        ( spi_sdio_o         ),
    .o_spi_oen         ( spi_oen            ),
    // I2C Interface
    .i_i2c_scl         ( i2c_scl            ),
    .i_i2c_sda         ( i2c_sda            ),
    .o_i2c_scl_en      ( i2c_scl_en         ),
    .o_i2c_sda_en      ( i2c_sda_en         ),
    // GPIO
    .o_gpio            ( {cam_gpio_out, gpio_out} ),
    .i_gpio            ( {CAM_GPIO, GPIO}   ),
  //------------------------------------------------------------------------------
  // sensor reset
  //------------------------------------------------------------------------------
    .o_sw_sys_rst      ( sw_sys_rst         ),
    .o_sw_sen_rst      ( sw_sen_rst         ),
  //------------------------------------------------------------------------------
  // PTP
  //------------------------------------------------------------------------------
    .i_ptp_clk         ( ptp_clk            ),
    .o_ptp_rst         ( ptp_rst            ),
    .o_ptp_sec         (                    ),
    .o_ptp_nanosec     ( ptp_nsec           ),
    .o_pps             ( sys_pps            )

  );

endmodule
