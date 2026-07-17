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

// Lattice Sensor Board Example Design
`include "HOLOLINK_def.svh"

module FPGA_top
  import HOLOLINK_pkg::*;
  import apb_pkg::*;
#(
  parameter BUILD_REV = 48'h0
)(
  input           RESET_N,  // board reset push button
  // 10GbE SFP
  input           ETH_REFCLK_P,
  input           ETH_REFCLK_N,
  input           ETH_RXD_P,
  input           ETH_RXD_N,
  output          ETH_TXD_P,
  output          ETH_TXD_N,

  // SFP Fiber Optics Disable    
  output          SFP_TX_DIS,

  // MIPI Interface
  inout   [1:0]   MIPI_CAM_CLK_P          ,
  inout   [1:0]   MIPI_CAM_CLK_N          ,
  inout   [3:0]   MIPI_CAM_DATA_P [1:0]   ,
  inout   [3:0]   MIPI_CAM_DATA_N [1:0]   ,

  // I2C Interfaces
  inout           CTRL_I2C_SCL,
  inout           CTRL_I2C_SDA,

  inout  [ 0:0]   CAM_I2C_SCL,
  inout  [ 0:0]   CAM_I2C_SDA,

  // SPI Flash Interfaces
  output          FLASH_SPI_MCSN,
  output          FLASH_SPI_MSCK,
  inout  [ 3:0]   FLASH_SPI_SDIO,
    
  // GPIO
  output [10:0] GPIO,
  // Camera power enable
  output        CAM_RST,
  // Camera clock
  output        CAM_MCLK,
  // POC
  output        POC_EN,
  input         POC_INT_L,
  inout         I2C_POC_SDA,
  inout         I2C_POC_SCL
);

//------------------------------------------------------------------------------
// Clock and Reset
//------------------------------------------------------------------------------

  logic [`SENSOR_RX_IF_INST-1:0] sw_sen_rst;
  logic                          sw_sys_rst;

  logic [`HOST_IF_INST  -1:0] usr_clk;     // pcs user clock out
  logic [`HOST_IF_INST  -1:0] usr_clk_rdy; // pcs user clock out ready
  logic                       usr_clk_locked;
  /* synthesis syn_keep=1 nomerge=""*/
  logic                       adc_clk;     // 50MHz ADC clock
  /* synthesis syn_keep=1 nomerge=""*/
  logic                       pcs_clk;     // 100-300 MHz PCS calibration clock
  /* synthesis syn_keep=1 nomerge=""*/
  logic                       apb_clk;     // ctrl plane clock
  /* synthesis syn_keep=1 nomerge=""*/
  logic                       hif_clk;     // data plane clock
  logic                       sys_rst;     // system active high reset
  logic                       apb_rst;     // apb active high reset
  logic                       hif_rst;     // host interface active high reset
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_rst;     // sensor interface active high reset
  logic                       pcs_rst_n;   // ethernet pcs active low reset
  /* synthesis syn_keep=1 nomerge=""*/
  logic        ptp_clk;
  logic        ptp_rst;
  logic [31:0] ptp_nsec;
  logic        vsync;
  logic [47:0] ptp_sec;
  logic        ptp_cam_clk;
  logic [7:0]  gpio_mux_en;
  logic [15:0] gpio_out;

  assign usr_clk_locked = &usr_clk_rdy;

  logic        sys_pps;
  logic        sys_pps_stretch;
  logic [17:0] timer_cnt;
  logic        timer_done;

  clk_n_rst u_clk_n_rst (
    .i_refclk      ( usr_clk [0]    ), // pcs user clock output
    .i_locked      ( usr_clk_locked ), // pcs user clock locked

    .o_mipi_clk    ( mipi_clk       ), // 27.043MHz clock for MIPI
    .o_pcs_clk     ( pcs_clk        ), // pcs calibration clock
    .o_hif_clk     ( hif_clk        ), // host interface clock
    .o_apb_clk     ( apb_clk        ), // apb interface clock
    .o_ptp_clk     ( ptp_clk        ), // ptp interface clock

    .i_ptp_nsec    ( ptp_nsec       ),
    .o_ptp_cam_clk ( ptp_cam_clk    ), //ptp 24MHz clock

    .i_pb_rst_n    ( RESET_N        ), // asynchronous active low board reset
    .i_sw_rst      ( sw_sys_rst     ), // software controlled active high reset

    .o_sys_rst     ( sys_rst        ), // system active high reset
    .o_pcs_rst_n   ( pcs_rst_n      )  // ethernet pcs active low reset
  );


//------------------------------------------------------------------------------
// FPGA Board Control
//------------------------------------------------------------------------------

  logic                     ptp_cam_24m_clk /* synthesis syn_keep=1 nomerge=""*/;

  // Fiber Optics Disable Input Jumper
  assign SFP_TX_DIS       = 1'b0;
  // Deserializer IC MAX96716 power down
  assign CAM_RST          = ~sw_sen_rst[0]; // power down active low
  // Cameras enable 
  assign POC_EN           = gpio_out[0];

  // MIPI clock is 27.043MHz
  ODDRX1 u_mipi_clk (
    .D0   ( 1'b1            ),
    .D1   ( 1'b0            ),
    .SCLK ( mipi_clk        ), // 27.043MHz clock
    .RST  ( !usr_clk_locked ),
    .Q    ( CAM_MCLK        )  // 27.043MHz clock
  );

//------------------------------------------------------------------------------
// PPS
//------------------------------------------------------------------------------
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

  logic init_done;

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
  logic [31         :0] apb_prdata [`REG_INST-1:0];
  logic [`REG_INST-1:0] apb_pserr;

  //Tie off unused REG_INST APB signals
  genvar i;

  //for(i=7; i<`REG_INST; i++) begin
  //  assign apb_pready[i] = '0;
  //  assign apb_prdata[i] = '0;
  //  assign apb_pserr [i] = '0;
  //end

//------------------------------------------------------------------------------
// Lattice 10GbE Host Interface
//------------------------------------------------------------------------------

  logic [`HOST_IF_INST-1  :0] hif_tx_axis_tvalid;
  logic [`HOST_IF_INST-1  :0] hif_tx_axis_tlast;
  logic [`HOST_WIDTH  -1  :0] hif_tx_axis_tdata [`HOST_IF_INST-1:0];
  logic [`HOSTKEEP_WIDTH-1:0] hif_tx_axis_tkeep [`HOST_IF_INST-1:0];
  logic [`HOSTUSER_WIDTH-1:0] hif_tx_axis_tuser [`HOST_IF_INST-1:0];
  logic [`HOST_IF_INST-1  :0] hif_tx_axis_tready;

  logic [`HOST_IF_INST-1  :0] hif_rx_axis_tvalid;
  logic [`HOST_IF_INST-1  :0] hif_rx_axis_tlast;
  logic [`HOST_WIDTH  -1  :0] hif_rx_axis_tdata [`HOST_IF_INST-1:0];
  logic [`HOSTKEEP_WIDTH-1:0] hif_rx_axis_tkeep [`HOST_IF_INST-1:0];
  logic [`HOSTUSER_WIDTH-1:0] hif_rx_axis_tuser [`HOST_IF_INST-1:0];
  logic [`HOST_IF_INST-1  :0] hif_rx_axis_tready;

  generate
    for (i=0; i<`HOST_IF_INST; i++) begin: ethernet_10gb

      eth_10gb_top #(
        .ID               ( 0                         )
      ) u_10gbe (
        // clock and reset
        .i_refclk_p       ( ETH_REFCLK_P              ),
        .i_refclk_n       ( ETH_REFCLK_N              ),
        // SERDES IO
        .i_pad_rx_p       ( ETH_RXD_P                 ),
        .i_pad_rx_n       ( ETH_RXD_N                 ),
        .o_pad_tx_p       ( ETH_TXD_P                 ),
        .o_pad_tx_n       ( ETH_TXD_N                 ),
        // PCS clock
        .i_pcs_clk        ( pcs_clk                   ),
        .i_pcs_rst_n      ( pcs_rst_n                 ),
        .i_sys_rst_n      ( ~sys_rst                  ),
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
  //logic       ctrl_spi_miso_sync;

  data_sync    #(
    .DATA_WIDTH ( 4                  )
  ) spi_glitch_filter (
    .clk        ( hif_clk            ),
    .rst_n      (~hif_rst            ),
    .sync_in    (FLASH_SPI_SDIO      ),
    .sync_out   (flsh_spi_sdio_sync  )
  );

  // SPI Interface, QSPI compatible
  logic [`SPI_INST-1:0] spi_csn;
  logic [`SPI_INST-1:0] spi_sck;
  logic [3          :0] spi_sdio_i [`SPI_INST-1:0];
  logic [3          :0] spi_sdio_o [`SPI_INST-1:0];
  logic [`SPI_INST-1:0] spi_oen;

  //// Regular SPI
  //assign CTRL_SPI_MSCK = spi_sck   [0];
  //assign CTRL_SPI_MCSN = spi_csn   [0];
  //assign CTRL_SPI_MOSI = spi_sdio_o[0];
  //assign spi_sdio_i[0] = {2'b0, ctrl_spi_miso_sync, 1'b0};
  // QSPI Flash
  assign FLASH_SPI_MSCK = spi_sck  [0];
  assign FLASH_SPI_MCSN = spi_csn  [0];
  assign FLASH_SPI_SDIO = spi_oen  [0] ? spi_sdio_o[0] : 4'hz;
  assign spi_sdio_i[0]  = flsh_spi_sdio_sync;
  // Adding a glitch filter, but can be removed if IO pads itself provides glitch filtering
  logic                 ctrl_i2c_scl_sync;
  logic                 ctrl_i2c_sda_sync;
  logic [0:0]           cam_i2c_scl_sync;
  logic [0:0]           cam_i2c_sda_sync;
  logic                 poc_i2c_scl_sync;
  logic                 poc_i2c_sda_sync;

  glitch_filter  #(
    .DATA_WIDTH   ( 6                                     ),
    .RESET_VALUE  ( 1'b1                                  ),
    .FILTER_DEPTH ( 8                                     )
  ) i2c_glitch_filter (
    .clk          ( hif_clk                               ),
    .rst_n        (~hif_rst                               ),
    .sync_in      ({CTRL_I2C_SDA, CTRL_I2C_SCL,
                    CAM_I2C_SDA , CAM_I2C_SCL,
                    I2C_POC_SDA , I2C_POC_SCL            }),
    .sync_out     ({ctrl_i2c_sda_sync, ctrl_i2c_scl_sync,
                    cam_i2c_sda_sync , cam_i2c_scl_sync,
                    poc_i2c_sda_sync , poc_i2c_scl_sync  })
  );

  logic [`I2C_INST-1:0] i2c_scl;
  logic [`I2C_INST-1:0] i2c_sda;
  logic [`I2C_INST-1:0] i2c_scl_en;
  logic [`I2C_INST-1:0] i2c_sda_en;

  assign i2c_scl[0]   = i2c_scl_en[0] ? ctrl_i2c_scl_sync : 1'b0;
  assign i2c_sda[0]   = i2c_sda_en[0] ? ctrl_i2c_sda_sync : 1'b0;
  assign CTRL_I2C_SCL = i2c_scl_en[0] ? 1'bz : 1'b0;
  assign CTRL_I2C_SDA = i2c_sda_en[0] ? 1'bz : 1'b0;

  
  assign i2c_scl[1]     = i2c_scl_en[1] ? cam_i2c_scl_sync[0] : 1'b0;
  assign i2c_sda[1]     = i2c_sda_en[1] ? cam_i2c_sda_sync[0] : 1'b0;
  assign CAM_I2C_SCL[0] = i2c_scl_en[1] ? 1'bz : 1'b0;
  assign CAM_I2C_SDA[0] = i2c_sda_en[1] ? 1'bz : 1'b0;

  assign i2c_scl[2]     = i2c_scl_en[2] ? poc_i2c_scl_sync : 1'b0;
  assign i2c_sda[2]     = i2c_sda_en[2] ? poc_i2c_sda_sync : 1'b0;
  assign I2C_POC_SCL    = i2c_scl_en[2] ? 1'bz : 1'b0;
  assign I2C_POC_SDA    = i2c_sda_en[2] ? 1'bz : 1'b0;

  
//------------------------------------------------------------------------------
// Camera LVDS Data
//------------------------------------------------------------------------------

  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_clk;
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_axis_tvalid;
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_axis_tlast;
  logic [`DATAPATH_WIDTH-1:0] sif_rx_axis_tdata [`SENSOR_RX_IF_INST-1:0];
  logic [`DATAKEEP_WIDTH-1:0] sif_rx_axis_tkeep [`SENSOR_RX_IF_INST-1:0];
  logic [`DATAUSER_WIDTH-1:0] sif_rx_axis_tuser [`SENSOR_RX_IF_INST-1:0];
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_axis_tready;


  logic [15:0] sif_event;
  
  logic [`SENSOR_RX_IF_INST-1:0] sof, sof_done;

  // Generate SOF logic for each SIF RX channel on its respective clock domain
  generate
    for (i=0; i<`SENSOR_RX_IF_INST; i++) begin: sof_gen
      always_ff @ (posedge sif_rx_clk[i]) begin
        if (sif_rx_rst[i]) begin
          sof[i] <= 1'b0;
          sof_done[i] <= 1'b0;
        end else begin
          sof[i] <= sif_rx_axis_tvalid[i] && ~sof_done[i];
          if(sif_rx_axis_tvalid[i]) sof_done[i] <= 1'b1;
          if(sif_rx_axis_tlast[i]) sof_done[i] <= 1'b0;
        end
      end
    end
  endgenerate
  
  assign sif_event = {14'h0, sif_rx_axis_tlast[1:0]};

  generate
    for (i=0; i<2; i++) begin: cam_sensor_rcvr

      assign sif_rx_clk[i] = hif_clk;

      mipi_cam_rcvr u_cam_rcvr (
        // LVDS PAD IO
        .i_mipi_sync_clk    ( pcs_clk                ),
        .i_rst_n            ( RESET_N                ),
        .i_sclk             ( sif_rx_clk[i]          ),
        .i_srst             ( sif_rx_rst[i]          ),
        .i_pll_locked       ( usr_clk_locked         ),
        .i_apb_clk          ( apb_clk                ),
        .i_apb_rst          ( apb_rst                ),
        .i_apb_sel          ( apb_psel         [i+4] ),
        .i_apb_enable       ( apb_penable            ),
        .i_apb_addr         ( apb_paddr              ),
        .i_apb_wdata        ( apb_pwdata             ),
        .i_apb_write        ( apb_pwrite             ),
        .o_apb_ready        ( apb_pready       [i+4] ),
        .o_apb_rdata        ( apb_prdata       [i+4] ),
        .o_apb_serr         ( apb_pserr        [i+4] ),
        // User AXIS Interface
        .o_axis_tvalid      ( sif_rx_axis_tvalid [i] ),
        .o_axis_tlast       ( sif_rx_axis_tlast  [i] ),
        .o_axis_tdata       ( sif_rx_axis_tdata  [i] ),
        .o_axis_tkeep       ( sif_rx_axis_tkeep  [i] ),
        .o_axis_tuser       ( sif_rx_axis_tuser  [i] ),
        .o_axis_tidx        (                        ),
        .i_axis_tready      ( sif_rx_axis_tready [i] ), 
        .mipi_cam_clk_n_io  ( MIPI_CAM_CLK_N     [i] ),
        .mipi_cam_clk_p_io  ( MIPI_CAM_CLK_P     [i] ),
        .mipi_cam_data_n_io ( MIPI_CAM_DATA_N    [i] ),
        .mipi_cam_data_p_io ( MIPI_CAM_DATA_P    [i] )
     );

    end
  endgenerate


//------------------------------------------------------------------------------
// Apb Interconnect
//------------------------------------------------------------------------------

apb_m2s sw_apb_m2s [1];
apb_s2m sw_apb_s2m [1];

apb_m2s s_apb_m2s [8];
apb_s2m s_apb_s2m [8];

assign sw_apb_m2s [0].psel   = apb_psel     [7];
assign sw_apb_m2s [0].penable= apb_penable     ;
assign sw_apb_m2s [0].paddr  = apb_paddr       ;
assign sw_apb_m2s [0].pwdata = apb_pwdata      ;
assign sw_apb_m2s [0].pwrite = apb_pwrite      ;
assign apb_pready [7]        = sw_apb_s2m[0].pready; 
assign apb_prdata [7]        = sw_apb_s2m[0].prdata; 
assign apb_pserr  [7]        = sw_apb_s2m[0].pserr; 

// PERI apb switch
apb_switch #(
  .N_MPORT         ( 1                ),
  .N_SPORT         ( 8                ),
  .W_OFSET         ( 20               ),
  .W_SW            ( 12               ),
  .MERGE_COMPLETER_SIG ( 0            )
) u_apb_switch     (
  .i_apb_clk       ( apb_clk          ),
  .i_apb_reset     ( apb_rst          ),
  .i_apb_m2s       ( sw_apb_m2s[0:0]  ),
  .o_apb_s2m       ( sw_apb_s2m[0:0]  ),
  .o_apb_m2s       ( s_apb_m2s        ),
  .i_apb_s2m       ( s_apb_s2m        ),
  .i_apb_timeout   ( 1'b0             )
);

//------------------------------------------------------------------------------
// APB Interface for Index 6 (VSYNC)
//------------------------------------------------------------------------------

  apb_m2s vsync_apb_m2s;
  apb_s2m vsync_apb_s2m;

  assign vsync_apb_m2s.psel    = apb_psel[6];
  assign vsync_apb_m2s.penable = apb_penable;
  assign vsync_apb_m2s.paddr   = apb_paddr;
  assign vsync_apb_m2s.pwdata  = apb_pwdata;
  assign vsync_apb_m2s.pwrite  = apb_pwrite;

  assign apb_prdata[6] = vsync_apb_s2m.prdata;
  assign apb_pready[6] = vsync_apb_s2m.pready;
  assign apb_pserr[6]  = vsync_apb_s2m.pserr;

//-------------------------------------------------------------------------
// VSYNC
//-------------------------------------------------------------------------
  vsync_gen u_vsync_gen (
    .i_clk           ( ptp_clk           ),
    .i_rst           ( ptp_rst           ),

    .i_apb_clk       ( apb_clk                    ),
    .i_apb_rst       ( apb_rst                    ),
    .i_apb_sel       ( vsync_apb_m2s.psel         ),  // Use switch signal (only 0x000-0x7FF)
    .i_apb_enable    ( vsync_apb_m2s.penable      ),  // Use switch signal
    .i_apb_addr      ( vsync_apb_m2s.paddr        ),  // Use switch signal
    .i_apb_wdata     ( vsync_apb_m2s.pwdata       ),  // Use switch signal
    .i_apb_write     ( vsync_apb_m2s.pwrite       ),  // Use switch signal
    .o_apb_ready     ( vsync_apb_s2m.pready       ),  // Connect to switch response
    .o_apb_rdata     ( vsync_apb_s2m.prdata       ),  // Connect to switch response
    .o_apb_serr      ( vsync_apb_s2m.pserr        ),  // Connect to switch response

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
      synth_lock_sync[0] <= '0;
      synth_lock_sync[1] <= synth_lock_sync[0];
    end
  end 

  logic poc_int_l_sync;
  data_sync #(
    .DATA_WIDTH  ( 1    ),
    .RESET_VALUE ( 1'b1 )
  ) poc_int_l_synchronizer (
    .clk         ( hif_clk        ),
    .rst_n       (~hif_rst        ),
    .sync_in     ( POC_INT_L      ),
    .sync_out    ( poc_int_l_sync )
  );

  ///////////////////////////////////////
  //MUX DEBUG SIGNALS INTO GPIO
  ///////////////////////////////////////
  logic [15:0] gpio_in;    // Input read values
  logic [15:0] gpio_dir;   // 0=output, 1=input
  logic [15:0] gpio_tri;

  // Read inputs (separate from driving logic)
  assign gpio_in[0]     = gpio_out[0];
  assign gpio_in[1]     = poc_int_l_sync;
  assign gpio_in[15:2]  = '0;

  // Generate blocks for pass-through assignments
  generate
    for (i=0; i<4; i++) begin: gpio_passthru_1
      assign gpio_tri[i] = gpio_dir[i] ? 1'bz : gpio_out[i];
    end
  endgenerate

  generate
    for (i=4; i<8; i++) begin: gpio_passthru_2
      assign gpio_tri[i] = gpio_dir[i] ? 1'bz : gpio_out[i];
    end
  endgenerate

  generate
    for (i=9; i<11; i++) begin: gpio_passthru_3
      assign gpio_tri[i] = gpio_dir[i] ? 1'bz : gpio_out[i];
    end
  endgenerate

  // Special function pins with mux
  assign gpio_tri[8]  = gpio_dir[8]  ? 1'bz : (gpio_mux_en[4] ? vsync              : gpio_out[8]);
  assign gpio_tri[13] = gpio_dir[13] ? 1'bz : (gpio_mux_en[2] ? synth_lock_sync[1] : gpio_out[13]);
  assign gpio_tri[14] = gpio_dir[14] ? 1'bz : (gpio_mux_en[1] ? vsync              : gpio_out[14]);
  assign gpio_tri[15] = gpio_dir[15] ? 1'bz : (gpio_mux_en[0] ? sys_pps_stretch    : gpio_out[15]);

  assign GPIO = gpio_tri[10:0];


//------------------------------------------------------------------------------
// HOLOLINK Top Instantiation
//------------------------------------------------------------------------------

  /*
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_axis_tvalid;
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_axis_tlast;
  logic [`DATAPATH_WIDTH-1:0] sif_tx_axis_tdata [`SENSOR_TX_IF_INST];
  logic [`DATAKEEP_WIDTH-1:0] sif_tx_axis_tkeep [`SENSOR_TX_IF_INST];
  logic [`DATAUSER_WIDTH-1:0] sif_tx_axis_tuser [`SENSOR_TX_IF_INST];
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_axis_tready;
  */

  logic [3:0] dummy;

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
`ifndef ENUM_EEPROM
    .i_mac_addr        ( '{`MAC_ADDR}       ),
    .i_board_sn        ( '{`BOARD_SN}       ),
    .i_enum_vld        ( 1'b1               ),
`endif
    .o_init_done       ( init_done          ),
  //------------------------------------------------------------------------------
  // Sensor IF
  //------------------------------------------------------------------------------
  // Sensor Interface Clock and Reset
    .i_sif_rx_clk      ( sif_rx_clk         ),
    .o_sif_rx_rst      ( sif_rx_rst         ),
    // Sensor Rx Streaming Interface
    .i_sif_axis_tvalid ( sif_rx_axis_tvalid ),
    .i_sif_axis_tlast  ( sif_rx_axis_tlast  ),
    .i_sif_axis_tdata  ( sif_rx_axis_tdata  ),
    .i_sif_axis_tkeep  ( sif_rx_axis_tkeep  ),
    .i_sif_axis_tuser  ( sif_rx_axis_tuser  ),
    .o_sif_axis_tready ( sif_rx_axis_tready ),
    // Sensor Tx Streaming Interface (Unimplemented)
/*
    .i_sif_tx_clk      ( sif_tx_clk         ),
    .o_sif_tx_rst      ( sif_tx_rst         ),
    .o_sif_axis_tvalid ( sif_tx_axis_tvalid ),
    .o_sif_axis_tlast  ( sif_tx_axis_tlast  ),
    .o_sif_axis_tdata  ( sif_tx_axis_tdata  ),
    .o_sif_axis_tkeep  ( sif_tx_axis_tkeep  ),
    .o_sif_axis_tuser  ( sif_tx_axis_tuser  ),
    .i_sif_axis_tready ( sif_tx_axis_tready ),
*/
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
    .o_gpio            ( gpio_out           ),
    .o_gpio_dir        ( gpio_dir           ),
    .i_gpio            ( gpio_in            ),
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
    .o_ptp_sec         ( ptp_sec            ),
    .o_ptp_nanosec     ( ptp_nsec           ),
    .o_pps             ( sys_pps            )
  
  );


endmodule
