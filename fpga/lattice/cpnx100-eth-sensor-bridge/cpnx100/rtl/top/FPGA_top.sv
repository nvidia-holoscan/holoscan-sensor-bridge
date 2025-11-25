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

  logic [31:0]                sw_sen_rst;
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
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_rst;     // sensor Rx interface active high reset
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_rst;     // sensor Tx interface active high reset
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
  logic                     i2s_ref_clk;
  /* synthesis syn_keep=1 nomerge=""*/

  logic                     uart_tx;
  logic                     uart_rx;  

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
    // .o_i2s_clk_ext   ( i2s_clk_ext    ),
    // .o_i2s_mclk_ext  ( i2s_mclk_ext   ),
    .o_i2s_ref_clk   ( i2s_ref_clk    ),

    .i_pb_rst_n      ( RESET_N        ), // asynchronous active low board reset
    .i_sw_rst        ( sw_sys_rst     ), // software controlled active high reset

    .o_sys_rst       ( sys_rst        ), // system active high reset
    .o_pcs_rst_n     ( pcs_rst_n      )  // ethernet pcs active low reset
  );

  assign SWRST_N = '1; //!sys_rst;
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
  logic [31         :0] apb_prdata [`REG_INST-1:0];
  logic [`REG_INST-1:0] apb_pserr;


  //Tie off unused REG_INST APB signals
  genvar i;

  // Note: All APB indices (0-7) are now in use since REG_INST=8

//------------------------------------------------------------------------------
// Lattice 10GbE Host Interface
//------------------------------------------------------------------------------

  logic [`HOST_IF_INST-1  :0] hif_tx_axis_tvalid;
  logic [`HOST_IF_INST-1  :0] hif_tx_axis_tlast;
  logic [`HOST_WIDTH-1    :0] hif_tx_axis_tdata [`HOST_IF_INST-1:0];
  logic [`HOSTKEEP_WIDTH-1:0] hif_tx_axis_tkeep [`HOST_IF_INST-1:0];
  logic [`HOSTUSER_WIDTH-1:0] hif_tx_axis_tuser [`HOST_IF_INST-1:0];
  logic [`HOST_IF_INST-1  :0] hif_tx_axis_tready;

  logic [`HOST_IF_INST-1  :0] hif_rx_axis_tvalid;
  logic [`HOST_IF_INST-1  :0] hif_rx_axis_tlast;
  logic [`HOST_WIDTH-1    :0] hif_rx_axis_tdata [`HOST_IF_INST-1:0];
  logic [`HOSTKEEP_WIDTH-1:0] hif_rx_axis_tkeep [`HOST_IF_INST-1:0];
  logic [`HOSTUSER_WIDTH-1:0] hif_rx_axis_tuser [`HOST_IF_INST-1:0];
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
  logic [3          :0] spi_sdio_i [`SPI_INST-1:0];
  logic [3          :0] spi_sdio_o [`SPI_INST-1:0];
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
// UART Peripheral
//------------------------------------------------------------------------------
  logic uart_rx_sync;

  glitch_filter #(
    .DATA_WIDTH   ( 1    ),
    .RESET_VALUE  ( 1'b1 ),
    .FILTER_DEPTH ( 8    )
  ) uart_sync (
    .clk      ( hif_clk      ),
    .rst_n    ( ~hif_rst     ),
    .sync_in  ( uart_rx      ),
    .sync_out ( uart_rx_sync )
  );

//------------------------------------------------------------------------------
// Delayed I/O Interface
//------------------------------------------------------------------------------

  logic delayed_io_pin;
  logic delayed_io_busy;

  // APB interface structs for delayed I/O (driven by APB switch below)
  apb_m2s delayed_io_apb_m2s;
  apb_s2m delayed_io_apb_s2m;

  FPGA_delayed_io #(
    .NUM_OUTPUTS  ( 1 ),
    .W_OFSET      ( 8 )
  ) u_delayed_io (
    .cmd_clk      ( apb_clk              ),
    .cmd_rst_n    ( ~apb_rst             ),
    .proc_clk     ( apb_clk              ),  // Using same clock for simplicity
    .proc_rst_n   ( ~apb_rst             ),  // Using same reset for simplicity
    // APB interface
    .i_apb_m2s    ( delayed_io_apb_m2s   ),
    .o_apb_s2m    ( delayed_io_apb_s2m   ),
    // Delayed I/O outputs
    .o_io_pins    ( delayed_io_pin       ),
    .o_io_busy    ( delayed_io_busy      )
  );

//------------------------------------------------------------------------------
// I2S IF
//------------------------------------------------------------------------------
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_clk;
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_rst;
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_axis_tvalid;
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_axis_tlast;
  logic [`DATAPATH_WIDTH-1:0]    sif_tx_axis_tdata [`SENSOR_TX_IF_INST-1:0];
  logic [`DATAKEEP_WIDTH-1:0]    sif_tx_axis_tkeep [`SENSOR_TX_IF_INST-1:0];
  logic [`DATAUSER_WIDTH-1:0]    sif_tx_axis_tuser [`SENSOR_TX_IF_INST-1:0];
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_axis_tready;

  logic                       i2s_tx_axis_tvalid;
  logic                       i2s_tx_axis_tlast;
  logic [31:0]                i2s_tx_axis_tdata;
  logic [3:0]                 i2s_tx_axis_tkeep;
  logic                       i2s_tx_axis_tuser;
  logic                       i2s_tx_axis_tready;

  assign sif_tx_clk[0] = apb_clk;

//------------------------------------------------------------------------------
// I2S Instantiation
//------------------------------------------------------------------------------
  logic I2S_LRCLK;
  logic I2S_SDATA;

  // APB interface structs for both modules
  apb_m2s i2s_apb_m2s;
  apb_s2m i2s_apb_s2m;
  apb_m2s switch_apb6_m2s [1];
  apb_s2m switch_apb6_s2m [1];
  
  // APB switch on index 6 - shared with vsync module (matches original hardware)
  assign switch_apb6_m2s[0].psel    = apb_psel[6];
  assign switch_apb6_m2s[0].penable = apb_penable;
  assign switch_apb6_m2s[0].paddr   = apb_paddr;
  assign switch_apb6_m2s[0].pwdata  = apb_pwdata;
  assign switch_apb6_m2s[0].pwrite  = apb_pwrite;
  
  // Connect switch output back to main APB bus (index 6)
  assign apb_prdata[6] = switch_apb6_s2m[0].prdata;
  assign apb_pready[6] = switch_apb6_s2m[0].pready;
  assign apb_pserr[6]  = switch_apb6_s2m[0].pserr;

  // I2S gets direct connection to APB index 7 (original location)
  assign i2s_apb_m2s.psel    = apb_psel[7];
  assign i2s_apb_m2s.penable = apb_penable;
  assign i2s_apb_m2s.paddr   = apb_paddr;
  assign i2s_apb_m2s.pwdata  = apb_pwdata;
  assign i2s_apb_m2s.pwrite  = apb_pwrite;
  
  assign apb_prdata[7] = i2s_apb_s2m.prdata;
  assign apb_pready[7] = i2s_apb_s2m.pready;
  assign apb_pserr[7]  = i2s_apb_s2m.pserr;
  
  // Create Completer APB interfaces
  apb_m2s vsync_apb_m2s;
  apb_s2m vsync_apb_s2m;
  
  // Arrays for apb_switch interface
  apb_m2s apb6_m2s_array [2];
  apb_s2m apb6_s2m_array [2];
  
  // APB switch instantiation using existing hololink module
  // W_OFSET=11, W_SW=1 means bit [11] is used for decode, bits [10:0] passed to Completer
  // vsync gets addr[11]=0 (0x000-0x7FF), delayed_io gets addr[11]=1 (0x800-0xFFF)
  apb_switch #(
    .N_MPORT              ( 1  ),   // Single Requester port
    .N_SPORT              ( 2  ),   // Two Completer ports: vsync and delayed_io
    .W_OFSET              ( 11 ),   // Pass address bits [10:0] to Completer
    .W_SW                 ( 1  ),   // Use 1 bit (bit [11]) for decode
    .MERGE_COMPLETER_SIG  ( 1  )    // Merge completer signals
  ) u_apb6_switch (
    .i_apb_clk     ( apb_clk              ),
    .i_apb_reset   ( apb_rst              ),
    .i_apb_timeout ( 1'b0                 ), // No timeout
    
    // Requester interface (from main APB bus) - single port array
    .i_apb_m2s     ( switch_apb6_m2s[0:0] ),
    .o_apb_s2m     ( switch_apb6_s2m[0:0] ),
    
    // Completer interfaces: [0] = vsync (addr[11]=0), [1] = delayed_io (addr[11]=1)  
    .o_apb_m2s     ( apb6_m2s_array       ),
    .i_apb_s2m     ( apb6_s2m_array       )
  );
  
  // Extract single element from arrays
  assign vsync_apb_m2s      = apb6_m2s_array[0];
  assign delayed_io_apb_m2s = apb6_m2s_array[1];

  assign apb6_s2m_array[0] = vsync_apb_s2m;
  assign apb6_s2m_array[1] = delayed_io_apb_s2m;

  // New i2s_peripheral_top instantiation
  i2s_top #(
    .AXI_DWIDTH     ( 32           ),  // Match the 32-bit interface
    .I2S_DWIDTH     ( 32           ),  // I2S data width
    .TX_FIFO_DEPTH  ( 8            ),  // TX buffer depth
    .RX_FIFO_DEPTH  ( 8            ),  // RX buffer depth  
    .APB_ADDR_WIDTH ( 12           )   // APB address space width
  ) u_i2s (
    // Clock and Reset
    .i_apb_clk         ( apb_clk            ),
    .i_apb_rst         ( apb_rst            ),
    // .i_ref_clk         ( apb_clk            ), // Use APB clock as reference
    .i_ref_clk         ( i2s_ref_clk        ), // Use APB clock as reference
    .i_axis_clk        ( apb_clk            ), // Use APB clock for AXI-Stream
    
    // APB Configuration Interface
    .i_apb_m2s         ( i2s_apb_m2s        ),
    .o_apb_s2m         ( i2s_apb_s2m        ),
    
    // Host AXI-Stream TX Interface (Host to I2S)
    .i_tx_axis_tvalid  ( sif_tx_axis_tvalid[0] ),
    .i_tx_axis_tdata   ( sif_tx_axis_tdata[0]  ),
    .i_tx_axis_tkeep   ( sif_tx_axis_tkeep[0]  ),
    .i_tx_axis_tlast   ( sif_tx_axis_tlast[0]  ),
    .i_tx_axis_tuser   ( sif_tx_axis_tuser[0]  ),
    .o_tx_axis_tready  ( sif_tx_axis_tready[0] ),
    
    // Host AXI-Stream RX Interface (I2S to Host) - Unused
    .o_rx_axis_tvalid  (                    ), // Not connected - TX only
    .o_rx_axis_tdata   (                    ), // Not connected - TX only
    .o_rx_axis_tkeep   (                    ), // Not connected - TX only
    .o_rx_axis_tlast   (                    ), // Not connected - TX only
    .o_rx_axis_tuser   (                    ), // Not connected - TX only
    .i_rx_axis_tready  ( 1'b0               ), // Not ready - TX only
    
    // I2S Physical Interface
    .o_i2s_bclk        ( i2s_clk_ext        ), // Not connected
    .o_i2s_lrclk       ( I2S_LRCLK          ),
    .o_i2s_sdata_tx    ( I2S_SDATA          ),
    .i_i2s_sdata_rx    ( 1'b0               ), // Not connected - TX only
    .o_i2s_mclk_out    ( i2s_mclk_ext       ), // Not connected
    
    // Status and Interrupts
    .o_tx_underrun     (                    ), // Not connected
    .o_rx_overrun      (                    ), // Not connected
    .o_pll_locked      (                    ), // Not connected
    .o_irq             (                    )  // Not connected
  );

//------------------------------------------------------------------------------
// Camera LVDS Data
//------------------------------------------------------------------------------

  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_clk;
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_rst;
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_axis_tvalid;
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_axis_tlast;
  logic [`DATAPATH_WIDTH-1:0]    sif_rx_axis_tdata [`SENSOR_RX_IF_INST-1:0];
  logic [`DATAKEEP_WIDTH-1:0]    sif_rx_axis_tkeep [`SENSOR_RX_IF_INST-1:0];
  logic [`DATAUSER_WIDTH-1:0]    sif_rx_axis_tuser [`SENSOR_RX_IF_INST-1:0];
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_axis_tready;

  logic [`SENSOR_RX_IF_INST-1:0] cam_drdy_i;

  assign CAM_DRDY = &cam_drdy_i;

  generate
    for (i=0; i<`SENSOR_RX_IF_INST; i++) begin: cam_sensor_rcvr
      assign sif_rx_clk[i] = hif_clk;

      cam_rcvr u_cam_rcvr (
        // LVDS PAD IO
        .i_rx_sclk     ( apb_clk                ),
        .i_rx_dclk     ( CAM_DCLK           [i] ), // 500MHz lvds data clock
        .i_rx_data     ( CAM_DATA           [i] ), // 500MHz ddr lvds data (1000 Mbps)
        .o_rx_drdy     ( cam_drdy_i         [i] ),
        // clock and reset
        .i_pclk        ( sif_rx_clk         [i] ),
        .i_prst        ( sif_rx_rst         [i] ),
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
    .i_clk           ( ptp_clk                ),
    .i_rst           ( ptp_rst                ),

    .i_apb_clk       ( apb_clk                ),
    .i_apb_rst       ( apb_rst                ),
    .i_apb_sel       ( vsync_apb_m2s.psel     ),  // Use switch signal (only 0x000-0x7FF)
    .i_apb_enable    ( vsync_apb_m2s.penable  ),  // Use switch signal  
    .i_apb_addr      ( vsync_apb_m2s.paddr    ),  // Use switch signal
    .i_apb_wdata     ( vsync_apb_m2s.pwdata   ),  // Use switch signal
    .i_apb_write     ( vsync_apb_m2s.pwrite   ),  // Use switch signal
    .o_apb_ready     ( vsync_apb_s2m.pready   ),  // Connect to switch response
    .o_apb_rdata     ( vsync_apb_s2m.prdata   ),  // Connect to switch response
    .o_apb_serr      ( vsync_apb_s2m.pserr    ),  // Connect to switch response

    .i_pps           ( sys_pps                ),
    .i_ptp_nanosec   ( ptp_nsec               ),

    .o_vsync_strb    ( vsync                  ),
    .o_gpio_mux_en   ( gpio_mux_en            )
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

  //TODO: GPIO MUX for UART?
  assign GPIO = {gpio_tri[15:1], delayed_io_pin};
  //assign GPIO = {gpio_tri[15:12], 1'bz, uart_tx, gpio_tri[9:1], delayed_io_pin};
  //assign uart_rx = GPIO[11];

  ///////////////////////////////////////
  //CAM GPIO
  ///////////////////////////////////////
  logic [14:0] cam_gpio_out;
  logic [14:0] cam_gpio_tri;

  generate
    for (i=0; i<3; i++) begin
      assign cam_gpio_tri[i] = cam_gpio_out[i];
    end
  endgenerate
  
  generate
    for (i=4; i<6; i++) begin
      assign cam_gpio_tri[i] = cam_gpio_out[i];
    end
  endgenerate
  
  generate
    for (i=9; i<15; i++) begin
      assign cam_gpio_tri[i] = cam_gpio_out[i];
    end
  endgenerate

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
  
  
  //assign sif_event = {10'h0, sof, GPIO[1], gpio_out[0], sif_rx_axis_tlast[1:0]};
  assign sif_event = {10'h0, sof, GPIO[1], delayed_io_pin, sif_rx_axis_tlast[1:0]};

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
    .i_sif_rx_clk      ( sif_rx_clk         ),
    .o_sif_rx_rst      ( sif_rx_rst         ),
    .i_sif_tx_clk      ( sif_tx_clk         ),
    .o_sif_tx_rst      ( sif_tx_rst         ),
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
