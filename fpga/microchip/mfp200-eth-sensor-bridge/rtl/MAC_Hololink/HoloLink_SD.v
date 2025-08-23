//////////////////////////////////////////////////////////////////////
// Created by SmartDesign Tue Aug 12 12:27:35 2025
// Version: 2025.1 2025.1.0.14
//////////////////////////////////////////////////////////////////////

`timescale 1ns / 100ps

// HoloLink_SD
module HoloLink_SD(
    // Inputs
    APB_INITIATOR_i_apb_prdata,
    APB_INITIATOR_i_apb_pready,
    APB_INITIATOR_i_apb_pserr,
    DI,
    FLASH,
    HIF_INITIATOR_0_i_hif_axis_tready_0,
    HIF_INITIATOR_1_i_hif_axis_tready_1,
    HIF_TARGET_0_i_hif_axis_tdata_0,
    HIF_TARGET_0_i_hif_axis_tkeep_0,
    HIF_TARGET_0_i_hif_axis_tlast_0,
    HIF_TARGET_0_i_hif_axis_tuser_0,
    HIF_TARGET_0_i_hif_axis_tvalid_0,
    HIF_TARGET_1_i_hif_axis_tdata_1,
    HIF_TARGET_1_i_hif_axis_tkeep_1,
    HIF_TARGET_1_i_hif_axis_tlast_1,
    HIF_TARGET_1_i_hif_axis_tuser_1,
    HIF_TARGET_1_i_hif_axis_tvalid_1,
    IFACE,
    SIF_TARGET_0_i_sif_axis_tdata_0,
    SIF_TARGET_0_i_sif_axis_tkeep_0,
    SIF_TARGET_0_i_sif_axis_tlast_0,
    SIF_TARGET_0_i_sif_axis_tuser_0,
    SIF_TARGET_0_i_sif_axis_tvalid_0,
    SIF_TARGET_1_i_sif_axis_tdata_1,
    SIF_TARGET_1_i_sif_axis_tkeep_1,
    SIF_TARGET_1_i_sif_axis_tlast_1,
    SIF_TARGET_1_i_sif_axis_tuser_1,
    SIF_TARGET_1_i_sif_axis_tvalid_1,
    i_apb_clk,
    i_hif_clk,
    i_ptp_clk,
    i_sif_clk,
    i_sys_rst,
    // Outputs
    APB_INITIATOR_o_apb_paddr,
    APB_INITIATOR_o_apb_penable,
    APB_INITIATOR_o_apb_psel,
    APB_INITIATOR_o_apb_pwdata,
    APB_INITIATOR_o_apb_pwrite,
    DO,
    HIF_INITIATOR_0_o_hif_axis_tdata_0,
    HIF_INITIATOR_0_o_hif_axis_tkeep_0,
    HIF_INITIATOR_0_o_hif_axis_tlast_0,
    HIF_INITIATOR_0_o_hif_axis_tuser_0,
    HIF_INITIATOR_0_o_hif_axis_tvalid_0,
    HIF_INITIATOR_1_o_hif_axis_tdata_1,
    HIF_INITIATOR_1_o_hif_axis_tkeep_1,
    HIF_INITIATOR_1_o_hif_axis_tlast_1,
    HIF_INITIATOR_1_o_hif_axis_tuser_1,
    HIF_INITIATOR_1_o_hif_axis_tvalid_1,
    HIF_TARGET_0_o_hif_axis_tready_0,
    HIF_TARGET_1_o_hif_axis_tready_1,
    SIF_TARGET_0_o_sif_axis_tready_0,
    SIF_TARGET_1_o_sif_axis_tready_1,
    o_init_done,
    o_pps,
    o_ptp_rst,
    // Inouts
    CAM1_SCL,
    CAM1_SDA,
    CAM2_SCL,
    CAM2_SDA,
    CLK,
    EEPROM_SCL,
    EEPROM_SDA,
    GPIO,
    SS
);

//--------------------------------------------------------------------
// Input
//--------------------------------------------------------------------
input  [31:0] APB_INITIATOR_i_apb_prdata;
input         APB_INITIATOR_i_apb_pready;
input         APB_INITIATOR_i_apb_pserr;
input         DI;
input         FLASH;
input         HIF_INITIATOR_0_i_hif_axis_tready_0;
input         HIF_INITIATOR_1_i_hif_axis_tready_1;
input  [63:0] HIF_TARGET_0_i_hif_axis_tdata_0;
input  [7:0]  HIF_TARGET_0_i_hif_axis_tkeep_0;
input         HIF_TARGET_0_i_hif_axis_tlast_0;
input  [0:0]  HIF_TARGET_0_i_hif_axis_tuser_0;
input         HIF_TARGET_0_i_hif_axis_tvalid_0;
input  [63:0] HIF_TARGET_1_i_hif_axis_tdata_1;
input  [7:0]  HIF_TARGET_1_i_hif_axis_tkeep_1;
input         HIF_TARGET_1_i_hif_axis_tlast_1;
input  [0:0]  HIF_TARGET_1_i_hif_axis_tuser_1;
input         HIF_TARGET_1_i_hif_axis_tvalid_1;
input         IFACE;
input  [63:0] SIF_TARGET_0_i_sif_axis_tdata_0;
input  [7:0]  SIF_TARGET_0_i_sif_axis_tkeep_0;
input         SIF_TARGET_0_i_sif_axis_tlast_0;
input  [0:0]  SIF_TARGET_0_i_sif_axis_tuser_0;
input         SIF_TARGET_0_i_sif_axis_tvalid_0;
input  [63:0] SIF_TARGET_1_i_sif_axis_tdata_1;
input  [7:0]  SIF_TARGET_1_i_sif_axis_tkeep_1;
input         SIF_TARGET_1_i_sif_axis_tlast_1;
input  [0:0]  SIF_TARGET_1_i_sif_axis_tuser_1;
input         SIF_TARGET_1_i_sif_axis_tvalid_1;
input         i_apb_clk;
input         i_hif_clk;
input         i_ptp_clk;
input         i_sif_clk;
input         i_sys_rst;
//--------------------------------------------------------------------
// Output
//--------------------------------------------------------------------
output [31:0] APB_INITIATOR_o_apb_paddr;
output        APB_INITIATOR_o_apb_penable;
output        APB_INITIATOR_o_apb_psel;
output [31:0] APB_INITIATOR_o_apb_pwdata;
output        APB_INITIATOR_o_apb_pwrite;
output        DO;
output [63:0] HIF_INITIATOR_0_o_hif_axis_tdata_0;
output [7:0]  HIF_INITIATOR_0_o_hif_axis_tkeep_0;
output        HIF_INITIATOR_0_o_hif_axis_tlast_0;
output [0:0]  HIF_INITIATOR_0_o_hif_axis_tuser_0;
output        HIF_INITIATOR_0_o_hif_axis_tvalid_0;
output [63:0] HIF_INITIATOR_1_o_hif_axis_tdata_1;
output [7:0]  HIF_INITIATOR_1_o_hif_axis_tkeep_1;
output        HIF_INITIATOR_1_o_hif_axis_tlast_1;
output [0:0]  HIF_INITIATOR_1_o_hif_axis_tuser_1;
output        HIF_INITIATOR_1_o_hif_axis_tvalid_1;
output        HIF_TARGET_0_o_hif_axis_tready_0;
output        HIF_TARGET_1_o_hif_axis_tready_1;
output        SIF_TARGET_0_o_sif_axis_tready_0;
output        SIF_TARGET_1_o_sif_axis_tready_1;
output        o_init_done;
output        o_pps;
output        o_ptp_rst;
//--------------------------------------------------------------------
// Inout
//--------------------------------------------------------------------
inout         CAM1_SCL;
inout         CAM1_SDA;
inout         CAM2_SCL;
inout         CAM2_SDA;
inout         CLK;
inout         EEPROM_SCL;
inout         EEPROM_SDA;
inout  [15:0] GPIO;
inout         SS;
//--------------------------------------------------------------------
// Nets
//--------------------------------------------------------------------
wire   [31:0] APB_INITIATOR_PADDR;
wire          APB_INITIATOR_PENABLE;
wire   [31:0] APB_INITIATOR_i_apb_prdata;
wire          APB_INITIATOR_i_apb_pready;
wire          APB_INITIATOR_PSELx;
wire          APB_INITIATOR_i_apb_pserr;
wire   [31:0] APB_INITIATOR_PWDATA;
wire          APB_INITIATOR_PWRITE;
wire          BIBUF_0_Y;
wire          BIBUF_1_Y;
wire          BIBUF_2_Y;
wire          BIBUF_3_Y;
wire          BIBUF_4_Y;
wire          BIBUF_5_Y;
wire          CAM1_SCL;
wire          CAM1_SDA;
wire          CAM2_SCL;
wire          CAM2_SDA;
wire          CLK;
wire          DI;
wire          DO_net_0;
wire          EEPROM_SCL;
wire          EEPROM_SDA;
wire          FLASH;
wire   [15:0] GPIO;
wire   [63:0] HIF_INITIATOR_0_TDATA;
wire   [7:0]  HIF_INITIATOR_0_TKEEP;
wire          HIF_INITIATOR_0_TLAST;
wire          HIF_INITIATOR_0_i_hif_axis_tready_0;
wire   [0:0]  HIF_INITIATOR_0_TUSER;
wire          HIF_INITIATOR_0_TVALID;
wire   [63:0] HIF_INITIATOR_1_TDATA;
wire   [7:0]  HIF_INITIATOR_1_TKEEP;
wire          HIF_INITIATOR_1_TLAST;
wire          HIF_INITIATOR_1_i_hif_axis_tready_1;
wire   [0:0]  HIF_INITIATOR_1_TUSER;
wire          HIF_INITIATOR_1_TVALID;
wire   [63:0] HIF_TARGET_0_i_hif_axis_tdata_0;
wire   [7:0]  HIF_TARGET_0_i_hif_axis_tkeep_0;
wire          HIF_TARGET_0_i_hif_axis_tlast_0;
wire          HIF_TARGET_0_TREADY;
wire   [0:0]  HIF_TARGET_0_i_hif_axis_tuser_0;
wire          HIF_TARGET_0_i_hif_axis_tvalid_0;
wire   [63:0] HIF_TARGET_1_i_hif_axis_tdata_1;
wire   [7:0]  HIF_TARGET_1_i_hif_axis_tkeep_1;
wire          HIF_TARGET_1_i_hif_axis_tlast_1;
wire          HIF_TARGET_1_TREADY;
wire   [0:0]  HIF_TARGET_1_i_hif_axis_tuser_1;
wire          HIF_TARGET_1_i_hif_axis_tvalid_1;
wire   [0:0]  HOLOLINK_top_0_o_i2c_scl_en0to0;
wire   [1:1]  HOLOLINK_top_0_o_i2c_scl_en1to1;
wire   [2:2]  HOLOLINK_top_0_o_i2c_scl_en2to2;
wire   [0:0]  HOLOLINK_top_0_o_i2c_sda_en0to0;
wire   [1:1]  HOLOLINK_top_0_o_i2c_sda_en1to1;
wire   [2:2]  HOLOLINK_top_0_o_i2c_sda_en2to2;
wire          HOLOLINK_top_0_o_spi_csn;
wire          HOLOLINK_top_0_o_spi_oen;
wire          HOLOLINK_top_0_o_spi_sck;
wire   [0:0]  HOLOLINK_top_0_o_spi_sdio0to0;
wire          i_apb_clk;
wire          i_hif_clk;
wire          i_ptp_clk;
wire          i_sif_clk;
wire          i_sys_rst;
wire          IFACE;
wire          o_init_done_net_0;
wire          o_pps_net_0;
wire          o_ptp_rst_net_0;
wire          PF_SPI_0_D_I;
wire   [63:0] SIF_TARGET_0_i_sif_axis_tdata_0;
wire   [7:0]  SIF_TARGET_0_i_sif_axis_tkeep_0;
wire          SIF_TARGET_0_i_sif_axis_tlast_0;
wire          SIF_TARGET_0_TREADY;
wire   [0:0]  SIF_TARGET_0_i_sif_axis_tuser_0;
wire          SIF_TARGET_0_i_sif_axis_tvalid_0;
wire   [63:0] SIF_TARGET_1_i_sif_axis_tdata_1;
wire   [7:0]  SIF_TARGET_1_i_sif_axis_tkeep_1;
wire          SIF_TARGET_1_i_sif_axis_tlast_1;
wire          SIF_TARGET_1_TREADY;
wire   [0:0]  SIF_TARGET_1_i_sif_axis_tuser_1;
wire          SIF_TARGET_1_i_sif_axis_tvalid_1;
wire          SS;
wire          DO_net_1;
wire          o_init_done_net_1;
wire          o_pps_net_1;
wire          HIF_INITIATOR_0_TVALID_net_0;
wire   [63:0] HIF_INITIATOR_0_TDATA_net_0;
wire   [7:0]  HIF_INITIATOR_0_TKEEP_net_0;
wire          HIF_INITIATOR_0_TLAST_net_0;
wire   [0:0]  HIF_INITIATOR_0_TUSER_net_0;
wire          HIF_TARGET_0_TREADY_net_0;
wire          SIF_TARGET_0_TREADY_net_0;
wire          HIF_INITIATOR_1_TVALID_net_0;
wire   [63:0] HIF_INITIATOR_1_TDATA_net_0;
wire   [7:0]  HIF_INITIATOR_1_TKEEP_net_0;
wire          HIF_INITIATOR_1_TLAST_net_0;
wire   [0:0]  HIF_INITIATOR_1_TUSER_net_0;
wire          HIF_TARGET_1_TREADY_net_0;
wire          SIF_TARGET_1_TREADY_net_0;
wire   [31:0] APB_INITIATOR_PADDR_net_0;
wire          APB_INITIATOR_PENABLE_net_0;
wire          APB_INITIATOR_PWRITE_net_0;
wire   [31:0] APB_INITIATOR_PWDATA_net_0;
wire          APB_INITIATOR_PSELx_net_0;
wire          o_ptp_rst_net_1;
wire   [3:1]  o_spi_sdio_slice_0;
wire   [3:0]  i_spi_sdio_net_0;
wire   [3:0]  o_spi_sdio_net_0;
wire   [2:0]  i_i2c_scl_net_0;
wire   [2:0]  i_i2c_sda_net_0;
wire   [2:0]  o_i2c_scl_en_net_0;
wire   [2:0]  o_i2c_sda_en_net_0;
//--------------------------------------------------------------------
// TiedOff Nets
//--------------------------------------------------------------------
wire          GND_net;
wire   [15:0] i_sif_event_const_net_0;
wire   [3:2]  i_spi_sdio_const_net_0;
wire          VCC_net;
//--------------------------------------------------------------------
// Inverted Nets
//--------------------------------------------------------------------
wire          E_IN_POST_INV0_0;
wire          E_IN_POST_INV1_0;
wire          E_IN_POST_INV2_0;
wire          E_IN_POST_INV3_0;
wire          E_IN_POST_INV4_0;
wire          E_IN_POST_INV5_0;
//--------------------------------------------------------------------
// Constant assignments
//--------------------------------------------------------------------
assign GND_net                 = 1'b0;
assign i_sif_event_const_net_0 = 16'h0000;
assign i_spi_sdio_const_net_0  = 2'h0;
assign VCC_net                 = 1'b1;
//--------------------------------------------------------------------
// Inversions
//--------------------------------------------------------------------
assign E_IN_POST_INV0_0 = ~ HOLOLINK_top_0_o_i2c_scl_en1to1[1];
assign E_IN_POST_INV1_0 = ~ HOLOLINK_top_0_o_i2c_sda_en1to1[1];
assign E_IN_POST_INV2_0 = ~ HOLOLINK_top_0_o_i2c_sda_en0to0[0];
assign E_IN_POST_INV3_0 = ~ HOLOLINK_top_0_o_i2c_scl_en0to0[0];
assign E_IN_POST_INV4_0 = ~ HOLOLINK_top_0_o_i2c_scl_en2to2[2];
assign E_IN_POST_INV5_0 = ~ HOLOLINK_top_0_o_i2c_sda_en2to2[2];
//--------------------------------------------------------------------
// Top level output port assignments
//--------------------------------------------------------------------
assign DO_net_1                                 = DO_net_0;
assign DO                                       = DO_net_1;
assign o_init_done_net_1                        = o_init_done_net_0;
assign o_init_done                              = o_init_done_net_1;
assign o_pps_net_1                              = o_pps_net_0;
assign o_pps                                    = o_pps_net_1;
assign HIF_INITIATOR_0_TVALID_net_0             = HIF_INITIATOR_0_TVALID;
assign HIF_INITIATOR_0_o_hif_axis_tvalid_0      = HIF_INITIATOR_0_TVALID_net_0;
assign HIF_INITIATOR_0_TDATA_net_0              = HIF_INITIATOR_0_TDATA;
assign HIF_INITIATOR_0_o_hif_axis_tdata_0[63:0] = HIF_INITIATOR_0_TDATA_net_0;
assign HIF_INITIATOR_0_TKEEP_net_0              = HIF_INITIATOR_0_TKEEP;
assign HIF_INITIATOR_0_o_hif_axis_tkeep_0[7:0]  = HIF_INITIATOR_0_TKEEP_net_0;
assign HIF_INITIATOR_0_TLAST_net_0              = HIF_INITIATOR_0_TLAST;
assign HIF_INITIATOR_0_o_hif_axis_tlast_0       = HIF_INITIATOR_0_TLAST_net_0;
assign HIF_INITIATOR_0_TUSER_net_0[0]           = HIF_INITIATOR_0_TUSER[0];
assign HIF_INITIATOR_0_o_hif_axis_tuser_0[0:0]  = HIF_INITIATOR_0_TUSER_net_0[0];
assign HIF_TARGET_0_TREADY_net_0                = HIF_TARGET_0_TREADY;
assign HIF_TARGET_0_o_hif_axis_tready_0         = HIF_TARGET_0_TREADY_net_0;
assign SIF_TARGET_0_TREADY_net_0                = SIF_TARGET_0_TREADY;
assign SIF_TARGET_0_o_sif_axis_tready_0         = SIF_TARGET_0_TREADY_net_0;
assign HIF_INITIATOR_1_TVALID_net_0             = HIF_INITIATOR_1_TVALID;
assign HIF_INITIATOR_1_o_hif_axis_tvalid_1      = HIF_INITIATOR_1_TVALID_net_0;
assign HIF_INITIATOR_1_TDATA_net_0              = HIF_INITIATOR_1_TDATA;
assign HIF_INITIATOR_1_o_hif_axis_tdata_1[63:0] = HIF_INITIATOR_1_TDATA_net_0;
assign HIF_INITIATOR_1_TKEEP_net_0              = HIF_INITIATOR_1_TKEEP;
assign HIF_INITIATOR_1_o_hif_axis_tkeep_1[7:0]  = HIF_INITIATOR_1_TKEEP_net_0;
assign HIF_INITIATOR_1_TLAST_net_0              = HIF_INITIATOR_1_TLAST;
assign HIF_INITIATOR_1_o_hif_axis_tlast_1       = HIF_INITIATOR_1_TLAST_net_0;
assign HIF_INITIATOR_1_TUSER_net_0[0]           = HIF_INITIATOR_1_TUSER[0];
assign HIF_INITIATOR_1_o_hif_axis_tuser_1[0:0]  = HIF_INITIATOR_1_TUSER_net_0[0];
assign HIF_TARGET_1_TREADY_net_0                = HIF_TARGET_1_TREADY;
assign HIF_TARGET_1_o_hif_axis_tready_1         = HIF_TARGET_1_TREADY_net_0;
assign SIF_TARGET_1_TREADY_net_0                = SIF_TARGET_1_TREADY;
assign SIF_TARGET_1_o_sif_axis_tready_1         = SIF_TARGET_1_TREADY_net_0;
assign APB_INITIATOR_PADDR_net_0                = APB_INITIATOR_PADDR;
assign APB_INITIATOR_o_apb_paddr[31:0]          = APB_INITIATOR_PADDR_net_0;
assign APB_INITIATOR_PENABLE_net_0              = APB_INITIATOR_PENABLE;
assign APB_INITIATOR_o_apb_penable              = APB_INITIATOR_PENABLE_net_0;
assign APB_INITIATOR_PWRITE_net_0               = APB_INITIATOR_PWRITE;
assign APB_INITIATOR_o_apb_pwrite               = APB_INITIATOR_PWRITE_net_0;
assign APB_INITIATOR_PWDATA_net_0               = APB_INITIATOR_PWDATA;
assign APB_INITIATOR_o_apb_pwdata[31:0]         = APB_INITIATOR_PWDATA_net_0;
assign APB_INITIATOR_PSELx_net_0                = APB_INITIATOR_PSELx;
assign APB_INITIATOR_o_apb_psel                 = APB_INITIATOR_PSELx_net_0;
assign o_ptp_rst_net_1                          = o_ptp_rst_net_0;
assign o_ptp_rst                                = o_ptp_rst_net_1;
//--------------------------------------------------------------------
// Slices assignments
//--------------------------------------------------------------------
assign HOLOLINK_top_0_o_i2c_scl_en0to0[0] = o_i2c_scl_en_net_0[0:0];
assign HOLOLINK_top_0_o_i2c_scl_en1to1[1] = o_i2c_scl_en_net_0[1:1];
assign HOLOLINK_top_0_o_i2c_scl_en2to2[2] = o_i2c_scl_en_net_0[2:2];
assign HOLOLINK_top_0_o_i2c_sda_en0to0[0] = o_i2c_sda_en_net_0[0:0];
assign HOLOLINK_top_0_o_i2c_sda_en1to1[1] = o_i2c_sda_en_net_0[1:1];
assign HOLOLINK_top_0_o_i2c_sda_en2to2[2] = o_i2c_sda_en_net_0[2:2];
assign HOLOLINK_top_0_o_spi_sdio0to0[0]   = o_spi_sdio_net_0[0:0];
assign o_spi_sdio_slice_0                 = o_spi_sdio_net_0[3:1];
//--------------------------------------------------------------------
// Concatenation assignments
//--------------------------------------------------------------------
assign i_spi_sdio_net_0 = { 2'h0 , PF_SPI_0_D_I , 1'b0 };
assign i_i2c_scl_net_0  = { BIBUF_4_Y , BIBUF_0_Y , BIBUF_3_Y };
assign i_i2c_sda_net_0  = { BIBUF_5_Y , BIBUF_1_Y , BIBUF_2_Y };
//--------------------------------------------------------------------
// Component instances
//--------------------------------------------------------------------
//--------BIBUF
BIBUF BIBUF_0(
        // Inputs
        .D   ( GND_net ),
        .E   ( E_IN_POST_INV0_0 ),
        // Outputs
        .Y   ( BIBUF_0_Y ),
        // Inouts
        .PAD ( CAM1_SCL ) 
        );

//--------BIBUF
BIBUF BIBUF_1(
        // Inputs
        .D   ( GND_net ),
        .E   ( E_IN_POST_INV1_0 ),
        // Outputs
        .Y   ( BIBUF_1_Y ),
        // Inouts
        .PAD ( CAM1_SDA ) 
        );

//--------BIBUF
BIBUF BIBUF_2(
        // Inputs
        .D   ( GND_net ),
        .E   ( E_IN_POST_INV2_0 ),
        // Outputs
        .Y   ( BIBUF_2_Y ),
        // Inouts
        .PAD ( EEPROM_SDA ) 
        );

//--------BIBUF
BIBUF BIBUF_3(
        // Inputs
        .D   ( GND_net ),
        .E   ( E_IN_POST_INV3_0 ),
        // Outputs
        .Y   ( BIBUF_3_Y ),
        // Inouts
        .PAD ( EEPROM_SCL ) 
        );

//--------BIBUF
BIBUF BIBUF_4(
        // Inputs
        .D   ( GND_net ),
        .E   ( E_IN_POST_INV4_0 ),
        // Outputs
        .Y   ( BIBUF_4_Y ),
        // Inouts
        .PAD ( CAM2_SCL ) 
        );

//--------BIBUF
BIBUF BIBUF_5(
        // Inputs
        .D   ( GND_net ),
        .E   ( E_IN_POST_INV5_0 ),
        // Outputs
        .Y   ( BIBUF_5_Y ),
        // Inouts
        .PAD ( CAM2_SDA ) 
        );

//--------HOLOLINK_top_wrapper
HOLOLINK_top_wrapper HOLOLINK_top_0(
        // Inputs
        .i_sys_rst           ( i_sys_rst ),
        .i_apb_clk           ( i_apb_clk ),
        .i_apb_pready        ( APB_INITIATOR_i_apb_pready ),
        .i_apb_prdata        ( APB_INITIATOR_i_apb_prdata ),
        .i_apb_pserr         ( APB_INITIATOR_i_apb_pserr ),
        .i_sif_clk           ( i_sif_clk ),
        .i_sif_axis_tvalid_0 ( SIF_TARGET_0_i_sif_axis_tvalid_0 ),
        .i_sif_axis_tlast_0  ( SIF_TARGET_0_i_sif_axis_tlast_0 ),
        .i_sif_axis_tdata_0  ( SIF_TARGET_0_i_sif_axis_tdata_0 ),
        .i_sif_axis_tkeep_0  ( SIF_TARGET_0_i_sif_axis_tkeep_0 ),
        .i_sif_axis_tuser_0  ( SIF_TARGET_0_i_sif_axis_tuser_0 ),
        .i_sif_axis_tvalid_1 ( SIF_TARGET_1_i_sif_axis_tvalid_1 ),
        .i_sif_axis_tlast_1  ( SIF_TARGET_1_i_sif_axis_tlast_1 ),
        .i_sif_axis_tdata_1  ( SIF_TARGET_1_i_sif_axis_tdata_1 ),
        .i_sif_axis_tkeep_1  ( SIF_TARGET_1_i_sif_axis_tkeep_1 ),
        .i_sif_axis_tuser_1  ( SIF_TARGET_1_i_sif_axis_tuser_1 ),
        .i_sif_event         ( i_sif_event_const_net_0 ),
        .i_hif_clk           ( i_hif_clk ),
        .i_hif_axis_tvalid_0 ( HIF_TARGET_0_i_hif_axis_tvalid_0 ),
        .i_hif_axis_tlast_0  ( HIF_TARGET_0_i_hif_axis_tlast_0 ),
        .i_hif_axis_tdata_0  ( HIF_TARGET_0_i_hif_axis_tdata_0 ),
        .i_hif_axis_tkeep_0  ( HIF_TARGET_0_i_hif_axis_tkeep_0 ),
        .i_hif_axis_tuser_0  ( HIF_TARGET_0_i_hif_axis_tuser_0 ),
        .i_hif_axis_tready_0 ( HIF_INITIATOR_0_i_hif_axis_tready_0 ),
        .i_hif_axis_tvalid_1 ( HIF_TARGET_1_i_hif_axis_tvalid_1 ),
        .i_hif_axis_tlast_1  ( HIF_TARGET_1_i_hif_axis_tlast_1 ),
        .i_hif_axis_tdata_1  ( HIF_TARGET_1_i_hif_axis_tdata_1 ),
        .i_hif_axis_tkeep_1  ( HIF_TARGET_1_i_hif_axis_tkeep_1 ),
        .i_hif_axis_tuser_1  ( HIF_TARGET_1_i_hif_axis_tuser_1 ),
        .i_hif_axis_tready_1 ( HIF_INITIATOR_1_i_hif_axis_tready_1 ),
        .i_spi_sdio          ( i_spi_sdio_net_0 ),
        .i_i2c_scl           ( i_i2c_scl_net_0 ),
        .i_i2c_sda           ( i_i2c_sda_net_0 ),
        .i_ptp_clk           ( i_ptp_clk ),
        // Outputs
        .o_apb_rst           (  ),
        .o_apb_psel          ( APB_INITIATOR_PSELx ),
        .o_apb_penable       ( APB_INITIATOR_PENABLE ),
        .o_apb_paddr         ( APB_INITIATOR_PADDR ),
        .o_apb_pwdata        ( APB_INITIATOR_PWDATA ),
        .o_apb_pwrite        ( APB_INITIATOR_PWRITE ),
        .o_init_done         ( o_init_done_net_0 ),
        .o_sif_rst           (  ),
        .o_sif_axis_tready_0 ( SIF_TARGET_0_TREADY ),
        .o_sif_axis_tready_1 ( SIF_TARGET_1_TREADY ),
        .o_hif_rst           (  ),
        .o_hif_axis_tready_0 ( HIF_TARGET_0_TREADY ),
        .o_hif_axis_tvalid_0 ( HIF_INITIATOR_0_TVALID ),
        .o_hif_axis_tlast_0  ( HIF_INITIATOR_0_TLAST ),
        .o_hif_axis_tdata_0  ( HIF_INITIATOR_0_TDATA ),
        .o_hif_axis_tkeep_0  ( HIF_INITIATOR_0_TKEEP ),
        .o_hif_axis_tuser_0  ( HIF_INITIATOR_0_TUSER ),
        .o_hif_axis_tready_1 ( HIF_TARGET_1_TREADY ),
        .o_hif_axis_tvalid_1 ( HIF_INITIATOR_1_TVALID ),
        .o_hif_axis_tlast_1  ( HIF_INITIATOR_1_TLAST ),
        .o_hif_axis_tdata_1  ( HIF_INITIATOR_1_TDATA ),
        .o_hif_axis_tkeep_1  ( HIF_INITIATOR_1_TKEEP ),
        .o_hif_axis_tuser_1  ( HIF_INITIATOR_1_TUSER ),
        .o_spi_csn           ( HOLOLINK_top_0_o_spi_csn ),
        .o_spi_sck           ( HOLOLINK_top_0_o_spi_sck ),
        .o_spi_sdio          ( o_spi_sdio_net_0 ),
        .o_spi_oen           ( HOLOLINK_top_0_o_spi_oen ),
        .o_i2c_scl_en        ( o_i2c_scl_en_net_0 ),
        .o_i2c_sda_en        ( o_i2c_sda_en_net_0 ),
        .o_sw_sys_rst        (  ),
        .o_sw_sen_rst        (  ),
        .o_ptp_rst           ( o_ptp_rst_net_0 ),
        .o_ptp_sec           (  ),
        .o_ptp_nanosec       (  ),
        .o_pps               ( o_pps_net_0 ),
        // Inouts
        .GPIO                ( GPIO ) 
        );

//--------PF_SPI
PF_SPI PF_SPI_0(
        // Inputs
        .CLK_OE        ( VCC_net ),
        .CLK_O         ( HOLOLINK_top_0_o_spi_sck ),
        .D_OE          ( HOLOLINK_top_0_o_spi_oen ),
        .D_O           ( HOLOLINK_top_0_o_spi_sdio0to0 ),
        .SS_OE         ( VCC_net ),
        .SS_O          ( HOLOLINK_top_0_o_spi_csn ),
        .DI            ( DI ),
        .IFACE         ( IFACE ),
        .FLASH         ( FLASH ),
        // Outputs
        .CLK_I         (  ),
        .D_I           ( PF_SPI_0_D_I ),
        .SS_I          (  ),
        .FAB_SPI_OWNER (  ),
        .DO            ( DO_net_0 ),
        // Inouts
        .CLK           ( CLK ),
        .SS            ( SS ) 
        );


endmodule
