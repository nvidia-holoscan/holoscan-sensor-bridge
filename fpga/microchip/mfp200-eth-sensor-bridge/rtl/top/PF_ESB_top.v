//=================================================================================================
//-- File Name                           : PF_ESB_top.v

//-- Targeted device                     : PolarFire
//-- Description                         : A top-level file for the PolarFire Ethernet sensor bridge  
//--                                       Libero SoC design provides an overview of the connections.  
//--                                       The complete design is available at 
//--                                       https://www.microchip.com/en-us/application-notes/an5522
//--
//-- © [2025] Microchip Technology Inc. and its subsidiaries

//-- Subject to your compliance with the terms and conditions of the license agreement accompanying 
//-- this RTL, you may use this Microchip RTL and any derivatives exclusively with Microchip 
//-- products. You are responsible for complying with third party license terms applicable to your 
//-- use of third party RTL (including open source RTL) that may accompany this Microchip RTL.
//-- RTL IS “AS IS.” NO WARRANTIES, WHETHER EXPRESS, IMPLIED OR STATUTORY, APPLY TO THIS RTL,
//-- INCLUDING ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR
//-- A PARTICULAR PURPOSE. IN NO EVENT WILL MICROCHIP BE LIABLE FOR ANY INDIRECT, SPECIAL,
//-- PUNITIVE, INCIDENTAL OR CONSEQUENTIAL LOSS, DAMAGE, COST OR EXPENSE OF ANY KIND 
//-- WHATSOEVER RELATED TO THE RTL, HOWEVER CAUSED, EVEN IF MICROCHIP HAS BEEN ADVISED OF 
//-- THE POSSIBILITY OR THE DAMAGES ARE FORESEEABLE. TO THE FULLEST EXTENT ALLOWED BY LAW, 
//-- MICROCHIP’S TOTAL LIABILITY ON ALL CLAIMS RELATED TO THE RTL WILL NOT EXCEED AMOUNT 
//-- OF FEES, IF ANY, YOU PAID DIRECTLY TO MICROCHIP FOR THIS RTL.
//--
//=================================================================================================

`timescale 1ns / 100ps

// PF_ESB_top
module PF_ESB_top(
    // Inputs
    CAM1_RX_CLK_N,
    CAM1_RX_CLK_N_0,
    CAM1_RX_CLK_P,
    CAM1_RX_CLK_P_0,
    CLK_IN,
    DI,
    FLASH,
    IFACE,
    LANE0_RXD_N,
    LANE0_RXD_N_0,
    LANE0_RXD_P,
    LANE0_RXD_P_0,
    REF_CLK_PAD_N,
    REF_CLK_PAD_P,
    RXD,
    RXD_0,
    RXD_N,
    RXD_N_0,
    // Outputs
    CAM1_CLK,
    CAM1_EN,
    CAM2_CLK,
    CAM2_EN,
    DO,
    LANE0_TXD_N,
    LANE0_TXD_N_0,
    LANE0_TXD_P,
    LANE0_TXD_P_0,
    LED1,
    RX_CLK_G,
    RX_CLK_G_0,
    TEN,
    c1_frame_valid_o,
    pps_stretch_o,
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
input         CAM1_RX_CLK_N;
input         CAM1_RX_CLK_N_0;
input         CAM1_RX_CLK_P;
input         CAM1_RX_CLK_P_0;
input         CLK_IN;
input         DI;
input         FLASH;
input         IFACE;
input         LANE0_RXD_N;
input         LANE0_RXD_N_0;
input         LANE0_RXD_P;
input         LANE0_RXD_P_0;
input         REF_CLK_PAD_N;
input         REF_CLK_PAD_P;
input  [3:0]  RXD;
input  [3:0]  RXD_0;
input  [3:0]  RXD_N;
input  [3:0]  RXD_N_0;
//--------------------------------------------------------------------
// Output
//--------------------------------------------------------------------
output        CAM1_CLK;
output        CAM1_EN;
output        CAM2_CLK;
output        CAM2_EN;
output        DO;
output        LANE0_TXD_N;
output        LANE0_TXD_N_0;
output        LANE0_TXD_P;
output        LANE0_TXD_P_0;
output        LED1;
output        RX_CLK_G;
output        RX_CLK_G_0;
output        TEN;
output        c1_frame_valid_o;
output        pps_stretch_o;
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
wire          c1_frame_valid_o_net_0;
wire          CAM1_EN_net_0;
wire          CAM1_RX_CLK_N;
wire          CAM1_RX_CLK_N_0;
wire          CAM1_RX_CLK_P;
wire          CAM1_RX_CLK_P_0;
wire          CAM1_SCL;
wire          CAM1_SDA;
wire          CAM2_CLK_net_0;
wire          CAM2_EN_net_0;
wire          CAM2_SCL;
wire          CAM2_SDA;
wire          CLK;
wire          CLK_IN;
wire          CORERESET_PF_C1_0_FABRIC_RESET_N;
wire          DI;
wire          DO_net_0;
wire          EEPROM_SCL;
wire          EEPROM_SDA;
wire          FLASH;
wire   [15:0] GPIO;
wire          IFACE;
wire   [63:0] IMX477_IF_TOP_0_BIF_1_TDATA;
wire   [7:0]  IMX477_IF_TOP_0_BIF_1_TKEEP;
wire          IMX477_IF_TOP_0_BIF_1_TLAST;
wire          IMX477_IF_TOP_0_BIF_1_TREADY;
wire   [7:0]  IMX477_IF_TOP_0_BIF_1_TSTRB;
wire          IMX477_IF_TOP_0_BIF_1_TVALID;
wire   [63:0] IMX477_IF_TOP_1_BIF_1_TDATA;
wire   [7:0]  IMX477_IF_TOP_1_BIF_1_TKEEP;
wire          IMX477_IF_TOP_1_BIF_1_TLAST;
wire          IMX477_IF_TOP_1_BIF_1_TREADY;
wire   [7:0]  IMX477_IF_TOP_1_BIF_1_TSTRB;
wire          IMX477_IF_TOP_1_BIF_1_TVALID;
wire          LANE0_RXD_N;
wire          LANE0_RXD_N_0;
wire          LANE0_RXD_P;
wire          LANE0_RXD_P_0;
wire          LANE0_TXD_N_net_0;
wire          LANE0_TXD_N_0_net_0;
wire          LANE0_TXD_P_net_0;
wire          LANE0_TXD_P_0_net_0;
wire          LED1_net_0;
wire          MAC_Hololink_0_CAM1_TRNG_RSTN;
wire          MAC_Hololink_0_CAM2_TRNG_RSTN;
wire          MAC_Hololink_0_LANE0_RX_CLK_R;
wire          MAC_Hololink_0_OUT2_FABCLK_0;
wire          PF_CCC_C3_0_OUT1_FABCLK_0;
wire          PF_CCC_C3_0_PLL_LOCK_0;
wire          PF_INIT_MONITOR_C0_0_AUTOCALIB_DONE;
wire          PF_INIT_MONITOR_C0_0_DEVICE_INIT_DONE;
wire          PF_INIT_MONITOR_C0_0_FABRIC_POR_N;
wire          pps_stretch_o_net_0;
wire          REF_CLK_PAD_N;
wire          REF_CLK_PAD_P;
wire          RX_CLK_G_net_0;
wire          RX_CLK_G_0_net_0;
wire   [3:0]  RXD;
wire   [3:0]  RXD_0;
wire   [3:0]  RXD_N;
wire   [3:0]  RXD_N_0;
wire          SS;
wire          CAM2_CLK_net_1;
wire          CAM1_EN_net_1;
wire          CAM2_CLK_net_2;
wire          CAM2_EN_net_1;
wire          DO_net_1;
wire          LANE0_TXD_N_0_net_1;
wire          LANE0_TXD_N_net_1;
wire          LANE0_TXD_P_0_net_1;
wire          LANE0_TXD_P_net_1;
wire          LED1_net_1;
wire          RX_CLK_G_0_net_1;
wire          RX_CLK_G_net_1;
wire          c1_frame_valid_o_net_1;
wire          pps_stretch_o_net_1;
//--------------------------------------------------------------------
// TiedOff Nets
//--------------------------------------------------------------------
wire          GND_net;
wire          VCC_net;
//--------------------------------------------------------------------
// Bus Interface Nets Declarations - Unequal Pin Widths
//--------------------------------------------------------------------
wire   [3:0]  IMX477_IF_TOP_0_BIF_1_TUSER;
wire   [0:0]  IMX477_IF_TOP_0_BIF_1_TUSER_0;
wire   [0:0]  IMX477_IF_TOP_0_BIF_1_TUSER_0_0to0;
wire   [3:0]  IMX477_IF_TOP_1_BIF_1_TUSER;
wire   [0:0]  IMX477_IF_TOP_1_BIF_1_TUSER_0;
wire   [0:0]  IMX477_IF_TOP_1_BIF_1_TUSER_0_0to0;
//--------------------------------------------------------------------
// Constant assignments
//--------------------------------------------------------------------
assign GND_net = 1'b0;
assign VCC_net = 1'b1;
//--------------------------------------------------------------------
// TieOff assignments
//--------------------------------------------------------------------
assign TEN                    = 1'b0;
//--------------------------------------------------------------------
// Top level output port assignments
//--------------------------------------------------------------------
assign CAM2_CLK_net_1         = CAM2_CLK_net_0;
assign CAM1_CLK               = CAM2_CLK_net_1;
assign CAM1_EN_net_1          = CAM1_EN_net_0;
assign CAM1_EN                = CAM1_EN_net_1;
assign CAM2_CLK_net_2         = CAM2_CLK_net_0;
assign CAM2_CLK               = CAM2_CLK_net_2;
assign CAM2_EN_net_1          = CAM2_EN_net_0;
assign CAM2_EN                = CAM2_EN_net_1;
assign DO_net_1               = DO_net_0;
assign DO                     = DO_net_1;
assign LANE0_TXD_N_0_net_1    = LANE0_TXD_N_0_net_0;
assign LANE0_TXD_N_0          = LANE0_TXD_N_0_net_1;
assign LANE0_TXD_N_net_1      = LANE0_TXD_N_net_0;
assign LANE0_TXD_N            = LANE0_TXD_N_net_1;
assign LANE0_TXD_P_0_net_1    = LANE0_TXD_P_0_net_0;
assign LANE0_TXD_P_0          = LANE0_TXD_P_0_net_1;
assign LANE0_TXD_P_net_1      = LANE0_TXD_P_net_0;
assign LANE0_TXD_P            = LANE0_TXD_P_net_1;
assign LED1_net_1             = LED1_net_0;
assign LED1                   = LED1_net_1;
assign RX_CLK_G_0_net_1       = RX_CLK_G_0_net_0;
assign RX_CLK_G_0             = RX_CLK_G_0_net_1;
assign RX_CLK_G_net_1         = RX_CLK_G_net_0;
assign RX_CLK_G               = RX_CLK_G_net_1;
assign c1_frame_valid_o_net_1 = c1_frame_valid_o_net_0;
assign c1_frame_valid_o       = c1_frame_valid_o_net_1;
assign pps_stretch_o_net_1    = pps_stretch_o_net_0;
assign pps_stretch_o          = pps_stretch_o_net_1;
//--------------------------------------------------------------------
// Bus Interface Nets Assignments - Unequal Pin Widths
//--------------------------------------------------------------------
assign IMX477_IF_TOP_0_BIF_1_TUSER_0 = { IMX477_IF_TOP_0_BIF_1_TUSER_0_0to0 };
assign IMX477_IF_TOP_0_BIF_1_TUSER_0_0to0 = IMX477_IF_TOP_0_BIF_1_TUSER[0:0];

assign IMX477_IF_TOP_1_BIF_1_TUSER_0 = { IMX477_IF_TOP_1_BIF_1_TUSER_0_0to0 };
assign IMX477_IF_TOP_1_BIF_1_TUSER_0_0to0 = IMX477_IF_TOP_1_BIF_1_TUSER[0:0];

//--------------------------------------------------------------------
// Component instances
//--------------------------------------------------------------------
//--------CORERESET_PF_C1
CORERESET_PF_C1 CORERESET_PF_C1_0(
        // Inputs
        .CLK                ( MAC_Hololink_0_OUT2_FABCLK_0 ),
        .EXT_RST_N          ( VCC_net ),
        .BANK_x_VDDI_STATUS ( VCC_net ),
        .BANK_y_VDDI_STATUS ( VCC_net ),
        .PLL_LOCK           ( PF_CCC_C3_0_PLL_LOCK_0 ),
        .SS_BUSY            ( GND_net ),
        .INIT_DONE          ( PF_INIT_MONITOR_C0_0_DEVICE_INIT_DONE ),
        .FF_US_RESTORE      ( GND_net ),
        .FPGA_POR_N         ( PF_INIT_MONITOR_C0_0_FABRIC_POR_N ),
        // Outputs
        .PLL_POWERDOWN_B    (  ),
        .FABRIC_RESET_N     ( CORERESET_PF_C1_0_FABRIC_RESET_N ) 
        );

//--------IMX477_IF_TOP
IMX477_IF_TOP IMX477_IF_TOP_0(
        // Inputs
        .ARST_N           ( PF_INIT_MONITOR_C0_0_AUTOCALIB_DONE ),
        .BIF_1_TREADY_I   ( IMX477_IF_TOP_0_BIF_1_TREADY ),
        .CAM1_RX_CLK_N    ( CAM1_RX_CLK_N ),
        .CAM1_RX_CLK_P    ( CAM1_RX_CLK_P ),
        .FPGA_POR_N       ( PF_INIT_MONITOR_C0_0_FABRIC_POR_N ),
        .INIT_DONE        ( PF_INIT_MONITOR_C0_0_DEVICE_INIT_DONE ),
        .RESET_N          ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .TRNG_RST_N       ( MAC_Hololink_0_CAM1_TRNG_RSTN ),
        .sif_clk          ( MAC_Hololink_0_OUT2_FABCLK_0 ),
        .RXD_N            ( RXD_N ),
        .RXD              ( RXD ),
        // Outputs
        .BIF_1_TLAST_O    ( IMX477_IF_TOP_0_BIF_1_TLAST ),
        .BIF_1_TVALID_O   ( IMX477_IF_TOP_0_BIF_1_TVALID ),
        .RX_CLK_G         ( RX_CLK_G_net_0 ),
        .c1_frame_valid_o (  ),
        .BIF_1_TDATA_O    ( IMX477_IF_TOP_0_BIF_1_TDATA ),
        .BIF_1_TKEEP_O    ( IMX477_IF_TOP_0_BIF_1_TKEEP ),
        .BIF_1_TSTRB_O    ( IMX477_IF_TOP_0_BIF_1_TSTRB ),
        .BIF_1_TUSER_O    ( IMX477_IF_TOP_0_BIF_1_TUSER ) 
        );

//--------IMX477_IF_TOP
IMX477_IF_TOP IMX477_IF_TOP_1(
        // Inputs
        .ARST_N           ( PF_INIT_MONITOR_C0_0_AUTOCALIB_DONE ),
        .BIF_1_TREADY_I   ( IMX477_IF_TOP_1_BIF_1_TREADY ),
        .CAM1_RX_CLK_N    ( CAM1_RX_CLK_N_0 ),
        .CAM1_RX_CLK_P    ( CAM1_RX_CLK_P_0 ),
        .FPGA_POR_N       ( PF_INIT_MONITOR_C0_0_FABRIC_POR_N ),
        .INIT_DONE        ( PF_INIT_MONITOR_C0_0_DEVICE_INIT_DONE ),
        .RESET_N          ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .TRNG_RST_N       ( MAC_Hololink_0_CAM2_TRNG_RSTN ),
        .sif_clk          ( MAC_Hololink_0_OUT2_FABCLK_0 ),
        .RXD_N            ( RXD_N_0 ),
        .RXD              ( RXD_0 ),
        // Outputs
        .BIF_1_TLAST_O    ( IMX477_IF_TOP_1_BIF_1_TLAST ),
        .BIF_1_TVALID_O   ( IMX477_IF_TOP_1_BIF_1_TVALID ),
        .RX_CLK_G         ( RX_CLK_G_0_net_0 ),
        .c1_frame_valid_o ( c1_frame_valid_o_net_0 ),
        .BIF_1_TDATA_O    ( IMX477_IF_TOP_1_BIF_1_TDATA ),
        .BIF_1_TKEEP_O    ( IMX477_IF_TOP_1_BIF_1_TKEEP ),
        .BIF_1_TSTRB_O    ( IMX477_IF_TOP_1_BIF_1_TSTRB ),
        .BIF_1_TUSER_O    ( IMX477_IF_TOP_1_BIF_1_TUSER ) 
        );

//--------MAC_Hololink
MAC_Hololink MAC_Hololink_0(
        // Inputs
        .DI                               ( DI ),
        .Device_Init_Done                 ( PF_INIT_MONITOR_C0_0_DEVICE_INIT_DONE ),
        .FLASH                            ( FLASH ),
        .IFACE                            ( IFACE ),
        .LANE0_RXD_N_0                    ( LANE0_RXD_N_0 ),
        .LANE0_RXD_N                      ( LANE0_RXD_N ),
        .LANE0_RXD_P_0                    ( LANE0_RXD_P_0 ),
        .LANE0_RXD_P                      ( LANE0_RXD_P ),
        .REF_CLK_0                        ( CLK_IN ),
        .REF_CLK_PAD_N                    ( REF_CLK_PAD_N ),
        .REF_CLK_PAD_P                    ( REF_CLK_PAD_P ),
        .i_sif_clk                        ( MAC_Hololink_0_OUT2_FABCLK_0 ),
        .SIF_TARGET_0_i_sif_axis_tvalid_0 ( IMX477_IF_TOP_0_BIF_1_TVALID ),
        .SIF_TARGET_0_i_sif_axis_tlast_0  ( IMX477_IF_TOP_0_BIF_1_TLAST ),
        .SIF_TARGET_1_i_sif_axis_tvalid_1 ( IMX477_IF_TOP_1_BIF_1_TVALID ),
        .SIF_TARGET_1_i_sif_axis_tlast_1  ( IMX477_IF_TOP_1_BIF_1_TLAST ),
        .i_ptp_clk                        ( PF_CCC_C3_0_OUT1_FABCLK_0 ),
        .SIF_TARGET_0_i_sif_axis_tdata_0  ( IMX477_IF_TOP_0_BIF_1_TDATA ),
        .SIF_TARGET_0_i_sif_axis_tkeep_0  ( IMX477_IF_TOP_0_BIF_1_TKEEP ),
        .SIF_TARGET_0_i_sif_axis_tuser_0  ( IMX477_IF_TOP_0_BIF_1_TUSER_0 ),
        .SIF_TARGET_1_i_sif_axis_tdata_1  ( IMX477_IF_TOP_1_BIF_1_TDATA ),
        .SIF_TARGET_1_i_sif_axis_tkeep_1  ( IMX477_IF_TOP_1_BIF_1_TKEEP ),
        .SIF_TARGET_1_i_sif_axis_tuser_1  ( IMX477_IF_TOP_1_BIF_1_TUSER_0 ),
        // Outputs
        .CAM1_EN                          ( CAM1_EN_net_0 ),
        .CAM1_TRNG_RSTN                   ( MAC_Hololink_0_CAM1_TRNG_RSTN ),
        .CAM2_EN                          ( CAM2_EN_net_0 ),
        .CAM2_TRNG_RSTN                   ( MAC_Hololink_0_CAM2_TRNG_RSTN ),
        .DO                               ( DO_net_0 ),
        .LANE0_TXD_N_0                    ( LANE0_TXD_N_0_net_0 ),
        .LANE0_TXD_N                      ( LANE0_TXD_N_net_0 ),
        .LANE0_TXD_P_0                    ( LANE0_TXD_P_0_net_0 ),
        .LANE0_TXD_P                      ( LANE0_TXD_P_net_0 ),
        .LED1                             ( LED1_net_0 ),
        .OUT2_FABCLK_0                    ( MAC_Hololink_0_OUT2_FABCLK_0 ),
        .pps_stretch_o                    ( pps_stretch_o_net_0 ),
        .SIF_TARGET_0_o_sif_axis_tready_0 ( IMX477_IF_TOP_0_BIF_1_TREADY ),
        .SIF_TARGET_1_o_sif_axis_tready_1 ( IMX477_IF_TOP_1_BIF_1_TREADY ),
        .LANE0_RX_CLK_R                   ( MAC_Hololink_0_LANE0_RX_CLK_R ),
        // Inouts
        .CAM1_SCL                         ( CAM1_SCL ),
        .CAM1_SDA                         ( CAM1_SDA ),
        .CAM2_SCL                         ( CAM2_SCL ),
        .CAM2_SDA                         ( CAM2_SDA ),
        .CLK                              ( CLK ),
        .EEPROM_SCL                       ( EEPROM_SCL ),
        .EEPROM_SDA                       ( EEPROM_SDA ),
        .SS                               ( SS ),
        .GPIO                             ( GPIO ) 
        );

//--------PF_CCC_C3
PF_CCC_C3 PF_CCC_C3_0(
        // Inputs
        .REF_CLK_0     ( MAC_Hololink_0_LANE0_RX_CLK_R ),
        // Outputs
        .OUT0_FABCLK_0 ( CAM2_CLK_net_0 ),
        .OUT1_FABCLK_0 ( PF_CCC_C3_0_OUT1_FABCLK_0 ),
        .PLL_LOCK_0    ( PF_CCC_C3_0_PLL_LOCK_0 ) 
        );

//--------PF_INIT_MONITOR_C0
PF_INIT_MONITOR_C0 PF_INIT_MONITOR_C0_0(
        // Outputs
        .FABRIC_POR_N               ( PF_INIT_MONITOR_C0_0_FABRIC_POR_N ),
        .PCIE_INIT_DONE             (  ),
        .USRAM_INIT_DONE            (  ),
        .SRAM_INIT_DONE             (  ),
        .DEVICE_INIT_DONE           ( PF_INIT_MONITOR_C0_0_DEVICE_INIT_DONE ),
        .XCVR_INIT_DONE             (  ),
        .USRAM_INIT_FROM_SNVM_DONE  (  ),
        .USRAM_INIT_FROM_UPROM_DONE (  ),
        .USRAM_INIT_FROM_SPI_DONE   (  ),
        .SRAM_INIT_FROM_SNVM_DONE   (  ),
        .SRAM_INIT_FROM_UPROM_DONE  (  ),
        .SRAM_INIT_FROM_SPI_DONE    (  ),
        .AUTOCALIB_DONE             ( PF_INIT_MONITOR_C0_0_AUTOCALIB_DONE ) 
        );


endmodule
