//////////////////////////////////////////////////////////////////////
// Created by SmartDesign Wed Aug 13 10:29:16 2025
// Version: 2025.1 2025.1.0.14
//////////////////////////////////////////////////////////////////////

`timescale 1ns / 100ps

// MAC_Hololink
module MAC_Hololink(
    // Inputs
    DI,
    Device_Init_Done,
    FLASH,
    IFACE,
    LANE0_RXD_N,
    LANE0_RXD_N_0,
    LANE0_RXD_P,
    LANE0_RXD_P_0,
    REF_CLK_0,
    REF_CLK_PAD_N,
    REF_CLK_PAD_P,
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
    i_ptp_clk,
    i_sif_clk,
    // Outputs
    CAM1_EN,
    CAM1_TRNG_RSTN,
    CAM2_EN,
    CAM2_TRNG_RSTN,
    DO,
    LANE0_RX_CLK_R,
    LANE0_TXD_N,
    LANE0_TXD_N_0,
    LANE0_TXD_P,
    LANE0_TXD_P_0,
    LED1,
    OUT2_FABCLK_0,
    SIF_TARGET_0_o_sif_axis_tready_0,
    SIF_TARGET_1_o_sif_axis_tready_1,
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
input         DI;
input         Device_Init_Done;
input         FLASH;
input         IFACE;
input         LANE0_RXD_N;
input         LANE0_RXD_N_0;
input         LANE0_RXD_P;
input         LANE0_RXD_P_0;
input         REF_CLK_0;
input         REF_CLK_PAD_N;
input         REF_CLK_PAD_P;
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
input         i_ptp_clk;
input         i_sif_clk;
//--------------------------------------------------------------------
// Output
//--------------------------------------------------------------------
output        CAM1_EN;
output        CAM1_TRNG_RSTN;
output        CAM2_EN;
output        CAM2_TRNG_RSTN;
output        DO;
output        LANE0_RX_CLK_R;
output        LANE0_TXD_N;
output        LANE0_TXD_N_0;
output        LANE0_TXD_P;
output        LANE0_TXD_P_0;
output        LED1;
output        OUT2_FABCLK_0;
output        SIF_TARGET_0_o_sif_axis_tready_0;
output        SIF_TARGET_1_o_sif_axis_tready_1;
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
wire          AND2_0_Y;
wire          AND2_1_Y;
wire          AND2_2_Y;
wire          AND2_3_Y;
wire          CAM1_EN_net_0;
wire          CAM1_SCL;
wire          CAM1_SDA;
wire          CAM2_SCL;
wire          CAM2_SDA;
wire          CLK;
wire   [31:0] CoreAPB3_C1_0_APBmslave1_PADDR;
wire          CoreAPB3_C1_0_APBmslave1_PENABLE;
wire   [31:0] CoreAPB3_C1_0_APBmslave1_PRDATA;
wire          CoreAPB3_C1_0_APBmslave1_PREADY;
wire          CoreAPB3_C1_0_APBmslave1_PSELx;
wire          CoreAPB3_C1_0_APBmslave1_PSLVERR;
wire   [31:0] CoreAPB3_C1_0_APBmslave1_PWDATA;
wire          CoreAPB3_C1_0_APBmslave1_PWRITE;
wire          CORERESET_PF_C1_0_FABRIC_RESET_N;
wire          Device_Init_Done;
wire          DI;
wire          DO_net_0;
wire          EEPROM_SCL;
wire          EEPROM_SDA;
wire          FLASH;
wire   [15:0] GPIO;
wire   [31:0] HoloLink_SD_0_APB_INITIATOR_PADDR;
wire          HoloLink_SD_0_APB_INITIATOR_PENABLE;
wire   [31:0] HoloLink_SD_0_APB_INITIATOR_PRDATA;
wire          HoloLink_SD_0_APB_INITIATOR_PREADY;
wire          HoloLink_SD_0_APB_INITIATOR_PSELx;
wire          HoloLink_SD_0_APB_INITIATOR_PSLVERR;
wire   [31:0] HoloLink_SD_0_APB_INITIATOR_PWDATA;
wire          HoloLink_SD_0_APB_INITIATOR_PWRITE;
wire   [63:0] HoloLink_SD_0_HIF_INITIATOR_0_TDATA;
wire   [7:0]  HoloLink_SD_0_HIF_INITIATOR_0_TKEEP;
wire          HoloLink_SD_0_HIF_INITIATOR_0_TLAST;
wire          HoloLink_SD_0_HIF_INITIATOR_0_TREADY;
wire          HoloLink_SD_0_HIF_INITIATOR_0_TVALID;
wire   [63:0] HoloLink_SD_0_HIF_INITIATOR_1_TDATA;
wire   [7:0]  HoloLink_SD_0_HIF_INITIATOR_1_TKEEP;
wire          HoloLink_SD_0_HIF_INITIATOR_1_TLAST;
wire          HoloLink_SD_0_HIF_INITIATOR_1_TREADY;
wire          HoloLink_SD_0_HIF_INITIATOR_1_TVALID;
wire          HoloLink_SD_0_o_pps;
wire          i_ptp_clk;
wire          i_sif_clk;
wire          IFACE;
wire          LANE0_RX_CLK_R_net_0;
wire          LANE0_RXD_N;
wire          LANE0_RXD_N_0;
wire          LANE0_RXD_P;
wire          LANE0_RXD_P_0;
wire          LANE0_TXD_N_net_0;
wire          LANE0_TXD_N_0_net_0;
wire          LANE0_TXD_P_net_0;
wire          LANE0_TXD_P_0_net_0;
wire          LED1_net_0;
wire   [63:0] MAC_BaseR_0_AXI4S_INITR_TDATA;
wire   [7:0]  MAC_BaseR_0_AXI4S_INITR_TKEEP;
wire          MAC_BaseR_0_AXI4S_INITR_TLAST;
wire          MAC_BaseR_0_AXI4S_INITR_TREADY;
wire          MAC_BaseR_0_AXI4S_INITR_TVALID;
wire          MAC_BaseR_0_LANE0_RX_VAL;
wire          MAC_BaseR_0_LANE0_TX_CLK_STABLE;
wire   [63:0] MAC_BaseR_1_AXI4S_INITR_TDATA;
wire   [7:0]  MAC_BaseR_1_AXI4S_INITR_TKEEP;
wire          MAC_BaseR_1_AXI4S_INITR_TLAST;
wire          MAC_BaseR_1_AXI4S_INITR_TREADY;
wire          MAC_BaseR_1_AXI4S_INITR_TVALID;
wire          MAC_BaseR_1_LANE0_RX_VAL;
wire          MAC_BaseR_1_LANE0_TX_CLK_STABLE;
wire          OUT2_FABCLK_0_net_0;
wire          PF_CCC_C1_0_OUT0_FABCLK_0;
wire          PF_CCC_C1_0_OUT1_FABCLK_0;
wire          PF_CCC_C1_0_PLL_LOCK_0;
wire          PF_XCVR_REF_CLK_C0_0_REF_CLK;
wire          REF_CLK_0;
wire          REF_CLK_PAD_N;
wire          REF_CLK_PAD_P;
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
wire          CAM1_EN_net_1;
wire          CAM1_EN_net_2;
wire          CAM1_EN_net_3;
wire          CAM1_EN_net_4;
wire          DO_net_1;
wire          LANE0_TXD_N_0_net_1;
wire          LANE0_TXD_N_net_1;
wire          LANE0_TXD_P_0_net_1;
wire          LANE0_TXD_P_net_1;
wire          LED1_net_1;
wire          OUT2_FABCLK_0_net_1;
wire          LED1_net_2;
wire          SIF_TARGET_0_TREADY_net_0;
wire          SIF_TARGET_1_TREADY_net_0;
wire          LANE0_RX_CLK_R_net_1;
//--------------------------------------------------------------------
// TiedOff Nets
//--------------------------------------------------------------------
wire          VCC_net;
wire          GND_net;
wire   [31:0] PRDATAS0_const_net_0;
//--------------------------------------------------------------------
// Inverted Nets
//--------------------------------------------------------------------
wire          i_sys_rst_IN_POST_INV0_0;
//--------------------------------------------------------------------
// Bus Interface Nets Declarations - Unequal Pin Widths
//--------------------------------------------------------------------
wire   [0:0]  HoloLink_SD_0_HIF_INITIATOR_0_TUSER;
wire   [2:0]  HoloLink_SD_0_HIF_INITIATOR_0_TUSER_0;
wire   [0:0]  HoloLink_SD_0_HIF_INITIATOR_0_TUSER_0_0to0;
wire   [2:1]  HoloLink_SD_0_HIF_INITIATOR_0_TUSER_0_2to1;
wire   [0:0]  HoloLink_SD_0_HIF_INITIATOR_1_TUSER;
wire   [2:0]  HoloLink_SD_0_HIF_INITIATOR_1_TUSER_0;
wire   [0:0]  HoloLink_SD_0_HIF_INITIATOR_1_TUSER_0_0to0;
wire   [2:1]  HoloLink_SD_0_HIF_INITIATOR_1_TUSER_0_2to1;
wire   [7:0]  MAC_BaseR_0_AXI4S_INITR_TUSER;
wire   [0:0]  MAC_BaseR_0_AXI4S_INITR_TUSER_0;
wire   [0:0]  MAC_BaseR_0_AXI4S_INITR_TUSER_0_0to0;
wire   [7:0]  MAC_BaseR_1_AXI4S_INITR_TUSER;
wire   [0:0]  MAC_BaseR_1_AXI4S_INITR_TUSER_0;
wire   [0:0]  MAC_BaseR_1_AXI4S_INITR_TUSER_0_0to0;
//--------------------------------------------------------------------
// Constant assignments
//--------------------------------------------------------------------
assign VCC_net              = 1'b1;
assign GND_net              = 1'b0;
assign PRDATAS0_const_net_0 = 32'h00000000;
//--------------------------------------------------------------------
// Inversions
//--------------------------------------------------------------------
assign i_sys_rst_IN_POST_INV0_0 = ~ CORERESET_PF_C1_0_FABRIC_RESET_N;
//--------------------------------------------------------------------
// Top level output port assignments
//--------------------------------------------------------------------
assign CAM1_EN_net_1                    = CAM1_EN_net_0;
assign CAM1_EN                          = CAM1_EN_net_1;
assign CAM1_EN_net_2                    = CAM1_EN_net_0;
assign CAM1_TRNG_RSTN                   = CAM1_EN_net_2;
assign CAM1_EN_net_3                    = CAM1_EN_net_0;
assign CAM2_EN                          = CAM1_EN_net_3;
assign CAM1_EN_net_4                    = CAM1_EN_net_0;
assign CAM2_TRNG_RSTN                   = CAM1_EN_net_4;
assign DO_net_1                         = DO_net_0;
assign DO                               = DO_net_1;
assign LANE0_TXD_N_0_net_1              = LANE0_TXD_N_0_net_0;
assign LANE0_TXD_N_0                    = LANE0_TXD_N_0_net_1;
assign LANE0_TXD_N_net_1                = LANE0_TXD_N_net_0;
assign LANE0_TXD_N                      = LANE0_TXD_N_net_1;
assign LANE0_TXD_P_0_net_1              = LANE0_TXD_P_0_net_0;
assign LANE0_TXD_P_0                    = LANE0_TXD_P_0_net_1;
assign LANE0_TXD_P_net_1                = LANE0_TXD_P_net_0;
assign LANE0_TXD_P                      = LANE0_TXD_P_net_1;
assign LED1_net_1                       = LED1_net_0;
assign LED1                             = LED1_net_1;
assign OUT2_FABCLK_0_net_1              = OUT2_FABCLK_0_net_0;
assign OUT2_FABCLK_0                    = OUT2_FABCLK_0_net_1;
assign LED1_net_2                       = LED1_net_0;
assign pps_stretch_o                    = LED1_net_2;
assign SIF_TARGET_0_TREADY_net_0        = SIF_TARGET_0_TREADY;
assign SIF_TARGET_0_o_sif_axis_tready_0 = SIF_TARGET_0_TREADY_net_0;
assign SIF_TARGET_1_TREADY_net_0        = SIF_TARGET_1_TREADY;
assign SIF_TARGET_1_o_sif_axis_tready_1 = SIF_TARGET_1_TREADY_net_0;
assign LANE0_RX_CLK_R_net_1             = LANE0_RX_CLK_R_net_0;
assign LANE0_RX_CLK_R                   = LANE0_RX_CLK_R_net_1;
//--------------------------------------------------------------------
// Bus Interface Nets Assignments - Unequal Pin Widths
//--------------------------------------------------------------------
assign HoloLink_SD_0_HIF_INITIATOR_0_TUSER_0 = { HoloLink_SD_0_HIF_INITIATOR_0_TUSER_0_2to1, HoloLink_SD_0_HIF_INITIATOR_0_TUSER_0_0to0 };
assign HoloLink_SD_0_HIF_INITIATOR_0_TUSER_0_0to0 = HoloLink_SD_0_HIF_INITIATOR_0_TUSER[0:0];
assign HoloLink_SD_0_HIF_INITIATOR_0_TUSER_0_2to1 = 2'h0;

assign HoloLink_SD_0_HIF_INITIATOR_1_TUSER_0 = { HoloLink_SD_0_HIF_INITIATOR_1_TUSER_0_2to1, HoloLink_SD_0_HIF_INITIATOR_1_TUSER_0_0to0 };
assign HoloLink_SD_0_HIF_INITIATOR_1_TUSER_0_0to0 = HoloLink_SD_0_HIF_INITIATOR_1_TUSER[0:0];
assign HoloLink_SD_0_HIF_INITIATOR_1_TUSER_0_2to1 = 2'h0;

assign MAC_BaseR_0_AXI4S_INITR_TUSER_0 = { MAC_BaseR_0_AXI4S_INITR_TUSER_0_0to0 };
assign MAC_BaseR_0_AXI4S_INITR_TUSER_0_0to0 = MAC_BaseR_0_AXI4S_INITR_TUSER[0:0];

assign MAC_BaseR_1_AXI4S_INITR_TUSER_0 = { MAC_BaseR_1_AXI4S_INITR_TUSER_0_0to0 };
assign MAC_BaseR_1_AXI4S_INITR_TUSER_0_0to0 = MAC_BaseR_1_AXI4S_INITR_TUSER[0:0];

//--------------------------------------------------------------------
// Component instances
//--------------------------------------------------------------------
//--------AND2
AND2 AND2_0(
        // Inputs
        .A ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .B ( MAC_BaseR_0_LANE0_TX_CLK_STABLE ),
        // Outputs
        .Y ( AND2_0_Y ) 
        );

//--------AND2
AND2 AND2_1(
        // Inputs
        .A ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .B ( MAC_BaseR_0_LANE0_RX_VAL ),
        // Outputs
        .Y ( AND2_1_Y ) 
        );

//--------AND2
AND2 AND2_2(
        // Inputs
        .A ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .B ( MAC_BaseR_1_LANE0_TX_CLK_STABLE ),
        // Outputs
        .Y ( AND2_2_Y ) 
        );

//--------AND2
AND2 AND2_3(
        // Inputs
        .A ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .B ( MAC_BaseR_1_LANE0_RX_VAL ),
        // Outputs
        .Y ( AND2_3_Y ) 
        );

//--------CoreAPB3_C1
CoreAPB3_C1 CoreAPB3_C1_0(
        // Inputs
        .PSEL      ( HoloLink_SD_0_APB_INITIATOR_PSELx ),
        .PENABLE   ( HoloLink_SD_0_APB_INITIATOR_PENABLE ),
        .PWRITE    ( HoloLink_SD_0_APB_INITIATOR_PWRITE ),
        .PREADYS0  ( VCC_net ), // tied to 1'b1 from definition
        .PSLVERRS0 ( GND_net ), // tied to 1'b0 from definition
        .PREADYS1  ( CoreAPB3_C1_0_APBmslave1_PREADY ),
        .PSLVERRS1 ( CoreAPB3_C1_0_APBmslave1_PSLVERR ),
        .PADDR     ( HoloLink_SD_0_APB_INITIATOR_PADDR ),
        .PWDATA    ( HoloLink_SD_0_APB_INITIATOR_PWDATA ),
        .PRDATAS0  ( PRDATAS0_const_net_0 ), // tied to 32'h00000000 from definition
        .PRDATAS1  ( CoreAPB3_C1_0_APBmslave1_PRDATA ),
        // Outputs
        .PREADY    ( HoloLink_SD_0_APB_INITIATOR_PREADY ),
        .PSLVERR   ( HoloLink_SD_0_APB_INITIATOR_PSLVERR ),
        .PSELS0    (  ),
        .PENABLES  ( CoreAPB3_C1_0_APBmslave1_PENABLE ),
        .PWRITES   ( CoreAPB3_C1_0_APBmslave1_PWRITE ),
        .PSELS1    ( CoreAPB3_C1_0_APBmslave1_PSELx ),
        .PRDATA    ( HoloLink_SD_0_APB_INITIATOR_PRDATA ),
        .PADDRS    ( CoreAPB3_C1_0_APBmslave1_PADDR ),
        .PWDATAS   ( CoreAPB3_C1_0_APBmslave1_PWDATA ) 
        );

//--------CORERESET_PF_C1
CORERESET_PF_C1 CORERESET_PF_C1_0(
        // Inputs
        .CLK                ( PF_CCC_C1_0_OUT0_FABCLK_0 ),
        .EXT_RST_N          ( VCC_net ),
        .BANK_x_VDDI_STATUS ( VCC_net ),
        .BANK_y_VDDI_STATUS ( VCC_net ),
        .PLL_LOCK           ( PF_CCC_C1_0_PLL_LOCK_0 ),
        .SS_BUSY            ( GND_net ),
        .INIT_DONE          ( Device_Init_Done ),
        .FF_US_RESTORE      ( GND_net ),
        .FPGA_POR_N         ( VCC_net ),
        // Outputs
        .PLL_POWERDOWN_B    (  ),
        .FABRIC_RESET_N     ( CORERESET_PF_C1_0_FABRIC_RESET_N ) 
        );

//--------HoloLink_SD
HoloLink_SD HoloLink_SD_0(
        // Inputs
        .DI                                  ( DI ),
        .FLASH                               ( FLASH ),
        .IFACE                               ( IFACE ),
        .i_apb_clk                           ( PF_CCC_C1_0_OUT1_FABCLK_0 ),
        .i_hif_clk                           ( PF_CCC_C1_0_OUT0_FABCLK_0 ),
        .i_sif_clk                           ( i_sif_clk ),
        .i_sys_rst                           ( i_sys_rst_IN_POST_INV0_0 ),
        .HIF_INITIATOR_0_i_hif_axis_tready_0 ( HoloLink_SD_0_HIF_INITIATOR_0_TREADY ),
        .HIF_TARGET_0_i_hif_axis_tvalid_0    ( MAC_BaseR_0_AXI4S_INITR_TVALID ),
        .HIF_TARGET_0_i_hif_axis_tlast_0     ( MAC_BaseR_0_AXI4S_INITR_TLAST ),
        .SIF_TARGET_0_i_sif_axis_tvalid_0    ( SIF_TARGET_0_i_sif_axis_tvalid_0 ),
        .SIF_TARGET_0_i_sif_axis_tlast_0     ( SIF_TARGET_0_i_sif_axis_tlast_0 ),
        .HIF_INITIATOR_1_i_hif_axis_tready_1 ( HoloLink_SD_0_HIF_INITIATOR_1_TREADY ),
        .HIF_TARGET_1_i_hif_axis_tvalid_1    ( MAC_BaseR_1_AXI4S_INITR_TVALID ),
        .HIF_TARGET_1_i_hif_axis_tlast_1     ( MAC_BaseR_1_AXI4S_INITR_TLAST ),
        .SIF_TARGET_1_i_sif_axis_tvalid_1    ( SIF_TARGET_1_i_sif_axis_tvalid_1 ),
        .SIF_TARGET_1_i_sif_axis_tlast_1     ( SIF_TARGET_1_i_sif_axis_tlast_1 ),
        .APB_INITIATOR_i_apb_pready          ( HoloLink_SD_0_APB_INITIATOR_PREADY ),
        .APB_INITIATOR_i_apb_pserr           ( HoloLink_SD_0_APB_INITIATOR_PSLVERR ),
        .i_ptp_clk                           ( i_ptp_clk ),
        .HIF_TARGET_0_i_hif_axis_tdata_0     ( MAC_BaseR_0_AXI4S_INITR_TDATA ),
        .HIF_TARGET_0_i_hif_axis_tkeep_0     ( MAC_BaseR_0_AXI4S_INITR_TKEEP ),
        .HIF_TARGET_0_i_hif_axis_tuser_0     ( MAC_BaseR_0_AXI4S_INITR_TUSER_0 ),
        .SIF_TARGET_0_i_sif_axis_tdata_0     ( SIF_TARGET_0_i_sif_axis_tdata_0 ),
        .SIF_TARGET_0_i_sif_axis_tuser_0     ( SIF_TARGET_0_i_sif_axis_tuser_0 ),
        .SIF_TARGET_0_i_sif_axis_tkeep_0     ( SIF_TARGET_0_i_sif_axis_tkeep_0 ),
        .HIF_TARGET_1_i_hif_axis_tdata_1     ( MAC_BaseR_1_AXI4S_INITR_TDATA ),
        .HIF_TARGET_1_i_hif_axis_tkeep_1     ( MAC_BaseR_1_AXI4S_INITR_TKEEP ),
        .HIF_TARGET_1_i_hif_axis_tuser_1     ( MAC_BaseR_1_AXI4S_INITR_TUSER_0 ),
        .SIF_TARGET_1_i_sif_axis_tdata_1     ( SIF_TARGET_1_i_sif_axis_tdata_1 ),
        .SIF_TARGET_1_i_sif_axis_tuser_1     ( SIF_TARGET_1_i_sif_axis_tuser_1 ),
        .SIF_TARGET_1_i_sif_axis_tkeep_1     ( SIF_TARGET_1_i_sif_axis_tkeep_1 ),
        .APB_INITIATOR_i_apb_prdata          ( HoloLink_SD_0_APB_INITIATOR_PRDATA ),
        // Outputs
        .DO                                  ( DO_net_0 ),
        .o_init_done                         ( CAM1_EN_net_0 ),
        .o_pps                               ( HoloLink_SD_0_o_pps ),
        .HIF_INITIATOR_0_o_hif_axis_tvalid_0 ( HoloLink_SD_0_HIF_INITIATOR_0_TVALID ),
        .HIF_INITIATOR_0_o_hif_axis_tlast_0  ( HoloLink_SD_0_HIF_INITIATOR_0_TLAST ),
        .HIF_TARGET_0_o_hif_axis_tready_0    ( MAC_BaseR_0_AXI4S_INITR_TREADY ),
        .SIF_TARGET_0_o_sif_axis_tready_0    ( SIF_TARGET_0_TREADY ),
        .HIF_INITIATOR_1_o_hif_axis_tvalid_1 ( HoloLink_SD_0_HIF_INITIATOR_1_TVALID ),
        .HIF_INITIATOR_1_o_hif_axis_tlast_1  ( HoloLink_SD_0_HIF_INITIATOR_1_TLAST ),
        .HIF_TARGET_1_o_hif_axis_tready_1    ( MAC_BaseR_1_AXI4S_INITR_TREADY ),
        .SIF_TARGET_1_o_sif_axis_tready_1    ( SIF_TARGET_1_TREADY ),
        .APB_INITIATOR_o_apb_penable         ( HoloLink_SD_0_APB_INITIATOR_PENABLE ),
        .APB_INITIATOR_o_apb_pwrite          ( HoloLink_SD_0_APB_INITIATOR_PWRITE ),
        .APB_INITIATOR_o_apb_psel            ( HoloLink_SD_0_APB_INITIATOR_PSELx ),
        .o_ptp_rst                           (  ),
        .HIF_INITIATOR_0_o_hif_axis_tdata_0  ( HoloLink_SD_0_HIF_INITIATOR_0_TDATA ),
        .HIF_INITIATOR_0_o_hif_axis_tkeep_0  ( HoloLink_SD_0_HIF_INITIATOR_0_TKEEP ),
        .HIF_INITIATOR_0_o_hif_axis_tuser_0  ( HoloLink_SD_0_HIF_INITIATOR_0_TUSER ),
        .HIF_INITIATOR_1_o_hif_axis_tdata_1  ( HoloLink_SD_0_HIF_INITIATOR_1_TDATA ),
        .HIF_INITIATOR_1_o_hif_axis_tkeep_1  ( HoloLink_SD_0_HIF_INITIATOR_1_TKEEP ),
        .HIF_INITIATOR_1_o_hif_axis_tuser_1  ( HoloLink_SD_0_HIF_INITIATOR_1_TUSER ),
        .APB_INITIATOR_o_apb_paddr           ( HoloLink_SD_0_APB_INITIATOR_PADDR ),
        .APB_INITIATOR_o_apb_pwdata          ( HoloLink_SD_0_APB_INITIATOR_PWDATA ),
        // Inouts
        .CAM1_SCL                            ( CAM1_SCL ),
        .CAM1_SDA                            ( CAM1_SDA ),
        .CAM2_SCL                            ( CAM2_SCL ),
        .CAM2_SDA                            ( CAM2_SDA ),
        .CLK                                 ( CLK ),
        .EEPROM_SCL                          ( EEPROM_SCL ),
        .EEPROM_SDA                          ( EEPROM_SDA ),
        .SS                                  ( SS ),
        .GPIO                                ( GPIO ) 
        );

//--------MAC_BaseR
MAC_BaseR MAC_BaseR_0(
        // Inputs
        .AXI4S_INITR_AXI4S_DT_INITR_TREADY ( MAC_BaseR_0_AXI4S_INITR_TREADY ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TLAST    ( HoloLink_SD_0_HIF_INITIATOR_0_TLAST ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TVALID   ( HoloLink_SD_0_HIF_INITIATOR_0_TVALID ),
        .Device_Init_Done                  ( Device_Init_Done ),
        .I_SYS_CLK                         ( PF_CCC_C1_0_OUT0_FABCLK_0 ),
        .I_SYS_RX_SRESETN                  ( AND2_1_Y ),
        .I_SYS_TX_SRESETN                  ( AND2_0_Y ),
        .LANE0_RXD_N                       ( LANE0_RXD_N ),
        .LANE0_RXD_P                       ( LANE0_RXD_P ),
        .PCLK                              ( PF_CCC_C1_0_OUT1_FABCLK_0 ),
        .PRESETN                           ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .REF_CLK                           ( PF_XCVR_REF_CLK_C0_0_REF_CLK ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TDATA    ( HoloLink_SD_0_HIF_INITIATOR_0_TDATA ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TKEEP    ( HoloLink_SD_0_HIF_INITIATOR_0_TKEEP ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TUSER    ( HoloLink_SD_0_HIF_INITIATOR_0_TUSER_0 ),
        // Outputs
        .AXI4S_INITR_AXI4S_DT_INITR_TLAST  ( MAC_BaseR_0_AXI4S_INITR_TLAST ),
        .AXI4S_INITR_AXI4S_DT_INITR_TVALID ( MAC_BaseR_0_AXI4S_INITR_TVALID ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TREADY   ( HoloLink_SD_0_HIF_INITIATOR_0_TREADY ),
        .LANE0_RX_VAL                      ( MAC_BaseR_0_LANE0_RX_VAL ),
        .LANE0_TXD_N                       ( LANE0_TXD_N_net_0 ),
        .LANE0_TXD_P                       ( LANE0_TXD_P_net_0 ),
        .LANE0_TX_CLK_STABLE               ( MAC_BaseR_0_LANE0_TX_CLK_STABLE ),
        .AXI4S_INITR_AXI4S_DT_INITR_TDATA  ( MAC_BaseR_0_AXI4S_INITR_TDATA ),
        .AXI4S_INITR_AXI4S_DT_INITR_TKEEP  ( MAC_BaseR_0_AXI4S_INITR_TKEEP ),
        .AXI4S_INITR_AXI4S_DT_INITR_TUSER  ( MAC_BaseR_0_AXI4S_INITR_TUSER ),
        .LANE0_RX_CLK_R                    ( LANE0_RX_CLK_R_net_0 ) 
        );

//--------MAC_BaseR
MAC_BaseR MAC_BaseR_1(
        // Inputs
        .AXI4S_INITR_AXI4S_DT_INITR_TREADY ( MAC_BaseR_1_AXI4S_INITR_TREADY ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TLAST    ( HoloLink_SD_0_HIF_INITIATOR_1_TLAST ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TVALID   ( HoloLink_SD_0_HIF_INITIATOR_1_TVALID ),
        .Device_Init_Done                  ( Device_Init_Done ),
        .I_SYS_CLK                         ( PF_CCC_C1_0_OUT0_FABCLK_0 ),
        .I_SYS_RX_SRESETN                  ( AND2_3_Y ),
        .I_SYS_TX_SRESETN                  ( AND2_2_Y ),
        .LANE0_RXD_N                       ( LANE0_RXD_N_0 ),
        .LANE0_RXD_P                       ( LANE0_RXD_P_0 ),
        .PCLK                              ( PF_CCC_C1_0_OUT1_FABCLK_0 ),
        .PRESETN                           ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .REF_CLK                           ( PF_XCVR_REF_CLK_C0_0_REF_CLK ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TDATA    ( HoloLink_SD_0_HIF_INITIATOR_1_TDATA ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TKEEP    ( HoloLink_SD_0_HIF_INITIATOR_1_TKEEP ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TUSER    ( HoloLink_SD_0_HIF_INITIATOR_1_TUSER_0 ),
        // Outputs
        .AXI4S_INITR_AXI4S_DT_INITR_TLAST  ( MAC_BaseR_1_AXI4S_INITR_TLAST ),
        .AXI4S_INITR_AXI4S_DT_INITR_TVALID ( MAC_BaseR_1_AXI4S_INITR_TVALID ),
        .AXI4S_TRGT_AXI4S_DT_TARG_TREADY   ( HoloLink_SD_0_HIF_INITIATOR_1_TREADY ),
        .LANE0_RX_VAL                      ( MAC_BaseR_1_LANE0_RX_VAL ),
        .LANE0_TXD_N                       ( LANE0_TXD_N_0_net_0 ),
        .LANE0_TXD_P                       ( LANE0_TXD_P_0_net_0 ),
        .LANE0_TX_CLK_STABLE               ( MAC_BaseR_1_LANE0_TX_CLK_STABLE ),
        .AXI4S_INITR_AXI4S_DT_INITR_TDATA  ( MAC_BaseR_1_AXI4S_INITR_TDATA ),
        .AXI4S_INITR_AXI4S_DT_INITR_TKEEP  ( MAC_BaseR_1_AXI4S_INITR_TKEEP ),
        .AXI4S_INITR_AXI4S_DT_INITR_TUSER  ( MAC_BaseR_1_AXI4S_INITR_TUSER ),
        .LANE0_RX_CLK_R                    (  ) 
        );

//--------PF_CCC_C1
PF_CCC_C1 PF_CCC_C1_0(
        // Inputs
        .REF_CLK_0     ( REF_CLK_0 ),
        // Outputs
        .OUT0_FABCLK_0 ( PF_CCC_C1_0_OUT0_FABCLK_0 ),
        .OUT1_FABCLK_0 ( PF_CCC_C1_0_OUT1_FABCLK_0 ),
        .OUT2_FABCLK_0 ( OUT2_FABCLK_0_net_0 ),
        .PLL_LOCK_0    ( PF_CCC_C1_0_PLL_LOCK_0 ) 
        );

//--------PF_SYSTEM_SERVICES_C0
PF_SYSTEM_SERVICES_C0 PF_SYSTEM_SERVICES_C0_0(
        // Inputs
        .CLK              ( PF_CCC_C1_0_OUT1_FABCLK_0 ),
        .RESETN           ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .APBS_PSEL        ( CoreAPB3_C1_0_APBmslave1_PSELx ),
        .APBS_PENABLE     ( CoreAPB3_C1_0_APBmslave1_PENABLE ),
        .APBS_PWRITE      ( CoreAPB3_C1_0_APBmslave1_PWRITE ),
        .APBS_PADDR       ( CoreAPB3_C1_0_APBmslave1_PADDR ),
        .APBS_PWDATA      ( CoreAPB3_C1_0_APBmslave1_PWDATA ),
        // Outputs
        .USR_CMD_ERROR    (  ),
        .USR_BUSY         (  ),
        .SS_BUSY          (  ),
        .USR_RDVLD        (  ),
        .SYSSERV_INIT_REQ (  ),
        .APBS_PREADY      ( CoreAPB3_C1_0_APBmslave1_PREADY ),
        .APBS_PSLVERR     ( CoreAPB3_C1_0_APBmslave1_PSLVERR ),
        .APBS_PRDATA      ( CoreAPB3_C1_0_APBmslave1_PRDATA ) 
        );

//--------PF_XCVR_REF_CLK_C0
PF_XCVR_REF_CLK_C0 PF_XCVR_REF_CLK_C0_0(
        // Inputs
        .REF_CLK_PAD_P ( REF_CLK_PAD_P ),
        .REF_CLK_PAD_N ( REF_CLK_PAD_N ),
        // Outputs
        .REF_CLK       ( PF_XCVR_REF_CLK_C0_0_REF_CLK ) 
        );

//--------pps_stretch
pps_stretch pps_stretch_0(
        // Inputs
        .hif_clk_i     ( PF_CCC_C1_0_OUT0_FABCLK_0 ),
        .rstn_i        ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .pps_i         ( HoloLink_SD_0_o_pps ),
        // Outputs
        .pps_stretch_o ( LED1_net_0 ) 
        );


endmodule
