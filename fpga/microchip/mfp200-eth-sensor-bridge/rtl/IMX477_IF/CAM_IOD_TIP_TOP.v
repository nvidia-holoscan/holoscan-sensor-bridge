//////////////////////////////////////////////////////////////////////
// Created by SmartDesign Sat Jul 26 13:16:23 2025
// Version: 2025.1 2025.1.0.14
//////////////////////////////////////////////////////////////////////

`timescale 1ns / 100ps

// CAM_IOD_TIP_TOP
module CAM_IOD_TIP_TOP(
    // Inputs
    ARST_N,
    FPGA_POR_N,
    HS_IO_CLK_PAUSE,
    HS_SEL,
    INIT_DONE,
    RXD,
    RXD_N,
    RX_CLK_N,
    RX_CLK_P,
    TRAINING_RESETN,
    // Outputs
    L0_LP_DATA,
    L0_LP_DATA_N,
    L0_RXD_DATA,
    L1_LP_DATA,
    L1_LP_DATA_N,
    L1_RXD_DATA,
    L2_LP_DATA,
    L2_LP_DATA_N,
    L2_RXD_DATA,
    L3_LP_DATA,
    L3_LP_DATA_N,
    L3_RXD_DATA,
    RX_CLK_G,
    training_done_o
);

//--------------------------------------------------------------------
// Input
//--------------------------------------------------------------------
input        ARST_N;
input        FPGA_POR_N;
input        HS_IO_CLK_PAUSE;
input        HS_SEL;
input        INIT_DONE;
input  [3:0] RXD;
input  [3:0] RXD_N;
input        RX_CLK_N;
input        RX_CLK_P;
input        TRAINING_RESETN;
//--------------------------------------------------------------------
// Output
//--------------------------------------------------------------------
output       L0_LP_DATA;
output       L0_LP_DATA_N;
output [7:0] L0_RXD_DATA;
output       L1_LP_DATA;
output       L1_LP_DATA_N;
output [7:0] L1_RXD_DATA;
output       L2_LP_DATA;
output       L2_LP_DATA_N;
output [7:0] L2_RXD_DATA;
output       L3_LP_DATA;
output       L3_LP_DATA_N;
output [7:0] L3_RXD_DATA;
output       RX_CLK_G;
output       training_done_o;
//--------------------------------------------------------------------
// Nets
//--------------------------------------------------------------------
wire         ARST_N;
wire         CORERESET_PF_C1_0_FABRIC_RESET_N;
wire         FPGA_POR_N;
wire         HS_IO_CLK_PAUSE;
wire         HS_SEL;
wire         INIT_DONE;
wire         L0_LP_DATA_net_0;
wire         L0_LP_DATA_N_net_0;
wire   [7:0] L0_RXD_DATA_net_0;
wire         L1_LP_DATA_net_0;
wire         L1_LP_DATA_N_net_0;
wire   [7:0] L1_RXD_DATA_net_0;
wire         L2_LP_DATA_net_0;
wire         L2_LP_DATA_N_net_0;
wire   [7:0] L2_RXD_DATA_net_0;
wire         L3_LP_DATA_net_0;
wire         L3_LP_DATA_N_net_0;
wire   [7:0] L3_RXD_DATA_net_0;
wire         MIPI_TRAINING_LITE_C0_0_BIT_ALGN_CLR_FLAGS_O;
wire         MIPI_TRAINING_LITE_C0_0_BIT_ALGN_DIRECTION_O;
wire         MIPI_TRAINING_LITE_C0_0_BIT_ALGN_LOAD_O;
wire         MIPI_TRAINING_LITE_C0_0_BIT_ALGN_MOVE_O;
wire         MIPI_TRAINING_LITE_C0_0_TRAINING_DONE_O;
wire         MIPI_TRAINING_LITE_C1_0_BIT_ALGN_CLR_FLAGS_O;
wire         MIPI_TRAINING_LITE_C1_0_BIT_ALGN_DIRECTION_O;
wire         MIPI_TRAINING_LITE_C1_0_BIT_ALGN_LOAD_O;
wire         MIPI_TRAINING_LITE_C1_0_BIT_ALGN_MOVE_O;
wire         MIPI_TRAINING_LITE_C1_0_TRAINING_DONE_O;
wire         MIPI_TRAINING_LITE_C2_0_BIT_ALGN_CLR_FLAGS_O;
wire         MIPI_TRAINING_LITE_C2_0_BIT_ALGN_DIRECTION_O;
wire         MIPI_TRAINING_LITE_C2_0_BIT_ALGN_LOAD_O;
wire         MIPI_TRAINING_LITE_C2_0_BIT_ALGN_MOVE_O;
wire         MIPI_TRAINING_LITE_C2_0_TRAINING_DONE_O;
wire         MIPI_TRAINING_LITE_C3_0_BIT_ALGN_CLR_FLAGS_O;
wire         MIPI_TRAINING_LITE_C3_0_BIT_ALGN_DIRECTION_O;
wire         MIPI_TRAINING_LITE_C3_0_BIT_ALGN_LOAD_O;
wire         MIPI_TRAINING_LITE_C3_0_BIT_ALGN_MOVE_O;
wire         MIPI_TRAINING_LITE_C3_0_TRAINING_DONE_O;
wire   [0:0] PF_IOD_0_DELAY_LINE_OUT_OF_RANGE0to0;
wire   [1:1] PF_IOD_0_DELAY_LINE_OUT_OF_RANGE1to1;
wire   [2:2] PF_IOD_0_DELAY_LINE_OUT_OF_RANGE2to2;
wire   [3:3] PF_IOD_0_DELAY_LINE_OUT_OF_RANGE3to3;
wire   [0:0] PF_IOD_0_EYE_MONITOR_EARLY0to0;
wire   [1:1] PF_IOD_0_EYE_MONITOR_EARLY1to1;
wire   [2:2] PF_IOD_0_EYE_MONITOR_EARLY2to2;
wire   [3:3] PF_IOD_0_EYE_MONITOR_EARLY3to3;
wire   [0:0] PF_IOD_0_EYE_MONITOR_LATE0to0;
wire   [1:1] PF_IOD_0_EYE_MONITOR_LATE1to1;
wire   [2:2] PF_IOD_0_EYE_MONITOR_LATE2to2;
wire   [3:3] PF_IOD_0_EYE_MONITOR_LATE3to3;
wire         RX_CLK_G_net_0;
wire         RX_CLK_N;
wire         RX_CLK_P;
wire   [3:0] RXD;
wire   [3:0] RXD_N;
wire         training_done_o_net_0;
wire         TRAINING_RESETN;
wire         L0_LP_DATA_N_net_1;
wire         L0_LP_DATA_net_1;
wire         L1_LP_DATA_N_net_1;
wire         L1_LP_DATA_net_1;
wire         L2_LP_DATA_N_net_1;
wire         L2_LP_DATA_net_1;
wire         L3_LP_DATA_N_net_1;
wire         L3_LP_DATA_net_1;
wire         RX_CLK_G_net_1;
wire         training_done_o_net_1;
wire   [7:0] L0_RXD_DATA_net_1;
wire   [7:0] L1_RXD_DATA_net_1;
wire   [7:0] L2_RXD_DATA_net_1;
wire   [7:0] L3_RXD_DATA_net_1;
wire   [3:0] EYE_MONITOR_CLEAR_FLAGS_net_0;
wire   [3:0] EYE_MONITOR_EARLY_net_0;
wire   [3:0] EYE_MONITOR_LATE_net_0;
wire   [3:0] DELAY_LINE_MOVE_net_0;
wire   [3:0] DELAY_LINE_DIRECTION_net_0;
wire   [3:0] DELAY_LINE_LOAD_net_0;
wire   [3:0] DELAY_LINE_OUT_OF_RANGE_net_0;
//--------------------------------------------------------------------
// TiedOff Nets
//--------------------------------------------------------------------
wire         VCC_net;
wire         GND_net;
wire   [7:0] TAP_DELAYS_I_const_net_0;
wire   [7:0] VALID_WIN_LEN_I_const_net_0;
wire   [3:0] FALSE_FLG_THRESHOLD_I_const_net_0;
wire   [7:0] TAP_DELAYS_I_const_net_1;
wire   [7:0] VALID_WIN_LEN_I_const_net_1;
wire   [3:0] FALSE_FLG_THRESHOLD_I_const_net_1;
wire   [7:0] TAP_DELAYS_I_const_net_2;
wire   [7:0] VALID_WIN_LEN_I_const_net_2;
wire   [3:0] FALSE_FLG_THRESHOLD_I_const_net_2;
wire   [7:0] TAP_DELAYS_I_const_net_3;
wire   [7:0] VALID_WIN_LEN_I_const_net_3;
wire   [3:0] FALSE_FLG_THRESHOLD_I_const_net_3;
wire   [2:0] EYE_MONITOR_WIDTH_const_net_0;
//--------------------------------------------------------------------
// Constant assignments
//--------------------------------------------------------------------
assign VCC_net                           = 1'b1;
assign GND_net                           = 1'b0;
assign TAP_DELAYS_I_const_net_0          = 8'h14;
assign VALID_WIN_LEN_I_const_net_0       = 8'h14;
assign FALSE_FLG_THRESHOLD_I_const_net_0 = 4'h5;
assign TAP_DELAYS_I_const_net_1          = 8'h14;
assign VALID_WIN_LEN_I_const_net_1       = 8'h14;
assign FALSE_FLG_THRESHOLD_I_const_net_1 = 4'h5;
assign TAP_DELAYS_I_const_net_2          = 8'h14;
assign VALID_WIN_LEN_I_const_net_2       = 8'h14;
assign FALSE_FLG_THRESHOLD_I_const_net_2 = 4'h5;
assign TAP_DELAYS_I_const_net_3          = 8'h14;
assign VALID_WIN_LEN_I_const_net_3       = 8'h14;
assign FALSE_FLG_THRESHOLD_I_const_net_3 = 4'h5;
assign EYE_MONITOR_WIDTH_const_net_0     = 3'h2;
//--------------------------------------------------------------------
// Top level output port assignments
//--------------------------------------------------------------------
assign L0_LP_DATA_N_net_1    = L0_LP_DATA_N_net_0;
assign L0_LP_DATA_N          = L0_LP_DATA_N_net_1;
assign L0_LP_DATA_net_1      = L0_LP_DATA_net_0;
assign L0_LP_DATA            = L0_LP_DATA_net_1;
assign L1_LP_DATA_N_net_1    = L1_LP_DATA_N_net_0;
assign L1_LP_DATA_N          = L1_LP_DATA_N_net_1;
assign L1_LP_DATA_net_1      = L1_LP_DATA_net_0;
assign L1_LP_DATA            = L1_LP_DATA_net_1;
assign L2_LP_DATA_N_net_1    = L2_LP_DATA_N_net_0;
assign L2_LP_DATA_N          = L2_LP_DATA_N_net_1;
assign L2_LP_DATA_net_1      = L2_LP_DATA_net_0;
assign L2_LP_DATA            = L2_LP_DATA_net_1;
assign L3_LP_DATA_N_net_1    = L3_LP_DATA_N_net_0;
assign L3_LP_DATA_N          = L3_LP_DATA_N_net_1;
assign L3_LP_DATA_net_1      = L3_LP_DATA_net_0;
assign L3_LP_DATA            = L3_LP_DATA_net_1;
assign RX_CLK_G_net_1        = RX_CLK_G_net_0;
assign RX_CLK_G              = RX_CLK_G_net_1;
assign training_done_o_net_1 = training_done_o_net_0;
assign training_done_o       = training_done_o_net_1;
assign L0_RXD_DATA_net_1     = L0_RXD_DATA_net_0;
assign L0_RXD_DATA[7:0]      = L0_RXD_DATA_net_1;
assign L1_RXD_DATA_net_1     = L1_RXD_DATA_net_0;
assign L1_RXD_DATA[7:0]      = L1_RXD_DATA_net_1;
assign L2_RXD_DATA_net_1     = L2_RXD_DATA_net_0;
assign L2_RXD_DATA[7:0]      = L2_RXD_DATA_net_1;
assign L3_RXD_DATA_net_1     = L3_RXD_DATA_net_0;
assign L3_RXD_DATA[7:0]      = L3_RXD_DATA_net_1;
//--------------------------------------------------------------------
// Slices assignments
//--------------------------------------------------------------------
assign PF_IOD_0_DELAY_LINE_OUT_OF_RANGE0to0[0] = DELAY_LINE_OUT_OF_RANGE_net_0[0:0];
assign PF_IOD_0_DELAY_LINE_OUT_OF_RANGE1to1[1] = DELAY_LINE_OUT_OF_RANGE_net_0[1:1];
assign PF_IOD_0_DELAY_LINE_OUT_OF_RANGE2to2[2] = DELAY_LINE_OUT_OF_RANGE_net_0[2:2];
assign PF_IOD_0_DELAY_LINE_OUT_OF_RANGE3to3[3] = DELAY_LINE_OUT_OF_RANGE_net_0[3:3];
assign PF_IOD_0_EYE_MONITOR_EARLY0to0[0]       = EYE_MONITOR_EARLY_net_0[0:0];
assign PF_IOD_0_EYE_MONITOR_EARLY1to1[1]       = EYE_MONITOR_EARLY_net_0[1:1];
assign PF_IOD_0_EYE_MONITOR_EARLY2to2[2]       = EYE_MONITOR_EARLY_net_0[2:2];
assign PF_IOD_0_EYE_MONITOR_EARLY3to3[3]       = EYE_MONITOR_EARLY_net_0[3:3];
assign PF_IOD_0_EYE_MONITOR_LATE0to0[0]        = EYE_MONITOR_LATE_net_0[0:0];
assign PF_IOD_0_EYE_MONITOR_LATE1to1[1]        = EYE_MONITOR_LATE_net_0[1:1];
assign PF_IOD_0_EYE_MONITOR_LATE2to2[2]        = EYE_MONITOR_LATE_net_0[2:2];
assign PF_IOD_0_EYE_MONITOR_LATE3to3[3]        = EYE_MONITOR_LATE_net_0[3:3];
//--------------------------------------------------------------------
// Concatenation assignments
//--------------------------------------------------------------------
assign EYE_MONITOR_CLEAR_FLAGS_net_0 = { MIPI_TRAINING_LITE_C3_0_BIT_ALGN_CLR_FLAGS_O , MIPI_TRAINING_LITE_C2_0_BIT_ALGN_CLR_FLAGS_O , MIPI_TRAINING_LITE_C1_0_BIT_ALGN_CLR_FLAGS_O , MIPI_TRAINING_LITE_C0_0_BIT_ALGN_CLR_FLAGS_O };
assign DELAY_LINE_MOVE_net_0         = { MIPI_TRAINING_LITE_C3_0_BIT_ALGN_MOVE_O , MIPI_TRAINING_LITE_C2_0_BIT_ALGN_MOVE_O , MIPI_TRAINING_LITE_C1_0_BIT_ALGN_MOVE_O , MIPI_TRAINING_LITE_C0_0_BIT_ALGN_MOVE_O };
assign DELAY_LINE_DIRECTION_net_0    = { MIPI_TRAINING_LITE_C3_0_BIT_ALGN_DIRECTION_O , MIPI_TRAINING_LITE_C2_0_BIT_ALGN_DIRECTION_O , MIPI_TRAINING_LITE_C1_0_BIT_ALGN_DIRECTION_O , MIPI_TRAINING_LITE_C0_0_BIT_ALGN_DIRECTION_O };
assign DELAY_LINE_LOAD_net_0         = { MIPI_TRAINING_LITE_C3_0_BIT_ALGN_LOAD_O , MIPI_TRAINING_LITE_C2_0_BIT_ALGN_LOAD_O , MIPI_TRAINING_LITE_C1_0_BIT_ALGN_LOAD_O , MIPI_TRAINING_LITE_C0_0_BIT_ALGN_LOAD_O };
//--------------------------------------------------------------------
// Component instances
//--------------------------------------------------------------------
//--------AND4
AND4 AND4_0(
        // Inputs
        .A ( MIPI_TRAINING_LITE_C0_0_TRAINING_DONE_O ),
        .B ( MIPI_TRAINING_LITE_C1_0_TRAINING_DONE_O ),
        .C ( MIPI_TRAINING_LITE_C2_0_TRAINING_DONE_O ),
        .D ( MIPI_TRAINING_LITE_C3_0_TRAINING_DONE_O ),
        // Outputs
        .Y ( training_done_o_net_0 ) 
        );

//--------CORERESET_PF_C1
CORERESET_PF_C1 CORERESET_PF_C1_0(
        // Inputs
        .CLK                ( RX_CLK_G_net_0 ),
        .EXT_RST_N          ( TRAINING_RESETN ),
        .BANK_x_VDDI_STATUS ( VCC_net ),
        .BANK_y_VDDI_STATUS ( VCC_net ),
        .PLL_LOCK           ( VCC_net ),
        .SS_BUSY            ( GND_net ),
        .INIT_DONE          ( INIT_DONE ),
        .FF_US_RESTORE      ( GND_net ),
        .FPGA_POR_N         ( FPGA_POR_N ),
        // Outputs
        .PLL_POWERDOWN_B    (  ),
        .FABRIC_RESET_N     ( CORERESET_PF_C1_0_FABRIC_RESET_N ) 
        );

//--------MIPI_TRAINING_LITE_C0
MIPI_TRAINING_LITE_C0 MIPI_TRAINING_LITE_C0_0(
        // Inputs
        .CLK_I                 ( RX_CLK_G_net_0 ),
        .RESET_N_I             ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .PLL_LOCK_I            ( VCC_net ),
        .TAP_DELAYS_I          ( TAP_DELAYS_I_const_net_0 ),
        .VALID_WIN_LEN_I       ( VALID_WIN_LEN_I_const_net_0 ),
        .IOD_EARLY_I           ( PF_IOD_0_EYE_MONITOR_EARLY0to0 ),
        .IOD_LATE_I            ( PF_IOD_0_EYE_MONITOR_LATE0to0 ),
        .IOD_OOR_I             ( PF_IOD_0_DELAY_LINE_OUT_OF_RANGE0to0 ),
        .LP_DATA_N_I           ( L0_LP_DATA_N_net_0 ),
        .FALSE_FLG_THRESHOLD_I ( FALSE_FLG_THRESHOLD_I_const_net_0 ),
        // Outputs
        .BIT_ALGN_CLR_FLAGS_O  ( MIPI_TRAINING_LITE_C0_0_BIT_ALGN_CLR_FLAGS_O ),
        .BIT_ALGN_MOVE_O       ( MIPI_TRAINING_LITE_C0_0_BIT_ALGN_MOVE_O ),
        .BIT_ALGN_DIRECTION_O  ( MIPI_TRAINING_LITE_C0_0_BIT_ALGN_DIRECTION_O ),
        .BIT_ALGN_LOAD_O       ( MIPI_TRAINING_LITE_C0_0_BIT_ALGN_LOAD_O ),
        .TRAINING_ACTIVE_O     (  ),
        .TRAINING_DONE_O       ( MIPI_TRAINING_LITE_C0_0_TRAINING_DONE_O ),
        .TAP_CNT_FINAL_O       (  ) 
        );

//--------MIPI_TRAINING_LITE_C1
MIPI_TRAINING_LITE_C1 MIPI_TRAINING_LITE_C1_0(
        // Inputs
        .CLK_I                 ( RX_CLK_G_net_0 ),
        .RESET_N_I             ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .PLL_LOCK_I            ( VCC_net ),
        .TAP_DELAYS_I          ( TAP_DELAYS_I_const_net_1 ),
        .VALID_WIN_LEN_I       ( VALID_WIN_LEN_I_const_net_1 ),
        .IOD_EARLY_I           ( PF_IOD_0_EYE_MONITOR_EARLY1to1 ),
        .IOD_LATE_I            ( PF_IOD_0_EYE_MONITOR_LATE1to1 ),
        .IOD_OOR_I             ( PF_IOD_0_DELAY_LINE_OUT_OF_RANGE1to1 ),
        .LP_DATA_N_I           ( L1_LP_DATA_N_net_0 ),
        .FALSE_FLG_THRESHOLD_I ( FALSE_FLG_THRESHOLD_I_const_net_1 ),
        // Outputs
        .BIT_ALGN_CLR_FLAGS_O  ( MIPI_TRAINING_LITE_C1_0_BIT_ALGN_CLR_FLAGS_O ),
        .BIT_ALGN_MOVE_O       ( MIPI_TRAINING_LITE_C1_0_BIT_ALGN_MOVE_O ),
        .BIT_ALGN_DIRECTION_O  ( MIPI_TRAINING_LITE_C1_0_BIT_ALGN_DIRECTION_O ),
        .BIT_ALGN_LOAD_O       ( MIPI_TRAINING_LITE_C1_0_BIT_ALGN_LOAD_O ),
        .TRAINING_ACTIVE_O     (  ),
        .TRAINING_DONE_O       ( MIPI_TRAINING_LITE_C1_0_TRAINING_DONE_O ),
        .TAP_CNT_FINAL_O       (  ) 
        );

//--------MIPI_TRAINING_LITE_C2
MIPI_TRAINING_LITE_C2 MIPI_TRAINING_LITE_C2_0(
        // Inputs
        .CLK_I                 ( RX_CLK_G_net_0 ),
        .RESET_N_I             ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .PLL_LOCK_I            ( VCC_net ),
        .TAP_DELAYS_I          ( TAP_DELAYS_I_const_net_2 ),
        .VALID_WIN_LEN_I       ( VALID_WIN_LEN_I_const_net_2 ),
        .IOD_EARLY_I           ( PF_IOD_0_EYE_MONITOR_EARLY2to2 ),
        .IOD_LATE_I            ( PF_IOD_0_EYE_MONITOR_LATE2to2 ),
        .IOD_OOR_I             ( PF_IOD_0_DELAY_LINE_OUT_OF_RANGE2to2 ),
        .LP_DATA_N_I           ( L2_LP_DATA_N_net_0 ),
        .FALSE_FLG_THRESHOLD_I ( FALSE_FLG_THRESHOLD_I_const_net_2 ),
        // Outputs
        .BIT_ALGN_CLR_FLAGS_O  ( MIPI_TRAINING_LITE_C2_0_BIT_ALGN_CLR_FLAGS_O ),
        .BIT_ALGN_MOVE_O       ( MIPI_TRAINING_LITE_C2_0_BIT_ALGN_MOVE_O ),
        .BIT_ALGN_DIRECTION_O  ( MIPI_TRAINING_LITE_C2_0_BIT_ALGN_DIRECTION_O ),
        .BIT_ALGN_LOAD_O       ( MIPI_TRAINING_LITE_C2_0_BIT_ALGN_LOAD_O ),
        .TRAINING_ACTIVE_O     (  ),
        .TRAINING_DONE_O       ( MIPI_TRAINING_LITE_C2_0_TRAINING_DONE_O ),
        .TAP_CNT_FINAL_O       (  ) 
        );

//--------MIPI_TRAINING_LITE_C3
MIPI_TRAINING_LITE_C3 MIPI_TRAINING_LITE_C3_0(
        // Inputs
        .CLK_I                 ( RX_CLK_G_net_0 ),
        .RESET_N_I             ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .PLL_LOCK_I            ( VCC_net ),
        .TAP_DELAYS_I          ( TAP_DELAYS_I_const_net_3 ),
        .VALID_WIN_LEN_I       ( VALID_WIN_LEN_I_const_net_3 ),
        .IOD_EARLY_I           ( PF_IOD_0_EYE_MONITOR_EARLY3to3 ),
        .IOD_LATE_I            ( PF_IOD_0_EYE_MONITOR_LATE3to3 ),
        .IOD_OOR_I             ( PF_IOD_0_DELAY_LINE_OUT_OF_RANGE3to3 ),
        .LP_DATA_N_I           ( L3_LP_DATA_N_net_0 ),
        .FALSE_FLG_THRESHOLD_I ( FALSE_FLG_THRESHOLD_I_const_net_3 ),
        // Outputs
        .BIT_ALGN_CLR_FLAGS_O  ( MIPI_TRAINING_LITE_C3_0_BIT_ALGN_CLR_FLAGS_O ),
        .BIT_ALGN_MOVE_O       ( MIPI_TRAINING_LITE_C3_0_BIT_ALGN_MOVE_O ),
        .BIT_ALGN_DIRECTION_O  ( MIPI_TRAINING_LITE_C3_0_BIT_ALGN_DIRECTION_O ),
        .BIT_ALGN_LOAD_O       ( MIPI_TRAINING_LITE_C3_0_BIT_ALGN_LOAD_O ),
        .TRAINING_ACTIVE_O     (  ),
        .TRAINING_DONE_O       ( MIPI_TRAINING_LITE_C3_0_TRAINING_DONE_O ),
        .TAP_CNT_FINAL_O       (  ) 
        );

//--------PF_IOD_GENERIC_RX_C1
PF_IOD_GENERIC_RX_C1 PF_IOD_0(
        // Inputs
        .RX_CLK_P                ( RX_CLK_P ),
        .RX_CLK_N                ( RX_CLK_N ),
        .RXD                     ( RXD ),
        .RXD_N                   ( RXD_N ),
        .HS_SEL                  ( HS_SEL ),
        .EYE_MONITOR_CLEAR_FLAGS ( EYE_MONITOR_CLEAR_FLAGS_net_0 ),
        .DELAY_LINE_MOVE         ( DELAY_LINE_MOVE_net_0 ),
        .DELAY_LINE_DIRECTION    ( DELAY_LINE_DIRECTION_net_0 ),
        .DELAY_LINE_LOAD         ( DELAY_LINE_LOAD_net_0 ),
        .ARST_N                  ( ARST_N ),
        .HS_IO_CLK_PAUSE         ( HS_IO_CLK_PAUSE ),
        .EYE_MONITOR_WIDTH       ( EYE_MONITOR_WIDTH_const_net_0 ),
        // Outputs
        .L0_RXD_DATA             ( L0_RXD_DATA_net_0 ),
        .L1_RXD_DATA             ( L1_RXD_DATA_net_0 ),
        .L2_RXD_DATA             ( L2_RXD_DATA_net_0 ),
        .L3_RXD_DATA             ( L3_RXD_DATA_net_0 ),
        .L0_LP_DATA              ( L0_LP_DATA_net_0 ),
        .L0_LP_DATA_N            ( L0_LP_DATA_N_net_0 ),
        .L1_LP_DATA              ( L1_LP_DATA_net_0 ),
        .L1_LP_DATA_N            ( L1_LP_DATA_N_net_0 ),
        .L2_LP_DATA              ( L2_LP_DATA_net_0 ),
        .L2_LP_DATA_N            ( L2_LP_DATA_N_net_0 ),
        .L3_LP_DATA              ( L3_LP_DATA_net_0 ),
        .L3_LP_DATA_N            ( L3_LP_DATA_N_net_0 ),
        .RX_CLK_G                ( RX_CLK_G_net_0 ),
        .EYE_MONITOR_EARLY       ( EYE_MONITOR_EARLY_net_0 ),
        .EYE_MONITOR_LATE        ( EYE_MONITOR_LATE_net_0 ),
        .DELAY_LINE_OUT_OF_RANGE ( DELAY_LINE_OUT_OF_RANGE_net_0 ),
        .CLK_TRAIN_DONE          (  ),
        .CLK_TRAIN_ERROR         (  ) 
        );


endmodule
