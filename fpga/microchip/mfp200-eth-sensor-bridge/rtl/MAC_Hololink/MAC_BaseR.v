//////////////////////////////////////////////////////////////////////
// Created by SmartDesign Wed Aug 13 10:29:00 2025
// Version: 2025.1 2025.1.0.14
//////////////////////////////////////////////////////////////////////

`timescale 1ns / 100ps

// MAC_BaseR
module MAC_BaseR(
    // Inputs
    AXI4S_INITR_AXI4S_DT_INITR_TREADY,
    AXI4S_TRGT_AXI4S_DT_TARG_TDATA,
    AXI4S_TRGT_AXI4S_DT_TARG_TKEEP,
    AXI4S_TRGT_AXI4S_DT_TARG_TLAST,
    AXI4S_TRGT_AXI4S_DT_TARG_TUSER,
    AXI4S_TRGT_AXI4S_DT_TARG_TVALID,
    Device_Init_Done,
    I_SYS_CLK,
    I_SYS_RX_SRESETN,
    I_SYS_TX_SRESETN,
    LANE0_RXD_N,
    LANE0_RXD_P,
    PCLK,
    PRESETN,
    REF_CLK,
    // Outputs
    AXI4S_INITR_AXI4S_DT_INITR_TDATA,
    AXI4S_INITR_AXI4S_DT_INITR_TKEEP,
    AXI4S_INITR_AXI4S_DT_INITR_TLAST,
    AXI4S_INITR_AXI4S_DT_INITR_TUSER,
    AXI4S_INITR_AXI4S_DT_INITR_TVALID,
    AXI4S_TRGT_AXI4S_DT_TARG_TREADY,
    LANE0_RX_CLK_R,
    LANE0_RX_VAL,
    LANE0_TXD_N,
    LANE0_TXD_P,
    LANE0_TX_CLK_STABLE
);

//--------------------------------------------------------------------
// Input
//--------------------------------------------------------------------
input         AXI4S_INITR_AXI4S_DT_INITR_TREADY;
input  [63:0] AXI4S_TRGT_AXI4S_DT_TARG_TDATA;
input  [7:0]  AXI4S_TRGT_AXI4S_DT_TARG_TKEEP;
input         AXI4S_TRGT_AXI4S_DT_TARG_TLAST;
input  [2:0]  AXI4S_TRGT_AXI4S_DT_TARG_TUSER;
input         AXI4S_TRGT_AXI4S_DT_TARG_TVALID;
input         Device_Init_Done;
input         I_SYS_CLK;
input         I_SYS_RX_SRESETN;
input         I_SYS_TX_SRESETN;
input         LANE0_RXD_N;
input         LANE0_RXD_P;
input         PCLK;
input         PRESETN;
input         REF_CLK;
//--------------------------------------------------------------------
// Output
//--------------------------------------------------------------------
output [63:0] AXI4S_INITR_AXI4S_DT_INITR_TDATA;
output [7:0]  AXI4S_INITR_AXI4S_DT_INITR_TKEEP;
output        AXI4S_INITR_AXI4S_DT_INITR_TLAST;
output [7:0]  AXI4S_INITR_AXI4S_DT_INITR_TUSER;
output        AXI4S_INITR_AXI4S_DT_INITR_TVALID;
output        AXI4S_TRGT_AXI4S_DT_TARG_TREADY;
output        LANE0_RX_CLK_R;
output        LANE0_RX_VAL;
output        LANE0_TXD_N;
output        LANE0_TXD_P;
output        LANE0_TX_CLK_STABLE;
//--------------------------------------------------------------------
// Nets
//--------------------------------------------------------------------
wire          AND2_0_Y;
wire   [63:0] AXI4S_INITR_TDATA;
wire   [7:0]  AXI4S_INITR_TKEEP;
wire          AXI4S_INITR_TLAST;
wire          AXI4S_INITR_AXI4S_DT_INITR_TREADY;
wire   [7:0]  AXI4S_INITR_TUSER;
wire          AXI4S_INITR_TVALID;
wire   [63:0] AXI4S_TRGT_AXI4S_DT_TARG_TDATA;
wire   [7:0]  AXI4S_TRGT_AXI4S_DT_TARG_TKEEP;
wire          AXI4S_TRGT_AXI4S_DT_TARG_TLAST;
wire          AXI4S_TRGT_TREADY;
wire   [2:0]  AXI4S_TRGT_AXI4S_DT_TARG_TUSER;
wire          AXI4S_TRGT_AXI4S_DT_TARG_TVALID;
wire   [63:0] CORE10GMAC_C0_0_O_PMA49_TX_GRBX_DATA;
wire   [3:0]  CORE10GMAC_C0_0_O_PMA49_TX_GRBX_HDR;
wire          CORE10GMAC_C0_0_O_PMA49_TX_GRBX_SOS;
wire   [2:0]  CORE10GMAC_C0_0_O_SYS_MAC_RX_BC2to0;
wire   [63:0] CORE10GMAC_C0_0_O_SYS_MAC_RX_DATA;
wire          CORE10GMAC_C0_0_O_SYS_MAC_RX_EN;
wire          CORE10GMAC_C0_0_O_SYS_MAC_RX_EOP;
wire          CORE10GMAC_C0_0_O_SYS_MAC_RX_ERR;
wire          CORE10GMAC_C0_0_O_SYS_MAC_RX_RDY;
wire          CORE10GMAC_C0_0_O_SYS_MAC_RX_SOP;
wire          CORE10GMAC_C0_0_O_SYS_MAC_TX_FIFO_AF;
wire          CORE10GMAC_C0_0_O_SYS_MAC_TX_RDY;
wire          COREABC_C0_0_APB3Initiator_PENABLE;
wire   [31:0] COREABC_C0_0_APB3Initiator_PRDATA;
wire          COREABC_C0_0_APB3Initiator_PREADY;
wire          COREABC_C0_0_APB3Initiator_PSELx;
wire          COREABC_C0_0_APB3Initiator_PSLVERR;
wire   [31:0] COREABC_C0_0_APB3Initiator_PWDATA;
wire          COREABC_C0_0_APB3Initiator_PWRITE;
wire          CoreAPB3_C0_0_APBmslave0_PENABLE;
wire   [31:0] CoreAPB3_C0_0_APBmslave0_PRDATA;
wire          CoreAPB3_C0_0_APBmslave0_PREADY;
wire          CoreAPB3_C0_0_APBmslave0_PSELx;
wire          CoreAPB3_C0_0_APBmslave0_PSLVERR;
wire   [31:0] CoreAPB3_C0_0_APBmslave0_PWDATA;
wire          CoreAPB3_C0_0_APBmslave0_PWRITE;
wire          Device_Init_Done;
wire          I_SYS_CLK;
wire          LANE0_RX_CLK_R_net_0;
wire          LANE0_RX_VAL_net_0;
wire          LANE0_RXD_N;
wire          LANE0_RXD_P;
wire          LANE0_TX_CLK_STABLE_net_0;
wire          LANE0_TXD_N_net_0;
wire          LANE0_TXD_P_net_0;
wire   [2:0]  MAC_to_AXIS_0_MAC_TX_BC;
wire   [63:0] MAC_to_AXIS_0_MAC_TX_DATA;
wire          MAC_to_AXIS_0_MAC_TX_EN;
wire          MAC_to_AXIS_0_MAC_TX_EOP;
wire          MAC_to_AXIS_0_MAC_TX_SOP;
wire          PCLK;
wire          PF_TX_PLL_C0_0_CLKS_TO_XCVR_BIT_CLK;
wire          PF_TX_PLL_C0_0_CLKS_TO_XCVR_LOCK;
wire          PF_TX_PLL_C0_0_CLKS_TO_XCVR_REF_CLK_TO_LANE;
wire   [63:0] PF_XCVR_ERM_C0_0_LANE0_RX_DATA;
wire          PF_XCVR_ERM_C0_0_LANE0_RX_DATA_VAL;
wire   [3:0]  PF_XCVR_ERM_C0_0_LANE0_RX_HDR;
wire          PF_XCVR_ERM_C0_0_LANE0_RX_HDR_VAL;
wire          PF_XCVR_ERM_C0_0_LANE0_RX_SOS;
wire          PF_XCVR_ERM_C0_0_LANE0_STATUS_LOCK;
wire          PF_XCVR_ERM_C0_0_LANE0_TX_CLK_R;
wire          PRESETN;
wire          REF_CLK;
wire          AXI4S_INITR_TLAST_net_0;
wire          AXI4S_INITR_TVALID_net_0;
wire          AXI4S_TRGT_TREADY_net_0;
wire          LANE0_RX_VAL_net_1;
wire          LANE0_TXD_N_net_1;
wire          LANE0_TXD_P_net_1;
wire          LANE0_TX_CLK_STABLE_net_1;
wire   [63:0] AXI4S_INITR_TDATA_net_0;
wire   [7:0]  AXI4S_INITR_TKEEP_net_0;
wire   [7:0]  AXI4S_INITR_TUSER_net_0;
wire          LANE0_RX_CLK_R_net_1;
wire   [7:3]  O_SYS_MAC_RX_BC_slice_0;
wire   [7:0]  I_SYS_MAC_TX_BC_net_0;
wire   [7:0]  O_SYS_MAC_RX_BC_net_0;
//--------------------------------------------------------------------
// TiedOff Nets
//--------------------------------------------------------------------
wire          GND_net;
wire          VCC_net;
wire   [7:3]  I_SYS_MAC_TX_BC_const_net_0;
wire   [57:0] I_CFG_PCS49_TX_TEST_PATTERN_SEED_A_const_net_0;
wire   [57:0] I_CFG_PCS49_TX_TEST_PATTERN_SEED_B_const_net_0;
//--------------------------------------------------------------------
// Inverted Nets
//--------------------------------------------------------------------
wire          I_SYS_RX_SRESETN;
wire          I_SYS_RX_SRESETN_IN_POST_INV0_0;
wire          I_SYS_TX_SRESETN;
wire          I_SYS_TX_SRESETN_IN_POST_INV1_0;
wire          B_IN_POST_INV2_0;
wire          RESETN_IN_POST_INV3_0;
//--------------------------------------------------------------------
// Bus Interface Nets Declarations - Unequal Pin Widths
//--------------------------------------------------------------------
wire   [19:0] COREABC_C0_0_APB3Initiator_PADDR;
wire   [31:0] COREABC_C0_0_APB3Initiator_PADDR_0;
wire   [19:0] COREABC_C0_0_APB3Initiator_PADDR_0_19to0;
wire   [31:20]COREABC_C0_0_APB3Initiator_PADDR_0_31to20;
wire   [31:0] CoreAPB3_C0_0_APBmslave0_PADDR;
wire   [15:0] CoreAPB3_C0_0_APBmslave0_PADDR_0;
wire   [15:0] CoreAPB3_C0_0_APBmslave0_PADDR_0_15to0;
//--------------------------------------------------------------------
// Constant assignments
//--------------------------------------------------------------------
assign GND_net                                        = 1'b0;
assign VCC_net                                        = 1'b1;
assign I_SYS_MAC_TX_BC_const_net_0                    = 5'h1F;
assign I_CFG_PCS49_TX_TEST_PATTERN_SEED_A_const_net_0 = 58'h000000000000000;
assign I_CFG_PCS49_TX_TEST_PATTERN_SEED_B_const_net_0 = 58'h000000000000000;
//--------------------------------------------------------------------
// Inversions
//--------------------------------------------------------------------
assign I_SYS_RX_SRESETN_IN_POST_INV0_0 = ~ I_SYS_RX_SRESETN;
assign I_SYS_TX_SRESETN_IN_POST_INV1_0 = ~ I_SYS_TX_SRESETN;
assign B_IN_POST_INV2_0                = ~ CORE10GMAC_C0_0_O_SYS_MAC_TX_FIFO_AF;
assign RESETN_IN_POST_INV3_0           = ~ I_SYS_TX_SRESETN_IN_POST_INV1_0;
//--------------------------------------------------------------------
// Top level output port assignments
//--------------------------------------------------------------------
assign AXI4S_INITR_TLAST_net_0                = AXI4S_INITR_TLAST;
assign AXI4S_INITR_AXI4S_DT_INITR_TLAST       = AXI4S_INITR_TLAST_net_0;
assign AXI4S_INITR_TVALID_net_0               = AXI4S_INITR_TVALID;
assign AXI4S_INITR_AXI4S_DT_INITR_TVALID      = AXI4S_INITR_TVALID_net_0;
assign AXI4S_TRGT_TREADY_net_0                = AXI4S_TRGT_TREADY;
assign AXI4S_TRGT_AXI4S_DT_TARG_TREADY        = AXI4S_TRGT_TREADY_net_0;
assign LANE0_RX_VAL_net_1                     = LANE0_RX_VAL_net_0;
assign LANE0_RX_VAL                           = LANE0_RX_VAL_net_1;
assign LANE0_TXD_N_net_1                      = LANE0_TXD_N_net_0;
assign LANE0_TXD_N                            = LANE0_TXD_N_net_1;
assign LANE0_TXD_P_net_1                      = LANE0_TXD_P_net_0;
assign LANE0_TXD_P                            = LANE0_TXD_P_net_1;
assign LANE0_TX_CLK_STABLE_net_1              = LANE0_TX_CLK_STABLE_net_0;
assign LANE0_TX_CLK_STABLE                    = LANE0_TX_CLK_STABLE_net_1;
assign AXI4S_INITR_TDATA_net_0                = AXI4S_INITR_TDATA;
assign AXI4S_INITR_AXI4S_DT_INITR_TDATA[63:0] = AXI4S_INITR_TDATA_net_0;
assign AXI4S_INITR_TKEEP_net_0                = AXI4S_INITR_TKEEP;
assign AXI4S_INITR_AXI4S_DT_INITR_TKEEP[7:0]  = AXI4S_INITR_TKEEP_net_0;
assign AXI4S_INITR_TUSER_net_0                = AXI4S_INITR_TUSER;
assign AXI4S_INITR_AXI4S_DT_INITR_TUSER[7:0]  = AXI4S_INITR_TUSER_net_0;
assign LANE0_RX_CLK_R_net_1                   = LANE0_RX_CLK_R_net_0;
assign LANE0_RX_CLK_R                         = LANE0_RX_CLK_R_net_1;
//--------------------------------------------------------------------
// Slices assignments
//--------------------------------------------------------------------
assign CORE10GMAC_C0_0_O_SYS_MAC_RX_BC2to0 = O_SYS_MAC_RX_BC_net_0[2:0];
assign O_SYS_MAC_RX_BC_slice_0             = O_SYS_MAC_RX_BC_net_0[7:3];
//--------------------------------------------------------------------
// Concatenation assignments
//--------------------------------------------------------------------
assign I_SYS_MAC_TX_BC_net_0 = { 5'h1F , MAC_to_AXIS_0_MAC_TX_BC };
//--------------------------------------------------------------------
// Bus Interface Nets Assignments - Unequal Pin Widths
//--------------------------------------------------------------------
assign COREABC_C0_0_APB3Initiator_PADDR_0 = { COREABC_C0_0_APB3Initiator_PADDR_0_31to20, COREABC_C0_0_APB3Initiator_PADDR_0_19to0 };
assign COREABC_C0_0_APB3Initiator_PADDR_0_19to0 = COREABC_C0_0_APB3Initiator_PADDR[19:0];
assign COREABC_C0_0_APB3Initiator_PADDR_0_31to20 = 12'h0;

assign CoreAPB3_C0_0_APBmslave0_PADDR_0 = { CoreAPB3_C0_0_APBmslave0_PADDR_0_15to0 };
assign CoreAPB3_C0_0_APBmslave0_PADDR_0_15to0 = CoreAPB3_C0_0_APBmslave0_PADDR[15:0];

//--------------------------------------------------------------------
// Component instances
//--------------------------------------------------------------------
//--------AND2
AND2 AND2_0(
        // Inputs
        .A ( CORE10GMAC_C0_0_O_SYS_MAC_TX_RDY ),
        .B ( B_IN_POST_INV2_0 ),
        // Outputs
        .Y ( AND2_0_Y ) 
        );

//--------CORE10GMAC_C0
CORE10GMAC_C0 CORE10GMAC_C0_0(
        // Inputs
        .I_SYS_CLK                            ( I_SYS_CLK ),
        .I_SYS_TX_SRESET                      ( I_SYS_TX_SRESETN_IN_POST_INV1_0 ),
        .I_SYS_RX_SRESET                      ( I_SYS_RX_SRESETN_IN_POST_INV0_0 ),
        .I_CORE_TX_CLK                        ( PF_XCVR_ERM_C0_0_LANE0_TX_CLK_R ),
        .I_CORE_RX_CLK                        ( LANE0_RX_CLK_R_net_0 ),
        .I_SYS_MAC_TX_EN                      ( MAC_to_AXIS_0_MAC_TX_EN ),
        .I_SYS_MAC_TX_SOP                     ( MAC_to_AXIS_0_MAC_TX_SOP ),
        .I_SYS_MAC_TX_EOP                     ( MAC_to_AXIS_0_MAC_TX_EOP ),
        .I_CFG_RS_TX_FAULT_EN                 ( GND_net ),
        .I_CFG_RS_TX_FAULT_LOCAL              ( GND_net ),
        .I_CFG_RS_TX_FAULT_REMOTE             ( GND_net ),
        .I_CFG_RS_TX_IDLE                     ( GND_net ),
        .I_RS_RX_SRESET                       ( I_SYS_RX_SRESETN_IN_POST_INV0_0 ),
        .I_PCS49_TX_SRESET                    ( I_SYS_TX_SRESETN_IN_POST_INV1_0 ),
        .I_CFG_PCS49_TX_BYPASS_SCRAMBLER      ( VCC_net ),
        .I_CFG_PCS49_TX_TEST_PRBS31_EN        ( GND_net ),
        .I_CFG_PCS49_TX_TEST_PATTERN_EN       ( GND_net ),
        .I_CFG_PCS49_TX_TEST_PATTERN_TYPE_SEL ( GND_net ),
        .I_CFG_PCS49_TX_TEST_PATTERN_DATA_SEL ( GND_net ),
        .I_PCS49_RX_SRESET                    ( I_SYS_RX_SRESETN_IN_POST_INV0_0 ),
        .I_CFG_PCS49_RX_BYPASS_SCRAMBLER      ( VCC_net ),
        .I_CFG_PCS49_RX_TEST_PRBS31_EN        ( GND_net ),
        .I_CFG_PCS49_RX_TEST_PATTERN_EN       ( GND_net ),
        .I_CFG_PCS49_RX_TEST_PATTERN_TYPE_SEL ( GND_net ),
        .I_CFG_PCS49_RX_TEST_PATTERN_DATA_SEL ( GND_net ),
        .PCLK                                 ( PCLK ),
        .PRESETN                              ( PRESETN ),
        .I_PMA49_RX_GRBX_LOCK                 ( PF_XCVR_ERM_C0_0_LANE0_STATUS_LOCK ),
        .I_PMA49_RX_GRBX_SOS                  ( PF_XCVR_ERM_C0_0_LANE0_RX_SOS ),
        .I_PMA49_RX_GRBX_HDR_EN               ( PF_XCVR_ERM_C0_0_LANE0_RX_HDR_VAL ),
        .I_PMA49_RX_GRBX_DATA_EN              ( PF_XCVR_ERM_C0_0_LANE0_RX_DATA_VAL ),
        .PENABLE                              ( CoreAPB3_C0_0_APBmslave0_PENABLE ),
        .PWRITE                               ( CoreAPB3_C0_0_APBmslave0_PWRITE ),
        .PSEL                                 ( CoreAPB3_C0_0_APBmslave0_PSELx ),
        .I_SYS_MAC_TX_BC                      ( I_SYS_MAC_TX_BC_net_0 ),
        .I_SYS_MAC_TX_DATA                    ( MAC_to_AXIS_0_MAC_TX_DATA ),
        .I_CFG_PCS49_TX_TEST_PATTERN_SEED_A   ( I_CFG_PCS49_TX_TEST_PATTERN_SEED_A_const_net_0 ),
        .I_CFG_PCS49_TX_TEST_PATTERN_SEED_B   ( I_CFG_PCS49_TX_TEST_PATTERN_SEED_B_const_net_0 ),
        .I_PMA49_RX_GRBX_HDR                  ( PF_XCVR_ERM_C0_0_LANE0_RX_HDR ),
        .I_PMA49_RX_GRBX_DATA                 ( PF_XCVR_ERM_C0_0_LANE0_RX_DATA ),
        .PADDR                                ( CoreAPB3_C0_0_APBmslave0_PADDR_0 ),
        .PWDATA                               ( CoreAPB3_C0_0_APBmslave0_PWDATA ),
        // Outputs
        .O_CORE_TX_SRESET                     (  ),
        .O_CORE_RX_SRESET                     (  ),
        .O_SYS_MAC_TX_RDY                     ( CORE10GMAC_C0_0_O_SYS_MAC_TX_RDY ),
        .O_SYS_MAC_TX_FAULT                   (  ),
        .O_SYS_MAC_TX_FIFO_AF                 ( CORE10GMAC_C0_0_O_SYS_MAC_TX_FIFO_AF ),
        .O_SYS_MAC_TX_ERR_BUS_PROTOCOL        (  ),
        .O_SYS_MAC_TX_ERR_FIFO_OVERFLOW       (  ),
        .O_SYS_MAC_TX_ERR_FIFO_UNDERRUN       (  ),
        .O_SYS_MAC_RX_RDY                     ( CORE10GMAC_C0_0_O_SYS_MAC_RX_RDY ),
        .O_SYS_MAC_RX_EN                      ( CORE10GMAC_C0_0_O_SYS_MAC_RX_EN ),
        .O_SYS_MAC_RX_SOP                     ( CORE10GMAC_C0_0_O_SYS_MAC_RX_SOP ),
        .O_SYS_MAC_RX_EOP                     ( CORE10GMAC_C0_0_O_SYS_MAC_RX_EOP ),
        .O_SYS_MAC_RX_ERR                     ( CORE10GMAC_C0_0_O_SYS_MAC_RX_ERR ),
        .O_SYS_MAC_RX_ERR_FIFO_OVERFLOW       (  ),
        .O_RS_RX_FAULT_LOCAL                  (  ),
        .O_RS_RX_FAULT_REMOTE                 (  ),
        .O_PCS49_RX_BLOCK_LOCK                (  ),
        .O_PCS49_RX_HI_BER                    (  ),
        .O_PCS49_RX_STATUS                    (  ),
        .O_PCS49_RX_BER_STRB                  (  ),
        .O_PCS49_RX_TEST_MODE_ERR_STRB        (  ),
        .O_PCS49_RX_ERRORED_BLOCK_CNT_STRB    (  ),
        .O_PMA49_TX_GRBX_SOS                  ( CORE10GMAC_C0_0_O_PMA49_TX_GRBX_SOS ),
        .O_PMA49_TX_GRBX_HDR_EN               (  ),
        .O_PMA49_TX_GRBX_DATA_EN              (  ),
        .PREADY                               ( CoreAPB3_C0_0_APBmslave0_PREADY ),
        .PSLVERR                              ( CoreAPB3_C0_0_APBmslave0_PSLVERR ),
        .O_SYS_MAC_TX_STATS_VECTOR            (  ),
        .O_SYS_MAC_RX_ERR_W                   (  ),
        .O_SYS_MAC_RX_BC                      ( O_SYS_MAC_RX_BC_net_0 ),
        .O_SYS_MAC_RX_DATA                    ( CORE10GMAC_C0_0_O_SYS_MAC_RX_DATA ),
        .O_SYS_MAC_RX_STATS_VECTOR            (  ),
        .O_PCS49_RX_BER_CNT                   (  ),
        .O_PCS49_RX_TEST_MODE_ERR_CNT         (  ),
        .O_PMA49_TX_GRBX_HDR                  ( CORE10GMAC_C0_0_O_PMA49_TX_GRBX_HDR ),
        .O_PMA49_TX_GRBX_DATA                 ( CORE10GMAC_C0_0_O_PMA49_TX_GRBX_DATA ),
        .O_SYS_MAC_RX_PREAMBLE                (  ),
        .PRDATA                               ( CoreAPB3_C0_0_APBmslave0_PRDATA ) 
        );

//--------COREABC_C0
COREABC_C0 COREABC_C0_0(
        // Inputs
        .NSYSRESET ( CORE10GMAC_C0_0_O_SYS_MAC_RX_RDY ),
        .PCLK      ( PCLK ),
        .PREADY_M  ( COREABC_C0_0_APB3Initiator_PREADY ),
        .PSLVERR_M ( COREABC_C0_0_APB3Initiator_PSLVERR ),
        .IO_IN     ( GND_net ),
        .PRDATA_M  ( COREABC_C0_0_APB3Initiator_PRDATA ),
        // Outputs
        .PRESETN   (  ),
        .PSEL_M    ( COREABC_C0_0_APB3Initiator_PSELx ),
        .PENABLE_M ( COREABC_C0_0_APB3Initiator_PENABLE ),
        .PWRITE_M  ( COREABC_C0_0_APB3Initiator_PWRITE ),
        .IO_OUT    (  ),
        .PADDR_M   ( COREABC_C0_0_APB3Initiator_PADDR ),
        .PWDATA_M  ( COREABC_C0_0_APB3Initiator_PWDATA ) 
        );

//--------CoreAPB3_C0
CoreAPB3_C0 CoreAPB3_C0_0(
        // Inputs
        .PSEL      ( COREABC_C0_0_APB3Initiator_PSELx ),
        .PENABLE   ( COREABC_C0_0_APB3Initiator_PENABLE ),
        .PWRITE    ( COREABC_C0_0_APB3Initiator_PWRITE ),
        .PREADYS0  ( CoreAPB3_C0_0_APBmslave0_PREADY ),
        .PSLVERRS0 ( CoreAPB3_C0_0_APBmslave0_PSLVERR ),
        .PADDR     ( COREABC_C0_0_APB3Initiator_PADDR_0 ),
        .PWDATA    ( COREABC_C0_0_APB3Initiator_PWDATA ),
        .PRDATAS0  ( CoreAPB3_C0_0_APBmslave0_PRDATA ),
        // Outputs
        .PREADY    ( COREABC_C0_0_APB3Initiator_PREADY ),
        .PSLVERR   ( COREABC_C0_0_APB3Initiator_PSLVERR ),
        .PSELS0    ( CoreAPB3_C0_0_APBmslave0_PSELx ),
        .PENABLES  ( CoreAPB3_C0_0_APBmslave0_PENABLE ),
        .PWRITES   ( CoreAPB3_C0_0_APBmslave0_PWRITE ),
        .PRDATA    ( COREABC_C0_0_APB3Initiator_PRDATA ),
        .PADDRS    ( CoreAPB3_C0_0_APBmslave0_PADDR ),
        .PWDATAS   ( CoreAPB3_C0_0_APBmslave0_PWDATA ) 
        );

//--------MAC_to_AXIS
MAC_to_AXIS #( 
        .AXI4S_WIDTH      ( 64 ),
        .EMPCS_MAC_DWIDTH ( 64 ),
        .RX_USER_ENABLE   ( 0 ),
        .RXBUF_DEPTH      ( 64 ),
        .TX_USER_ENABLE   ( 0 ),
        .TXBUF_DEPTH      ( 64 ) )
MAC_to_AXIS_0(
        // Inputs
        .SYS_CLK               ( I_SYS_CLK ),
        .TX_CORE_CLK           ( I_SYS_CLK ),
        .RX_CORE_CLK           ( I_SYS_CLK ),
        .RESETN                ( RESETN_IN_POST_INV3_0 ),
        .AXI4S_DT_TARG_TVALID  ( AXI4S_TRGT_AXI4S_DT_TARG_TVALID ),
        .AXI4S_DT_TARG_TLAST   ( AXI4S_TRGT_AXI4S_DT_TARG_TLAST ),
        .AXI4S_DT_INITR_TREADY ( AXI4S_INITR_AXI4S_DT_INITR_TREADY ),
        .MAC_TX_RDY            ( AND2_0_Y ),
        .MAC_TX_FAULT          ( GND_net ),
        .MAC_TX_FIFO_FF        ( GND_net ),
        .MAC_TX_FIFO_PAF       ( GND_net ),
        .MAC_TX_FIFO_PAE       ( GND_net ),
        .MAC_RX_RDY            ( CORE10GMAC_C0_0_O_SYS_MAC_RX_RDY ),
        .MAC_RX_EN             ( CORE10GMAC_C0_0_O_SYS_MAC_RX_EN ),
        .MAC_RX_SOP            ( CORE10GMAC_C0_0_O_SYS_MAC_RX_SOP ),
        .MAC_RX_EOP            ( CORE10GMAC_C0_0_O_SYS_MAC_RX_EOP ),
        .MAC_RX_ERR            ( CORE10GMAC_C0_0_O_SYS_MAC_RX_ERR ),
        .MAC_RX_ERR_FRAME      ( GND_net ),
        .MAC_RX_ERR_CRC        ( GND_net ),
        .MAC_RX_ERR_STOMP      ( GND_net ),
        .MAC_RX_ERR_SHORT      ( GND_net ),
        .MAC_RX_ERR_LONG       ( GND_net ),
        .MAC_RX_ERR_TYPE       ( GND_net ),
        .MAC_RX_ERR_PAUSE      ( GND_net ),
        .AXI4S_DT_TARG_TDATA   ( AXI4S_TRGT_AXI4S_DT_TARG_TDATA ),
        .AXI4S_DT_TARG_TKEEP   ( AXI4S_TRGT_AXI4S_DT_TARG_TKEEP ),
        .AXI4S_DT_TARG_TUSER   ( AXI4S_TRGT_AXI4S_DT_TARG_TUSER ),
        .MAC_RX_DATA           ( CORE10GMAC_C0_0_O_SYS_MAC_RX_DATA ),
        .MAC_RX_BC             ( CORE10GMAC_C0_0_O_SYS_MAC_RX_BC2to0 ),
        // Outputs
        .AXI4S_DT_TARG_TREADY  ( AXI4S_TRGT_TREADY ),
        .AXI4S_DT_INITR_TVALID ( AXI4S_INITR_TVALID ),
        .AXI4S_DT_INITR_TLAST  ( AXI4S_INITR_TLAST ),
        .MAC_TX_EN             ( MAC_to_AXIS_0_MAC_TX_EN ),
        .MAC_TX_SOP            ( MAC_to_AXIS_0_MAC_TX_SOP ),
        .MAC_TX_EOP            ( MAC_to_AXIS_0_MAC_TX_EOP ),
        .MAC_TX_FCS_SWAP       (  ),
        .MAC_TX_FCS_INS        (  ),
        .MAC_TX_FCS_ERR        (  ),
        .MAC_TX_FCS_STOMP      (  ),
        .AXI4S_DT_INITR_TDATA  ( AXI4S_INITR_TDATA ),
        .AXI4S_DT_INITR_TKEEP  ( AXI4S_INITR_TKEEP ),
        .AXI4S_DT_INITR_TUSER  ( AXI4S_INITR_TUSER ),
        .MAC_TX_DATA           ( MAC_to_AXIS_0_MAC_TX_DATA ),
        .MAC_TX_BC             ( MAC_to_AXIS_0_MAC_TX_BC ) 
        );

//--------PF_TX_PLL_C0
PF_TX_PLL_C0 PF_TX_PLL_C0_0(
        // Inputs
        .REF_CLK         ( REF_CLK ),
        // Outputs
        .PLL_LOCK        (  ),
        .LOCK            ( PF_TX_PLL_C0_0_CLKS_TO_XCVR_LOCK ),
        .BIT_CLK         ( PF_TX_PLL_C0_0_CLKS_TO_XCVR_BIT_CLK ),
        .REF_CLK_TO_LANE ( PF_TX_PLL_C0_0_CLKS_TO_XCVR_REF_CLK_TO_LANE ) 
        );

//--------PF_XCVR_ERM_C0
PF_XCVR_ERM_C0 PF_XCVR_ERM_C0_0(
        // Inputs
        .LANE0_RXD_P          ( LANE0_RXD_P ),
        .LANE0_RXD_N          ( LANE0_RXD_N ),
        .LANE0_CDR_REF_CLK_0  ( REF_CLK ),
        .LANE0_PCS_ARST_N     ( Device_Init_Done ),
        .LANE0_PMA_ARST_N     ( Device_Init_Done ),
        .LANE0_TX_SOS         ( CORE10GMAC_C0_0_O_PMA49_TX_GRBX_SOS ),
        .TX_PLL_LOCK_0        ( PF_TX_PLL_C0_0_CLKS_TO_XCVR_LOCK ),
        .TX_BIT_CLK_0         ( PF_TX_PLL_C0_0_CLKS_TO_XCVR_BIT_CLK ),
        .TX_PLL_REF_CLK_0     ( PF_TX_PLL_C0_0_CLKS_TO_XCVR_REF_CLK_TO_LANE ),
        .LANE0_TX_DATA        ( CORE10GMAC_C0_0_O_PMA49_TX_GRBX_DATA ),
        .LANE0_TX_HDR         ( CORE10GMAC_C0_0_O_PMA49_TX_GRBX_HDR ),
        // Outputs
        .LANE0_TXD_P          ( LANE0_TXD_P_net_0 ),
        .LANE0_TXD_N          ( LANE0_TXD_N_net_0 ),
        .LANE0_TX_CLK_R       ( PF_XCVR_ERM_C0_0_LANE0_TX_CLK_R ),
        .LANE0_RX_CLK_R       ( LANE0_RX_CLK_R_net_0 ),
        .LANE0_JA_CLK         (  ),
        .LANE0_RX_IDLE        (  ),
        .LANE0_RX_READY       (  ),
        .LANE0_RX_VAL         ( LANE0_RX_VAL_net_0 ),
        .LANE0_TX_CLK_STABLE  ( LANE0_TX_CLK_STABLE_net_0 ),
        .LANE0_RX_HDR_VAL     ( PF_XCVR_ERM_C0_0_LANE0_RX_HDR_VAL ),
        .LANE0_RX_SOS         ( PF_XCVR_ERM_C0_0_LANE0_RX_SOS ),
        .LANE0_RX_DATA_VAL    ( PF_XCVR_ERM_C0_0_LANE0_RX_DATA_VAL ),
        .LANE0_RX_BYPASS_DATA (  ),
        .LANE0_STATUS_LOCK    ( PF_XCVR_ERM_C0_0_LANE0_STATUS_LOCK ),
        .LANE0_STATUS_HI_BER  (  ),
        .LANE0_RX_DATA        ( PF_XCVR_ERM_C0_0_LANE0_RX_DATA ),
        .LANE0_RX_HDR         ( PF_XCVR_ERM_C0_0_LANE0_RX_HDR ) 
        );


endmodule
