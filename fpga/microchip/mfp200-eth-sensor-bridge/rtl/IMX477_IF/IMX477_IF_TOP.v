//////////////////////////////////////////////////////////////////////
// Created by SmartDesign Sat Jul 26 13:16:42 2025
// Version: 2025.1 2025.1.0.14
//////////////////////////////////////////////////////////////////////

`timescale 1ns / 100ps

// IMX477_IF_TOP
module IMX477_IF_TOP(
    // Inputs
    ARST_N,
    BIF_1_TREADY_I,
    CAM1_RX_CLK_N,
    CAM1_RX_CLK_P,
    FPGA_POR_N,
    INIT_DONE,
    RESET_N,
    RXD,
    RXD_N,
    TRNG_RST_N,
    sif_clk,
    // Outputs
    BIF_1_TDATA_O,
    BIF_1_TKEEP_O,
    BIF_1_TLAST_O,
    BIF_1_TSTRB_O,
    BIF_1_TUSER_O,
    BIF_1_TVALID_O,
    RX_CLK_G,
    c1_frame_valid_o
);

//--------------------------------------------------------------------
// Input
//--------------------------------------------------------------------
input         ARST_N;
input         BIF_1_TREADY_I;
input         CAM1_RX_CLK_N;
input         CAM1_RX_CLK_P;
input         FPGA_POR_N;
input         INIT_DONE;
input         RESET_N;
input  [3:0]  RXD;
input  [3:0]  RXD_N;
input         TRNG_RST_N;
input         sif_clk;
//--------------------------------------------------------------------
// Output
//--------------------------------------------------------------------
output [63:0] BIF_1_TDATA_O;
output [7:0]  BIF_1_TKEEP_O;
output        BIF_1_TLAST_O;
output [7:0]  BIF_1_TSTRB_O;
output [3:0]  BIF_1_TUSER_O;
output        BIF_1_TVALID_O;
output        RX_CLK_G;
output        c1_frame_valid_o;
//--------------------------------------------------------------------
// Nets
//--------------------------------------------------------------------
wire          AND2_0_Y;
wire          ARST_N;
wire   [63:0] BIF_1_TDATA;
wire   [7:0]  BIF_1_TKEEP;
wire          BIF_1_TLAST;
wire          BIF_1_TREADY_I;
wire   [7:0]  BIF_1_TSTRB;
wire   [3:0]  BIF_1_TUSER;
wire          BIF_1_TVALID;
wire          buffer_lp_0_DATA_A_O;
wire          buffer_lp_0_DATA_B_O;
wire          buffer_lp_0_DATA_C_O;
wire          buffer_lp_0_DATA_D_O;
wire          c1_frame_valid_o_net_0;
wire          CAM1_RX_CLK_N;
wire          CAM1_RX_CLK_P;
wire          CORERESET_PF_C1_0_FABRIC_RESET_N;
wire          FPGA_POR_N;
wire          INIT_DONE;
wire   [31:0] mipicsi2rxdecoderPF_0_DATA_O;
wire          mipicsi2rxdecoderPF_0_FRAME_END_O;
wire          mipicsi2rxdecoderPF_0_LINE_VALID_O;
wire          PF_CCC_C2_0_OUT0_FABCLK_0;
wire          PF_CCC_C2_0_PLL_LOCK_0;
wire          PF_IOD_GENERIC_RX_C0_0_L0_LP_DATA_N;
wire   [7:0]  PF_IOD_GENERIC_RX_C0_0_L0_RXD_DATA;
wire          PF_IOD_GENERIC_RX_C0_0_L1_LP_DATA_N;
wire   [7:0]  PF_IOD_GENERIC_RX_C0_0_L1_RXD_DATA;
wire          PF_IOD_GENERIC_RX_C0_0_L2_LP_DATA_N;
wire   [7:0]  PF_IOD_GENERIC_RX_C0_0_L2_RXD_DATA;
wire          PF_IOD_GENERIC_RX_C0_0_L3_LP_DATA_N;
wire   [7:0]  PF_IOD_GENERIC_RX_C0_0_L3_RXD_DATA;
wire          PF_IOD_GENERIC_RX_C0_0_training_done_o;
wire          RESET_N;
wire          RX_CLK_G_net_0;
wire   [3:0]  RXD;
wire   [3:0]  RXD_N;
wire          sif_clk;
wire          TRNG_RST_N;
wire          BIF_1_TLAST_net_0;
wire          BIF_1_TVALID_net_0;
wire          RX_CLK_G_net_1;
wire          c1_frame_valid_o_net_1;
wire   [63:0] BIF_1_TDATA_net_0;
wire   [7:0]  BIF_1_TKEEP_net_0;
wire   [7:0]  BIF_1_TSTRB_net_0;
wire   [3:0]  BIF_1_TUSER_net_0;
//--------------------------------------------------------------------
// TiedOff Nets
//--------------------------------------------------------------------
wire          GND_net;
wire          VCC_net;
wire   [31:0] AWADDR_I_const_net_0;
wire   [31:0] WDATA_I_const_net_0;
wire   [31:0] ARADDR_I_const_net_0;
//--------------------------------------------------------------------
// Constant assignments
//--------------------------------------------------------------------
assign GND_net              = 1'b0;
assign VCC_net              = 1'b1;
assign AWADDR_I_const_net_0 = 32'h00000000;
assign WDATA_I_const_net_0  = 32'h00000000;
assign ARADDR_I_const_net_0 = 32'h00000000;
//--------------------------------------------------------------------
// Top level output port assignments
//--------------------------------------------------------------------
assign BIF_1_TLAST_net_0      = BIF_1_TLAST;
assign BIF_1_TLAST_O          = BIF_1_TLAST_net_0;
assign BIF_1_TVALID_net_0     = BIF_1_TVALID;
assign BIF_1_TVALID_O         = BIF_1_TVALID_net_0;
assign RX_CLK_G_net_1         = RX_CLK_G_net_0;
assign RX_CLK_G               = RX_CLK_G_net_1;
assign c1_frame_valid_o_net_1 = c1_frame_valid_o_net_0;
assign c1_frame_valid_o       = c1_frame_valid_o_net_1;
assign BIF_1_TDATA_net_0      = BIF_1_TDATA;
assign BIF_1_TDATA_O[63:0]    = BIF_1_TDATA_net_0;
assign BIF_1_TKEEP_net_0      = BIF_1_TKEEP;
assign BIF_1_TKEEP_O[7:0]     = BIF_1_TKEEP_net_0;
assign BIF_1_TSTRB_net_0      = BIF_1_TSTRB;
assign BIF_1_TSTRB_O[7:0]     = BIF_1_TSTRB_net_0;
assign BIF_1_TUSER_net_0      = BIF_1_TUSER;
assign BIF_1_TUSER_O[3:0]     = BIF_1_TUSER_net_0;
//--------------------------------------------------------------------
// Component instances
//--------------------------------------------------------------------
//--------AND2
AND2 AND2_0(
        // Inputs
        .A ( TRNG_RST_N ),
        .B ( PF_CCC_C2_0_PLL_LOCK_0 ),
        // Outputs
        .Y ( AND2_0_Y ) 
        );

//--------buffer_lp
buffer_lp buffer_lp_0(
        // Inputs
        .SYS_CLK_I ( RX_CLK_G_net_0 ),
        .RESETN_I  ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .DATA_A_I  ( PF_IOD_GENERIC_RX_C0_0_L0_LP_DATA_N ),
        .DATA_B_I  ( PF_IOD_GENERIC_RX_C0_0_L1_LP_DATA_N ),
        .DATA_C_I  ( PF_IOD_GENERIC_RX_C0_0_L2_LP_DATA_N ),
        .DATA_D_I  ( PF_IOD_GENERIC_RX_C0_0_L3_LP_DATA_N ),
        .DATA_E_I  ( GND_net ),
        .DATA_F_I  ( GND_net ),
        .DATA_G_I  ( GND_net ),
        .DATA_H_I  ( GND_net ),
        // Outputs
        .DATA_A_O  ( buffer_lp_0_DATA_A_O ),
        .DATA_B_O  ( buffer_lp_0_DATA_B_O ),
        .DATA_C_O  ( buffer_lp_0_DATA_C_O ),
        .DATA_D_O  ( buffer_lp_0_DATA_D_O ),
        .DATA_E_O  (  ),
        .DATA_F_O  (  ),
        .DATA_G_O  (  ),
        .DATA_H_O  (  ) 
        );

//--------CORERESET_PF_C1
CORERESET_PF_C1 CORERESET_PF_C1_0(
        // Inputs
        .CLK                ( PF_CCC_C2_0_OUT0_FABCLK_0 ),
        .EXT_RST_N          ( PF_IOD_GENERIC_RX_C0_0_training_done_o ),
        .BANK_x_VDDI_STATUS ( VCC_net ),
        .BANK_y_VDDI_STATUS ( VCC_net ),
        .PLL_LOCK           ( PF_CCC_C2_0_PLL_LOCK_0 ),
        .SS_BUSY            ( GND_net ),
        .INIT_DONE          ( RESET_N ),
        .FF_US_RESTORE      ( GND_net ),
        .FPGA_POR_N         ( FPGA_POR_N ),
        // Outputs
        .PLL_POWERDOWN_B    (  ),
        .FABRIC_RESET_N     ( CORERESET_PF_C1_0_FABRIC_RESET_N ) 
        );

//--------HSB_AXIS
HSB_AXIS #( 
        .g_AXI_DWIDTH ( 64 ),
        .g_DWIDTH_IN  ( 32 ) )
HSB_AXIS_0(
        // Inputs
        .CLK_I        ( PF_CCC_C2_0_OUT0_FABCLK_0 ),
        .RESETN_I     ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .DATA_VALID_I ( mipicsi2rxdecoderPF_0_LINE_VALID_O ),
        .DATA_I       ( mipicsi2rxdecoderPF_0_DATA_O ),
        .EOF_I        ( mipicsi2rxdecoderPF_0_FRAME_END_O ),
        .ACLK_I       ( sif_clk ),
        .TREADY_I     ( BIF_1_TREADY_I ),
        // Outputs
        .TDATA_O      ( BIF_1_TDATA ),
        .TSTRB_O      ( BIF_1_TSTRB ),
        .TKEEP_O      ( BIF_1_TKEEP ),
        .TVALID_O     ( BIF_1_TVALID ),
        .TLAST_O      ( BIF_1_TLAST ),
        .TUSER_O      ( BIF_1_TUSER ) 
        );

//--------mipicsi2rxdecoderPF_C0
mipicsi2rxdecoderPF_C0 mipicsi2rxdecoderPF_0(
        // Inputs
        .CAM_CLOCK_I       ( RX_CLK_G_net_0 ),
        .PARALLEL_CLOCK_I  ( PF_CCC_C2_0_OUT0_FABCLK_0 ),
        .RESET_N_I         ( CORERESET_PF_C1_0_FABRIC_RESET_N ),
        .L0_HS_DATA_I      ( PF_IOD_GENERIC_RX_C0_0_L0_RXD_DATA ),
        .L1_HS_DATA_I      ( PF_IOD_GENERIC_RX_C0_0_L1_RXD_DATA ),
        .L2_HS_DATA_I      ( PF_IOD_GENERIC_RX_C0_0_L2_RXD_DATA ),
        .L3_HS_DATA_I      ( PF_IOD_GENERIC_RX_C0_0_L3_RXD_DATA ),
        .L0_LP_DATA_I      ( buffer_lp_0_DATA_A_O ),
        .L0_LP_DATA_N_I    ( buffer_lp_0_DATA_A_O ),
        .L1_LP_DATA_I      ( buffer_lp_0_DATA_B_O ),
        .L1_LP_DATA_N_I    ( buffer_lp_0_DATA_B_O ),
        .L2_LP_DATA_I      ( buffer_lp_0_DATA_C_O ),
        .L2_LP_DATA_N_I    ( buffer_lp_0_DATA_C_O ),
        .L3_LP_DATA_I      ( buffer_lp_0_DATA_D_O ),
        .L3_LP_DATA_N_I    ( buffer_lp_0_DATA_D_O ),
        .CAM_PLL_LOCK_I    ( PF_CCC_C2_0_PLL_LOCK_0 ),
        .TRAINING_DONE_I   ( PF_IOD_GENERIC_RX_C0_0_training_done_o ),
        .ACLK_I            ( GND_net ),
        .ARESETN_I         ( GND_net ),
        .AWVALID_I         ( GND_net ),
        .AWADDR_I          ( AWADDR_I_const_net_0 ),
        .WDATA_I           ( WDATA_I_const_net_0 ),
        .WVALID_I          ( GND_net ),
        .BREADY_I          ( GND_net ),
        .ARADDR_I          ( ARADDR_I_const_net_0 ),
        .ARVALID_I         ( GND_net ),
        .RREADY_I          ( GND_net ),
        // Outputs
        .FRAME_VALID_O     ( c1_frame_valid_o_net_0 ),
        .FRAME_START_O     (  ),
        .FRAME_END_O       ( mipicsi2rxdecoderPF_0_FRAME_END_O ),
        .LINE_VALID_O      ( mipicsi2rxdecoderPF_0_LINE_VALID_O ),
        .LINE_START_O      (  ),
        .LINE_END_O        (  ),
        .DATA_O            ( mipicsi2rxdecoderPF_0_DATA_O ),
        .VIRTUAL_CHANNEL_O (  ),
        .DATA_TYPE_O       (  ),
        .ECC_ERROR_O       (  ),
        .CRC_ERROR_O       (  ),
        .WORD_COUNT_O      (  ),
        .EBD_VALID_O       (  ),
        .MIPI_INTERRUPT_O  (  ),
        .AWREADY_O         (  ),
        .WREADY_O          (  ),
        .BRESP_O           (  ),
        .BVALID_O          (  ),
        .ARREADY_O         (  ),
        .RDATA_O           (  ),
        .RRESP_O           (  ),
        .RVALID_O          (  ) 
        );

//--------PF_CCC_C2
PF_CCC_C2 PF_CCC_C2_0(
        // Inputs
        .REF_CLK_0     ( RX_CLK_G_net_0 ),
        // Outputs
        .OUT0_FABCLK_0 ( PF_CCC_C2_0_OUT0_FABCLK_0 ),
        .PLL_LOCK_0    ( PF_CCC_C2_0_PLL_LOCK_0 ) 
        );

//--------CAM_IOD_TIP_TOP
CAM_IOD_TIP_TOP PF_IOD_GENERIC_RX_C0_0(
        // Inputs
        .ARST_N          ( ARST_N ),
        .FPGA_POR_N      ( FPGA_POR_N ),
        .HS_IO_CLK_PAUSE ( GND_net ),
        .HS_SEL          ( VCC_net ),
        .INIT_DONE       ( INIT_DONE ),
        .RX_CLK_N        ( CAM1_RX_CLK_N ),
        .RX_CLK_P        ( CAM1_RX_CLK_P ),
        .TRAINING_RESETN ( AND2_0_Y ),
        .RXD_N           ( RXD_N ),
        .RXD             ( RXD ),
        // Outputs
        .L0_LP_DATA_N    ( PF_IOD_GENERIC_RX_C0_0_L0_LP_DATA_N ),
        .L0_LP_DATA      (  ),
        .L1_LP_DATA_N    ( PF_IOD_GENERIC_RX_C0_0_L1_LP_DATA_N ),
        .L1_LP_DATA      (  ),
        .L2_LP_DATA_N    ( PF_IOD_GENERIC_RX_C0_0_L2_LP_DATA_N ),
        .L2_LP_DATA      (  ),
        .L3_LP_DATA_N    ( PF_IOD_GENERIC_RX_C0_0_L3_LP_DATA_N ),
        .L3_LP_DATA      (  ),
        .RX_CLK_G        ( RX_CLK_G_net_0 ),
        .training_done_o ( PF_IOD_GENERIC_RX_C0_0_training_done_o ),
        .L0_RXD_DATA     ( PF_IOD_GENERIC_RX_C0_0_L0_RXD_DATA ),
        .L1_RXD_DATA     ( PF_IOD_GENERIC_RX_C0_0_L1_RXD_DATA ),
        .L2_RXD_DATA     ( PF_IOD_GENERIC_RX_C0_0_L2_RXD_DATA ),
        .L3_RXD_DATA     ( PF_IOD_GENERIC_RX_C0_0_L3_RXD_DATA ) 
        );


endmodule
