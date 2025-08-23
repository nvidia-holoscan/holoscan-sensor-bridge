///////////////////////////////////////////////////////////////////////////////////////////////////
// Company: <Name>
//
// File: HOLOLINK_top_wrapper.v
// File history:
//      <Revision number>: <Date>: <Comments>
//      <Revision number>: <Date>: <Comments>
//      <Revision number>: <Date>: <Comments>
//
// Description: 
//
// <Description here>
//
// Targeted device: <Family::PolarFire> <Die::MPF200T> <Package::FCG784>
// Author: <Name>
//
/////////////////////////////////////////////////////////////////////////////////////////////////// 

//`timescale <time_units> / <precision>
`include "HOLOLINK_def.svh"
module HOLOLINK_top_wrapper(
  input                        i_sys_rst,
//------------------------------------------------------------------------------
// User Reg Interface
//------------------------------------------------------------------------------
  // Control Plane
  input                        i_apb_clk,
  output                       o_apb_rst,
  // APB Register Interface
  output                       o_apb_psel,
  output                       o_apb_penable,
  output [31               :0] o_apb_paddr,
  output [31               :0] o_apb_pwdata,
  output                       o_apb_pwrite,
  input                        i_apb_pready,
  input  [31               :0] i_apb_prdata ,
  input                        i_apb_pserr,
//------------------------------------------------------------------------------
// User Auto Initialization Complete
//------------------------------------------------------------------------------
  output                       o_init_done,
//------------------------------------------------------------------------------
// Sensor IF
//------------------------------------------------------------------------------
  // Sensor Interface Clock and Reset
  input                        i_sif_clk,
  output                       o_sif_rst,
  // Sensor Rx Streaming Interface 0
  input                        i_sif_axis_tvalid_0,
  input                        i_sif_axis_tlast_0,
  input  [`DATAPATH_WIDTH-1:0] i_sif_axis_tdata_0,
  input  [`DATAKEEP_WIDTH-1:0] i_sif_axis_tkeep_0,
  input  [`DATAUSER_WIDTH-1:0] i_sif_axis_tuser_0,
  output                       o_sif_axis_tready_0,
  // Sensor Rx Streaming Interface 1
  input                        i_sif_axis_tvalid_1,
  input                        i_sif_axis_tlast_1,
  input  [`DATAPATH_WIDTH-1:0] i_sif_axis_tdata_1,
  input  [`DATAKEEP_WIDTH-1:0] i_sif_axis_tkeep_1,
  input  [`DATAUSER_WIDTH-1:0] i_sif_axis_tuser_1,
  output                       o_sif_axis_tready_1,
  // Sensor Tx Streaming Interface (Unimplemented)
  // output                       o_sif_axis_tvalid,
  // output                       o_sif_axis_tlast,
  // output [`DATAPATH_WIDTH-1:0] o_sif_axis_tdata,
  // output [`DATAKEEP_WIDTH-1:0] o_sif_axis_tkeep,
  // output [`DATAUSER_WIDTH-1:0] o_sif_axis_tuser,
  // input                        i_sif_axis_tready,
  // Sensor Event
  input  [15:0]    i_sif_event,
//------------------------------------------------------------------------------
// Host IF
//------------------------------------------------------------------------------
  // Host Interface Clock and Reset
  input                        i_hif_clk,
  output                       o_hif_rst,
  // Host Rx Interface 0
  input                        i_hif_axis_tvalid_0,
  input                        i_hif_axis_tlast_0,
  input  [`HOST_WIDTH-1    :0] i_hif_axis_tdata_0 ,
  input  [`HOSTKEEP_WIDTH-1:0] i_hif_axis_tkeep_0 ,
  input  [`HOSTUSER_WIDTH-1:0] i_hif_axis_tuser_0 ,
  output                       o_hif_axis_tready_0,
  // Host Tx Interface 0
  output                       o_hif_axis_tvalid_0,
  output                       o_hif_axis_tlast_0,
  output [`HOST_WIDTH-1    :0] o_hif_axis_tdata_0 ,
  output [`HOSTKEEP_WIDTH-1:0] o_hif_axis_tkeep_0 ,
  output [`HOSTUSER_WIDTH-1:0] o_hif_axis_tuser_0 ,
  input                        i_hif_axis_tready_0,
  // Host Rx Interface 1
  input                        i_hif_axis_tvalid_1,
  input                        i_hif_axis_tlast_1,
  input  [`HOST_WIDTH-1    :0] i_hif_axis_tdata_1 ,
  input  [`HOSTKEEP_WIDTH-1:0] i_hif_axis_tkeep_1 ,
  input  [`HOSTUSER_WIDTH-1:0] i_hif_axis_tuser_1 ,
  output                       o_hif_axis_tready_1,
  // Host Tx Interface 1
  output                       o_hif_axis_tvalid_1,
  output                       o_hif_axis_tlast_1,
  output [`HOST_WIDTH-1    :0] o_hif_axis_tdata_1 ,
  output [`HOSTKEEP_WIDTH-1:0] o_hif_axis_tkeep_1 ,
  output [`HOSTUSER_WIDTH-1:0] o_hif_axis_tuser_1 ,
  input                        i_hif_axis_tready_1,
//------------------------------------------------------------------------------
// Peripheral IF
//------------------------------------------------------------------------------
  // SPI Interface, QSPI compatible
  output                       o_spi_csn,
  output                       o_spi_sck,
  input  [3                :0] i_spi_sdio ,
  output [3                :0] o_spi_sdio ,
  output                       o_spi_oen,
  // I2C Interface
  input  [`I2C_INST-1      :0] i_i2c_scl,
  input  [`I2C_INST-1      :0] i_i2c_sda,
  output [`I2C_INST-1      :0] o_i2c_scl_en,
  output [`I2C_INST-1      :0] o_i2c_sda_en,
  // GPIO
  //output [`GPIO_INST-1      :0] o_gpio,
  //input  [`GPIO_INST-1       :0] i_gpio,
  inout  [`GPIO_INST-1     :0] GPIO, 
//------------------------------------------------------------------------------
// sensor reset
//------------------------------------------------------------------------------
  output                       o_sw_sys_rst,
  output [`SENSOR_IF_INST-1:0] o_sw_sen_rst,
//------------------------------------------------------------------------------
// PPS
//------------------------------------------------------------------------------
  input                        i_ptp_clk,
  output                       o_ptp_rst,
  output [47               :0] o_ptp_sec,
  output [31               :0] o_ptp_nanosec,
  output                       o_pps
);

wire [31:0]                apb_prdata     [0:0];
wire [`DATAPATH_WIDTH-1:0] sif_axis_tdata [0:`SENSOR_IF_INST-1];
wire [`DATAKEEP_WIDTH-1:0] sif_axis_tkeep [0:`SENSOR_IF_INST-1];
wire [`DATAUSER_WIDTH-1:0] sif_axis_tuser [0:`SENSOR_IF_INST-1];
wire [`HOST_WIDTH-1:0]     hif_axis_tdata [0:`HOST_IF_INST-1];
wire [`HOSTKEEP_WIDTH-1:0] hif_axis_tkeep [0:`HOST_IF_INST-1];
wire [`HOSTUSER_WIDTH-1:0] hif_axis_tuser [0:`HOST_IF_INST-1];
wire [`HOST_WIDTH-1:0]     hif_axis_tdata_out [0:`HOST_IF_INST-1];
wire [`HOSTKEEP_WIDTH-1:0] hif_axis_tkeep_out [0:`HOST_IF_INST-1];
wire [`HOSTUSER_WIDTH-1:0] hif_axis_tuser_out [0:`HOST_IF_INST-1];
wire [3:0]                 spi_sdin   [0:0];
wire [3:0]                 spi_sdout  [0:0];

assign apb_prdata[0]       = i_apb_prdata;
assign sif_axis_tdata[0]   = i_sif_axis_tdata_0;
assign sif_axis_tkeep[0]   = i_sif_axis_tkeep_0;
assign sif_axis_tuser[0]   = i_sif_axis_tuser_0;
assign hif_axis_tdata[0]   = i_hif_axis_tdata_0;
assign hif_axis_tkeep[0]   = i_hif_axis_tkeep_0;
assign hif_axis_tuser[0]   = i_hif_axis_tuser_0;
assign sif_axis_tdata[1]   = i_sif_axis_tdata_1;
assign sif_axis_tkeep[1]   = i_sif_axis_tkeep_1;
assign sif_axis_tuser[1]   = i_sif_axis_tuser_1;
assign hif_axis_tdata[1]   = i_hif_axis_tdata_1;
assign hif_axis_tkeep[1]   = i_hif_axis_tkeep_1;
assign hif_axis_tuser[1]   = i_hif_axis_tuser_1;
assign spi_sdin[0]         = i_spi_sdio;
assign o_hif_axis_tdata_0  = hif_axis_tdata_out[0];
assign o_hif_axis_tkeep_0  = hif_axis_tkeep_out[0];
assign o_hif_axis_tuser_0  = hif_axis_tuser_out[0];
assign o_hif_axis_tdata_1  = hif_axis_tdata_out[1];
assign o_hif_axis_tkeep_1  = hif_axis_tkeep_out[1];
assign o_hif_axis_tuser_1  = hif_axis_tuser_out[1];
assign o_spi_sdio          = spi_sdout[0];
//--------------------------------------------------------------------
// Component instances
//--------------------------------------------------------------------
//--------HOLOLINK_top
HOLOLINK_top HOLOLINK_top_0(
        // Inputs
        .i_sys_rst         ( i_sys_rst ),
        .i_apb_clk         ( i_apb_clk ),
        .i_apb_pready      ( i_apb_pready ),
        .i_apb_prdata      ( apb_prdata ),
        .i_apb_pserr       ( i_apb_pserr ),
        .i_sif_clk         ( i_sif_clk ),
        .i_sif_axis_tvalid ( {i_sif_axis_tvalid_1,i_sif_axis_tvalid_0} ),
        .i_sif_axis_tlast  ( {i_sif_axis_tlast_1,i_sif_axis_tlast_0} ),
        .i_sif_axis_tdata  ( sif_axis_tdata ),
        .i_sif_axis_tkeep  ( sif_axis_tkeep ),
        .i_sif_axis_tuser  ( sif_axis_tuser ),
        .i_sif_axis_tready ( 2'b11 ), 
        .i_sif_event       ( i_sif_event ),
        .i_hif_clk         ( i_hif_clk ),
        .i_hif_axis_tvalid ( {i_hif_axis_tvalid_1,i_hif_axis_tvalid_0} ),
        .i_hif_axis_tlast  ( {i_hif_axis_tlast_1,i_hif_axis_tlast_0} ),
        .i_hif_axis_tdata  ( hif_axis_tdata ),
        .i_hif_axis_tkeep  ( hif_axis_tkeep ),
        .i_hif_axis_tuser  ( hif_axis_tuser ),
        .i_hif_axis_tready ( {i_hif_axis_tready_1,i_hif_axis_tready_0} ),
        .i_spi_sdio        ( spi_sdin ),
        .i_i2c_scl         ( i_i2c_scl ),
        .i_i2c_sda         ( i_i2c_sda ),
        .i_gpio            ( GPIO ),
        // Outputs
        .o_apb_rst         ( o_apb_rst ),
        .o_apb_psel        ( o_apb_psel ),
        .o_apb_penable     ( o_apb_penable ),
        .o_apb_paddr       ( o_apb_paddr ),
        .o_apb_pwdata      ( o_apb_pwdata ),
        .o_apb_pwrite      ( o_apb_pwrite ),
        .o_init_done       ( o_init_done ),
        .o_sif_rst         ( o_sif_rst ),
        .o_sif_axis_tready ( {o_sif_axis_tready_1,o_sif_axis_tready_0} ),
        .o_sif_axis_tvalid (  ),
        .o_sif_axis_tlast  (  ),
        .o_sif_axis_tdata  (  ),
        .o_sif_axis_tkeep  (  ),
        .o_sif_axis_tuser  (  ),
        .o_hif_rst         ( o_hif_rst ),
        .o_hif_axis_tready ( {o_hif_axis_tready_1,o_hif_axis_tready_0} ),
        .o_hif_axis_tvalid ( {o_hif_axis_tvalid_1,o_hif_axis_tvalid_0} ),
        .o_hif_axis_tlast  ( {o_hif_axis_tlast_1,o_hif_axis_tlast_0} ),
        .o_hif_axis_tdata  ( hif_axis_tdata_out ),
        .o_hif_axis_tkeep  ( hif_axis_tkeep_out ),
        .o_hif_axis_tuser  ( hif_axis_tuser_out ),
        .o_spi_csn         ( o_spi_csn    ),
        .o_spi_sck         ( o_spi_sck    ),
        .o_spi_sdio        ( spi_sdout    ),
        .o_spi_oen         ( o_spi_oen    ),
        .o_i2c_scl_en      ( o_i2c_scl_en ),
        .o_i2c_sda_en      ( o_i2c_sda_en ),
        .o_gpio            ( GPIO       ),
        .o_sw_sys_rst      ( o_sw_sys_rst ),
        .o_sw_sen_rst      ( o_sw_sen_rst ),
        .i_ptp_clk         ( i_ptp_clk    ),
        .o_ptp_rst         ( o_ptp_rst    ),
        .o_ptp_sec         ( o_ptp_sec    ),
        .o_ptp_nanosec     ( o_ptp_nanosec ),
        .o_pps             ( o_pps )
        );
endmodule

