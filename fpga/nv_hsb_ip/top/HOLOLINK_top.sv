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

// Config parameters are defined in the svh file
`include "HOLOLINK_def.svh"

module HOLOLINK_top
  import HOLOLINK_pkg::*;
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter BUILD_REV = 48'h0
)(
  input                           i_sys_rst,
//------------------------------------------------------------------------------------------------//
// User Reg Interface
//------------------------------------------------------------------------------------------------//
  // Control Plane
  input                           i_apb_clk,
  output                          o_apb_rst,
  // APB Register Interface
  output [`REG_INST-1      :0]    o_apb_psel,
  output                          o_apb_penable,
  output [31               :0]    o_apb_paddr,
  output [31               :0]    o_apb_pwdata,
  output                          o_apb_pwrite,
  input  [`REG_INST-1      :0]    i_apb_pready,
  input  [31               :0]    i_apb_prdata      [`REG_INST-1:0],
  input  [`REG_INST-1      :0]    i_apb_pserr,
//------------------------------------------------------------------------------------------------//
// User Auto Initialization Complete
//------------------------------------------------------------------------------------------------//
`ifndef ENUM_EEPROM
  input  [47                :0]   i_mac_addr        [`HOST_IF_INST-1:0],
  input  [55                :0]   i_board_sn,
  input                           i_enum_vld,
`endif
  output                          o_init_done,
//------------------------------------------------------------------------------------------------//
// Sensor IF
//------------------------------------------------------------------------------------------------//
`ifdef SENSOR_RX_IF_INST
  // Sensor RX Interface Clock and Reset
  input  [`SENSOR_RX_IF_INST-1:0] i_sif_rx_clk,
  output [`SENSOR_RX_IF_INST-1:0] o_sif_rx_rst,
  // Sensor Rx Streaming Interface
  input  [`SENSOR_RX_IF_INST-1:0] i_sif_axis_tvalid,
  input  [`SENSOR_RX_IF_INST-1:0] i_sif_axis_tlast,
  input  [`DATAPATH_WIDTH-1   :0] i_sif_axis_tdata  [`SENSOR_RX_IF_INST-1:0],
  input  [`DATAKEEP_WIDTH-1   :0] i_sif_axis_tkeep  [`SENSOR_RX_IF_INST-1:0],
  input  [`DATAUSER_WIDTH-1   :0] i_sif_axis_tuser  [`SENSOR_RX_IF_INST-1:0],
  output [`SENSOR_RX_IF_INST-1:0] o_sif_axis_tready,
`endif

`ifdef SENSOR_TX_IF_INST
  // Sensor Tx Interface Clock and Reset
  input  [`SENSOR_TX_IF_INST-1:0] i_sif_tx_clk,
  output [`SENSOR_TX_IF_INST-1:0] o_sif_tx_rst,
  // Sensor Tx Streaming Interface
  output [`SENSOR_TX_IF_INST-1:0] o_sif_axis_tvalid,
  output [`SENSOR_TX_IF_INST-1:0] o_sif_axis_tlast,
  output [`DATAPATH_WIDTH-1   :0] o_sif_axis_tdata  [`SENSOR_TX_IF_INST-1:0],
  output [`DATAKEEP_WIDTH-1   :0] o_sif_axis_tkeep  [`SENSOR_TX_IF_INST-1:0],
  output [`DATAUSER_WIDTH-1   :0] o_sif_axis_tuser  [`SENSOR_TX_IF_INST-1:0],
  input  [`SENSOR_TX_IF_INST-1:0] i_sif_axis_tready,
`endif
  // Event
  input  [15:0]                   i_sif_event,
//------------------------------------------------------------------------------------------------//
// Host IF
//------------------------------------------------------------------------------------------------//
  // Host Interface Clock and Reset
  input                           i_hif_clk,
  output                          o_hif_rst,
  // Host Rx Interface
  input  [`HOST_IF_INST-1  :0]    i_hif_axis_tvalid,
  input  [`HOST_IF_INST-1  :0]    i_hif_axis_tlast,
  input  [`HOST_WIDTH-1    :0]    i_hif_axis_tdata  [`HOST_IF_INST-1:0],
  input  [`HOSTKEEP_WIDTH-1:0]    i_hif_axis_tkeep  [`HOST_IF_INST-1:0],
  input  [`HOSTUSER_WIDTH-1:0]    i_hif_axis_tuser  [`HOST_IF_INST-1:0],
  output [`HOST_IF_INST-1  :0]    o_hif_axis_tready,
  // Host Tx Interface
  output [`HOST_IF_INST-1  :0]    o_hif_axis_tvalid,
  output [`HOST_IF_INST-1  :0]    o_hif_axis_tlast,
  output [`HOST_WIDTH-1    :0]    o_hif_axis_tdata  [`HOST_IF_INST-1:0],
  output [`HOSTKEEP_WIDTH-1:0]    o_hif_axis_tkeep  [`HOST_IF_INST-1:0],
  output [`HOSTUSER_WIDTH-1:0]    o_hif_axis_tuser  [`HOST_IF_INST-1:0],
  input  [`HOST_IF_INST-1  :0]    i_hif_axis_tready,
//------------------------------------------------------------------------------------------------//
// Peripheral IF
//------------------------------------------------------------------------------------------------//
`ifdef SPI_INST
  // SPI Interface, QSPI compatible
  output [`SPI_INST-1      :0]    o_spi_csn,
  output [`SPI_INST-1      :0]    o_spi_sck,
  input  [3                :0]    i_spi_sdio        [`SPI_INST-1:0],
  output [3                :0]    o_spi_sdio        [`SPI_INST-1:0],
  output [`SPI_INST-1      :0]    o_spi_oen,
`endif
`ifdef I2C_INST
  // I2C Interface
  input  [`I2C_INST-1      :0]    i_i2c_scl,
  input  [`I2C_INST-1      :0]    i_i2c_sda,
  output [`I2C_INST-1      :0]    o_i2c_scl_en,
  output [`I2C_INST-1      :0]    o_i2c_sda_en,
`endif
`ifdef UART_INST
  // UART Interface
  output                       o_uart_tx,
  input                        i_uart_rx,
  output                       o_uart_busy,
  input                        i_uart_cts,
  output                       o_uart_rts,
`endif
  // GPIO
  input  [`GPIO_INST-1     :0]    i_gpio,
  output [`GPIO_INST-1     :0]    o_gpio,
  output [`GPIO_INST-1     :0]    o_gpio_dir,
//------------------------------------------------------------------------------------------------//
// Sensor Reset
//------------------------------------------------------------------------------------------------//
  output                          o_sw_sys_rst,
  output [31               :0]    o_sw_sen_rst,
//------------------------------------------------------------------------------------------------//
// PPS
//------------------------------------------------------------------------------------------------//
  input                           i_ptp_clk,
  output                          o_ptp_rst,
`ifndef EXT_PTP
  output [47               :0]    o_ptp_sec,
  output [31               :0]    o_ptp_nanosec,
  output                          o_pps
`else
  input  [47               :0]    i_ptp_sec,
  input  [31               :0]    i_ptp_nanosec
`endif
);

localparam HOLOLINK_REV = 16'h2603;

`ifdef SYNC_CLK_HIF_APB
  localparam SYNC_CLK_HIF_APB = 1;
`else
  localparam SYNC_CLK_HIF_APB = 0;
`endif

`ifdef SYNC_CLK_HIF_PTP
  localparam SYNC_CLK_HIF_PTP = 1;
`else
  localparam SYNC_CLK_HIF_PTP = 0;
`endif

//------------------------------------------------------------------------------------------------//
// Reset Generator
//------------------------------------------------------------------------------------------------//
`ifdef SENSOR_RX_IF_INST
  localparam NUM_SENSOR_RX = `SENSOR_RX_IF_INST;

  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_clk;
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_rst;

  apb_m2s s_apb_m2s_sen_rx [`SENSOR_RX_IF_INST * num_sen_rx_mod];
  apb_s2m s_apb_s2m_sen_rx [`SENSOR_RX_IF_INST * num_sen_rx_mod];

  assign sif_rx_clk   = i_sif_rx_clk;
  assign o_sif_rx_rst = sif_rx_rst;
`else
  localparam NUM_SENSOR_RX = 1;

  logic [NUM_SENSOR_RX-1:0] sif_rx_clk;
  logic [NUM_SENSOR_RX-1:0] sif_rx_rst;

  apb_m2s s_apb_m2s_sen_rx [num_sen_rx_mod];
  apb_s2m s_apb_s2m_sen_rx [num_sen_rx_mod];

  assign s_apb_s2m_sen_rx = '{default:0};
  assign sif_rx_clk       = 1'b0;
`endif

`ifdef SENSOR_TX_IF_INST
  localparam NUM_SENSOR_TX = `SENSOR_TX_IF_INST;

  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_clk;
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_rst;

  apb_m2s s_apb_m2s_sen_tx [`SENSOR_TX_IF_INST * num_sen_tx_mod];
  apb_s2m s_apb_s2m_sen_tx [`SENSOR_TX_IF_INST * num_sen_tx_mod];

  assign sif_tx_clk   = i_sif_tx_clk;
  assign o_sif_tx_rst = sif_tx_rst;
`else
  localparam NUM_SENSOR_TX = 1;
  logic [NUM_SENSOR_TX-1:0] sif_tx_clk;
  logic [NUM_SENSOR_TX-1:0] sif_tx_rst;

  apb_m2s s_apb_m2s_sen_tx [num_sen_tx_mod];
  apb_s2m s_apb_s2m_sen_tx [num_sen_tx_mod];

  assign s_apb_s2m_sen_tx = '{default:0};
  assign sif_tx_clk       = 1'b0;
`endif

reset_gen #(
  .NUM_SENSOR_RX ( NUM_SENSOR_RX      ),
  .NUM_SENSOR_TX ( NUM_SENSOR_TX      )
) u_rst_gen(
  .i_sys_rst     ( i_sys_rst          ),
  .i_cfg_rst     ( o_sw_sys_rst       ),
  .i_sif_rx_clk  ( sif_rx_clk         ),
  .i_sif_tx_clk  ( sif_tx_clk         ),
  .i_hif_clk     ( i_hif_clk          ),
  .i_apb_clk     ( i_apb_clk          ),
  .i_ptp_clk     ( i_ptp_clk          ),
  .o_sif_rx_rst  ( sif_rx_rst         ),
  .o_sif_tx_rst  ( sif_tx_rst         ),
  .o_hif_rst     ( o_hif_rst          ),
  .o_apb_rst     ( o_apb_rst          ),
  .o_ptp_rst     ( o_ptp_rst          )
);

apb_m2s s_apb_m2s_dev  [4];
apb_s2m s_apb_s2m_dev  [4];

apb_m2s s_apb_m2s_host [`HOST_IF_INST * num_host_mod];
apb_s2m s_apb_s2m_host [`HOST_IF_INST * num_host_mod];

apb_m2s s_apb_m2s_ram  [2];
apb_s2m s_apb_s2m_ram  [2];

apb_m2s s_apb_m2s_bridge;
apb_s2m s_apb_s2m_bridge;
//------------------------------------------------------------------------------------------------//
// RX Parser
//------------------------------------------------------------------------------------------------//

localparam PORT_NUM       = 5;
localparam NUM_LS_TX_INST = 4;
localparam NUM_LS_RX_INST = 6;
localparam AXI_LS_DWIDTH  = 8;
localparam AXI_LS_KWIDTH  = AXI_LS_DWIDTH/8;
localparam ENUM_DWIDTH    = 296;

logic                       init_done_hif_clk;
logic                       pps;
logic [47:0]                ptp_sec;
logic [ENUM_DWIDTH-1:0]     enum_data;

logic [47               :0] dev_mac_addr                  [`HOST_IF_INST];
logic [31               :0] dev_ip_addr                   [`HOST_IF_INST];
logic [31 :0]               eeprom_base_ip_addr;
logic                       eeprom_ip_addr_vld_hif_clk;

logic [`HOST_IF_INST-1  :0] stx_axis_tvalid;
logic [`HOST_WIDTH-1    :0] stx_axis_tdata               [`HOST_IF_INST];
logic [`HOST_IF_INST-1  :0] stx_axis_tlast;
logic [`HOST_IF_INST-1  :0] stx_axis_tuser;
logic [`HOSTKEEP_WIDTH-1:0] stx_axis_tkeep               [`HOST_IF_INST];

logic [`HOST_IF_INST-1  :0] brx_axis_tvalid;
logic [`HOST_WIDTH-1    :0] brx_axis_tdata               [`HOST_IF_INST-1:0];
logic [`HOST_IF_INST-1  :0] brx_axis_tlast;
logic [                1:0] brx_axis_tuser               [`HOST_IF_INST-1:0];
logic [`HOSTKEEP_WIDTH-1:0] brx_axis_tkeep               [`HOST_IF_INST-1:0];
logic [`HOST_IF_INST-1  :0] brx_axis_tready;

logic [`HOST_IF_INST-1  :0] btx_axis_tvalid;
logic [`HOST_WIDTH-1    :0] btx_axis_tdata               [`HOST_IF_INST-1:0];
logic [`HOST_IF_INST-1  :0] btx_axis_tlast;
logic [`HOST_IF_INST-1  :0] btx_axis_tuser;
logic [`HOSTKEEP_WIDTH-1:0] btx_axis_tkeep               [`HOST_IF_INST-1:0];
logic [`HOST_IF_INST-1:0]   btx_axis_tready;

logic [71               :0] dest_info                    [`HOST_IF_INST];

logic [`HOST_IF_INST-1  :0] ptp_rx_axis_tvalid;
logic [`HOST_WIDTH-1    :0] ptp_rx_axis_tdata            [`HOST_IF_INST];
logic [`HOST_IF_INST-1  :0] ptp_rx_axis_tlast;
logic [`HOST_IF_INST-1  :0] ptp_rx_axis_tuser;
logic [`HOSTKEEP_WIDTH-1:0] ptp_rx_axis_tkeep            [`HOST_IF_INST];
logic [`HOST_IF_INST-1  :0] ptp_rx_axis_tready;

logic [`HOST_IF_INST-1  :0] lso_axis_tvalid;
logic [`HOST_WIDTH-1    :0] lso_axis_tdata;
logic                       lso_axis_tlast;
logic                       lso_axis_tuser;
logic [`HOSTKEEP_WIDTH-1:0] lso_axis_tkeep;
logic [`HOST_IF_INST-1  :0] lso_axis_tready;

logic [`HOST_IF_INST-1 :0]  lsi_axis_tvalid;
logic [AXI_LS_DWIDTH-1 :0]  lsi_axis_tdata               [`HOST_IF_INST];
logic [`HOST_IF_INST-1:0]   lsi_axis_tlast;
logic [NUM_LS_RX_INST-1:0]  lsi_axis_tuser               [`HOST_IF_INST];
logic [AXI_LS_KWIDTH-1:0]   lsi_axis_tkeep               [`HOST_IF_INST];
logic [`HOST_IF_INST-1:0]   lsi_axis_tready;

logic [`HOST_IF_INST-1  :0] pause_axis_tvalid;
logic [`HOST_WIDTH-1    :0] pause_axis_tdata             [`HOST_IF_INST];
logic [`HOST_IF_INST-1  :0] pause_axis_tlast;
logic [`HOST_IF_INST-1  :0] pause_axis_tuser;
logic [`HOSTKEEP_WIDTH-1:0] pause_axis_tkeep             [`HOST_IF_INST];
logic [`HOST_IF_INST-1  :0] pause_axis_tready;

logic [47:0] ptp_sec_sync_hif;
logic [31:0] ptp_nano_sec_sync_hif;
logic        ptp_sync_hif_valid;
logic [47:0] ptp_sec_sync_apb;
logic [31:0] ptp_nano_sec_sync_apb;
logic        ptp_sync_apb_valid;

logic [`HOST_IF_INST-1  :0] pkt_inc;

logic [`HOST_IF_INST-1:0] ptp_val;
logic [31:0]              ptp_nano_sec;
logic [47:0]              ptp_frac_nano_sec;
logic                     ptp_en_sync_rx;
logic [7:0]               hsb_stat;

assign hsb_stat = {7'h0, ptp_en_sync_rx};

logic [31:0]  event_vec;
logic         spi_busy;
logic         i2c_busy;
logic         uart_irq;

logic [`HOST_IF_INST-1  :0] is_ptp_sync_msg;

apb_m2s m_apb_m2s [4];
apb_s2m m_apb_s2m [4];
genvar i, j;
generate
  for (i=0; i<`HOST_IF_INST; i++) begin: gen_rx_parser

    assign  o_hif_axis_tready             [i] = init_done_hif_clk;

    rx_parser #(
      .AXI_DWIDTH                        ( `HOST_WIDTH                                ),
      .AXI_LS_DWIDTH                     ( 8                                          ),
      .MTU                               ( `HOST_MTU                                  ),
      .NUM_LS_INST                       ( NUM_LS_RX_INST                             ),
      .SYNC_CLK                          ( SYNC_CLK_HIF_APB                           )
    ) rx_parser (
      .host_clk                          ( i_hif_clk                                  ),
      .host_rst                          ( o_hif_rst                                  ),
      .apb_clk                           ( i_apb_clk                                  ),
      .apb_rst                           ( o_apb_rst                                  ),
      //Configuration
      .i_dev_mac_addr                    ( dev_mac_addr             [i]               ),
      .i_dev_ip_addr                     ( dev_ip_addr              [i]               ),
      //Register APB Interfaces
      .i_apb_m2s                         ( s_apb_m2s_host [(i*num_host_mod)+inst_dec] ),
      .o_apb_s2m                         ( s_apb_s2m_host [(i*num_host_mod)+inst_dec] ),
      .o_dest_info                       ( dest_info                [i]               ),
      .o_ptp_sync_msg                    ( is_ptp_sync_msg          [i]               ),
      //AXI RX Interface inbound from packet buffer (via Ethernet MAC)
      .i_axis_rx_tvalid                  ( i_hif_axis_tvalid [i] && init_done_hif_clk ),
      .i_axis_rx_tdata                   ( i_hif_axis_tdata         [i]               ),
      .i_axis_rx_tlast                   ( i_hif_axis_tlast         [i]               ),
      .i_axis_rx_tuser                   ( i_hif_axis_tuser         [i]               ),
      .i_axis_rx_tkeep                   ( i_hif_axis_tkeep         [i]               ),
      // Sensor TX Interface
      .o_stx_axis_tvalid                 ( stx_axis_tvalid          [i]               ),
      .o_stx_axis_tdata                  ( stx_axis_tdata           [i]               ),
      .o_stx_axis_tlast                  ( stx_axis_tlast           [i]               ),
      .o_stx_axis_tuser                  ( stx_axis_tuser           [i]               ),
      .o_stx_axis_tkeep                  ( stx_axis_tkeep           [i]               ),
      // PTP Interface
      .o_ptp_axis_tvalid                 ( ptp_rx_axis_tvalid       [i]               ),
      .o_ptp_axis_tdata                  ( ptp_rx_axis_tdata        [i]               ),
      .o_ptp_axis_tlast                  ( ptp_rx_axis_tlast        [i]               ),
      .o_ptp_axis_tuser                  ( ptp_rx_axis_tuser        [i]               ),
      .o_ptp_axis_tkeep                  ( ptp_rx_axis_tkeep        [i]               ),
      //Bridge TX AXIS Interface
      .o_btx_axis_tvalid                 ( brx_axis_tvalid          [i]               ),
      .o_btx_axis_tdata                  ( brx_axis_tdata           [i]               ),
      .o_btx_axis_tlast                  ( brx_axis_tlast           [i]               ),
      .o_btx_axis_tuser                  ( brx_axis_tuser           [i]               ),
      .o_btx_axis_tkeep                  ( brx_axis_tkeep           [i]               ),
      .i_btx_axis_tready                 ( brx_axis_tready          [i]               ),
      //Low Speed AXIS Interface to datapath
      .o_ls_axis_tvalid                  ( lsi_axis_tvalid          [i]               ),
      .o_ls_axis_tdata                   ( lsi_axis_tdata           [i]               ),
      .o_ls_axis_tlast                   ( lsi_axis_tlast           [i]               ),
      .o_ls_axis_tuser                   ( lsi_axis_tuser           [i]               ),
      .o_ls_axis_tkeep                   ( lsi_axis_tkeep           [i]               ),
      .i_ls_axis_tready                  ( lsi_axis_tready          [i]               )
    );

    assign s_apb_s2m_host [(i*num_host_mod)+ctrl_evt].pready   = 0;
    assign s_apb_s2m_host [(i*num_host_mod)+ctrl_evt].prdata   = 0;
    assign s_apb_s2m_host [(i*num_host_mod)+ctrl_evt].pserr    = 0;

  end
endgenerate

rx_ls_parser #(
  .AXI_DWIDTH                         ( `HOST_WIDTH                               ),
  .AXI_LS_DWIDTH                      ( 8                                         ),
  .ENUM_DWIDTH                        ( ENUM_DWIDTH                               ),
  .MTU                                ( `HOST_MTU                                 ),
  .NUM_LS_RX_INST                     ( NUM_LS_RX_INST                            ),
  .NUM_LS_TX_INST                     ( NUM_LS_TX_INST                            ),
  .NUM_HOST                           ( `HOST_IF_INST                             ),
  .UUID                               ( `UUID                                     ),
  .SYNC_CLK                           ( SYNC_CLK_HIF_APB                          )
) u_rx_ls_parser (
  .i_pclk                             ( i_hif_clk                                 ),
  .i_prst                             ( o_hif_rst                                 ),
  .i_aclk                             ( i_apb_clk                                 ),
  .i_arst                             ( o_apb_rst                                 ),
  .i_apb_m2s_ecb                      ( s_apb_m2s_dev [3]                         ),
  .o_apb_s2m_ecb                      ( s_apb_s2m_dev [3]                         ),
  .i_apb_m2s_evt                      ( s_apb_m2s_dev [2]                         ),
  .o_apb_s2m_evt                      ( s_apb_s2m_dev [2]                         ),
  .o_apb_m2s_evt                      ( m_apb_m2s     [3]                         ),
  .i_apb_s2m_evt                      ( m_apb_s2m     [3]                         ),
  .i_apb_m2s_ram                      ( s_apb_m2s_ram [1]                         ),
  .o_apb_s2m_ram                      ( s_apb_s2m_ram [1]                         ),
  //Low Speed AXIS Interface to Packet proc
  .i_axis_tvalid                      ( lsi_axis_tvalid                           ),
  .i_axis_tdata                       ( lsi_axis_tdata                            ),
  .i_axis_tlast                       ( lsi_axis_tlast                            ),
  .i_axis_tuser                       ( lsi_axis_tuser                            ),
  .i_axis_tkeep                       ( lsi_axis_tkeep                            ),
  .o_axis_tready                      ( lsi_axis_tready                           ),
  // Input Control Data
  .i_init_done                        ( init_done_hif_clk                         ),
  .i_ptp_clk                          ( i_ptp_clk                                 ),
  .i_ptp_rst                          ( o_ptp_rst                                 ),
  .i_ptp                              ( {ptp_sec, ptp_nano_sec}                   ),
  .i_ptp_hif                          ( {ptp_sec_sync_hif, ptp_nano_sec_sync_hif} ),
  .i_dev_mac_addr                     ( dev_mac_addr                              ),
  .i_eeprom_ip_addr                   ( eeprom_base_ip_addr                       ),
  .i_eeprom_ip_addr_vld               ( eeprom_ip_addr_vld_hif_clk                ),
  .i_hsb_stat                         ( hsb_stat                                  ),
  .o_dev_ip_addr                      ( dev_ip_addr                               ),
  .i_enum_data                        ( enum_data                                 ),
  .i_evt_vec                          ( event_vec                                 ),
  .i_pkt_inc                          ( pkt_inc                                   ),
  .i_pps                              ( pps                                       ),
  // APB interface
  .o_apb_m2s_ecb                      ( m_apb_m2s [0]                             ),
  .i_apb_s2m_ecb                      ( m_apb_s2m [0]                             ),
  //Low Speed AXIS Interface to ethernet output at full speed
  .o_axis_tvalid                      ( lso_axis_tvalid                           ),
  .o_axis_tdata                       ( lso_axis_tdata                            ),
  .o_axis_tlast                       ( lso_axis_tlast                            ),
  .o_axis_tuser                       ( lso_axis_tuser                            ),
  .o_axis_tkeep                       ( lso_axis_tkeep                            ),
  .i_axis_tready                      ( lso_axis_tready                           )
);

logic [1:0] event_internal_int;
assign event_internal_int = '0; // Driven within ctrl_bus_evt_int

assign event_vec = {i_sif_event,11'h0,uart_irq,event_internal_int,spi_busy,i2c_busy};

//------------------------------------------------------------------------------------------------//
// System Initialization
//------------------------------------------------------------------------------------------------//

logic         init_done;
logic         sys_init_done;
logic         eeprom_dval;

sys_init #(
  .N_REG      ( `N_INIT_REG    )
) u_sys_init   (
  // clock and reset
  .i_aclk     ( i_apb_clk      ),
  .i_arst     ( o_apb_rst      ),
  .o_apb_m2s  ( m_apb_m2s [1]  ),
  .i_apb_s2m  ( m_apb_s2m [1]  ),
  // control
  .i_init     ( 1'b1           ),
  .i_init_reg ( init_reg       ),
  .o_done     ( sys_init_done  )
);

logic eeprom_dval_r;
always_ff @(posedge i_apb_clk) begin
  if (o_apb_rst) begin
    eeprom_dval_r <= 1'b0;
  end
  else begin
    if (eeprom_dval) begin
      eeprom_dval_r <= eeprom_dval;
    end
  end
end

assign init_done = eeprom_dval_r && sys_init_done;
assign o_init_done = init_done;

data_sync u_init_done_sync ( .clk (i_hif_clk), .rst_n (!o_hif_rst), .sync_in(init_done), .sync_out(init_done_hif_clk) );

//------------------------------------------------------------------------------------------------//
// APB Interconnect
//------------------------------------------------------------------------------------------------//

// Peripheral ports (SPI[0], I2C[1], UART[2])
apb_m2s s_apb_m2s_per [3];
apb_s2m s_apb_s2m_per [3];

apb_intc_top #(
  .N_EXT_APB        ( `REG_INST          ),
  .N_SENSOR_RX      ( NUM_SENSOR_RX      ),
  .N_SENSOR_RX_MOD  ( num_sen_rx_mod     ),
  .N_SENSOR_TX      ( NUM_SENSOR_TX      ),
  .N_SENSOR_TX_MOD  ( num_sen_tx_mod     ),
  .N_HOST           ( `HOST_IF_INST      ),
  .N_HOST_MOD       ( num_host_mod       ),
  .N_PER            ( 3                  ),
  .N_MPORT          ( 4                  )
) u_apb_intc (
  .i_aclk           ( i_apb_clk          ),
  .i_arst           ( o_apb_rst          ),
  // Connect to Decoder
  .i_apb_m2s        ( m_apb_m2s          ),
  .o_apb_s2m        ( m_apb_s2m          ),
  // Connect to Device Modules
  .i_apb_s2m_dev    ( s_apb_s2m_dev      ),
  .o_apb_m2s_dev    ( s_apb_m2s_dev      ),
  // Connect to Sensor RX Module
  .i_apb_s2m_sen_rx ( s_apb_s2m_sen_rx   ),
  .o_apb_m2s_sen_rx ( s_apb_m2s_sen_rx   ),
  // Connect to Sensor TX Module
  .i_apb_s2m_sen_tx ( s_apb_s2m_sen_tx   ),
  .o_apb_m2s_sen_tx ( s_apb_m2s_sen_tx   ),
  // Connect to Host Module
  .i_apb_s2m_host   ( s_apb_s2m_host     ),
  .o_apb_m2s_host   ( s_apb_m2s_host     ),
  // Connect to Host ROCE Module
  .i_apb_s2m_ram    ( s_apb_s2m_ram      ),
  .o_apb_m2s_ram    ( s_apb_m2s_ram      ),
  // Connect to Peripheral Modules
  .i_apb_s2m_per    ( s_apb_s2m_per      ),
  .o_apb_m2s_per    ( s_apb_m2s_per      ),
  // Connect to Bridge Module
  .i_apb_s2m_bridge ( s_apb_s2m_bridge   ),
  .o_apb_m2s_bridge ( s_apb_m2s_bridge   ),
  // Connect to User Modules, not using defined type struct
  .o_apb_psel       ( o_apb_psel         ),
  .o_apb_penable    ( o_apb_penable      ),
  .o_apb_paddr      ( o_apb_paddr        ),
  .o_apb_pwdata     ( o_apb_pwdata       ),
  .o_apb_pwrite     ( o_apb_pwrite       ),
  .i_apb_pready     ( i_apb_pready       ),
  .i_apb_prdata     ( i_apb_prdata       ),
  .i_apb_pserr      ( i_apb_pserr        )
);

//------------------------------------------------------------------------------------------------//
// System Timer
//------------------------------------------------------------------------------------------------//

logic [`HOST_IF_INST  -1:0] ptp_tx_axis_tvalid;
logic [`HOST_IF_INST  -1:0] ptp_tx_axis_tvalid_gated;
logic [`HOST_WIDTH    -1:0] ptp_tx_axis_tdata [`HOST_IF_INST];
logic [`HOST_IF_INST  -1:0] ptp_tx_axis_tlast;
logic [`HOST_IF_INST  -1:0] ptp_tx_axis_tuser;
logic [`HOSTKEEP_WIDTH-1:0] ptp_tx_axis_tkeep [`HOST_IF_INST];
logic [`HOST_IF_INST  -1:0] ptp_tx_axis_tready;

`ifdef BRIDGE_IF_INST
  assign ptp_tx_axis_tvalid_gated = 1'b0;
`else
  always_comb begin
    ptp_tx_axis_tvalid_gated = '0;
    ptp_tx_axis_tvalid_gated[0] = ptp_tx_axis_tvalid;
  end
`endif

`ifndef EXT_PTP
  ptp_top #(
    .HIF_CLK_FREQ     ( `HIF_CLK_FREQ          ),  // clock frequency in Hz
    .PTP_CLK_FREQ     ( `PTP_CLK_FREQ          ),
    .AXI_DWIDTH       ( `HOST_WIDTH            ),
    .NUM_HOST         ( `HOST_IF_INST          ),
    .SYNC_CLK         ( SYNC_CLK_HIF_PTP       )
  ) u_ptp_top (
    .i_pclk           ( i_ptp_clk              ),
    .i_prst           ( o_ptp_rst              ),
    .i_apb_clk        ( i_apb_clk              ),
    .i_apb_rst        ( o_apb_rst              ),
    .i_apb_m2s        ( s_apb_m2s_dev[1]       ),
    .o_apb_s2m        ( s_apb_s2m_dev[1]       ),
    .i_hif_clk        ( i_hif_clk              ),
    .i_hif_rst        ( o_hif_rst              ),
    .i_axis_tdata     ( ptp_rx_axis_tdata[0]   ),
    .i_axis_tkeep     ( ptp_rx_axis_tkeep[0]   ),
    .i_axis_tvalid    ( ptp_rx_axis_tvalid[0]  ),
    .i_axis_tuser     ( ptp_rx_axis_tuser[0]   ),
    .i_axis_tlast     ( ptp_rx_axis_tlast[0]   ),
    .o_axis_tready    ( ptp_rx_axis_tready[0]  ),
    .o_axis_tdata     ( ptp_tx_axis_tdata[0]   ),
    .o_axis_tkeep     ( ptp_tx_axis_tkeep[0]   ),
    .o_axis_tvalid    ( ptp_tx_axis_tvalid[0]  ),
    .o_axis_tuser     ( ptp_tx_axis_tuser[0]   ),
    .o_axis_tlast     ( ptp_tx_axis_tlast[0]   ),
    .i_axis_tready    ( ptp_tx_axis_tready[0]  ),
    .i_dev_mac_addr   ( dev_mac_addr[0]        ),
    .i_ptp_tx_ts_vld  ( ptp_val[0]             ),
    .o_ptp_en_sync_rx ( ptp_en_sync_rx         ),
    .i_enable         ( init_done_hif_clk      ),
    .o_pps            ( pps                    ), // pulse per second
    .o_sec            ( ptp_sec                ), // v2 PTP seconds
    .o_nano_sec       ( ptp_nano_sec           ), // v2 PTP nano seconds round to the nearest nano second
    .o_frac_nano_sec  ( ptp_frac_nano_sec      )  // v2 PTP fraction nano seconds
  );

  assign o_ptp_sec     = ptp_sec;
  assign o_ptp_nanosec = ptp_nano_sec;
  assign o_pps         = pps;

  generate
    for (i=1; i<`HOST_IF_INST; i++) begin: gen_ptp_tx_axis_tdata
      assign ptp_tx_axis_tdata[i] = '0;
      assign ptp_tx_axis_tkeep[i] = '0;
      assign ptp_tx_axis_tvalid[i] = '0;
      assign ptp_tx_axis_tlast[i] = '0;
      assign ptp_tx_axis_tuser[i] = '0;
    end
  endgenerate

`else
  assign ptp_tx_axis_tdata  = '{default:0};
  assign ptp_tx_axis_tkeep  = '{default:0};
  assign ptp_tx_axis_tvalid = '{default:0};
  assign ptp_tx_axis_tlast  = '{default:0};
  assign ptp_tx_axis_tuser  = '{default:0};

  assign ptp_en_sync_rx     = 1'b0;
  assign ptp_rx_axis_tready = 1'b0;
  assign s_apb_s2m_dev[1]   = '{default:0};
  assign ptp_sec            = i_ptp_sec;
  assign ptp_nano_sec       = i_ptp_nanosec;
  assign ptp_frac_nano_sec  = 0;

  logic sec_r;
  always_ff @(posedge i_ptp_clk) begin
    if (o_ptp_rst) begin
      sec_r <= 1'b0;
      pps   <= 1'b0;
    end
    else begin
      sec_r <= i_ptp_sec[0];
      pps   <= (sec_r != i_ptp_sec[0]) ? 1'b1 : 1'b0;
    end
  end
`endif

//------------------------------------------------------------------------------------------------//
// Global Registers
//------------------------------------------------------------------------------------------------//

logic [63:0] hsb_ctrl;


glb_ctrl_top #(
  .N_GPIO           ( `GPIO_INST                       ),
  .GPIO_RST_VAL     ( GPIO_RESET_VALUE                 ),
  .SYNC_CLK         ( SYNC_CLK_HIF_APB                 )
) u_glb_ctrl   (
  // clock and reset
  .i_pclk      ( i_hif_clk                             ),
  .i_prst      ( o_hif_rst                             ),
  // Register Map, abp clk domain
  .i_aclk      ( i_apb_clk                             ),
  .i_arst      ( o_apb_rst                             ),
  .i_apb_m2s   ( s_apb_m2s_dev[0]                      ),
  .o_apb_s2m   ( s_apb_s2m_dev[0]                      ),
  // HSB Control
  .i_hsb_ver   ( {BUILD_REV[15 :0],HOLOLINK_REV[15:0]} ),
  .i_hsb_date  ( BUILD_REV [47:16]                     ),
  .i_hsb_stat  ( '0                                    ),
  .o_hsb_ctrl  ( hsb_ctrl                              ),
  // GPIO
  .i_gpio      ( i_gpio                                ),
  .o_gpio      ( o_gpio                                ),
  .o_gpio_dir  ( o_gpio_dir                            )
);

assign o_sw_sen_rst = ~hsb_ctrl[63:32];
assign o_sw_sys_rst =  hsb_ctrl[3];



//------------------------------------------------------------------------------------------------//
// Enumeration Packet
//------------------------------------------------------------------------------------------------//

// EEPROM Information
logic [47 :0] eeprom_mac_addr;
logic         eeprom_ip_addr_vld;
logic [15 :0] eeprom_board_id;
logic [159:0] eeprom_board_ver;
logic [55 :0] eeprom_board_sn;
logic [15 :0] eeprom_dev_ver;
logic [15 :0] eeprom_dev_crc;
logic [31 :0] eeprom_misc;
logic         eeprom_crc_err;

`ifdef ENUM_EEPROM

  eeprom_info #(
    .CLK_FREQ              ( `APB_CLK_FREQ         ),
    .REG_ADDR_BITS         ( `EEPROM_REG_ADDR_BITS )
  ) u_eeprom_info (
    // clock and reset
    .i_aclk                ( i_apb_clk                ),
    .i_arst                ( o_apb_rst                ),
    // control
    .i_init                ( sys_init_done            ),
    // APB Interface
    .o_apb_m2s             ( m_apb_m2s       [2]      ),
    .i_apb_s2m             ( m_apb_s2m       [2]      ),
    // EEPROM Fields
    .eeprom_dval           ( eeprom_dval              ),
    .mac_addr              ( eeprom_mac_addr          ),
    .ip_addr               ( eeprom_base_ip_addr      ),
    .ip_addr_vld           ( eeprom_ip_addr_vld       ),
    .board_version         ( eeprom_board_ver         ),
    .board_sn              ( eeprom_board_sn          ),
    .eeprom_crc_err        ( eeprom_crc_err           )
  );

  generate
    for (i=0; i<`HOST_IF_INST; i++) begin: gen_mac_addr
      always_ff @(posedge i_apb_clk) begin
        dev_mac_addr[i]  <= eeprom_mac_addr + i;
      end
    end
  endgenerate

`else

  assign eeprom_dval      = i_enum_vld;
  assign eeprom_mac_addr  = 0;
  assign eeprom_board_ver = 0;
  assign eeprom_board_sn  = i_board_sn;
  assign m_apb_m2s[2]     = '{default:0};

  assign eeprom_base_ip_addr = 32'h00000000;
  assign eeprom_ip_addr_vld  = 1'b0;

  generate
    for (i=0; i<`HOST_IF_INST; i++) begin: gen_mac_addr
      always_ff @(posedge i_apb_clk) begin
        dev_mac_addr[i]  <= i_mac_addr[i];
      end
    end
  endgenerate

`endif
assign eeprom_dev_crc   = 'd0;
assign eeprom_misc      = 'd0;
assign eeprom_dev_ver   = HOLOLINK_REV[15:0];
assign eeprom_board_id  = 16'd0;

assign enum_data = {
  eeprom_misc[31:0],
  eeprom_dev_crc,
  eeprom_dev_ver,
  eeprom_board_sn,
  eeprom_board_ver,
  eeprom_board_id
};

data_sync u_eeprom_ip_addr_vld_sync (
  .clk      ( i_hif_clk                  ),
  .rst_n    ( !o_hif_rst                 ),
  .sync_in  ( eeprom_ip_addr_vld         ),
  .sync_out ( eeprom_ip_addr_vld_hif_clk )
);

//------------------------------------------------------------------------------------------------//
// Ethernet Packetizer
//------------------------------------------------------------------------------------------------//

// Lower ports have higher priority
// port 0       = DATA PLANE
// port 1       = PTP
// port 2       = Low Speed Control
// port 3       = Pause
// port 4       = Bridge


logic [`HOST_IF_INST-1  :0] dp_axis_tvalid;
logic [`HOST_IF_INST-1  :0] dp_axis_tlast;
logic [`HOST_WIDTH-1    :0] dp_axis_tdata    [`HOST_IF_INST];
logic [`HOSTKEEP_WIDTH-1:0] dp_axis_tkeep    [`HOST_IF_INST];
logic [`HOST_IF_INST-1  :0] dp_axis_tuser;
logic [`HOST_IF_INST-1  :0] dp_axis_tready;


// AXIS Interface
logic [PORT_NUM-1       :0] tx_axis_tvalid  [`HOST_IF_INST];
logic [PORT_NUM-1       :0] tx_axis_tlast   [`HOST_IF_INST];
logic [`HOST_WIDTH-1    :0] tx_axis_tdata   [`HOST_IF_INST][PORT_NUM];
logic [`HOSTKEEP_WIDTH-1:0] tx_axis_tkeep   [`HOST_IF_INST][PORT_NUM];
logic [PORT_NUM-1       :0] tx_axis_tuser   [`HOST_IF_INST];
logic [PORT_NUM-1       :0] tx_axis_tready  [`HOST_IF_INST];

logic [`HOST_IF_INST-1  :0] del_req_val;
generate
  for (i=0; i<`HOST_IF_INST; i++) begin: gen_eth_pkt

    assign tx_axis_tvalid[i]             = {btx_axis_tvalid                [i],  // bridge
                                            pause_axis_tvalid              [i],  // pause
                                            lso_axis_tvalid                [i],  // arp/icmp/udp_loopback
                                            ptp_tx_axis_tvalid_gated       [i],  // ptp
                                            dp_axis_tvalid                 [i]}; // sensor data

    assign tx_axis_tlast[i]              = {btx_axis_tlast                 [i],
                                            pause_axis_tlast               [i],
                                            lso_axis_tlast                    ,
                                            ptp_tx_axis_tlast              [i],
                                            dp_axis_tlast                  [i]};

    assign tx_axis_tdata[i]              = '{ dp_axis_tdata                  [i],
                                              ptp_tx_axis_tdata             [i],
                                              lso_axis_tdata                   ,
                                              pause_axis_tdata              [i],
                                              btx_axis_tdata                [i]};

    assign tx_axis_tkeep[i]              = '{ dp_axis_tkeep                  [i],
                                              ptp_tx_axis_tkeep             [i],
                                              lso_axis_tkeep                   ,
                                              pause_axis_tkeep              [i],
                                              btx_axis_tkeep                [i]};

    assign tx_axis_tuser[i]              =  { btx_axis_tuser                 [i],
                                              pause_axis_tuser              [i],
                                              lso_axis_tuser                   ,
                                              ptp_tx_axis_tuser             [i],
                                              dp_axis_tuser                 [i]};

    assign {btx_axis_tready               [i],
            pause_axis_tready             [i],
            lso_axis_tready               [i],
            ptp_tx_axis_tready            [i],
            dp_axis_tready                [i]} = tx_axis_tready[i];

    eth_pkt #(
      .N_INPT                   ( PORT_NUM                                  ),
      .W_DATA                   ( `HOST_WIDTH                               ),
      .SYNC_CLK                 ( SYNC_CLK_HIF_APB                          )
    ) u_eth_pkt  (
      .i_pclk                   ( i_hif_clk                                 ),
      .i_prst                   ( o_hif_rst                                 ),
      // Register Map, abp clk domain
      .i_aclk                   ( i_apb_clk                                 ),
      .i_arst                   ( o_apb_rst                                 ),
      .i_apb_m2s                ( s_apb_m2s_host [(i*num_host_mod)+eth_pkt] ),
      .o_apb_s2m                ( s_apb_s2m_host [(i*num_host_mod)+eth_pkt] ),
      .o_pkt_inc                ( pkt_inc                 [i]               ),
      //PTP Timestamp
      .o_ptp_val                ( ptp_val                 [i]               ),
      .o_del_req_val            ( del_req_val             [i]               ),
      // AXIS From Multiple Sources
      .i_axis_tvalid            ( tx_axis_tvalid          [i]               ),
      .i_axis_tlast             ( tx_axis_tlast           [i]               ),
      .i_axis_tkeep             ( tx_axis_tkeep           [i]               ),
      .i_axis_tdata             ( tx_axis_tdata           [i]               ),
      .i_axis_tuser             ( tx_axis_tuser           [i]               ),
      .o_axis_tready            ( tx_axis_tready          [i]               ),
      // AXIS to MAC
      .o_axis_tvalid            ( o_hif_axis_tvalid       [i]               ),
      .o_axis_tlast             ( o_hif_axis_tlast        [i]               ),
      .o_axis_tdata             ( o_hif_axis_tdata        [i]               ),
      .o_axis_tkeep             ( o_hif_axis_tkeep        [i]               ),
      .o_axis_tuser             ( o_hif_axis_tuser        [i]               ),
      .i_axis_tready            ( i_hif_axis_tready       [i]               )
    );

  end
endgenerate

//------------------------------------------------------------------------------------------------//
// Peripheral Control
//------------------------------------------------------------------------------------------------//


streaming_cdc #(
  .DATA_WIDTH ( 80                                        ),
  .SRC_FREQ   ( `PTP_CLK_FREQ                             ),
  .DST_FREQ   ( `HIF_CLK_FREQ                             )
) u_ptp_hif_cdc (
  .i_src_clk  ( i_ptp_clk                                 ),
  .i_dst_clk  ( i_hif_clk                                 ),
  .i_src_rst  ( o_ptp_rst                                 ),
  .i_dst_rst  ( o_hif_rst                                 ),
  .i_src_data ( {ptp_sec, ptp_nano_sec}                   ),
  .o_dst_data ( {ptp_sec_sync_hif, ptp_nano_sec_sync_hif} ),
  .o_dst_valid ( ptp_sync_hif_valid                       )
);

streaming_cdc #(
  .DATA_WIDTH ( 80                                        ),
  .SRC_FREQ   ( `PTP_CLK_FREQ                             ),
  .DST_FREQ   ( `APB_CLK_FREQ                             )
) u_ptp_apb_cdc (
  .i_src_clk  ( i_ptp_clk                                 ),
  .i_dst_clk  ( i_apb_clk                                 ),
  .i_src_rst  ( o_ptp_rst                                 ),
  .i_dst_rst  ( o_apb_rst                                 ),
  .i_src_data ( {ptp_sec, ptp_nano_sec}                   ),
  .o_dst_data ( {ptp_sec_sync_apb, ptp_nano_sec_sync_apb} ),
  .o_dst_valid ( ptp_sync_apb_valid                       )
);

`ifdef SPI_INST

  logic                         spi_csn;
  logic                         spi_sck;
  logic                         spi_oen;
  logic [3                  :0] spi_sdio_i;
  logic [3                  :0] spi_sdio_o;
  logic [$clog2(`SPI_INST)-1:0] spi_bus_en;

  assign spi_sdio_i = i_spi_sdio[spi_bus_en];
  generate
    for (i=0; i<`SPI_INST; i++) begin : spi_output
      assign o_spi_csn [i] = (spi_bus_en == i) ? spi_csn    : 1'b1;
      assign o_spi_sck [i] = (spi_bus_en == i) ? spi_sck    : 1'b0;
      assign o_spi_sdio[i] = (spi_bus_en == i) ? spi_sdio_o : 4'h0;
      assign o_spi_oen [i] = (spi_bus_en == i) ? spi_oen    : 1'b0;
    end
  endgenerate

  spi_ctrl_fsm  #(
  `ifdef PERI_RAM_DEPTH
    .RAM_DEPTH        ( `PERI_RAM_DEPTH  ),
  `endif
    .NUM_INST         ( `SPI_INST        )
  ) spi_ctrl_inst (
    .i_aclk           ( i_apb_clk                  ),
    .i_arst           ( o_apb_rst                  ),
    .i_apb_m2s        ( s_apb_m2s_per[apb_spi_idx] ),
    .o_apb_s2m        ( s_apb_s2m_per[apb_spi_idx] ),
    .CS_N             ( spi_csn                    ),
    .SCK              ( spi_sck                    ),
    .SDIO_en          ( spi_oen                    ),
    .SDIO_in          ( spi_sdio_i                 ),
    .SDIO_out         ( spi_sdio_o                 ),
    .o_busy           ( spi_busy                   ),
    .o_bus_en         ( spi_bus_en                 )
  );

`else

  assign s_apb_s2m_per[apb_spi_idx].pready = 0;
  assign s_apb_s2m_per[apb_spi_idx].prdata = 0;
  assign s_apb_s2m_per[apb_spi_idx].pserr = 0;

`endif

//------------------------------------------------------------------------------------------------//
// I2C Peripheral
//------------------------------------------------------------------------------------------------//

`ifdef I2C_INST

  logic                         i2c_scl_i;
  logic                         i2c_sda_i;
  logic                         i2c_scl_en;
  logic                         i2c_sda_en;
  logic [$clog2(`I2C_INST)-1:0] i2c_bus_en;

  assign i2c_scl_i = i_i2c_scl[i2c_bus_en];
  assign i2c_sda_i = i_i2c_sda[i2c_bus_en];

  generate
    for (i=0; i<`I2C_INST; i++) begin : i2c_en
      assign o_i2c_scl_en[i] = (i2c_bus_en == i) ? i2c_scl_en : 1'b1;
      assign o_i2c_sda_en[i] = (i2c_bus_en == i) ? i2c_sda_en : 1'b1;
    end
  endgenerate


  i2c_ctrl_fsm  #(
  `ifdef PERI_RAM_DEPTH
    .RAM_DEPTH        ( `PERI_RAM_DEPTH  ),
  `endif
    .NUM_INST         ( `I2C_INST        )
  ) i2c_ctrl_inst (
    .i_aclk           ( i_apb_clk                  ),
    .i_arst           ( o_apb_rst                  ),
    .scl_i            ( i2c_scl_i                  ),
    .sda_i            ( i2c_sda_i                  ),
    .scl_o            ( i2c_scl_en                 ),
    .sda_o            ( i2c_sda_en                 ),
    .i_start          ( '1                         ),
    .o_data           (                            ),
    .o_data_valid     (                            ),
    .o_busy           ( i2c_busy                   ),
    .o_bus_en         ( i2c_bus_en                 ),
    .i_apb_m2s        ( s_apb_m2s_per[apb_i2c_idx] ),
    .o_apb_s2m        ( s_apb_s2m_per[apb_i2c_idx] )
  );
`else

  assign s_apb_s2m_per[apb_i2c_idx].pready = 0;
  assign s_apb_s2m_per[apb_i2c_idx].prdata = 0;
  assign s_apb_s2m_per[apb_i2c_idx].pserr = 0;

`endif

//------------------------------------------------------------------------------------------------//
// UART Peripheral
//------------------------------------------------------------------------------------------------//
`ifdef UART_INST
  uart_ctrl_fsm #(
    .NUM_INST        ( `UART_INST ),
    `ifdef SIMULATION
      .FIFO_DEPTH      ( 64        ),
    `else
      .FIFO_DEPTH      ( 256       ),
    `endif
    .UART_ADDR_WIDTH ( 9          )
  ) uart_ctrl_inst (
    .i_aclk               ( i_apb_clk                   ),
    .i_arst               ( o_apb_rst                   ),
    .i_apb_m2s            ( s_apb_m2s_per[apb_uart_idx] ),
    .o_apb_s2m            ( s_apb_s2m_per[apb_uart_idx] ),
    .uart_tx              ( o_uart_tx                   ),
    .uart_rx              ( i_uart_rx                   ),
    .uart_busy            ( o_uart_busy                 ),
    .uart_interrupt       ( uart_irq                    ),
    .uart_enable_override ( 1'b0                        ),
    .uart_cts             ( i_uart_cts                   ),
    .uart_rts             ( o_uart_rts                   )
  );
`else
  assign s_apb_s2m_per[apb_uart_idx] = '{default:0};
  assign uart_irq = 0;
`endif

// assign s_apb_s2m_per[apb_uart_idx] = '{default:0};

//------------------------------------------------------------------------------------------------//
// Hololink Bridge
//------------------------------------------------------------------------------------------------//
`ifdef BRIDGE_IF_INST

//------------------------------------------------------------------------------------------------//
// SIF Buffer
//------------------------------------------------------------------------------------------------//
  genvar m2;
  logic   [`BRIDGE_IF_INST-1:0] sif_buf_axis_tvalid;
  logic   [`HOST_WIDTH-1:0]     sif_buf_axis_tdata [`BRIDGE_IF_INST-1:0];
  logic   [`BRIDGE_IF_INST-1:0] sif_buf_axis_tlast;
  logic   [`HOSTKEEP_WIDTH-1:0] sif_buf_axis_tkeep [`BRIDGE_IF_INST-1:0];
  logic   [`BRIDGE_IF_INST-1:0] sif_buf_axis_tready;

  localparam HOST_BUF_DEPTH = `HOST_MTU * 2 / (`HOST_WIDTH / 8);

  generate
    for (m2=0; m2<`BRIDGE_IF_INST; m2++) begin : gen_sif_buffer
      axis_buffer # (
        .IN_DWIDTH         ( SIF_RX_WIDTH[m2]                                ),
        .OUT_DWIDTH        ( `HOST_WIDTH                                     ),
        .WAIT2SEND         ( 1                                               ),
        .BUF_DEPTH         ( HOST_BUF_DEPTH                                  ),
        .W_USER            ( 2                                               ),
        .DUAL_CLOCK        ( 1                                               )
      ) u_axis_sif_rx_buffer (
        .in_clk            ( sif_rx_clk [m2]                                 ),
        .in_rst            ( o_sif_rx_rst [m2]                               ),
        .out_clk           ( i_hif_clk                                       ),
        .out_rst           ( o_hif_rst                                       ),
        .i_axis_rx_tvalid  ( i_sif_axis_tvalid   [m2]                        ),
        .i_axis_rx_tdata   ( i_sif_axis_tdata    [m2][0+:SIF_RX_WIDTH[m2]]   ),
        .i_axis_rx_tlast   ( i_sif_axis_tlast    [m2]                        ),
        .i_axis_rx_tuser   ( '0                                              ),
        .i_axis_rx_tkeep   ( i_sif_axis_tkeep    [m2][0+:SIF_RX_WIDTH[m2]/8] ),
        .o_axis_rx_tready  ( o_sif_axis_tready   [m2]                        ),
        .o_fifo_aempty     (                                                 ),
        .o_fifo_afull      (                                                 ),
        .o_fifo_empty      (                                                 ),
        .o_fifo_full       (                                                 ),
        .o_axis_tx_tvalid  ( sif_buf_axis_tvalid [m2]                        ),
        .o_axis_tx_tdata   ( sif_buf_axis_tdata  [m2]                        ),
        .o_axis_tx_tlast   ( sif_buf_axis_tlast  [m2]                        ),
        .o_axis_tx_tuser   (                                                 ),
        .o_axis_tx_tkeep   ( sif_buf_axis_tkeep  [m2]                        ),
        .i_axis_tx_tready  ( sif_buf_axis_tready [m2]                        )
      );
    end
  endgenerate

//------------------------------------------------------------------------------------------------//
// Routing Table
//------------------------------------------------------------------------------------------------//

  logic [47:0]                              rt_mac_addr [`HOST_IF_INST];
  logic [`HOST_IF_INST-1:0]                 rt_mac_req;
  logic [`BRIDGE_IF_INST+`HOST_IF_INST-1:0] rt_dest_port;
  logic [`HOST_IF_INST-1:0]                 rt_dest_val;
  logic [`BRIDGE_IF_INST-1:0]               chk_del_req;
  logic [`BRIDGE_IF_INST-1:0]               is_del_req;

  routing_table #(
    .N_INPT ( `BRIDGE_IF_INST   ),
    .N_HOST ( `HOST_IF_INST     ),
    .W_DATA ( `HOST_WIDTH       ),
    .DEPTH  ( 64                )
  ) u_routing_table(
    .i_clk          ( i_hif_clk                                ),
    .i_rst          ( o_hif_rst                                ),
    .i_axis_tvalid  ( sif_buf_axis_tvalid                      ),
    .i_axis_tready  ( sif_buf_axis_tready                      ),
    .i_axis_tlast   ( sif_buf_axis_tlast                       ),
    .i_axis_tdata   ( sif_buf_axis_tdata                       ),
    .i_axis_tkeep   ( sif_buf_axis_tkeep                       ),
    .i_axis_tuser   (                                          ),
    .i_mac_addr     ( rt_mac_addr                              ),
    .i_mac_req      ( rt_mac_req                               ),
    .o_dest_port    ( rt_dest_port                             ),
    .o_dest_val     ( rt_dest_val                              ),
    .o_chk_del_req  ( chk_del_req                              ),
    .o_is_del_req   ( is_del_req                               )
  );


//------------------------------------------------------------------------------------------------//
// Bridge TX
//------------------------------------------------------------------------------------------------//

  logic [47:0] dest_mac_addr [`HOST_IF_INST];
  generate
    for (i=0; i<`HOST_IF_INST; i++) begin
      assign dest_mac_addr[i] = dest_info[i][47:0];
    end
  endgenerate

  bridge_pkt_proc #(
    .AXI_DWIDTH       ( `HOST_WIDTH                                   ),
    .NUM_HOST         ( `HOST_IF_INST                                 ),
    .NUM_SENSOR       ( `BRIDGE_IF_INST                               ),
    .FREQ             ( `HIF_CLK_FREQ                                 ),
    .SIF_DWIDTH       ( `DATAPATH_WIDTH                               ),
    .W_SIF_TX         ( SIF_TX_WIDTH                                  ),
    .HOST_MTU         ( `HOST_MTU                                     ),
    .SYNC_CLK         ( SYNC_CLK_HIF_APB                              )
  ) u_bridge_pkt_proc (
    .i_pclk           ( i_hif_clk                                     ),
    .i_prst           ( o_hif_rst                                     ),
    .i_sif_clk        ( sif_tx_clk                                    ),
    .i_sif_rst        ( o_sif_tx_rst                                  ),
    .i_aclk           ( i_apb_clk                                     ),
    .i_arst           ( o_apb_rst                                     ),
    .i_apb_m2s        ( s_apb_m2s_bridge                              ),
    .o_apb_s2m        ( s_apb_s2m_bridge                              ),
    .i_dest_info      ( dest_mac_addr                                 ),
    .o_mac_addr       ( rt_mac_addr                                   ),
    .o_mac_req        ( rt_mac_req                                    ),
    .i_dest_port      ( rt_dest_port                                  ),
    .i_dest_val       ( rt_dest_val                                   ),
    .i_ptp_sync_msg   ( is_ptp_sync_msg[0]                            ),
    .i_del_req_val    ( del_req_val                                   ),
    .i_chk_del_req    ( chk_del_req                                   ),
    .i_is_del_req     ( is_del_req                                    ),
    // Input HIF interface
    .i_hif_axis_tvalid( brx_axis_tvalid                               ),
    .i_hif_axis_tdata ( brx_axis_tdata                                ),
    .i_hif_axis_tlast ( brx_axis_tlast                                ),
    .i_hif_axis_tuser ( brx_axis_tuser                                ),
    .i_hif_axis_tkeep ( brx_axis_tkeep                                ),
    .o_hif_axis_tready( brx_axis_tready                               ),
    // Input SIF Interface
    .i_sif_axis_tvalid( sif_buf_axis_tvalid                           ),
    .i_sif_axis_tdata ( sif_buf_axis_tdata                            ),
    .i_sif_axis_tlast ( sif_buf_axis_tlast                            ),
    .i_sif_axis_tuser ( '0                                            ),
    .i_sif_axis_tkeep ( sif_buf_axis_tkeep                            ),
    .o_sif_axis_tready( sif_buf_axis_tready                           ),
    // Output HIF Interface
    .o_hif_axis_tvalid( btx_axis_tvalid                               ),
    .o_hif_axis_tdata ( btx_axis_tdata                                ),
    .o_hif_axis_tlast ( btx_axis_tlast                                ),
    .o_hif_axis_tuser ( btx_axis_tuser                                ),
    .o_hif_axis_tkeep ( btx_axis_tkeep                                ),
    .i_hif_axis_tready( btx_axis_tready                               ),
    // Output SIF Interface
    .o_sif_axis_tvalid( o_sif_axis_tvalid                             ),
    .o_sif_axis_tdata ( o_sif_axis_tdata                              ),
    .o_sif_axis_tlast ( o_sif_axis_tlast                              ),
    .o_sif_axis_tuser (                                               ),
    .o_sif_axis_tkeep ( o_sif_axis_tkeep                              ),
    .i_sif_axis_tready( i_sif_axis_tready                             )
  );

  assign o_sif_axis_tuser = '{default:0};

`else
  assign btx_axis_tvalid = '{default:0};
  assign btx_axis_tdata  = '{default:0};
  assign btx_axis_tlast  = '{default:0};
  assign btx_axis_tuser  = '{default:0};
  assign btx_axis_tkeep  = '{default:0};
  assign brx_axis_tready = '{default:'1};
  assign s_apb_s2m_bridge.pready   = 0;
  assign s_apb_s2m_bridge.prdata   = 0;
  assign s_apb_s2m_bridge.pserr    = 0;
`endif

//------------------------------------------------------------------------------------------------//
// Hololink Data Plane
//------------------------------------------------------------------------------------------------//
`ifndef BRIDGE_IF_INST
  `ifdef SENSOR_RX_IF_INST
    // Function to calculate total SOF width needed
    function automatic int calc_total_sof_width();
      int total_width = 0;
      for (int j = 0; j < `SENSOR_RX_IF_INST; j++) begin
        int sif_vp_inst = SIF_RX_PACKETIZER_EN[j] ? (SIF_RX_VP_COUNT[j] == 0) ? 1 : SIF_RX_VP_COUNT[j] : 1;
        total_width += sif_vp_inst;
      end
      return total_width;
    endfunction

    localparam TOTAL_VP_WIDTH = calc_total_sof_width();

    logic [TOTAL_VP_WIDTH-1:0]   sif_sof;
    logic [TOTAL_VP_WIDTH-1:0]   pkt_rx_axis_tvalid;
    logic [TOTAL_VP_WIDTH-1:0]   pkt_rx_axis_tlast;
    logic [`HOST_WIDTH-1:0]      pkt_rx_axis_tdata [TOTAL_VP_WIDTH-1:0];
    logic [`HOSTKEEP_WIDTH-1:0]  pkt_rx_axis_tkeep [TOTAL_VP_WIDTH-1:0];
    logic [TOTAL_VP_WIDTH-1:0]   pkt_rx_axis_tuser;
    logic [TOTAL_VP_WIDTH-1:0]   pkt_rx_axis_tready;

    logic [`SENSOR_RX_IF_INST-1  :0] sif_axis_tready;

    logic [`SENSOR_RX_IF_INST-1  :0] sen_rx_data_gen_axis_tvalid;
    logic [`DATAPATH_WIDTH-1     :0] sen_rx_data_gen_axis_tdata  [`SENSOR_RX_IF_INST];
    logic [`SENSOR_RX_IF_INST-1  :0] sen_rx_data_gen_axis_tlast;
    logic [`DATAUSER_WIDTH-1     :0] sen_rx_data_gen_axis_tuser  [`SENSOR_RX_IF_INST];
    logic [`DATAKEEP_WIDTH-1     :0] sen_rx_data_gen_axis_tkeep  [`SENSOR_RX_IF_INST];
    logic [`SENSOR_RX_IF_INST-1  :0] sen_rx_data_gen_axis_tready;
    logic [`SENSOR_RX_IF_INST-1  :0] sen_rx_data_gen_axis_mux;

    logic [`SENSOR_RX_IF_INST-1  :0] sen_rx_mux_axis_tvalid;
    logic [`DATAPATH_WIDTH-1     :0] sen_rx_mux_axis_tdata  [`SENSOR_RX_IF_INST];
    logic [`SENSOR_RX_IF_INST-1  :0] sen_rx_mux_axis_tlast;
    logic [`DATAUSER_WIDTH-1     :0] sen_rx_mux_axis_tuser  [`SENSOR_RX_IF_INST];
    logic [`DATAKEEP_WIDTH-1     :0] sen_rx_mux_axis_tkeep  [`SENSOR_RX_IF_INST];
    logic [`SENSOR_RX_IF_INST-1  :0] sen_rx_mux_axis_tready;

    `ifdef SIF_RX_DATA_GEN
//------------------------------------------------------------------------------------------------//
// Sensor RX Data Generator
//------------------------------------------------------------------------------------------------//
      generate
        for (i=0; i<`SENSOR_RX_IF_INST; i++) begin: gen_data_gen_inst
          data_gen # (
            .W_DATA              ( SIF_RX_WIDTH[i]                                          )
          ) u_sen_rx_data_gen (
            .i_apb_clk           ( i_apb_clk                                                ),
            .i_apb_rst           ( o_apb_rst                                                ),
            .i_sif_clk           ( sif_rx_clk[i]                                            ),
            .i_sif_rst           ( o_sif_rx_rst[i]                                          ),
            .i_apb_m2s           ( s_apb_m2s_sen_rx[(i*num_sen_rx_mod)+sen_rx_data_gen]     ),
            .o_apb_s2m           ( s_apb_s2m_sen_rx[(i*num_sen_rx_mod)+sen_rx_data_gen]     ),
            .o_data_gen_axis_mux ( sen_rx_data_gen_axis_mux      [i]                        ),
            .o_axis_tvalid       ( sen_rx_data_gen_axis_tvalid   [i]                        ),
            .o_axis_tdata        ( sen_rx_data_gen_axis_tdata    [i][SIF_RX_WIDTH[i]-1:0]   ),
            .o_axis_tkeep        ( sen_rx_data_gen_axis_tkeep    [i][SIF_RX_WIDTH[i]/8-1:0] ),
            .o_axis_tuser        (                                                          ),
            .o_axis_tlast        ( sen_rx_data_gen_axis_tlast    [i]                        ),
            .i_axis_tready       ( sif_axis_tready               [i]                        )
          );

          if (SIF_RX_WIDTH[i] != `DATAPATH_WIDTH) begin
            assign sen_rx_data_gen_axis_tdata    [i][`DATAPATH_WIDTH-1:SIF_RX_WIDTH[i]]   = '0;
            assign sen_rx_data_gen_axis_tkeep    [i][`DATAKEEP_WIDTH-1:SIF_RX_WIDTH[i]/8] = '0;
          end
          assign sen_rx_data_gen_axis_tuser[i]  = '0;
        end
      endgenerate

    `else //SIF_RX_DATA_GEN

      assign sen_rx_data_gen_axis_mux = '0;

      generate
        for (i=0; i<`SENSOR_RX_IF_INST; i++) begin: gen_data_gen_inst

          assign sen_rx_data_gen_axis_tvalid[i] = '0;
          assign sen_rx_data_gen_axis_tdata[i]  = '0;
          assign sen_rx_data_gen_axis_tkeep[i]  = '0;
          assign sen_rx_data_gen_axis_tlast[i]  = '0;
          assign sen_rx_data_gen_axis_tuser[i]  = '0;

          assign s_apb_s2m_sen_rx[(i*num_sen_rx_mod)+sen_rx_data_gen].pready = 1'b0;
          assign s_apb_s2m_sen_rx[(i*num_sen_rx_mod)+sen_rx_data_gen].prdata = '0;
          assign s_apb_s2m_sen_rx[(i*num_sen_rx_mod)+sen_rx_data_gen].pserr  = 1'b0;

        end
      endgenerate

    `endif // ifdef SIF_RX_DATA_GEN
  `endif   // ifdef SENSOR_RX_IF_INST
`endif     // ifndef BRIDGE_IF_INST

//------------------------------------------------------------------------------------------------//
// Packetizer
//------------------------------------------------------------------------------------------------//

`ifndef BRIDGE_IF_INST
  `ifdef SENSOR_RX_IF_INST
  // Function to calculate bit position for each instance
  function automatic int calc_bit_position(int inst_num);
    int pos = 0;
    for (int j = 0; j < inst_num; j++) begin
      int sif_vp_inst = SIF_RX_PACKETIZER_EN[j] ? (SIF_RX_VP_COUNT[j] == 0) ? 1 : SIF_RX_VP_COUNT[j] : 1;
      pos += sif_vp_inst;
    end
    return pos;
  endfunction

    generate
      for (i=0; i<`SENSOR_RX_IF_INST; i++) begin: gen_pack_inst
        assign sen_rx_mux_axis_tvalid[i] = sen_rx_data_gen_axis_mux[i] ? sen_rx_data_gen_axis_tvalid[i] : i_sif_axis_tvalid[i];
        assign sen_rx_mux_axis_tdata[i]  = sen_rx_data_gen_axis_mux[i] ? sen_rx_data_gen_axis_tdata[i]  : i_sif_axis_tdata[i];
        assign sen_rx_mux_axis_tkeep[i]  = sen_rx_data_gen_axis_mux[i] ? sen_rx_data_gen_axis_tkeep[i]  : i_sif_axis_tkeep[i];
        assign sen_rx_mux_axis_tuser[i]  = sen_rx_data_gen_axis_mux[i] ? sen_rx_data_gen_axis_tuser[i]  : i_sif_axis_tuser[i];
        assign sen_rx_mux_axis_tlast[i]  = sen_rx_data_gen_axis_mux[i] ? sen_rx_data_gen_axis_tlast[i]  : i_sif_axis_tlast[i];

        localparam SIF_VP_INST          = SIF_RX_PACKETIZER_EN[i] ? (SIF_RX_VP_COUNT[i]==0) ? 1 : SIF_RX_VP_COUNT [i] : 1;
        localparam PACK_SORT_RESOLUTION = SIF_RX_PACKETIZER_EN[i] ? SIF_RX_SORT_RESOLUTION[i] : 2;
        localparam PACK_VP_COUNT        = SIF_RX_PACKETIZER_EN[i] ? SIF_RX_VP_COUNT[i] : 1;
        localparam PACK_VP_SIZE         = SIF_RX_PACKETIZER_EN[i] ? SIF_RX_VP_SIZE[i] : 32;
        localparam PACK_MIXED_VP_SIZE   = 0;
        localparam PACK_DYN_VP          = 0;
        localparam PACK_NUM_CYCLES      = SIF_RX_PACKETIZER_EN[i] ? SIF_RX_NUM_CYCLES[i] : 1;

        logic [SIF_VP_INST-1:0]     pkt_raw_sof;
        logic [SIF_VP_INST-1:0]     pkt_raw_axis_tvalid;
        logic [SIF_VP_INST-1:0]     pkt_raw_axis_tlast;
        logic [`HOST_WIDTH-1:0]     pkt_raw_axis_tdata [SIF_VP_INST-1:0];
        logic [`HOSTKEEP_WIDTH-1:0] pkt_raw_axis_tkeep [SIF_VP_INST-1:0];
        logic [SIF_VP_INST-1:0]     pkt_raw_axis_tuser;
        logic [SIF_VP_INST-1:0]     pkt_raw_axis_tready;

        packetizer_top # (
          .DIN_WIDTH        ( SIF_RX_WIDTH[i]         ),
          .DOUT_WIDTH       ( `HOST_WIDTH             ),
          .W_USER           ( `DATAUSER_WIDTH         ),
          .SORT_RESOLUTION  ( PACK_SORT_RESOLUTION    ),
          .VP_COUNT         ( PACK_VP_COUNT           ),
          .VP_SIZE          ( PACK_VP_SIZE            ),
          .MIXED_VP_SIZE    ( PACK_MIXED_VP_SIZE      ),
          .DYNAMIC_VP       ( PACK_DYN_VP             ),
          .NUM_CYCLES       ( PACK_NUM_CYCLES         ),
          .PACKETIZER_ENABLE( SIF_RX_PACKETIZER_EN[i] ),
          .MTU              ( `HOST_MTU               )
        ) packetizer_top_inst (
          // Register Map, abp clk domain
          .i_aclk         ( i_apb_clk                                             ),
          .i_arst         ( o_apb_rst                                             ),
          .i_apb_m2s      ( s_apb_m2s_sen_rx [(i*num_sen_rx_mod)+sen_rx_pkt_ctrl] ),
          .o_apb_s2m      ( s_apb_s2m_sen_rx [(i*num_sen_rx_mod)+sen_rx_pkt_ctrl] ),
          .o_sof          ( pkt_raw_sof                                           ),
          //  MIPI Incoming Data
          .i_sclk         ( sif_rx_clk[i]                                         ),
          .i_srst         ( o_sif_rx_rst[i]                                       ),
          .i_axis_tvalid  ( sen_rx_mux_axis_tvalid   [i]                          ),
          .i_axis_tlast   ( sen_rx_mux_axis_tlast    [i]                          ),
          .i_axis_tdata   ( sen_rx_mux_axis_tdata    [i][SIF_RX_WIDTH[i]-1:0]     ),
          .i_axis_tkeep   ( sen_rx_mux_axis_tkeep    [i][SIF_RX_WIDTH[i]/8-1:0]   ),
          .i_axis_tuser   ( sen_rx_mux_axis_tuser    [i]                          ),
          .o_axis_tready  ( sif_axis_tready          [i]                          ),
          // Outgoing Virtual Port Interface
          .i_pclk         ( i_hif_clk                                             ),
          .i_prst         ( o_hif_rst                                             ),
          .o_axis_tvalid  ( pkt_raw_axis_tvalid                                   ),
          .o_axis_tlast   ( pkt_raw_axis_tlast                                    ),
          .o_axis_tdata   ( pkt_raw_axis_tdata                                    ),
          .o_axis_tkeep   ( pkt_raw_axis_tkeep                                    ),
          .o_axis_tuser   ( pkt_raw_axis_tuser                                    ),
          .i_axis_tready  ( pkt_raw_axis_tready                                   )
        );

        assign o_sif_axis_tready[i] = sen_rx_data_gen_axis_mux[i] ? 1'b0 : sif_axis_tready[i];

        localparam int BIT_POS       = calc_bit_position(i);
        localparam int SIF_VP_INST_I = SIF_RX_PACKETIZER_EN[i] ? (SIF_RX_VP_COUNT[i] == 0) ? 1 : SIF_RX_VP_COUNT[i] : 1;

        assign sif_sof[BIT_POS +: SIF_VP_INST_I] = pkt_raw_sof;
        assign pkt_rx_axis_tvalid[BIT_POS +: SIF_VP_INST_I] = pkt_raw_axis_tvalid;
        assign pkt_rx_axis_tlast[BIT_POS +: SIF_VP_INST_I] = pkt_raw_axis_tlast;
        assign pkt_rx_axis_tdata[BIT_POS +: SIF_VP_INST_I] = pkt_raw_axis_tdata;
        assign pkt_rx_axis_tkeep[BIT_POS +: SIF_VP_INST_I] = pkt_raw_axis_tkeep;
        assign pkt_rx_axis_tuser[BIT_POS +: SIF_VP_INST_I] = pkt_raw_axis_tuser;
        assign pkt_raw_axis_tready = pkt_rx_axis_tready[BIT_POS +: SIF_VP_INST_I];
      end
    endgenerate


//------------------------------------------------------------------------------------------------//
// Dataplane Packetizer
//------------------------------------------------------------------------------------------------//

    apb_m2s dp_pkt_apb_m2s [`HOST_IF_INST];
    apb_s2m dp_pkt_apb_s2m [`HOST_IF_INST];

    `ifndef SIMULATION
      `ifdef DISABLE_COE
        `define REMOVE_COE 1
      `else
        `define REMOVE_COE 0
      `endif
    `else
      `define REMOVE_COE 0
    `endif

    dp_pkt_top #(
      .N_INPT        ( TOTAL_VP_WIDTH   ),
      .W_DATA        (`HOST_WIDTH       ),
      .N_HOST        (`HOST_IF_INST     ),
      .MTU           (`HOST_MTU         ),
      .BUFFER_4K_REG ( 1                ),
      .DIS_COE       (`REMOVE_COE       ),
      .SYNC_CLK      ( SYNC_CLK_HIF_APB )
    ) u_dp_pkt_top (
      // clock and reset
      .i_pclk             ( i_hif_clk                                 ),
      .i_prst             ( o_hif_rst                                 ),
      .i_aclk             ( i_apb_clk                                 ),
      .i_arst             ( o_apb_rst                                 ),
      .i_apb_m2s          ( dp_pkt_apb_m2s                            ),
      .o_apb_s2m          ( dp_pkt_apb_s2m                            ),
      .i_apb_m2s_cfg      ( s_apb_m2s_ram[0]                          ),
      .o_apb_s2m_cfg      ( s_apb_s2m_ram[0]                          ),
      .i_sof              ( sif_sof                                   ),
      .i_cur_ptp          ( {ptp_sec_sync_hif, ptp_nano_sec_sync_hif} ),
      .i_dev_mac_addr     ( dev_mac_addr                              ),
      .i_dev_ip_addr      ( dev_ip_addr                               ),
      .i_axis_tvalid      ( pkt_rx_axis_tvalid                        ),
      .i_axis_tlast       ( pkt_rx_axis_tlast                         ),
      .i_axis_tdata       ( pkt_rx_axis_tdata                         ),
      .i_axis_tkeep       ( pkt_rx_axis_tkeep                         ),
      .i_axis_tuser       ( pkt_rx_axis_tuser                         ),
      .o_axis_tready      ( pkt_rx_axis_tready                        ),
      .o_axis_tvalid      ( dp_axis_tvalid                            ),
      .o_axis_tlast       ( dp_axis_tlast                             ),
      .o_axis_tdata       ( dp_axis_tdata                             ),
      .o_axis_tkeep       ( dp_axis_tkeep                             ),
      .o_axis_tuser       ( dp_axis_tuser                             ),
      .i_axis_tready      ( dp_axis_tready                            )
    );

    generate
      for (i=0; i<`HOST_IF_INST; i++) begin : gen_dp_pkt_apb
        assign dp_pkt_apb_m2s [i] = s_apb_m2s_host [dp_pkt+(num_host_mod*i)];
        assign s_apb_s2m_host [dp_pkt+(num_host_mod*i)] = dp_pkt_apb_s2m [i];
      end
    endgenerate

  `else //SENSOR_RX_IF_INST
    assign dp_axis_tvalid = '{default:0};
    assign dp_axis_tlast  = '{default:0};
    assign dp_axis_tdata  = '{default:0};
    assign dp_axis_tkeep  = '{default:0};
    assign dp_axis_tuser  = '{default:0};

    generate
      for (i=0; i<`HOST_IF_INST; i++) begin: gen_dp_pkt
        assign s_apb_s2m_host [(i*num_host_mod)+dp_pkt] = '{default:0};
      end
    endgenerate
    assign s_apb_s2m_ram[0]                             = '0;

  `endif //SENSOR_RX_IF_INST

`else //BRIDGE_IF_INST
  assign dp_axis_tvalid = '{default:0};
  assign dp_axis_tlast  = '{default:0};
  assign dp_axis_tdata  = '{default:0};
  assign dp_axis_tkeep  = '{default:0};
  assign dp_axis_tuser  = '{default:0};

  generate
    for (i=0; i<`SENSOR_RX_IF_INST*num_sen_rx_mod; i++) begin: gen_pack_inst
      assign s_apb_s2m_sen_rx [i]  = '{default:0};
    end
    for (i=0; i<`HOST_IF_INST; i++) begin: gen_dp_pkt
      assign s_apb_s2m_host [(i*num_host_mod)+dp_pkt]               = '{default:0};
    end
  endgenerate
  assign s_apb_s2m_ram[0]                                             = '0;

  assign o_sif_rx_rst = sif_rx_rst;
`endif //BRIDGE_IF_INST

//------------------------------------------------------------------------------------------------//
// Sensor Tx Interface
//------------------------------------------------------------------------------------------------//
`ifndef BRIDGE_IF_INST
  `ifdef SENSOR_TX_IF_INST

    logic [`SENSOR_TX_IF_INST-1:0] sbuf_axis_tvalid;
    logic [`HOST_WIDTH-1       :0] sbuf_axis_tdata         [`SENSOR_TX_IF_INST];
    logic [`SENSOR_TX_IF_INST-1:0] sbuf_axis_tlast;
    logic [`SENSOR_TX_IF_INST-1:0] sbuf_axis_tuser;
    logic [`HOSTKEEP_WIDTH-1   :0] sbuf_axis_tkeep         [`SENSOR_TX_IF_INST];

    logic [`SENSOR_TX_IF_INST-1:0] sbuf_axis_tvalid_out;
    logic [`DATAPATH_WIDTH-1   :0] sbuf_axis_tdata_out     [`SENSOR_TX_IF_INST];
    logic [`SENSOR_TX_IF_INST-1:0] sbuf_axis_tlast_out;
    logic [`SENSOR_TX_IF_INST-1:0] sbuf_axis_tuser_out;
    logic [`DATAKEEP_WIDTH-1   :0] sbuf_axis_tkeep_out     [`SENSOR_TX_IF_INST];

    logic [`SENSOR_TX_IF_INST-1:0] tx_axis_tvalid_out;
    logic [`DATAPATH_WIDTH-1   :0] tx_axis_tdata_out       [`SENSOR_TX_IF_INST];
    logic [`SENSOR_TX_IF_INST-1:0] tx_axis_tlast_out;
    logic [`SENSOR_TX_IF_INST-1:0] tx_axis_tuser_out;
    logic [`DATAKEEP_WIDTH-1   :0] tx_axis_tkeep_out       [`SENSOR_TX_IF_INST];

    logic [`HOST_IF_INST-1     :0] tx_pause_req            [`SENSOR_TX_IF_INST];
    logic [`HOST_IF_INST-1     :0] host_pause_req;
    logic [15                  :0] tx_pause_quanta         [`SENSOR_TX_IF_INST];
    logic [`SENSOR_TX_IF_INST-1:0] tx_strm_buffer_ready;


    sensor_tx_pkt_proc #(
      .AXI_DWIDTH       ( `HOST_WIDTH                                   ),
      .NUM_HOST         ( `HOST_IF_INST                                 ),
      .NUM_SENSOR       ( `SENSOR_TX_IF_INST                            )
    ) u_sensor_tx_pkt_proc (
      .host_clk         ( i_hif_clk                                     ),
      .host_rst         ( o_hif_rst                                     ),
      .i_dest_info      ( dest_info                                     ),
      .i_axis_tvalid    ( stx_axis_tvalid                               ),
      .i_axis_tdata     ( stx_axis_tdata                                ),
      .i_axis_tlast     ( stx_axis_tlast                                ),
      .i_axis_tuser     ( stx_axis_tuser                                ),
      .i_axis_tkeep     ( stx_axis_tkeep                                ),
      .o_axis_tvalid    ( sbuf_axis_tvalid                              ),
      .o_axis_tdata     ( sbuf_axis_tdata                               ),
      .o_axis_tlast     ( sbuf_axis_tlast                               ),
      .o_axis_tuser     ( sbuf_axis_tuser                               ),
      .o_axis_tkeep     ( sbuf_axis_tkeep                               )
    );

    generate
      for (i=0; i<`SENSOR_TX_IF_INST; i++) begin: gen_stx_buf
        assign tx_strm_buffer_ready[i]  = i_sif_axis_tready[i];
        assign o_sif_axis_tvalid[i]     = tx_axis_tvalid_out[i];
        assign o_sif_axis_tlast[i]      = tx_axis_tlast_out[i];
        assign o_sif_axis_tdata[i]      = {'0, tx_axis_tdata_out[i][SIF_TX_WIDTH[i]-1:0]};
        assign o_sif_axis_tkeep[i]      = {'0, tx_axis_tkeep_out[i][(SIF_TX_WIDTH[i]/8)-1:0]};
        assign o_sif_axis_tuser[i]      = tx_axis_tuser_out[i];

        tx_stream_buffer #
        (
          .HOST_DWIDTH            ( `HOST_WIDTH                                                      ),
          .NUM_HOSTS              ( `HOST_IF_INST                                                    ),
          .SENSOR_DWIDTH          ( SIF_TX_WIDTH[i]                                                  ),
          .FIFO_DEPTH             ( SIF_TX_BUF_SIZE                     [i]                          ),
          .MTU                    ( `HOST_MTU                                                        )
        ) u_tx_stream_buffer (
          //Clock and Reset
          .i_hif_clk              ( i_hif_clk                                                        ),
          .i_hif_rst              ( o_hif_rst                                                        ),
          .i_sif_clk              ( sif_tx_clk[i]                                                    ),
          .i_sif_rst              ( sif_tx_rst[i]                                                    ),
          .i_apb_clk              ( i_apb_clk                                                        ),
          .i_apb_rst              ( o_apb_rst                                                        ),
          //APB
          .i_apb_m2s              (s_apb_m2s_sen_tx      [i*num_sen_tx_mod]                          ),
          .o_apb_s2m              (s_apb_s2m_sen_tx      [i*num_sen_tx_mod]                          ),
          //TX Sensor Data AXIS Input
          .i_axis_tvalid          ( sbuf_axis_tvalid                    [i]                          ),
          .i_axis_tdata           ( sbuf_axis_tdata                     [i]                          ),
          .i_axis_tlast           ( sbuf_axis_tlast                     [i]                          ),
          .i_axis_tuser           ( sbuf_axis_tuser                     [i]                          ),
          .i_axis_tkeep           ( sbuf_axis_tkeep                     [i]                          ),
          //TX Sensor Data AXIS Output
          .o_axis_tvalid          ( tx_axis_tvalid_out                  [i]                          ),
          .o_axis_tdata           ( tx_axis_tdata_out                   [i][SIF_TX_WIDTH[i]-1:0]     ),
          .o_axis_tlast           ( tx_axis_tlast_out                   [i]                          ),
          .o_axis_tuser           ( tx_axis_tuser_out                   [i]                          ),
          .o_axis_tkeep           ( tx_axis_tkeep_out                   [i][(SIF_TX_WIDTH[i]/8)-1:0] ),
          .i_axis_tready          ( tx_strm_buffer_ready                [i]                          ),
          //Pause
          .o_eth_pause            ( tx_pause_req                        [i]                          ),
          .o_pause_quanta         ( tx_pause_quanta                     [i]                          )
        );

      end
    endgenerate

    //Generate the host pause request from all sensors.
    always_comb begin
      host_pause_req        = '0;
      for (int i = 0; i < `HOST_IF_INST; i++) begin
        for (int j = 0; j < `SENSOR_TX_IF_INST; j++) begin
          host_pause_req[i] = host_pause_req[i] | tx_pause_req[j][i];
        end
      end
    end

    generate
      for (i=0; i<`HOST_IF_INST; i++) begin: gen_pause
        pause_gen #(
          .AXI_DWIDTH             ( `HOST_WIDTH                           )
        ) u_pause_gen (
          .pclk                   ( i_hif_clk                             ),
          .rst                    ( o_hif_rst                             ),
          .heartbeat              ( 1'b0                                  ),
          .i_start                ( host_pause_req [i]                    ),
          .i_pause_time           ( tx_pause_quanta[0]                    ),
          .i_dev_mac_addr         ( dev_mac_addr  [i]                     ),
          .o_busy                 (                                       ),
          .o_axis_tvalid          ( pause_axis_tvalid[i]                  ),
          .o_axis_tdata           ( pause_axis_tdata[i]                   ),
          .o_axis_tlast           ( pause_axis_tlast[i]                   ),
          .o_axis_tuser           ( pause_axis_tuser[i]                   ),
          .o_axis_tkeep           ( pause_axis_tkeep[i]                   ),
          .i_axis_tready          ( pause_axis_tready[i]                  )
        );
      end
    endgenerate

  `else
    for (i=0; i<`HOST_IF_INST; i++) begin: pause_intf_zero_assign
      assign pause_axis_tvalid  [i] = '0;
      assign pause_axis_tdata   [i] = '0;
      assign pause_axis_tlast   [i] = '0;
      assign pause_axis_tuser   [i] = '0;
      assign pause_axis_tkeep   [i] = '0;
    end
  `endif
`else //Bridge is defined, tie off TX interfaces

  for (i=0; i<`HOST_IF_INST; i++) begin: pause_intf_zero_assign
    assign pause_axis_tvalid  [i] = '0;
    assign pause_axis_tdata   [i] = '0;
    assign pause_axis_tlast   [i] = '0;
    assign pause_axis_tuser   [i] = '0;
    assign pause_axis_tkeep   [i] = '0;
  end

  for (i=0; i<(`SENSOR_TX_IF_INST*num_sen_tx_mod); i++) begin: gen_tx_apb_tieoff
    assign s_apb_s2m_sen_tx[i].pready   = '0;
    assign s_apb_s2m_sen_tx[i].prdata   = '0;
    assign s_apb_s2m_sen_tx[i].pserr    = '0;
  end

`endif

//------------------------------------------------------------------------------------------------//
// Assertions
//------------------------------------------------------------------------------------------------//

`ifdef ASSERT_ON
  // SIF AXIS Assertions
  `ifdef SENSOR_RX_IF_INST
  logic [31:0] fv_axis_sif_inp_byt_cnt       [`SENSOR_RX_IF_INST];
  logic [31:0] fv_axis_sif_inp_byt_cnt_nxt   [`SENSOR_RX_IF_INST];
  generate
  for (genvar n0 = 0; n0 < `SENSOR_RX_IF_INST; n0++) begin
  //SIF Input AXIS Assertions
    axis_checker #(
    .STBL_CHECK  (1),
    .NLST_BT_B2B (0),
    .MIN_PKTL_CHK (0),
    .MAX_PKTL_CHK (0),
    .AXI_TDATA   (SIF_RX_WIDTH[n0]),
    .AXI_TUSER   (`DATAUSER_WIDTH),
`ifdef SIMULATION
    .SIMULATION(1),
`endif
    .PKT_MIN_LENGTH  (58),
    .PKT_MAX_LENGTH  (`HOST_MTU)
    ) assert_sif_input_axis (
    .clk            (sif_rx_clk[n0]),
    .rst            (sif_rx_rst[n0]),
    .axis_tvalid    (i_sif_axis_tvalid[n0]),
    .axis_tlast     (i_sif_axis_tlast[n0]),
    .axis_tkeep     (i_sif_axis_tkeep[n0][(SIF_RX_WIDTH[n0]/8)-1:0]),
    .axis_tdata     (i_sif_axis_tdata[n0][SIF_RX_WIDTH[n0]-1:0]),
    .axis_tuser     (i_sif_axis_tuser[n0]),
    .axis_tready    (o_sif_axis_tready[n0]),
    .byte_count     (fv_axis_sif_inp_byt_cnt[n0]),
    .byte_count_nxt (fv_axis_sif_inp_byt_cnt_nxt[n0])
    );
  end
  endgenerate
  `endif

  `ifdef SENSOR_TX_IF_INST
  logic [31:0] fv_axis_sif_out_byt_cnt       [`SENSOR_TX_IF_INST];
  logic [31:0] fv_axis_sif_out_byt_cnt_nxt   [`SENSOR_TX_IF_INST];
  generate
  for (genvar n2 = 0; n2 < `SENSOR_TX_IF_INST; n2++) begin
  //SIF Output AXIS Assertions
    // Tlast must be zero if tvalid is low
    assert_sif_output_axis_tlast_tvalid: assert property ( @(posedge sif_tx_clk[n2]) disable iff (sif_tx_rst[n2])
      (!o_sif_axis_tlast[n2] || (o_sif_axis_tvalid[n2] && o_sif_axis_tlast[n2])));

    axis_checker #(
    .STBL_CHECK  (1),
    .NLST_BT_B2B (1),
    .MIN_PKTL_CHK (0),
    .MAX_PKTL_CHK (0),
    .AXI_TDATA   (SIF_TX_WIDTH[n2]),
    .AXI_TUSER   (`DATAUSER_WIDTH),
`ifdef SIMULATION
    .SIMULATION(1),
`endif
    .PKT_MIN_LENGTH  (58),
    .PKT_MAX_LENGTH  (`HOST_MTU)
    ) assert_sif_output_axis (
    .clk            (sif_tx_clk[n2]),
    .rst            (sif_tx_rst[n2]),
    .axis_tvalid    (o_sif_axis_tvalid[n2]),
    .axis_tlast     (o_sif_axis_tlast[n2]),
    .axis_tkeep     (o_sif_axis_tkeep[n2][(SIF_TX_WIDTH[n2]/8)-1:0]),
    .axis_tdata     (o_sif_axis_tdata[n2][SIF_TX_WIDTH[n2]-1:0]),
    .axis_tuser     (o_sif_axis_tuser[n2]),
    .axis_tready    (i_sif_axis_tready[n2]),
    .byte_count     (fv_axis_sif_out_byt_cnt[n2]),
    .byte_count_nxt (fv_axis_sif_out_byt_cnt_nxt[n2])
    );
  end
  endgenerate
  `endif

  // HIF AXIS Assertions
  logic [31:0] fv_axis_hif_inp_byt_cnt       [`HOST_IF_INST];
  logic [31:0] fv_axis_hif_inp_byt_cnt_nxt   [`HOST_IF_INST];
  logic [31:0] fv_axis_hif_out_byt_cnt       [`HOST_IF_INST];
  logic [31:0] fv_axis_hif_out_byt_cnt_nxt   [`HOST_IF_INST];
  generate
  for (genvar n1 = 0; n1 < `HOST_IF_INST; n1++) begin

  //HIF Input AXIS Assertions
    axis_checker #(
    .STBL_CHECK  (1),
    .NLST_BT_B2B (0),
    .MIN_PKTL_CHK (0),
    .MAX_PKTL_CHK (0),
    .AXI_TDATA   (`HOST_WIDTH),
    .AXI_TUSER   (`HOSTUSER_WIDTH),
`ifdef SIMULATION
    .SIMULATION(1),
`endif
    .PKT_MIN_LENGTH  (58),
    .PKT_MAX_LENGTH  (`HOST_MTU)
    ) assert_hif_input_axis (
    .clk            (i_hif_clk),
    .rst            (o_hif_rst),
    .axis_tvalid    (i_hif_axis_tvalid[n1]),
    .axis_tlast     (i_hif_axis_tlast[n1]),
    .axis_tkeep     (i_hif_axis_tkeep[n1]),
    .axis_tdata     (i_hif_axis_tdata[n1]),
    .axis_tuser     (i_hif_axis_tuser[n1]),
    .axis_tready    (o_hif_axis_tready[n1]),
    .byte_count     (fv_axis_hif_inp_byt_cnt[n1]),
    .byte_count_nxt (fv_axis_hif_inp_byt_cnt_nxt[n1])
    );

  //HIF Output AXIS Assertions
    // Tlast must be zero if tvalid is low
    assert_hif_output_axis_tlast_tvalid: assert property ( @(posedge i_hif_clk) disable iff (o_hif_rst)
      (!o_hif_axis_tlast[n1] || (o_hif_axis_tvalid[n1] && o_hif_axis_tlast[n1])));

    axis_checker #(
    .STBL_CHECK  (1),
    .NLST_BT_B2B (1),
    .MIN_PKTL_CHK (1),
    .MAX_PKTL_CHK (1),
    .AXI_TDATA   (`HOST_WIDTH),
    .AXI_TUSER   (`HOSTUSER_WIDTH),
`ifdef SIMULATION
    .SIMULATION(1),
`endif
    .PKT_MIN_LENGTH  (58),
    .PKT_MAX_LENGTH  (`HOST_MTU)
    ) assert_hif_output_axis (
    .clk            (i_hif_clk),
    .rst            (o_hif_rst),
    .axis_tvalid    (o_hif_axis_tvalid[n1]),
    .axis_tlast     (o_hif_axis_tlast[n1]),
    .axis_tkeep     (o_hif_axis_tkeep[n1]),
    .axis_tdata     (o_hif_axis_tdata[n1]),
    .axis_tuser     (o_hif_axis_tuser[n1]),
    .axis_tready    (i_hif_axis_tready[n1]),
    .byte_count     (fv_axis_hif_out_byt_cnt[n1]),
    .byte_count_nxt (fv_axis_hif_out_byt_cnt_nxt[n1])
    );
  end
  endgenerate

`endif

endmodule
