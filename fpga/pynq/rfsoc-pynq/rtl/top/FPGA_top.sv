// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.	
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

`include "HOLOLINK_def.svh"

module FPGA_top
  import HOLOLINK_pkg::*;
  import apb_pkg::*;
#(
  parameter BUILD_REV = 48'h0
)(
  input         RESET,  // board reset push button
  // 10GbE SFP
  input         ETH_REFCLK_P,
  input         ETH_REFCLK_N,
  input  [ 3:0] gt_serial_port_0_grx_p ,
  input  [ 3:0] gt_serial_port_0_grx_n ,
  output [ 3:0] gt_serial_port_0_gtx_p ,
  output [ 3:0] gt_serial_port_0_gtx_n,
  input  sysclk_p,
  input  sysclk_n
);


//------------------------------------------------------------------------------
// Clock and Reset
//------------------------------------------------------------------------------

  logic [`HOST_IF_INST-1:0] usr_clk;     // pcs user clock out
  logic [`HOST_IF_INST-1:0] usr_clk_rdy; // pcs user clock out ready
  logic                     usr_clk_locked;
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
  logic                     cmac_sys_rst;   // ethernet cmac system active high
  /* synthesis syn_keep=1 nomerge=""*/
  logic [31:0]              ptp_nsec;
  logic [47:0]              ptp_sec;
  logic [`HOST_IF_INST-1:0] aligned;
  logic [`HOST_IF_INST-1  :0] pll_locked;



  rst u_rst (
    .i_apb_clk       ( apb_clk        ), // apb clock
    .i_locked        ( pll_locked ), // cmac pll locked
    .i_aligned       ( aligned   ), // ethernet cmac system aligned

    .i_pb_rst        ( RESET        ), // asynchronous active high board reset
    .i_sw_rst        ( sw_sys_rst     ), // software controlled active high reset

    .o_sys_rst       ( sys_rst        ), // system active high reset
    .o_cmac_sys_rst  ( cmac_sys_rst   )  // ethernet cmac system active high
  );

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

  genvar i;

//------------------------------------------------------------------------------
// xilinx 100GbE Host Interface
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
  logic [`HOST_IF_INST-1  :0] usr_rst; // TODO: add to rst
  logic sys_clk_100, init_clk;
  logic [3:0] gt_powergoodout;
  logic gt_ref_clk_out;
  
  IBUFDS sys_clk_inst (
   .O(sys_clk_100),  // Buffer output
   .I(sysclk_p),  // Diff_p buffer input (connect directly to top-level port)
   .IB(sysclk_n) // Diff_n buffer input (connect directly to top-level port)
  );
  BUFG init_clk_inst (
   .O(init_clk), // 1-bit output: Clock output
   .I(sys_clk_100)  // 1-bit input: Clock input
  );

logic [3:0] gt_pg_d, gt_pg_dd;
logic rst_all;
logic init_rst;
logic [11:0] init_rst_cnt;

always_ff@(posedge init_clk or posedge RESET) begin
  if(RESET) begin
    init_rst <= 1'b1;
    init_rst_cnt <= 1'b0;
  end
  else if(init_rst_cnt[11] == 1'b1) begin
    init_rst <= 1'b0;
  end
  else begin
    init_rst <= 1'b1;
    init_rst_cnt <= init_rst_cnt + 1'b1;
  end
end


always_ff@(posedge init_clk or posedge init_rst) begin
  if(init_rst) begin
    gt_pg_d <= 4'h0;
    gt_pg_dd <= 4'h0;
    rst_all <= 1'b1;
  end
  else begin
    gt_pg_d <= gt_powergoodout;
    gt_pg_dd <= gt_pg_d;
    rst_all <= (gt_pg_dd == 4'hf) ? 1'b0 : 1'b1;
  end
end

      eth_100gb_top u_100gbe (
        // clock and reset
        .i_refclk_p       ( ETH_REFCLK_P              ),
        .i_refclk_n       ( ETH_REFCLK_N              ),
        .o_pll_locked     ( pll_locked                ),
        .i_cmac_rst       ( rst_all                   ),
        .init_clk         ( init_clk                  ),
        .o_ptp_clk        ( ptp_clk                   ),
        // SERDES IO
        .gt_serial_port_0_grx_p       ( gt_serial_port_0_grx_p              ),
        .gt_serial_port_0_grx_n       ( gt_serial_port_0_grx_n              ),
        .gt_serial_port_0_gtx_p       ( gt_serial_port_0_gtx_p              ),
        .gt_serial_port_0_gtx_n       ( gt_serial_port_0_gtx_n              ),
        .o_usr_clk        ( usr_clk                ),
        .o_usr_rst        ( usr_rst                ),
        .o_aligned        ( aligned                ),
        .gt_powergoodout  (gt_powergoodout   ),
        .gt_ref_clk_out   (gt_ref_clk_out),
        // APB Interface, abp clk domain
        .o_aclk           ( apb_clk                   ),
        .axis_tx_0_tvalid   ( hif_tx_axis_tvalid     ),
        .axis_tx_0_tlast    ( hif_tx_axis_tlast      ),
        .axis_tx_0_tkeep    ( hif_tx_axis_tkeep[0]      ),
        .axis_tx_0_tdata    ( hif_tx_axis_tdata[0]      ),
        .axis_tx_0_tuser    ( hif_tx_axis_tuser[0]      ),
        .axis_tx_0_tready   ( hif_tx_axis_tready     ),

        .axis_rx_0_tvalid   ( hif_rx_axis_tvalid     ),
        .axis_rx_0_tlast    ( hif_rx_axis_tlast      ),
        .axis_rx_0_tkeep    ( hif_rx_axis_tkeep[0]      ),
        .axis_rx_0_tdata    ( hif_rx_axis_tdata[0]      ),
        .axis_rx_0_tuser    ( hif_rx_axis_tuser[0]      )
      );


  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_clk;
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_axis_tvalid;
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_axis_tlast;
  logic [`DATAPATH_WIDTH-1:0]    sif_tx_axis_tdata [`SENSOR_TX_IF_INST-1:0];
  logic [`DATAKEEP_WIDTH-1:0]    sif_tx_axis_tkeep [`SENSOR_TX_IF_INST-1:0];
  logic [`DATAUSER_WIDTH-1:0]    sif_tx_axis_tuser [`SENSOR_TX_IF_INST-1:0];
  logic [`SENSOR_TX_IF_INST-1:0] sif_tx_axis_tready;

  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_axis_tvalid;
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_axis_tlast;
  logic [`DATAPATH_WIDTH-1:0]    sif_rx_axis_tdata [`SENSOR_RX_IF_INST-1:0];
  logic [`DATAKEEP_WIDTH-1:0]    sif_rx_axis_tkeep [`SENSOR_RX_IF_INST-1:0];
  logic [`DATAUSER_WIDTH-1:0]    sif_rx_axis_tuser [`SENSOR_RX_IF_INST-1:0];
  logic [`SENSOR_RX_IF_INST-1:0] sif_rx_axis_tready;



//------------------------------------------------------------------------------
// APB
//------------------------------------------------------------------------------

//Tie off unused APB bus signals. 
assign apb_pserr[7:5]         = '0;
assign apb_pserr[1:0]         = '0;
assign apb_pready[7:5]        = '1;
assign apb_pready[1:0]        = '0;
assign apb_prdata[7]          = 'h017;
assign apb_prdata[6]          = 'h016;
assign apb_prdata[5]          = 'h015;
assign apb_prdata[1]          = 'h011;
assign apb_prdata[0]          = 'h010;


//------------------------------------------------------------------------------
// Sensor IF
//------------------------------------------------------------------------------

logic                sof;
logic                eof;
logic [79:0]         ptp_ts;
logic                ptp_ts_en ;
logic [11:0]         frame_cnt;

always_ff @ (posedge usr_clk) begin
  if (hif_rst) begin
    ptp_ts <= '0;
    ptp_ts_en <= '0;
    frame_cnt <= '0;
  end
  else begin
    ptp_ts <= (sof) ? sif_tx_axis_tdata[0][79:0] : ptp_ts;
    frame_cnt <= (sof) ? sif_tx_axis_tdata[0][91:80] : frame_cnt;
    ptp_ts_en <= sof;
  end
end 
assign sof = (sif_tx_axis_tvalid[0]);
assign eof = (sif_tx_axis_tlast[0]);

logic [47:0] ptp_sec_sync_usr;
logic [31:0] ptp_nsec_sync_usr;
logic        ptp_sync_usr_valid;

streaming_cdc #(
  .DATA_WIDTH ( 80                                        ),
  .SRC_FREQ   ( `PTP_CLK_FREQ                             ),
  .DST_FREQ   ( `HIF_CLK_FREQ                             )
) u_ptp_hif_cdc (
  .i_src_clk  ( ptp_clk                                 ),
  .i_dst_clk  ( usr_clk                                 ),
  .i_src_rst  ( ptp_rst                                 ),
  .i_dst_rst  ( hif_rst                                 ),
  .i_src_data ( {ptp_sec, ptp_nsec}                   ),
  .o_dst_data ( {ptp_sec_sync_usr, ptp_nsec_sync_usr} ),
  .o_dst_valid ( ptp_sync_usr_valid                       )
);

logic [31:0] cnt;

always_ff @ (posedge usr_clk) begin
  if (hif_rst) begin
    cnt <= '0;
  end
  else begin
    cnt <= cnt + 1'b1;
  end
end


localparam ILA_DATA_WIDTH = 256;
logic [ILA_DATA_WIDTH-1:0] ila_wr_data;
assign ila_wr_data[63:0] = ptp_ts[63:0];
assign ila_wr_data[127:64] = {ptp_sec_sync_usr[31:0], ptp_nsec_sync_usr[31:0]};
assign ila_wr_data[139:128] = frame_cnt;
assign ila_wr_data[140] = sof;
assign ila_wr_data[141] = eof;
assign ila_wr_data[223:142] = 'h123456789ABCDEF;
assign ila_wr_data[255:224] = cnt;

apb_m2s ila_apb_m2s;
apb_s2m ila_apb_s2m;

s_apb_ila #(
  .DEPTH            ( 16384                          ),
  .W_DATA           ( ILA_DATA_WIDTH                 )
) u_apb_ila (
  .i_aclk           ( apb_clk                        ),
  .i_arst           ( apb_rst                        ),
  .i_apb_m2s        ( ila_apb_m2s                    ),
  .o_apb_s2m        ( ila_apb_s2m                    ),
  .i_pclk           ( usr_clk                        ),
  .i_prst           ( hif_rst                        ),
  .i_trigger        ( '1                             ),
  .i_enable         ( '1                             ),
  .i_wr_data        ( ila_wr_data                    ),
  .i_wr_en          ( ptp_ts_en                      ),
  .o_ctrl_reg       (                                )
);



assign ila_apb_m2s.psel   = apb_psel     [2];
assign ila_apb_m2s.penable= apb_penable     ;
assign ila_apb_m2s.paddr  = apb_paddr       ;
assign ila_apb_m2s.pwdata = apb_pwdata      ;
assign ila_apb_m2s.pwrite = apb_pwrite      ;
assign apb_pready [2]        = ila_apb_s2m.pready; 
assign apb_prdata [2]        = ila_apb_s2m.prdata; 
assign apb_pserr  [2]        = ila_apb_s2m.pserr;



//------------------------------------------------------------------------------
// SIF Latch
//------------------------------------------------------------------------------

localparam SIF_ILA_DATA_WIDTH = 512 + 2 + 7 + 64;

localparam SIF_ILA_WSTRB = $clog2(`DATAPATH_WIDTH/8);

logic [SIF_ILA_DATA_WIDTH-1:0] sif_ila_wr_data;
logic [SIF_ILA_WSTRB-1:0] sif_ila_wr_tcnt;


assign sif_ila_wr_data[511:0]   = sif_tx_axis_tdata[0];
assign sif_ila_wr_data[512]     = sif_tx_axis_tvalid[0];
assign sif_ila_wr_data[513]     = sif_tx_axis_tlast[0];
assign sif_ila_wr_data[520:514] = sif_ila_wr_tcnt;
assign sif_ila_wr_data[584:521] = {ptp_sec_sync_usr[31:0], ptp_nsec_sync_usr[31:0]};

integer j;
always_comb begin
  sif_ila_wr_tcnt = '0;
  for (j=0;j<(`DATAPATH_WIDTH/8);j=j+1) begin
    if (sif_tx_axis_tkeep[0][j]) begin
      sif_ila_wr_tcnt = j;
    end
  end
end

apb_m2s sif_ila_apb_m2s;
apb_s2m sif_ila_apb_s2m;

s_apb_ila #(
  .DEPTH            ( 8192                           ),
  .W_DATA           ( SIF_ILA_DATA_WIDTH             )
) u_apb_sif_ila (
  .i_aclk           ( apb_clk                        ),
  .i_arst           ( apb_rst                        ),
  .i_apb_m2s        ( sif_ila_apb_m2s                ),
  .o_apb_s2m        ( sif_ila_apb_s2m                ),
  .i_pclk           ( usr_clk                        ),
  .i_prst           ( hif_rst                        ),
  .i_trigger        ( '1                             ),
  .i_enable         ( '1                             ),
  .i_wr_data        ( sif_ila_wr_data                ),
  .i_wr_en          ( sif_tx_axis_tvalid[0]          ),
  .o_ctrl_reg       (                                )
);



assign sif_ila_apb_m2s.psel   = apb_psel     [3];
assign sif_ila_apb_m2s.penable= apb_penable     ;
assign sif_ila_apb_m2s.paddr  = apb_paddr       ;
assign sif_ila_apb_m2s.pwdata = apb_pwdata      ;
assign sif_ila_apb_m2s.pwrite = apb_pwrite      ;
assign apb_pready [3]        = sif_ila_apb_s2m.pready; 
assign apb_prdata [3]        = sif_ila_apb_s2m.prdata; 
assign apb_pserr  [3]        = sif_ila_apb_s2m.pserr;


//------------------------------------------------------------------------------
// RAM Player
//------------------------------------------------------------------------------

apb_m2s ram_apb_m2s;
apb_s2m ram_apb_s2m;

ram_player #(
  .W_DATA           ( 512                                   )
) u_ram_player (
  .i_apb_clk        ( apb_clk                               ),
  .i_apb_rst        ( apb_rst                               ),
  .i_sif_clk        ( usr_clk                               ),
  .i_sif_rst        ( hif_rst                               ),
  .i_apb_m2s        ( ram_apb_m2s                           ),
  .o_apb_s2m        ( ram_apb_s2m                           ),
  .o_ram_axis_mux   (                                       ),
  .i_ptp            ( {ptp_sec_sync_usr, ptp_nsec_sync_usr} ),
  .o_axis_tvalid    ( sif_rx_axis_tvalid [0]                ),
  .o_axis_tdata     ( sif_rx_axis_tdata  [0]                ),
  .o_axis_tkeep     ( sif_rx_axis_tkeep  [0]                ),
  .o_axis_tuser     ( sif_rx_axis_tuser  [0]                ),
  .o_axis_tlast     ( sif_rx_axis_tlast  [0]                ),
  .i_axis_tready    ( sif_rx_axis_tready [0]                )
);

assign ram_apb_m2s.psel    = apb_psel     [4];
assign ram_apb_m2s.penable = apb_penable;
assign ram_apb_m2s.paddr   = apb_paddr;
assign ram_apb_m2s.pwdata  = apb_pwdata;
assign ram_apb_m2s.pwrite  = apb_pwrite;
assign apb_pready [4]      = ram_apb_s2m.pready; 
assign apb_prdata [4]      = ram_apb_s2m.prdata; 
assign apb_pserr  [4]      = ram_apb_s2m.pserr;

//------------------------------------------------------------------------------
// HOLOLINK Top Instantiation
//------------------------------------------------------------------------------
logic [47:0] mac_addr [`HOST_IF_INST];
    assign mac_addr[0] = 48'hCAFEC0FFEE00;
  HOLOLINK_top #(
    .BUILD_REV         ( BUILD_REV          )
  ) u_hololink_top (
    .i_sys_rst         ( sys_rst            ),
    .i_mac_addr        ( mac_addr           ),
    .i_board_sn        ( '0                 ),
    .i_enum_vld        ( '1                 ),    
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
    .i_sif_rx_clk      ( usr_clk            ),
    .o_sif_rx_rst      ( sif_rx_rst         ),
    .i_sif_tx_clk      ( usr_clk            ),
    .o_sif_tx_rst      ( sif_tx_rst         ),
    // Sensor Rx Streaming Interface
    .i_sif_axis_tvalid ( sif_rx_axis_tvalid ),
    .i_sif_axis_tlast  ( sif_rx_axis_tlast  ),
    .i_sif_axis_tdata  ( sif_rx_axis_tdata  ),
    .i_sif_axis_tkeep  ( sif_rx_axis_tkeep  ),
    .i_sif_axis_tuser  ( sif_rx_axis_tuser  ),
    .o_sif_axis_tready ( sif_rx_axis_tready ),
    // Sensor Tx Streaming Interface
    .o_sif_axis_tvalid ( sif_tx_axis_tvalid ),
    .o_sif_axis_tlast  ( sif_tx_axis_tlast  ),
    .o_sif_axis_tdata  ( sif_tx_axis_tdata  ),
    .o_sif_axis_tkeep  ( sif_tx_axis_tkeep  ),
    .o_sif_axis_tuser  ( sif_tx_axis_tuser  ),
    .i_sif_axis_tready ( '1 ),
    // Sensor Event
    .i_sif_event       ( '0         ),
  //------------------------------------------------------------------------------
  // Host IF
  //------------------------------------------------------------------------------
  // Host Interface Clock and Reset
    .i_hif_clk         ( usr_clk            ),
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
    .o_spi_csn         (             ),
    .o_spi_sck         (             ),
    .i_spi_sdio        ( '{1'b0}         ),
    .o_spi_sdio        (          ),
    .o_spi_oen         (             ),
    // I2C Interface
    .i_i2c_scl         ( '{1'b1}            ),
    .i_i2c_sda         ( '{1'b1}            ),
    .o_i2c_scl_en      (          ),
    .o_i2c_sda_en      (          ),
    // GPIO
    .o_gpio            ( ),
    .i_gpio            ( '0   ),
    .o_gpio_dir        (),
  //------------------------------------------------------------------------------
  // sensor reset
  //------------------------------------------------------------------------------
    .o_sw_sys_rst      ( sw_sys_rst         ),
    .o_sw_sen_rst      (            ),
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
