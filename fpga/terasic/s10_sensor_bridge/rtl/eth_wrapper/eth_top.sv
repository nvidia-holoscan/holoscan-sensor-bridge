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

module eth_top
#(
    parameter                               DWIDTH = 512,
    parameter                               RX_ERR_WIDTH = 6,
    parameter                               TX_ERR_WIDTH = 1,
    parameter                               EMPTY_WIDTH = 6,
    parameter                               KEEP_WIDTH = DWIDTH/8
) (             
              
    input   logic                           hif_clk             ,
    input   logic                           fpga_clk_100        ,
    input   logic                           eth_reset_in        , 
`ifdef ETH_25Gx4
    output  logic [3:0]                     eth_rdy             , 
    
`else
    output  logic                           eth_rdy             , 
`endif
    input   logic                           heartbeat           ,
    
`ifdef ETH_25Gx4
    output  logic [3:0]                     eth_clk             ,
    output  logic [3:0]                     eth_rst_n           ,
`else
    output  logic                           eth_clk             ,
    output  logic                           eth_rst_n           ,
`endif
                                  
`ifdef ETH_25Gx4                                  
    input   logic  [3:0]                    i_clk_ref           ,
`else
    input   logic                           i_clk_ref           ,
    
`endif
    input   logic  [3:0]                    i_rx_serial         ,
    output  logic  [3:0]                    o_tx_serial         ,
          
    //RX      
`ifdef ETH_25Gx4
    output  logic  [3:0]                    o_eth_axis_rx_tvalid,
    output  logic  [3:0]                    o_eth_axis_rx_tlast ,
    output  logic  [2*DWIDTH-1:0]           o_eth_axis_rx_tdata[4] ,
    output  logic                           o_eth_axis_rx_tuser[4] ,
    output  logic  [2*KEEP_WIDTH-1:0]       o_eth_axis_rx_tkeep[4] , 
`else
    output  logic                           o_eth_axis_rx_tvalid,
    output  logic                           o_eth_axis_rx_tlast ,
    output  logic  [DWIDTH-1:0]             o_eth_axis_rx_tdata ,
    output  logic                           o_eth_axis_rx_tuser ,
    output  logic  [KEEP_WIDTH-1:0]         o_eth_axis_rx_tkeep ,
       
`endif
          
    //TX      
`ifdef ETH_25Gx4
    input   logic  [3:0]                    i_eth_axis_tx_tvalid,
    input   logic  [3:0]                    i_eth_axis_tx_tlast ,
    output  logic  [3:0]                    o_eth_axis_tx_tready,
    input   logic  [2*DWIDTH-1:0]           i_eth_axis_tx_tdata[4] ,
    input   logic                           i_eth_axis_tx_tuser[4] ,
    input   logic  [2*KEEP_WIDTH-1:0]       i_eth_axis_tx_tkeep[4] ,
`else
    input   logic                           i_eth_axis_tx_tvalid,
    input   logic                           i_eth_axis_tx_tlast ,
    output  logic                           o_eth_axis_tx_tready,
    input   logic  [DWIDTH-1:0]             i_eth_axis_tx_tdata ,
    input   logic                           i_eth_axis_tx_tuser ,
    input   logic  [KEEP_WIDTH-1:0]         i_eth_axis_tx_tkeep ,
    
`endif
    
    //APB Interface
    input   logic                           i_eth_apb_psel      ,
    input   logic                           i_eth_apb_penable   ,
    input   logic  [31:0]                   i_eth_apb_paddr     ,  
    input   logic  [31:0]                   i_eth_apb_pwdata    , 
    input   logic                           i_eth_apb_pwrite    , 
    output  logic                           o_eth_apb_pready    , 
    output  logic  [31:0]                   o_eth_apb_prdata    , 
    output  logic                           o_eth_apb_pserr     
    
);

`ifdef ETH_25Gx4
    eth_25gb qsfp_inst  (
      .hif_clk                    ( hif_clk             ),
      .fpga_clk_100               ( fpga_clk_100        ),
      .eth_reset_in               ( eth_reset_in        ),
      .eth_rdy                    ( eth_rdy             ),
      .heartbeat                  ( heartbeat           ),
                                    
      .eth_clk                    ( eth_clk             ),
      .eth_rst_n                  ( eth_rst_n           ),
      // 25G IO                     
      .i_clk_ref                  ( i_clk_ref           ),
      .i_rx_serial                ( i_rx_serial         ),
      .o_tx_serial                ( o_tx_serial         ),
      //RX                          
      .o_eth_axis_rx_tvalid       ( o_eth_axis_rx_tvalid),
      .o_eth_axis_rx_tdata        ( o_eth_axis_rx_tdata ),
      .o_eth_axis_rx_tlast        ( o_eth_axis_rx_tlast ),
      .o_eth_axis_rx_tuser        ( o_eth_axis_rx_tuser ),
      .o_eth_axis_rx_tkeep        ( o_eth_axis_rx_tkeep ),
      //TX                          
      .i_eth_axis_tx_tvalid       ( i_eth_axis_tx_tvalid),
      .i_eth_axis_tx_tdata        ( i_eth_axis_tx_tdata ),
      .i_eth_axis_tx_tlast        ( i_eth_axis_tx_tlast ),
      .i_eth_axis_tx_tuser        ( i_eth_axis_tx_tuser ),
      .i_eth_axis_tx_tkeep        ( i_eth_axis_tx_tkeep ),
      .o_eth_axis_tx_tready       ( o_eth_axis_tx_tready),
      //APB                              
      .i_eth_apb_psel             ( i_eth_apb_psel      ),
      .i_eth_apb_penable          ( i_eth_apb_penable   ),
      .i_eth_apb_paddr            ( i_eth_apb_paddr     ),
      .i_eth_apb_pwdata           ( i_eth_apb_pwdata    ),
      .i_eth_apb_pwrite           ( i_eth_apb_pwrite    ),
      .o_eth_apb_pready           ( o_eth_apb_pready    ),
      .o_eth_apb_prdata           ( o_eth_apb_prdata    ),
      .o_eth_apb_pserr            ( o_eth_apb_pserr     )
    );
`else
    eth_100gb qsfp_inst  (
      .hif_clk                    ( hif_clk             ),
      .fpga_clk_100               ( fpga_clk_100        ),
      .eth_reset_in               ( eth_reset_in        ),
      .eth_rdy                    ( eth_rdy             ),
      .heartbeat                  ( heartbeat           ),
                                    
      .eth_clk                    ( eth_clk             ),
      .eth_rst_n                  ( eth_rst_n           ),
      // 100G IO                     
      .i_clk_ref                  ( i_clk_ref           ),
      .i_rx_serial                ( i_rx_serial         ),
      .o_tx_serial                ( o_tx_serial         ),
      //RX                          
      .o_eth_axis_rx_tvalid       ( o_eth_axis_rx_tvalid),
      .o_eth_axis_rx_tdata        ( o_eth_axis_rx_tdata ),
      .o_eth_axis_rx_tlast        ( o_eth_axis_rx_tlast ),
      .o_eth_axis_rx_tuser        ( o_eth_axis_rx_tuser ),
      .o_eth_axis_rx_tkeep        ( o_eth_axis_rx_tkeep ),
      //TX                          
      .i_eth_axis_tx_tvalid       ( i_eth_axis_tx_tvalid),
      .i_eth_axis_tx_tdata        ( i_eth_axis_tx_tdata ),
      .i_eth_axis_tx_tlast        ( i_eth_axis_tx_tlast ),
      .i_eth_axis_tx_tuser        ( i_eth_axis_tx_tuser ),
      .i_eth_axis_tx_tkeep        ( i_eth_axis_tx_tkeep ),
      .o_eth_axis_tx_tready       ( o_eth_axis_tx_tready),
      //APB                              
      .i_eth_apb_psel             ( i_eth_apb_psel      ),
      .i_eth_apb_penable          ( i_eth_apb_penable   ),
      .i_eth_apb_paddr            ( i_eth_apb_paddr     ),
      .i_eth_apb_pwdata           ( i_eth_apb_pwdata    ),
      .i_eth_apb_pwrite           ( i_eth_apb_pwrite    ),
      .o_eth_apb_pready           ( o_eth_apb_pready    ),
      .o_eth_apb_prdata           ( o_eth_apb_prdata    ),
      .o_eth_apb_pserr            ( o_eth_apb_pserr     )
    );
    
`endif

  

endmodule
