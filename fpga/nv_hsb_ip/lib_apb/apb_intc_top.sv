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

module apb_intc_top
  import HOLOLINK_pkg::*;
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter  N_EXT_APB        = 2,
  parameter  N_PER            = 2,
  parameter  N_MPORT          = 4,
  parameter  N_SENSOR_RX      = 2,
  parameter  N_SENSOR_RX_MOD  = 1,
  parameter  N_SENSOR_TX      = 0,
  localparam N_MIN_SENSOR_TX  = (N_SENSOR_TX==0) ? 1 : N_SENSOR_TX,
  parameter  N_SENSOR_TX_MOD  = 1,
  parameter  N_HOST           = 2,
  parameter  N_HOST_MOD       = 4
)(
  input                  i_aclk,
  input                  i_arst,
  // level 0 switch, ECB / sys_init / EEPROM FSM
  input  apb_m2s         i_apb_m2s        [N_MPORT],
  output apb_s2m         o_apb_s2m        [N_MPORT],
  //Device
  input  apb_s2m         i_apb_s2m_dev    [4],
  output apb_m2s         o_apb_m2s_dev    [4],
  //Sensor RX
  input  apb_s2m         i_apb_s2m_sen_rx [N_SENSOR_RX * N_SENSOR_RX_MOD],
  output apb_m2s         o_apb_m2s_sen_rx [N_SENSOR_RX * N_SENSOR_RX_MOD],
  //Sensor TX
  input  apb_s2m         i_apb_s2m_sen_tx [N_MIN_SENSOR_TX * N_SENSOR_TX_MOD],
  output apb_m2s         o_apb_m2s_sen_tx [N_MIN_SENSOR_TX * N_SENSOR_TX_MOD],
  //Host
  input  apb_s2m         i_apb_s2m_host   [N_HOST * N_HOST_MOD],
  output apb_m2s         o_apb_m2s_host   [N_HOST * N_HOST_MOD],
  //Host RAMs
  input  apb_s2m         i_apb_s2m_ram    [2],
  output apb_m2s         o_apb_m2s_ram    [2],
  // Connect to HOLOLINK Peripheral Modules
  input  apb_s2m         i_apb_s2m_per    [N_PER],
  output apb_m2s         o_apb_m2s_per    [N_PER],
  // Connect to Bridge Module
  input  apb_s2m         i_apb_s2m_bridge,
  output apb_m2s         o_apb_m2s_bridge,
  // Connect to User Modules, not using defined type struct
  output [N_EXT_APB-1:0] o_apb_psel,
  output                 o_apb_penable,
  output [31        :0]  o_apb_paddr,
  output [31        :0]  o_apb_pwdata,
  output                 o_apb_pwrite,
  input  [N_EXT_APB-1:0] i_apb_pready,
  input  [31        :0]  i_apb_prdata     [N_EXT_APB-1:0],
  input  [N_EXT_APB-1:0] i_apb_pserr
);

localparam SENSOR_TXRX = (N_SENSOR_TX>0) ? 2 : 1;
genvar j,k,l,m,n;

apb_m2s s_apb_m2s [N_EXT_APB+1];
apb_s2m s_apb_s2m [N_EXT_APB+1];

//------------------------------------------------------------------------------------------------//
// Watchdog Timeout
//------------------------------------------------------------------------------------------------//
logic m2s_penable;
always_comb begin
  m2s_penable = 0;
  for (int i=0;i<N_EXT_APB+1;i++) begin
    m2s_penable = m2s_penable | s_apb_m2s[i].penable;
  end
end

logic                apb_timeout;
logic [8:0]          err_wd_cnt;
assign apb_timeout = err_wd_cnt[8];

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    err_wd_cnt <= '0;
  end
  else begin
    // Error timer
    err_wd_cnt <= '0;
    if (m2s_penable && !apb_timeout) begin
      err_wd_cnt <= err_wd_cnt + 1'b1;
    end
  end
end

//------------------------------------------------------------------------------------------------//
// Switch between Hololink Reg and User Reg
//------------------------------------------------------------------------------------------------//

// Level 1 switch [31:28]
// Input : ECB rdwr     : 0x0000_0000 ~ 0xFFFF_FFFF
// Output: Hololink Reg : 0x0000_0000 - 0x0FFF_FFFF
// Output: User Reg     : 0x1000_0000 - 0xFFFF_FFFF

// Instruction Decoder, Sys Init, EEPROM fsm
apb_switch #(
  .N_MPORT             ( N_MPORT             ),
  .N_SPORT             ( N_EXT_APB+1         ),
  .W_OFSET             ( ADDR_SW_USER        ),
  .W_SW                ( W_ADDR-ADDR_SW_USER ),
  .MERGE_COMPLETER_SIG ( 0                   )
) u_apb_switch     (
  .i_apb_clk           ( i_aclk              ),
  .i_apb_reset         ( i_arst              ),
  // Connect to Decoder / EEPROM FSM / Sys Init
  .i_apb_m2s           ( i_apb_m2s           ),
  .o_apb_s2m           ( o_apb_s2m           ),
  // Connect to user reg switch / hololink reg switch
  .o_apb_m2s           ( s_apb_m2s           ),
  .i_apb_s2m           ( s_apb_s2m           ),
  // APB Timeout
  .i_apb_timeout       ( apb_timeout         )
);

//------------------------------------------------------------------------------------------------//
// Hololink IP Reg
//------------------------------------------------------------------------------------------------//
// Level 2 switch [27:24]
// IP Internal
// Input  : Level 1 switch   : 0x0000_0000 - 0x0FFF_FFFF
// Output : Global module    : 0x0000_0000 - 0x00FF_FFFF
// Output : Sensor module    : 0x0100_0000 - 0x01FF_FFFF
// Output : Host module      : 0x0200_0000 - 0x02FF_FFFF
// Output : Peripheral module: 0x0300_0000 - 0x03FF_FFFF

apb_m2s s_apb_m2s_int [7];
apb_s2m s_apb_s2m_int [7];

apb_m2s s_apb_m2s_ff [1];
apb_s2m s_apb_s2m_ff [1];

apb_ff u_apb_ff_lvl2 (
  .i_aclk          ( i_aclk           ),
  .i_arst          ( i_arst           ),
  .i_apb_m2s       ( s_apb_m2s [0]    ),
  .o_apb_s2m       ( s_apb_s2m [0]    ),
  .o_apb_m2s       ( s_apb_m2s_ff [0] ),
  .i_apb_s2m       ( s_apb_s2m_ff [0] )
);

apb_switch #(
  .N_MPORT         ( 1                ),
  .N_SPORT         ( 7                ), //0=Global 1=Sensor 2=Host 3=Peri ... 6=Bridge
  .W_OFSET         ( ADDR_SW_HOLOLINK ),
  .W_SW            ( ADDR_SW_USER-ADDR_SW_HOLOLINK)
) u_apb_switch_int (
  .i_apb_clk       ( i_aclk             ),
  .i_apb_reset     ( i_arst             ),
  .i_apb_m2s       ( s_apb_m2s_ff [0:0] ),
  .o_apb_s2m       ( s_apb_s2m_ff [0:0] ),
  .o_apb_m2s       ( s_apb_m2s_int      ),
  .i_apb_s2m       ( s_apb_s2m_int      ),
  // APB Timeout
  .i_apb_timeout   ( apb_timeout        )
);

//------------------------------------------------------------------------------------------------//
//SPI, I2C, UART Module
//------------------------------------------------------------------------------------------------//

apb_m2s s_apb_m2s_ls_peri [3];
apb_s2m s_apb_s2m_ls_peri [3];


apb_switch #(
  .N_MPORT         ( 1                ),
  .N_SPORT         ( 3                ), //0=SPI 1=I2C 2=UART
  .W_OFSET         ( ADDR_SW_PERI     ),
  .W_SW            ( ADDR_SW_HOLOLINK-ADDR_SW_PERI)
) u_apb_switch_peri (
  .i_apb_clk       ( i_aclk             ),
  .i_apb_reset     ( i_arst             ),
  .i_apb_m2s       ( s_apb_m2s_int[3:3] ),
  .o_apb_s2m       ( s_apb_s2m_int[3:3] ),
  .o_apb_m2s       ( s_apb_m2s_ls_peri  ),
  .i_apb_s2m       ( s_apb_s2m_ls_peri  ),
  // APB Timeout
  .i_apb_timeout   ( apb_timeout        )
);

assign o_apb_m2s_per[apb_spi_idx] = s_apb_m2s_ls_peri[0];
assign s_apb_s2m_ls_peri[0]       = i_apb_s2m_per[apb_spi_idx];

assign o_apb_m2s_per[apb_i2c_idx] = s_apb_m2s_ls_peri[1];
assign s_apb_s2m_ls_peri[1]       = i_apb_s2m_per[apb_i2c_idx];

assign o_apb_m2s_per[apb_uart_idx] = s_apb_m2s_ls_peri[2];
assign s_apb_s2m_ls_peri[2]        = i_apb_s2m_per[apb_uart_idx];

//------------------------------------------------------------------------------------------------//
// Bridge
//------------------------------------------------------------------------------------------------//

assign s_apb_s2m_int[4]           = '{default:0};
assign s_apb_s2m_int[5]           = '{default:0};

assign o_apb_m2s_bridge           = s_apb_m2s_int[6];
assign s_apb_s2m_int[6]           = i_apb_s2m_bridge;

//------------------------------------------------------------------------------------------------//
// ROCE
//------------------------------------------------------------------------------------------------//
apb_m2s s_apb_m2s_roce [3];
apb_s2m s_apb_s2m_roce [3];

apb_switch #(
  .N_MPORT         ( 1                             ),
  .N_SPORT         ( 3                             ), //0=Global 1=ROCE
  .W_OFSET         ( ADDR_SW_ROCE                  ),
  .W_SW            ( ADDR_SW_HOLOLINK-ADDR_SW_ROCE )
) u_apb_switch_roce (
  .i_apb_clk       ( i_aclk              ),
  .i_apb_reset     ( i_arst              ),
  .i_apb_m2s       ( s_apb_m2s_int [0:0] ),
  .o_apb_s2m       ( s_apb_s2m_int [0:0] ),
  .o_apb_m2s       ( s_apb_m2s_roce      ),
  .i_apb_s2m       ( s_apb_s2m_roce      ),
  .i_apb_timeout   ( apb_timeout         )
);

assign s_apb_s2m_roce[1] = i_apb_s2m_ram[0];
assign o_apb_m2s_ram[0]    = s_apb_m2s_roce[1];

assign s_apb_s2m_roce[2] = i_apb_s2m_ram[1];
assign o_apb_m2s_ram[1]     = s_apb_m2s_roce[2];

//------------------------------------------------------------------------------------------------//
// Global/PTP
//------------------------------------------------------------------------------------------------//

apb_switch #(
  .N_MPORT         ( 1               ),
  .N_SPORT         ( 4               ), //0=Global 1=PTP, 2=Event, 3=ROCE
  .W_OFSET         ( w_ofst          ),
  .W_SW            ( ADDR_SW_ROCE-w_ofst)
) u_apb_switch_glb (
  .i_apb_clk       ( i_aclk              ),
  .i_apb_reset     ( i_arst              ),
  .i_apb_m2s       ( s_apb_m2s_roce[0:0] ),
  .o_apb_s2m       ( s_apb_s2m_roce[0:0] ),
  .o_apb_m2s       ( o_apb_m2s_dev       ),
  .i_apb_s2m       ( i_apb_s2m_dev       ),
  .i_apb_timeout   ( apb_timeout         )
);

//------------------------------------------------------------------------------------------------//
// Sensor
//------------------------------------------------------------------------------------------------//
// Level 3 switch [23:20]
// Sensor RX/TX
// Input  : Level 2 switch   : 0x0100_0000 - 0x01FF_FFFF
// Output : Sensor RX module : 0x0100_0000 - 0x011F_FFFF
// Output : Sensor TX module : 0x0120_0000 - 0x013F_FFFF

apb_m2s s_apb_m2s_sen_txrx [SENSOR_TXRX];
apb_s2m s_apb_s2m_sen_txrx [SENSOR_TXRX];

// Sensor TX/RX Interconnect
apb_switch #(
  .N_MPORT         ( 1                          ),
  .N_SPORT         ( SENSOR_TXRX                ),
  .W_OFSET         ( ADDR_SW_IF                 ),
  .W_SW            ( ADDR_SW_HOLOLINK-ADDR_SW_IF)
) u_apb_switch_sensor_txrx (
  .i_apb_clk       ( i_aclk              ),
  .i_apb_reset     ( i_arst              ),
  .i_apb_m2s       ( s_apb_m2s_int [1:1] ),
  .o_apb_s2m       ( s_apb_s2m_int [1:1] ),
  .o_apb_m2s       ( s_apb_m2s_sen_txrx  ),
  .i_apb_s2m       ( s_apb_s2m_sen_txrx  ),
  .i_apb_timeout   ( apb_timeout         )
);

// Level 4 switch [23:20]
// Sensor RX CH
// Input  : Level 3 switch   : 0x0100_0000 - 0x011F_FFFF
// Output : Sensor RX CH0    : 0x0100_0000 - 0x0100_FFFF
// Output : Sensor RX CH1    : 0x0101_0000 - 0x0101_FFFF

apb_m2s s_apb_m2s_sen_rx_ch [N_SENSOR_RX];
apb_s2m s_apb_s2m_sen_rx_ch [N_SENSOR_RX];

// Sensor Channel Interconnect
apb_switch #(
  .N_MPORT         ( 1                     ),
  .N_SPORT         ( N_SENSOR_RX           ),
  .W_OFSET         ( ADDR_SW_CH            ),
  .W_SW            ( ADDR_SW_IF-ADDR_SW_CH )
) u_apb_switch_sensor_rx_ch (
  .i_apb_clk       ( i_aclk                   ),
  .i_apb_reset     ( i_arst                   ),
  .i_apb_m2s       ( s_apb_m2s_sen_txrx [0:0] ),
  .o_apb_s2m       ( s_apb_s2m_sen_txrx [0:0] ),
  .o_apb_m2s       ( s_apb_m2s_sen_rx_ch      ),
  .i_apb_s2m       ( s_apb_s2m_sen_rx_ch      ),
  .i_apb_timeout   ( apb_timeout              )
);

// Level 5 switch [23:20]
// Sensor RX MOD
// Input  : Level 4 switch : 0x0100_0000 - 0x0100_FFFF
// Output : Sensor RX MOD0 : 0x0100_0000 - 0x0100_00FF

generate
  for (k=0; k<N_SENSOR_RX; k++) begin
    // Sensor MOD Interconnect
    apb_switch #(
      .N_MPORT         ( 1                                                            ),
      .N_SPORT         ( N_SENSOR_RX_MOD                                              ),
      .W_OFSET         ( w_ofst                                                       ),
      .W_SW            ( ADDR_SW_CH-w_ofst                                            )
    ) u_apb_switch_sensor_rx (
      .i_apb_clk       ( i_aclk                                                       ),
      .i_apb_reset     ( i_arst                                                       ),
      .i_apb_m2s       ( s_apb_m2s_sen_rx_ch [k:k]                                    ),
      .o_apb_s2m       ( s_apb_s2m_sen_rx_ch [k:k]                                    ),
      .o_apb_m2s       ( o_apb_m2s_sen_rx[k*N_SENSOR_RX_MOD:((k+1)*N_SENSOR_RX_MOD)-1]),
      .i_apb_s2m       ( i_apb_s2m_sen_rx[k*N_SENSOR_RX_MOD:((k+1)*N_SENSOR_RX_MOD)-1]),
      .i_apb_timeout   ( apb_timeout                                                  )
    );
  end
endgenerate


if (N_SENSOR_TX>0) begin
  apb_m2s s_apb_m2s_sen_tx_ch [N_SENSOR_TX];
  apb_s2m s_apb_s2m_sen_tx_ch [N_SENSOR_TX];

  // Sensor TX Channel Interconnect
  apb_switch #(
    .N_MPORT         ( 1                     ),
    .N_SPORT         ( N_SENSOR_TX           ),
    .W_OFSET         ( ADDR_SW_CH            ),
    .W_SW            ( ADDR_SW_IF-ADDR_SW_CH )
  ) u_apb_switch_sensor_tx_ch (
    .i_apb_clk       ( i_aclk                   ),
    .i_apb_reset     ( i_arst                   ),
    .i_apb_m2s       ( s_apb_m2s_sen_txrx [1:1] ),
    .o_apb_s2m       ( s_apb_s2m_sen_txrx [1:1] ),
    .o_apb_m2s       ( s_apb_m2s_sen_tx_ch      ),
    .i_apb_s2m       ( s_apb_s2m_sen_tx_ch      ),
    .i_apb_timeout   ( apb_timeout              )
  );

  for (m=0; m<N_SENSOR_TX; m++) begin
    // Sensor MOD Interconnect
    apb_switch #(
      .N_MPORT         ( 1                                                            ),
      .N_SPORT         ( N_SENSOR_TX_MOD                                              ),
      .W_OFSET         ( w_ofst                                                       ),
      .W_SW            ( ADDR_SW_CH-w_ofst                                            )
    ) u_apb_switch_sensor_tx (
      .i_apb_clk       ( i_aclk                                                       ),
      .i_apb_reset     ( i_arst                                                       ),
      .i_apb_m2s       ( s_apb_m2s_sen_tx_ch [m:m]                                    ),
      .o_apb_s2m       ( s_apb_s2m_sen_tx_ch [m:m]                                    ),
      .o_apb_m2s       ( o_apb_m2s_sen_tx[m*N_SENSOR_TX_MOD:((m+1)*N_SENSOR_TX_MOD)-1]),
      .i_apb_s2m       ( i_apb_s2m_sen_tx[m*N_SENSOR_TX_MOD:((m+1)*N_SENSOR_TX_MOD)-1]),
      .i_apb_timeout   ( apb_timeout                                                  )
    );
  end
end
else begin
  assign o_apb_m2s_sen_tx = '{default:0};
end
//------------------------------------------------------------------------------------------------//
// Host
//------------------------------------------------------------------------------------------------//
// Level 3 switch [23:16]
// IP Internal
// Input  : Level 2 switch  : 0x0200_0000 - 0x02FF_FFFF
// Output : Host CH0 module : 0x0200_0000 - 0x0200_FFFF
// Output : Host CH1 module : 0x0201_0000 - 0x0201_FFFF
// ....
// Output : Host CH# module

apb_m2s s_apb_m2s_host_ch [N_HOST];
apb_s2m s_apb_s2m_host_ch [N_HOST];

// Host Register Interconnect
apb_switch #(
  .N_MPORT         ( 1                           ),
  .N_SPORT         ( N_HOST                      ),
  .W_OFSET         ( ADDR_SW_CH                  ),
  .W_SW            ( ADDR_SW_HOLOLINK-ADDR_SW_CH )
) u_apb_switch_host_ch (
  .i_apb_clk       ( i_aclk              ),
  .i_apb_reset     ( i_arst              ),
  .i_apb_m2s       ( s_apb_m2s_int [2:2] ),
  .o_apb_s2m       ( s_apb_s2m_int [2:2] ),
  .o_apb_m2s       ( s_apb_m2s_host_ch   ),
  .i_apb_s2m       ( s_apb_s2m_host_ch   ),
  .i_apb_timeout   ( apb_timeout         )
);

// Level 5 switch
// IP Internal
// Input  : Level 4 switch  : 0x0200_0000 - 0x0200_FFFF
// Output : Host MOD0 module : 0x0200_0000 - 0x0200_00FF
// Output : Host MOD1 module : 0x0200_0100 - 0x0200_01FF
// ....
// Output : Host MOD# module

generate
  for(m=0; m<N_HOST; m++) begin
    // Host Register Interconnect
    apb_switch #(
      .N_MPORT         ( 1                   ),
      .N_SPORT         ( N_HOST_MOD          ),
      .W_OFSET         ( w_ofst              ),
      .W_SW            ( ADDR_SW_CH-w_ofst   )
    ) u_apb_switch_host (
      .i_apb_clk       ( i_aclk                                            ),
      .i_apb_reset     ( i_arst                                            ),
      .i_apb_m2s       ( s_apb_m2s_host_ch[m:m]                            ),
      .o_apb_s2m       ( s_apb_s2m_host_ch[m:m]                            ),
      .o_apb_m2s       ( o_apb_m2s_host[m*N_HOST_MOD:((m+1)*N_HOST_MOD)-1] ),
      .i_apb_s2m       ( i_apb_s2m_host[m*N_HOST_MOD:((m+1)*N_HOST_MOD)-1] ),
      .i_apb_timeout   ( apb_timeout                                       )
    );
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// User Added Logic
//------------------------------------------------------------------------------------------------//

// Level 2 switch 1
// Input  : Level 1 switch   : 0x1000_0000 - 0xFFFF_FFFF
// Output : user modules     : 0x1000_0000 - 0x8FFF_FFFF

apb_m2s apb_m2s_ext [N_EXT_APB];
apb_s2m apb_s2m_ext [N_EXT_APB];

assign apb_m2s_ext            = s_apb_m2s[1:N_EXT_APB];
assign s_apb_s2m[1:N_EXT_APB] = apb_s2m_ext;

logic [N_EXT_APB-1:0] apb_psel;
logic [N_EXT_APB-1:0] apb_penable;
logic [31         :0] apb_paddr;
logic [31         :0] apb_pwdata;
logic [N_EXT_APB-1:0] apb_pwrite;

always_comb begin
  for (int i=0; i<N_EXT_APB; i++) begin
    apb_psel   [i]        = apb_m2s_ext [i].psel;
    apb_s2m_ext[i].pready = i_apb_pready[i];
    apb_s2m_ext[i].prdata = i_apb_prdata[i];
    apb_s2m_ext[i].pserr  = i_apb_pserr [i];
  end
end

assign o_apb_psel    = apb_psel;
assign o_apb_penable = apb_m2s_ext[0].penable;
assign o_apb_paddr   = apb_m2s_ext[0].paddr;
assign o_apb_pwdata  = apb_m2s_ext[0].pwdata;
assign o_apb_pwrite  = apb_m2s_ext[0].pwrite;

endmodule
