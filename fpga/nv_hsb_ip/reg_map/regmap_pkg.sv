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

`ifndef regmap_pkg
`define regmap_pkg

package regmap_pkg;

//------------------------------------------------------------------------------------------------//
// APB Return data for undefined read addr
//------------------------------------------------------------------------------------------------//

  localparam [31:0] INVAID_RD_DATA = 32'hBADADD12;

//----------------------------------------------------------------------------------
// APB Address Switching
//
// [31 :  28] [27  :  24] [23  :  20] [19  :  16] [15 : 12] [11  :  8] [ 7 : 0]
//  Hololink  (Hololink)     (SIF)     (SIF/HIF)   (Host)    (Submod)
//   / User    Global=0     RX = 0      # of CH   Submod=0
//              SIF  =1     TX = 8                ROCE  =1
//              HIF  =2
//              SPI  =3
//              I2C  =4
//----------------------------------------------------------------------------------
  localparam ADDR_SW_USER     = 28;
  localparam ADDR_SW_HOLOLINK = 24;
  localparam ADDR_SW_IF       = 21;
  localparam ADDR_SW_CH       = 16;
  localparam ADDR_SW_PERI     = 9 ;
  localparam ADDR_SW_ROCE     = 12;
  // user reg base address offset
  localparam w_ofst           = 8; // offset addr width

//------------------------------------------------------------------------------------------------//
// Submodules
//------------------------------------------------------------------------------------------------//
  localparam num_sen_rx_mod      = 2;
  localparam sen_rx_pkt_ctrl     = 0;
  localparam sen_rx_data_gen     = 1;

  localparam num_sen_tx_mod      = 1;

  localparam num_host_mod        = 4;
  localparam inst_dec            = 0;
  localparam eth_pkt             = 1;
  localparam ctrl_evt            = 2;
  localparam dp_pkt              = 3;

  //------------------------------------------------------------------------------------------------//
  // External User Config 0x1000_0000 - 0xFFFF_FFFF
  //   Define in customer facing file : HOLOLINK_def.svh
  //------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------------------------------------//
// max register num for each module
//------------------------------------------------------------------------------------------------//

  localparam max_n_reg = 64; // each module has 64 registers

//------------------------------------------------------------------------------------------------//
// Status Registers Offset
//------------------------------------------------------------------------------------------------//

  localparam stat_ofst = 32; // all status registers starts after 32 offset

//------------------------------------------------------------------------------------------------//
// Global Register
//------------------------------------------------------------------------------------------------//

  localparam glb_nctrl = 19;
  localparam glb_nstat = 11;
  // RW
  localparam glb_scratch = 0;  // 0x0000_0000
  localparam hsb_ctrl_0  = 1;  // 0x0000_0004
  localparam hsb_ctrl_1  = 2;  // 0x0000_0008
  localparam gpio_o_0    = 3;  // 0x0000_000C // gpio out 31:0
  localparam gpio_o_1    = 4;  // 0x0000_0010 // gpio out 63:32
  localparam gpio_o_2    = 5;  // 0x0000_0014 // gpio out 95:64
  localparam gpio_o_3    = 6;  // 0x0000_0018 // gpio out 127:96
  localparam gpio_o_4    = 7;  // 0x0000_001C // gpio out 159:128
  localparam gpio_o_5    = 8;  // 0x0000_0020 // gpio out 191:160
  localparam gpio_o_6    = 9;  // 0x0000_0024 // gpio out 223:192
  localparam gpio_o_7    = 10; // 0x0000_0028 // gpio out 255:224
  localparam gpio_t_0    = 11; // 0x0000_002C // gpio tri 31:0
  localparam gpio_t_1    = 12; // 0x0000_0030 // gpio tri 63:32
  localparam gpio_t_2    = 13; // 0x0000_0034 // gpio tri 95:64
  localparam gpio_t_3    = 14; // 0x0000_0038 // gpio tri 127:96
  localparam gpio_t_4    = 15; // 0x0000_003C // gpio tri 159:128
  localparam gpio_t_5    = 16; // 0x0000_0040 // gpio tri 191:160
  localparam gpio_t_6    = 17; // 0x0000_0044 // gpio tri 223:192
  localparam gpio_t_7    = 18; // 0x0000_0048 // gpio tri 255:224
  // RO
  localparam hsb_ver     = 32; // 0x0000_0080 // hsb_ip version
  localparam hsb_date    = 33; // 0x0000_0084 // hsb_ip date
  localparam hsb_stat    = 34; // 0x0000_0088 // dev_ver external
  localparam gpio_i_0    = 35; // 0x0000_008C // gpio in 31:0
  localparam gpio_i_1    = 36; // 0x0000_0090 // gpio in 63:32
  localparam gpio_i_2    = 37; // 0x0000_0094 // gpio in 95:64
  localparam gpio_i_3    = 38; // 0x0000_0098 // gpio in 127:96
  localparam gpio_i_4    = 39; // 0x0000_009C // gpio in 159:128
  localparam gpio_i_5    = 40; // 0x0000_00A0 // gpio in 191:160
  localparam gpio_i_6    = 41; // 0x0000_00A4 // gpio in 223:192
  localparam gpio_i_7    = 42; // 0x0000_00A8 // gpio in 255:224

//------------------------------------------------------------------------------------------------//
// PTP Register
//------------------------------------------------------------------------------------------------//

  localparam ptp_nctrl = 7;
  localparam ptp_nstat = 9;
  // RW
  localparam ptp_scratch        = 0; // 0x0000_0000
  localparam ptp_ctrl           = 1; // 0x0000_0004
  localparam ptp_ctrl_profile   = 2; // 0x0000_0008
  localparam ptp_ctrl_dly       = 3; // 0x0000_000C
  localparam ptp_ctrl_dpll_cfg1 = 4; // 0x0000_0010
  localparam ptp_ctrl_dpll_cfg2 = 5; // 0x0000_0014
  localparam ptp_ctrl_avg_fact  = 6; // 0x0000_0018

  // RO
  localparam ptp_sync_ts_0     = 32; // 0x0000_0080
  localparam ptp_sync_cf_0     = 33; // 0x0000_0084
  localparam ptp_sync_stat     = 34; // 0x0000_0088
  localparam ptp_ofm           = 35; // 0x0000_008C
  localparam ptp_mean_dly      = 36; // 0x0000_0090
  localparam ptp_inc           = 37; // 0x0000_0094
  localparam ptp_fa_adj        = 38; // 0x0000_0098
  localparam ptp_dly_cf_0      = 39; // 0x0000_009C
  localparam ptp_ip_dly_asymm  = 40; // 0x0000_00A0

//------------------------------------------------------------------------------------------------//
// SFP+ Register
//------------------------------------------------------------------------------------------------//

  // see SFF-8472-R12.4.pdf
  // Controlled

//------------------------------------------------------------------------------------------------//
// Instruction Decoder Registers
//------------------------------------------------------------------------------------------------//

  localparam inst_dec_nctrl = 10;
  localparam inst_dec_nstat = 2;
  // RW
  localparam inst_dec_scratch   = 0;  // 0x0000_0000
  localparam inst_dec_ctrl      = 1;  // 0x0000_0004
  localparam lpbk_udp_port      = 2;  // 0x0000_0008
  localparam roce_ecb_dest_qp   = 3;  // 0x0000_000C
  localparam inst_dec_rsvd_1    = 4;  // 0x0000_0010
  localparam inst_dec_rsvd_2    = 5;  // 0x0000_0018
  localparam inst_dec_rsvd_3    = 6;  // 0x0000_0018
  localparam inst_dec_rsvd_4    = 7;  // 0x0000_0018
  localparam ecb_udp_port       = 8;  // 0x0000_0020
  localparam stx_udp_port       = 9;  // 0x0000_0024
  // RO
  localparam inst_dec_stat       = 32; // 0x0000_0080
  localparam ipv4_chksum_err_cnt = 33; // 0x0000_0084

//------------------------------------------------------------------------------------------------//
// Ethernet Packet
//------------------------------------------------------------------------------------------------//

  localparam eth_pkt_nctrl = 3;
  localparam eth_pkt_nstat = 1;
  // RW
  localparam eth_pkt_scratch = 0;  // 0x0000_0000
  localparam eth_pkt_ctrl    = 1;  // 0x0000_0008
  localparam eth_pkt_hp_cnt  = 2;  // 0x0000_0004
  // RO
  localparam eth_pkt_stat    = 32; // 0x0000_0080


//------------------------------------------------------------------------------------------------//
// ECB
//------------------------------------------------------------------------------------------------//

  localparam ecb_nctrl = 6;
  localparam ecb_nstat = 1;
  // RW
  localparam ecb_scratch  = 0;  // 0x0000_0000
  localparam ecb_addr_lsb = 1;  // 0x0000_0008
  localparam ecb_addr_msb = 2;  // 0x0000_0004
  localparam ecb_rkey     = 3;  // 0x0000_000C
  localparam ecb_pkey     = 4;  // 0x0000_0010
  localparam ecb_dest_qp  = 5;  // 0x0000_0014
  // RO
  localparam ecb_stat    = 32; // 0x0000_0080

//------------------------------------------------------------------------------------------------//
// Control Bus Event
//------------------------------------------------------------------------------------------------//

  localparam ctrl_evt_nctrl = 12;
  localparam ctrl_evt_nstat = 2;
  // RW
  localparam ctrl_evt_scratch          = 0;  // 0x0000_0000
  localparam ctrl_evt_rising           = 1;  // 0x0000_0004
  localparam ctrl_evt_falling          = 2;  // 0x0000_0008
  localparam ctrl_evt_clear            = 3;  // 0x0000_000C
  localparam ctrl_evt_host_mac_addr_lo = 4;  // 0x0000_0010
  localparam ctrl_evt_host_mac_addr_hi = 5;  // 0x0000_0014
  localparam ctrl_evt_host_ip_addr     = 6;  // 0x0000_0018
  localparam ctrl_evt_host_udp_port    = 7;  // 0x0000_001C
  localparam ctrl_evt_dev_udp_port     = 8;  // 0x0000_0020
  localparam ctrl_evt_apb_interrupt_en = 9;  // 0x0000_0024
  localparam ctrl_evt_apb_timeout      = 10; // 0x0000_0028
  localparam ctrl_evt_sw_event         = 11; // 0x0000_002C
  // RO
  localparam ctrl_evt_stat             = 32; // 0x0000_0080

//------------------------------------------------------------------------------------------------//
// Dataplane Packetizer
//------------------------------------------------------------------------------------------------//

  localparam dp_pkt_nctrl = 6;
  localparam dp_pkt_nstat = 1;
  // RW
  localparam dp_pkt_scratch          = 0;  // 0x0000_0000
  localparam dp_pkt_len              = 1;  // 0x0000_0004
  localparam dp_pkt_dev_udp_port     = 2;  // 0x0000_0008
  localparam dp_pkt_vip_mask         = 3;  // 0x0000_000C
  localparam dp_pkt_crc_xor          = 4;  // 0x0000_0010
  localparam dp_pkt_1722B            = 5;  // 0x0000_0014
  // RO
  localparam dp_pkt_stat             = 32;


//------------------------------------------------------------------------------------------------//
// Packetizer Register
//------------------------------------------------------------------------------------------------//

  localparam pack_nctrl = 4;
  localparam pack_nstat = 2;

  localparam pack_scratch          = 0;  // 0x0000_0000
  localparam pack_ram_addr         = 1;  // 0x0000_0004
  localparam pack_ram_data         = 2;  // 0x0000_0008
  localparam pack_ctrl             = 3;  // 0x0000_000C
  // RO
  localparam pack_tvalid_cnt       = 32; // 0x0000_0080
  localparam pack_psn_cnt          = 33; // 0x0000_0084

//------------------------------------------------------------------------------------------------//
// Sensor RX Data Generator
//------------------------------------------------------------------------------------------------//

  localparam sen_rx_data_gen_nctrl    = 10;
  localparam sen_rx_data_gen_nstat    = 1;

  localparam sen_rx_data_gen_scratch      = 0;  // 0x0000_0000
  localparam sen_rx_data_gen_ena          = 1;  // 0x0000_0004
  localparam sen_rx_data_gen_mode         = 2;  // 0x0000_0008
  localparam sen_rx_data_gen_size         = 3;  // 0x0000_000C
  localparam sen_rx_data_gen_output_rate  = 4;  // 0x0000_0010
  localparam sen_rx_prbs_seed             = 9;  // 0x0000_0024
//------------------------------------------------------------------------------------------------//
// Peripheral Data Bus
//------------------------------------------------------------------------------------------------//

  localparam apb_spi_idx   = 0;
  localparam apb_i2c_idx   = 1;
  localparam apb_uart_idx  = 2;

endpackage

`endif
