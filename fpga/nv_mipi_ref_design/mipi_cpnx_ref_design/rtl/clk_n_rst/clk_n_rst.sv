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

module clk_n_rst (
  // Reference Clock
  input  i_refclk,    // pll reference clock
  input  i_locked,    // pll reference clock locked
  // System Clocks
  output o_mipi_clk,  // 27.043MHz clock for MIPI
  output o_pcs_clk,   // pcs calibration clock
  output o_hif_clk,   // processing clock
  output o_apb_clk,   // apb clock
  output o_ptp_clk,
  //PTP Clock
  input  [31:0] i_ptp_nsec,    //Nanosecond
    output        o_ptp_cam_clk, //PTP 24MHz
  //
  input  i_pb_rst_n,  // asynchronous active low pushbutton reset
  input  i_sw_rst,    // software controlled system active high reset

  output o_sys_rst,   // system active high reset
  output o_pcs_rst_n  // ethernet pcs reset
);

  // pcs calibration clock
  osc_clk u_pcs_clk (
    .hf_out_en_i  ( '1         ), // always enable
    .hf_clk_out_o ( o_pcs_clk  )
  );

  logic locked;
  logic ptp_locked;

  // Primary clock pll (NONE FRAC DIV)
  eclk_mipi_pll u_clk_pll (
    .clki_i   ( i_refclk  ),
    .rstn_i   ( i_locked  ),
    .clkop_o  (           ),
    .clkos_o  ( o_hif_clk ), // 156.25 MHz
    .clkos2_o ( o_apb_clk ), // 19.53125 MHz
    .clkos3_o ( o_mipi_clk), // 27.043 MHz
    .clkos4_o ( o_ptp_clk ), // 100.446545 MHz
    .lock_o   ( locked    )
  );

  logic ptp_sensor_pll_locked;
  //PTP generated 24MHz clock
  ptp_sensor_pll u_ptp_to_sensor_pll (
    .clki_i    ( i_ptp_nsec[2]         ), //25MHz
    .rstn_i    ( locked                ),
    .clkop_o   (                       ),
    .clkos_o   ( o_ptp_cam_clk         ), //24.038MHz
    .clkos2_o  (                       ), //36.765MHz
    .lock_o    ( ptp_sensor_pll_locked )
  );

//----------------------------------------------------------------------------
// System Reset
//----------------------------------------------------------------------------

  reset_sync u_sys_rst (
    .i_clk    ( o_apb_clk   ),
    .i_arst_n ( i_pb_rst_n  ),
    .i_srst   ( 1'b0        ),
    .i_locked ( locked      ),

    .o_arst   ( o_sys_rst   ),
    .o_arst_n (             ),
    .o_srst   (             ),
    .o_srst_n (             )
  );

//----------------------------------------------------------------------------
// MAC Reset
//----------------------------------------------------------------------------

  logic pcs_rst;

  data_sync #(
    .DATA_WIDTH  ( 1          ),
    .RESET_VALUE ( 0          ),
    .SYNC_DEPTH  ( 32         )
  ) u_sw_erst_sync (
    .clk         ( o_pcs_clk  ),
    .rst_n       ( i_pb_rst_n ),
    .sync_in     ( i_sw_rst   ),
    .sync_out    ( pcs_rst    )
  );

  reset_sync u_eth_rst (
    .i_clk    ( o_pcs_clk   ),
    .i_arst_n ( i_pb_rst_n  ),
    .i_srst   ( pcs_rst     ),
    .i_locked ( 1'b1        ),

    .o_arst   (             ),
    .o_arst_n ( o_pcs_rst_n ),
    .o_srst   (             ),
    .o_srst_n (             )
  );

endmodule
