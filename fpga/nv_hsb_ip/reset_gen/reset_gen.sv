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

module reset_gen #(
  parameter NUM_SENSOR_RX = 2,
  parameter NUM_SENSOR_TX = 1
)(
  input                       i_sys_rst,   // asynchronous active low pushbutton reset
  input                       i_cfg_rst,   // soft system active high reset

  input   [NUM_SENSOR_RX-1:0] i_sif_rx_clk,
  input   [NUM_SENSOR_TX-1:0] i_sif_tx_clk,
  input                       i_hif_clk,
  input                       i_apb_clk,
  input                       i_ptp_clk,

  output  [NUM_SENSOR_RX-1:0] o_sif_rx_rst,
  output  [NUM_SENSOR_TX-1:0] o_sif_tx_rst,
  output                      o_hif_rst,
  output                      o_apb_rst,
  output                      o_ptp_rst
);

// APB reset
reset_sync u_apb_rst (
  .i_clk    ( i_apb_clk  ),
  .i_arst_n (~i_sys_rst  ),
  .i_srst   ( i_cfg_rst  ),
  .i_locked ( 1'b1       ),
  .o_arst   (            ),
  .o_arst_n (            ),
  .o_srst   ( o_apb_rst  ),
  .o_srst_n (            )
);

// HIF reset
reset_sync u_hif_rst (
  .i_clk    ( i_hif_clk  ),
  .i_arst_n (~i_sys_rst  ),
  .i_srst   ( o_apb_rst  ),
  .i_locked ( 1'b1       ),
  .o_arst   (            ),
  .o_arst_n (            ),
  .o_srst   ( o_hif_rst  ),
  .o_srst_n (            )
);

generate
  for (genvar i=0; i<NUM_SENSOR_RX; i++) begin : gen_sif_rx_rst
    // SIF reset
    reset_sync u_sif_rx_rst (
      .i_clk    ( i_sif_rx_clk[i] ),
      .i_arst_n (~i_sys_rst       ),
      .i_srst   ( o_apb_rst       ),
      .i_locked ( 1'b1            ),
      .o_arst   (                 ),
      .o_arst_n (                 ),
      .o_srst   ( o_sif_rx_rst[i] ),
      .o_srst_n (                 )
    );
  end
endgenerate

generate
  for (genvar i=0; i<NUM_SENSOR_TX; i++) begin : gen_sif_tx_rst
    // SIF reset
    reset_sync u_sif_tx_rst (
      .i_clk    ( i_sif_tx_clk[i] ),
      .i_arst_n (~i_sys_rst       ),
      .i_srst   ( o_apb_rst       ),
      .i_locked ( 1'b1            ),
      .o_arst   (                 ),
      .o_arst_n (                 ),
      .o_srst   ( o_sif_tx_rst[i] ),
      .o_srst_n (                 )
    );
  end
endgenerate

// PTP reset
reset_sync u_ptp_rst (
  .i_clk    ( i_ptp_clk  ),
  .i_arst_n (~i_sys_rst  ),
  .i_srst   ( o_apb_rst  ),
  .i_locked ( 1'b1       ),
  .o_arst   (            ),
  .o_arst_n (            ),
  .o_srst   ( o_ptp_rst  ),
  .o_srst_n (            )
);


endmodule
