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

module rst (
  input  i_apb_clk,
  input  i_locked,    // pll reference clock locked
  input  i_pb_rst,  // asynchronous active high pushbutton reset
  input  i_sw_rst,    // software controlled system active high reset
  input  i_aligned,   // ethernet cmac system aligned

  output o_sys_rst,   // system active high reset
  output o_cmac_sys_rst  // ethernet cmac system reset
);

//----------------------------------------------------------------------------
// hololink System Reset
//----------------------------------------------------------------------------

  logic mac_aligned;

  data_sync #(
    .DATA_WIDTH  ( 1          ),
    .RESET_VALUE ( 0          ),
    .SYNC_DEPTH  ( 3          )
  ) u_sw_erst_sync (
    .clk         ( i_apb_clk  ),
    .rst_n       ( ~i_pb_rst ),
    .sync_in     ( i_aligned  ),
    .sync_out    ( mac_aligned)
  );

  reset_sync u_sys_rst (
    .i_clk    ( i_apb_clk   ),
    .i_arst_n ( ~i_pb_rst  ),
    .i_srst   ( mac_aligned ),
    .i_locked ( i_locked    ),

    .o_arst   ( o_sys_rst   ),
    .o_arst_n (             ),
    .o_srst   (             ),
    .o_srst_n (             )
  );

//----------------------------------------------------------------------------
// CMAC system reset
//----------------------------------------------------------------------------

  reset_sync u_cmac_sys_rst (
    .i_clk    ( i_apb_clk   ),
    .i_arst_n ( ~i_pb_rst  ),
    .i_srst   ( i_sw_rst    ),
    .i_locked ( i_locked    ),

    .o_arst   ( o_cmac_sys_rst),
    .o_arst_n (             ),
    .o_srst   (             ),
    .o_srst_n (             )
  );

endmodule
