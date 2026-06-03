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

`ifndef ptp_pkg
`define ptp_pkg

package ptp_pkg;

//------------------------------------------------------------------------------------------------//
// ETHERNET L2 LAYER
//------------------------------------------------------------------------------------------------//

  localparam        ETH_L2_HDR_WIDTH              = 14*8;
  localparam [47:0] PTP_FW_MULTI_ADDR             = 48'h011B19000000;
  localparam [47:0] PTP_NON_FW_MULTI_ADDR         = 48'h0180C200000E;
  localparam [15:0] PTP_ETH_TYPE                  = 16'h88F7;

  localparam [47:0] BE_PTP_FW_MULTI_ADDR          = 48'h000000191B01;
  localparam [47:0] BE_PTP_NON_FW_MULTI_ADDR      = 48'h0E0000C28001;
  localparam [15:0] BE_PTP_ETH_TYPE               = 16'hF788;

//------------------------------------------------------------------------------------------------//
// PTP MESSAGE TYPES
//------------------------------------------------------------------------------------------------//
  localparam [3:0]  VERSION_PTP                   = 4'd2;

  localparam [3:0]  MSG_SYNC                      = 4'h0;
  localparam [3:0]  MSG_DELAY_REQ                 = 4'h1;
  localparam [3:0]  MSG_PEER_DELAY_REQ            = 4'h2;
  localparam [3:0]  MSG_PEER_DELAY_RESP           = 4'h3;
  localparam [3:0]  MSG_FOLLOW_UP                 = 4'h8;
  localparam [3:0]  MSG_DELAY_RESP                = 4'h9;
  localparam [3:0]  MSG_PEER_DELAY_RESP_FOLLOW_UP = 4'hA;

//------------------------------------------------------------------------------------------------//
// PTP RTL SPECIFIC
//------------------------------------------------------------------------------------------------//
  localparam W_DLY = 24;
  localparam W_OFM = W_DLY;
  localparam W_FA  = W_OFM;

  localparam     [30:0] BILLION         = 10**9;
  localparam            W_FRAC_NS       = 24;
  localparam            W_NS            = 31 + W_FRAC_NS;

endpackage

`endif
