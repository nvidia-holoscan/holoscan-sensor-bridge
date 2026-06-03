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

`ifndef rx_parser_pkg
`define rx_parser_pkg

package rx_parser_pkg;

  typedef struct {
    logic     [47:0]                dest_mac;
    logic     [47:0]                src_mac;
    logic     [15:0]                ethertype;
  } eth_hdr;

  typedef struct {
    logic     [7:0]                 ver_ihl;
    logic     [7:0]                 dscp_ecn;
    logic     [15:0]                length;
    logic     [15:0]                id;
    logic     [15:0]                flags_frag;
    logic     [7:0]                 ttl;
    logic     [7:0]                 protocol;
    logic     [15:0]                chksum;
    logic     [31:0]                src_addr;
    logic     [31:0]                dest_addr;
  } ip_hdr;

  typedef struct {
    logic     [15:0]                src_port;
    logic     [15:0]                dest_port;
    logic     [15:0]                length;
    logic     [15:0]                chksum;
  } udp_hdr;

  typedef struct {
    logic     [7:0]                 icmp_type;
    logic     [7:0]                 code;
    logic     [15:0]                chksum;
    logic     [15:0]                id;
    logic     [15:0]                seq_num;
  } ping_hdr;

  typedef struct {
    logic     [15:0]                htype;
    logic     [15:0]                ptype;
    logic     [7:0]                 hlen;
    logic     [7:0]                 plen;
    logic     [15:0]                oper;
    logic     [47:0]                smac;
    logic     [31:0]                spa;
    logic     [47:0]                tmac;
    logic     [31:0]                tpa;
  } arp_hdr;

  typedef struct {
    logic     [7:0]                 opcode;
    logic     [7:0]                 s_m_pad_tver;
    logic     [15:0]                pkey;
    logic     [7:0]                 f_b_rsv6;
    logic     [23:0]                dest_qp;
    logic     [7:0]                 a_rsv7;
    logic     [23:0]                psn;
  } bth_hdr;


endpackage

`endif
