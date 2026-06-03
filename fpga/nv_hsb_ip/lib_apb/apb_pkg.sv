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

`ifndef apb_pkg
`define apb_pkg

package apb_pkg;

  localparam W_DATA = 32; // data bus width
  localparam W_ADDR = 32; // address bus width
//------------------------------------------------------------------------------
// APB Bus
//------------------------------------------------------------------------------
  
  typedef struct packed {
    logic              psel;
    logic              penable;
    logic [W_ADDR-1:0] paddr;
    logic [W_DATA-1:0] pwdata;
    logic              pwrite;
  } apb_m2s;
  
  typedef struct packed {
    logic              pready;
    logic [W_DATA-1:0] prdata;
    logic              pserr;
  } apb_s2m;

endpackage

`endif