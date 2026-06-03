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

`ifndef ecb_pkg
`define ecb_pkg

package ecb_pkg;

  localparam WR_WORD  = 8'h02;
  localparam WR_DWORD = 8'h04;
  localparam WR_BLOCK = 8'h09;
  localparam RMW_BYTE = 8'h0A;
  localparam RD_BYTE  = 8'h11;
  localparam RD_WORD  = 8'h12;
  localparam RD_DWORD = 8'h14;
  localparam RD_BLOCK = 8'h19;
  localparam GET_INFO = 8'h20;
  localparam WR_BYTE  = 8'h01;

endpackage

`endif