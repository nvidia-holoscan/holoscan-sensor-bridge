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

`ifndef axis_pkg
`define axis_pkg

`include "HOLOLINK_def.svh"

package axis_pkg;

  //Swap endianness of Host AXIS data bus
  function automatic [`HOST_WIDTH-1:0] host_axis_tdata_end_swap (input [`HOST_WIDTH-1:0] axis_tdata_in);
    begin
      for (int j = 0; j < (`HOST_WIDTH/8); j = j+1) begin
        host_axis_tdata_end_swap[j*8+:8] = axis_tdata_in[(((`HOST_WIDTH/8)-1)-j)*8+:8];
      end
    end
  endfunction

  //Swap endianness of Host AXIS keep bus
  function automatic [(`HOST_WIDTH/8)-1:0] host_axis_tkeep_end_swap (input [(`HOST_WIDTH/8)-1:0] axis_tkeep_in);
    begin
      for (int i = 0; i < (`HOST_WIDTH/8); i=i+1) begin
        host_axis_tkeep_end_swap[i] = axis_tkeep_in[((`HOST_WIDTH/8)-1)-i];
      end
    end
  endfunction

  //Swap endianness of Sensor AXIS data bus
  function automatic [`DATAPATH_WIDTH-1:0] sensor_axis_tdata_end_swap (input [`DATAPATH_WIDTH-1:0] axis_tdata_in);
    begin
      for (int j = 0; j < (`DATAPATH_WIDTH/8); j = j+1) begin
        sensor_axis_tdata_end_swap[j*8+:8] = axis_tdata_in[(((`DATAPATH_WIDTH/8)-1)-j)*8+:8];
      end
    end
  endfunction

  //Swap endianness of Host AXIS keep bus
  function automatic [(`DATAPATH_WIDTH/8)-1:0] sensor_axis_tkeep_end_swap (input [(`DATAPATH_WIDTH/8)-1:0] axis_tkeep_in);
    begin
      for (int i = 0; i < (`DATAPATH_WIDTH/8); i=i+1) begin
        sensor_axis_tkeep_end_swap[i] = axis_tkeep_in[((`DATAPATH_WIDTH/8)-1)-i];
      end
    end
  endfunction

  function automatic [(`DATAPATH_WIDTH/8)-1:0] dec_to_onehot (input [(`DATAPATH_WIDTH/8)-1:0] tkeep_dec);
    begin
      dec_to_onehot = '0;
      for (int i = 0; i < tkeep_dec; i=i+1) begin
        dec_to_onehot[i] = 1'b1;
      end
    end
  endfunction

  function automatic [($clog2(`HOST_WIDTH/8)):0] host_onehot_to_dec (input [(`HOST_WIDTH/8)-1:0] tkeep_onehot);
    begin
      host_onehot_to_dec = '0;
      for (int i = 0; i < `HOST_WIDTH/8; i=i+1) begin
        if (tkeep_onehot[i]) begin
          host_onehot_to_dec = i + 1'b1;
        end
      end
    end
  endfunction

endpackage

`endif
