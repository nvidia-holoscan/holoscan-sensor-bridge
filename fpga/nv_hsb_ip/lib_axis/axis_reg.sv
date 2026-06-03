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

module axis_reg #(
    parameter DWIDTH = 64,
    parameter SKID   = 1
)
(
    input                                       clk,
    input                                       rst,

    input                                       i_axis_rx_tvalid,
    input           [DWIDTH-1:0]                i_axis_rx_tdata,
    output                                      o_axis_rx_tready,

    output                                      o_axis_tx_tvalid,
    output          [DWIDTH-1:0]                o_axis_tx_tdata,
    input                                       i_axis_tx_tready
);

logic                 r_axis_tvalid;
logic   [DWIDTH-1:0]  r_axis_tdata;

logic                 sr_axis_tvalid;
logic   [DWIDTH-1:0]  sr_axis_tdata;

logic                 r_axis_ready;

logic                 load;
logic                 skid;

//------------------------------------------------------------------------------------------------//
// Control path
//------------------------------------------------------------------------------------------------//

always_ff @(posedge clk) begin
  if (rst) begin
    r_axis_tvalid  <= '0;
    r_axis_ready   <= '0;
    sr_axis_tvalid <= '0;
    skid           <= '0;
  end
  else begin
    if (skid && SKID) begin
      if (load) begin
        r_axis_tvalid <= sr_axis_tvalid;
        r_axis_ready  <= 1'b1;
        skid          <= 1'b0;
      end
    end
    else begin
      if (load) begin
        r_axis_tvalid <= i_axis_rx_tvalid;
        r_axis_ready  <= 1'b1;
        skid          <= 1'b0;
      end
      else begin
        sr_axis_tvalid <= i_axis_rx_tvalid;
        r_axis_ready   <= !i_axis_rx_tvalid;
        skid           <= i_axis_rx_tvalid;
      end
    end
  end
end

//------------------------------------------------------------------------------------------------//
// Data path (no reset needed)
//------------------------------------------------------------------------------------------------//
always_ff @(posedge clk) begin
  if (skid && SKID) begin
    if (load) begin
      r_axis_tdata  <= sr_axis_tdata;
    end
  end
  else begin
    if (load) begin
      r_axis_tdata  <= i_axis_rx_tdata;
    end
    else begin
      sr_axis_tdata  <= i_axis_rx_tdata;
    end
  end
end

// Load into register when data is valid and either reg is empty or reg is being sent
assign load  = (!r_axis_tvalid || i_axis_tx_tready);

assign o_axis_tx_tvalid = r_axis_tvalid;
assign o_axis_tx_tdata  = r_axis_tdata;
assign o_axis_rx_tready = (SKID) ? (r_axis_ready) : load;

endmodule
