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

module dp_ram #(
  parameter DATA_WIDTH = 32,                // bit width of data
  parameter RAM_DEPTH  = 256,               // bit width of internal ram addressing
  parameter ADDR_WIDTH = $clog2(RAM_DEPTH), // bit width of address
  parameter RAM_TYPE   = "SIMPLE",          // "SIMPLE" or "TRUE"
  parameter MEM_STYLE  = "AUTO"             // "LUT" / "BLOCK" / "AUTO"
)(
  // Port A
  input  wire                  clk_a,
  input  wire                  en_a,
  input  wire                  we_a,
  input  wire [DATA_WIDTH-1:0] din_a,
  input  wire [ADDR_WIDTH-1:0] addr_a,
  output      [DATA_WIDTH-1:0] dout_a,
  // Port B
  input  wire                  clk_b,
  input  wire                  en_b,
  input  wire                  we_b,
  input  wire [DATA_WIDTH-1:0] din_b,
  input  wire [ADDR_WIDTH-1:0] addr_b,
  output      [DATA_WIDTH-1:0] dout_b
);

logic [DATA_WIDTH-1:0] mem     [RAM_DEPTH] = '{default:'0};
(* ram_style = "block" *)      logic [DATA_WIDTH-1:0] blk_mem [RAM_DEPTH] = '{default:'0}/*synthesis syn_ramstyle = "block_ram"*/;
(* ram_style = "distributed" *)logic [DATA_WIDTH-1:0] lut_mem [RAM_DEPTH] = '{default:'0}/*synthesis syn_ramstyle = "distributed"*/;

//------------------------------------------------------------------------------------------------//
// Port A is always RD/WR
//------------------------------------------------------------------------------------------------//

logic [DATA_WIDTH-1:0] rd_data_a;

generate
  if (MEM_STYLE == "LUT") begin : LUT_RAM_A
    always @ (posedge clk_a) begin : port_a
      if (en_a) begin
        if (we_a) begin
          lut_mem[addr_a] <= din_a;
        end
        rd_data_a     <= lut_mem[addr_a];
      end
    end
  end

  else if (MEM_STYLE == "BLOCK") begin : BLOCK_RAM_A
    always @ (posedge clk_a) begin : port_a
      if (en_a) begin
        if (we_a) begin
          blk_mem[addr_a] <= din_a;
        end
        rd_data_a     <= blk_mem[addr_a];
      end
    end
  end

  else begin : AUTO_RAM_A
    always @ (posedge clk_a) begin : port_a
      if (en_a) begin
        if (we_a) begin
          mem[addr_a] <= din_a;
        end
        rd_data_a     <= mem[addr_a];
      end
    end
  end

endgenerate

//------------------------------------------------------------------------------------------------//
// Port B is only active on clock B when "true" option is selected for RAM_TYPE
//------------------------------------------------------------------------------------------------//

logic [DATA_WIDTH-1:0] rd_data_b;

generate

  if (MEM_STYLE == "LUT") begin : LUT_RAM_B
    if (RAM_TYPE == "TRUE") begin : TRUE_DP_RAM
      always @ (posedge clk_b) begin : tdp_port_b
        if (en_b) begin
          if (we_b) begin
            lut_mem[addr_b] <= din_b;
          end
          rd_data_b     <= lut_mem[addr_b];
        end
      end
    end
    else begin : SIMPLE_DP_RAM
      always @ (posedge clk_a) begin : sdp_port_b
        if (en_a) begin
          rd_data_b <= lut_mem[addr_b];
        end
      end
    end
  end

  else if (MEM_STYLE == "BLOCK") begin : BLOCK_RAM_B
    if (RAM_TYPE == "TRUE") begin : TRUE_DP_RAM
      always @ (posedge clk_b) begin : tdp_port_b
        if (en_b) begin
          if (we_b) begin
            blk_mem[addr_b] <= din_b;
          end
          rd_data_b     <= blk_mem[addr_b];
        end
      end
    end
    else begin : SIMPLE_DP_RAM
      always @ (posedge clk_a) begin : sdp_port_b
        if (en_a) begin
          rd_data_b <= blk_mem[addr_b];
        end
      end
    end
  end

  else begin : AUTO_RAM_B
    if (RAM_TYPE == "TRUE") begin : TRUE_DP_RAM
      always @ (posedge clk_b) begin : tdp_port_b
        if (en_b) begin
          if (we_b) begin
            mem[addr_b] <= din_b;
          end
          rd_data_b     <= mem[addr_b];
        end
      end
    end
    else begin : SIMPLE_DP_RAM
      always @ (posedge clk_a) begin : sdp_port_b
        if (en_a) begin
          rd_data_b <= mem[addr_b];
        end
      end
    end
  end

endgenerate

assign dout_a = rd_data_a;
assign dout_b = rd_data_b;

endmodule
