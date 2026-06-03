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

module sys_init
  import apb_pkg::*;
  // import regmap_pkg::*;
#(
  parameter W_DATA = 32,
  parameter W_ADDR = 32,
  parameter N_REG  = 1,
  parameter W_INIT = W_DATA + W_ADDR
)(
  // clock and reset
  input               i_aclk,
  input               i_arst,
  // control
  input               i_init,
  input  [W_INIT-1:0] i_init_reg [N_REG],
  output              o_done,
  // APB Interface
  output apb_m2s      o_apb_m2s,
  input  apb_s2m      i_apb_s2m
);

//------------------------------------------------------------------------------------------------//
// APB Register Initialization
//------------------------------------------------------------------------------------------------//

logic       init_q;
logic       init_en;

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    init_q     <= 1'b0;
    init_en    <= 1'b0;
  end
  else begin
    init_q     <= i_init;
    if (!init_en) begin
      init_en    <= i_init & !init_q;
    end
  end
end

logic [W_DATA-1:0] init_data [N_REG];
logic [W_ADDR-1:0] init_addr [N_REG];

genvar i;
generate
  for (i=0; i<N_REG; i++) begin
    assign init_addr [i] = i_init_reg [i][W_ADDR+W_DATA-1:W_DATA];
    assign init_data [i] = i_init_reg [i][       W_DATA-1:     0];
  end
endgenerate

logic              init_err;
logic              init_done;

m_apb_reg #(
  .N_REG         ( N_REG         ), // Number of registers in this sequence
  .N_DELAY       ( 32            )  // Number of clock cycle delay in between sequence, 1-255. 0: i_init triggered delay only
) u_reg_init     (
  // Clock and Reset
  .i_aclk        ( i_aclk        ),
  .i_arst        ( i_arst        ),
  // Register Sequence
  .i_data        ( init_data     ),
  .i_addr        ( init_addr     ),
  .i_init        ( init_en       ),
  .i_init_en     ( 1'b1          ),
  .i_wren        ( {N_REG{1'b1}} ),
  .o_done        ( init_done     ),
  .o_ready       (               ),
  .o_data        (               ),
  .o_addr        (               ),
  .o_err         ( init_err      ),
  //APB Interface
  .o_apb_m2s     ( o_apb_m2s     ),
  .i_apb_s2m     ( i_apb_s2m     )
);

logic done;

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    done <= 0;
  end
  else begin
    done <= done | init_done;
  end
end

assign o_done = done;

endmodule
