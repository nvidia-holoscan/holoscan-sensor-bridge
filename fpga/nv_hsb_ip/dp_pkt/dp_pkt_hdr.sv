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

module dp_pkt_hdr #(
    parameter W_DATA         = 64,
    parameter W_KEEP         = W_DATA/8,
    parameter ROCE_HDR_WIDTH = 128,
    parameter COE_HDR_WIDTH  = 128,
    parameter WR_IMM_SIZE    = 128,
    parameter METADATA_SIZE  = 128,
    parameter PADDING_SIZE   = 128
)(
  input                           i_clk,
  input                           i_rst,
  // Header inputs (vectors)
  input  [ROCE_HDR_WIDTH-1:0]     i_roce_header,
  input  [COE_HDR_WIDTH-1:0]      i_coe_header,
  input  [WR_IMM_SIZE-1:0]        i_wr_imm,
  input  [METADATA_SIZE-1:0]      i_metadata,
  // Control
  input  [2:0]                    i_header_type,
  input                           i_trigger,
  output                          o_busy,
  // AXIS output
  output                          o_axis_tvalid,
  output                          o_axis_tlast,
  output [W_DATA-1:0]             o_axis_tdata,
  output [W_KEEP-1:0]             o_axis_tkeep,
  output                          o_axis_tuser,
  input                           i_axis_tready
);

// 00=RoCE, 01=COE, 10=ROCE+IMM+Metadata+Padding, 11=COE+Metadata+Padding

logic [W_DATA-1:0] r_axis_tdata;
logic              r_axis_tlast;
logic              r_axis_tready;
logic [W_KEEP-1:0] r_axis_tkeep;
logic              r_axis_tvalid;


localparam HDR_FULL_WIDTH = ROCE_HDR_WIDTH + WR_IMM_SIZE + PADDING_SIZE;
localparam HDR_FULL_CYCLES = (HDR_FULL_WIDTH-1)/W_DATA + 1;
localparam W_HDR           = HDR_FULL_CYCLES * W_DATA;

logic [W_HDR-1:0] hdr;

localparam HDR_00_WIDTH  = ROCE_HDR_WIDTH;
localparam HDR_01_WIDTH  = COE_HDR_WIDTH;
localparam HDR_10_WIDTH  = ROCE_HDR_WIDTH + WR_IMM_SIZE + PADDING_SIZE;
localparam HDR_11_WIDTH  = COE_HDR_WIDTH + PADDING_SIZE;
localparam HDR_100_WIDTH = ROCE_HDR_WIDTH + WR_IMM_SIZE;

localparam HDR_00_DATA_WIDTH  = ROCE_HDR_WIDTH;
localparam HDR_01_DATA_WIDTH  = COE_HDR_WIDTH;
localparam HDR_10_DATA_WIDTH  = ROCE_HDR_WIDTH + WR_IMM_SIZE + METADATA_SIZE;
localparam HDR_11_DATA_WIDTH  = COE_HDR_WIDTH + METADATA_SIZE;
localparam HDR_100_DATA_WIDTH = ROCE_HDR_WIDTH + WR_IMM_SIZE;

localparam HDR_00_CYCLES  = (HDR_00_WIDTH-1)/W_DATA + 1;
localparam HDR_01_CYCLES  = (HDR_01_WIDTH-1)/W_DATA + 1;
localparam HDR_10_CYCLES  = (HDR_10_WIDTH-1)/W_DATA + 1;
localparam HDR_11_CYCLES  = (HDR_11_WIDTH-1)/W_DATA + 1;
localparam HDR_100_CYCLES = (HDR_100_WIDTH-1)/W_DATA + 1;

localparam HDR_00_VEC_WIDTH  = HDR_00_CYCLES * W_DATA;
localparam HDR_01_VEC_WIDTH  = HDR_01_CYCLES * W_DATA;
localparam HDR_10_VEC_WIDTH  = HDR_10_CYCLES * W_DATA;
localparam HDR_11_VEC_WIDTH  = HDR_11_CYCLES * W_DATA;
localparam HDR_100_VEC_WIDTH = HDR_100_CYCLES * W_DATA;

localparam HDR_00_VAL_BITS  = (HDR_00_WIDTH%W_DATA) == 0 ? W_DATA : (HDR_00_WIDTH%W_DATA);
localparam HDR_01_VAL_BITS  = (HDR_01_WIDTH%W_DATA) == 0 ? W_DATA : (HDR_01_WIDTH%W_DATA);
localparam HDR_10_VAL_BITS  = (HDR_10_WIDTH%W_DATA) == 0 ? W_DATA : (HDR_10_WIDTH%W_DATA);
localparam HDR_11_VAL_BITS  = (HDR_11_WIDTH%W_DATA) == 0 ? W_DATA : (HDR_11_WIDTH%W_DATA);
localparam HDR_100_VAL_BITS = (HDR_100_WIDTH%W_DATA) == 0 ? W_DATA : (HDR_100_WIDTH%W_DATA);

localparam HDR_00_TKEEP  = {'0,{(HDR_00_VAL_BITS/8){1'b1}}};
localparam HDR_01_TKEEP  = {'0,{(HDR_01_VAL_BITS/8){1'b1}}};
localparam HDR_10_TKEEP  = {'0,{(HDR_10_VAL_BITS/8){1'b1}}};
localparam HDR_11_TKEEP  = {'0,{(HDR_11_VAL_BITS/8){1'b1}}};
localparam HDR_100_TKEEP = {'0,{(HDR_100_VAL_BITS/8){1'b1}}};

localparam HDR_00_DATA_CYCLES  = (HDR_00_DATA_WIDTH-1)/W_DATA + 1;
localparam HDR_01_DATA_CYCLES  = (HDR_01_DATA_WIDTH-1)/W_DATA + 1;
localparam HDR_10_DATA_CYCLES  = (HDR_10_DATA_WIDTH-1)/W_DATA + 1;
localparam HDR_11_DATA_CYCLES  = (HDR_11_DATA_WIDTH-1)/W_DATA + 1;
localparam HDR_100_DATA_CYCLES = (HDR_100_DATA_WIDTH-1)/W_DATA + 1;

logic [HDR_00_VEC_WIDTH-1:0]  hdr_00;
logic [HDR_01_VEC_WIDTH-1:0]  hdr_01;
logic [HDR_10_VEC_WIDTH-1:0]  hdr_10;
logic [HDR_11_VEC_WIDTH-1:0]  hdr_11;
logic [HDR_100_VEC_WIDTH-1:0] hdr_100;

assign hdr_00  = {'0,i_roce_header};
assign hdr_01  = {'0,i_coe_header};
assign hdr_10  = {'0,{i_metadata,i_wr_imm,i_roce_header}};
assign hdr_11  = {'0,{i_metadata,i_coe_header}};
assign hdr_100 = {i_wr_imm,i_roce_header};


typedef enum logic [2:0] {
  AXI_IDLE     = 3'b001,
  AXI_SEND     = 3'b010,
  AXI_LAST     = 3'b100
} vec_states;
vec_states vec_state;

logic [$clog2(HDR_10_WIDTH+1)+1:0]  cnt;
logic                               r_trigger;
logic                               r_busy;
logic                               is_last;
logic [W_KEEP-1:0]                  tkeep;
logic                               data_is_zero;


always_ff @(posedge i_clk) begin
  if (i_rst) begin
    cnt <= '0;
  end 
  else begin
    cnt <= (data_is_zero) ? '0 : (cnt + r_axis_tready);
  end
end


always_ff @(posedge i_clk) begin
  if (i_rst) begin
    vec_state               <= AXI_IDLE;
    r_trigger               <= '0;
    r_busy                  <= '0;
    data_is_zero            <= '1;
  end
  else begin
    r_trigger <= i_trigger;
    case (vec_state)
      AXI_IDLE: begin
        if ({r_trigger,i_trigger} == 2'b01) begin
          vec_state           <=  is_last ? AXI_LAST : AXI_SEND;
          r_busy              <= '1;
          data_is_zero        <= '0;
        end
      end
      AXI_SEND: begin
        if (r_axis_tready) begin
          vec_state               <= is_last     ? AXI_LAST : vec_state;
        end
      end
      AXI_LAST: begin
        vec_state               <= r_axis_tready ? AXI_IDLE : vec_state;
        r_busy                  <= !r_axis_tready;
        data_is_zero            <= r_axis_tready;
      end
      default: begin
        vec_state               <= AXI_IDLE;
      end
    endcase
  end
end



always_comb begin
  is_last = '0;
  casez(i_header_type)
    3'b000: is_last = (vec_state == AXI_IDLE) ? (HDR_00_CYCLES == 1)  : (cnt == HDR_00_CYCLES-2);
    3'b?01: is_last = (vec_state == AXI_IDLE) ? (HDR_01_CYCLES == 1)  : (cnt == HDR_01_CYCLES-2);
    3'b010: is_last = (vec_state == AXI_IDLE) ? (HDR_10_CYCLES == 1)  : (cnt == HDR_10_CYCLES-2);
    3'b?11: is_last = (vec_state == AXI_IDLE) ? (HDR_11_CYCLES == 1)  : (cnt == HDR_11_CYCLES-2);
    3'b100: is_last = (vec_state == AXI_IDLE) ? (HDR_100_CYCLES == 1) : (cnt == HDR_100_CYCLES-2);
  endcase
end

always_comb begin
  tkeep = '0;
  casez(i_header_type)
    3'b000: tkeep = HDR_00_TKEEP;
    3'b?01: tkeep = HDR_01_TKEEP;
    3'b010: tkeep = HDR_10_TKEEP;
    3'b?11: tkeep = HDR_11_TKEEP;
    3'b100: tkeep = HDR_100_TKEEP;
  endcase
end


always_comb begin
  hdr = '0;
  casez (i_header_type)
    3'b000: hdr = {'0,hdr_00};
    3'b?01: hdr = {'0,hdr_01};
    3'b010: hdr = {'0,hdr_10};
    3'b?11: hdr = {'0,hdr_11};
    3'b100: hdr = {'0,hdr_100};
  endcase
end

always_comb begin
  r_axis_tdata = '0;
  if (!data_is_zero) begin
    r_axis_tdata = hdr[W_DATA*cnt+:W_DATA];
  end
end

assign r_axis_tlast  = (vec_state == AXI_LAST);
assign r_axis_tvalid = (vec_state != AXI_IDLE);
assign r_axis_tkeep  = r_axis_tlast ? tkeep : '1;


axis_reg # (
  .DWIDTH             ( W_DATA + W_KEEP + 1                       ),
  .SKID               ( 1'b0                                      )
) u_axis_reg (
  .clk                ( i_clk                                     ),
  .rst                ( i_rst                                     ),
  .i_axis_rx_tvalid   ( r_axis_tvalid                             ),
  .i_axis_rx_tdata    ( {r_axis_tdata,r_axis_tlast,r_axis_tkeep}  ),
  .o_axis_rx_tready   ( r_axis_tready                             ),
  .o_axis_tx_tvalid   ( o_axis_tvalid                             ),
  .o_axis_tx_tdata    ( {o_axis_tdata,o_axis_tlast,o_axis_tkeep}  ),
  .i_axis_tx_tready   ( i_axis_tready                             )
);

assign o_axis_tuser  = 1'b0;
assign o_busy       = r_busy || o_axis_tvalid;

endmodule
