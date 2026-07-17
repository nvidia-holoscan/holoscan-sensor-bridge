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

module axis_remove
#(
    parameter                               AXI_DWIDTH   = 64,
    parameter                               REMOVE_OFFSET  = 16,
    parameter                               REMOVE_WIDTH   = 32,
    parameter                               W_USER       = 4,
    localparam                              W_KEEP       = AXI_DWIDTH/8
)(
    input                                   clk,
    input                                   rst,

    input  logic                            i_enable,
    output logic   [REMOVE_WIDTH-1:0]       o_removed_data,
    output logic                            o_gated_enable,


    input  logic                            i_axis_rx_tvalid,
    input  logic   [AXI_DWIDTH-1:0]         i_axis_rx_tdata,
    input  logic                            i_axis_rx_tlast,
    input  logic   [W_USER-1:0]             i_axis_rx_tuser,
    input  logic   [(AXI_DWIDTH/8)-1:0]     i_axis_rx_tkeep,
    output                                  o_axis_rx_tready,
  //AXIS Interface
    output  logic                           o_axis_tx_tvalid,
    output  logic   [AXI_DWIDTH-1:0]        o_axis_tx_tdata,
    output  logic                           o_axis_tx_tlast,
    output  logic   [W_USER-1:0]            o_axis_tx_tuser,
    output  logic   [(AXI_DWIDTH/8)-1:0]    o_axis_tx_tkeep,
    input                                   i_axis_tx_tready
);

localparam BIT_OFFSET                   = (REMOVE_OFFSET%AXI_DWIDTH);

localparam D_CYCLES                     = (REMOVE_OFFSET-1)/AXI_DWIDTH + 1;
localparam R_CYCLES                     = (BIT_OFFSET + REMOVE_WIDTH-1)/AXI_DWIDTH + 1;
localparam D_CNT_WIDTH                  = $clog2(D_CYCLES+1);
localparam R_CNT_WIDTH                  = $clog2(R_CYCLES+1);

localparam BIT_OFFSET_NEXT_CYCLE        = (BIT_OFFSET + REMOVE_WIDTH) % AXI_DWIDTH;
localparam VAL_BITS_SECOND_HALF         = REMOVE_WIDTH % AXI_DWIDTH;

localparam REMOVE_IS_EVEN_FILL           = (((BIT_OFFSET + REMOVE_WIDTH) % AXI_DWIDTH) == 0);
localparam REMOVE_IS_EVEN_START          = (BIT_OFFSET == 0);
localparam REMOVE_IS_ENCAPSULATED        = ((BIT_OFFSET + REMOVE_WIDTH) < AXI_DWIDTH) && !REMOVE_IS_EVEN_START;
localparam REMOVE_IS_MULTI_CYCLE         = (R_CYCLES > 1);
localparam REMOVE_IS_FIRST_CYCLE         = (REMOVE_OFFSET < AXI_DWIDTH);
localparam REMOVE_FITS_IN_FIRST_CYCLE    = REMOVE_IS_FIRST_CYCLE && !REMOVE_IS_MULTI_CYCLE;
localparam REMOVE_ENDS_ON_CYCLE_BOUNDARY = (VAL_BITS_SECOND_HALF == 0);

localparam VAL_BITS_NEXT_CYCLE           = AXI_DWIDTH - VAL_BITS_SECOND_HALF;

localparam VAL_BITS                     = (AXI_DWIDTH - BIT_OFFSET);

localparam BIT_OFFSET_FIRST_SECOND_HALF = (REMOVE_IS_EVEN_START) ? AXI_DWIDTH-BIT_OFFSET : AXI_DWIDTH - BIT_OFFSET - REMOVE_WIDTH;

localparam REMOVE_LAST_BYTES            = ((REMOVE_WIDTH%AXI_DWIDTH) == 0) ? W_KEEP : ((REMOVE_WIDTH%AXI_DWIDTH)/8);
localparam TLAST_VAL_BYTES              = REMOVE_LAST_BYTES;



logic [W_KEEP*2-1:0] tkeep_final;   //tkeep for final 2 cycles
logic [W_KEEP-1:0]   r_tkeep;

logic [D_CNT_WIDTH-1:0] d_cnt;
logic [R_CNT_WIDTH-1:0] r_cnt;

logic                 extra_cycle;
logic                 is_extra_cycle;
logic                 is_fill_first_cycle_remove;
logic                 is_fill_last;
logic                 fill_last_pending;
logic                 capture_removed_data;
logic                 pkt_active;
logic                 pkt_enable;
logic                 effective_enable;
logic                 pending_output;

logic [REMOVE_WIDTH-1:0] r_removed_data;
logic [W_USER-1:0]       r_axis_tuser;
logic [W_KEEP-1:0]       r_axis_tkeep;



//------------------------------------------------------------------------------------------------//
// Register Pipeline
//------------------------------------------------------------------------------------------------//

logic [AXI_DWIDTH-1:0] r_axis_tdata [R_CYCLES-1:0];
logic [R_CYCLES*AXI_DWIDTH-1:0] r_axis_tdata_flat;

integer i;
always_ff @(posedge clk) begin
  if (rst) begin
    pkt_active <= '0;
    pkt_enable <= '0;
  end
  else begin
    if (!pkt_active && i_axis_rx_tvalid) begin
      pkt_active <= !(o_axis_rx_tready && i_axis_rx_tlast);
      pkt_enable <= i_enable;
    end
    else if (pkt_active && i_axis_rx_tvalid && o_axis_rx_tready && i_axis_rx_tlast) begin
      pkt_active <= '0;
    end
  end
end

always_ff @(posedge clk) begin
  if (rst) begin
    r_axis_tdata[0] <= '0;
    r_axis_tuser    <= '0;
    r_axis_tkeep    <= '0;
    for (i = 1; i < R_CYCLES; i = i + 1) begin
      r_axis_tdata[i] <= '0;
    end
  end
  else if (i_axis_tx_tready && i_axis_rx_tvalid) begin
    r_axis_tdata[0] <= i_axis_rx_tdata;
    r_axis_tuser    <= i_axis_rx_tuser;
    r_axis_tkeep    <= i_axis_rx_tkeep;
    for (i = 1; i < R_CYCLES; i = i + 1) begin
      r_axis_tdata[i] <= r_axis_tdata[i-1];
    end
  end
end

always_comb begin
  for (int k = 0; k < R_CYCLES; k = k + 1) begin
    r_axis_tdata_flat[k*AXI_DWIDTH+:AXI_DWIDTH] = r_axis_tdata[R_CYCLES-1-k];
  end
end


//------------------------------------------------------------------------------------------------//
// Data Mux
//------------------------------------------------------------------------------------------------//

logic [AXI_DWIDTH-1:0] tdata;
logic                  tvalid;
logic                  tlast;
logic [W_KEEP-1:0]     tkeep;
logic [W_USER-1:0]     tuser;

typedef enum logic [3:0] {
   S_REG_FILL        = 4'h0,
   S_REG             = 4'h1,
   S_REG_FIRST       = 4'h2,
   S_REG_FIRST_ENCAP = 4'h3,
   S_REG_STEADY      = 4'h4,
   S_REG_EMPTY       = 4'h5,
   S_REG_LAST        = 4'h6
} data_states;
data_states data_state;

logic [AXI_DWIDTH-1:0] d_idle;
logic [AXI_DWIDTH-1:0] d_reg_fill;
logic [AXI_DWIDTH-1:0] d_reg;
logic [AXI_DWIDTH-1:0] d_reg_first;
logic [AXI_DWIDTH-1:0] d_reg_first_encap;
logic [AXI_DWIDTH-1:0] d_reg_steady;
logic [AXI_DWIDTH-1:0] d_reg_empty;
logic [AXI_DWIDTH-1:0] d_reg_last;

generate

  assign d_idle             = i_axis_rx_tdata;
  if (REMOVE_IS_FIRST_CYCLE && REMOVE_IS_ENCAPSULATED && !REMOVE_IS_MULTI_CYCLE) begin
    assign d_reg_fill       = {{VAL_BITS_SECOND_HALF{1'b0}}, r_axis_tdata[0][AXI_DWIDTH-1-:BIT_OFFSET_FIRST_SECOND_HALF], r_axis_tdata[R_CYCLES-1][0+:BIT_OFFSET]};
  end else begin
    assign d_reg_fill       = r_axis_tdata[R_CYCLES-1];
  end
  
  if (REMOVE_ENDS_ON_CYCLE_BOUNDARY && REMOVE_IS_EVEN_START) begin
    assign d_reg_steady     = i_axis_rx_tdata;
  end else if (REMOVE_ENDS_ON_CYCLE_BOUNDARY) begin
    assign d_reg_steady     = r_axis_tdata[0][VAL_BITS_SECOND_HALF+:VAL_BITS_NEXT_CYCLE];
  end else begin
    assign d_reg_steady     = {i_axis_rx_tdata[0+:VAL_BITS_SECOND_HALF], r_axis_tdata[0][VAL_BITS_SECOND_HALF+:VAL_BITS_NEXT_CYCLE]};
  end

  assign d_reg_empty        = r_axis_tdata[R_CYCLES-1];
  assign d_reg_last         = {'0,r_axis_tdata[0][VAL_BITS_SECOND_HALF+:VAL_BITS_NEXT_CYCLE]};

  if (REMOVE_IS_EVEN_START) begin
    assign d_reg_first        = d_reg_steady;
  end else if (REMOVE_IS_EVEN_FILL && !REMOVE_IS_MULTI_CYCLE) begin
    assign d_reg_first        = {i_axis_rx_tdata[BIT_OFFSET_NEXT_CYCLE+:VAL_BITS_SECOND_HALF], r_axis_tdata[R_CYCLES-1][0+:BIT_OFFSET]};
  end else begin
    assign d_reg_first = d_reg;
  end

  if (REMOVE_IS_ENCAPSULATED && !REMOVE_IS_MULTI_CYCLE) begin
    assign d_reg_first_encap  = {i_axis_rx_tdata[0+:VAL_BITS_SECOND_HALF], r_axis_tdata[0][AXI_DWIDTH-1-:BIT_OFFSET_FIRST_SECOND_HALF], r_axis_tdata[R_CYCLES-1][0+:BIT_OFFSET]};
  end else begin
    assign d_reg_first_encap = d_reg;
  end

  assign d_reg              = (REMOVE_IS_FIRST_CYCLE) ? 
                               (REMOVE_IS_ENCAPSULATED) ? d_reg_first_encap : d_reg_first 
                              : r_axis_tdata[R_CYCLES-1];

endgenerate

always_comb begin
  case (data_state)
    S_REG_FILL        : tdata = (effective_enable || fill_last_pending) ? d_reg_fill : d_idle;
    S_REG             : tdata = d_reg;
    S_REG_FIRST       : tdata = d_reg_first;
    S_REG_FIRST_ENCAP : tdata = d_reg_first_encap;
    S_REG_STEADY      : tdata = d_reg_steady;
    S_REG_EMPTY       : tdata = d_reg_empty;
    S_REG_LAST        : tdata = (effective_enable || pending_output) ? d_reg_last : d_idle;
    default           : tdata = d_reg_fill;
  endcase
end

//------------------------------------------------------------------------------------------------//
// Latched Removed Data
//------------------------------------------------------------------------------------------------//

generate
  if (REMOVE_FITS_IN_FIRST_CYCLE) begin
    always_ff @(posedge clk) begin
      if (rst) begin
        r_removed_data <= '0;
      end
      else if (i_axis_tx_tready) begin
        if (effective_enable && is_fill_first_cycle_remove) begin
          r_removed_data <= i_axis_rx_tdata[BIT_OFFSET+:REMOVE_WIDTH];
        end else if (capture_removed_data || (REMOVE_IS_FIRST_CYCLE && REMOVE_IS_MULTI_CYCLE && data_state == S_REG && i_axis_rx_tvalid)) begin
          r_removed_data <= r_axis_tdata_flat[BIT_OFFSET+:REMOVE_WIDTH];
        end else if (!effective_enable) 
        begin
          r_removed_data <= '0;
        end
      end
    end
  end
  else begin
    always_ff @(posedge clk) begin
      if (rst) begin
        r_removed_data <= '0;
      end
      else if (i_axis_tx_tready) begin
        if (capture_removed_data || (REMOVE_IS_FIRST_CYCLE && REMOVE_IS_MULTI_CYCLE && data_state == S_REG && i_axis_rx_tvalid)) begin
          r_removed_data <= r_axis_tdata_flat[BIT_OFFSET+:REMOVE_WIDTH];
        end
        else if (!effective_enable) 
        begin
          r_removed_data <= '0;
        end
      end
    end
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// State
//------------------------------------------------------------------------------------------------//

logic is_last_d_cycle;

always_ff @(posedge clk) begin
  if (rst) begin
    data_state     <= S_REG_FILL;
    d_cnt          <= '0;
    r_cnt          <= '0;
    r_tkeep        <= '0;
    is_extra_cycle <= '0;
    fill_last_pending <= '0;
    capture_removed_data <= '0;
  end
  else begin
    if (!i_axis_tx_tready) begin
      data_state     <= data_state;
      d_cnt          <= d_cnt;
      r_cnt          <= r_cnt;
      r_tkeep        <= r_tkeep;
      is_extra_cycle <= is_extra_cycle;
      fill_last_pending <= fill_last_pending;
      capture_removed_data <= capture_removed_data;
    end
    else begin
      is_extra_cycle <= '0;
      fill_last_pending <= '0;
      capture_removed_data <= '0;
      r_tkeep        <= tkeep_final[W_KEEP+:W_KEEP];
      d_cnt          <= '0;
      r_cnt          <= '0;
      case(data_state)
        S_REG_FILL:   begin
          if (effective_enable) begin
            r_cnt      <= ((r_cnt == R_CYCLES-1) && i_axis_rx_tvalid) ? '0 : r_cnt + (i_axis_rx_tvalid);
            fill_last_pending <= is_fill_last;
            data_state <= is_fill_last ? S_REG_FILL :
                          ((r_cnt == R_CYCLES-1) && (i_axis_rx_tvalid)) ? S_REG : S_REG_FILL;
          end else begin
            data_state <= S_REG_FILL;
          end
        end
        S_REG: begin
          d_cnt      <= (is_last_d_cycle && i_axis_rx_tvalid) ? '0 : d_cnt + (i_axis_rx_tvalid);
          data_state <= (is_last_d_cycle && (i_axis_rx_tvalid)) ? 
                        (REMOVE_IS_FIRST_CYCLE) ? ((i_axis_rx_tlast) ? S_REG_LAST : S_REG_STEADY) : 
                        (REMOVE_IS_EVEN_START ? S_REG_STEADY : (REMOVE_IS_ENCAPSULATED ? S_REG_FIRST_ENCAP : S_REG_FIRST)) : S_REG;
          is_extra_cycle <= REMOVE_IS_FIRST_CYCLE && i_axis_rx_tlast && extra_cycle;
          capture_removed_data <= !REMOVE_IS_FIRST_CYCLE && is_last_d_cycle && i_axis_rx_tvalid;
        end
        S_REG_FIRST: begin
          data_state     <= i_axis_rx_tvalid ? S_REG_STEADY : S_REG_FIRST;
        end
        S_REG_FIRST_ENCAP: begin
          data_state     <= i_axis_rx_tvalid ? S_REG_STEADY : S_REG_FIRST_ENCAP;
        end
        S_REG_STEADY: begin
          data_state     <= (i_axis_rx_tvalid && i_axis_rx_tlast) ? 
                            ((extra_cycle) ? S_REG_LAST : S_REG_FILL) : S_REG_STEADY;
          is_extra_cycle <= i_axis_rx_tvalid && i_axis_rx_tlast && extra_cycle;
        end 
        S_REG_LAST: begin
          if (effective_enable) begin
            r_cnt      <= ((r_cnt == R_CYCLES-1) && i_axis_rx_tvalid) ? '0 : r_cnt + (i_axis_rx_tvalid);
            fill_last_pending <= is_fill_last;
            data_state <= is_fill_last ? S_REG_FILL :
                          ((r_cnt == R_CYCLES-1) && (i_axis_rx_tvalid)) ? S_REG : S_REG_FILL;
          end else begin
            data_state <= S_REG_FILL;
          end
        end
        default: begin
          data_state <= S_REG_FILL;
        end
      endcase
    end
  end
end

assign tkeep_final = ({i_axis_rx_tkeep,{W_KEEP{1'b1}}} >> TLAST_VAL_BYTES);
assign extra_cycle = tkeep_final[W_KEEP];

assign is_last_d_cycle = REMOVE_IS_FIRST_CYCLE ? 1'b1 :
                         REMOVE_IS_EVEN_START  ? (d_cnt == D_CYCLES-1) :
                                                 (d_cnt == D_CYCLES-2);
assign is_fill_first_cycle_remove = ((data_state == S_REG_FILL) || (data_state == S_REG_LAST)) &&
                                    (r_cnt == R_CYCLES-1) && i_axis_rx_tvalid;

assign is_fill_last = REMOVE_FITS_IN_FIRST_CYCLE && is_fill_first_cycle_remove && i_axis_rx_tlast;

assign effective_enable = pkt_active ? pkt_enable : i_enable;
assign pending_output = fill_last_pending || is_extra_cycle;

assign tlast  = pending_output     ? 1'b1 :
                !effective_enable ? i_axis_rx_tlast :
                                    (i_axis_rx_tlast && !extra_cycle);
assign tvalid = (!effective_enable && i_axis_rx_tvalid && !pending_output) ||
                (effective_enable && i_axis_rx_tvalid && (data_state != S_REG_FILL) && !((data_state==S_REG_LAST) && extra_cycle)) ||
                pending_output || (data_state == S_REG_LAST && is_extra_cycle);
assign tkeep  = fill_last_pending ? (r_axis_tkeep >> TLAST_VAL_BYTES) :
                is_extra_cycle    ? r_tkeep                :
                !effective_enable ? i_axis_rx_tkeep        :
                i_axis_rx_tlast   ? tkeep_final[0+:W_KEEP] : '1;
assign tuser  = (fill_last_pending || is_extra_cycle) ? r_axis_tuser : i_axis_rx_tuser;

assign o_axis_tx_tdata  = tdata;
assign o_axis_tx_tlast  = tlast;
assign o_axis_tx_tvalid = tvalid;
assign o_axis_tx_tkeep  = tkeep;
assign o_axis_tx_tuser  = tuser;
assign o_axis_rx_tready = i_axis_tx_tready && (effective_enable || !pending_output);

assign o_removed_data = r_removed_data;
assign o_gated_enable = effective_enable;

endmodule

