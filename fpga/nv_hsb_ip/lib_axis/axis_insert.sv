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

module axis_insert
#(
    parameter                               AXI_DWIDTH   = 64,
    parameter                               INSERT_OFFSET  = 16,
    parameter                               INSERT_WIDTH   = 32,
    localparam                              W_KEEP       = AXI_DWIDTH/8
)(
    input                                   clk,
    input                                   rst,

    input  logic  [INSERT_WIDTH-1:0]        i_insert_data,
    input  logic                            i_enable,

    input  logic                            i_axis_rx_tvalid,
    input  logic   [AXI_DWIDTH-1:0]         i_axis_rx_tdata,
    input  logic                            i_axis_rx_tlast,
    input  logic                            i_axis_rx_tuser,
    input  logic   [W_KEEP-1:0]             i_axis_rx_tkeep,
    output                                  o_axis_rx_tready,
  //AXIS Interface
    output  logic                           o_axis_tx_tvalid,
    output  logic   [AXI_DWIDTH-1:0]        o_axis_tx_tdata,
    output  logic                           o_axis_tx_tlast,
    output  logic                           o_axis_tx_tuser,
    output  logic   [W_KEEP-1:0]            o_axis_tx_tkeep,
    input                                   i_axis_tx_tready
);

localparam D_CYCLES                     = (INSERT_OFFSET-1)/AXI_DWIDTH + 1;
localparam I_CYCLES                     = (INSERT_WIDTH-1)/AXI_DWIDTH + 1;
localparam D_CNT_WIDTH                  = $clog2(D_CYCLES+1);
localparam I_CNT_WIDTH                  = $clog2(I_CYCLES+1);

localparam BIT_OFFSET                   = (INSERT_OFFSET%AXI_DWIDTH);


localparam INSERT_IS_FIRST_CYCLE        = (INSERT_OFFSET < AXI_DWIDTH);
localparam INSERT_IS_EVEN_FILL          = (((BIT_OFFSET + INSERT_WIDTH) % AXI_DWIDTH) == 0);
localparam INSERT_IS_EVEN_START         = (BIT_OFFSET == 0);
localparam INSERT_IS_ENCAPSULATED       = ((BIT_OFFSET + INSERT_WIDTH) < AXI_DWIDTH) && !INSERT_IS_EVEN_START;
localparam INSERT_IS_MULTI_CYCLE        = (I_CYCLES > 1);



localparam VAL_BITS                     = (INSERT_IS_EVEN_START)  ? 0 :
                                          (INSERT_IS_ENCAPSULATED ? (AXI_DWIDTH + INSERT_WIDTH - BIT_OFFSET) : (AXI_DWIDTH - BIT_OFFSET));
localparam LAST_BITS                    = (INSERT_WIDTH - ((I_CYCLES-1)*AXI_DWIDTH) - VAL_BITS);

localparam OFFSET_NEXT_CYCLE            = INSERT_IS_ENCAPSULATED ? (AXI_DWIDTH - INSERT_WIDTH) :
                                          (INSERT_IS_EVEN_FILL && !INSERT_IS_EVEN_START) ? BIT_OFFSET :
                                          ((INSERT_IS_MULTI_CYCLE && !INSERT_IS_EVEN_FILL) || (INSERT_IS_EVEN_START)) ? (BIT_OFFSET + INSERT_WIDTH) % AXI_DWIDTH : 0;

localparam INSERT_LAST_BYTES            = ((INSERT_WIDTH%AXI_DWIDTH) == 0) ? W_KEEP : ((INSERT_WIDTH%AXI_DWIDTH)/8);
localparam TLAST_VAL_BYTES              = (INSERT_IS_EVEN_START || INSERT_IS_ENCAPSULATED || (INSERT_IS_MULTI_CYCLE && !INSERT_IS_EVEN_FILL) ? INSERT_LAST_BYTES : (VAL_BITS/8));

localparam [W_KEEP-1:0] TKEEP_ADD = {'0,{TLAST_VAL_BYTES{1'b1}}};

logic [W_KEEP*2-1:0] tkeep_final;   //tkeep for final 2 cycles
logic [W_KEEP-1:0]   r_tkeep;

logic [D_CNT_WIDTH-1:0] d_cnt;
logic [I_CNT_WIDTH-1:0] i_cnt;

logic                 extra_cycle;
logic                 is_extra_cycle;
logic                 is_full_cycle;
logic                 pkt_active;
logic                 pkt_enable;
logic                 effective_enable;

//------------------------------------------------------------------------------------------------//
// Register Pipeline
//------------------------------------------------------------------------------------------------//

logic [AXI_DWIDTH-1:0] r_axis_tdata;

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
    r_axis_tdata <= '0;
  end
  else if (o_axis_rx_tready) begin
    r_axis_tdata <= i_axis_rx_tdata;
  end
end


//------------------------------------------------------------------------------------------------//
// Data Mux
//------------------------------------------------------------------------------------------------//

logic [AXI_DWIDTH-1:0] tdata;
logic                  tvalid;
logic                  tlast;
logic [W_KEEP-1:0]     tkeep;
logic                  tuser;

typedef enum logic [3:0] {
   S_DATA             = 4'h0,
   S_DATA_INSERT      = 4'h1,
   S_INSERT_FULL      = 4'h2,
   S_INSERT_LAST      = 4'h3,
   S_INSERT_DATA      = 4'h4,
   S_REG_DATA         = 4'h5,
   S_REG_PAD          = 4'h6,
   S_DATA_INSERT_DATA = 4'h7
} data_states;
data_states data_state;


logic [AXI_DWIDTH-1:0] d_data;
logic [AXI_DWIDTH-1:0] d_data_insert;
logic [AXI_DWIDTH-1:0] d_insert_full;
logic [AXI_DWIDTH-1:0] d_insert_last;
logic [AXI_DWIDTH-1:0] d_insert_data;
logic [AXI_DWIDTH-1:0] d_reg_data;
logic [AXI_DWIDTH-1:0] d_reg_pad;
logic [AXI_DWIDTH-1:0] d_data_insert_data;

generate
  
  if (OFFSET_NEXT_CYCLE == 0) begin
    assign d_reg_data       = r_axis_tdata;
  end else begin
    assign d_reg_data       = {i_axis_rx_tdata[OFFSET_NEXT_CYCLE-1:0], r_axis_tdata[AXI_DWIDTH-1:OFFSET_NEXT_CYCLE]};
  end
  assign d_reg_pad          = {'0, r_axis_tdata[AXI_DWIDTH-1:OFFSET_NEXT_CYCLE]};

  if ((INSERT_IS_EVEN_FILL || (INSERT_IS_MULTI_CYCLE && !INSERT_IS_EVEN_START)) && !INSERT_IS_EVEN_START) begin
    assign d_data_insert = {i_insert_data[VAL_BITS-1:0], i_axis_rx_tdata[BIT_OFFSET-1:0]};
  end else begin
    assign d_data_insert = d_data;
  end

  if (INSERT_IS_EVEN_START) begin
    assign d_insert_data = {i_axis_rx_tdata[LAST_BITS-1:0], i_insert_data[INSERT_WIDTH-1-:LAST_BITS]};
  end else begin
    assign d_insert_data = d_data;
  end

  if (INSERT_IS_MULTI_CYCLE || (INSERT_IS_EVEN_FILL && INSERT_IS_EVEN_START)) begin
    assign d_insert_full = {i_insert_data[VAL_BITS+(i_cnt*AXI_DWIDTH)+:AXI_DWIDTH]};
  end else begin
    assign d_insert_full = d_data;
  end

  if (INSERT_IS_MULTI_CYCLE && !INSERT_IS_EVEN_START && !INSERT_IS_EVEN_FILL) begin
    assign d_insert_last = {i_axis_rx_tdata[LAST_BITS-1:0], i_insert_data[INSERT_WIDTH-1-:LAST_BITS]};
  end else begin
    assign d_insert_last = d_data;
  end

  if (!INSERT_IS_MULTI_CYCLE && INSERT_IS_ENCAPSULATED) begin
    assign d_data_insert_data = {i_axis_rx_tdata[AXI_DWIDTH-INSERT_WIDTH-1:BIT_OFFSET],i_insert_data,i_axis_rx_tdata[BIT_OFFSET-1:0]};
  end else begin
    assign d_data_insert_data = d_data;
  end

  assign d_data             = (!effective_enable) ? i_axis_rx_tdata :
                              (INSERT_IS_FIRST_CYCLE) ? 
                                (INSERT_IS_ENCAPSULATED) ? d_data_insert_data : d_data_insert
                              : i_axis_rx_tdata;

endgenerate

always_comb begin
  case (data_state)
    S_DATA             : tdata = d_data;
    S_DATA_INSERT      : tdata = d_data_insert;
    S_INSERT_FULL      : tdata = d_insert_full;
    S_INSERT_LAST      : tdata = d_insert_last;
    S_INSERT_DATA      : tdata = d_insert_data;
    S_REG_DATA         : tdata = d_reg_data;
    S_REG_PAD          : tdata = d_reg_pad;
    S_DATA_INSERT_DATA : tdata = d_data_insert_data;
    default            : tdata = i_axis_rx_tdata;
  endcase
end



//------------------------------------------------------------------------------------------------//
// State
//------------------------------------------------------------------------------------------------//


logic is_last_d_cycle;

always_ff @(posedge clk) begin
  if (rst) begin
    data_state     <= S_DATA;
    d_cnt          <= '0;
    i_cnt          <= '0;
    r_tkeep        <= '0;
    is_extra_cycle <= '0;
    is_full_cycle  <= '0;
  end
  else begin
    if (!i_axis_tx_tready) begin
      data_state     <= data_state;
      d_cnt          <= d_cnt;
      i_cnt          <= i_cnt;
      r_tkeep        <= r_tkeep;
      is_extra_cycle <= is_extra_cycle;
      is_full_cycle  <= is_full_cycle;
    end
    else begin
      is_extra_cycle <= '0;
      is_full_cycle  <= '0;
      d_cnt          <= '0;
      i_cnt          <= '0;
      r_tkeep        <= tkeep_final[W_KEEP+:W_KEEP];
      case(data_state)
        S_DATA:   begin
          if (effective_enable) begin
            d_cnt         <= (is_last_d_cycle && i_axis_rx_tvalid) ? '0 : d_cnt + (i_axis_rx_tvalid);
            is_full_cycle <= is_last_d_cycle && i_axis_rx_tvalid && INSERT_IS_MULTI_CYCLE && INSERT_IS_EVEN_START;
            is_extra_cycle <= is_last_d_cycle && i_axis_rx_tvalid && INSERT_IS_FIRST_CYCLE &&
                              (INSERT_IS_EVEN_FILL || INSERT_IS_ENCAPSULATED) && i_axis_rx_tlast && extra_cycle;
            data_state    <= (is_last_d_cycle && i_axis_rx_tvalid) ? 
                              (INSERT_IS_FIRST_CYCLE) ? 
                                      (INSERT_IS_MULTI_CYCLE)                         ? S_INSERT_FULL : 
                                      (INSERT_IS_EVEN_FILL || INSERT_IS_ENCAPSULATED) ? 
                                        (i_axis_rx_tlast) ? ((extra_cycle) ? S_REG_PAD : S_DATA) : S_REG_DATA :
                                        S_INSERT_LAST :                                        
                              (INSERT_IS_EVEN_START)  ? (INSERT_IS_MULTI_CYCLE ? S_INSERT_FULL : S_INSERT_DATA) :
                              (INSERT_IS_ENCAPSULATED ? S_DATA_INSERT_DATA : S_DATA_INSERT) 
                                              : S_DATA;
          end
        end
        S_DATA_INSERT: begin
          is_full_cycle <= i_axis_rx_tvalid && INSERT_IS_MULTI_CYCLE;
          data_state    <= !i_axis_rx_tvalid ? S_DATA_INSERT :
                           (INSERT_IS_MULTI_CYCLE) ? S_INSERT_FULL : 
                           (!INSERT_IS_EVEN_FILL) ? S_INSERT_LAST : S_REG_DATA;
        end
        S_INSERT_FULL: begin
          is_full_cycle <= !(i_cnt == I_CYCLES-2);
          i_cnt         <= (i_cnt == I_CYCLES-2) ? '0 : i_cnt + 1;
          data_state    <= (i_cnt == I_CYCLES-2) ? 
                            (INSERT_IS_EVEN_START) ? S_INSERT_DATA :
                            (INSERT_IS_EVEN_FILL)  ? S_REG_DATA    : S_INSERT_LAST :
                                                    S_INSERT_FULL;
        end
        S_INSERT_DATA: begin
          data_state <= S_REG_DATA;
        end
        S_INSERT_LAST: begin
          data_state <= S_REG_DATA;
        end
        S_REG_DATA: begin
          data_state     <= (i_axis_rx_tlast) ? ((extra_cycle) ? S_REG_PAD : S_DATA) : S_REG_DATA;
          is_extra_cycle <= i_axis_rx_tlast && extra_cycle;
        end
        S_REG_PAD: begin
          data_state <= S_DATA;
        end
        S_DATA_INSERT_DATA: begin
          data_state <= (i_axis_rx_tvalid) ? S_REG_DATA : S_DATA_INSERT_DATA;
        end
        default: begin
          data_state <= S_DATA;
        end
      endcase
    end
  end
end

assign tkeep_final = (TKEEP_ADD | (i_axis_rx_tkeep << TLAST_VAL_BYTES));
assign extra_cycle = tkeep_final[W_KEEP];

assign is_last_d_cycle = INSERT_IS_FIRST_CYCLE ? 1'b1 :
                         INSERT_IS_EVEN_START  ? (d_cnt == D_CYCLES-1) :
                                                 (d_cnt == D_CYCLES-2);
assign effective_enable = pkt_active ? pkt_enable : i_enable;

assign tlast  = is_extra_cycle  ? 1'b1 :
                !effective_enable ? i_axis_rx_tlast :
                                  (i_axis_rx_tlast && !extra_cycle);
                                  
assign tvalid = i_axis_rx_tvalid || is_extra_cycle;
assign tkeep  = is_extra_cycle  ? r_tkeep                : 
                !effective_enable ? i_axis_rx_tkeep      :
                i_axis_rx_tlast ? tkeep_final[0+:W_KEEP] : '1;
assign tuser  = i_axis_rx_tuser;

assign o_axis_tx_tdata  = tdata;
assign o_axis_tx_tlast  = tlast;
assign o_axis_tx_tvalid = tvalid;
assign o_axis_tx_tkeep  = tkeep;
assign o_axis_tx_tuser  = tuser;
assign o_axis_rx_tready = i_axis_tx_tready && !is_extra_cycle && !is_full_cycle;

endmodule
