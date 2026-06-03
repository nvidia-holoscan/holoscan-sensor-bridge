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

module ctrl_bus_evt_int
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter         AXI_DWIDTH = 64,
  parameter         NUM_HOST   = 1,
  parameter         W_EVENT    = 32,
  parameter         SYNC_CLK   = 0,
  localparam        AXI_KWIDTH = AXI_DWIDTH/8,
  localparam        W_NUM_HOST = (NUM_HOST==1) ? 1 :$clog2(NUM_HOST)
)(
  input                    pclk,
  input                    rst,
  // Register Map, abp clk domain
  input                    aclk,
  input                    arst,
  input  apb_m2s           i_apb_m2s,
  output apb_s2m           o_apb_s2m,

  input  apb_m2s           i_apb_m2s_ram,
  output apb_s2m           o_apb_s2m_ram,

  output  apb_m2s          o_apb_m2s,
  input   apb_s2m          i_apb_s2m,

  input  [79:0]            i_ptp,

  input  [W_EVENT-1:0]     evt_vec,
  // Ethernet Address
  output [47:0]            o_host_mac_addr,
  output [31:0]            o_host_ip_addr,
  output [15:0]            o_host_udp_port,
  output [15:0]            o_dev_udp_port,
  output [15:0]            o_pld_len,
  //AXIS Interface
  output                   o_int_axis_tx_tvalid,
  output [AXI_DWIDTH-1:0]  o_int_axis_tx_tdata,
  output                   o_int_axis_tx_tlast,
  output [W_NUM_HOST-1:0]  o_int_axis_tx_tuser,
  output [AXI_KWIDTH-1:0]  o_int_axis_tx_tkeep,
  input                    i_int_axis_tx_tready
);


logic [W_NUM_HOST-1:0] host_idx;
localparam W_EVENT_IDX = 5;

logic sw_event;
logic fsm_timeout;
logic fsm_error;
logic fsm_evt;

always_ff @(posedge aclk) begin
  if (arst) begin
    fsm_evt <= '0;
  end
  else begin
    fsm_evt <= fsm_timeout || fsm_error;
  end
end

//------------------------------------------------------------------------------------------------//
// Data Sync
//------------------------------------------------------------------------------------------------//

logic [W_EVENT-1:0]    evt_vec_sync;
logic [W_EVENT-1:0]    w_evt_vec;

assign w_evt_vec[1:0]  = evt_vec[1:0];
assign w_evt_vec[2]    = sw_event;
assign w_evt_vec[3]    = fsm_evt;
assign w_evt_vec[31:4] = evt_vec[31:4];

data_sync    #(
  .DATA_WIDTH ( W_EVENT         )
) data_glitch_filter (
  .clk        ( pclk            ),
  .rst_n      ( !rst            ),
  .sync_in    ( w_evt_vec       ),
  .sync_out   ( evt_vec_sync    )
);

//------------------------------------------------------------------------------------------------//
// Register Map
//------------------------------------------------------------------------------------------------//

logic [31:0] ctrl_reg [ctrl_evt_nctrl];
logic [31:0] stat_reg [ctrl_evt_nstat];

s_apb_reg #(
  .N_CTRL     ( ctrl_evt_nctrl ),
  .N_STAT     ( ctrl_evt_nstat ),
  .W_OFST     ( w_ofst         ),
  .SYNC_CLK   ( SYNC_CLK       )
) u_reg_map   (
  // APB Interface
  .i_aclk     ( aclk           ),
  .i_arst     ( arst           ),
  .i_apb_m2s  ( i_apb_m2s      ),
  .o_apb_s2m  ( o_apb_s2m      ),
  // User Control Signals
  .i_pclk     ( pclk           ),
  .i_prst     ( rst            ),
  .o_ctrl     ( ctrl_reg       ),
  .i_stat     ( stat_reg       )
);

logic [W_EVENT-1:0]  falling_mask;
logic [W_EVENT-1:0]  rising_mask;
logic [W_EVENT-1:0]  clear_event_mask;
logic [W_EVENT-1:0]  clear_event_mask_q;
logic [W_EVENT-1:0]  int_active;
logic [31:0]         apb_interrupt_en;
logic [23:0]         timeout;
logic                fsm_timeout_sync;
logic                fsm_error_sync;

logic [W_EVENT_IDX-1:0]  evt_vec_timeout;
logic [W_EVENT_IDX-1:0]  evt_vec_timeout_sync;


assign o_pld_len = 18;

// Ethernet address
assign o_host_mac_addr = {ctrl_reg[ctrl_evt_host_mac_addr_hi][15:0], ctrl_reg[ctrl_evt_host_mac_addr_lo]};
assign o_host_ip_addr  =  ctrl_reg[ctrl_evt_host_ip_addr];
assign o_host_udp_port =  ctrl_reg[ctrl_evt_host_udp_port];
assign o_dev_udp_port  =  ctrl_reg[ctrl_evt_dev_udp_port];

assign falling_mask = ctrl_reg[ctrl_evt_falling];
assign rising_mask = ctrl_reg[ctrl_evt_rising];
assign clear_event_mask = ctrl_reg[ctrl_evt_clear];
assign timeout = ctrl_reg[ctrl_evt_apb_timeout][23:0];

assign apb_interrupt_en = ctrl_reg[ctrl_evt_apb_interrupt_en];
assign sw_event = ctrl_reg[ctrl_evt_sw_event][0];

assign stat_reg[0] = int_active[31:0];
assign stat_reg[1][W_EVENT_IDX-1:0] = evt_vec_timeout_sync;
assign stat_reg[1][31:28] = {2'b0, fsm_timeout_sync, fsm_error_sync};
assign stat_reg[1][27:5] = '0;


data_sync #(
  .DATA_WIDTH ( 2 )
) fsm_timeout_sync_inst (
  .clk        ( pclk                              ),
  .rst_n      ( !rst                              ),
  .sync_in    ( {fsm_timeout, fsm_error}          ),
  .sync_out   ( {fsm_timeout_sync, fsm_error_sync} )
);

//------------------------------------------------------------------------------------------------//
// Int Detect
//------------------------------------------------------------------------------------------------//

typedef enum logic [1:0] {
  IDLE   =  2'h0,
  TRIG   =  2'h1,
  XFER   =  2'h2
} evt_axi_state;

evt_axi_state        evt_state;
logic  [W_EVENT-1:0] int_det;
logic  [W_EVENT-1:0] evt_vec_q;
logic  [W_EVENT-1:0] int_det_apb;

integer i;

always_ff @(posedge pclk) begin
  if (rst) begin
    evt_vec_q               <= '0;
    int_det                 <= '0;
    int_active              <= '0;
    clear_event_mask_q      <= '0;
    int_det_apb             <= '0;
  end
  else begin
    evt_vec_q               <= evt_vec_sync;
    clear_event_mask_q      <= clear_event_mask;
    int_det_apb             <= int_det & apb_interrupt_en;
    for (i=0;i<W_EVENT;i=i+1) begin
      if (!int_active[i]) begin
        if (rising_mask[i] && falling_mask[i]) begin
          int_det[i]    <= (evt_vec_sync[i] != evt_vec_q[i]);
          int_active[i] <= (evt_vec_sync[i] != evt_vec_q[i]);
        end
        else if (rising_mask[i]) begin
          int_det[i]    <= (evt_vec_sync[i] && !evt_vec_q[i]);
          int_active[i] <= (evt_vec_sync[i] && !evt_vec_q[i]);
        end
        else if (falling_mask[i]) begin
          int_det[i]    <= (!evt_vec_sync[i] && evt_vec_q[i]);
          int_active[i] <= (!evt_vec_sync[i] && evt_vec_q[i]);
        end
        else begin
          int_det[i]    <= '0;
          int_active[i] <= int_active[i];
        end
      end
      else begin
        int_det[i]    <= '0;
        int_active[i] <= (clear_event_mask[i] && !clear_event_mask_q[i]) ? '0 : int_active[i];
      end
    end
  end
end

//------------------------------------------------------------------------------------------------//
// APB Interrupt
//------------------------------------------------------------------------------------------------//

logic [23:0]        timeout_sync;
logic [W_EVENT-1:0] int_det_sync;

genvar j;
generate
  for (j=0; j<W_EVENT; j++) begin : gen_int_det_sync
    pulse_sync int_det_sync_inst (
      .src_clk    ( pclk             ),
      .src_rst    ( rst              ),
      .dst_clk    ( aclk             ),
      .dst_rst    ( arst             ),
      .i_src_pulse( int_det_apb[j]   ),
      .o_dst_pulse( int_det_sync[j]  )
    );
  end
endgenerate

data_sync #(
  .DATA_WIDTH ( 24 )
) timeout_sync_inst (
  .clk        ( aclk          ),
  .rst_n      ( !arst         ),
  .sync_in    ( timeout       ),
  .sync_out   ( timeout_sync  )
);

data_sync #(
  .DATA_WIDTH ( W_EVENT_IDX            )
) evt_vec_timeout_sync_inst (
  .clk        ( pclk                  ),
  .rst_n      ( !rst                  ),
  .sync_in    ( evt_vec_timeout       ),
  .sync_out   ( evt_vec_timeout_sync  )
);

ctrl_bus_evt_fsm #(
  .W_EVENT    ( W_EVENT        )
) u_ctrl_bus_evt_fsm (
  .i_aclk     ( aclk           ),
  .i_arst     ( arst           ),
  .i_apb_m2s  ( i_apb_m2s_ram  ),
  .o_apb_s2m  ( o_apb_s2m_ram  ),
  .i_apb_s2m  ( i_apb_s2m      ),
  .o_apb_m2s  ( o_apb_m2s      ),
  .i_evt_vec  ( int_det_sync   ),
  .i_timeout  ( timeout_sync   ),
  .o_timeout  ( fsm_timeout    ),
  .o_evt_vec  ( evt_vec_timeout),
  .o_error    ( fsm_error      )
);


//------------------------------------------------------------------------------------------------//
// FIFO
//------------------------------------------------------------------------------------------------//

localparam W_FIFO = W_EVENT + W_EVENT+64;

logic              evt_fifo_wren;
logic [W_FIFO-1:0] evt_fifo_wrdata;
logic              evt_fifo_full;
logic              evt_fifo_afull;
logic              evt_fifo_over;
logic              evt_fifo_rden;
logic [W_FIFO-1:0] evt_fifo_rddata;
logic              evt_fifo_rdval;
logic              evt_fifo_empty;
logic              evt_fifo_aempty;
logic              evt_fifo_under;

logic [W_FIFO-1:0]    int_det_fifo_q;
logic                 trigger;
logic                 axis_is_busy;
logic [W_FIFO+16-1:0] axis_data;
logic [W_FIFO+16-1:0] axis_data_be;

sc_fifo #(
  .DATA_WIDTH ( W_FIFO      ),
  .FIFO_DEPTH ( W_EVENT     ),
  .MEM_STYLE  ( "BLOCK"     )
) u_ctrl_evt_fifo (
  .clk   ( pclk             ),
  .rst   ( rst              ),
  .wr    ( evt_fifo_wren    ),
  .din   ( evt_fifo_wrdata  ),
  .full  ( evt_fifo_full    ),
  .afull ( evt_fifo_afull   ),
  .over  ( evt_fifo_over    ),
  .rd    ( evt_fifo_rden    ),
  .dout  ( evt_fifo_rddata  ),
  .dval  ( evt_fifo_rdval   ),
  .empty ( evt_fifo_empty   ),
  .aempty( evt_fifo_aempty  ),
  .under ( evt_fifo_under   ),
  .count (                  )
);

assign evt_fifo_wren        = |int_det && ~evt_fifo_full;
assign evt_fifo_wrdata      = {int_active, evt_vec_q,i_ptp[63:0]};


genvar k;
generate
  for (k=0; k<(W_FIFO+16)/8; k++) begin : gen_axis_data_align
    assign axis_data_be[k*8+:8] = axis_data[((W_FIFO+16)/8-1-k)*8+:8];
  end
endgenerate

logic        ptp_fifo_wren;
assign ptp_fifo_wren = evt_fifo_wren;

//------------------------------------------------------------------------------------------------//
// FSM
//------------------------------------------------------------------------------------------------//


always_ff @(posedge pclk) begin
  if (rst) begin
    evt_state               <= IDLE;
    host_idx                <= '0;
    evt_fifo_rden           <= 0;
  end
  else begin
    case(evt_state)
      IDLE:   begin
        evt_state     <= (!evt_fifo_empty && !evt_fifo_rden) ? TRIG : IDLE;
        evt_fifo_rden <= (!evt_fifo_empty && !evt_fifo_rden);
      end
      TRIG: begin
        evt_state     <= XFER;
        evt_fifo_rden <= '0;
      end
      XFER: begin
        if (!axis_is_busy) begin
          if (host_idx < (NUM_HOST-1)) begin
            evt_state    <= TRIG;
            host_idx     <= host_idx + 1'b1;
          end
          else begin
            evt_state <= IDLE;
            host_idx  <= '0;
          end
        end
      end
      default: begin
        evt_state    <= IDLE;
      end
    endcase
  end
end


//------------------------------------------------------------------------------------------------//
// Output
//------------------------------------------------------------------------------------------------//

vec_to_axis #(
  .AXI_DWIDTH       ( AXI_DWIDTH           ),
  .DATA_WIDTH       ( W_FIFO+16            ),
  .PADDED_WIDTH     ( 18*8                 ),
  .REG_DATA         ( 0                    )
) int_to_axis (
  .clk              ( pclk                 ),
  .rst              ( rst                  ),
  .trigger          ( trigger              ),
  .data             ( axis_data_be         ),
  .is_busy          ( axis_is_busy         ),
  .o_axis_tx_tvalid ( o_int_axis_tx_tvalid ),
  .o_axis_tx_tdata  ( o_int_axis_tx_tdata  ),
  .o_axis_tx_tlast  ( o_int_axis_tx_tlast  ),
  .o_axis_tx_tuser  (                      ),
  .o_axis_tx_tkeep  ( o_int_axis_tx_tkeep  ),
  .i_axis_tx_tready ( i_int_axis_tx_tready )
);

assign trigger = (evt_state == TRIG);
assign axis_data = {evt_fifo_rddata[W_FIFO-1:64],16'h0,evt_fifo_rddata[63:0]};
assign o_int_axis_tx_tuser = host_idx;


endmodule
