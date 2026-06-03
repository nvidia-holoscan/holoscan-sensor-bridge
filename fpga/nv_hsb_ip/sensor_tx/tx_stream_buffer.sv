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


module tx_stream_buffer
  import apb_pkg::*;
  import regmap_pkg::*;
  import axis_pkg::*;
#(
  parameter   HOST_DWIDTH       = 512,
  parameter   NUM_HOSTS         = 1,
  parameter   SENSOR_DWIDTH     = 128,
  parameter   FIFO_DEPTH        = 8192,
  localparam  HOST_KEEP_WIDTH   = (HOST_DWIDTH / 8),
  localparam  SENSOR_KEEP_WIDTH = (SENSOR_DWIDTH / 8),
  parameter   MTU               = 1500
) (
  //Clock and Reset
  input   logic                         i_hif_clk,
  input   logic                         i_hif_rst,
  input   logic                         i_sif_clk,
  input   logic                         i_sif_rst,
  input   logic                         i_apb_clk,
  input   logic                         i_apb_rst,
  //APB
  input   apb_m2s                       i_apb_m2s,
  output  apb_s2m                       o_apb_s2m,
  //TX Sensor Data AXIS Input
  input   logic                         i_axis_tvalid,
  input   logic [HOST_DWIDTH-1:0]       i_axis_tdata,
  input   logic                         i_axis_tlast,
  input   logic                         i_axis_tuser,
  input   logic [HOST_KEEP_WIDTH-1:0]   i_axis_tkeep,
  //TX Sensor Data AXIS Output
  output  logic                         o_axis_tvalid,
  output  logic [SENSOR_DWIDTH-1:0]     o_axis_tdata,
  output  logic                         o_axis_tlast,
  output  logic                         o_axis_tuser,
  output  logic [SENSOR_KEEP_WIDTH-1:0] o_axis_tkeep,
  input   logic                         i_axis_tready,
  //Pause
  output  logic [NUM_HOSTS-1:0]         o_eth_pause,
  output  logic [15:0]                  o_pause_quanta
);


localparam N_CTRL_REG   = 5;
localparam N_STAT_REG   = 3;
localparam HALF_SECOND  = 26'h2FAF07F;
localparam FIFO_DEPTH_W = $clog2(FIFO_DEPTH)+1;

logic [31:0]                    ctrl_reg [N_CTRL_REG];
logic [31:0]                    stat_reg [N_STAT_REG];
logic                           buffer_empty;
logic                           buffer_aempty;
logic                           buffer_full;
logic                           buffer_afull;
logic [FIFO_DEPTH_W-1:0]        buffer_count;
logic                           buffer_underflow;
logic                           buffer_underflow_hif_sync;
logic                           buffer_overflow;
logic                           aempty_seen;
logic                           afull_seen;
logic                           afull_seen_sync;
logic                           tx_data_output_ena;
logic                           pause;
logic                           pause_ena;
logic                           prog_thresh_ena;
logic [FIFO_DEPTH_W-1:0]        almost_full_thresh;
logic [FIFO_DEPTH_W-1:0]        almost_empty_thresh;
logic [FIFO_DEPTH_W-1:0]        sw_almost_full_thresh;
logic [FIFO_DEPTH_W-1:0]        sw_almost_empty_thresh;
logic [25:0]                    second_cnt;
logic                           heartbeat;
logic                           axis_buffer_tx_tready;
logic                           axis_buffer_tvalid;
logic                           prog_thresh_ena_reg;
logic [FIFO_DEPTH_W-1:0]        high_watermark;
logic [FIFO_DEPTH_W-1:0]        low_watermark;
logic                           saw_overflow;
logic                           saw_underflow;
logic                           is_quanta_overwrite;
logic [NUM_HOSTS-1:0]           host_pause_mapping;
logic                           dbg_ena_output_sync;
logic                           dbg_ena_output;
logic                           stat_rst;


//------------------------------------------------------------------------------------------------//
// Second Count Logic
//------------------------------------------------------------------------------------------------//
always_ff @(posedge i_apb_clk) begin
  if (i_apb_rst) begin
    second_cnt                            <= 26'b0;
    heartbeat                             <= 1'b0;
  end
  else begin
    if (second_cnt == HALF_SECOND) begin
      second_cnt                          <= 26'b0;
      heartbeat                           <= ~heartbeat;
    end
    else begin
      second_cnt                          <= second_cnt + 1'b1;
    end
  end
end


//APB Register Block
s_apb_reg #(
  .N_CTRL                     ( N_CTRL_REG            ),
  .N_STAT                     ( N_STAT_REG            )
) u_sensor_tx_data_regs (
  // APB Interface
  .i_aclk                     ( i_apb_clk             ),
  .i_arst                     ( i_apb_rst             ),
  .i_apb_m2s                  ( i_apb_m2s             ),
  .o_apb_s2m                  ( o_apb_s2m             ),
  // User Control Signals
  .i_pclk                     ( i_hif_clk             ),
  .i_prst                     ( i_hif_rst             ),
  .o_ctrl                     ( ctrl_reg              ),
  .i_stat                     ( stat_reg              )
);

data_sync #
(
  .DATA_WIDTH                 ( 1                     ),
  .RESET_VALUE                ( 1'b0                  ),
  .SYNC_DEPTH                 ( 2                     )
) u_afull_seen_sync (
  .clk                        ( i_sif_clk             ),
  .rst_n                      ( !i_sif_rst            ),
  .sync_in                    ( afull_seen            ),
  .sync_out                   ( afull_seen_sync       )
);

assign axis_buffer_tx_tready = (afull_seen_sync & i_axis_tready) || dbg_ena_output_sync;
assign o_axis_tvalid         = afull_seen_sync & axis_buffer_tvalid;

//Transmit AXIS buffer
axis_buffer_tx #
(
  .IN_DWIDTH                  ( HOST_DWIDTH           ),
  .OUT_DWIDTH                 ( SENSOR_DWIDTH         ),
  .BUF_DEPTH                  ( FIFO_DEPTH            ),
  .DUAL_CLOCK                 ( 1                     )
) u_axis_tx_buffer (
  .in_clk                     ( i_hif_clk             ),
  .in_rst                     ( i_hif_rst             ),
  .out_clk                    ( i_sif_clk             ),
  .out_rst                    ( i_sif_rst             ),
  .i_axis_rx_tvalid           ( i_axis_tvalid         ),
  .i_axis_rx_tdata            ( i_axis_tdata          ),
  .i_axis_rx_tlast            ( i_axis_tlast          ),
  .i_axis_rx_tuser            ( i_axis_tuser          ),
  .i_axis_rx_tkeep            ( i_axis_tkeep          ),
  .o_axis_rx_tready           (                       ),
  .o_axis_tx_tvalid           ( axis_buffer_tvalid    ),
  .o_axis_tx_tdata            ( o_axis_tdata          ),
  .o_axis_tx_tlast            ( o_axis_tlast          ),
  .o_axis_tx_tuser            ( o_axis_tuser          ),
  .o_axis_tx_tkeep            ( o_axis_tkeep          ),
  .i_axis_tx_tready           ( axis_buffer_tx_tready ),
  .o_buffer_afull             ( /*NC*/                ),
  .o_buffer_aempty            ( /*NC*/                ),
  .o_buffer_full              ( /*buffer_full*/       ),
  .o_buffer_empty             ( /*buffer_empty*/      ),
  .o_buffer_wrcnt             ( buffer_count          ),
  .o_buffer_rdcnt             (                       ),
  .o_buffer_over              ( /*buffer_overflow*/   ),
  .o_buffer_under             ( /*buffer_underflow*/  )
);

//Register signals
assign tx_data_output_ena     = ctrl_reg[0][0];
assign pause_ena              = ctrl_reg[0][1];
assign prog_thresh_ena        = ctrl_reg[0][2];
assign stat_rst               = ctrl_reg[0][4];
assign dbg_ena_output         = ctrl_reg[0][8];
assign sw_almost_full_thresh  = ctrl_reg[1];
assign sw_almost_empty_thresh = ctrl_reg[2];
assign host_pause_mapping     = ctrl_reg[3];
assign o_pause_quanta         = is_quanta_overwrite ? ctrl_reg[4][15:0] : 16'hFFFF;

assign stat_reg[0]            = high_watermark;
assign stat_reg[1]            = low_watermark;
assign stat_reg[2]            = {saw_overflow, saw_underflow};



//Almost empty and almost full signaling.
always_ff @(posedge i_hif_clk) begin
  if (i_hif_rst) begin
    almost_full_thresh          <= (FIFO_DEPTH-(FIFO_DEPTH/4));
    almost_empty_thresh         <= (FIFO_DEPTH/4);
    buffer_aempty               <= 0;
    buffer_afull                <= 0;
    prog_thresh_ena_reg         <= 0;
  end
  else begin
    prog_thresh_ena_reg           <= prog_thresh_ena;
    if (prog_thresh_ena && !prog_thresh_ena_reg) begin
      almost_full_thresh          <= sw_almost_full_thresh;
      almost_empty_thresh         <= sw_almost_empty_thresh;
    end

    if (buffer_count >= almost_full_thresh) begin
      buffer_afull                <= 1;
    end
    else begin
      buffer_afull                <= 0;
    end

    if (buffer_count <= almost_empty_thresh) begin
      buffer_aempty               <= 1;
    end
    else begin
      buffer_aempty               <= 0;
    end
  end
end


//Almost full and almost empty seen logic
always_ff @(posedge i_hif_clk) begin
  if (i_hif_rst) begin
    aempty_seen                       <= 0;
    afull_seen                        <= 0;
  end
  else begin
    if (tx_data_output_ena) begin
      if (!buffer_aempty) begin
        aempty_seen                   <= 1'b1;
      end

      if (buffer_afull) begin
        afull_seen                    <= 1'b1;
      end

    end
    else begin
      aempty_seen                     <= 0;
      afull_seen                      <= 0;
    end
  end
end


//Pause signal logic
always_ff @(posedge i_hif_clk) begin
  if (i_hif_rst) begin
    pause                             <= 0;
  end
  else begin
    if (pause_ena) begin
      if (buffer_afull) begin
        pause                         <= 1'b1;
      end
      else if (buffer_aempty) begin
        pause                         <= 1'b0;
      end
    end
    else begin
      pause                           <= 0;
    end
  end
end

assign o_eth_pause = {NUM_HOSTS{pause}} & host_pause_mapping;


//High and low watermark logic
always_ff @(posedge i_hif_clk) begin
  if (i_hif_rst) begin
    high_watermark                    <= 0;
    low_watermark                     <= '1;
    saw_overflow                      <= 0;
    saw_underflow                     <= 0;
  end
  else begin
    if (stat_rst) begin
      high_watermark                  <= '0;
      low_watermark                   <= '1;
      saw_overflow                    <= 0;
      saw_underflow                   <= 0;
    end
    else if (tx_data_output_ena) begin
      saw_overflow                    <= saw_overflow || (afull_seen & buffer_count == FIFO_DEPTH-1);
      //Logic in the buffer module will prevent reads when FIFO is empty so signal underflow if buffer hits 0 and output is enabled.
      saw_underflow                   <= saw_underflow || (afull_seen & buffer_count == 0);

      //Track how high the buffer count gets after output is enabled.
      if (buffer_count > high_watermark) begin
        high_watermark                <= buffer_count;
      end

      //Lowest buffer count is tracked after data starts being read from FIFO. This occurs after almost full is reached for the first time.
      //If the buffer goes empty, then reset low watermark to highest value to indicate overflow.
      if (afull_seen) begin
        if (buffer_count == 0) begin
          low_watermark               <= '1;
        end
        else if (buffer_count < low_watermark) begin
          low_watermark               <= buffer_count;
        end
      end

    end
    else begin
      high_watermark                  <= 0;
      low_watermark                   <= '1;
      saw_overflow                    <= 0;
      saw_underflow                   <= 0;
    end
  end
end

data_sync #(
  .DATA_WIDTH     ( 1                   )
) u_dbg_en_sync (
  .clk            ( i_sif_clk           ),
  .rst_n          ( !i_sif_rst          ),
  .sync_in        ( dbg_ena_output      ),
  .sync_out       ( dbg_ena_output_sync )
);


logic [15:0] quanta_reg;
always_ff @(posedge i_hif_clk) begin
  if (i_hif_rst) begin
    is_quanta_overwrite <= 1'b0;
    quanta_reg          <= ctrl_reg[4][15:0];
  end
  else begin
    quanta_reg <= ctrl_reg[4][15:0];
    if (ctrl_reg[4][15:0] != quanta_reg) begin
      is_quanta_overwrite <= 1'b1;
    end
  end
end

endmodule
