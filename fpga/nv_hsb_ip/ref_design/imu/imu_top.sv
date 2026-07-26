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


module imu_top
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter  RAM_DEPTH = 128,
  parameter  W_DATA    = 64,
  localparam W_KEEP    = W_DATA/8
) (
  // PTP
  input                           i_ptp_clk,
  input                           i_ptp_rst,
  input    [63:0]                 i_ptp_ts,
  // Register Interface
  input                           i_aclk,
  input                           i_arst,
  input  apb_m2s                  i_apb_m2s,
  output apb_s2m                  o_apb_s2m,
  // AXIS Interface
  output   [W_DATA-1:0]           o_axis_tdata,
  output                          o_axis_tvalid,
  output                          o_axis_tlast,
  output                          o_axis_tuser,
  output   [W_KEEP-1:0]           o_axis_tkeep,
  input                           i_axis_tready,
  // Interrupt
  input    [1:0]                  i_start,
  //I2C Ports
  input                           scl_i,
  output                          scl_o,
  input                           sda_i,
  output                          sda_o           
);


logic        imu_data_valid;
logic [7:0]  imu_data;
logic        eop;
logic        sop;
logic        sample_last;

logic [63:0] imu_ts;
logic        imu_ts_vld;
logic        imu_sop_ptp_sync;

logic [7:0]  axis_tdata;
logic        axis_tvalid;


imu  #(
  .NUM_INST         ( 1                     ),
  .RAM_DEPTH        ( RAM_DEPTH             )
) imu_inst (
  .i_aclk           ( i_aclk                ),
  .i_arst           ( i_arst                ),
  .scl_i            ( scl_i                 ),
  .sda_i            ( sda_i                 ),
  .scl_o            ( scl_o                 ),
  .sda_o            ( sda_o                 ),
  .i_start          ( i_start               ),
  .o_data           ( imu_data              ),
  .o_data_valid     ( imu_data_valid        ),
  .o_busy           (                       ),
  .o_sample_last    ( sample_last           ),
  .o_eop            ( eop                   ),
  .o_sop            ( sop                   ),
  .o_bus_en         (                       ),
  .i_apb_m2s        ( i_apb_m2s             ),
  .o_apb_s2m        ( o_apb_s2m             )
);


// Append PTP timestamp to each sample
pulse_sync imu_data_pulse_ptp_sync_inst (
  .src_clk     ( i_aclk              ),
  .src_rst     ( i_arst              ),
  .dst_clk     ( i_ptp_clk           ),
  .dst_rst     ( i_ptp_rst           ),
  .i_src_pulse ( sop                 ),
  .o_dst_pulse ( imu_sop_ptp_sync    )
);

reg_cdc # (
  .NBITS         ( 64     ),
  .REG_RST_VALUE ( '0     )
) u_ptp_cdc (
  .i_a_clk ( i_ptp_clk                ),
  .i_a_rst ( i_ptp_rst                ),
  .i_a_val ( imu_sop_ptp_sync         ),
  .i_a_reg ( i_ptp_ts                 ),
  .i_b_clk ( i_aclk                   ),
  .i_b_rst ( i_arst                   ),
  .o_b_val ( imu_ts_vld               ),
  .o_b_reg ( imu_ts                   )
);

typedef enum logic [3:0] {
  IMU_IDLE,
  IMU_WAIT_TS,
  IMU_TS,
  IMU_DATA,
  IMU_SOP
} imu_fsm_t;

imu_fsm_t imu_state;

logic [2:0] cnt;
logic [7:0] imu_data_r;

always_ff @ (posedge i_aclk) begin
  if (i_arst) begin
    imu_state  <= IMU_IDLE;
    cnt        <= '0;
    imu_data_r <= '0;
  end else begin
    imu_state <= imu_state;
    case (imu_state)
      IMU_IDLE: begin
        if (sop) begin
          imu_state  <= IMU_WAIT_TS;
          imu_data_r <= imu_data;
        end
        cnt <= '0;
      end
      IMU_WAIT_TS: begin
        if (imu_ts_vld) begin
          imu_state <= IMU_TS;
        end
      end
      IMU_TS: begin
        cnt <= cnt + 1;
        if (cnt == 3'd7) begin
          imu_state <= IMU_SOP;
        end
      end
      IMU_SOP: begin
        imu_state <= IMU_DATA;
      end
      IMU_DATA: begin
        if (eop) begin
          imu_state <= IMU_IDLE;
        end
      end
    endcase
  end  
end

always_comb begin
  case (imu_state)
    IMU_IDLE: begin
      axis_tdata  = '0;
      axis_tvalid = '0;
    end
    IMU_WAIT_TS: begin
      axis_tdata  = '0;
      axis_tvalid = '0;
    end
    IMU_TS: begin
    axis_tdata  = imu_ts[cnt*8+:8];
    axis_tvalid = '1;
    end
    IMU_SOP: begin
      axis_tdata  = imu_data_r;
      axis_tvalid = '1;
    end
    IMU_DATA: begin
      axis_tdata  = imu_data;
      axis_tvalid = imu_data_valid;
    end
  endcase
end

axis_buffer #(
  .IN_DWIDTH        ( 8             ),
  .OUT_DWIDTH       ( W_DATA        ),
  .BUF_DEPTH        ( 16            ),
  .WAIT2SEND        ( 0             ),
  .DUAL_CLOCK       ( 0             )
) u_axis_buffer (
  .in_clk           ( i_aclk        ),
  .in_rst           ( i_arst        ),
  .out_clk          ( i_aclk        ),
  .out_rst          ( i_arst        ),
  .i_axis_rx_tvalid ( axis_tvalid   ),
  .i_axis_rx_tdata  ( axis_tdata    ),
  .i_axis_rx_tlast  ( sample_last   ),
  .i_axis_rx_tuser  ( '0            ),
  .i_axis_rx_tkeep  ( '1            ),
  .o_axis_rx_tready (               ),
  .o_axis_tx_tvalid ( o_axis_tvalid ),
  .o_axis_tx_tdata  ( o_axis_tdata  ),
  .o_axis_tx_tlast  ( o_axis_tlast  ),
  .o_axis_tx_tuser  ( o_axis_tuser  ),
  .o_axis_tx_tkeep  ( o_axis_tkeep  ),
  .i_axis_tx_tready ( i_axis_tready )
);

endmodule