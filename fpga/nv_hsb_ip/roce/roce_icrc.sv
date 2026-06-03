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


module roce_icrc #(
  parameter  W_DATA     = 64,
  parameter  W_KEEP     = W_DATA/8,
  parameter  W_USER     = 1
) (
  input                 pclk,
  input                 prst,

  input                 i_crc_en,

  input                 i_axis_rx_tvalid,
  input                 i_axis_rx_tlast,
  input  [W_KEEP-1:0]   i_axis_rx_tkeep,
  input  [W_DATA-1:0]   i_axis_rx_tdata,
  input  [W_USER-1:0]   i_axis_rx_tuser,
  output                o_axis_rx_tready,

  output                o_axis_tx_tvalid,
  output                o_axis_tx_tlast,
  output [W_KEEP-1:0]   o_axis_tx_tkeep,
  output [W_DATA-1:0]   o_axis_tx_tdata,
  output [W_USER-1:0]   o_axis_tx_tuser,
  input                 i_axis_tx_tready
);

logic [31:0] fcs_cmb_r      ;
logic        fcs_cmb_valid_r;

logic [4       :0] del_axis_tvalid;
logic [4       :0] del_axis_tlast;
logic [W_KEEP-1:0] del_axis_tkeep [4:0];
logic [W_DATA-1:0] del_axis_tdata [4:0];
logic [W_USER-1:0] del_axis_tuser [4:0];
logic [4       :0] del_axis_tready;
logic [4       :0] del_crc_en;

//------------------------------------------------------------------------------------------------//
// CRC Calc
//------------------------------------------------------------------------------------------------//

logic              drp_valid;
logic              tlast_prev;
logic              fcs_tready;
logic              fcs_tvalid;

logic              drp_axis_tvalid;
logic              drp_axis_tlast;
logic [W_KEEP-1:0] drp_axis_tkeep;
logic [W_DATA-1:0] drp_axis_tdata;
logic              drp_axis_tuser;
logic              drp_axis_tready;
logic              o_drp_axis_tready;

logic              msk_axis_tvalid;
logic              msk_axis_tlast;
logic [W_KEEP-1:0] msk_axis_tkeep;
logic [W_DATA-1:0] msk_axis_tdata;
logic              msk_axis_tuser;
logic              msk_axis_tready;

logic              mskr_axis_tvalid;
logic              mskr_axis_tlast;
logic [W_KEEP-1:0] mskr_axis_tkeep;
logic [W_DATA-1:0] mskr_axis_tdata;
logic              mskr_axis_tuser;
logic              mskr_axis_tready;

axis_drop #(
  .DROP_WIDTH  (6*8),
  .DWIDTH     (W_DATA)
) axis_drop_hdr (
  .clk                  ( pclk               ),
  .rst                  ( prst               ),
  .i_axis_rx_tvalid     ( drp_valid          ),
  .i_axis_rx_tdata      ( i_axis_rx_tdata    ),
  .i_axis_rx_tlast      ( i_axis_rx_tlast    ),
  .i_axis_rx_tuser      ( '0                 ),
  .i_axis_rx_tkeep      ( i_axis_rx_tkeep    ),
  .o_axis_rx_tready     ( o_drp_axis_tready  ),
  .o_axis_tx_tvalid     ( drp_axis_tvalid    ),
  .o_axis_tx_tdata      ( drp_axis_tdata     ),
  .o_axis_tx_tlast      ( drp_axis_tlast     ),
  .o_axis_tx_tuser      ( drp_axis_tuser     ),
  .o_axis_tx_tkeep      ( drp_axis_tkeep     ),
  .i_axis_tx_tready     ( drp_axis_tready    )
);

assign drp_valid = (o_axis_rx_tready && i_crc_en && i_axis_rx_tvalid);

logic [5:0][63:0] hdr_mask;
logic [64*6-1:0]  mask;

assign hdr_mask   = {64'hffffffffffffffff,         //Initialize with all 1's to mimic a LRH
                     64'h000000000000ff00,        //Set TOS to 1's
                     64'h00000000ffff00ff,        //Set TTL and IP Checksum to 1's
                     64'h0000000000000000,
                     64'h00000000ffff0000,        //Set UDP Checksum to 1's
                     64'h00000000000000ff         //Set BTH F, B, and Rsrv6 to 1's
                    };

assign mask = {hdr_mask[0],hdr_mask[1],hdr_mask[2],hdr_mask[3],hdr_mask[4],hdr_mask[5]};

axis_mask #(
  .MASK_WIDTH           ( 64*6               ),
  .DWIDTH               ( W_DATA             ),
  .OP                   ( "OR"               )
) axis_mask_data (
  .clk                  ( pclk               ),
  .rst                  ( prst               ),
  .i_mask               ( mask               ),
  .i_mask_en            ( 1'b1               ),
  .i_axis_rx_tvalid     ( drp_axis_tvalid    ),
  .i_axis_rx_tdata      ( drp_axis_tdata     ),
  .i_axis_rx_tlast      ( drp_axis_tlast     ),
  .i_axis_rx_tuser      ( drp_axis_tuser     ),
  .i_axis_rx_tkeep      ( drp_axis_tkeep     ),
  .o_axis_rx_tready     ( drp_axis_tready    ),
  .o_axis_tx_tvalid     ( msk_axis_tvalid    ),
  .o_axis_tx_tdata      ( msk_axis_tdata     ),
  .o_axis_tx_tlast      ( msk_axis_tlast     ),
  .o_axis_tx_tuser      ( msk_axis_tuser     ),
  .o_axis_tx_tkeep      ( msk_axis_tkeep     ),
  .i_axis_tx_tready     ( msk_axis_tready    )
);

axis_reg # (
  .DWIDTH             ( W_DATA + W_KEEP + 1 + 1                                                 )
) u_axis_reg_crc (
  .clk                ( pclk                                                                    ),
  .rst                ( prst                                                                    ),
  .i_axis_rx_tvalid   ( msk_axis_tvalid                                                         ),
  .i_axis_rx_tdata    ( {msk_axis_tdata,msk_axis_tlast,msk_axis_tuser,msk_axis_tkeep }          ),
  .o_axis_rx_tready   ( msk_axis_tready                                                         ),
  .o_axis_tx_tvalid   ( mskr_axis_tvalid                                                        ),
  .o_axis_tx_tdata    ( {mskr_axis_tdata,mskr_axis_tlast,mskr_axis_tuser,mskr_axis_tkeep}       ),
  .i_axis_tx_tready   ( mskr_axis_tready                                                        )
);

//------------------------------------------------------------------------------------------------//
// CRC Loop
//------------------------------------------------------------------------------------------------//


logic [W_KEEP-1:0] tkeep_reduced;
localparam ROCE_WRD_SIZE     = (((28+42-6) % W_KEEP)==0)    ? W_KEEP : ((28+42-6) % W_KEEP);
localparam ROCE_WRI_SIZE     = (((32+42-6) % W_KEEP)==0)    ? W_KEEP : ((32+42-6) % W_KEEP);
localparam ROCE_WRI_32B_SIZE = (((32+42-6+32) % W_KEEP)==0) ? W_KEEP : ((32+42-6+32) % W_KEEP);


localparam [W_KEEP-1:0] ROCE_WRD_KEEP     = {'0,{(ROCE_WRD_SIZE){1'b1}}};
localparam [W_KEEP-1:0] ROCE_WRI_KEEP     = {'0,{(ROCE_WRI_SIZE){1'b1}}};
localparam [W_KEEP-1:0] ROCE_WRI_32B_KEEP = {'0,{(ROCE_WRI_32B_SIZE){1'b1}}};


assign tkeep_reduced = (mskr_axis_tkeep == ROCE_WRD_KEEP                         ) ? ROCE_WRD_KEEP     :
                       (mskr_axis_tkeep == ROCE_WRI_KEEP                         ) ? ROCE_WRI_KEEP     :
                       ((mskr_axis_tkeep == ROCE_WRI_32B_KEEP) && (W_DATA >= 512)) ? ROCE_WRI_32B_KEEP :
                                                                                     '1                ;


logic [31:0] crc_s;
logic [31:0] fcs_r;
logic [31:0] fcs_w;
logic        fcs_w_valid;
logic [31:0] crc_next[W_KEEP-1:0];

generate
for (genvar i=0;i<W_KEEP;i=i+1) begin : crc_tkeep_loop
  lfsr_hc #(
    .DATA_WIDTH ( W_DATA/W_KEEP*(i+1))                     )
  eth_crc_inst (
    .data_in    ( mskr_axis_tdata[W_DATA/W_KEEP*(i+1)-1:0] ),
    .state_in   ( crc_s                                    ),
    .state_out  ( crc_next[i]                              )
  );
end
endgenerate

integer j;
always_comb begin
  fcs_w = '0;
  fcs_w_valid = '0;
  if (mskr_axis_tlast & mskr_axis_tvalid & mskr_axis_tready) begin
    fcs_w_valid = '1;
    fcs_w = ~crc_next[0];
    for (j=0;j<W_KEEP;j=j+1) begin
      if (tkeep_reduced[j]) begin
        fcs_w = ~crc_next[j];
      end
    end
  end
end

always @(posedge pclk) begin
  if (prst) begin
    crc_s  <= '1;
    fcs_r  <= '0;
  end
  else begin
    if (mskr_axis_tvalid) begin
      crc_s <= crc_next[W_KEEP-1];
      if (mskr_axis_tlast & mskr_axis_tvalid & mskr_axis_tready) begin
        fcs_r <= fcs_w;
        crc_s <= '1;
      end
    end
  end
end


assign mskr_axis_tready = '1;

//------------------------------------------------------------------------------------------------//
// Latch CRC
//------------------------------------------------------------------------------------------------//


logic fifo_rden;
logic fifo_empty;

reg_fifo #(
  .DATA_WIDTH                ( 32                ),
  .DEPTH                     ( 2                 )
) crc_buffer (
  .clk                       ( pclk              ),
  .rst                       ( prst              ),
  .wr                        ( fcs_w_valid       ),
  .din                       ( fcs_w             ),
  .full                      (                   ),
  .rd                        ( fifo_rden         ),
  .dout                      ( fcs_cmb_r         ),
  .dval                      (                   ),
  .over                      (                   ),
  .under                     (                   ),
  .empty                     ( fifo_empty        )
);

always @(posedge pclk) begin
  if (prst) begin
    fifo_rden <= '0;
  end
  else begin
    fifo_rden <= o_axis_tx_tlast && i_axis_tx_tready && o_axis_tx_tvalid && !fifo_empty;
  end
end


assign fcs_cmb_valid_r = !fifo_empty;
//------------------------------------------------------------------------------------------------//
// AXIS Output Path
//------------------------------------------------------------------------------------------------//


generate
  for (genvar i=0;i<4;i=i+1) begin
    // Delay data path to align CRC to same cycle
    axis_reg # (
      .DWIDTH             ( W_DATA + W_KEEP + W_USER + 1 + 1                                                                  ),
      .SKID               ( (i==0)                                                                                            )
    ) u_axis_reg_ftr (
      .clk                ( pclk                                                                                              ),
      .rst                ( prst                                                                                              ),
      .i_axis_rx_tvalid   ( del_axis_tvalid[i]                                                                                ),
      .i_axis_rx_tdata    ( {del_axis_tdata[i],del_axis_tlast[i],del_axis_tuser[i],del_axis_tkeep[i],del_crc_en[i] }          ),
      .o_axis_rx_tready   ( del_axis_tready[i]                                                                                ),
      .o_axis_tx_tvalid   ( del_axis_tvalid[i+1]                                                                              ),
      .o_axis_tx_tdata    ( {del_axis_tdata[i+1],del_axis_tlast[i+1],del_axis_tuser[i+1],del_axis_tkeep[i+1],del_crc_en[i+1]} ),
      .i_axis_tx_tready   ( del_axis_tready[i+1]                                                                              )
    );
  end
endgenerate

assign del_axis_tvalid [0] = i_axis_rx_tvalid ;
assign del_axis_tlast  [0] = i_axis_rx_tlast  ;
assign del_axis_tkeep  [0] = i_axis_rx_tkeep  ;
assign del_axis_tdata  [0] = i_axis_rx_tdata  ;
assign del_axis_tuser  [0] = i_axis_rx_tuser  ;
assign del_crc_en      [0] = i_crc_en         ;
assign o_axis_rx_tready    = del_axis_tready [0];


axis_ftr #(
  .FTR_WIDTH            ( 32                   ),
  .DWIDTH               ( W_DATA               ),
  .W_USER               ( W_USER               )
) axis_append_ftr (
  .clk                  ( pclk                 ),
  .rst                  ( prst                 ),
  .i_ftr                ( fcs_cmb_r            ),
  .i_ftr_val            ( fcs_cmb_valid_r      ),
  .i_ftr_en             ( del_crc_en      [4]  ),
  .i_axis_rx_tvalid     ( del_axis_tvalid [4]  ),
  .i_axis_rx_tdata      ( del_axis_tdata  [4]  ),
  .i_axis_rx_tlast      ( del_axis_tlast  [4]  ),
  .i_axis_rx_tuser      ( del_axis_tuser  [4]  ),
  .i_axis_rx_tkeep      ( del_axis_tkeep  [4]  ),
  .o_axis_rx_tready     ( del_axis_tready [4]  ),
  .o_axis_tx_tvalid     ( o_axis_tx_tvalid     ),
  .o_axis_tx_tdata      ( o_axis_tx_tdata      ),
  .o_axis_tx_tlast      ( o_axis_tx_tlast      ),
  .o_axis_tx_tuser      ( o_axis_tx_tuser      ),
  .o_axis_tx_tkeep      ( o_axis_tx_tkeep      ),
  .i_axis_tx_tready     ( i_axis_tx_tready     )
);

endmodule