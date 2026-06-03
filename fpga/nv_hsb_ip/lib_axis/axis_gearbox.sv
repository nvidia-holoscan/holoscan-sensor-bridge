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

// MIXED WIDTH Buffer, DOUT/DIN must be an integer
module axis_gearbox # (
  parameter  DIN_WIDTH     = 8,
  parameter  DOUT_WIDTH    = 32,
  parameter  W_USER        = 1,
  localparam EVEN_MULTIPLE = (((DIN_WIDTH%DOUT_WIDTH) == '0) || ((DOUT_WIDTH%DIN_WIDTH) == '0)),
  localparam IN_W_KEEP     = DIN_WIDTH/8,
  localparam OUT_W_KEEP    = DOUT_WIDTH/8,
  localparam IN_W_USER     = W_USER,
  localparam OUT_W_USER    = (EVEN_MULTIPLE) ? W_USER : (IN_W_USER * DOUT_WIDTH / DIN_WIDTH)
) (
  input                                       clk,
  input                                       rst,
  input                                       i_axis_rx_tvalid,
  input     [DIN_WIDTH-1:0]                   i_axis_rx_tdata,
  input                                       i_axis_rx_tlast,
  input     [IN_W_USER-1:0]                   i_axis_rx_tuser,
  input     [IN_W_KEEP-1:0]                   i_axis_rx_tkeep,
  output                                      o_axis_rx_tready,
  output                                      o_axis_tx_tvalid,
  output    [DOUT_WIDTH-1:0]                  o_axis_tx_tdata,
  output                                      o_axis_tx_tlast,
  output    [OUT_W_USER-1:0]                  o_axis_tx_tuser,
  output    [OUT_W_KEEP-1:0]                  o_axis_tx_tkeep,
  input                                       i_axis_tx_tready
);


generate

//------------------------------------------------------------------------------------------------//
// DOUT == DIN
//------------------------------------------------------------------------------------------------//

  if (DOUT_WIDTH == DIN_WIDTH) begin

    assign o_axis_tx_tvalid    = i_axis_rx_tvalid;
    assign o_axis_tx_tdata     = i_axis_rx_tdata ;
    assign o_axis_tx_tlast     = i_axis_rx_tlast ;
    assign o_axis_tx_tuser     = i_axis_rx_tuser ;
    assign o_axis_tx_tkeep     = i_axis_rx_tkeep ;
    assign o_axis_rx_tready    = i_axis_tx_tready;

  end

//------------------------------------------------------------------------------------------------//
// DOUT > DIN : Even Multiple
//------------------------------------------------------------------------------------------------//
  else if (DOUT_WIDTH > DIN_WIDTH) begin
    if (EVEN_MULTIPLE) begin
      localparam MW_RATIO  = DOUT_WIDTH/DIN_WIDTH;
      localparam CNT_WIDTH = $clog2(MW_RATIO);

      logic [DOUT_WIDTH-1:0] tdata, tdata_r;
      logic [OUT_W_KEEP-1:0] tkeep, tkeep_r;
      logic [CNT_WIDTH-1:0]  cnt_r;
      logic                  send;


      always_comb begin
        tdata = tdata_r;
        tkeep = tkeep_r;
        if (i_axis_rx_tvalid) begin
          tdata[cnt_r*DIN_WIDTH+:DIN_WIDTH] = i_axis_rx_tdata;
          tkeep[cnt_r*IN_W_KEEP+:IN_W_KEEP] = i_axis_rx_tkeep;
        end
      end


      always_ff @(posedge clk) begin
        if (rst) begin
          cnt_r   <= '0;
          tdata_r <= '0;
          tkeep_r <= '0;
        end
        else begin
          cnt_r   <= (!i_axis_tx_tready) ? cnt_r    :
                  (send)              ? '0       :
                  (!i_axis_rx_tvalid) ? cnt_r    :
                                          cnt_r + 1;
          tdata_r <= (o_axis_tx_tvalid && i_axis_tx_tready) ? '0 : tdata;
          tkeep_r <= (o_axis_tx_tvalid && i_axis_tx_tready) ? '0 : tkeep;
        end
      end


      assign o_axis_rx_tready = i_axis_tx_tready;
      assign o_axis_tx_tuser  = i_axis_rx_tuser;
      assign o_axis_tx_tdata  = tdata;
      assign o_axis_tx_tkeep  = tkeep;
      assign o_axis_tx_tlast  = i_axis_rx_tlast;
      assign o_axis_tx_tvalid = (i_axis_rx_tvalid && ((cnt_r == MW_RATIO-1) || (i_axis_rx_tlast)));
      assign send             = o_axis_tx_tvalid && i_axis_tx_tready;
    end
//------------------------------------------------------------------------------------------------//
// DOUT > DIN : Uneven Multiple
//------------------------------------------------------------------------------------------------//
    else begin
      localparam GEARBOX_WIDTH = DIN_WIDTH + DOUT_WIDTH;
      localparam GEARBOX_WKEEP = GEARBOX_WIDTH / 8;
      localparam CNT_WIDTH     = $clog2(GEARBOX_WKEEP) + 3;
      localparam GEARBOX_WUSER = IN_W_USER + OUT_W_USER;
      localparam UCNT_WIDTH    = $clog2(GEARBOX_WUSER) + 3;

      logic [GEARBOX_WIDTH-1:0] tdata, tdata_r, tdin;
      logic [GEARBOX_WKEEP-1:0] tkeep, tkeep_r, tkin;
      logic [GEARBOX_WUSER-1:0] tuser, tuser_r, tuin;
      logic                     tvalid;
      logic                     flush;
      logic [CNT_WIDTH:0]       cnt, cnt_r;
      logic [UCNT_WIDTH:0]      ucnt, ucnt_r;
      logic                     even_last;

      always_comb begin
        tdata = tdata_r;
        tkeep = tkeep_r;
        tuser = tuser_r;
        cnt   = cnt_r;
        ucnt  = ucnt_r;
        tvalid = '0;
        tdin = '0;
        tkin = '0;
        tuin = '0;

        if (i_axis_tx_tready) begin
          // Shift In
          if (flush) begin
            tdin = '0;
            tkin = '0;
            tuin = '0;
          end
          else if (i_axis_rx_tvalid) begin
            tdin = i_axis_rx_tdata << (cnt << 3);
            tkin = i_axis_rx_tkeep << cnt;
            tuin = i_axis_rx_tuser << ucnt;
          end

          if (i_axis_rx_tvalid) begin
            cnt = cnt + IN_W_KEEP;
            ucnt = ucnt + IN_W_USER;
          end

          tdata = tdin | tdata;
          tkeep = tkin | tkeep;
          tuser = tuin | tuser;

          // Shift Out
          if (o_axis_tx_tlast) begin
            cnt = '0;
            ucnt = '0;
            tvalid = '1;
          end
          else if (cnt >= OUT_W_KEEP) begin
            cnt = cnt - OUT_W_KEEP;
            ucnt = ucnt - OUT_W_USER;
            tvalid = '1;
          end
        end
      end

      always_ff @(posedge clk) begin
        if (rst) begin
          cnt_r   <= '0;
          ucnt_r  <= '0;
          tdata_r <= '0;
          tkeep_r <= '0;
          tuser_r <= '0;
          flush   <= '0;
        end
        else begin
          cnt_r   <= cnt;
          ucnt_r  <= ucnt;
          tdata_r <= (o_axis_tx_tvalid) ? tdata >> DOUT_WIDTH : tdata ;
          tkeep_r <= (o_axis_tx_tvalid) ? tkeep >> OUT_W_KEEP : tkeep ;
          tuser_r <= (o_axis_tx_tvalid) ? tuser >> OUT_W_USER : tuser ;
          flush   <= (i_axis_rx_tvalid && i_axis_rx_tlast && o_axis_rx_tready && !even_last);
        end
      end

      assign o_axis_rx_tready = i_axis_tx_tready && !flush;
      assign o_axis_tx_tuser  = tuser;
      assign o_axis_tx_tdata  = tdata;
      assign o_axis_tx_tkeep  = tkeep;
      assign o_axis_tx_tlast  = ((i_axis_rx_tvalid && i_axis_rx_tlast && even_last) || flush);
      assign o_axis_tx_tvalid = tvalid;
      assign even_last        = (tkeep[GEARBOX_WKEEP-1:OUT_W_KEEP] == '0);
    end
  end
//------------------------------------------------------------------------------------------------//
// DOUT < DIN : Even Multiple
//------------------------------------------------------------------------------------------------//

  else begin
    if (EVEN_MULTIPLE) begin
      localparam MW_RATIO  = DIN_WIDTH/DOUT_WIDTH;
      localparam CNT_WIDTH = $clog2(MW_RATIO);

      logic [CNT_WIDTH-1:0] cnt;
      logic [IN_W_KEEP:0]   w_tkeep;

      always_ff @(posedge clk) begin
        if (rst) begin
          cnt   <= '0;
        end
        else begin
          if (i_axis_rx_tvalid && i_axis_tx_tready) begin
            if ((cnt == MW_RATIO-1) || (o_axis_tx_tlast)) begin
              cnt <= '0;
            end
            else begin
              cnt <= cnt + 1'b1;
            end
          end
        end
      end

      assign o_axis_tx_tuser  = i_axis_rx_tuser;
      assign o_axis_tx_tdata  = i_axis_rx_tdata[cnt*DOUT_WIDTH+:DOUT_WIDTH];
      assign o_axis_tx_tkeep  = i_axis_rx_tkeep[cnt*OUT_W_KEEP+:OUT_W_KEEP];
      assign o_axis_tx_tlast  = i_axis_rx_tlast && ((w_tkeep[cnt*OUT_W_KEEP+:OUT_W_KEEP+1]) != '1);
      assign o_axis_rx_tready = ((w_tkeep[cnt*OUT_W_KEEP+:OUT_W_KEEP+1]) != '1) && i_axis_tx_tready;
      assign o_axis_tx_tvalid = i_axis_rx_tvalid;
      assign w_tkeep          = {1'b0,i_axis_rx_tkeep};
    end
//------------------------------------------------------------------------------------------------//
// DOUT < DIN : Unven Multiple
//------------------------------------------------------------------------------------------------//
    else begin
      localparam GEARBOX_WIDTH = DIN_WIDTH + DOUT_WIDTH;
      localparam GEARBOX_WKEEP = GEARBOX_WIDTH / 8;
      localparam GEARBOX_WUSER = OUT_W_USER + IN_W_USER;
      localparam CNT_WIDTH     = $clog2(GEARBOX_WKEEP) + 3;
      localparam UCNT_WIDTH    = $clog2(GEARBOX_WUSER) + 3;

      logic [GEARBOX_WIDTH-1:0] tdata, tdata_r;
      logic [GEARBOX_WIDTH-1:0] tdin;
      logic [GEARBOX_WUSER-1:0] tuin, tuser, tuser_r;
      logic [GEARBOX_WKEEP-1:0] tkin;
      logic [GEARBOX_WKEEP-1:0] tkeep, tkeep_r;
      logic                     tvalid;
      logic                     tready;
      logic                     flush;
      logic [CNT_WIDTH:0]       cnt, cnt_r;
      logic [UCNT_WIDTH:0]      ucnt, ucnt_r;
      logic                     even_last;


      always_comb begin
        tdata = tdata_r;
        tkeep = tkeep_r;
        tuser = tuser_r;
        tdin  = '0;
        tkin  = '0;
        tuin  = '0;
        cnt   = cnt_r;
        ucnt  = ucnt_r;
        tvalid = '0;
        tready = '0;

        if (i_axis_tx_tready) begin
          // Shift In
          if (flush) begin
            tdin = '0;
            tkin = '0;
            tuin = '0;
          end
          else if (i_axis_rx_tvalid && (cnt < OUT_W_KEEP)) begin
            tdin = i_axis_rx_tdata << (cnt << 3);
            tkin = i_axis_rx_tkeep << cnt;
            tuin = i_axis_rx_tuser << ucnt;
            cnt = cnt + IN_W_KEEP;
            ucnt = ucnt + IN_W_USER;
            tready = 1'b1;
          end

          tdata = tdin | tdata;
          tkeep = tkin | tkeep;
          tuser = tuin | tuser;

          // Shift Out
          if (o_axis_tx_tlast) begin
            cnt = '0;
            ucnt = '0;
            tvalid = '1;
          end
          else if (cnt >= OUT_W_KEEP) begin
            cnt = cnt - OUT_W_KEEP;
            ucnt = ucnt - OUT_W_USER;
            tvalid = '1;
          end
        end
      end

      always_ff @(posedge clk) begin
        if (rst) begin
          cnt_r   <= '0;
          ucnt_r  <= '0;
          tdata_r <= '0;
          tkeep_r <= '0;
          tuser_r <= '0;
          flush   <= '0;
        end
        else begin
          cnt_r   <= cnt;
          ucnt_r  <= ucnt;
          tdata_r <= (o_axis_tx_tvalid) ? tdata >> DOUT_WIDTH : tdata;
          tkeep_r <= (o_axis_tx_tvalid) ? tkeep >> OUT_W_KEEP : tkeep;
          tuser_r <= (o_axis_tx_tvalid) ? tuser >> OUT_W_USER : tuser;
          flush   <= (i_axis_rx_tvalid && i_axis_rx_tlast && o_axis_rx_tready && !even_last);
        end
      end

      assign o_axis_rx_tready = tready;
      assign o_axis_tx_tuser  = tuser;
      assign o_axis_tx_tdata  = tdata;
      assign o_axis_tx_tkeep  = tkeep;
      assign o_axis_tx_tlast  = ((i_axis_rx_tvalid && i_axis_rx_tlast && even_last && tready) || flush);
      assign o_axis_tx_tvalid = tvalid;
      assign even_last        = (tkeep[GEARBOX_WKEEP-1:OUT_W_KEEP] == '0);

    end
  end
endgenerate

endmodule
