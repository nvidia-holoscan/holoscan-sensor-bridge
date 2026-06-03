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

module ecb_rdwr_ctrl
  import                                              ecb_pkg::*;
  import                                              apb_pkg::*;
  import                                              regmap_pkg::*;      
#(
  parameter                                           AXI_DWIDTH = 64,
  parameter                                           W_USER     = 1,
  parameter                                           MTU        = 1500,
  parameter                                           SYNC_CLK   = 0
) (
  //Clocks and resets
  input   logic                                       i_aclk,
  input   logic                                       i_arst,
  input   logic                                       i_pclk,
  input   logic                                       i_prst,
  input   apb_m2s                                     i_apb_m2s,
  output  apb_s2m                                     o_apb_s2m,
  //AXIS Input Interface
  input  logic   [1:0]                                i_axis_tvalid,
  input  logic   [AXI_DWIDTH-1:0]                     i_axis_tdata,
  input  logic                                        i_axis_tlast,
  input  logic   [W_USER-1:0]                         i_axis_tuser,
  input  logic   [(AXI_DWIDTH/8)-1:0]                 i_axis_tkeep,
  output logic   [1:0]                                o_axis_tready,
  // HDR info
  output  logic   [47:0]                              o_host_mac_addr,
  output  logic   [31:0]                              o_host_ip_addr,
  output  logic   [15:0]                              o_host_udp_port,
  output  logic   [15:0]                              o_pld_len,
  output  logic   [15:0]                              o_dev_udp_port,
  //APB Control Interface
  output  apb_m2s                                     o_apb_m2s,
  input   apb_s2m                                     i_apb_s2m,
  //ECB Response Interface
  output  logic                                       o_axis_tvalid,
  output  logic   [AXI_DWIDTH-1:0]                    o_axis_tdata,
  output  logic                                       o_axis_tlast,
  output  logic   [W_USER-1:0]                        o_axis_tuser,
  output  logic   [(AXI_DWIDTH/8)-1:0]                o_axis_tkeep,
  output  logic                                       o_axis_is_roce,
  input   logic                                       i_axis_tready
);

localparam RESP_WIDTH   = 6;
localparam ROCE_HDR_WIDTH = 32;

//------------------------------------------------------------------------------
// Registers
//------------------------------------------------------------------------------
logic [31:0] ctrl_reg [ecb_nctrl];
logic [31:0] stat_reg [ecb_nstat];

assign stat_reg[ecb_stat-stat_ofst] = '{default:0};

logic [63:0] roce_vaddr;
logic [31:0] roce_rkey;
logic [15:0] roce_pkey;
logic [23:0] roce_dest_qp;

s_apb_reg #(
  .N_CTRL    ( ecb_nctrl      ),
  .N_STAT    ( ecb_nstat      ),
  .W_OFST    ( w_ofst         ),
  .SYNC_CLK  ( SYNC_CLK       )
) u_reg_map  (
  // APB Interface
  .i_aclk    ( i_aclk         ), 
  .i_arst    ( i_arst         ),
  .i_apb_m2s ( i_apb_m2s      ),
  .o_apb_s2m ( o_apb_s2m      ),
  // User Control Signals
  .i_pclk    ( i_pclk         ), 
  .i_prst    ( i_prst         ),
  .o_ctrl    ( ctrl_reg       ),
  .i_stat    ( stat_reg       )
);

assign roce_vaddr   = {ctrl_reg[ecb_addr_msb][31:0], ctrl_reg[ecb_addr_lsb][31:0]};
assign roce_rkey    = ctrl_reg[ecb_rkey][31:0];
assign roce_pkey    = ctrl_reg[ecb_pkey][15:0];
assign roce_dest_qp = ctrl_reg[ecb_dest_qp][23:0];

//------------------------------------------------------------------------------------------------//
// Parse Incoming Data
//------------------------------------------------------------------------------------------------//

enum logic [2:0] {DCD_IDLE, DCD_HDR,DCD_DATA, DCD_DONE, DCD_FLUSH} decode_state;

logic [7:0]   ecb_cmd;
logic [7:0]   ecb_flags;
logic [15:0]  ecb_seq;

logic [7:0]   roce_opcode;

logic [15:0]  udp_len;

logic [5:0]   byte_cnt;
logic [7:0]   data_cnt;
logic         tready;

logic [31:0] cmd_addr        /* synthesis syn_keep=1 nomerge=""*/;
logic [31:0] cmd_addr_sync   /* synthesis syn_keep=1 nomerge=""*/;
logic [31:0] cmd_dout;

logic   [8:0]  num_pairs;
logic   [8:0]  total_pairs;
logic          response_sent;
logic          drop_packet;

logic decode_hdr;
logic decode_data;
logic decode_flush;
logic decode_done;
logic decode_start;
logic decode_last;
logic decode_tlast_seen;

logic decode_hdr_sync;
logic decode_data_sync;
logic decode_flush_sync;
logic decode_done_sync;
logic decode_start_sync;
logic decode_last_sync;

logic [W_USER-1:0] tuser;

logic ecb_is_roce;
logic ecb_is_roce_sync;
 
// Adaptive ECB addr
always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    o_host_mac_addr <= '0;
    o_host_ip_addr  <= '0;
    o_host_udp_port <= '0;
    o_dev_udp_port  <= '0;
    ecb_cmd         <= '0;
    ecb_flags       <= '0;
    ecb_seq         <= '0;
    tuser           <= '0;
    udp_len         <= '0;
    total_pairs     <= '0;
    roce_opcode     <= '0;
  end
  else begin
    total_pairs <= (ecb_is_roce) ? (udp_len-'d38)>>3 : (udp_len-'d22)>>3; // UDP header + RoCE header + ICRC + 1 pair
    if (decode_state == DCD_HDR) begin
      tuser           <= i_axis_tuser;
      case(byte_cnt)
        'd06: o_host_mac_addr[5*8+:8]   <= i_axis_tdata;
        'd07: o_host_mac_addr[4*8+:8]   <= i_axis_tdata;
        'd08: o_host_mac_addr[3*8+:8]   <= i_axis_tdata;
        'd09: o_host_mac_addr[2*8+:8]   <= i_axis_tdata;
        'd10: o_host_mac_addr[1*8+:8]   <= i_axis_tdata;
        'd11: o_host_mac_addr[0*8+:8]   <= i_axis_tdata;
        'd26: o_host_ip_addr [3*8+:8]   <= i_axis_tdata;
        'd27: o_host_ip_addr [2*8+:8]   <= i_axis_tdata;
        'd28: o_host_ip_addr [1*8+:8]   <= i_axis_tdata;
        'd29: o_host_ip_addr [0*8+:8]   <= i_axis_tdata;
        'd34: begin // Switch UDP port for host and FPGA in ECB case. Otherwise, use same port.
          if (ecb_is_roce) o_dev_udp_port[1*8+:8]  <= i_axis_tdata;
          else             o_host_udp_port[1*8+:8] <= i_axis_tdata;
        end
        'd35: begin
          if (ecb_is_roce) o_dev_udp_port[0*8+:8]  <= i_axis_tdata;
          else             o_host_udp_port[0*8+:8] <= i_axis_tdata;
        end
        'd36: begin
          if (ecb_is_roce) o_host_udp_port[1*8+:8] <= i_axis_tdata;
          else             o_dev_udp_port[1*8+:8]  <= i_axis_tdata;
        end
        'd37: begin
          if (ecb_is_roce) o_host_udp_port[0*8+:8] <= i_axis_tdata;
          else             o_dev_udp_port[0*8+:8]  <= i_axis_tdata;
        end
        'd38: udp_len[1*8+:8] <= i_axis_tdata;
        'd39: udp_len[0*8+:8] <= i_axis_tdata;
        'd42: begin
          ecb_cmd                       <= i_axis_tdata;
          roce_opcode                   <= i_axis_tdata;
        end
        'd43: ecb_flags                 <= i_axis_tdata;
        'd44: ecb_seq    [8+:8]         <= i_axis_tdata;
        'd45: ecb_seq    [0+:8]         <= i_axis_tdata;
        'd54: ecb_cmd                   <= i_axis_tdata;
        'd55: ecb_flags                 <= i_axis_tdata;
        'd56: ecb_seq        [8+:8]     <= i_axis_tdata;
        'd57: ecb_seq        [0+:8]     <= i_axis_tdata;
      endcase
    end
  end
end

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    cmd_addr <= '0;
    cmd_dout <= '0;
  end
  else begin
    if (decode_state == DCD_DATA) begin
      case(data_cnt)
        'd00: cmd_addr   [3*8+:8]  <= i_axis_tdata;
        'd01: cmd_addr   [2*8+:8]  <= i_axis_tdata;
        'd02: cmd_addr   [1*8+:8]  <= i_axis_tdata;
        'd03: cmd_addr   [0*8+:8]  <= i_axis_tdata;
        'd04: cmd_dout   [3*8+:8]  <= i_axis_tdata;
        'd05: cmd_dout   [2*8+:8]  <= i_axis_tdata;
        'd06: cmd_dout   [1*8+:8]  <= i_axis_tdata;
        'd07: cmd_dout   [0*8+:8]  <= i_axis_tdata;
      endcase
    end
    else if (decode_state == DCD_FLUSH) begin
      cmd_addr <= '0;
      cmd_dout <= '0;
    end
  end
end

//Command State Machine
always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    decode_state                                          <= DCD_IDLE;
    tready                                                <= '0;
    decode_done                                           <= '0;
    decode_start                                          <= '0;
    decode_last                                           <= '0;
    decode_tlast_seen                                     <= '0;
    response_sent                                         <= '0;
    num_pairs                                             <= '0;
    ecb_is_roce                                           <= '0;
    byte_cnt                                              <= '0;
    data_cnt                                              <= '0;
  end
  else begin
    case (decode_state)
      DCD_IDLE        : begin
        decode_start   <= (i_axis_tvalid[0] || i_axis_tvalid[1]);
        decode_state   <= (decode_hdr_sync )  ? DCD_HDR   :
                          (decode_data_sync)  ? DCD_DATA  :
                          (decode_flush_sync) ? DCD_FLUSH :
                                                DCD_IDLE  ;
        tready          <= (decode_hdr_sync || decode_data_sync || (decode_flush_sync && !decode_tlast_seen));
        byte_cnt        <= '0;
        data_cnt        <= '0;
        decode_done     <= '0;
        response_sent   <= '0;
      end
      DCD_HDR    : begin
        ecb_is_roce       <= i_axis_tvalid[1];
        decode_start      <= '0;
        decode_last       <= '0;
        decode_tlast_seen <= '0;
        byte_cnt          <= byte_cnt + 1'b1;
        decode_state      <= (byte_cnt == (ecb_is_roce ? 'd59 : 'd47)) ? DCD_DONE: DCD_HDR;
        tready            <= !(byte_cnt == (ecb_is_roce ? 'd59 : 'd47));
        num_pairs         <= '0;
      end
      DCD_DATA      : begin
        data_cnt          <= data_cnt + 1;
        decode_state      <= (data_cnt == 'd7) ? DCD_DONE : (i_axis_tlast) ? DCD_DONE : DCD_DATA;
        tready            <= !(data_cnt == 'd7);
        decode_last       <= (i_axis_tlast) || ((num_pairs == total_pairs) && (ecb_is_roce)) ? 1 : decode_last;
        decode_tlast_seen <= (i_axis_tlast) ? 1 : decode_tlast_seen;
        num_pairs         <= (data_cnt == 'd7) ? num_pairs + 1 : num_pairs;
      end
      DCD_FLUSH: begin
        decode_state      <= (response_sent && decode_tlast_seen) ? DCD_DONE : DCD_FLUSH;
        tready            <= (i_axis_tlast) ? '0 : tready;
        decode_last       <= (i_axis_tlast) || ((num_pairs == total_pairs) && (ecb_is_roce)) ? 1 : decode_last;
        decode_tlast_seen <= (i_axis_tlast) ? 1 : decode_tlast_seen;
        response_sent     <= (o_axis_tlast && ((i_axis_tready && o_axis_tvalid) || (drop_packet))) ? '1 : response_sent;
      end
      DCD_DONE:  begin
        decode_state <= (!decode_hdr_sync && !decode_data_sync && !decode_flush_sync) ? DCD_IDLE : DCD_DONE;
        decode_done  <= '1;
        tready       <= '0;
      end
      default : begin
        decode_state   <= DCD_IDLE;
      end
    endcase
  end
end

assign o_axis_tready = {tready,tready};

//------------------------------------------------------------------------------------------------//
// APB FSM
//------------------------------------------------------------------------------------------------//

enum logic [3:0] {CMD_IDLE, CMD_DECODE,CMD_GET_DATA, CMD_SEND,CMD_WAIT, CMD_RSP, CMD_RSP_WAIT, CMD_ERR} cmd_state;

logic                                       ecb_cmd_error;
logic                                       ecb_flag_error;
logic                                       ecb_addr_error;
logic                                       ecb_seq_chk_error;
logic                                       ecb_seq_chk_error_latched;
logic                                       ecb_error;
logic   [7:0]                               ecb_resp_code;
logic   [7:0]                               ecb_cmd_code;
logic                                       cmd_wren;
logic                                       cmd_en;
logic                                       cmd_err;
logic   [(RESP_WIDTH*8)-1:0]                resp_data;
logic                                       resp_trigger;
logic                                       resp_trigger_sync;
logic                                       resp_vec_is_busy;
logic                                       resp_vec_is_busy_sync;
logic   [15:0]                              seq_num_latch;
logic   [15:0]                              seq_num_latch_sync;
logic   [15:0]                              prev_seq_num_latch;
logic                                       ecb_wr;
logic                                       ecb_rd;
logic                                       ecb_block;
logic                                       r_ecb_error;

assign ecb_wr             = ((ecb_cmd == WR_DWORD) || (ecb_cmd == WR_BLOCK));
assign ecb_rd             = ((ecb_cmd == RD_DWORD) || (ecb_cmd == RD_BLOCK));
assign ecb_block          = ((ecb_cmd == WR_BLOCK) || (ecb_cmd == RD_BLOCK));

//Command State Machine
always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    cmd_state                                         <= CMD_IDLE;
    decode_hdr                                        <= '0;
    decode_flush                                      <= '0;
    decode_data                                       <= '0;
    cmd_wren                                          <= '0;
    cmd_en                                            <= '0;
    resp_trigger                                      <= '0;
    seq_num_latch                                     <= '0;
    o_apb_m2s.penable                                 <= '0;
    cmd_err                                           <= '0;
    r_ecb_error                                       <= '0;
    ecb_seq_chk_error_latched                         <= '0;
    prev_seq_num_latch                                <= '0;
    resp_data                                         <= '0;
  end
  else begin
    case (cmd_state)
      CMD_IDLE        : begin
        cmd_state                                         <= (decode_start_sync) ? CMD_DECODE : CMD_IDLE;
        decode_hdr                                        <= (decode_start_sync);
        decode_flush                                      <= '0;
        decode_data                                       <= '0;
        cmd_wren                                          <= '0;
        cmd_en                                            <= '0;
        resp_trigger                                      <= '0;
        o_apb_m2s.penable                                 <= '0;
        r_ecb_error                                       <= '0;
      end

      //Check for available ECB commands on available interfaces
      CMD_DECODE    : begin
        if (decode_hdr) begin
          if (decode_done_sync) begin
            decode_hdr <= '0;
            //If there is no sequence check error or other ecb errors, latch the current sequence number.
            //In case of an error, latched value stays the same.
            if (!ecb_error) begin
              seq_num_latch                                   <= ecb_seq;
              prev_seq_num_latch                              <= seq_num_latch;
            end
            ecb_seq_chk_error_latched                         <= ecb_seq_chk_error;
            r_ecb_error                                       <= ecb_error;
          end
        end
        else begin
          if (!decode_done_sync) begin
            decode_data <= '1;
            cmd_state   <= CMD_GET_DATA;
          end
        end
      end
      CMD_GET_DATA      : begin
        if (decode_done_sync) begin
          decode_data <= '0;
          cmd_state   <= CMD_SEND;
          //Check for errors in latched command. If detected, send error response, else send the read/write command.
          cmd_wren    <= ecb_wr && !r_ecb_error;
          cmd_err     <= '0;
        end
      end
      CMD_SEND:  begin
        if (!decode_done_sync) begin
          cmd_en    <= !r_ecb_error;
          cmd_state <= CMD_WAIT;
        end
      end
      //Wait for the APB to signal done for the sent command.
      CMD_WAIT : begin
        if (i_apb_s2m.pready || r_ecb_error) begin
          cmd_wren                                        <= 1'b0;
          cmd_en                                          <= 1'b0;
          o_apb_m2s.penable                               <= 1'b0;
          cmd_err                                         <= i_apb_s2m.pserr;
          if (i_apb_s2m.pserr) begin
            cmd_state <= CMD_RSP;
          end
          else if (!decode_last_sync && (ecb_block) && !r_ecb_error) begin
            cmd_state   <= CMD_GET_DATA;
            decode_data <= '1;
          end
          else begin
            cmd_state <= CMD_RSP;
          end
          if (i_apb_s2m.pserr && !r_ecb_error) begin
            seq_num_latch                                 <= prev_seq_num_latch;
          end
        end
        else begin
          o_apb_m2s.penable                               <= !r_ecb_error;
        end
      end
      //Send a response for the command.
      CMD_RSP : begin
        if (!resp_vec_is_busy_sync) begin
          resp_trigger                                    <= 1'b1;
          resp_data                                       <= resp_end_swap({ecb_cmd_code, ecb_flags, ecb_seq, ecb_resp_code, 8'b0});
          cmd_state                                       <= CMD_RSP_WAIT;
          decode_flush                                    <= '1;
          cmd_err                                         <= 1'b0;
        end
      end
      //Wait for the response to egress the vec_to_axis module before cycling back to check for new commands.
      CMD_RSP_WAIT : begin
        resp_trigger                                      <= 1'b0;
        if (decode_flush) begin
          decode_flush <= !decode_done_sync;
        end
        else begin
          cmd_state <= (decode_done_sync) ? CMD_RSP_WAIT : CMD_IDLE;
          resp_data <= '0;
        end
      end
      default : begin
        cmd_state                                         <= CMD_IDLE;
      end
    endcase
  end
end

//------------------------------------------------------------------------------------------------//
// Response <Addr> <Data> Buffer
//------------------------------------------------------------------------------------------------//

logic   [2:0]  apb_cnt;
logic          pkt_active;
logic   [15:0] pld_len;

logic          apb_axis_tlast;
logic   [63:0] apb_axis_tdata;
logic          apb_axis_tvalid;
logic          apb_axis_tready;

logic          rsp_axis_tlast;
logic   [7:0]  rsp_axis_tdata;
logic          rsp_axis_tvalid;
logic          rsp_axis_tready;

logic          vec_axis_tlast;
logic   [7:0]  vec_axis_tdata;
logic          vec_axis_tvalid;
logic          vec_axis_tready;

logic          roce_axis_tlast;
logic   [7:0]  roce_axis_tdata;
logic          roce_axis_tvalid;
logic          roce_axis_tready;

always_ff @(posedge i_aclk) begin
  if (i_arst) begin
    apb_axis_tdata  <= '0;
    apb_axis_tvalid <= '0;
    apb_cnt         <= '0;
  end
  else begin
    if (i_apb_s2m.pready || ((cmd_state == CMD_WAIT) && r_ecb_error)) begin
      apb_axis_tdata[63:32] <= ( cmd_wren || r_ecb_error ) ? '0 :
                                                            {i_apb_s2m.prdata[7:0],i_apb_s2m.prdata[15:8],i_apb_s2m.prdata[23:16],i_apb_s2m.prdata[31:24]};
      apb_axis_tdata[31:0]  <= {o_apb_m2s.paddr [7:0],o_apb_m2s.paddr [15:8],o_apb_m2s.paddr [23:16],o_apb_m2s.paddr [31:24]};
      apb_axis_tvalid       <= '1;
    end
    else begin
      if (apb_axis_tvalid) begin
        if (apb_cnt == '1) begin
          apb_axis_tvalid <= '0;
        end
        apb_cnt <= apb_cnt + 1;
      end
      else begin
        apb_cnt         <= '0;
        apb_axis_tvalid <= '0;
      end
    end
  end
end

assign apb_axis_tlast = ((apb_cnt == '1) && (decode_flush));

axis_buffer # (
  .IN_DWIDTH         ( 8              ),
  .OUT_DWIDTH        ( AXI_DWIDTH     ),
  .WAIT2SEND         ( 1              ),
  .BUF_DEPTH         ( MTU            ),
  .ALMOST_FULL_DEPTH ( MTU-40         ),
  .DUAL_CLOCK        ( 1              )
) u_axis_buffer (
  .in_clk            ( i_aclk                       ),
  .in_rst            ( i_arst                       ),
  .out_clk           ( i_pclk                       ),
  .out_rst           ( i_prst                       ),
  .i_axis_rx_tvalid  ( apb_axis_tvalid              ),
  .i_axis_rx_tdata   ( apb_axis_tdata[apb_cnt*8+:8] ),
  .i_axis_rx_tlast   ( apb_axis_tlast               ),
  .i_axis_rx_tuser   ( '0                           ),
  .i_axis_rx_tkeep   ( '1                           ),
  .o_axis_rx_tready  (                              ),
  .o_fifo_aempty     (                              ),
  .o_fifo_afull      (                              ),
  .o_fifo_empty      (                              ),
  .o_fifo_full       (                              ),
  .o_axis_tx_tvalid  ( rsp_axis_tvalid              ),
  .o_axis_tx_tdata   ( rsp_axis_tdata               ),
  .o_axis_tx_tlast   ( rsp_axis_tlast               ),
  .o_axis_tx_tuser   (                              ),
  .o_axis_tx_tkeep   (                              ),
  .i_axis_tx_tready  ( rsp_axis_tready              )
);

//------------------------------------------------------------------------------------------------//
// Response
//------------------------------------------------------------------------------------------------//


//APB Control Interface
assign o_apb_m2s.pwdata         = cmd_dout;
assign o_apb_m2s.paddr          = {cmd_addr_sync[31:2],2'b00};
assign o_apb_m2s.pwrite         = cmd_wren;
assign o_apb_m2s.psel           = cmd_en;

//ECB Error Logic
assign ecb_cmd_error            = !(ecb_cmd inside {WR_DWORD, RD_DWORD, WR_BLOCK,RD_BLOCK});
assign ecb_flag_error           = |ecb_flags[7:2];
assign ecb_addr_error           = cmd_err;
assign ecb_seq_chk_error        = ecb_flags[1] && (ecb_seq != seq_num_latch + 1'b1);
assign ecb_error                = ecb_cmd_error || ecb_flag_error || ecb_addr_error || ecb_seq_chk_error;

//ECB Response Command Code. If there is a command error, return that command that came in, else encode to match ECB definitions
assign ecb_cmd_code             = ecb_cmd_error ? ecb_cmd : {3'b100, ecb_rd, ecb_cmd[3:0]};

//ECB Response Code
always_comb begin
  if (ecb_cmd_error) begin
    ecb_resp_code               = 8'h04;
  end
  else if (ecb_flag_error) begin
    ecb_resp_code               = 8'h06;
  end
  else if (ecb_seq_chk_error_latched) begin
    ecb_resp_code               = 8'h0B;
  end
  else if (ecb_addr_error) begin
    ecb_resp_code               = 8'h03;
  end
  else begin
    ecb_resp_code               = 8'h00;
  end
end

//ECB Response Vector to AXIS Module. The ECB response is input to this module (triggered by a synced signal from the FSM).
//The FSM ensures this module is not busy before triggering it with new response data.
//The module then sends out an AXIS packet with the response information in it based on the width of the data path.
vec_to_axis #(
  .AXI_DWIDTH( AXI_DWIDTH   ),
  .DATA_WIDTH( RESP_WIDTH*8 )
) ecb_to_axis_rdresp (
  .clk              ( i_pclk                 ),
  .rst              ( i_prst                 ),
  .trigger          ( resp_trigger_sync      ),
  .data             ( resp_data              ),
  .is_busy          ( resp_vec_is_busy       ),
//AXIS Interface
  .o_axis_tx_tvalid ( vec_axis_tvalid        ),
  .o_axis_tx_tdata  ( vec_axis_tdata         ),
  .o_axis_tx_tlast  ( vec_axis_tlast         ),
  .o_axis_tx_tuser  (                        ),
  .o_axis_tx_tkeep  (                        ),
  .i_axis_tx_tready ( vec_axis_tready        )
);
//------------------------------------------------------------------------------
// RoCE Header
//------------------------------------------------------------------------------

logic [ROCE_HDR_WIDTH*8-1:0] hdr_roce;
logic [ROCE_HDR_WIDTH*8-1:0] roce_hdr_data;
logic                        roce_hdr_is_busy;

vec_to_axis #(
  .AXI_DWIDTH( AXI_DWIDTH       ),
  .DATA_WIDTH( ROCE_HDR_WIDTH*8 )
) ecb_to_axis_roce_hdr (
  .clk              ( i_pclk                             ),
  .rst              ( i_prst                             ),
  .trigger          ( resp_trigger_sync && ecb_is_roce   ),
  .data             ( roce_hdr_data                      ),
  .is_busy          ( roce_hdr_is_busy                   ),
//AXIS Interface        
  .o_axis_tx_tvalid ( roce_axis_tvalid                   ),
  .o_axis_tx_tdata  ( roce_axis_tdata                    ),
  .o_axis_tx_tlast  ( roce_axis_tlast                    ),
  .o_axis_tx_tuser  (                                    ),
  .o_axis_tx_tkeep  (                                    ),
  .i_axis_tx_tready ( roce_axis_tready                   )
);
 
logic [7:0] opcode;
logic [23:0] psn;
logic [7:0] se_m_pad_tver;
logic [31:0] wr_imm;

assign opcode = 'h2B;
assign se_m_pad_tver = {1'b0, 1'b0, 2'h0, 4'h0};
assign wr_imm = {16'h0, ecb_seq};

  assign hdr_roce = {
              opcode,se_m_pad_tver,roce_pkey,8'h0,roce_dest_qp,
              8'h0,psn,roce_vaddr,roce_rkey,
              16'h0,pld_len,wr_imm};

genvar j;
generate
  for (j=0; j<ROCE_HDR_WIDTH; j=j+1) begin
    assign roce_hdr_data[j*8+:8] = hdr_roce[(ROCE_HDR_WIDTH-1-j)*8+:8];
  end
endgenerate

assign o_axis_tuser = tuser;

//------------------------------------------------------------------------------------------------//
// CDC
//------------------------------------------------------------------------------------------------//

data_sync #(
  .DATA_WIDTH     ( 3                                                    ),
  .RESET_VALUE    ( 1'b0                                                 ),
  .SYNC_DEPTH     ( 2                                                    )
) u_apb_stat_sync (
  .clk            ( i_pclk                                               ),
  .rst_n          ( !i_prst                                              ),
  .sync_in        ( {decode_hdr,decode_data,decode_flush}                ),
  .sync_out       ( {decode_hdr_sync,decode_data_sync,decode_flush_sync} )
);

pulse_sync u_resp_trigger_sync (
  .src_clk        ( i_aclk                                               ),
  .src_rst        ( i_arst                                               ),
  .dst_clk        ( i_pclk                                               ),
  .dst_rst        ( i_prst                                               ),
  .i_src_pulse    ( resp_trigger                                         ),
  .o_dst_pulse    ( resp_trigger_sync                                    )
);

data_sync #(
  .DATA_WIDTH     ( 5                                                                                           ),
  .RESET_VALUE    ( 1'b0                                                                                        ),
  .SYNC_DEPTH     ( 2                                                                                           )
) u_apb_ctrl_sync (
  .clk            ( i_aclk                                                                                      ),
  .rst_n          ( !i_arst                                                                                     ),
  .sync_in        ( {resp_vec_is_busy,decode_done,decode_start,decode_last,ecb_is_roce}                         ),
  .sync_out       ( {resp_vec_is_busy_sync,decode_done_sync,decode_start_sync,decode_last_sync,ecb_is_roce_sync})
);

data_sync #(
  .DATA_WIDTH     ( 32                                                   ),
  .RESET_VALUE    ( 1'b0                                                 ),
  .SYNC_DEPTH     ( 2                                                    )
) u_cmd_addr (
  .clk            ( i_aclk                                               ),
  .rst_n          ( !i_arst                                              ),
  .sync_in        ( cmd_addr                                             ),
  .sync_out       ( cmd_addr_sync                                        )
);

data_sync #(
  .DATA_WIDTH     ( 16                                                   ),
  .RESET_VALUE    ( 1'b0                                                 ),
  .SYNC_DEPTH     ( 2                                                    )
) u_seq_num (
  .clk            ( i_pclk                                               ),
  .rst_n          ( !i_prst                                              ),
  .sync_in        ( seq_num_latch                                        ),
  .sync_out       ( seq_num_latch_sync                                   )
);

//------------------------------------------------------------------------------------------------//
// Combine Response + <addr><data> pairs
//------------------------------------------------------------------------------------------------//

// Adaptive ECB addr
enum logic [3:0] {RESP_IDLE, RESP_HDR,RESP_ROCE_HDR,RESP_DATA,RESP_SEQ,RESP_SEQ2,RESP_PAD,RESP_PAD2} resp_state;
logic        is_pad;

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    resp_state    <= RESP_IDLE;
    pld_len       <= '0;
    drop_packet   <= '0;
    is_pad        <= '0;
    psn           <= '0;
  end
  else begin
    pld_len      <= (is_pad) ? 16'h12 : (RESP_WIDTH+2) + (num_pairs<<3);
    is_pad       <= ((num_pairs == 'd1) || (num_pairs == 'd0));
    drop_packet  <= !(ecb_flags[0] || ecb_rd );
    if (i_axis_tready) begin
      case (resp_state)
        RESP_IDLE        : begin
          resp_state <= (vec_axis_tvalid && rsp_axis_tvalid) ? (ecb_is_roce) ? RESP_ROCE_HDR : RESP_HDR : RESP_IDLE;
        end
        RESP_ROCE_HDR    : begin
          resp_state <= (roce_axis_tlast) ? RESP_HDR : RESP_ROCE_HDR;
        end
        RESP_HDR         : begin
          resp_state <= (vec_axis_tlast) ? RESP_DATA : RESP_HDR;
        end
        RESP_DATA        : begin
          resp_state <= (rsp_axis_tlast) ? RESP_SEQ : RESP_DATA;
        end
        RESP_SEQ        : begin
          resp_state <= RESP_SEQ2;
          psn        <= psn+ecb_is_roce;
        end
        RESP_SEQ2        : begin
          resp_state <= (is_pad) ? RESP_PAD : RESP_IDLE;
        end
        RESP_PAD         : begin
          resp_state <= RESP_PAD2;
        end
        RESP_PAD2        : begin
          resp_state <= RESP_IDLE;
        end
      endcase
    end
  end
end

assign o_axis_tvalid  = (resp_state != RESP_IDLE) && !drop_packet;
assign o_axis_tdata   = (resp_state == RESP_ROCE_HDR) ? roce_axis_tdata  :
                        (resp_state == RESP_HDR ) ? vec_axis_tdata          :
                        (resp_state == RESP_DATA) ? rsp_axis_tdata          :
                        (resp_state == RESP_SEQ ) ? seq_num_latch_sync[15:8]:
                        (resp_state == RESP_SEQ2) ? seq_num_latch_sync[7:0] :
                                                   '0                       ;
assign o_axis_is_roce = ecb_is_roce;
assign o_axis_tlast   = ((resp_state == RESP_SEQ2) && !is_pad) || (resp_state == RESP_PAD2);
assign o_axis_tkeep   = '1;

assign vec_axis_tready = (i_axis_tready || drop_packet) && (resp_state == RESP_HDR);
assign roce_axis_tready = (i_axis_tready || drop_packet) && (resp_state == RESP_ROCE_HDR);
assign rsp_axis_tready = (i_axis_tready || drop_packet) && (resp_state == RESP_DATA);

//Map output AXIS to vec_to_axis module, depending on which interface is active

assign o_pld_len = pld_len + (ecb_is_roce ? ROCE_HDR_WIDTH+4 : 'd0);

//Swap endianness of resp bus
function [(RESP_WIDTH*8)-1:0] resp_end_swap (input [(RESP_WIDTH*8)-1:0] resp_in);
  begin
    for (int j = 0; j < RESP_WIDTH; j = j+1) begin
      resp_end_swap[j*8+:8] = resp_in[((RESP_WIDTH-1)-j)*8+:8];
    end
  end
endfunction

endmodule
