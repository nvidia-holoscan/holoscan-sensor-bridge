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


`define STATE_SPI_WIDTH                 3
`define BITS_PER_WORD                   8
`define CPOL                            0
`define CPHA                            1
`define READ                            2
`define PAUSE                           3
`define WRITE                           4

module spi #(
    parameter  NUM_TARGETS = 1,
    localparam MAX_TARGET  = NUM_TARGETS -1
)
(
    // System
    input                 clk,
    input                 rst_n,

    input  [3:0]          prescaler,  // Indicates number of clocks for each edge of the SCLK, 0 = 1 clock cycle.  This allows slowing down the bus
    input  [1:0]          spi_mode,   // {CPHA,CPOL}: CPOL has to change at least one clock cycle before a new command to allow correct SCK edge direction
    input  [2:0]          targ_sel,   // Selects which downstream target to access.

    input                 start,      // command the start of a SPI transaction.  Must have write =1 at the same time
    input                 write,      // command a single byte write
    input                 turnaround, // assert to trigger a turnaround in which clock is driven for <turnaround_len> cycle
    input                 read,       // command a single byte read
    input                 stop,       // indicate that the current OP is the end of the transaction


    input  [1:0]          spi_width,      // 0x0 = Single SPI , 0x1 = Dual SPI , 0x2 = Quad SPI
    input  [7:0]          turnaround_len, //number of idle cycle for a turnaround
    input  [7:0]          wr_data,        //Data to be written.  This value is latched when write =1 and cmd_ack=1

    output                cmd_ack,        //pulse to indicate current command has been accepted and queued in the engine

    output [7:0]          rd_data,        //last byte read on the spi interface
    output                rd_data_valid,  //strobe indicating a new valid value on rd_data for a read byte

    output                busy,           //0 = bus idle , 1 = transaction in process

    output [MAX_TARGET:0] CS_N, //SPI interface pins
    output                SCK,
    output                SDIO_en,       // Tri-State output enable.  "Pad" is at the top level.
    input  [3:0]          SDIO_in,
    output [3:0]          SDIO_out
);



reg  [`STATE_SPI_WIDTH-1:0]      rSpiState;

localparam [`STATE_SPI_WIDTH-1:0]
STATE_SPI_IDLE                         = `STATE_SPI_WIDTH'h0,
STATE_SPI_SEND                         = `STATE_SPI_WIDTH'h1,
STATE_SPI_WAIT                         = `STATE_SPI_WIDTH'h2,
STATE_SPI_TURNAROUND                   = `STATE_SPI_WIDTH'h3,
STATE_SPI_CS_SETUP                     = `STATE_SPI_WIDTH'h4;  // Fixed CS setup delay before first SCK edge

// SPI FSM

reg [3:0]           rSDIO;
reg                 rSDIO_en;
reg                 rSCK;
reg [MAX_TARGET:0]  rCS_N;

// Counts number of prescaler expirations to implement a fixed CS setup delay (2 SCK cycles = 4 half cycles)
reg [2:0]           rCsSetupCnt;

reg [3:0]                    rCycleCnt;
reg [`BITS_PER_WORD-1:0]     rWrData;
reg [5:0]                    rSentCnt;
reg [5:0]                    rSentCntMax;
reg [7:0]                    rRdData;

reg rCmdAck;
reg rRdDataValid;

reg [4:0]                    rSpiCmd;
reg [7:0]                    rTurnaroundLen;
reg [1:0]                    rSpiWidth;
reg [3:0]                    rPreScaler;
reg [2:0]                    incr;


always_comb begin
  case (rSpiWidth)
    2'h0:   begin           // Single SPI
      incr        = 3'h1;
      rSentCntMax = 6'h7;
    end
    2'h1:   begin           // Dual SPI
      incr        = 3'h2;
      rSentCntMax = 6'h6;
    end
    2'h2:   begin           // Quad SPI
      incr        = 3'h4;
      rSentCntMax = 6'h4;
    end
    default:    begin
      incr        = 3'h1;
      rSentCntMax = 6'h7;
    end
  endcase
end

always@(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    rSpiState       <=  STATE_SPI_IDLE ;
    rWrData         <=  8'h00;
    rRdData         <=  8'h00;
    rCS_N           <=  {NUM_TARGETS{1'b1}};
    rCmdAck         <= 1'b0;
    rSCK            <= 1'b0;
    rPreScaler      <= 4'h0;
    rSpiCmd         <= 5'h0;
    rTurnaroundLen  <= 8'h0;
    rSpiWidth       <= 2'h0;
    rRdDataValid    <= 1'b0;
    rCycleCnt       <= 4'h0;
    rSentCnt        <= 6'h0;
    rSDIO_en        <= 1'b0;
    rSDIO           <= 4'b0000;
    rCsSetupCnt     <= 3'h0;
  end
  else begin
    case(rSpiState)
      STATE_SPI_IDLE: begin
        rPreScaler          <= prescaler;
        rSpiCmd             <= {write,stop, read, spi_mode};
        rTurnaroundLen      <= turnaround_len;
        rSpiWidth           <= spi_width;
        rRdDataValid        <= 1'b0;
        rRdData             <= 8'h00;
        rWrData             <= wr_data;
        rCycleCnt           <= 4'h0;
        rSentCnt            <= 6'h0;
        if (start) begin
          // Assert CS and hold SCK at CPOL. Insert fixed CS setup delay (2 SCK cycles) before first SCK edge.
          rSpiState           <= STATE_SPI_CS_SETUP;
          rCmdAck             <= 1'b1;
          if (targ_sel < NUM_TARGETS) begin
            rCS_N[targ_sel]     <= 1'b0;
          end
          rSCK         <= spi_mode[`CPOL];
          rCsSetupCnt  <= 3'h0;
          rSDIO_en     <= (!write) ? 1'b0 : 1'b1;
          rSDIO        <= (!write)            ? 4'b0000 :
                          (spi_width == 2'h1) ? {2'b11, wr_data[7:6]} :  // Dual Spi
                          (spi_width == 2'h2) ? wr_data[7:4] :           // Quad Spi
                                                {3'b110, wr_data[7]};    // Single Spi
        end
        else begin
          rSpiState       <= STATE_SPI_IDLE;
          rCmdAck         <= 1'b0;
          rSCK            <= spi_mode[`CPOL];
          rCS_N           <= {NUM_TARGETS{1'b1}};
          rSDIO_en        <= 1'b0;
          rSDIO           <= 4'b0000;
        end
      end
      // Fixed CS setup delay: wait 4 prescaler intervals (2 full SCK cycles) before first toggle
      STATE_SPI_CS_SETUP: begin
        rCmdAck      <= 1'b0;
        rRdDataValid <= 1'b0;
        if (rCycleCnt == rPreScaler) begin
          rCycleCnt    <= 4'h0;
          if (rCsSetupCnt == 3'h3) begin
            // After 4 half-cycles worth of time, move to CPHA-selected phase
            rSpiState <= (rSpiCmd[`CPHA]) ? STATE_SPI_WAIT : STATE_SPI_SEND;
          end
          else begin
            rCsSetupCnt <= rCsSetupCnt + 1'b1;
            rSpiState   <= STATE_SPI_CS_SETUP;
          end
        end
        else begin
          rCycleCnt <= rCycleCnt + 1'b1;
        end
      end

      STATE_SPI_SEND: begin
        rCmdAck        <= 1'b0;
        if (rCycleCnt == rPreScaler) begin
          rCycleCnt   <= 4'h0;
          rSpiState   <= STATE_SPI_WAIT;
          rSCK        <= ~rSCK;
          rWrData     <= rWrData << incr;
          rSentCnt    <= rSentCnt + incr;
          if (rSpiCmd[`READ]) begin
            if (rSpiWidth == 2'h0) begin
              rRdData <= {rRdData[6:0],SDIO_in[1]};
            end
            else if (rSpiWidth == 2'h1) begin
              rRdData <= {rRdData[5:0],SDIO_in[1:0]};
            end
            else if (rSpiWidth == 2'h2) begin
              rRdData <= {rRdData[3:0],SDIO_in[3:0]};
            end
            else begin
              rRdData <= {rRdData[6:0],SDIO_in[1]};
            end
          end
          rRdDataValid <= ((rSentCnt == rSentCntMax) && rSpiCmd[`READ]);
        end
        else begin
          rCycleCnt <= rCycleCnt + 1'b1;
          rSpiState <= STATE_SPI_SEND;
        end
      end

      STATE_SPI_WAIT: begin
        rRdDataValid <= 1'b0;
        if (rCycleCnt == rPreScaler) begin
          rCycleCnt   <= 4'h0;
          if (rSentCnt < 6'h8) begin
            rSCK      <= ~rSCK;
            rSpiState <=  STATE_SPI_SEND;
            rSDIO_en  <= (!rSpiCmd[`WRITE])  ? 1'b0 : 1'b1;             // No write
            rSDIO     <= (!rSpiCmd[`WRITE])  ? 4'b0000 :                // No write
                         (rSpiWidth == 2'h1) ? {2'b11, rWrData[7:6]} :  // Dual Spi
                         (rSpiWidth == 2'h2) ? rWrData[7:4] :           // Quad Spi
                                               {3'b110, rWrData[7]};    // Single Spi
          end
          else if (rSpiCmd[`PAUSE]) begin
            rSDIO_en          <= 1'b0;
            rSDIO             <= 4'b0000;
            rSpiState         <= STATE_SPI_IDLE;
            rCS_N             <= {NUM_TARGETS{1'b1}};
          end
          else if (turnaround) begin
            rSentCnt        <= 6'h0;
            rSpiCmd         <= {write, stop, read,spi_mode};
            rSpiState       <= STATE_SPI_TURNAROUND;
            rCmdAck         <= 1'b1;
            rSCK            <= ~rSCK;
            rSDIO_en        <= 1'b0;
            rSDIO           <= 4'b0000;
          end
          else begin
            rSCK                <= ~rSCK;
            rSpiCmd             <= {write, stop, read,spi_mode};
            rTurnaroundLen      <= turnaround_len;
            rSpiWidth           <= spi_width;
            rPreScaler          <= prescaler;
            rSpiState           <= STATE_SPI_SEND;
            rCmdAck             <= 1'b1;
            rSDIO_en            <= (!write)            ? 1'b0 : 1'b1;
            rSDIO               <= (!write)            ? 4'b0000 :
                                   (spi_width == 2'h1) ? {2'b11, wr_data[7:6]} :  // Dual Spi
                                   (spi_width == 2'h2) ? wr_data[7:4] :           // Quad Spi
                                                         {3'b110, wr_data[7]};    // Single Spi
            rWrData             <= wr_data;
            rSentCnt            <= 6'h0;
          end
        end
        else begin
          rCycleCnt <= rCycleCnt + 1'b1;
        end
      end
      STATE_SPI_TURNAROUND: begin
        rRdDataValid <= 1'b0;
        rCmdAck      <= 1'b0;
        if (rCycleCnt == rPreScaler) begin
          rCycleCnt   <= 4'h0;
          rSCK        <= ~rSCK;
          if (rSentCnt == ((rTurnaroundLen-1)*2)) begin   // Last SCK edge in WAIT state
            rSpiState   <= STATE_SPI_WAIT;
            rCycleCnt   <= 4'h0;
            rSentCnt    <= 6'h8;
          end
          else begin
            rSentCnt <= rSentCnt + 1'b1;
          end
        end
        else begin
          rCycleCnt <= rCycleCnt + 1'b1;
        end
      end
    endcase
  end
end



assign SCK = rSCK;
assign SDIO_en  = rSDIO_en;
assign SDIO_out = rSDIO;
assign CS_N = rCS_N;
assign rd_data = rRdData;
assign rd_data_valid = rRdDataValid;
assign cmd_ack = rCmdAck;
assign busy = (rSpiState != STATE_SPI_IDLE);


endmodule
