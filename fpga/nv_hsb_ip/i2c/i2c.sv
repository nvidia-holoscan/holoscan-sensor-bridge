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


module i2c (
    input         clk,     // controller clock
    input         rst,     // synchronous active high reset
    input         nReset,  // asynchronous active low reset
    input         ena,     // core enable signal
    input  [31:0] i2c_scl_timeout,
    input  [15:0] clk_cnt,        // 4x SCL (default)

    // control inputs
    input         start,
    input         stop,
    input         read,
    input         write,
    input         ack_stretch,
    input         ack_in,
    input  [7:0]  din,

    // status outputs
    output logic  ack_out,
    output        cmd_ack,
    output        i2c_busy,
    output        i2c_al,
    output        idle,
    output [7:0]  dout,

    // I2C signals
    input         scl_i,
    output        scl_o,
    output        scl_oen,
    input         sda_i,
    output        sda_o,
    output        sda_oen
);

typedef enum logic [3:0] {
  I2C_IDLE,
  I2C_START,
  I2C_WRITE,
  I2C_READ,
  I2C_ACK_STRETCH,
  I2C_ACK,
  I2C_STOP,
  I2C_RESTART
} i2c_states;
i2c_states state, state_nxt, state_prev;

//------------------------------------------------------------------------------------------------//
// Outputs
//------------------------------------------------------------------------------------------------//

logic [7:0]  din_r;
logic [7:0]  dout_r;
logic [2:0]  idx;
logic [1:0]  pcnt;
logic [15:0] clk_div_cnt;
logic        clk_inc;
logic [31:0] timeout_cnt;
logic        al;
logic        tick;
logic        tick_high;
logic        tick_low;
logic        nack;
logic        ack;
logic        sda,sda_r;
logic        scl,scl_r;
logic        peri_ack;
logic        idle_delay;
logic        is_busy;
logic        peri_stretch;
logic        peri_stretch_r;

always_ff @(posedge clk) begin
  if (!nReset) begin
    state      <= I2C_IDLE;
    sda_r      <= '1;
    scl_r      <= '1;
    ack_out    <= '1;
    state_prev <= I2C_IDLE;
    al         <= '0;
    peri_ack   <= '1;
    idle_delay <= '0;
    is_busy    <= '0;
    dout_r     <= '0;
    idx        <= '0;
  end
  else begin
    idle_delay <= (state_prev == I2C_IDLE);  // Delay in Idle state for 1 extra cycle
    is_busy    <= (state==I2C_WRITE) ? 1'b1    :
                  (stop)             ? 1'b0    :
                                        is_busy ;
    if ((state != I2C_WRITE) && (state != I2C_READ)) begin
      idx   <= '1;
    end
    else if (tick) begin
      idx   <= idx - (pcnt == '1);
    end

    if (al) begin
      state <= I2C_IDLE;
    end
    else if (((pcnt == '1) && tick) || (state == I2C_IDLE && idle_delay)) begin
      state <= state_nxt;
      if (state == I2C_READ) begin
        dout_r[idx] <= sda_i;
      end
    end
    sda_r      <= (tick) ? sda : sda_r;
    scl_r      <= (((pcnt == 2'b11) || (pcnt == 2'b01)) && tick) ? scl : scl_r;
    state_prev <= state;
    ack_out    <= (state == I2C_IDLE)                                     ? 1'b1   :
                  ((state == I2C_ACK) && ((pcnt == 'd2) || peri_stretch)) ? sda_i  :
                                                                            ack_out;
    al         <= ((timeout_cnt == i2c_scl_timeout) && (state != I2C_IDLE));
    peri_ack   <= (state == I2C_WRITE) ? 1'b1     :
                  (state == I2C_READ ) ? ack_in   :
                                         peri_ack;
  end
end


// Register Inputs
always_ff @(posedge clk) begin
  if (!nReset) begin
    din_r <= '0;
  end
  else begin
    if (state == I2C_IDLE) begin
      din_r <= din;
    end
  end
end


//------------------------------------------------------------------------------------------------//
// Clock Divider + Peripheral Clock Stretch
//------------------------------------------------------------------------------------------------//

always_ff @(posedge clk) begin
  if (!nReset) begin
    pcnt           <= '0;
    clk_div_cnt    <= 'd1;
    timeout_cnt    <= '0;
    peri_stretch_r <= '0;
  end
  else begin
    if (state == I2C_IDLE) begin
      pcnt        <= '0;
      timeout_cnt <= '0;
    end
    else begin
      if (tick) begin
        clk_div_cnt <= 'd1;
        pcnt        <= pcnt + 1'b1;
      end
      else if (clk_inc && peri_stretch) begin // Reset the counter when the stretch starts
        clk_div_cnt    <= 'd1;
        peri_stretch_r <= '1;
      end
      else if (peri_stretch_r) begin  // Don't increment until the stretch is over
        clk_div_cnt    <= clk_div_cnt;
        peri_stretch_r <= peri_stretch;
      end
      else begin
        clk_div_cnt <= clk_div_cnt + 1'b1;
      end
      timeout_cnt <= (!sda_i || !scl || (is_busy)) ? timeout_cnt + 1'b1 : '0;
    end
  end
end
assign tick_low  = (clk_div_cnt >= ((clk_cnt>>1) + clk_cnt[0]));
assign tick_high = (clk_div_cnt >= (clk_cnt>>1));
assign clk_inc = ((tick_high && (pcnt[1])) || (tick_low));
assign tick = clk_inc && !peri_stretch;
assign peri_stretch = (scl_oen && !scl_i);

//------------------------------------------------------------------------------------------------//
// State Transitions
//------------------------------------------------------------------------------------------------//
always_comb begin
  state_nxt = state;
  case (state)
    I2C_IDLE: begin
      if ((read | write | stop) & (!cmd_ack)) begin
        state_nxt = (start && is_busy) ? I2C_RESTART:
                    (start)            ? I2C_START:
                    (read )            ? I2C_READ:
                    (write)            ? I2C_WRITE:
                    (stop )            ? I2C_STOP:
                                          I2C_IDLE;
      end
    end
    I2C_START: begin
      state_nxt = I2C_WRITE;
    end
    I2C_WRITE: begin
      state_nxt = (idx == '0) ? I2C_ACK : I2C_WRITE;
    end
    I2C_READ: begin
      state_nxt = (idx == '0) ? (ack_stretch) ? I2C_ACK_STRETCH : I2C_ACK : I2C_READ;
    end
    I2C_ACK_STRETCH: begin
      state_nxt = (ack_stretch) ? I2C_ACK_STRETCH : I2C_ACK;
    end
    I2C_ACK: begin
      state_nxt = (stop || (peri_ack & ack_out)) ? I2C_STOP : I2C_IDLE;
    end
    I2C_STOP: begin
      state_nxt = I2C_IDLE;
    end
    I2C_RESTART: begin
      state_nxt = I2C_WRITE;
    end
  endcase
end

//------------------------------------------------------------------------------------------------//
// I2C IO
//------------------------------------------------------------------------------------------------//

always_comb begin
  sda = sda_r;
  scl = scl_r;
  case (state)
    I2C_IDLE: begin
      sda = sda_r;
      scl = scl_r;
    end
    I2C_START: begin
      if (pcnt == 'd2) begin
        sda = 1'b0;
      end
      scl = 1'b1;
    end
    I2C_WRITE: begin
      if (pcnt == 'd1) begin
        sda = din_r[idx];
      end
      scl = !scl;
    end
    I2C_READ: begin
      if (pcnt == 'd1) begin
        sda = 1'b1;
      end
      scl = !scl;
    end
    I2C_ACK_STRETCH: begin
      sda = 1'b0;
      scl = !scl;
    end
    I2C_ACK: begin
      if (pcnt == 'd1) begin
        sda = peri_ack;
      end
      scl = !scl;
    end
    I2C_STOP: begin
      if (pcnt == 'd3) begin
        sda = 1'b1;
      end
      else if (pcnt == 'd1) begin
        sda = 1'b0;
      end
      scl = !scl;
    end
    I2C_RESTART: begin
      if (pcnt == 'd3) begin
        sda = 1'b0;
      end
      scl = !scl_r;
    end
  endcase
end

assign cmd_ack  = ((state == I2C_IDLE) && (state_prev != I2C_IDLE));
assign i2c_busy = (state != I2C_IDLE);
assign i2c_al   = al;
assign idle     = !i2c_busy;
assign dout     = dout_r;

// I2C signals
assign scl_o   = '0;
assign scl_oen = scl;
assign sda_o   = '0;
assign sda_oen = sda;






endmodule
