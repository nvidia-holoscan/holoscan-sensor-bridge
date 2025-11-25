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

module dbg_led (

  input   logic           clk,
  input   logic           rst,
  output  logic [15:0]    led

);

localparam LED_TICK = 27'h455BAE;
localparam SECOND   = 27'h5F5E0F4;


logic   [26:0]      led_cnt;


enum logic [1:0] {RIGHT, LEFT, PAUSE} state;


always_ff @(posedge clk) begin
  if (rst) begin
    state                 <= RIGHT;
    led_cnt               <= '0;
    led                   <= 16'b0000000000000001;
  end else begin
    case (state)
    
      RIGHT: begin
        if (led_cnt == LED_TICK) begin
          led_cnt         <= '0;
          if (led[15]) begin
            state         <= LEFT;
          end else begin
            led           <= {led[14:0], 1'b0};
          end
        end else begin
          led_cnt         <= led_cnt + 1'b1;
        end
      end
      
      LEFT: begin
        if (led_cnt == LED_TICK) begin
          led_cnt         <= '0;
          if (led[0]) begin
            state         <= PAUSE;
          end else begin
            led           <= {1'b0, led[15:1]};
          end
        end else begin
          led_cnt         <= led_cnt + 1'b1;
        end
      end
      
      PAUSE: begin
        if (led_cnt == SECOND) begin
          led_cnt         <= '0;
          state           <= RIGHT;
        end else begin
          led_cnt         <= led_cnt + 1'b1;
        end
      end
    endcase
  end
end  

endmodule
