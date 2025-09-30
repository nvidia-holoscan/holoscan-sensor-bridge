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
