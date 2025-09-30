module avmm_to_apb #(
  parameter AVMM_ADDR_WIDTH = 32,
  parameter AVMM_DATA_WIDTH = 32,
  parameter USE_AVMM_READDATAVALID = 1,
  parameter AVMM_RD_LATENCY = 0         //Used if "USE_AVMM_READDATAVALID"=0
) (
  input                              clk,
  input                              rst,
  
  input                              psel,
  input                              penable,
  input        [31:0]                paddr,
  input                              pwrite,
  input        [31:0]                pwdata,
  output logic                       pready,
  output logic [31:0]                prdata,
  output logic                       pserr, 
  
  output logic [AVMM_ADDR_WIDTH-1:0] avmm_address,
  output logic                       avmm_write,
  output logic [AVMM_DATA_WIDTH-1:0] avmm_writedata,
  output logic                       avmm_read,
  input        [AVMM_DATA_WIDTH-1:0] avmm_readdata,
  input                              avmm_readdatavalid,
  input                              avmm_waitrequest
);

logic [2:0]       avmm_latency_cnt;
logic [8:0]       timer;
logic             timer_en;
logic             timeout;

localparam TIMEOUT_VAL          = 9'd250; 

typedef enum logic [2:0] {IDLE, APB_SETUP, AVMM_WR, AVMM_RD, APB_ACCESS} states;
states state;

always_ff @ (posedge clk) begin
  if (rst) begin
    pready         <= 1'b0;
    prdata         <= 0;
    pserr          <= 0;
    
    avmm_address   <= 0;
    avmm_write     <= 1'b0;
    avmm_writedata <= 0;
    avmm_read      <= 1'b0;

    avmm_latency_cnt <= 0;
    
    timer_en       <= 0;
    
    state          <= IDLE;
  end else begin
    case (state)
    IDLE: begin
      pready <= 1'b0;
      prdata <= 0;
    
      avmm_address   <= 0;
      avmm_write     <= 1'b0;
      avmm_writedata <= 0;
      avmm_read      <= 1'b0;
      avmm_latency_cnt <= 0;
      if (psel) begin
        state <= APB_SETUP;
      end
    end
    
    APB_SETUP: begin
      if (penable) begin
        avmm_address     <= paddr;
        timer_en         <= 1'b1;
        if (pwrite) begin
          avmm_write     <= pwrite;
          avmm_writedata <= pwdata;
          state          <= AVMM_WR;
        end else begin
          avmm_read      <= !pwrite;
          state          <= AVMM_RD;
        end
      end
    end
    
    AVMM_WR: begin
      if (!avmm_waitrequest || timeout) begin
        pready         <= 1'b1;
        pserr          <= timeout;
        avmm_address   <= 0;
        avmm_write     <= 1'b0;
        avmm_writedata <= 0;

        state  <= APB_ACCESS;
      end
    end
    
    AVMM_RD: begin
      if (USE_AVMM_READDATAVALID) begin
        if (!avmm_waitrequest) begin
          avmm_address   <= 0;
          avmm_read      <= 1'b0;
        end

        if (avmm_readdatavalid || timeout) begin
          pready <= 1'b1;
          prdata <= timeout ? 32'hBADADD12 : avmm_readdata;
          pserr  <= timeout;
          state  <= APB_ACCESS;
        end
      end else begin
        if ((!avmm_waitrequest && avmm_latency_cnt==AVMM_RD_LATENCY) || timeout) begin
          pready         <= 1'b1;
          prdata         <= timeout ? 32'hBADADD12 : avmm_readdata;
          pserr          <= timeout;

          avmm_address   <= 0;
          avmm_read      <= 1'b0;
          avmm_latency_cnt <= 0;
          state          <= APB_ACCESS;
        end else if (!avmm_waitrequest) begin
          avmm_latency_cnt <= avmm_latency_cnt + 1'b1;
        end
      end
    end

    APB_ACCESS: begin
      if (!psel) begin
        pready <= 1'b0;
        prdata <= 0;
        pserr  <= 1'b0;
        timer_en <= 0;
        avmm_address   <= 0;
        avmm_write     <= 1'b0;
        avmm_writedata <= 0;
        avmm_read      <= 1'b0;
        state  <= IDLE;
      end
    end
    
    default: begin
      state <= IDLE;
    end
    endcase
  end
end


always_ff @(posedge clk) begin
  if (rst) begin
    timer                 <= 0;
    timeout               <= 0;
  end else begin
    if (timer_en) begin
      if (timer != TIMEOUT_VAL) begin
        timer             <= timer + 1'b1;
      end else begin
        timeout           <= 1'b1;
      end
    end else begin
      timer               <= 0;
      timeout             <= 0;
    end
  end
end

endmodule
