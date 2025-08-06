
`define STATE_SPI_WIDTH                 3
`define CPOL                            0
`define CPHA                            1

module FPGA_spi_peri #(
)
(
    // System
    input       clk                 ,
    input       rst_n               ,

    input [1:0] spi_mode            ,// {CPHA,CPOL}

    input       write               ,//command a single byte write
    input       turnaround          ,//assert to trigger a turnaround in which clock is driven for <turnaround_len> cycle
    input       read                ,//command a single byte read

    input [1:0] spi_width           ,// 0x0 = Single SPI , 0x1 = Dual SPI , 0x2 = Quad SPI
    input [3:0] turnaround_len      ,//number of idle cycle for a turnaround 
    input [7:0] wr_data             ,//Data to be written.  This value is latched when write =1 and cmd_ack=1
    
    output logic      cmd_ack       , //pulse to indicate current command has been accepted and queued in the engine
    output logic [7:0] rd_data      , //last byte read on the spi interface
    output logic      rd_data_valid , //strobe indicating a new valid value on rd_data for a read byte

    output      busy                , //0 = bus idle , 1 = transaction in process 

    input CS_N                     , //SPI interface pins
    input SCK                      ,
    inout [3:0] SDIO                
);

typedef enum logic [3:0] {
    IDLE       = 4'h0, 
    READ       = 4'h1, 
    WRITE      = 4'h2, 
    TURN       = 4'h3
} spi_peri_fsm;
    
spi_peri_fsm state, nxt_state;



logic [3:0] SDIO_sync;

data_sync #(
    .DATA_WIDTH ( 4         )  
    ) spi_sync (
    .clk        ( clk       ),
    .rst_n      ( rst_n     ),
    .sync_in    ( SDIO      ),
    .sync_out   ( SDIO_sync )
);

wire wSCK;
//assign wSCK = (nxt_state != WRITE) ? SCK : !SCK;
logic wSCK_rise;
logic wSCK_fall;
logic SCK_prev;

wire [2:0] incr;
assign incr = (spi_width == 2'h1) ? 3'h2:   // Dual Spi
              (spi_width == 2'h2) ? 3'h4:   // Quad Spi
                                    3'h1;   // Single Spi

logic [3:0] sent_cnt, sent_cnt_r;
logic [7:0] rd_data_c;


assign SDIO[3:1] = (state == WRITE) ? {2'bz, wr_data[4'd7-sent_cnt_r]} : 3'bz;   // Single Spi;
assign SDIO[0]   = 1'bz;


always_ff @ (posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        SCK_prev <= 1'b0;
        state  <= IDLE;
        sent_cnt_r <= 8'h00;
        rd_data  <= 8'h00;
    end
    else if (CS_N ) begin
        SCK_prev <= 1'b0;
        state  <= IDLE;
        sent_cnt_r <= 8'h00;
        rd_data  <= 8'h00;
    end
    else begin
        SCK_prev <= SCK;
        if (((nxt_state != WRITE) && wSCK_fall) ||  ((nxt_state == WRITE) && wSCK_rise)) begin
          state <= nxt_state;
          sent_cnt_r <= sent_cnt;
          rd_data <= rd_data_c;
        end
    end
end

assign wSCK_rise = ({SCK,SCK_prev}==2'b10);
assign wSCK_fall = ({SCK,SCK_prev}==2'b01);

assign busy = !CS_N;

always_comb begin
    // Default signal values
    nxt_state     = state;
    sent_cnt      = sent_cnt_r;
    rd_data_valid = 1'b0;
      rd_data_c       = rd_data;
    cmd_ack       = 1'b0;
    
    if (CS_N) begin
      nxt_state = IDLE;
    end
    else begin
      case (state)
        IDLE: begin
          nxt_state = READ;
        end
        READ: begin
          if ((sent_cnt+incr) == 4'h8) begin
            cmd_ack = 1'b1;
            rd_data_valid = 1'b1;
            nxt_state = (write) ? WRITE : READ;
            sent_cnt = 4'h0;
          end 
          else begin
            sent_cnt = sent_cnt + incr;
            cmd_ack = 1'b0;
          end
          rd_data_c  = (spi_width == 2'h1) ? {rd_data[5:0], SDIO_sync[1:0]}:  // Dual Spi
                      (spi_width == 2'h2) ? {rd_data[3:0], SDIO_sync[3:0]}:  // Quad Spi
                                            {rd_data[6:0], SDIO_sync[0]  };  // Single Spi
        end
        WRITE: begin
            if ((sent_cnt+incr) == 4'h8) begin
              nxt_state = (write)       ? WRITE      :
                        (turnaround) ? TURN:
                                          READ       ;
              sent_cnt = 4'h0;
              cmd_ack = 1'b1;
            end 
            else begin
              cmd_ack = 1'b0;
              sent_cnt = sent_cnt + incr;
            end
        end
        TURN: begin
          if ((sent_cnt+1'b1) == turnaround_len) begin
            cmd_ack = 1'b1;
            nxt_state = (write) ? WRITE : READ;
            sent_cnt = 4'h0;
          end 
          else begin
            cmd_ack = 1'b0;
            sent_cnt = sent_cnt + 1'b1;
          end
          rd_data_c = 8'h0;
        end
        
      endcase
    end
end



endmodule

