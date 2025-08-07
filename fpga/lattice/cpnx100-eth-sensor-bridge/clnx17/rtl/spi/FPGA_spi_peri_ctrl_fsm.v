
`define STATE_CTRL_WIDTH   3

`define CMD_WR_BYTE         8'h01
`define CMD_RD_BYTE         8'h11
`define CMD_RMW_BYTE        8'h0A       //READ MODIFY WRITE Byte

module FPGA_spi_peri_ctrl_fsm  # (
        parameter [15:0] FPGA_VERSION = 16'h0100
    ) (
    input           clk             ,
    input           rst_n           ,
    // Regtbl Ports
    output [7:0]    mipi_debug_en   ,
    output [1:0]    mipi_en         ,
    output [1:0]    soft_rstn       ,
    output          spi_flash_fwd   ,
    output [7:0]    debug_length    ,
    // LMMI Ctrl Interface
    input  [1:0]    lmmi_ready      ,
    input  [7:0]    lmmi_rdata [1:0],
    input  [1:0]    lmmi_rdata_valid,
    output [7:0]    lmmi_wdata      ,
    output          lmmi_wr_rdn     ,
    output [7:0]    lmmi_offset     ,
    output [1:0]    lmmi_request    ,
    //Spi Ports
    input           CS_N            ,
    input           SCK             ,
    inout  [3:0]    SDIO
);

typedef enum logic [3:0] {
    IDLE,
    CMD ,
    RD_ADDR  ,
    RD_DATA ,
    RD_MASK,
    WR_DATA ,
    WAIT_ACK,
    ERROR,
    DONE
} spi_ctrl_peri_fsm;
    
spi_ctrl_peri_fsm ctrl_state, nxt_ctrl_state;

//Regtbl:      0x000 - 0x00F
// Ctrl Regtb
wire        busy           ;
reg [7:0]   regtbl [15:0]  ;

reg        wr_en    ;
reg  [8:0]  addr     ;
reg  [7:0]  data_wr  ;
reg  [7:0]  data_rd  ;


// Interface
wire        write          ;
wire        turnaround     ;
wire        read           ;
wire        rd_data_valid  ;
wire        cmd_ack        ;

reg [7:0]   spi_wr_data    ;
wire [7:0]  rd_data        ;



FPGA_spi_peri spi_peri (
    .clk             ( clk              ),
    .rst_n           ( rst_n            ),
    .spi_mode        ( 2'b11            ),
    .write           ( write            ),
    .turnaround      ( turnaround       ),
    .read            ( read             ),
    .spi_width       ( 2'b00            ),
    .turnaround_len  ( 4'h0             ), 
    .wr_data         ( spi_wr_data      ),
    .cmd_ack         ( cmd_ack          ),
    .rd_data         ( rd_data          ),
    .rd_data_valid   ( rd_data_valid    ),
    .busy            ( busy             ),
    .CS_N            ( CS_N             ),
    .SCK             ( SCK              ),
    .SDIO            ( SDIO             )
);

// Spi Interface FSM
//spyglass disable_block STARC05-2.10.3.1
//spyglass disable_block W362

reg cmd_ack_prev;
logic [7:0] cmd, cmd_r;
logic [7:0] rt_addr, rt_addr_r;
logic [7:0] rt_wr_data, rt_wr_data_r;
integer i;
logic state_change, state_change_r;
logic rt_wr_en;


always_ff @ (posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        ctrl_state  <= IDLE;
        cmd_ack_prev <= 1'b0;
    end
    else if (CS_N ) begin
        ctrl_state  <= IDLE;
        cmd_ack_prev <= 1'b0;
    end
    else begin
        ctrl_state <= nxt_ctrl_state;
        cmd_ack_prev <= cmd_ack;
        cmd_r <= cmd;
        rt_addr_r <= rt_addr;
        rt_wr_data_r <= rt_wr_data;
        state_change_r <= state_change;
    end
end

assign write      = (ctrl_state == WR_DATA);
assign turnaround = 1'b0;
assign read       = ((ctrl_state == CMD) || (ctrl_state == RD_ADDR) || (ctrl_state == RD_DATA) || (ctrl_state == RD_MASK));

always_comb begin
    // Default signal values
    nxt_ctrl_state     = ctrl_state;
    rt_wr_en = 1'b0;
    spi_wr_data = 8'h00;
    cmd = cmd_r;
    rt_addr = rt_addr_r;
    rt_wr_data = rt_wr_data_r;
    state_change = state_change_r;

    if (CS_N) begin
      nxt_ctrl_state = IDLE;
      state_change = 1'b0;
    end
    else begin
        if (!spi_flash_fwd) begin
            case (ctrl_state) // spyglass disable DefaultState STARC05-2.8.1.4 W71
                IDLE: begin
                nxt_ctrl_state = CMD;
                state_change = 1'b0;     
                end
                CMD: begin
                    if (rd_data_valid) begin
                        cmd = rd_data;
                        nxt_ctrl_state = (rd_data == `CMD_WR_BYTE) || (rd_data[4]) || (rd_data == `CMD_RMW_BYTE) ? RD_ADDR : ERROR;
                        state_change = 1'b1;
                    end
                end
                RD_ADDR: begin
                    if (state_change && !cmd_ack) begin
                        state_change = 1'b0;
                    end
                    else if (!state_change && rd_data_valid) begin
                        rt_addr = rd_data;
                        nxt_ctrl_state =  (cmd == `CMD_WR_BYTE) || (cmd == `CMD_RMW_BYTE) ? RD_DATA : 
                                          (cmd[4])                           ? WR_DATA :
                                                                                            ERROR   ;
                        state_change = 1'b1;
                    end
                end
                RD_DATA: begin
                    if (state_change && !cmd_ack) begin
                        state_change = 1'b0;
                    end
                    else if (!state_change && rd_data_valid) begin
                        if (cmd == `CMD_WR_BYTE) begin
                            rt_wr_en = 1'b1;
                        end
                        rt_wr_data = rd_data;
                        nxt_ctrl_state = (cmd == `CMD_WR_BYTE) ? DONE    : 
                                        (cmd == `CMD_RMW_BYTE) ? RD_MASK :
                                                                ERROR   ;     
                        state_change = 1'b1;                                                    
                    end
                end
                RD_MASK: begin
                    if (state_change && !cmd_ack) begin
                        state_change = 1'b0;
                    end
                    else if (!state_change && rd_data_valid) begin
                        rt_wr_en = 1'b1;
                        nxt_ctrl_state = DONE;            
                        state_change = 1'b1;                                            
                    end
                end
                WR_DATA: begin
                    spi_wr_data = regtbl[rt_addr];
                    if (state_change && !cmd_ack) begin
                        state_change = 1'b0;
                    end
                    else if (!state_change && cmd_ack) begin
                        nxt_ctrl_state = ({cmd_ack,cmd_ack_prev} == 2'b01) ? DONE : WR_DATA;  
                        state_change = 1'b1;                                                    
                    end
                end
                ERROR: begin
                    nxt_ctrl_state = ERROR;
                end
            DONE: begin
                    nxt_ctrl_state = DONE;
                end
            endcase
        end
    end
end


//spyglass enable_block STARC05-2.10.3.1
//spyglass enable_block SelfDeterminedExpr-ML
//spyglass enable_block W362

//------------------------------------------------------------------------------
// REGTBL
//------------------------------------------------------------------------------


reg [7:0] CS_N_cnt;
reg CS_N_prev;
logic [1:0] lmmi_rdata_valid_prev;

always@(posedge clk or negedge rst_n)  
begin
    if (!rst_n) begin
        // Reset regtbl
        for (i=0; i<16; i=i+1) begin
            regtbl[i] <= 8'h00;
        end
        // Set non-zero default values
        regtbl[0]  <= FPGA_VERSION[7:0];
        regtbl[1]  <= FPGA_VERSION[15:8];
        regtbl[7]  <= 8'h0F;
        regtbl[11] <= 8'hB0;
        // Other Signals
        CS_N_cnt   <= 8'h0;
        CS_N_prev  <= 1'b1;
        lmmi_rdata_valid_prev <= 2'b00;
    end
    else begin
        CS_N_prev <= CS_N;
        lmmi_rdata_valid_prev <= lmmi_rdata_valid;
        // Write user addressable bytes
        if (!CS_N && rt_wr_en) begin
            if (cmd == `CMD_WR_BYTE) begin
                regtbl[rt_addr] <= rd_data;
            end
            if (cmd == `CMD_RMW_BYTE) begin
                for (i=0;i<8;i=i+1) begin
                    regtbl[rt_addr][i] <= rd_data[i] ? rt_wr_data[i] : regtbl[rt_addr][i];
                end
            end
        end
        
        // SPI Flash Forward Count
        if ({CS_N,CS_N_prev} == 2'b10) begin // End of transaction detect
            if (regtbl[6][0]) begin
                if (CS_N_cnt >= regtbl[6][7:4]) begin
                    CS_N_cnt <= 8'h0;
                    regtbl[6][0] <= 1'b0;
                end
                else begin
                    CS_N_cnt <= CS_N_cnt + 1'b1;
                end
            end
            
        end
        //Write status signals to regtbl data bytes
        else if (CS_N) begin 
            regtbl[12][3:2] <= lmmi_ready;
            regtbl[12][5:4] <= 2'b0;    // Clear Start after SPI Transaction termination
            
            if ({lmmi_rdata_valid_prev[0],lmmi_rdata_valid[0]} == 2'b10) begin
                regtbl[14] <= lmmi_rdata[0];
            end
            else if ({lmmi_rdata_valid_prev[1],lmmi_rdata_valid[1]} == 2'b10) begin
                regtbl[14] <= lmmi_rdata[1];
            end
        end 

    end
end

assign mipi_debug_en   = regtbl[8];
assign mipi_en         = regtbl[7][1:0];
assign soft_rstn       = regtbl[7][3:2];
assign spi_flash_fwd   = (CS_N_cnt != 0);

assign lmmi_wdata      = regtbl[13];
assign lmmi_wr_rdn     = regtbl[12][0];
assign lmmi_offset     = regtbl[15];
assign lmmi_request    = regtbl[12][5:4];

assign debug_length    = regtbl[11];

endmodule