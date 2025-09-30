
module eth_adapt_sequencer (
  input   logic                           clk,
  input   logic                           rst,
              
  input   logic                           start_seq,
  input   logic   [7:0]                   seq_start_addr,
  input   logic   [7:0]                   seq_end_addr,
  output  logic   [31:0]                  seq_return_data,
  output  logic   [31:0]                  seq_last_attr_info,
  output  logic   [31:0]                  seq_status,
  
  output  logic                           psel,
  output  logic                           penable,
  output  logic   [31:0]                  paddr,
  output  logic                           pwrite,
  output  logic   [31:0]                  pwdata,
  input   logic                           pready,
  input   logic   [31:0]                  prdata,
  input   logic                           pserr,
  //Status output
  output  logic   [31:0]                  stat_cmd_addr,
  output  logic   [7:0]                   stat_cmd_data,
  output  logic   [7:0]                   stat_cmd_mask,
  output  logic   [7:0]                   stat_cmd_dataval,
  output  logic   [2:0]                   stat_cmd_type,
  output  logic   [7:0]                   stat_rom_addr,
  output  logic   [3:0]                   stat_seq_state
);


//1) IDLE - Wait here until a rising edge is seen to start a sequence.
//  - Start address and end address need to be valid at start.
//  - XCVR channel needs to be valid at start.
//  - If the cmd_data is to be overriden, then override data needs to be valid at start.
//  - Transition to READ_ROM at start. 
//
//2) READ_ROM - grab data at ROM address
//  - Transition to DECODE
//
//3) DECODE - Grab all the fields from the ROM data output.
//  - Transition to SEND_CMD
//
//4) SEND_CMD - Issue the read or write command to the selected XCVR. 
//  - If the write data is to be overriden, then send the override data with write command.
//  - Wait until command is completed and transition as appropriate.
//    - If POLL or READ-AND-VALIDATE then transition to CHECK_DATA.
//    - If READ-MODIFY-WRITE then transition to MOD_DATA.
//    - Else, transition to READ_ROM.
//    - If POLL, a timer is started so that a timeout error can occur if POLL response is never valid.
//  
//5) CHECK_DATA - Use the mask and the data value from the command and check that the returned data matches
//  - If data does not match:
//    - If POLL and timer has not expired, then go back to SEND_CMD and issue command again.
//    - If POLL and timer has expired, transition to CMD_ERROR.
//    - If READ-AND-VALIDATE, set error status transition to CMD_ERROR.
//  - If data does match, stop timer and transition to READ_ROM.
//
//6) MOD_DATA - Modify the read back data with the mask (OR mask into read data)
//  - Transition to SEND_CMD to issue a write with the modified data.
//
//7) CMD_ERROR - Transition to IDLE if error is encountered.

localparam TMR_VAL  =  32'h05F5E100;
localparam CMD_WR   =  3'h1;
localparam CMD_RD   =  3'h2;
localparam CMD_VAL  =  3'h3;
localparam CMD_POLL =  3'h4;
localparam CMD_RMW  =  3'h5;

logic             start_seq_reg;
logic             seq_done;
logic [7:0]       rom_addr;
logic [3:0]       cmd_chan;
logic [31:0]      cmd_addr;
logic [7:0]       cmd_data;
logic [7:0]       cmd_mask;
logic             cmd_override;
logic [7:0]       cmd_override_data;
logic [7:0]       cmd_dataval;
logic [2:0]       cmd_type;
logic [7:0]       read_data;
logic [31:0]      timer_cnt;
logic             timer_en;
logic [71:0]      rom_dout;
logic [31:0]      return_val;
logic [15:0]      last_attr_data;
logic [15:0]      last_attr_code;
logic             data_verify_fail;

enum logic [3:0] {IDLE, READ_ROM, DECODE, SEND_CMD, MOD_DATA, CHECK_DATA, CMD_ERROR} seq_state;

assign seq_done           = rom_addr > seq_end_addr;
assign data_verify_fail   = !((read_data & cmd_mask) == cmd_dataval);


always_ff @(posedge clk) begin
  if (rst) begin
    seq_state                       <= IDLE;
    rom_addr                        <= 8'b0;
    start_seq_reg                   <= 1'b0;
    cmd_chan                        <= 4'b0;
    cmd_addr                        <= 32'b0;
    cmd_data                        <= 8'b0;
    cmd_mask                        <= 8'b0;
    cmd_dataval                     <= 8'b0;
    cmd_type                        <= 3'b0;
    cmd_override                    <= 1'b0;
    cmd_override_data               <= 8'b0;
    read_data                       <= 8'b0;
    timer_en                        <= 1'b0;
    seq_return_data                 <= 32'b0;
    seq_status                      <= 32'b0;
    stat_cmd_addr                   <= 32'b0;
    stat_cmd_data                   <= 8'b0;
    stat_cmd_mask                   <= 8'b0;
    stat_cmd_dataval                <= 8'b0;
    stat_cmd_type                   <= 3'b0;
    stat_rom_addr                   <= 8'b0;
    stat_seq_state                  <= 4'b0;
  end else begin
    //Stats
    stat_cmd_addr                   <= cmd_addr;
    stat_cmd_data                   <= cmd_data;
    stat_cmd_mask                   <= cmd_mask;
    stat_cmd_dataval                <= cmd_dataval;
    stat_cmd_type                   <= cmd_type;
    stat_rom_addr                   <= rom_addr;
    stat_seq_state                  <= seq_state;
  
    start_seq_reg                   <= start_seq;
    seq_return_data                 <= return_val;
    
    case (seq_state) 
      
      IDLE:   begin
        seq_status                  <= 0;
        if (start_seq && !start_seq_reg) begin
          seq_state                 <= READ_ROM;
          rom_addr                  <= seq_start_addr;
        end
      end
      
      READ_ROM: begin
        if (seq_done) begin
          seq_state                 <= IDLE;
          seq_status                <= 32'h00000001;
        end else begin
          seq_state                 <= DECODE;  
        end
      end
      
      DECODE: begin
        cmd_addr                    <= rom_dout[59:28];
        cmd_data                    <= rom_dout[27:20];
        cmd_mask                    <= rom_dout[19:12];
        cmd_dataval                 <= rom_dout[11:4];
        cmd_type                    <= rom_dout[2:0];
        cmd_override                <= rom_dout[3];
        rom_addr                    <= rom_addr + 1'b1;
        seq_state                   <= SEND_CMD;
      end
      
      SEND_CMD:  begin
        if (pready) begin
          psel                      <= 1'b0;
          penable                   <= 1'b0;
          paddr                     <= 32'b0;
          pwrite                    <= 1'b0;     
          pwdata                    <= 32'b0;
          read_data                 <= (cmd_type != CMD_WR) ? prdata[7:0] : 8'b0;
          if (pserr) begin
            seq_state               <= CMD_ERROR;
            seq_status              <= 32'h00000033;      //Bit[0]=done, Bit[1]=error, Bit[5:4]=3=command timeout (waitrequest never deasserted)
          end else if ((cmd_type == CMD_POLL) || (cmd_type == CMD_VAL)) begin
            seq_state                     <= CHECK_DATA;
          end else if (cmd_type == CMD_RMW) begin
            seq_state                     <= MOD_DATA;
          end else begin
            seq_state                     <= READ_ROM;
          end
        end else begin
          psel                            <= 1'b1;
          penable                         <= 1'b1;
          paddr                           <= cmd_addr;
          pwrite                          <= (cmd_type == CMD_WR) ? 1'b1 : 1'b0;
          pwdata                          <= {24'b0, cmd_data};
          timer_en                        <= cmd_type == CMD_POLL;
        end
      end

      CHECK_DATA:  begin
        if (data_verify_fail) begin
          if (cmd_type == CMD_POLL) begin
            if (timer_cnt == TMR_VAL) begin
              seq_state                   <= CMD_ERROR;
              timer_en                    <= 1'b0;
              seq_status                  <= 32'h00000013;      //Bit[0]=done, Bit[1]=error, Bit[5:4]=1=poll timeout
            end else begin
              seq_state                   <= SEND_CMD;
            end
          end else begin
            seq_state                     <= CMD_ERROR;
            timer_en                      <= 1'b0;
            seq_status                    <= 32'h00000023;      //Bit[0]=done, Bit[1]=error, Bit[5:4]=2=data validation error
          end
        end else begin
          seq_state                       <= READ_ROM;
          timer_en                        <= 1'b0;
        end
        read_data                         <= 8'b0;
      end
      
      MOD_DATA: begin
        cmd_data                          <= read_data | cmd_mask;
        cmd_type                          <= CMD_WR;
        seq_state                         <= SEND_CMD;
      end
    
      CMD_ERROR:  begin
        seq_state                         <= IDLE;
      end
      
    endcase
  end
end

//Latch the last PMA attribute and data that was written. Latch return data from PMA command.
//always_ff @(posedge clk) begin
//  if (rst) begin
//    last_attr_data                        <= 16'b0;
//    last_attr_code                        <= 16'b0;
//    seq_last_attr_info                    <= 32'b0;
//    return_val                            <= 32'b0;
//  end else begin
//    if ((seq_state == SEND_CMD) && seq_xcvr_ctrl_done[cmd_chan]) begin
//      case (cmd_addr[9:2]) 
//        8'h84 : begin
//          last_attr_data[7:0]             <= cmd_data;
//        end
//        8'h85 : begin
//          last_attr_data[15:8]            <= cmd_data;
//        end
//        8'h86 : begin
//          last_attr_code[7:0]             <= cmd_data;
//        end
//        8'h87 : begin
//          last_attr_code[15:8]            <= cmd_data;
//        end
//        8'h88 : begin
//          return_val                      <= {seq_xcvr_ctrl_rdata[cmd_chan][7:0], return_val[31:8]};
//        end
//        8'h89 : begin
//          return_val                      <= {seq_xcvr_ctrl_rdata[cmd_chan][7:0], return_val[31:8]};
//        end
//        8'h90 : begin
//          seq_last_attr_info              <= {last_attr_code, last_attr_data};
//        end
//        default : begin
//          last_attr_data                  <= last_attr_data;
//          last_attr_code                  <= last_attr_code;
//          seq_last_attr_info              <= seq_last_attr_info;
//          return_val                      <= return_val;
//        end
//      endcase
//    end
//  end
//end

//Timer
always_ff @(posedge clk) begin
  if (rst) begin
    timer_cnt                             <= 32'b0;
  end else begin
    if (timer_en) begin
      if (timer_cnt != TMR_VAL) begin
        timer_cnt                         <= timer_cnt + 1'b1;
      end
    end else begin
      timer_cnt                           <= 32'b0;
    end
  end
end

//Sequence ROM
eth_adapt_seq_rom eth_adapt_seq_rom (
  .data                         (rom_dout           ),       
  .addr                         (rom_addr           ),
  .clk                          (clk                ) 
);
      

endmodule
