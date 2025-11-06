
module axis_to_avst 
  import axis_pkg::*;
#(
  parameter     DWIDTH      = 512,
  parameter     USER_WIDTH  = 1,
  parameter     ERR_WIDTH   = 1, 
  localparam    KEEP_WIDTH  = DWIDTH/8,
  localparam    EMPTY_MSB   = $clog2(KEEP_WIDTH)
) (
  
  input  logic                     clk,
  input  logic                     rst,
  //AXIS Input
  input  logic                     axis_tvalid,
  input  logic   [DWIDTH-1:0]      axis_tdata,
  input  logic                     axis_tlast,
  input  logic   [USER_WIDTH-1:0]     axis_tuser,
  input  logic   [KEEP_WIDTH-1:0]  axis_tkeep,
  output logic                     axis_tready,
  //AVST Output
  output logic                     avst_valid,
  output logic                     avst_start,
  output logic                     avst_end,
  output logic   [DWIDTH-1:0]      avst_data,
  output logic   [EMPTY_MSB-1:0]   avst_empty,
  output logic   [ERR_WIDTH-1:0]   avst_error,
  input  logic                     avst_ready
);

logic   axis_sop_next;

//Create a start signal. This is basically defined as the first axis_tvalid following an
//axis_tlast.
always_ff @(posedge clk) begin
  if (rst) begin
    axis_sop_next                 <= 1'b1;
  end else if (axis_tvalid && avst_ready) begin
    if (axis_tlast) begin
      axis_sop_next               <= 1'b1;
    end else begin
      axis_sop_next               <= 1'b0;
    end
  end
end


//Translate the AXIS tkeep to AVST empty. This is simply subtracting the number of ones in
//the tkeep at tlast from the width of tkeep.
assign avst_empty       = KEEP_WIDTH - $countones(axis_tkeep);

//avst_end is just tlast
assign avst_end         = axis_tlast;

//Error is mapped from tuser. If the axis_tuser is wider than the avst_error, then OR
//the tuser together. This is more of a debug feature.
assign avst_error       = ERR_WIDTH < USER_WIDTH ? |axis_tuser : axis_tuser;

//avst_valid 
assign avst_valid       = axis_tvalid;

//avst_start
assign avst_start       = axis_sop_next;

//avst_data
assign avst_data        = axis_tdata_end_swap(axis_tdata);

//axis_tready
assign axis_tready      = avst_ready;

//Swap endianness of AXIS tdata bus
function automatic [DWIDTH-1:0] axis_tdata_end_swap (input [DWIDTH-1:0] axis_tdata_in);
  begin
    for (int j = 0; j < (DWIDTH/8); j = j+1) begin
      axis_tdata_end_swap[j*8+:8] = axis_tdata_in[(((DWIDTH/8)-1)-j)*8+:8];
    end
  end
endfunction

  
endmodule
