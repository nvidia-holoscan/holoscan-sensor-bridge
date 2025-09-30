
module avst_to_axis 
  import axis_pkg::*;
#(
  parameter     DWIDTH      = 512,
  parameter     ERR_WIDTH   = 6,
  localparam    KEEP_WIDTH  = DWIDTH/8,
  localparam    EMPTY_MSB   = $clog2(KEEP_WIDTH)
) (
  
  input   logic                     clk,
  input   logic                     rst,
  //Avalon Streaming Input
  input   logic                     avst_valid,
  input   logic                     avst_start,
  input   logic                     avst_end,
  input   logic   [DWIDTH-1:0]      avst_data,
  input   logic   [EMPTY_MSB-1:0]   avst_empty,
  input   logic   [ERR_WIDTH-1:0]   avst_error,
  
  //AXIS Output
  output  logic                     axis_tvalid,
  output  logic   [DWIDTH-1:0]      axis_tdata,
  output  logic                     axis_tlast,
  output  logic                     axis_tuser,
  output  logic   [KEEP_WIDTH-1:0]  axis_tkeep
);


logic   [KEEP_WIDTH-1:0]    keep_array [2**EMPTY_MSB];
logic   [KEEP_WIDTH-1:0]    axis_tkeep_i;

//Translate the AVST empty to AXIS tkeep. AVST is number of empty bytes in the last data beat. AXIS tkeep is the
//number of valid bytes in the last beat, encoded as a one-hot vector.
genvar j;
generate
  for (j=0; j<=KEEP_WIDTH-1; j=j+1) begin
    assign keep_array[j]                  = {{KEEP_WIDTH-j{1'b1}}, {j{1'b0}}};
  end
endgenerate 

assign axis_tkeep_i = keep_array[avst_empty]; 

always_ff @(posedge clk) begin
  if (rst) begin
    axis_tvalid                   <= '0;
    axis_tdata                    <= '0;
    axis_tlast                    <= '0;
    axis_tuser                    <= '0;
    axis_tkeep                    <= '0;
  end else begin
    axis_tvalid                   <= avst_valid;
    axis_tdata                    <= axis_tdata_end_swap(avst_data);
    axis_tlast                    <= avst_end;
    axis_tuser                    <= |avst_error;
    axis_tkeep                    <= axis_tkeep_end_swap(axis_tkeep_i);
  end
end

//Swap endianness of AXIS tdata bus
  function automatic [DWIDTH-1:0] axis_tdata_end_swap (input [DWIDTH-1:0] axis_tdata_in);
    begin
      for (int j = 0; j < (DWIDTH/8); j = j+1) begin
        axis_tdata_end_swap[j*8+:8] = axis_tdata_in[(((DWIDTH/8)-1)-j)*8+:8];
      end
    end
  endfunction
  
  //Swap endianness of AXIS tkeep bus
  function automatic [(DWIDTH/8)-1:0] axis_tkeep_end_swap (input [(DWIDTH/8)-1:0] axis_tkeep_in);
    begin
      for (int i = 0; i < (DWIDTH/8); i=i+1) begin
        axis_tkeep_end_swap[i] = axis_tkeep_in[((DWIDTH/8)-1)-i];
      end
    end
  endfunction 

  
endmodule
