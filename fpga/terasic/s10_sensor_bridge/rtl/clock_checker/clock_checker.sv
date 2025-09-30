
module clock_checker #(
) (

  input logic                 test_clk,                             //Clock that is being measured
  input logic                 rst,                                  //Reset
  input logic                 en,
  input logic                 latch_count,
  input logic   [31:0]        test_clk_target,
  input logic   [31:0]        test_clk_tol,
  
  output logic                clk_in_tolerance,
  output logic  [31:0]        last_clk_count

);

logic         [31:0]          test_clk_cnt            /* synthesis noprune */;
logic         [31:0]          test_clk_cnt_latch      /* synthesis noprune */;
logic                         latch_clk_count_sync    /* synthesis noprune */;
logic                         latch_clk_count_sync_r1;
logic                         en_sync;

reset_sync test_clk_rst_sync (
  .i_clk        (test_clk),
  .i_arst_n     (!rst),
  .i_srst       (1'b0),
  .i_locked     (1'b1),
  .o_arst       (),
  .o_arst_n     (),
  .o_srst       (rst_test_clk),
  .o_srst_n     ()
);

data_sync    #(
  .DATA_WIDTH ( 2                              )
) rst_and_lock_sync (
  .clk        ( test_clk                       ),
  .rst_n      ( ~rst_test_clk                  ),
  .sync_in    ({latch_count, en               }),
  .sync_out   ({latch_clk_count_sync, en_sync })
);

always_ff @(posedge test_clk or posedge rst_test_clk) begin
  if (rst_test_clk) begin
    test_clk_cnt              <= 32'b0;
    test_clk_cnt_latch        <= 32'b0;
    latch_clk_count_sync_r1   <= 1'b0;
  end else begin
    latch_clk_count_sync_r1   <= latch_clk_count_sync;
    if (en_sync) begin
      if (latch_clk_count_sync && !latch_clk_count_sync_r1) begin
        test_clk_cnt            <= 32'b0;
        test_clk_cnt_latch      <= test_clk_cnt;
        if ( (test_clk_cnt <= (test_clk_target + test_clk_tol)) && (test_clk_cnt >= test_clk_target - test_clk_tol) ) begin
          clk_in_tolerance      <= 1'b1;
        end else begin
          clk_in_tolerance      <= 1'b0;
        end
      end else begin
        test_clk_cnt            <= test_clk_cnt + 1'b1;
      end
    end else begin
      test_clk_cnt_latch        <= 32'hFFFFFFFF;
    end
  end
end

assign last_clk_count = test_clk_cnt_latch;


endmodule


