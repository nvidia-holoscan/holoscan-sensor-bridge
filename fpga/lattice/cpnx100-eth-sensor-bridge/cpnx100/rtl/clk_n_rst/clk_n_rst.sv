module clk_n_rst (
  // Reference Clock
  input  i_refclk,    // pll reference clock
  input  i_locked,    // pll reference clock locked
  // System Clocks
  output o_adc_clk,   // 50MHz clock for ADC
  output o_pcs_clk,   // pcs calibration clock
  output o_hif_clk,   // processing clock
  output o_apb_clk,   // apb clock
  output o_ptp_clk,   // ptp clock
  //PTP to Sensor Clock
  input  [31:0] i_ptp_nsec,    //Nanosecond
  output        o_ptp_sensor_pll_lock,
  output        o_ptp_cam_clk, //PTP 24MHz
  //I2S Clock
  output o_i2s_clk_int,  //I2S Clock for Internal I2S IP
  output o_i2s_clk_ext,  //I2S Clock for External device
  output o_i2s_mclk_ext, //I2S MCLK for External device
  //
  input  i_pb_rst_n,  // asynchronous active low pushbutton reset
  input  i_sw_rst,    // software controlled system active high reset

  output o_sys_rst,   // system active high reset
  output o_pcs_rst_n  // ethernet pcs reset
);

  // pcs calibration clock
  osc_clk u_pcs_clk (
    .hf_out_en_i  ( i_pb_rst_n ),
    .hf_clk_out_o ( o_pcs_clk  )
  );

  logic locked;

`ifndef SIMULATION

  // Primary clock pll (NONE FRAC DIV)
  eclk_pll u_clk_pll (
    .clki_i   ( i_refclk  ),
    .rstn_i   ( i_locked  ),
    .clkop_o  (           ),
    .clkos_o  ( o_hif_clk ), // 156.25 MHz
    .clkos2_o ( o_apb_clk ), // 19.53125 MHz
    .clkos3_o ( o_adc_clk ), // 50 MHz
    .clkos4_o ( o_ptp_clk ), // 100.446545 MHz
    .lock_o   ( locked    )
  );

`else
  // Primary clock pll (FRAC DIV, SIM ONLY)
  eclk_pll_sim u_clk_pll (
    .clki_i   ( i_refclk  ),
    .rstn_i   ( i_locked  ),
    .clkop_o  ( o_hif_clk ),
    .clkos_o  ( o_apb_clk ),
    .lock_o   ( locked    )
  );

  // Primary clock pll (INT DIV, SIM ONLY)
  eclk_ptp_pll_sim u_ptp_clk_pll (
    .clki_i   ( i_refclk     ),
    .rstn_i   ( i_locked     ),
    .clkop_o  (              ),
    .clkos_o  ( o_ptp_clk    ), //100.446545 MHz
    .lock_o   (              )
  );

  logic adc_clk;
  initial begin
    adc_clk = 0;
  end
  always #100000 adc_clk <= ~adc_clk;
  assign o_adc_clk = adc_clk;

`endif

  logic ptp_sensor_pll_locked;
  //PTP generated 24MHz clock
  ptp_sensor_pll u_ptp_to_sensor_pll (
    .clki_i    ( i_ptp_nsec[2]         ), //25MHz
    .rstn_i    ( locked                ),
    .clkop_o   (                       ),
    .clkos_o   ( o_ptp_cam_clk         ), //24.038MHz
    .clkos2_o  (                       ), //36.765MHz For I2S
    .lock_o    ( ptp_sensor_pll_locked )
  );

  assign o_ptp_sensor_pll_lock = ptp_sensor_pll_locked;

  /////////////////////////
  //I2S Clock
  /////////////////////////
  logic i2s_pll_clk;
  logic i2s_pll_locked;

  // I2S clock pll 
  i2s_pll u_i2s_clk_pll (
    .clki_i   ( o_hif_clk      ),
    .rstn_i   ( locked         ),
    .clkop_o  (                ),
    .clkos_o  ( i2s_pll_clk    ), //37MHz
    .lock_o   ( i2s_pll_locked )
  );

  //PCLKDIV primitive
  PCLKDIVSP #(
    .DIV_PCLKDIV ("X16")
  ) u_I2S_CLKDIV (
    .CLKIN   ( i2s_pll_clk            ),
    .LSRPDIV ( !i2s_pll_locked        ),
    .CLKOUT  ( o_i2s_clk_int          )
  );

  logic [3:0] div_cnt;
  /* synthesis syn_keep=1 nomerge=""*/
  logic       div2_q;

  always_ff @ (posedge i2s_pll_clk) begin
    if (!i2s_pll_locked) begin
      div_cnt      <= 'd0;
      div2_q       <= 1'b0;
    end else begin
      div_cnt <= div_cnt + 1'b1;
      div2_q  <= div_cnt[0];
    end
  end

  assign o_i2s_mclk_ext = div2_q;
  assign o_i2s_clk_ext  = div_cnt[3];


//----------------------------------------------------------------------------
// System Reset
//----------------------------------------------------------------------------

  reset_sync u_sys_rst (
    .i_clk    ( o_apb_clk   ),
    .i_arst_n ( i_pb_rst_n  ),
    .i_srst   ( 1'b0        ),
    .i_locked ( locked      ),

    .o_arst   ( o_sys_rst   ),
    .o_arst_n (             ),
    .o_srst   (             ),
    .o_srst_n (             )
  );

//----------------------------------------------------------------------------
// MAC Reset
//----------------------------------------------------------------------------

  logic pcs_rst;

  data_sync #(
    .DATA_WIDTH  ( 1          ),
    .RESET_VALUE ( 0          ),
    .SYNC_DEPTH  ( 32         )
  ) u_sw_erst_sync (
    .clk         ( o_pcs_clk  ),
    .rst_n       ( i_pb_rst_n ),
    .sync_in     ( i_sw_rst   ),
    .sync_out    ( pcs_rst    )
  );

  reset_sync u_eth_rst (
    .i_clk    ( o_pcs_clk   ),
    .i_arst_n ( i_pb_rst_n  ),
    .i_srst   ( pcs_rst     ),
    .i_locked ( 1'b1        ),

    .o_arst   (             ),
    .o_arst_n ( o_pcs_rst_n ),
    .o_srst   (             ),
    .o_srst_n (             )
  );

endmodule
