module cam_rcvr (
  // LVDS PAD IO
  input             i_rx_sclk,
  input             i_rx_dclk,  // 500MHz lvds data clock
  input      [10:0] i_rx_data,  // 500MHz ddr lvds data (1000 Mbps)
  output            o_rx_drdy,
  // clock and reset
  input             i_pclk,
  input             i_prst,     // active high async reset
  //Double ECC Detected or SotSyncHS err
  output            o_phy_err_det,
  // User AXIS Interface
  output            o_axis_tvalid,
  output            o_axis_tlast,
  output     [63:0] o_axis_tdata,
  output     [ 7:0] o_axis_tkeep,
  output     [ 1:0] o_axis_tuser,
  input             i_axis_tready,
  // APB Interface
  input             i_apb_clk,
  input             i_apb_rst,
  input             i_apb_sel,
  input             i_apb_enable,
  input      [31:0] i_apb_addr,
  input      [31:0] i_apb_wdata,
  input             i_apb_write,
  output reg        o_apb_ready,
  output reg [31:0] o_apb_rdata,
  output reg        o_apb_serr
);

//------------------------------------------------------------------------------
// LVDS Rx Interface
//------------------------------------------------------------------------------

  // LVDS calibration reset
  logic        cal_rst;
  logic        cal_reset;
  logic        cal_reset_n;

  reset_sync u_cal_rst (
    .i_clk     ( i_rx_sclk   ),
    .i_arst_n  (~i_prst      ),
    .i_srst    ( cal_rst     ),
    .i_locked  ( 1'b1        ),
    .o_arst    ( cal_reset   ),
    .o_arst_n  ( cal_reset_n ),
    .o_srst    (             ),
    .o_srst_n  (             )
  );

  logic sync_start;

  always_ff @ (posedge i_rx_sclk) begin
    if (cal_reset) begin
      sync_start <= '0;
    end else begin
      sync_start <= '1;
    end
  end

  logic        cal_align;
  logic        cal_done;
  logic [15:0] cal_wait;
  logic [31:0] cal_fail;

  logic        rx_dclk;
  logic [87:0] rx_data;
  logic        rx_ready;

  lvds_ddr_rx u_lvds_rx (
    .sync_clk_i   ( i_rx_sclk     ),
    .sync_rst_i   ( cal_reset     ),
    .sync_start_i ( sync_start    ),
    .alignwd_i    ( cal_align     ),
    .data_i       ( i_rx_data     ),
    .clk_i        ( i_rx_dclk     ),
    .data_o       ( rx_data       ),
    .sclk_o       ( rx_dclk       ),
    .ready_o      ( rx_ready      )
  );


assign o_phy_err_det = '0;

//------------------------------------------------------------------------------
// Calibration
//------------------------------------------------------------------------------

  logic rx_drst;

  reset_sync u_rst (
    .i_clk     ( rx_dclk     ),
    .i_arst_n  ( cal_reset_n ),
    .i_srst    ( 1'b0        ),
    .i_locked  ( 1'b1        ),
    .o_arst    ( rx_drst     ),
    .o_arst_n  (             ),
    .o_srst    (             ),
    .o_srst_n  (             )
  );

  localparam logic [87:0] cal_pattern = 88'h005A55FEDCBA9876543210;
  
  always_ff @ (posedge rx_dclk or posedge rx_drst) begin
    if (rx_drst) begin
      cal_rst         <= 0;
      cal_wait        <= 1;
      cal_align       <= 0;
      cal_fail        <= 0;
      cal_done        <= 0;
    end else begin
      cal_rst         <= 0;
      cal_wait        <= 1;
      cal_align       <= 0;
      cal_fail        <= 0;
      cal_done        <= 0;
      if (rx_ready) begin
        cal_rst       <= cal_fail[31];
        cal_wait      <= 1;
        cal_align     <= 0;
        cal_fail      <= 0;
        cal_done      <= cal_done;
        if (!cal_done) begin
          cal_wait    <= {cal_wait[14:0], cal_wait[15]};
          cal_fail    <= cal_fail;
          cal_done    <= rx_data == cal_pattern;
          if (rx_data != cal_pattern && cal_wait[15] && |rx_data) begin
            cal_align <= ~cal_align;
            cal_fail  <= {cal_fail[30:0], 1'b1};
	        end
        end
      end
    end
  end

  assign o_rx_drdy = cal_done;

//------------------------------------------------------------------------------
// AXIS Buffer
//------------------------------------------------------------------------------
  logic [87:0] r_rx_data;


  logic [79:0] lvds_axis_tdata;
  logic [9:0]  lvds_axis_tkeep;
  logic [2:0]  lvds_axis_metadata;
  logic [4:0]  lvds_axis_tuser;
  logic        lvds_axis_tvalid;
  logic        lvds_axis_tlast;

  logic [63:0] buf_axis_tdata;
  logic [7:0]  buf_axis_tkeep;
  logic [3:0]  buf_axis_tuser;
  logic        buf_axis_tvalid;
  logic        buf_axis_tlast;
  logic        buf_axis_tready;

  logic [3:0]  w_axis_tuser;

  always_ff @ (posedge rx_dclk or posedge rx_drst) begin
    if (rx_drst) begin
      r_rx_data <= '0;
    end else begin
      r_rx_data <= cal_done ? rx_data : '0;
    end
  end

  assign {lvds_axis_tuser,lvds_axis_metadata,lvds_axis_tdata} = r_rx_data;

  assign lvds_axis_tvalid     = (lvds_axis_metadata != 'd0);
  assign lvds_axis_tlast      = (lvds_axis_metadata >= 'd2);
  assign lvds_axis_tkeep[9:8] = (lvds_axis_metadata <= 'd2) ? '1 : '0;
  assign lvds_axis_tkeep[7:6] = (lvds_axis_metadata <= 'd3) ? '1 : '0;
  assign lvds_axis_tkeep[5:4] = (lvds_axis_metadata <= 'd4) ? '1 : '0;
  assign lvds_axis_tkeep[3:2] = (lvds_axis_metadata <= 'd5) ? '1 : '0;
  assign lvds_axis_tkeep[1:0] = (lvds_axis_metadata <= 'd6) ? '1 : '0;

  axis_buffer # (
      .IN_DWIDTH     ( 80     ),
      .OUT_DWIDTH    ( 64     ),
      .WAIT2SEND     ( 0      ),
      .DUAL_CLOCK    ( 1      ),
      .W_USER        ( 5      ),
      .OUT_W_USER    ( 4      )
    ) u_axis_buffer (
      .in_clk            ( rx_dclk               ),
      .in_rst            ( rx_drst               ),
      .out_clk           ( i_pclk                ),
      .out_rst           ( i_prst                ),
      .i_axis_rx_tvalid  ( lvds_axis_tvalid      ),
      .i_axis_rx_tdata   ( lvds_axis_tdata       ),
      .i_axis_rx_tlast   ( lvds_axis_tlast       ),
      .i_axis_rx_tuser   ( lvds_axis_tuser       ),
      .i_axis_rx_tkeep   ( lvds_axis_tkeep       ),
      .o_axis_rx_tready  (                       ),
      .o_axis_tx_tvalid  ( buf_axis_tvalid       ),
      .o_axis_tx_tdata   ( buf_axis_tdata        ),
      .o_axis_tx_tlast   ( buf_axis_tlast        ),
      .o_axis_tx_tuser   ( buf_axis_tuser        ),
      .o_axis_tx_tkeep   ( buf_axis_tkeep        ),
      .i_axis_tx_tready  ( buf_axis_tready       )
    );

  axis_reg # (
    .DWIDTH             ( 64 + 8 + 1 + 4                                                  )
  ) u_axis_reg (
    .clk                ( i_pclk                                                          ),
    .rst                ( i_prst                                                          ),
    .i_axis_rx_tvalid   ( buf_axis_tvalid                                                 ),
    .i_axis_rx_tdata    ( {buf_axis_tdata,buf_axis_tlast,buf_axis_tuser,buf_axis_tkeep}   ),
    .o_axis_rx_tready   ( buf_axis_tready                                                 ),
    .o_axis_tx_tvalid   ( o_axis_tvalid                                                   ),
    .o_axis_tx_tdata    ( {o_axis_tdata,o_axis_tlast,w_axis_tuser,o_axis_tkeep}           ),
    .i_axis_tx_tready   ( i_axis_tready                                                   )
  );


assign o_axis_tuser = w_axis_tuser[1:0];

//////////////////////////////////////////
// APB Interface
//////////////////////////////////////////

always_ff @ (posedge i_apb_clk) begin
  if (i_apb_rst) begin
    o_apb_rdata <= 32'd0;
    o_apb_serr <= 1'b0;
    o_apb_ready <= 1'b0;
  end else begin
    if (i_apb_sel && i_apb_enable) begin
      o_apb_rdata  <= 32'd0;
      o_apb_serr <= 1'b0;
      o_apb_ready  <= 1'b0;
    end else if (i_apb_sel) begin
      o_apb_ready <= 1'b1;
      case(i_apb_addr[27:2])
        0:  begin o_apb_rdata <=  '0; end
        1:  begin o_apb_rdata <=  '0;        end
        2:  begin o_apb_rdata <=  '0;        end
        3:  begin o_apb_rdata <=  '0;        end
        4:  begin o_apb_rdata <=  '0;        end
        5:  begin o_apb_rdata <=  '0;        end
        6:  begin o_apb_rdata <=  '0;        end
        7:  begin o_apb_rdata <=  '0;        end
        8:  begin o_apb_rdata <=  '0;        end
        9:  begin o_apb_rdata <=  '0;        end
        10: begin o_apb_rdata <=  '0;        end
        11: begin o_apb_rdata <=  '0;        end
        12: begin o_apb_rdata <=  '0;        end
        13: begin o_apb_rdata <=  '0;        end
        14: begin o_apb_rdata <=  '0;        end
        15: begin o_apb_rdata <=  '0;        end
        16: begin o_apb_rdata <=  '0;        end
        default: begin o_apb_rdata <= 32'hBAADBAAD; o_apb_serr <= 1'b1; end
      endcase
    end
  end
end

endmodule
