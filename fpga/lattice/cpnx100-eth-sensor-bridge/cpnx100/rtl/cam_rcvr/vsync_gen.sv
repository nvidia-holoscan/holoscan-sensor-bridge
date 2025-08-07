module vsync_gen 
  import apb_pkg::*;
(
  input             i_clk,
  input             i_rst,
  
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
  output reg        o_apb_serr,

  input             i_pps,
  input      [31:0] i_ptp_nanosec,
  output            o_vsync_strb,
  output     [ 7:0] o_gpio_mux_en 
);

 localparam PERIOD_10HZ  = 30'h5F5_E100;
 localparam PERIOD_30HZ  = 30'h1FC_A055;
 localparam PERIOD_60HZ  = 30'hFE_502B;
 localparam PERIOD_90HZ  = 30'hA9_8AC7;
 localparam PERIOD_120HZ = 30'h7F_2815;

  enum logic [2:0] {IDLE, WAIT, SET_DLY, SET_EDGE_TIMER, CHK_STRB_CNT, SET_NXT_TIMER, GEN_VSYNC_STRB} state;

  logic        cfg_vsync_en;
  logic [ 3:0] cfg_vsync_mode;
  logic [31:0] cfg_vsync_dly;
  logic        cfg_vsync_start_val;
  logic [31:0] cfg_vsync_exp_time;
  logic [ 7:0] cfg_gpio_mux_en;

  logic        vsync_strb;
  logic [ 6:0] max_strb_cnt;
  logic [ 6:0] strb_cnt;
  logic [29:0] strb_timer /* synthesis syn_keep=1 */;
  logic [29:0] prev_timer;
  logic [29:0] strb_period;

  logic        pps;
  logic [31:0] nanosec;

  always_ff @ (posedge i_clk) begin
    if (i_rst) begin 
      pps     <= 1'b0;
      nanosec <= 'd0;
    end else begin
      pps     <= i_pps;
      nanosec <= i_ptp_nanosec;
    end
  end

//------------------------------------------------------------------------------
// Register Map
//------------------------------------------------------------------------------
  logic [31:0] ctrl_reg [6];
  logic [31:0] stat_reg [1];
  apb_m2s apb_m2s_w;
  apb_s2m apb_s2m_w;  

  assign apb_m2s_w.psel    = i_apb_sel;
  assign apb_m2s_w.penable = i_apb_enable;
  assign apb_m2s_w.paddr   = i_apb_addr;
  assign apb_m2s_w.pwdata  = i_apb_wdata;
  assign apb_m2s_w.pwrite  = i_apb_write;

  assign o_apb_ready = apb_s2m_w.pready;
  assign o_apb_rdata = apb_s2m_w.prdata;
  assign o_apb_serr  = apb_s2m_w.pserr;

  assign stat_reg[0] = 'd0;

  s_apb_reg #(
    .N_CTRL           ( 6                 ),
    .N_STAT           ( 1                 ),
    .W_OFST           ( 28                )
  ) u_reg_map  (
    // APB Interface
    .i_aclk           ( i_apb_clk         ), 
    .i_arst           ( i_apb_rst         ),
    .i_apb_m2s        ( apb_m2s_w         ),
    .o_apb_s2m        ( apb_s2m_w         ),
    // User Control Signals
    .i_pclk           ( i_clk             ), 
    .i_prst           ( i_rst             ),
    .o_ctrl           ( ctrl_reg          ),
    .i_stat           ( stat_reg          )
  );  

  assign cfg_vsync_en        = ctrl_reg[0][0];
  assign cfg_vsync_mode      = ctrl_reg[1][3:0];
  assign cfg_vsync_dly       = ctrl_reg[2][31:0];
  assign cfg_vsync_start_val = ctrl_reg[3][0];
  assign cfg_vsync_exp_time  = ctrl_reg[4][31:0];
  assign cfg_gpio_mux_en     = ctrl_reg[5][7:0];

  always_ff @ (posedge i_clk) begin
    if (i_rst) begin 
      vsync_strb   <= 1'b0;
      strb_cnt     <= 'd0;
      max_strb_cnt <= 'd0;
      strb_period  <= 'd0;
      strb_timer   <= 'd0;
      prev_timer   <= 'd0;
      state        <= IDLE;
    end else begin
      case (state)
        IDLE: begin
          prev_timer   <= 'd0;
          vsync_strb   <= cfg_vsync_start_val;
          strb_cnt     <= 'd0;
          strb_timer   <= cfg_vsync_dly[29:0];
          case(cfg_vsync_mode)
            4'd0:    begin strb_period <= PERIOD_10HZ;  max_strb_cnt <= 7'd10;  end
            4'd1:    begin strb_period <= PERIOD_30HZ;  max_strb_cnt <= 7'd30;  end
            4'd2:    begin strb_period <= PERIOD_60HZ;  max_strb_cnt <= 7'd60;  end
            4'd3:    begin strb_period <= PERIOD_90HZ;  max_strb_cnt <= 7'd90;  end
            4'd4:    begin strb_period <= PERIOD_120HZ; max_strb_cnt <= 7'd120; end
            default: begin strb_period <= PERIOD_60HZ;  max_strb_cnt <= 7'd60;  end
          endcase

          if (pps) begin
            state <= WAIT;
          end
        end
        WAIT: begin
          if (nanosec >= strb_timer) begin
            state <= GEN_VSYNC_STRB;
          end
        end
        GEN_VSYNC_STRB: begin
          vsync_strb <= ~vsync_strb;
          if (vsync_strb == cfg_vsync_start_val) begin
            prev_timer  <= strb_timer;
            state       <= SET_EDGE_TIMER;
          end else begin
            state       <= CHK_STRB_CNT;
            strb_cnt    <= strb_cnt + 1'b1;
          end
        end
        SET_EDGE_TIMER: begin
          strb_timer <= strb_timer + cfg_vsync_exp_time[23:0];
          state      <= WAIT;
        end
        CHK_STRB_CNT: begin
          if (strb_cnt == max_strb_cnt) begin
            state <= IDLE;
          end else begin
            state <= SET_NXT_TIMER;
          end
        end
        SET_NXT_TIMER: begin
          strb_timer <= prev_timer + strb_period;
          state      <= WAIT;
        end
      endcase

      if (!cfg_vsync_en) begin
        state <= IDLE;
      end
    end
  end

assign o_vsync_strb  = vsync_strb;
assign o_gpio_mux_en = cfg_gpio_mux_en;

endmodule
