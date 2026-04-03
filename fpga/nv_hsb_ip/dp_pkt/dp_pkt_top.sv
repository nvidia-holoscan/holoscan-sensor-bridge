
module dp_pkt_top
  import apb_pkg::*;
  import regmap_pkg::*;
#(
  parameter   W_DATA        = 64,
  localparam  W_KEEP        = W_DATA/8,
  parameter   N_INPT        = 2,
  parameter   N_HOST        = 2,
  parameter   DIS_COE       = 0,
  parameter   MTU           = 1500,
  parameter   BUFFER_4K_REG = 0,
  parameter   SYNC_CLK      = 0
)(
  input                           i_pclk,
  input                           i_prst,
  // Register Map, abp clk domain
  input                           i_aclk,
  input                           i_arst,
  input  apb_m2s                  i_apb_m2s       [N_HOST],
  output apb_s2m                  o_apb_s2m       [N_HOST],
  input  apb_m2s                  i_apb_m2s_cfg,
  output apb_s2m                  o_apb_s2m_cfg,
  //PTP Timestamp
  input  [79:0]                   i_cur_ptp,
  input  [N_INPT-1:0]             i_sof,
  // Ethernet Address
  input  [47:0]                   i_dev_mac_addr  [N_HOST],
  input  [31:0]                   i_dev_ip_addr   [N_HOST],
  // Input Data
  input  [N_INPT-1:0]             i_axis_tvalid,
  input  [N_INPT-1:0]             i_axis_tlast,
  input  [W_DATA-1:0]             i_axis_tdata    [N_INPT-1:0],
  input  [W_KEEP-1:0]             i_axis_tkeep    [N_INPT-1:0],
  input  [N_INPT-1:0]             i_axis_tuser,
  output [N_INPT-1:0]             o_axis_tready,
  // AXIS to MAC
  output [N_HOST-1:0]             o_axis_tvalid,
  output [N_HOST-1:0]             o_axis_tlast,
  output [W_DATA-1:0]             o_axis_tdata    [N_HOST],
  output [W_KEEP-1:0]             o_axis_tkeep    [N_HOST],
  output [N_HOST-1:0]             o_axis_tuser,
  input  [N_HOST-1:0]             i_axis_tready
);

//------------------------------------------------------------------------------------------------//
// Constants
//------------------------------------------------------------------------------------------------//

localparam ROCE_HDR_WIDTH     = 224 + 336; // ETH + IPV4 + UDP + ROCE
localparam ROCE_IMM_HDR_WIDTH = ROCE_HDR_WIDTH + 32;
localparam COE_HDR_WIDTH      = 368;
localparam NUM_PAD_BITS       = 7; //128B alignment
localparam ADDR_BITS          = 32;
localparam WIDE_DATA          = (W_DATA > 512) ? 1 : 0;
localparam PAGES_TO_CYCLES    = NUM_PAD_BITS-($clog2(W_KEEP));
localparam CYCLES_TO_PAGES    = NUM_PAD_BITS-($clog2(W_KEEP));
localparam CYCLES_TO_BYTES    = ($clog2(W_KEEP));
localparam BYTES_TO_CYCLES    = ($clog2(W_KEEP));
localparam W_CNT              = $clog2(MTU)+1;
localparam W_INPT             = $clog2(N_INPT)+1;
localparam W_HOST             = $clog2(N_HOST)+1;


//------------------------------------------------------------------------------------------------//
// DP Mux Signals
//------------------------------------------------------------------------------------------------//

// AXIS
logic [N_HOST-1:0] dp_axis_tvalid;
logic [N_HOST-1:0] dp_axis_tlast;
logic [W_DATA-1:0] dp_axis_tdata    [N_HOST];
logic [W_KEEP-1:0] dp_axis_tkeep    [N_HOST];
logic [N_HOST-1:0] dp_axis_tuser;
logic [N_HOST-1:0] dp_axis_tready;

logic [15:0]       w_dev_udp_port [N_HOST];

logic [N_INPT-1:0] w_axis_tready [N_HOST];
logic [N_INPT-1:0] axis_tready;

// Register Map
logic [31:0]        vp_mask     [N_HOST];
logic [7:0]         eth_pkt_len [N_HOST];
logic [31:0]        crc_xor     [N_HOST];

// Control
logic [N_HOST-1:0]  pkt_len_en;
logic [N_HOST-1:0]  sif_tlast_done;
logic [W_CNT-1:0]   pkt_len;
logic [W_CNT-1:0]   cur_pkt_len;
logic [W_CNT-1:0]   cur_pkt_len_pages;
logic [W_CNT-1:0]   unsync_cnt [N_HOST];
logic [W_CNT-1:0]   unsync_cnt_r;
logic [N_HOST-1:0]  dp_tlast;
logic [W_INPT-1:0]  sif_gnt_idx_r;
logic [W_INPT-1:0]  sif_gnt_idx_rq;
logic               is_last;
logic [N_INPT-1:0]  sif_tlast [N_HOST];
logic [N_HOST-1:0]  unsync;
// CRC
logic [31:0]       crc_in;
logic [31:0]       crc_out [N_HOST];
logic [W_INPT-1:0] crc_idx [N_HOST];
logic [N_HOST-1:0] crc_valid;

//------------------------------------------------------------------------------------------------//
// Register Map
//------------------------------------------------------------------------------------------------//
genvar i;

generate
  for (i=0;i<N_HOST;i=i+1) begin : gen_host_rx_axis
    logic [31:0] ctrl_reg [dp_pkt_nctrl];
    logic [31:0] stat_reg [dp_pkt_nstat];

    localparam  [(dp_pkt_nctrl*32)-1:0] RST_VAL = {'0,32'h3000,32'h5CE,32'h0};

    s_apb_reg #(
      .N_CTRL    ( dp_pkt_nctrl   ),
      .N_STAT    ( dp_pkt_nstat   ),
      .W_OFST    ( w_ofst         ),
      .RST_VAL   ( RST_VAL        ),
      .SYNC_CLK  ( SYNC_CLK       )
    ) u_reg_map  (
      // APB Interface
      .i_aclk    ( i_aclk         ),
      .i_arst    ( i_arst         ),
      .i_apb_m2s ( i_apb_m2s  [i] ),
      .o_apb_s2m ( o_apb_s2m  [i] ),
      // User Control Signals
      .i_pclk    ( i_pclk         ),
      .i_prst    ( i_prst         ),
      .o_ctrl    ( ctrl_reg       ),
      .i_stat    ( stat_reg       )
    );

    assign w_dev_udp_port  [i] = ctrl_reg[dp_pkt_dev_udp_port][15:0];
    assign vp_mask         [i] = ctrl_reg[dp_pkt_vip_mask];
    assign eth_pkt_len     [i] = ctrl_reg[dp_pkt_len][7:0];
    assign crc_xor         [i] = ctrl_reg[dp_pkt_crc_xor];
    assign stat_reg        [0] = '0;

//------------------------------------------------------------------------------------------------//
// Data Path MUX
//------------------------------------------------------------------------------------------------//

    dp_pkt # (
      .W_DATA         ( W_DATA                             ),
      .W_KEEP         ( W_KEEP                             ),
      .N_INPT         ( N_INPT                             ),
      .MTU            ( MTU                                )
    ) dp_data_inst (
      // Clocks and Reset
      .i_pclk         ( i_pclk                             ),
      .i_prst         ( i_prst                             ),
      // Control
      .i_pkt_len      ( cur_pkt_len                        ),
      .i_pkt_len_en   ( pkt_len_en                     [i] ),
      .i_gnt_idx      ( sif_gnt_idx_r                      ),
      .o_tlast        ( sif_tlast                      [i] ),
      .i_tlast_done   ( sif_tlast_done                 [i] ),
      .o_unsync_cnt   ( unsync_cnt                     [i] ),
      .o_unsync       ( unsync                         [i] ),
      // CRC
      .i_crc          ( crc_in                             ),
      .o_crc          ( crc_out                        [i] ),
      .o_crc_idx      ( crc_idx                        [i] ),
      .o_crc_valid    ( crc_valid                      [i] ),
      // Input Data
      .i_axis_tvalid  ( i_axis_tvalid                      ),
      .i_axis_tlast   ( i_axis_tlast                       ),
      .i_axis_tdata   ( i_axis_tdata                       ),
      .i_axis_tkeep   ( i_axis_tkeep                       ),
      .i_axis_tuser   ( i_axis_tuser                       ),
      .o_axis_tready  ( w_axis_tready                  [i] ),
      // Output Data
      .o_axis_tvalid  ( dp_axis_tvalid                 [i] ),
      .o_axis_tlast   ( dp_axis_tlast                  [i] ),
      .o_axis_tdata   ( dp_axis_tdata                  [i] ),
      .o_axis_tkeep   ( dp_axis_tkeep                  [i] ),
      .o_axis_tuser   ( dp_axis_tuser                  [i] ),
      .i_axis_tready  ( dp_axis_tready                 [i] )
    );

  end
endgenerate

integer m;
always_comb begin
  axis_tready = '0;
  for (m=0;m<N_HOST;m=m+1) begin
    axis_tready |= w_axis_tready[m];
  end
end

assign o_axis_tready = axis_tready;

//------------------------------------------------------------------------------------------------//
// SOF Timestamp
//------------------------------------------------------------------------------------------------//

logic [79:0] sof_ptp;
logic [15:0] frame_num;
logic        tlast_aligned;
logic        ram_rd;



dp_pkt_ts # (
  .N_INPT ( N_INPT )
) u_dp_pkt_ts (
  .i_pclk    ( i_pclk                       ),
  .i_prst    ( i_prst                       ),
  .i_req     ( i_sof                        ),
  .i_ptp     ( i_cur_ptp                    ),
  .i_ptp_sel ( sif_gnt_idx_r                ),
  .i_ram_rd  ( ram_rd                       ),
  .o_ptp     ( sof_ptp                      )
);


//------------------------------------------------------------------------------------------------//
// Configuration RAM
//------------------------------------------------------------------------------------------------//
// Config RAM Contents [512]:
  // [0] : is_1722b,roce_dest_qp     [23:0]
  // [1] : roce_rkey          [31:0]
  // [2] : roce_buf_start  [0][25:0]
  // [3] : roce_buf_start  [1][25:0]
  // [4] : roce_buf_start  [2][25:0]
  // [5] : roce_buf_start  [3][25:0]
  // [6] : roce_buf_len       [31:0]
  // [7] : roce_buf_mask      [ 3:0]
  // [8] : host_mac_lo        [31:0]
  // [9] : host_mac_hi        [15:0]
  // [A] : host_ip_addr       [31:0]
  // [B] : md_disable, host_udp_port [15:0]
  // [C] : roce_buf_fixed_msb [31:0]
  // [D] : RSVD               [31:0]
  // [E] : RSVD               [31:0]
  // [F] : PSN                [23:0]

logic [31:0] cfg_ram_data;
logic [8:0]  cfg_ram_addr;
logic [8:0]  w_cfg_ram_addr;
logic        cfg_ram_rd_valid;
logic        cfg_ram_wr_en;
logic        cfg_ram_rd_en;
logic [31:0] cfg_ram_wrdata;
logic [11:0] buf_ptr;

s_apb_ram #(
  .R_CTRL           ( 512               ),
  .R_WIDTH          ( 32                ),
  .R_TOTL           ( 2**ADDR_SW_ROCE   )
) u_config_ram  (
  .i_aclk           ( i_aclk            ),
  .i_arst           ( i_arst            ),
  .i_apb_m2s        ( i_apb_m2s_cfg     ),
  .o_apb_s2m        ( o_apb_s2m_cfg     ),
  // User Control Signals
  .i_pclk           ( i_pclk            ),
  .i_prst           ( i_prst            ),
  .i_addr           ( w_cfg_ram_addr    ),
  .o_rd_data        ( cfg_ram_data      ),
  .o_rd_data_valid  ( cfg_ram_rd_valid  ),
  .i_wr_data        ( cfg_ram_wrdata    ),
  .i_wr_en          ( cfg_ram_wr_en     ),
  .i_rd_en          ( cfg_ram_rd_en     )
);

assign w_cfg_ram_addr[8:4] = cfg_ram_addr[8:4];
generate
  if (BUFFER_4K_REG) begin
    assign w_cfg_ram_addr[3:0] = (cfg_ram_addr[3:0] == 4'h0) ? 4'h6               :
                                 (cfg_ram_addr[3:0] == 4'h1) ? 4'h4               :
                                 (cfg_ram_addr[3:0] == 4'h2) ? 4'h2               :
                                 (cfg_ram_addr[3:0] == 4'h3) ? 4'h3               :
                                 (cfg_ram_addr[3:0] == 4'h4) ? 4'h0               :
                                 (cfg_ram_addr[3:0] == 4'h5) ? 4'h1               :
                                 (cfg_ram_addr[3:0] == 4'h6) ? 4'h5               :
                                 (cfg_ram_addr[3:0] == 4'h7) ? 4'hA               :
                                 (cfg_ram_addr[3:0] == 4'h8) ? 4'hB               :
                                 (cfg_ram_addr[3:0] == 4'h9) ? 4'h8               :
                                 (cfg_ram_addr[3:0] == 4'hA) ? 4'h9               :
                                                              cfg_ram_addr[3:0];
  end 
  else begin
    assign w_cfg_ram_addr[3:0] =(cfg_ram_addr[3:0] == 4'h0) ? 4'h6               :
                                (cfg_ram_addr[3:0] == 4'h1) ? 4'h0               :
                                (cfg_ram_addr[3:0] == 4'h2) ? 4'hC               :
                                (cfg_ram_addr[3:0] == 4'h3) ? 4'h2 + buf_ptr[1:0]:
                                (cfg_ram_addr[3:0] == 4'h4) ? 4'h1               :
                                (cfg_ram_addr[3:0] == 4'h5) ? 4'h7               :
                                (cfg_ram_addr[3:0] == 4'h6) ? 4'h8               :
                                (cfg_ram_addr[3:0] == 4'h7) ? 4'h9               :
                                (cfg_ram_addr[3:0] == 4'h8) ? 4'hA               :
                                (cfg_ram_addr[3:0] == 4'h9) ? 4'hB               :
                                                              cfg_ram_addr[3:0];
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// Status RAMs
//------------------------------------------------------------------------------------------------//
localparam STATUS_RAM_SIZE = 1+11+1+4+24+ADDR_BITS+16+1;
// Counter RAM Contents [16]:
  // [0]   : N'addr, 24'psn, 12'buf_ptr, 1'is_new, 1'is_last, 1'not_first // VIP 0

logic [STATUS_RAM_SIZE-1:0] sts_ram_data;
logic [W_INPT-1:0]          sts_ram_addr;
logic [STATUS_RAM_SIZE-1:0] sts_ram_wrdata;
logic                       sts_ram_wren;
logic                       sts_ram_rden;

(* ram_style = "distributed" *) logic [STATUS_RAM_SIZE-1:0] sts_ram [N_INPT] = '{default:'0}/*synthesis syn_ramstyle = "distributed"*/;

always @ (posedge i_pclk) begin
  if (i_prst) begin
    sts_ram      <= '{default:'0};
    sts_ram_data <= '0;
  end
  else begin
    if (sts_ram_wren) begin
      sts_ram[sts_ram_addr] <= sts_ram_wrdata;
    end
    if (sts_ram_rden) begin
      sts_ram_data     <= sts_ram[sts_ram_addr];
    end
  end
end

logic [31:0]        crc_ram_data;
logic [W_INPT-1:0]  crc_ram_addr;
logic [31:0]        crc_ram_wrdata;
logic               crc_ram_wren;
logic               crc_ram_rden;

(* ram_style = "distributed" *) logic [31:0] crc_ram [N_INPT]/*synthesis syn_ramstyle = "distributed"*/;

always @ (posedge i_pclk) begin
  if (i_prst) begin
    crc_ram      <= '{default:'0};
    crc_ram_data <= '0;
  end
  else begin
    if (crc_ram_wren) begin
      crc_ram[crc_ram_addr] <= crc_ram_wrdata;
    end
    if (crc_ram_rden) begin
      crc_ram_data <= crc_ram[crc_ram_addr];
    end
  end
end

localparam ETH_RAM_SIZE = 48+32+16+16+1;

logic [ETH_RAM_SIZE-1:0] eth_ram_data;
logic [W_INPT-1:0]       eth_ram_addr;
logic [ETH_RAM_SIZE-1:0] eth_ram_wrdata;
logic                    eth_ram_wren;
logic                    eth_ram_rden;

(* ram_style = "distributed" *) logic [ETH_RAM_SIZE-1:0] eth_ram [N_INPT]/*synthesis syn_ramstyle = "distributed"*/;

always @ (posedge i_pclk) begin
  if (i_prst) begin
    eth_ram      <= '{default:'0};
    eth_ram_data <= '0;
  end
  else begin
    if (eth_ram_wren) begin
      eth_ram[eth_ram_addr] <= eth_ram_wrdata;
    end
    if (eth_ram_rden) begin
      eth_ram_data <= eth_ram[eth_ram_addr];
    end
  end
end


//------------------------------------------------------------------------------------------------//
// Status RAM Read Data
//------------------------------------------------------------------------------------------------//

logic                 is_new;
logic                 line_end;
logic [23:0]          psn;
logic [ADDR_BITS-1:0] roce_vaddr_sts;
logic                 nxt_pkt_is_last;
logic                 pkt_is_done;
logic                 pkt_is_imm;
logic [25:0]          roce_nxt_vaddr;
logic [25:0]          roce_nxt_vaddr2;
logic [11:0]          buf_ptr_nxt;
logic                 pkt_is_1722b;
logic [N_HOST-1:0]    hif_is_1722b;
logic [N_HOST-1:0]    hif_is_data_wr_imm;
logic                 is_data_wr_imm;
logic                 nxt_pkt_le;
logic                 threshold_val;
logic                 threshold_val_last;
logic                 single_pkt;
logic                 md_disable;
logic                 is_first_run;

assign is_first_run    = !sts_ram_data[0];
assign is_last         = sts_ram_data[1];
assign pkt_is_done     = sts_ram_data[2];
assign is_new          = !sts_ram_data[3];
assign buf_ptr         = sts_ram_data[15:4];
assign line_end        = sts_ram_data[16];
assign psn             = sts_ram_data[40:17];
assign roce_vaddr_sts  = sts_ram_data[41+:ADDR_BITS];
assign frame_num       = sts_ram_data[41+ADDR_BITS+:16];

assign sts_ram_wrdata[0]                = '1;
assign sts_ram_wrdata[1]                = nxt_pkt_is_last;
assign sts_ram_wrdata[2]                = (is_last && !pkt_is_imm) || (single_pkt);
assign sts_ram_wrdata[3]                = !pkt_is_imm;
assign sts_ram_wrdata[15:4]             = (pkt_is_imm) ? buf_ptr_nxt : buf_ptr;
assign sts_ram_wrdata[16]               = nxt_pkt_le;
assign sts_ram_wrdata[40:17]            = (pkt_is_imm && pkt_is_1722b) ? '0 : psn + (md_disable ? !pkt_is_imm : 1'b1);
assign sts_ram_wrdata[41+:ADDR_BITS]    = roce_nxt_vaddr; // In pages
assign sts_ram_wrdata[41+ADDR_BITS+:16] = frame_num + (pkt_is_imm);

//------------------------------------------------------------------------------------------------//
// Config RAM Read Data
//------------------------------------------------------------------------------------------------//

typedef enum logic [3:0] {
  ROCE_IDLE,
  ROCE_LOAD_STS,
  ROCE_GRANT,
  ROCE_CALC,
  ROCE_STORE,
  ROCE_WR_IMM,
  ROCE_HDR_WAIT
} roce_fsm_t;

roce_fsm_t roce_state;

logic [23:0]       roce_dest_qp;
logic [N_INPT-1:0] sensor_is_1722b;
logic [N_INPT-1:0] sensor_is_md_disable;
logic [31:0]       rkey;
logic [31:0]       roce_buf_fixed_msb;
logic [25:0]       roce_buf_inc;
logic [11:0]       roce_buf_ptr_start;
logic [11:0]       roce_buf_ptr_end;
logic [25:0]       roce_buf_start;
logic [31:0]       roce_buf_len;
logic [3:0]        roce_buf_mask;
logic [6:0]        threshold_idx;
logic [6:0]        r_threshold_idx;

logic [47:0] sif_mac_addr;
logic [31:0] sif_ip_addr ;
logic [15:0] sif_udp_port;

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    sensor_is_1722b <= '0;
    sensor_is_md_disable <= '0;
  end
  else begin
    if (BUFFER_4K_REG) begin
      sensor_is_1722b[sif_gnt_idx_r]      <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h5)) ? cfg_ram_data[24] : sensor_is_1722b[sif_gnt_idx_r];
      sensor_is_md_disable[sif_gnt_idx_r] <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h9)) ? cfg_ram_data[16] : sensor_is_md_disable[sif_gnt_idx_r];
    end
    else begin
      sensor_is_1722b[sif_gnt_idx_r]      <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h2)) ? cfg_ram_data[24] : sensor_is_1722b[sif_gnt_idx_r];
      sensor_is_md_disable[sif_gnt_idx_r] <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'hA)) ? cfg_ram_data[16] : sensor_is_md_disable[sif_gnt_idx_r];
    end
  end
end

generate
  if (BUFFER_4K_REG) begin
    always_ff @(posedge i_pclk) begin
      roce_buf_len        <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h1)) ? cfg_ram_data[31:0] : roce_buf_len;
      roce_buf_inc        <= pkt_is_1722b ? '0 : (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h2)) ? cfg_ram_data[25:0] : roce_buf_inc;
      roce_buf_start[24:0]<= pkt_is_1722b ? '0 : (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h3)) ? cfg_ram_data[31:7] : roce_buf_start[24:0];
      roce_buf_start[25]  <= pkt_is_1722b ? '0 : (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h4)) ? cfg_ram_data[0]    : roce_buf_start[25];
      roce_buf_fixed_msb  <= pkt_is_1722b ? '0 : (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h4)) ? {cfg_ram_data[31:1],1'b0} : roce_buf_fixed_msb;
      roce_dest_qp        <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h5)) ? cfg_ram_data[23:0]  : roce_dest_qp;
      threshold_idx       <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h5)) ? cfg_ram_data[31:25] : threshold_idx;
      rkey                <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h6)) ? cfg_ram_data[31:0] : rkey;
      roce_buf_ptr_start  <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h7)) ? cfg_ram_data[27:16]: roce_buf_ptr_start;
      roce_buf_ptr_end    <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h7)) ? cfg_ram_data[11:0] : roce_buf_ptr_end;
      sif_ip_addr         <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h8)) ? cfg_ram_data[31:0] : sif_ip_addr;
      sif_udp_port        <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h9)) ? cfg_ram_data[15:0] : sif_udp_port;
      sif_mac_addr[31:0]  <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'hA)) ? cfg_ram_data[31:0] : sif_mac_addr[31:0];
      sif_mac_addr[47:32] <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'hB)) ? cfg_ram_data[31:0] : sif_mac_addr[47:32];
    end
  end
  else begin
    always_ff @(posedge i_pclk) begin
      roce_buf_len        <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h1)) ? cfg_ram_data[31:0] : roce_buf_len;
      roce_dest_qp        <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h2)) ? cfg_ram_data[23:0]  : roce_dest_qp;
      threshold_idx       <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h2)) ? cfg_ram_data[31:25] : threshold_idx;
      roce_buf_fixed_msb  <= pkt_is_1722b ? '0 : (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h3)) ? cfg_ram_data[31:0] : roce_buf_fixed_msb;
      roce_buf_start      <= pkt_is_1722b ? '0 : (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h4)) ? cfg_ram_data[31:0] : roce_buf_start;
      rkey                <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h5)) ? cfg_ram_data[31:0] : rkey;
      roce_buf_mask       <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h6)) ? cfg_ram_data[3:0]  : roce_buf_mask;
      sif_mac_addr[31:0]  <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h7)) ? cfg_ram_data[31:0] : sif_mac_addr[31:0];
      sif_mac_addr[47:32] <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h8)) ? cfg_ram_data[31:0] : sif_mac_addr[47:32];
      sif_ip_addr         <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'h9)) ? cfg_ram_data[31:0] : sif_ip_addr;
      sif_udp_port        <= (cfg_ram_rd_valid && (cfg_ram_addr[3:0] == 4'hA)) ? cfg_ram_data[15:0] : sif_udp_port;
    end
  end
endgenerate


assign ram_rd = (roce_state == ROCE_IDLE) && tlast_aligned;
assign cfg_ram_rd_en = (roce_state != ROCE_IDLE);

//------------------------------------------------------------------------------------------------//
// Eth RAM
//------------------------------------------------------------------------------------------------//

logic  [47:0] hif_mac_addr;
logic  [31:0] hif_ip_addr ;
logic  [15:0] hif_udp_port;
logic  [15:0] hif_ipv4_chksum ;
logic  [15:0] sif_ipv4_chksum ;

assign hif_mac_addr    = eth_ram_data[0+:48];
assign hif_ip_addr     = eth_ram_data[48+:32];
assign hif_udp_port    = eth_ram_data[80+:16];
assign hif_ipv4_chksum = eth_ram_data[96+:16];

assign eth_ram_wrdata[0+:48]  = sif_mac_addr;
assign eth_ram_wrdata[48+:32] = sif_ip_addr;
assign eth_ram_wrdata[80+:16] = sif_udp_port;
assign eth_ram_wrdata[96+:16] = (pkt_is_imm && md_disable) ? hif_ipv4_chksum : sif_ipv4_chksum; // Used for single packet mode, no Metadata packet. Prevents recalculation of IPv4 checksum.

//------------------------------------------------------------------------------------------------//
// Sensor ARB
//------------------------------------------------------------------------------------------------//

logic [N_INPT-1:0] sif_req;
logic [N_INPT-1:0] sif_gnt;
logic [W_INPT-1:0] sif_gnt_idx;
logic [N_INPT-1:0] sif_eof;
logic [N_INPT-1:0] sif_wtlast;

logic [N_HOST-1:0] hif_busy;
logic [N_INPT-1:0] hif_mask;
logic [N_HOST-1:0] hif_data_en;
logic [N_HOST-1:0] hif_gnt;
logic [W_HOST-1:0] hif_gnt_idx;
logic [W_HOST-1:0] hif_gnt_idx_r;
logic              hdr_is_busy;
logic              imm_is_done;

logic [W_DATA-1:0] dout_axis_tdata [N_HOST];
logic [W_KEEP-1:0] dout_axis_tkeep [N_HOST];
logic [N_HOST-1:0] dout_axis_tlast;
logic [N_HOST-1:0] dout_axis_tuser;
logic [N_HOST-1:0] dout_axis_tvalid;
logic [N_HOST-1:0] dout_axis_tready;

logic              arb_idle;


always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    sif_req     <= '0;
    hif_busy    <= '0;
    hif_data_en <= '0;
  end
  else begin
    for (int s=0;s<N_INPT;s=s+1) begin
      sif_req[s] <= !hif_mask[s] ? '0         : // No SIFs can transmit if host is busy
                    (|sif_eof  ) ? sif_eof[s] : // Prioritize EOF packets
                    (i_axis_tvalid[s]); // Otherwise, prioritize valid packets
    end
    for (int h=0;h<N_HOST;h=h+1) begin
      hif_busy[h]    <= (hif_busy[h]) ? !((dout_axis_tvalid[h] && dout_axis_tlast[h] && dout_axis_tready[h]) || (imm_is_done && (h == hif_gnt_idx_r))) : hif_gnt[h];
      hif_data_en[h] <= (hif_data_en[h]) ? hif_busy[h] : (h==hif_gnt_idx_r) && hdr_is_busy && !pkt_is_imm;
    end
  end
end

assign imm_is_done = (md_disable && (roce_state == ROCE_HDR_WAIT) && pkt_is_imm); // Metadata is skipped and Done

always_comb begin
  hif_mask = '0;
  sif_wtlast = '0;
  for (int i=0;i<N_HOST;i=i+1) begin
    hif_mask |= hif_busy[i] ? '0 : (vp_mask[i]);
    sif_wtlast |= sif_tlast[i];
  end
end

rrarb #(
  .WIDTH ( N_INPT                      )
) axis_sif_arb (
  .clk   ( i_pclk                      ),
  .rst_n ( !i_prst                     ),
  .rst   ( 1'b0                        ),
  .idle  ( arb_idle                    ),
  .req   ( sif_req                     ),
  .gnt   ( sif_gnt                     )
);

integer j;
always_comb begin
  sif_gnt_idx = '0;
  for (j=0;j<N_INPT;j=j+1) begin
    if (sif_gnt[j]) begin
      sif_gnt_idx = j;
    end
  end
end

always_comb begin
  for (int i=0;i<N_HOST;i=i+1) begin
    hif_gnt[i] = |(sif_gnt & vp_mask[i]);
  end
end

always_comb begin
  hif_gnt_idx = '0;
  for (int i=0;i<N_HOST;i=i+1) begin
    if (hif_gnt[i]) begin
      hif_gnt_idx = i;
    end
  end
end

assign arb_idle = (roce_state == ROCE_IDLE);

//------------------------------------------------------------------------------------------------//
// ROCE Header FSM
//------------------------------------------------------------------------------------------------//

logic [4:0]           fsm_cnt;
logic [4:0]           fsm_cnt_max;
logic [4:0]           fsm_cnt_comb;
logic [79:0]          cur_ptp_sync;
logic [79:0]          cur_ptp;
logic [31:0]          frame_crc;
logic [25:0]          roce_buf_offset;
logic [25:0]          roce_buf_addr;
logic [25:0]          roce_rel_addr;
logic [25:0]          roce_len_left;
logic [25:0]          roce_rel_addr_cycles;
logic [25:0]          roce_imm_vaddr;
logic                 le_last;
logic [25:0]          roce_nxt_len;
logic                 threshold_is_lt;
logic                 round_up;
logic [ADDR_BITS-1:0] roce_vaddr;
logic [1:0]           hdr_type;
logic                 hdr_trigger;
logic [15:0]          dma_len;
logic                 unsync_flag;
logic                 bpw_only_flag;
logic [63:0]          roce_val_addr;
logic                 ptp_ts_en;
logic                 round_cur_pkt_len;
logic                 cycle_32byte;
logic                 sif_gnt_unchanged;

assign cur_ptp_sync = i_cur_ptp;


always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    roce_state         <= ROCE_IDLE;
    fsm_cnt            <= '0;
    fsm_cnt_max        <= '0;
    pkt_len_en         <= '0;
    pkt_len            <= '0;
    sif_eof            <= '0;
    pkt_is_imm         <= '0;
    hdr_type           <= '0;
    hdr_trigger        <= '0;
    sif_gnt_idx_r      <= '0;
    sif_gnt_idx_rq     <= '0;
    hif_gnt_idx_r      <= '0;
    dma_len            <= '0;
    hif_is_1722b       <= '0;
    hif_is_data_wr_imm <= '0;
    tlast_aligned      <= '0;
    sts_ram_wren       <= '0;
    sif_gnt_unchanged  <= '0;
    sif_tlast_done     <= '0;
  end
  else begin
    case (roce_state)
      ROCE_IDLE: begin
        if (|sif_gnt) begin
          roce_state         <= ROCE_LOAD_STS;
          hif_gnt_idx_r      <= hif_gnt_idx;
          sif_gnt_idx_r      <= sif_gnt_idx;
          sif_gnt_idx_rq     <= sif_gnt_idx_r;
        end
        fsm_cnt        <= '0;
        fsm_cnt_max    <= '0;
        hdr_trigger    <= '0;
        sif_eof        <= sif_eof | sif_wtlast; // Early tlast from Sensor
        sif_tlast_done <= '0;
        tlast_aligned  <= '0;
        sts_ram_wren   <= '0;
      end
      ROCE_LOAD_STS: begin
        sif_gnt_unchanged <= (sif_gnt_idx_r == sif_gnt_idx_rq);
        if (cfg_ram_rd_valid) begin
          fsm_cnt                      <= fsm_cnt + 1;
          roce_state                   <= (cfg_ram_rd_valid) ? ROCE_GRANT : ROCE_LOAD_STS;
          pkt_is_imm                   <= sif_eof[sif_gnt_idx_r];
          pkt_len                      <= (eth_pkt_len[hif_gnt_idx_r]<<PAGES_TO_CYCLES);
        end
      end
      ROCE_GRANT: begin
        if (cfg_ram_rd_valid) begin
          fsm_cnt                           <= fsm_cnt + 1;
          sif_tlast_done[hif_gnt_idx_r]     <= '1;
          tlast_aligned                     <= sif_wtlast[sif_gnt_idx_r] && pkt_is_imm;
          roce_state                        <= ROCE_CALC;
          hdr_type                          <= {pkt_is_imm, pkt_is_1722b};
          hif_is_1722b[hif_gnt_idx_r]       <= pkt_is_1722b;
          hif_is_data_wr_imm[hif_gnt_idx_r] <= is_data_wr_imm;
          // Either Read all of Config RAM, or just enough to read + calculate the next packet
          fsm_cnt_max                       <= (is_new || is_last || pkt_is_imm) ? 5'd18 : 5'd7;
          dma_len                           <= (pkt_is_imm) ? 'd128: (cycle_32byte) ? 'd32 : cur_pkt_len << CYCLES_TO_BYTES;
        end
      end
      ROCE_CALC: begin
        sif_tlast_done              <= '0;
        fsm_cnt                     <= fsm_cnt + cfg_ram_rd_valid;
        hdr_trigger                 <= '0;
        pkt_len_en                  <= '0;
        dma_len                     <= (pkt_is_imm) ? 'd128 : (cycle_32byte) ? 'd32 : cur_pkt_len << CYCLES_TO_BYTES;
        if ((fsm_cnt == fsm_cnt_comb) && cfg_ram_rd_valid) begin
          pkt_len_en[hif_gnt_idx_r]         <= !pkt_is_imm;
          hdr_trigger                       <= md_disable ? !pkt_is_imm : '1;
          hdr_type                          <= {pkt_is_imm, pkt_is_1722b};
          hif_is_1722b[hif_gnt_idx_r]       <= pkt_is_1722b;
          hif_is_data_wr_imm[hif_gnt_idx_r] <= is_data_wr_imm;
        end
        if (fsm_cnt == fsm_cnt_max && cfg_ram_rd_valid) begin
          roce_state <= ROCE_STORE;
        end
        sif_eof                  <= sif_eof | sif_wtlast; // Early tlast from Sensor
      end
      ROCE_STORE: begin
        hdr_trigger              <= '0;
        pkt_len_en               <= '0;
        if (fsm_cnt >= 'd9) begin
          roce_state               <= ROCE_HDR_WAIT;
          sif_eof[sif_gnt_idx_r]   <= (pkt_is_imm) ? '0 : (single_pkt || is_last);
          sts_ram_wren             <= '1;
        end
        fsm_cnt <= fsm_cnt + 1;
      end
      ROCE_HDR_WAIT: begin
        fsm_cnt           <= '0;
        roce_state        <= (hdr_is_busy) ? ROCE_HDR_WAIT : ROCE_IDLE;
        sif_eof           <= sif_eof | sif_wtlast; // Early tlast from Sensor
        sts_ram_wren      <= '0;
      end
    endcase
  end
end

assign sts_ram_addr   = sif_gnt_idx_r;
assign cfg_ram_addr   = {sif_gnt_idx_r, fsm_cnt[3:0]};
assign sts_ram_rden   = (roce_state == ROCE_LOAD_STS);
assign crc_ram_wren   = (roce_state == ROCE_LOAD_STS) && crc_valid[hif_gnt_idx_r]; // Saves CRC data from previous packet
assign crc_ram_wrdata = crc_out[hif_gnt_idx_r];
assign eth_ram_wren   = (roce_state == ROCE_CALC) && (fsm_cnt == 5'd17);
assign eth_ram_addr   = sif_gnt_idx_r;
assign cfg_ram_wrdata = {8'h0,psn};
assign cfg_ram_wr_en  = ((fsm_cnt == 'd15) && (roce_state == ROCE_CALC));
assign crc_ram_rden  = (roce_state != ROCE_IDLE);
assign eth_ram_rden  = (roce_state != ROCE_IDLE);

localparam DUMP_CLK_CYCLES = (BUFFER_4K_REG) ? 'd7 : 'd5;

assign fsm_cnt_comb = (is_first_run)                    ? fsm_cnt_max     : 
                      (single_pkt && md_disable)        ? (sif_gnt_unchanged ? 'd2 : DUMP_CLK_CYCLES) :
                      (is_new || is_last || pkt_is_imm) ? fsm_cnt_max     : 
                                                          DUMP_CLK_CYCLES ;

//------------------------------------------------------------------------------------------------//
// RoCE Calculations
//------------------------------------------------------------------------------------------------//

generate
  if (W_DATA >= 512) begin
    always_ff @(posedge i_pclk) begin
      if (i_prst) begin
        cycle_32byte <= '0;
      end
      else begin
        cycle_32byte <= (roce_buf_len == 'd32);
      end
    end
  end
  else begin
    assign cycle_32byte = '0;
  end
endgenerate


always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    frame_crc                 <= '0;
    roce_rel_addr             <= '0;
    roce_len_left             <= '0;
    roce_imm_vaddr            <= '0;
    le_last                   <= '0;
    pkt_is_1722b              <= '0;
    single_pkt                <= '0;
    roce_nxt_vaddr            <= '0;
    cur_pkt_len               <= '0;
    roce_vaddr                <= '0;
    nxt_pkt_is_last           <= '0;
    unsync_flag               <= '0;
    unsync_cnt_r              <= '0;
    roce_val_addr             <= '0;
    round_up                  <= '0;
    threshold_val             <= '0;
    threshold_is_lt           <= '0;
    nxt_pkt_le                <= '0;
    r_threshold_idx           <= '0;
    roce_buf_offset           <= '0;
    roce_buf_addr             <= '0;
    roce_nxt_vaddr2           <= '0;
    md_disable                <= '0;
  end
  else begin
    roce_buf_offset           <= (BUFFER_4K_REG) ? (roce_buf_inc * buf_ptr) : '0;
    roce_buf_addr             <= roce_buf_start + roce_buf_offset;
    roce_vaddr                <= (is_new) ? roce_buf_addr : roce_vaddr_sts;
    frame_crc                 <= crc_ram_data ^ crc_xor[hif_gnt_idx_r];
    roce_rel_addr             <= (roce_vaddr - roce_buf_addr); // In Pages
    roce_len_left             <= ((roce_buf_len == 'd32) && (W_DATA >= 512)) ? 'd1 : ((roce_buf_len >> BYTES_TO_CYCLES) - roce_rel_addr_cycles);  // In cycles
    round_up                  <= |(roce_buf_len[0+:BYTES_TO_CYCLES+CYCLES_TO_PAGES]);
    roce_imm_vaddr            <= (roce_buf_len >> (BYTES_TO_CYCLES + CYCLES_TO_PAGES)) + (roce_buf_addr + round_up);
    r_threshold_idx           <= threshold_idx - (PAGES_TO_CYCLES + CYCLES_TO_BYTES);
    le_last                   <= roce_buf_len[threshold_idx] != roce_vaddr[r_threshold_idx];
    pkt_is_1722b              <= sensor_is_1722b[sif_gnt_idx_r];
    md_disable                <= sensor_is_md_disable[sif_gnt_idx_r];
    roce_nxt_vaddr            <= roce_vaddr + (cur_pkt_len_pages + round_cur_pkt_len); // Round Up
    roce_nxt_vaddr2           <= roce_vaddr + (cur_pkt_len_pages<<1);
    cur_pkt_len               <= single_pkt ? (cycle_32byte) ? 'd1 : (roce_buf_len >> BYTES_TO_CYCLES) : (is_last && !is_new) ? roce_len_left : pkt_len;
    single_pkt                <= ((roce_buf_len >> BYTES_TO_CYCLES) <= pkt_len);
    nxt_pkt_is_last           <= (roce_len_left <= (pkt_len<<1)) && !pkt_is_imm;
    unsync_flag               <= unsync[hif_gnt_idx_r] ? '1 : (pkt_is_imm && !pkt_is_done);
    unsync_cnt_r              <= unsync_cnt[hif_gnt_idx_r];
    roce_val_addr             <= ((roce_rel_addr_cycles - unsync_cnt_r) << CYCLES_TO_BYTES);
    threshold_val             <= roce_nxt_vaddr2[r_threshold_idx];
    threshold_is_lt           <= ((1 << r_threshold_idx) <= eth_pkt_len [hif_gnt_idx_r]); // Threshold is smaller than line size
    nxt_pkt_le                <= (roce_nxt_vaddr[r_threshold_idx] != threshold_val);
  end
end

assign cur_pkt_len_pages    = (cur_pkt_len >> PAGES_TO_CYCLES);
assign roce_rel_addr_cycles = (roce_rel_addr << PAGES_TO_CYCLES);
assign crc_ram_addr         = crc_idx[hif_gnt_idx_r];
assign crc_in               = (is_new) ? '1 : crc_ram_data;

generate
  if (WIDE_DATA) begin
    assign round_cur_pkt_len    = '0;
  end else begin
    assign round_cur_pkt_len = |cur_pkt_len[0+:CYCLES_TO_PAGES] ? 1 : 0;
  end
endgenerate

// Find the next enabled buffer after the current buffer
always_comb begin
  if (BUFFER_4K_REG) begin
    buf_ptr_nxt = (buf_ptr == roce_buf_ptr_end) ? roce_buf_ptr_start : (buf_ptr + 1);
  end 
  else begin
    case(buf_ptr[1:0])
    2'd0: begin
      buf_ptr_nxt = (roce_buf_mask[1]) ? 'd1 :
                    (roce_buf_mask[2]) ? 'd2 :
                    (roce_buf_mask[3]) ? 'd3 :
                                         'd0 ;
    end
    2'd1: begin
      buf_ptr_nxt = (roce_buf_mask[2]) ? 'd2 :
                    (roce_buf_mask[3]) ? 'd3 :
                    (roce_buf_mask[0]) ? 'd0 :
                                         'd1 ;
    end
    2'd2: begin
      buf_ptr_nxt = (roce_buf_mask[3]) ? 'd3 :
                    (roce_buf_mask[0]) ? 'd0 :
                    (roce_buf_mask[1]) ? 'd1 :
                                         'd2 ;
    end
    2'd3: begin
      buf_ptr_nxt = (roce_buf_mask[0]) ? 'd0 :
                    (roce_buf_mask[1]) ? 'd1 :
                    (roce_buf_mask[2]) ? 'd2 :
                                         'd3 ;
    end
  endcase
  end
end


//------------------------------------------------------------------------------------------------//
// ROCE Header
//------------------------------------------------------------------------------------------------//

logic [ROCE_HDR_WIDTH-1:0] hdr_roce;
logic [ROCE_HDR_WIDTH-1:0] hdr_roce_be;
logic [7:0]                se_m_pad_tver;
logic [15:0]               pkey;
logic [7:0]                opcode;
logic [63:0]               roce_addr;

  // UDP
localparam HEADER_WIDTH = 336;

logic [15:0] pld_len;
logic [15:0] udp_len;
logic [15:0] udp_chksum;

assign pld_len    = dma_len + (((hdr_type == 2'b10 || (is_data_wr_imm)) ? 'd36 : 'd32)); // Account for RoCE WR Imm;
assign udp_len    = pld_len + 'd8;
assign udp_chksum = 16'h0; // set to 0 if unused

// IPV4 Header
logic [ 3:0] ipv4_version;
logic [ 3:0] ipv4_ihl;
logic [ 5:0] ipv4_dscp;
logic [ 1:0] ipv4_ecn;
logic [15:0] ipv4_len;
logic [15:0] ipv4_id;
logic [ 2:0] ipv4_flag;
logic [12:0] ipv4_offset;
logic [ 7:0] ipv4_ttl;
logic [ 7:0] ipv4_protocol;
logic [31:0] ipv4_src_addr;
logic [31:0] ipv4_dst_addr;
logic [31:0] w_ipv4_dst_addr;
logic [15:0] dev_udp_port;
logic [47:0] src_mac_addr;
logic [18:0] ipv4_chksum_calc [7];

assign ipv4_version  = 4'h4;
assign ipv4_ihl      = 4'h5;
assign ipv4_dscp     = 6'h0;
assign ipv4_ecn      = 2'h0;
assign ipv4_len      = udp_len + 20;
assign ipv4_id       = 16'h0000;
assign ipv4_flag     = 3'h2;
assign ipv4_offset   = 13'h0;
assign ipv4_ttl      = 8'h40;
assign ipv4_protocol = 8'h11;
assign ipv4_src_addr = i_dev_ip_addr[hif_gnt_idx_r];
assign ipv4_dst_addr = hif_ip_addr;

assign dev_udp_port  = w_dev_udp_port[hif_gnt_idx_r];

// Ethernet Header

logic [15:0] eth_type;

assign eth_type     = 16'h0800; // IPv4 EtherType
assign src_mac_addr = i_dev_mac_addr[hif_gnt_idx_r];

assign se_m_pad_tver       = {1'b0, 1'b0, 2'h0, 4'h0}; //Solicted Event, MigReq, Pad Count, Transport Version
assign pkey                = 16'hFFFF;
assign opcode              = (pkt_is_imm || is_data_wr_imm) ? 8'h2B : 8'h2A;
assign is_data_wr_imm      = md_disable ? (is_last || single_pkt) : '0;

assign hdr_roce = {
  // Ethernet Header 14 Bytes
  hif_mac_addr, src_mac_addr, eth_type,
  // IPv4 Header 20 Bytes
  ipv4_version, ipv4_ihl, ipv4_dscp, ipv4_ecn, ipv4_len,
  ipv4_id, ipv4_flag, ipv4_offset,
  ipv4_ttl, ipv4_protocol, hif_ipv4_chksum,
  ipv4_src_addr,
  ipv4_dst_addr,
  // UDP Header 8 bytes
  dev_udp_port, hif_udp_port, udp_len, udp_chksum,
  // RoCE Header 28 Bytes
  opcode,se_m_pad_tver,pkey,8'h0,roce_dest_qp,
  8'h0,psn,roce_addr,rkey,
  16'h0,dma_len
};


assign roce_addr[32:0]  = ((hdr_type[1]) ? roce_imm_vaddr : roce_vaddr) << (PAGES_TO_CYCLES + CYCLES_TO_BYTES);
assign roce_addr[63:33] = roce_buf_fixed_msb[31:1];


always_ff @(posedge i_pclk) begin
  // C0
  ipv4_chksum_calc[0] <= ipv4_src_addr[31:16] + ipv4_src_addr[15: 0];
  ipv4_chksum_calc[1] <= sif_ip_addr[31:16] + sif_ip_addr[15: 0];
  ipv4_chksum_calc[2] <= pld_len[15:0] + 16'hC52D;
  // C1
  ipv4_chksum_calc[3] <= ipv4_chksum_calc[1] + ipv4_chksum_calc[2];
  // C2
  ipv4_chksum_calc[4] <= ipv4_chksum_calc[0] + ipv4_chksum_calc[3];
  // C3
  ipv4_chksum_calc[5] <= ipv4_chksum_calc[4][15:0] + ipv4_chksum_calc[4][18:16];
  // C4
  ipv4_chksum_calc[6] <= ipv4_chksum_calc[5][15:0] + ipv4_chksum_calc[5][16];
end

assign sif_ipv4_chksum = ~ipv4_chksum_calc[6][15:0];



//------------------------------------------------------------------------------------------------//
// COE Header
//------------------------------------------------------------------------------------------------//

// From FSM
logic [7:0]  sequence_number;                 // PSN
logic [5:0]  channel_number;                  // Buffer index
logic [7:0]  flags;                           // SOF + EOF
logic [27:0] byte_offset;                     // vaddr
logic [27:0] byte_offset_imm;                 // vaddr

logic [COE_HDR_WIDTH-1:0] hdr_coe;
logic [COE_HDR_WIDTH-1:0] hdr_coe_be;

assign sequence_number = psn[7:0];
assign channel_number  = roce_dest_qp[5:0];  // RoCE Dest QP address space to channel_number
assign flags[3:0]      = is_new                       ? 4'h1:
                         pkt_is_imm || is_data_wr_imm ? 4'h2:
                                         4'h0;

assign flags[7:4]  = (threshold_is_lt || (is_last ? le_last : line_end)) && !pkt_is_imm && !is_new ? 4'h1: 4'h0;

assign byte_offset     = roce_addr[27:0];

// Constants
logic [11:0] vlan_id;
logic [2:0]  pcp;
logic        dei;
logic [15:0] ether_type;
logic [7:0]  sub_type;
logic        sv;
logic [7:0]  acf_msg_type;
logic [7:0]  acf_msg_length;
logic        mtv;
logic [4:0]  line_type;
logic [63:0] stream_id;
logic [7:0]  image_sensor_id;
logic        e;
logic        se;
logic        fcv;
logic [1:0]  pad;
logic [1:0]  version;
logic [1:0]  exposure;
logic [15:0] vlan;

assign vlan_id            = 12'hCCC;
assign vlan               = 16'h8100;
assign pcp                = 3'h6;
assign dei                = 1'b1;
assign ether_type         = 16'h22F0;
assign sub_type           = 8'h82;
assign sv                 = 1'b1;
assign mtv                = 1'b1;
assign stream_id          = 64'h0000_0000_AAAA_AAAA;
assign acf_msg_type       = 'h0C;
assign image_sensor_id    = sif_gnt_idx_r;
assign pad                = '0;
assign line_type          = 6'h3F;
assign e                  = '0;
assign se                 = '0;
assign fcv                = '0;
assign version            = '0;
assign exposure           = '0;
assign acf_msg_length     = '0;

assign hdr_coe = {
  hif_mac_addr,
  src_mac_addr,
  //vlan,pcp,dei, vlan_id,
  ether_type,
  sub_type, sv,23'h0,
  stream_id,
  acf_msg_type, acf_msg_length, pad,mtv,5'h0,image_sensor_id,
  cur_ptp[63:0],
  //27'h0, line_type,
  //32'h0,
  sequence_number, e, se, fcv, version, exposure, 3'h0, channel_number, flags,
  2'b00,frame_num[1:0],byte_offset
};

//------------------------------------------------------------------------------------------------//
// WR Immediate
//------------------------------------------------------------------------------------------------//

localparam WR_IMM_SIZE = 384;

logic              imm_axis_tvalid;
logic              imm_axis_tlast;
logic [W_DATA-1:0] imm_axis_tdata;
logic [W_KEEP-1:0] imm_axis_tkeep;
logic              imm_axis_tuser;
logic              imm_axis_tready;

logic [31:0]                wr_imm;
logic [31:0]                wr_imm_be;
logic [WR_IMM_SIZE-1:0]     wr_imm_data;
logic [WR_IMM_SIZE-1:0]     wr_imm_pkt_be;

logic [31:0]                md_flags;

assign bpw_only_flag = !tlast_aligned & !unsync_flag;
assign md_flags = {30'h0, bpw_only_flag,unsync_flag};

assign wr_imm         = (BUFFER_4K_REG) ? {psn[19:0],buf_ptr[11:0]} : {psn[23:0],6'h0,buf_ptr[1:0]};
assign wr_imm_data    = {md_flags,8'h0,psn,frame_crc,16'h0,sof_ptp,roce_val_addr,16'h0,frame_num,16'h0,cur_ptp};

always_ff @(posedge i_pclk) begin
  if (i_prst) begin
    cur_ptp <= '0;
  end
  else begin
    if (hdr_trigger) begin
      cur_ptp <= cur_ptp_sync;
    end
  end
end


//------------------------------------------------------------------------------------------------//
// HDR BE
//------------------------------------------------------------------------------------------------//

genvar k;
generate
  for (k=0; k<COE_HDR_WIDTH/8; k++) begin : gen_hdr_coe_byte_align
    assign hdr_coe_be[k*8+:8] = hdr_coe[(COE_HDR_WIDTH/8-1-k)*8+:8];
  end
  for (k=0; k<WR_IMM_SIZE/8; k++) begin : gen_imm_byte_align
    assign wr_imm_pkt_be[k*8+:8] = wr_imm_data[(WR_IMM_SIZE/8-1-k)*8+:8];
  end
  for (k=0; k<4; k++) begin : gen_wr_imm_byte_align
    assign wr_imm_be[k*8+:8] = wr_imm[(4-1-k)*8+:8];
  end
  for (k=0; k<ROCE_HDR_WIDTH/8; k++) begin : gen_hdr_roce_byte_align
    assign hdr_roce_be[k*8+:8] = hdr_roce[(ROCE_HDR_WIDTH/8-1-k)*8+:8];
  end
endgenerate


//------------------------------------------------------------------------------------------------//
// HDR Creation
//------------------------------------------------------------------------------------------------//

// Possible Headers (Header Type)
// 0. RoCE Header
// 1. CoE Header
// 2. RoCE Header + WR IMM + Metadata Packet
// 3. COE Header + Metadata Packet

localparam ROCE_HDR_CYCLES    = (ROCE_HDR_WIDTH-1)/W_DATA + 1;
localparam ROCE_HDR_DATA_BITS = (ROCE_HDR_WIDTH%W_DATA) == 0 ? W_DATA : (ROCE_HDR_WIDTH%W_DATA);
localparam ROCE_HDR_PAD_BITS  = W_DATA - ROCE_HDR_DATA_BITS;
localparam ROCE_HDR_TKEEP     = (ROCE_HDR_DATA_BITS == W_DATA) ? '1 : {'0,{(ROCE_HDR_DATA_BITS/8){1'b1}}};
localparam ROCE_HDR_TKEEP_32B = (ROCE_HDR_DATA_BITS == W_DATA) ? '1 : {'0,{((ROCE_HDR_DATA_BITS/8)+32){1'b1}}};

localparam ROCE_IMM_HDR_CYCLES    = (ROCE_IMM_HDR_WIDTH-1)/W_DATA + 1;
localparam ROCE_IMM_HDR_DATA_BITS = (ROCE_IMM_HDR_WIDTH%W_DATA) == 0 ? W_DATA : (ROCE_IMM_HDR_WIDTH%W_DATA);
localparam ROCE_IMM_HDR_PAD_BITS  = W_DATA - ROCE_IMM_HDR_DATA_BITS;
localparam ROCE_IMM_HDR_TKEEP     = (ROCE_IMM_HDR_DATA_BITS == W_DATA) ? '1 : {'0,{(ROCE_IMM_HDR_DATA_BITS/8){1'b1}}};
localparam ROCE_IMM_HDR_TKEEP_32B = (ROCE_IMM_HDR_DATA_BITS == W_DATA) ? '1 : {'0,{((ROCE_IMM_HDR_DATA_BITS/8)+32){1'b1}}};

localparam COE_HDR_CYCLES    = (COE_HDR_WIDTH-1)/W_DATA + 1;
localparam COE_HDR_DATA_BITS = (COE_HDR_WIDTH%W_DATA) == 0 ? W_DATA : (COE_HDR_WIDTH%W_DATA);
localparam COE_HDR_PAD_BITS  = W_DATA - COE_HDR_DATA_BITS;
localparam COE_HDR_TKEEP     = (COE_HDR_DATA_BITS == W_DATA) ? '1 : {'0,{(COE_HDR_DATA_BITS/8){1'b1}}};

logic [W_DATA-1:0] hdr_axis_tdata;
logic [W_KEEP-1:0] hdr_axis_tkeep;
logic              hdr_axis_tlast;
logic              hdr_axis_tuser;
logic              hdr_axis_tvalid;
logic              hdr_axis_tready;


logic [W_DATA-1:0] dwh_axis_tdata [N_HOST]; // Header and Data Cycle
logic [W_DATA-1:0] dwd_axis_tdata [N_HOST]; // Data and Previous Data Cycle

logic [W_DATA-1:0] dp_axis_tdata_r [N_HOST];
logic [N_HOST-1:0] dp_axis_tlast_r;
logic [W_KEEP-1:0] dp_axis_tkeep_r [N_HOST];
logic [N_HOST-1:0] dp_axis_tvalid_r;

logic [2:0]        w_hdr_type;

dp_pkt_hdr #(
  .W_DATA         ( W_DATA         ),
  .W_KEEP         ( W_KEEP         ),
  .ROCE_HDR_WIDTH ( ROCE_HDR_WIDTH ),
  .COE_HDR_WIDTH  ( COE_HDR_WIDTH  ),
  .WR_IMM_SIZE    ( 32             ),
  .METADATA_SIZE  ( WR_IMM_SIZE    ),
  .PADDING_SIZE   ( 128*8          )
) hdr_formation (
  .i_clk         ( i_pclk                                    ),
  .i_rst         ( i_prst                                    ),
  .i_roce_header ( hdr_roce_be                               ),
  .i_coe_header  ( (DIS_COE) ? '0 : hdr_coe_be               ),
  .i_wr_imm      ( wr_imm_be                                 ),
  .i_metadata    ( wr_imm_pkt_be                             ),
  .i_header_type ( w_hdr_type                                ),
  .i_trigger     ( hdr_trigger                               ),
  .o_busy        ( hdr_is_busy                               ),
  .o_axis_tdata  ( hdr_axis_tdata                            ),
  .o_axis_tlast  ( hdr_axis_tlast                            ),
  .o_axis_tvalid ( hdr_axis_tvalid                           ),
  .o_axis_tkeep  ( hdr_axis_tkeep                            ),
  .o_axis_tuser  ( hdr_axis_tuser                            ),
  .i_axis_tready ( hdr_axis_tready                           )
);

assign w_hdr_type = (DIS_COE) ? {is_data_wr_imm,hdr_type[1],1'b0} : {is_data_wr_imm,hdr_type };


genvar n;
generate
  for (n=0; n<N_HOST; n++) begin : gen_hdr_mux
    always_ff @(posedge i_pclk) begin
      if (i_prst) begin
        dp_axis_tdata_r[n]  <= '0;
        dp_axis_tlast_r[n]  <= '0;
        dp_axis_tvalid_r[n] <= '0;
        dp_axis_tkeep_r[n]  <= '0;
      end
      else begin
        if (dp_axis_tready[n]) begin
          dp_axis_tdata_r[n]  <= dp_axis_tdata[n];
          dp_axis_tlast_r[n]  <= dp_axis_tlast[n];
          dp_axis_tvalid_r[n] <= dp_axis_tvalid[n];
        end
        if (n==hif_gnt_idx_r && (hdr_axis_tvalid || (cycle_32byte && hdr_trigger))) begin
          dp_axis_tkeep_r[n] <= (hdr_type[0] ? COE_HDR_TKEEP : is_data_wr_imm ? 
                                (cycle_32byte ? ROCE_IMM_HDR_TKEEP_32B : ROCE_IMM_HDR_TKEEP) :
                                (cycle_32byte ? ROCE_HDR_TKEEP_32B : ROCE_HDR_TKEEP));
        end
      end
    end

    if (W_DATA <= 16) begin // Header goes evenly into axis stream
      assign dwh_axis_tdata[n] = hdr_axis_tdata;
      assign dwd_axis_tdata[n] = dp_axis_tdata_r[n];
    end
    else begin
      assign dwh_axis_tdata[n] = hif_is_1722b[n]       ? {dp_axis_tdata[n][COE_HDR_PAD_BITS-1:0], hdr_axis_tdata[0+:COE_HDR_DATA_BITS]} :
                                 hif_is_data_wr_imm[n] ? {dp_axis_tdata[n][ROCE_IMM_HDR_PAD_BITS-1:0], hdr_axis_tdata[0+:ROCE_IMM_HDR_DATA_BITS]} :
                                                         {dp_axis_tdata[n][ROCE_HDR_PAD_BITS-1:0], hdr_axis_tdata[0+:ROCE_HDR_DATA_BITS]};
      assign dwd_axis_tdata[n] = hif_is_1722b[n]       ? {dp_axis_tdata[n][COE_HDR_PAD_BITS-1:0], dp_axis_tdata_r[n][W_DATA-1:COE_HDR_PAD_BITS]} :
                                 hif_is_data_wr_imm[n] ? {dp_axis_tdata[n][ROCE_IMM_HDR_PAD_BITS-1:0], dp_axis_tdata_r[n][W_DATA-1:ROCE_IMM_HDR_PAD_BITS]} :
                                                         {dp_axis_tdata[n][ROCE_HDR_PAD_BITS-1:0], dp_axis_tdata_r[n][W_DATA-1:ROCE_HDR_PAD_BITS]};
    end

    always_comb begin
      if (n==hif_gnt_idx_r && !hdr_type[1] && hdr_axis_tvalid) begin // HDR + Data
        dout_axis_tdata[n] = (hdr_axis_tlast) ? dwh_axis_tdata[n] : hdr_axis_tdata;
        dout_axis_tlast[n] = (cycle_32byte) ? hdr_axis_tlast : dp_axis_tlast_r[n];
        dout_axis_tvalid[n] = (cycle_32byte) ? hdr_axis_tvalid : (dp_axis_tvalid_r[n] | dp_axis_tvalid[n]);
        dout_axis_tkeep[n] = dout_axis_tlast[n] ? dp_axis_tkeep_r[n] : '1;
        dp_axis_tready[n] = (hdr_axis_tlast && dout_axis_tready[n]);
      end
      else if (n==hif_gnt_idx_r && hdr_type[1] && hdr_axis_tvalid) begin // HDR only
        dout_axis_tdata[n] = hdr_axis_tdata;
        dout_axis_tlast[n] = hdr_axis_tlast;
        dout_axis_tvalid[n] = '1;
        dout_axis_tkeep[n] = hdr_axis_tkeep;
        dp_axis_tready[n] = '0;
      end
      else begin  // Steady State Data
        dout_axis_tdata[n] = dwd_axis_tdata[n];
        dout_axis_tlast[n] = (cycle_32byte) ? '0 : dp_axis_tlast_r[n];
        dout_axis_tvalid[n] = (cycle_32byte) ? '0 : dp_axis_tvalid_r[n];
        dout_axis_tkeep[n] = dout_axis_tlast[n] ? dp_axis_tkeep_r[n] : '1;
        dp_axis_tready[n] = dout_axis_tready[n] && hif_data_en[n];
      end

    end
    assign dout_axis_tuser[n] = hif_is_1722b[n];
  end
endgenerate

assign hdr_axis_tready = dout_axis_tready[hif_gnt_idx_r];

//------------------------------------------------------------------------------------------------//
// iCRC
//------------------------------------------------------------------------------------------------//

logic [W_DATA-1:0] dreg_axis_tdata [N_HOST];
logic [W_KEEP-1:0] dreg_axis_tkeep [N_HOST];
logic [N_HOST-1:0] dreg_axis_tlast;
logic [N_HOST-1:0] dreg_axis_tuser;
logic [N_HOST-1:0] dreg_axis_tvalid;
logic [N_HOST-1:0] dreg_axis_tready;

generate
  for (n=0; n<N_HOST; n++) begin : gen_roce_icrc

    if (N_INPT > 16) begin
      axis_reg # (
        .DWIDTH              ( W_DATA + W_KEEP + 1 + 1)
      ) u_axis_reg (
        .clk                ( i_pclk                                                                        ),
        .rst                ( i_prst                                                                        ),
        .i_axis_rx_tvalid   ( dout_axis_tvalid[n]                                                           ),
        .i_axis_rx_tdata    ( {dout_axis_tdata[n],dout_axis_tlast[n],dout_axis_tuser[n],dout_axis_tkeep[n]} ),
        .o_axis_rx_tready   ( dout_axis_tready[n]                                                           ),
        .o_axis_tx_tvalid   ( dreg_axis_tvalid[n]                                                           ),
        .o_axis_tx_tdata    ( {dreg_axis_tdata[n],dreg_axis_tlast[n],dreg_axis_tuser[n],dreg_axis_tkeep[n]} ),
        .i_axis_tx_tready   ( dreg_axis_tready[n]                                                           )
      );
    end
    else begin
      assign dreg_axis_tdata[n] = dout_axis_tdata[n];
      assign dreg_axis_tlast[n] = dout_axis_tlast[n];
      assign dreg_axis_tuser[n] = dout_axis_tuser[n];
      assign dreg_axis_tkeep[n] = dout_axis_tkeep[n];
      assign dreg_axis_tvalid[n] = dout_axis_tvalid[n];
      assign dout_axis_tready[n] = dreg_axis_tready[n];
    end

    roce_icrc #(
      .W_DATA                 ( W_DATA               )
    ) u_roce_icrc (
      .pclk                   ( i_pclk                ),
      .prst                   ( i_prst                ),
      .i_crc_en               ( !dreg_axis_tuser[n]   ),
      .i_axis_rx_tvalid       ( dreg_axis_tvalid[n]   ),
      .i_axis_rx_tlast        ( dreg_axis_tlast[n]    ),
      .i_axis_rx_tkeep        ( dreg_axis_tkeep[n]    ),
      .i_axis_rx_tdata        ( dreg_axis_tdata[n]    ),
      .i_axis_rx_tuser        ( '0                    ),
      .o_axis_rx_tready       ( dreg_axis_tready[n]   ),
      .o_axis_tx_tvalid       ( o_axis_tvalid[n]      ),
      .o_axis_tx_tlast        ( o_axis_tlast[n]       ),
      .o_axis_tx_tkeep        ( o_axis_tkeep[n]       ),
      .o_axis_tx_tdata        ( o_axis_tdata[n]       ),
      .o_axis_tx_tuser        ( o_axis_tuser[n]       ),
      .i_axis_tx_tready       ( i_axis_tready[n]      )
    );
  end
endgenerate

//------------------------------------------------------------------------------------------------//
// DV signals
//------------------------------------------------------------------------------------------------//

logic              wr_imm_trigger_coe;
logic [W_INPT-1:0] gnt_idx_r;
logic [N_HOST-1:0] gnt_en;
logic              wr_imm_trigger;



assign wr_imm_trigger_coe = hdr_trigger && (hdr_type == 2'b11);
assign wr_imm_trigger     = hdr_trigger && (hdr_type == 2'b10);
assign gnt_idx_r          = sif_gnt_idx_r;
assign gnt_en             = pkt_len_en;

//------------------------------------------------------------------------------------------------//
// AXIS Assertions
//------------------------------------------------------------------------------------------------//

`ifdef ASSERT_ON
  // Input AXIS Assertions (one per input port)
  logic [31:0] fv_axis_inp_byt_cnt       [N_INPT];
  logic [31:0] fv_axis_inp_byt_cnt_nxt   [N_INPT];
  generate
  for (genvar gi = 0; gi < N_INPT; gi++) begin : gen_axis_inp_check
    axis_checker #(
      .STBL_CHECK   (1),
      .NLST_BT_B2B  (0),
      .MIN_PKTL_CHK (0),
      .MAX_PKTL_CHK (0),
      .AXI_TDATA    (W_DATA),
      .AXI_TUSER    (1),
`ifdef SIMULATION
      .SIMULATION   (1),
`endif
      .PKT_MIN_LENGTH (58),
      .PKT_MAX_LENGTH (MTU)
    ) assert_dp_pkt_input_axis (
      .clk            (i_pclk),
      .rst            (i_prst),
      .axis_tvalid    (i_axis_tvalid[gi]),
      .axis_tlast     (i_axis_tlast[gi]),
      .axis_tkeep     (i_axis_tkeep[gi]),
      .axis_tdata     (i_axis_tdata[gi]),
      .axis_tuser     (i_axis_tuser[gi]),
      .axis_tready    (o_axis_tready[gi]),
      .byte_count     (fv_axis_inp_byt_cnt[gi]),
      .byte_count_nxt (fv_axis_inp_byt_cnt_nxt[gi])
    );

    assert_dp_pkt_input_axis_tkeep_all_1s: assert property (
      @(posedge i_pclk) disable iff (i_prst)
      i_axis_tvalid[gi] |-> i_axis_tkeep[gi] == '1
    );

    // dp_pkt expects the input TVALID to be always HIGH when TREADY is HIGH,
    // unless there's been a TLAST on the prior cycle:
    assert_dp_pkt_input_axis_tvalid_high_when_output_tready: assert property (
      @(posedge i_pclk) disable iff (i_prst)
      o_axis_tready[gi] && !$past(i_axis_tvalid[gi] && i_axis_tlast[gi], 1) |->
      i_axis_tvalid[gi]
    );

    assert_dp_pkt_input_axis_tvalid_falls_on_tlast_or_fell_tready: assert property(
      @(posedge i_pclk) disable iff (i_prst)
      $fell(i_axis_tvalid[gi]) |->
      $fell(o_axis_tready[gi]) || $past(i_axis_tlast[gi], 1)
    );
  end
  endgenerate

  localparam COE_MD_PKT_BYTES = COE_HDR_WIDTH/8 + 128;
  localparam MIN_OUTPUT_PKT_LEN = (COE_MD_PKT_BYTES <= (COE_HDR_WIDTH/8+W_KEEP)) ?
                                   COE_MD_PKT_BYTES :
                                  (COE_HDR_WIDTH/8+W_KEEP);

  // Output AXIS Assertions (one per output port)
  logic [31:0] fv_axis_out_byt_cnt       [N_HOST];
  logic [31:0] fv_axis_out_byt_cnt_nxt   [N_HOST];
  generate
  for (genvar go = 0; go < N_HOST; go++) begin : gen_axis_out_check
    // Tlast must be zero if tvalid is low
    assert_dp_pkt_output_axis_tlast_tvalid: assert property (
      @(posedge i_pclk) disable iff (i_prst)
      (!o_axis_tlast[go] || (o_axis_tvalid[go] && o_axis_tlast[go])));

    axis_checker #(
      .STBL_CHECK   (1),
      .NLST_BT_B2B  (1),
      .MIN_PKTL_CHK (1),
      .MAX_PKTL_CHK (1),
      .AXI_TDATA    (W_DATA),
      .AXI_TUSER    (1),
`ifdef SIMULATION
      .SIMULATION   (1),
`endif
      .PKT_MIN_LENGTH (MIN_OUTPUT_PKT_LEN),
      .PKT_MAX_LENGTH (MTU)
    ) assert_dp_pkt_output_axis (
      .clk            (i_pclk),
      .rst            (i_prst),
      .axis_tvalid    (o_axis_tvalid[go]),
      .axis_tlast     (o_axis_tlast[go]),
      .axis_tkeep     (o_axis_tkeep[go]),
      .axis_tdata     (o_axis_tdata[go]),
      .axis_tuser     (o_axis_tuser[go]),
      .axis_tready    (i_axis_tready[go]),
      .byte_count     (fv_axis_out_byt_cnt[go]),
      .byte_count_nxt (fv_axis_out_byt_cnt_nxt[go])
    );
  end
  endgenerate
`endif

endmodule
