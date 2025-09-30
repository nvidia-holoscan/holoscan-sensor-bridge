module eth_100gb 
    import apb_pkg::*;
#(
    parameter                               DWIDTH = 512,
    parameter                               RX_ERR_WIDTH = 6,
    parameter                               TX_ERR_WIDTH = 1,
    parameter                               EMPTY_WIDTH = 6,
    parameter                               KEEP_WIDTH = DWIDTH/8
) (             
              
    input   logic                           hif_clk             ,
    input   logic                           fpga_clk_100        ,
    input   logic                           eth_reset_in        , 
    output  logic                           eth_rdy             , 
    input   logic                           heartbeat           ,
                                  
    output  logic                           eth_clk             ,
    output  logic                           eth_rst_n           ,
                                  
    // 25G IO                           
    input   logic                           i_clk_ref           ,
    input   logic  [3:0]                    i_rx_serial         ,
    output  logic  [3:0]                    o_tx_serial         ,
          
    //RX      
    output  logic                           o_eth_axis_rx_tvalid,
    output  logic  [DWIDTH-1:0]             o_eth_axis_rx_tdata ,
    output  logic                           o_eth_axis_rx_tlast ,
    output  logic                           o_eth_axis_rx_tuser ,
    output  logic  [KEEP_WIDTH-1:0]         o_eth_axis_rx_tkeep ,
          
    //TX      
    input   logic                           i_eth_axis_tx_tvalid,
    input   logic  [DWIDTH-1:0]             i_eth_axis_tx_tdata ,
    input   logic                           i_eth_axis_tx_tlast ,
    input   logic                           i_eth_axis_tx_tuser ,
    input   logic  [KEEP_WIDTH-1:0]         i_eth_axis_tx_tkeep ,
    output  logic                           o_eth_axis_tx_tready,
    
    //APB Interface
    input   logic                           i_eth_apb_psel      ,
    input   logic                           i_eth_apb_penable   ,
    input   logic  [31:0]                   i_eth_apb_paddr     ,  
    input   logic  [31:0]                   i_eth_apb_pwdata    , 
    input   logic                           i_eth_apb_pwrite    , 
    output  logic                           o_eth_apb_pready    , 
    output  logic  [31:0]                   o_eth_apb_prdata    , 
    output  logic                           o_eth_apb_pserr     
    
);


//------------------------------------------------------------------------------
// Reset
//------------------------------------------------------------------------------
localparam                  NUM_RST               = 32;
logic [NUM_RST-1:0]         eth_rst_count_n       = '0 /* spyglass disable SYNTH_89 -- Waive initial assignment*/;
logic                       hif_rst;


//------------------------------------------------------------------------------
// AVMM Signals
//------------------------------------------------------------------------------
logic [20:0]                eth_rcfg_addr;
logic                       eth_rcfg_write;
logic                       eth_rcfg_read;
logic [31:0]                eth_rcfg_writedata;
logic [31:0]                eth_rcfg_readdata;
logic                       eth_rcfg_readdatavalid;
logic                       eth_rcfg_waitrequest;
          
logic [18:0]                xcvr0_rcfg_addr; 
logic                       xcvr0_rcfg_write;
logic                       xcvr0_rcfg_read;
logic [7:0]                 xcvr0_rcfg_writedata;
logic [7:0]                 xcvr0_rcfg_readdata;
logic                       xcvr0_rcfg_waitrequest;
          
logic [18:0]                xcvr1_rcfg_addr; 
logic                       xcvr1_rcfg_write;
logic                       xcvr1_rcfg_read;
logic [7:0]                 xcvr1_rcfg_writedata;
logic [7:0]                 xcvr1_rcfg_readdata;
logic                       xcvr1_rcfg_waitrequest;
          
logic [18:0]                xcvr2_rcfg_addr; 
logic                       xcvr2_rcfg_write;
logic                       xcvr2_rcfg_read;
logic [7:0]                 xcvr2_rcfg_writedata;
logic [7:0]                 xcvr2_rcfg_readdata;
logic                       xcvr2_rcfg_waitrequest;
          
logic [18:0]                xcvr3_rcfg_addr; 
logic                       xcvr3_rcfg_write;
logic                       xcvr3_rcfg_read;
logic [7:0]                 xcvr3_rcfg_writedata;
logic [7:0]                 xcvr3_rcfg_readdata;
logic                       xcvr3_rcfg_waitrequest;
          
logic [4*19-1:0]            xcvr_rcfg_addr;
logic [3:0]                 xcvr_rcfg_write;
logic [3:0]                 xcvr_rcfg_read;
logic [4*8-1:0]             xcvr_rcfg_writedata;
logic [4*8-1:0]             xcvr_rcfg_readdata;
logic [3:0]                 xcvr_rcfg_waitrequest;
          
logic [10:0]                rsfec_rcfg_addr;
logic                       rsfec_rcfg_write;
logic                       rsfec_rcfg_read;
logic [7:0]                 rsfec_rcfg_writedata;
logic [7:0]                 rsfec_rcfg_readdata;
logic                       rsfec_rcfg_waitrequest;

//------------------------------------------------------------------------------
// Adaptation FSM and Related Logic Signals
//------------------------------------------------------------------------------
enum logic [3:0]            {SEQ_WAIT_TO_START, SEQ_START, SEQ_STEP_0, SEQ_STEP_1, SEQ_STEP_2, SEQ_STEP_3, SEQ_STEP_4, SEQ_STEP_5, SEQ_STEP_6, SEQ_PAUSE, SEQ_WAIT, SEQ_ERROR, SEQ_WAIT_STATUS, SEQ_DONE} seq_state;

localparam                  PAUSE_TIME      = 32'd50000000;
localparam                  STAT_WAIT_TIME  = 32'd1000000000; 
  
logic [31:0]                seq_status;
logic [31:0]                seq_status_last;
logic [3:0]                 seq_step_cnt;
logic [31:0]                timer;
logic [31:0]                seq_time;
logic                       seq_time_en;
logic [7:0]                 seq_start_addr;
logic [7:0]                 seq_end_addr;
logic                       seq_has_started;
logic                       start_sequencer;
logic                       eth_rst_from_seq_n;
logic [2:0]                 seq_attempts;
logic                       seq_failed;
logic                       seq_status_good;

//------------------------------------------------------------------------------
// APB and GP Register Signals
//------------------------------------------------------------------------------
localparam NUM_CTRL_REG     = 2;
localparam NUM_STAT_REG     = 6;

//Initiator ports
apb_m2s                     sw_apb_m2s            [1];
apb_s2m                     sw_apb_s2m            [1];

//Recipient Ports
apb_m2s                     s_apb_m2s             [2];
apb_s2m                     s_apb_s2m             [2];


logic [31:0]                ctrl_reg              [NUM_CTRL_REG];
logic [31:0]                stat_reg              [NUM_STAT_REG];
logic [31:0]                stat_cmd_addr;
logic [7:0]                 stat_cmd_data;
logic [7:0]                 stat_cmd_mask;
logic [7:0]                 stat_cmd_dataval;
logic [2:0]                 stat_cmd_type;
logic [7:0]                 stat_rom_addr;
logic [3:0]                 stat_seq_state;



//------------------------------------------------------------------------------
// Clocks
//------------------------------------------------------------------------------
logic                       i_reconfig_clk;
logic                       o_clk_pll_div64;

//------------------------------------------------------------------------------
// Ethernet IP Status
//------------------------------------------------------------------------------
logic                       o_tx_lanes_stable; 
logic                       o_rx_pcs_ready; 
logic                       o_ehip_ready; 
logic                       o_rx_block_lock; 
logic                       o_rx_am_lock; 
logic                       o_rx_hi_ber; 
logic                       o_local_fault_status; 
logic                       o_remote_fault_status; 
logic                       i_stats_snapshot;
logic                       o_cdr_lock;
logic                       o_tx_pll_locked;
logic                       o_rxstatus_valid;
logic [39:0]                o_rxstatus_data;
logic [1:0]                 eth_rdy_sync;

//------------------------------------------------------------------------------
// Ethernet Rx AXIS
//------------------------------------------------------------------------------
logic                       rx_valid;
logic [DWIDTH-1:0]          rx_data;
logic                       rx_startofpacket;
logic                       rx_endofpacket;
logic [EMPTY_WIDTH-1:0]     rx_empty;
logic [RX_ERR_WIDTH-1:0]    rx_error;

//------------------------------------------------------------------------------
// Ethernet Tx AXIS
//------------------------------------------------------------------------------
logic                       tx_ready;
logic                       tx_valid;
logic [DWIDTH-1:0]          tx_data;
logic [TX_ERR_WIDTH-1:0]    tx_error;
logic                       tx_startofpacket;
logic                       tx_endofpacket;
logic [EMPTY_WIDTH-1:0]     tx_empty;
logic                       tx_chk_sop;

//------------------------------------------------------------------------------
// Ethernet AVMM
//------------------------------------------------------------------------------
logic [31:0]                eth_avmm_address;
logic                       eth_avmm_write;
logic [31:0]                eth_avmm_writedata;
logic                       eth_avmm_read;
logic [31:0]                eth_avmm_readdata;
logic                       eth_avmm_readdatavalid;
logic                       eth_avmm_waitrequest;


//------------------------------------------------------------------------------
// Ethernet Rx AVST to AXIS Signals
//------------------------------------------------------------------------------
localparam RX_FIFO_WIDTH    = DWIDTH + 1 + 1 + 6 + 6; //DATA + SOP + EOP + EMPTY + ERROR
localparam RX_FIFO_DEPTH    = 32;
localparam RX_ADDR_WIDTH    = $clog2(RX_FIFO_DEPTH);
localparam RX_RAM_TYPE      = "M20K";

logic [RX_FIFO_WIDTH-1:0]   rx_fifo_din;
logic [RX_FIFO_WIDTH-1:0]   rx_fifo_dout;
logic                       rx_fifo_empty;
logic                       rx_fifo_rdreq;
logic                       rx_fifo_avst_valid;
logic [511:0]               rx_fifo_avst_data;
logic                       rx_fifo_avst_sop;
logic                       rx_fifo_avst_eop;
logic [5:0]                 rx_fifo_avst_emp;
logic [5:0]                 rx_fifo_avst_error;

//------------------------------------------------------------------------------
// Ethernet Tx AXIS to AVST Signals
//------------------------------------------------------------------------------
localparam TX_FIFO_WIDTH      = DWIDTH + 1 + 1 + EMPTY_WIDTH + TX_ERR_WIDTH; //DATA + SOP + EOP + EMPTY + ERROR
localparam TX_DC_FIFO_DEPTH   = 32;
localparam TX_SC_FIFO_DEPTH   = 256;
localparam TX_DC_ADDR_WIDTH   = $clog2(TX_DC_FIFO_DEPTH);
localparam TX_SC_ADDR_WIDTH   = $clog2(TX_SC_FIFO_DEPTH);
localparam TX_DEPTH_MSB       = $clog2(TX_SC_FIFO_DEPTH);
localparam TX_RAM_TYPE        = "M20K";

logic                       tx_avst_valid;
logic                       tx_avst_startofpacket;
logic                       tx_avst_endofpacket;
logic [DWIDTH-1:0]          tx_avst_data;
logic [EMPTY_WIDTH-1:0]     tx_avst_empty;
logic [TX_ERR_WIDTH-1:0]    tx_avst_error;
logic [TX_FIFO_WIDTH-1:0]   tx_fifo_din;
logic [TX_FIFO_WIDTH-1:0]   tx_fifo_dout;
logic                       tx_fifo_empty;
logic                       tx_fifo_full;
logic                       tx_fifo_rdreq;
logic [DWIDTH-1:0]          tx_dc_fifo_data;
logic [TX_ERR_WIDTH-1:0]    tx_dc_fifo_error;
logic                       tx_dc_fifo_startofpacket;
logic                       tx_dc_fifo_endofpacket;
logic [EMPTY_WIDTH-1:0]     tx_dc_fifo_empty; 
logic                       tx_pkt_fifo_full;
logic                       tx_pkt_fifo_afull;
logic                       tx_pkt_fifo_empty;
logic                       tx_pkt_fifo_rdreq;
logic                       tx_pkt_fifo_wrreq;
logic                       tx_pkt_fifo_startofpacket;
logic                       tx_pkt_fifo_endofpacket;
logic                       tx_valid_sop_eop;
logic [TX_FIFO_WIDTH-1:0]   tx_pkt_fifo_din;
logic [TX_FIFO_WIDTH-1:0]   tx_pkt_fifo_dout;
logic [TX_DEPTH_MSB:0]      tx_pkt_fifo_count;

//------------------------------------------------------------------------------
// AVMM to APB and Sequencer Signals
//------------------------------------------------------------------------------
logic                       seq_psel;
logic                       seq_penable;
logic [31:0]                seq_paddr;
logic                       seq_pwrite;
logic [31:0]                seq_pwdata;
logic                       seq_pready;
logic [31:0]                seq_prdata;
logic                       seq_pserr;
logic                       avmm_to_apb_psel;
logic                       avmm_to_apb_penable;
logic [31:0]                avmm_to_apb_paddr;
logic                       avmm_to_apb_pwrite;
logic [31:0]                avmm_to_apb_pwdata;
logic                       avmm_to_apb_pready;
logic [31:0]                avmm_to_apb_prdata;
logic                       avmm_to_apb_pserr;
logic                       seq_active;

//------------------------------------------------------------------------------
// Ethernet Status Sync Signals
//------------------------------------------------------------------------------
logic [9:0]                 eth_status_sync_in;
logic [9:0]                 eth_status_sync_out;
logic                       tx_lanes_stable_sync;
logic                       rx_pcs_ready_sync;
logic                       ehip_rdy_sync;
logic                       rx_block_lock_sync;
logic                       rx_am_lock_sync;
logic                       rx_hi_ber_sync;
logic                       local_fault_sync;
logic                       remote_fault_sync;
logic                       cdr_locked_sync;
logic                       tx_pll_locked_sync;

//------------------------------------------------------------------------------
// Clocks
//------------------------------------------------------------------------------
assign eth_clk          = o_clk_pll_div64;
assign i_reconfig_clk   = fpga_clk_100;


//------------------------------------------------------------------------------
// Ethernet Ready
//------------------------------------------------------------------------------
always_ff @(posedge eth_clk) begin
  if (!eth_rst_n) begin
    eth_rdy_sync        <= 1'b0; 
  end else begin
    eth_rdy_sync        <= {eth_rdy_sync[0], o_tx_lanes_stable}; 
  end
end

assign eth_rdy          = eth_rdy_sync[1];

//------------------------------------------------------------------------------
// Reset
//------------------------------------------------------------------------------
always_ff @(posedge eth_clk ) begin
    eth_rst_count_n       <= {o_tx_pll_locked, eth_rst_count_n[NUM_RST-1:1]};
end

assign eth_rst_n        = eth_rst_count_n[0];


//------------------------------------------------------------------------------
// Reset sync for hif_rst
//------------------------------------------------------------------------------
reset_sync u_hif_rst_sync (
  .i_clk                                    ( hif_clk                           ),
  .i_arst_n                                 ( eth_rst_n                         ),
  .i_srst                                   ( 1'b0                              ),
  .i_locked                                 ( 1'b1                              ),
  .o_arst                                   (                                   ),
  .o_arst_n                                 (                                   ),
  .o_srst                                   ( hif_rst                           ),
  .o_srst_n                                 (                                   )
);

//------------------------------------------------------------------------------
// 100Gbps Ethernet IP Core
//------------------------------------------------------------------------------
`ifndef  SPYGLASS
    eth_100G#(
        .enforce_max_frame_size             ( "enable"                          )
    ) eth_100g_inst (                        //spyglass disable ErrorAnalyzeBBox
        .i_stats_snapshot                   ( '0                                ),
        .o_cdr_lock                         ( o_cdr_lock                        ),
        .o_tx_pll_locked                    ( o_tx_pll_locked                   ),

        .i_eth_reconfig_addr                ( eth_rcfg_addr                     ),
        .i_eth_reconfig_read                ( eth_rcfg_read                     ),
        .i_eth_reconfig_write               ( eth_rcfg_write                    ),
        .o_eth_reconfig_readdata            ( eth_rcfg_readdata                 ),
        .o_eth_reconfig_readdata_valid      ( eth_rcfg_readdatavalid            ),
        .i_eth_reconfig_writedata           ( eth_rcfg_writedata                ),
        .o_eth_reconfig_waitrequest         ( eth_rcfg_waitrequest              ),

        .i_rsfec_reconfig_addr              ( rsfec_rcfg_addr                   ),
        .i_rsfec_reconfig_read              ( rsfec_rcfg_read                   ),
        .i_rsfec_reconfig_write             ( rsfec_rcfg_write                  ),
        .o_rsfec_reconfig_readdata          ( rsfec_rcfg_readdata               ),
        .i_rsfec_reconfig_writedata         ( rsfec_rcfg_writedata              ),
        .o_rsfec_reconfig_waitrequest       ( rsfec_rcfg_waitrequest            ),

        .o_tx_lanes_stable                  ( o_tx_lanes_stable                 ),
        .o_rx_pcs_ready                     ( o_rx_pcs_ready                    ),
        .o_ehip_ready                       ( o_ehip_ready                      ),
        .o_rx_block_lock                    ( o_rx_block_lock                   ),
        .o_rx_am_lock                       ( o_rx_am_lock                      ),
        .o_rx_hi_ber                        ( o_rx_hi_ber                       ),
        .o_local_fault_status               ( o_local_fault_status              ),
        .o_remote_fault_status              ( o_remote_fault_status             ),

        .i_clk_tx                           ( eth_clk                           ),
        .i_clk_rx                           ( eth_clk                           ),

        .i_csr_rst_n                        ( eth_rst_from_seq_n                ),
        .i_tx_rst_n                         ( eth_rst_from_seq_n                ),
        .i_rx_rst_n                         ( eth_rst_from_seq_n                ),

        .o_tx_serial                        ( o_tx_serial                       ),
        .i_rx_serial                        ( i_rx_serial                       ),

        .o_tx_serial_n                      (                                   ),
        .i_rx_serial_n                      (                                   ),

        .i_reconfig_clk                     ( i_reconfig_clk                    ),
        .i_reconfig_reset                   ( !eth_rst_from_seq_n               ),

        .o_tx_ready                         ( tx_ready                          ),
        .i_tx_valid                         ( tx_valid                          ),
        .i_tx_data                          ( tx_data                           ),
        .i_tx_error                         ( tx_error                          ),
        .i_tx_startofpacket                 ( tx_startofpacket                  ),
        .i_tx_endofpacket                   ( tx_endofpacket                    ),
        .i_tx_empty                         ( tx_empty                          ),

        .o_rx_valid                         ( rx_valid                          ),
        .o_rx_data                          ( rx_data                           ),
        .o_rx_startofpacket                 ( rx_startofpacket                  ),
        .o_rx_endofpacket                   ( rx_endofpacket                    ),
        .o_rx_empty                         ( rx_empty                          ),
        .o_rx_error                         ( rx_error                          ),

        .i_xcvr_reconfig_address            ( xcvr_rcfg_addr                    ),
        .i_xcvr_reconfig_read               ( xcvr_rcfg_read                    ),
        .i_xcvr_reconfig_write              ( xcvr_rcfg_write                   ),
        .o_xcvr_reconfig_readdata           ( xcvr_rcfg_readdata                ),
        .i_xcvr_reconfig_writedata          ( xcvr_rcfg_writedata               ),
        .o_xcvr_reconfig_waitrequest        ( xcvr_rcfg_waitrequest             ),

        .i_clk_ref                          ( i_clk_ref                         ),
        .o_clk_pll_div64                    ( o_clk_pll_div64                   ),
        .o_clk_pll_div66                    (                                   ),
        .o_clk_rec_div64                    (                                   ),
        .o_clk_rec_div66                    (                                   ),

        .i_tx_skip_crc                      ( '0                                ),
        .o_rxstatus_data                    ( o_rxstatus_data                   ),
        .o_rxstatus_valid                   ( o_rxstatus_valid                  ),

        .i_tx_pfc                           ( '0                                ),
        .o_rx_pfc                           (                                   ),

        .i_tx_pause                         ( 1'b0                              ),
        .o_rx_pause                         (                                   ) 
    );
`else
//Add stub module for Spyglass runs in future to speed up runs



`endif



//------------------------------------------------------------------------------
// DC FIFO to cross the Rx AVST from the eth_clk domain to the hif_clk domain 
//------------------------------------------------------------------------------
assign rx_fifo_din        = {rx_data, rx_startofpacket, rx_endofpacket, rx_empty, rx_error};
assign rx_fifo_rdreq      = !rx_fifo_empty;

dcfifo #(
  .lpm_width                        ( RX_FIFO_WIDTH                             ),
  .lpm_numwords                     ( RX_FIFO_DEPTH                             ),
  .lpm_widthu                       ( RX_ADDR_WIDTH                             ),
  .lpm_showahead                    ( "OFF"                                     ),
  .lpm_type                         ( "DCFIFO"                                  ),
  .lpm_hint                         ( "DISABLE_EMBEDDED_TIMING_CONSTRAINT=TRUE" ),
  .add_usedw_msb_bit                ( "ON"                                      ),
  .clocks_are_synchronized          ( "FALSE"                                   ),
  .ram_block_type                   ( RX_RAM_TYPE                               ),
  .write_aclr_synch                 ( "ON"                                      ),
  .read_aclr_synch                  ( "ON"                                      ),
  .intended_device_family           ( "Stratix 10"                              ),
  .enable_ecc                       ( "FALSE"                                   ),
  .overflow_checking                ( "ON"                                      ),
  .underflow_checking               ( "ON"                                      ),
  .rdsync_delaypipe                 ( 4                                         ),
  .wrsync_delaypipe                 ( 4                                         )
) u_eth_rx_dc_fifo (         
  .aclr                             ( !eth_rst_n                                ),
  .wrclk                            ( eth_clk                                   ),
  .wrreq                            ( rx_valid                                  ),
  .data                             ( rx_fifo_din                               ),
  .wrfull                           (                                           ),
  .wrempty                          (                                           ),
  .wrusedw                          (                                           ),
  .rdclk                            ( hif_clk                                   ),
  .rdreq                            ( rx_fifo_rdreq                             ),
  .q                                ( rx_fifo_dout                              ),
  .rdfull                           (                                           ),
  .rdempty                          ( rx_fifo_empty                             ),
  .rdusedw                          (                                           )
);
  
assign {rx_fifo_avst_data, rx_fifo_avst_sop, rx_fifo_avst_eop, rx_fifo_avst_emp, rx_fifo_avst_error} = rx_fifo_dout;

//Valid signal is one cycle delay of the read request (b/c not showahead FIFO)
always_ff @(posedge hif_clk) begin
  rx_fifo_avst_valid    <= rx_fifo_rdreq;
end

//------------------------------------------------------------------------------
// Translate the Rx AVST to AXIS 
//------------------------------------------------------------------------------
avst_to_axis #(
  .DWIDTH                           ( DWIDTH                                    ),
  .ERR_WIDTH                        ( RX_ERR_WIDTH                              )
) u_avst_to_axis (                        
  .clk                              ( hif_clk                                   ),
  .rst                              ( hif_rst                                   ),
  //Avalon Streaming Input                        
  .avst_valid                       ( rx_fifo_avst_valid                        ),
  .avst_start                       ( rx_fifo_avst_sop                          ),
  .avst_end                         ( rx_fifo_avst_eop                          ),
  .avst_data                        ( rx_fifo_avst_data                         ),
  .avst_empty                       ( rx_fifo_avst_emp                          ),
  .avst_error                       ( rx_fifo_avst_error                        ),
  //AXIS Output                       
  .axis_tvalid                      ( o_eth_axis_rx_tvalid                      ),
  .axis_tdata                       ( o_eth_axis_rx_tdata                       ),
  .axis_tlast                       ( o_eth_axis_rx_tlast                       ),
  .axis_tuser                       ( o_eth_axis_rx_tuser                       ),
  .axis_tkeep                       ( o_eth_axis_rx_tkeep                       )
);


//------------------------------------------------------------------------------
// Translate the Tx AXIS to AVST 
//------------------------------------------------------------------------------
axis_to_avst u_axis_to_avst (
  .clk                              ( hif_clk                                   ),
  .rst                              ( hif_rst                                   ),
  //AXIS Input  
  .axis_tvalid                      ( i_eth_axis_tx_tvalid                      ),
  .axis_tdata                       ( i_eth_axis_tx_tdata                       ),
  .axis_tlast                       ( i_eth_axis_tx_tlast                       ),
  .axis_tuser                       ( i_eth_axis_tx_tuser                       ),
  .axis_tkeep                       ( i_eth_axis_tx_tkeep                       ),
  .axis_tready                      ( o_eth_axis_tx_tready                      ),
  //AVST Output 
  .avst_valid                       ( tx_avst_valid                             ),
  .avst_start                       ( tx_avst_startofpacket                     ),
  .avst_end                         ( tx_avst_endofpacket                       ),
  .avst_data                        ( tx_avst_data                              ),
  .avst_empty                       ( tx_avst_empty                             ),
  .avst_error                       ( tx_avst_error                             ),
  .avst_ready                       ( tx_avst_ready                             )
);

assign tx_avst_ready = !tx_fifo_full;

//------------------------------------------------------------------------------
// Cross the Tx AXIS from the hif_clk domain to the eth_clk domain
//------------------------------------------------------------------------------

//First, cross the data into the ethernet clock domain using a shallow DCFIFO. Then use a larger
//SCFIFO as a packet buffer. Once a packet begins egress to the ethernet IP, the user logic cannot
//stall the process. Meaning, the user side cannot deassert the tx_valid when tx_ready is asserted 
//during packet transmission (between a startofpacket and endofpacket). Therefore, the packet FIFO 
//read logic waits until there is a full packet in the buffer before allowing data to be read out. 

assign tx_fifo_din        = {tx_avst_data, tx_avst_startofpacket, tx_avst_endofpacket, tx_avst_empty, tx_avst_error};
assign tx_fifo_rdreq      = !tx_fifo_empty && !tx_pkt_fifo_afull; 

dcfifo #(
  .lpm_width                ( TX_FIFO_WIDTH                                     ),
  .lpm_numwords             ( TX_DC_FIFO_DEPTH                                  ),
  .lpm_widthu               ( TX_DC_ADDR_WIDTH                                  ),
  .lpm_showahead            ( "ON"                                              ),
  .lpm_type                 ( "DCFIFO"                                          ),
  .lpm_hint                 ( "DISABLE_EMBEDDED_TIMING_CONSTRAINT=TRUE"         ),
  .add_usedw_msb_bit        ( "ON"                                              ),
  .clocks_are_synchronized  ( "FALSE"                                           ),
  .ram_block_type           ( TX_RAM_TYPE                                       ),
  .write_aclr_synch         ( "ON"                                              ),
  .read_aclr_synch          ( "ON"                                              ),
  .intended_device_family   ( "Stratix 10"                                      ),
  .enable_ecc               ( "FALSE"                                           ),
  .overflow_checking        ( "ON"                                              ),
  .underflow_checking       ( "ON"                                              ),
  .rdsync_delaypipe         ( 4                                                 ),
  .wrsync_delaypipe         ( 4                                                 )
) u_eth_tx_dc_fifo (                                    
  .aclr                     ( hif_rst                                           ),
  .wrclk                    ( hif_clk                                           ),
  .wrreq                    ( tx_avst_valid                                     ),
  .data                     ( tx_fifo_din                                       ),
  .wrfull                   ( tx_fifo_full                                      ),
  .wrempty                  (                                                   ),
  .wrusedw                  (                                                   ),
  .rdclk                    ( eth_clk                                           ),
  .rdreq                    ( tx_fifo_rdreq                                     ),
  .q                        ( tx_fifo_dout                                      ),
  .rdfull                   (                                                   ),
  .rdempty                  ( tx_fifo_empty                                     ),
  .rdusedw                  (                                                   )
);
  
assign tx_pkt_fifo_din    = {tx_dc_fifo_data, tx_dc_fifo_startofpacket, tx_dc_fifo_endofpacket, tx_dc_fifo_empty, tx_dc_fifo_error};

always_ff @(posedge eth_clk) begin
  if (!eth_rst_n) begin
    tx_dc_fifo_data             <= 'b0;
    tx_dc_fifo_startofpacket    <= 'b0;
    tx_dc_fifo_endofpacket      <= 'b0;
    tx_dc_fifo_empty            <= 'b0;
    tx_dc_fifo_error            <= 'b0;
  end else begin
    {tx_dc_fifo_data, 
     tx_dc_fifo_startofpacket, 
     tx_dc_fifo_endofpacket, 
     tx_dc_fifo_empty, 
     tx_dc_fifo_error}          <= tx_fifo_dout;
    tx_pkt_fifo_wrreq           <= tx_fifo_rdreq;
  end
end

scfifo #(
  .lpm_width                ( TX_FIFO_WIDTH                                     ),
  .lpm_numwords             ( TX_SC_FIFO_DEPTH                                  ),
  .lpm_widthu               ( TX_SC_ADDR_WIDTH                                  ),
  .lpm_showahead            ( "OFF"                                             ),
  .add_ram_output_register  ( "ON"                                              ),
  .lpm_type                 ( "SCFIFO"                                          ),
  .ram_block_type           ( TX_RAM_TYPE                                       ),
  .almost_full_value        ( TX_SC_FIFO_DEPTH - 4                              ),
  .intended_device_family   ( "Stratix 10"                                      )
) u_eth_tx_pkt_fifo (                                 
  .sclr                     ( !eth_rst_n                                        ),
  .aclr                     ( 1'b0                                              ),
  .clock                    ( eth_clk                                           ),
  .wrreq                    ( tx_pkt_fifo_wrreq                                 ),
  .data                     ( tx_pkt_fifo_din                                   ),
  .full                     ( tx_pkt_fifo_full                                  ),
  .almost_full              ( tx_pkt_fifo_afull                                 ),
  .empty                    ( tx_pkt_fifo_empty                                 ),
  .almost_empty             (                                                   ),
  .usedw                    (                                                   ),
  .rdreq                    ( tx_pkt_fifo_rdreq                                 ),
  .q                        ( tx_pkt_fifo_dout                                  )
);

//Keep track of packets in the buffer. Counter is sized for FIFO depth + 1. If a valid endofpacket
//occurs on the input, increment the counter. If a valid endofpacket occurs on the output, decrement
//the counter. If a valid endofpacket occurs on both the input and output at the same time, don't 
//adjust the counter. 
always_ff @(posedge eth_clk) begin
  if (!eth_rst_n) begin
    tx_pkt_fifo_count           <= 'b0;
  end else begin
    if ((tx_pkt_fifo_wrreq && tx_dc_fifo_endofpacket) && !(tx_valid && tx_ready && tx_endofpacket)) begin
      tx_pkt_fifo_count         <= !(&tx_pkt_fifo_count) ? tx_pkt_fifo_count + 1'b1 : tx_pkt_fifo_count;
    end else if ((tx_valid && tx_ready && tx_endofpacket) && !(tx_pkt_fifo_wrreq && tx_dc_fifo_endofpacket)) begin
      tx_pkt_fifo_count         <= (|tx_pkt_fifo_count) ? tx_pkt_fifo_count - 1'b1 : tx_pkt_fifo_count;
    end
  end
end

//Allow data to be read from the packet buffer when there is at least one full packet in it and the
//ethernet IP is ready to accept data. 
assign tx_pkt_fifo_rdreq                                                         = tx_ready && |(tx_pkt_fifo_count) && !tx_pkt_fifo_empty && !tx_endofpacket;
assign {tx_data, tx_pkt_fifo_startofpacket, tx_pkt_fifo_endofpacket, tx_empty, tx_error}  = tx_pkt_fifo_dout;
assign tx_valid                                                                  = (tx_pkt_fifo_startofpacket && tx_chk_sop) || tx_valid_sop_eop; 
assign tx_startofpacket                                                          = tx_valid && tx_pkt_fifo_startofpacket;
assign tx_endofpacket                                                            = tx_valid && tx_pkt_fifo_endofpacket;

always_ff @(posedge eth_clk) begin
  if (!eth_rst_n) begin
    tx_valid_sop_eop <= 1'b0;
    tx_chk_sop       <= 1'b0;
  end
  else begin
    tx_chk_sop <= tx_pkt_fifo_rdreq;
    if (tx_endofpacket && tx_ready)
      tx_valid_sop_eop <= 1'b0;
    else if (tx_pkt_fifo_startofpacket && tx_chk_sop) begin
      tx_valid_sop_eop <= 1'b1;
    end
  end
end


//------------------------------------------------------------------------------
// Use APB Switch to carve up the incoming bus:
// 0xI000_0000 to 0xI7FF_FFFF --> Ethernet IP 
// 0xI800_0000 to 0xIFFF_FFFF --> General Purpose Registers
// Where I = base user index, 1 for interface 0, 2 for interface 1
//------------------------------------------------------------------------------


//Map the initiator structure signals to the APB signals on IO
assign sw_apb_m2s[0].psel     = i_eth_apb_psel;
assign sw_apb_m2s[0].penable  = i_eth_apb_penable;
assign sw_apb_m2s[0].paddr    = i_eth_apb_paddr;
assign sw_apb_m2s[0].pwdata   = i_eth_apb_pwdata;
assign sw_apb_m2s[0].pwrite   = i_eth_apb_pwrite;
assign o_eth_apb_pready       = sw_apb_s2m[0].pready;
assign o_eth_apb_prdata       = sw_apb_s2m[0].prdata;
assign o_eth_apb_pserr        = sw_apb_s2m[0].pserr;

//Instantiate the APB Switch
apb_switch
#(
  .N_MPORT                  ( 1                                                 ),
  .N_SPORT                  ( 2                                                 ),
  .W_OFSET                  ( 27                                                ), 
  .W_SW                     ( 5                                                 ),
  .MERGE_COMPLETER_SIG      ( 0                                                 )
)u_eth_apb_sw ( 
  .i_apb_clk                ( fpga_clk_100                                      ),
  .i_apb_reset              ( eth_reset_in                                      ),
  .i_apb_m2s                ( sw_apb_m2s[0:0]                                   ),
  .o_apb_s2m                ( sw_apb_s2m[0:0]                                   ),
  .i_apb_s2m                ( s_apb_s2m                                         ),
  .o_apb_m2s                ( s_apb_m2s                                         )
);

//------------------------------------------------------------------------------
// Translate the APB to AVMM for the Ethernet IP Module
//------------------------------------------------------------------------------

//Mux the avmm signals based on the sequencer active signal
assign avmm_to_apb_psel     = seq_active  ? seq_psel            : s_apb_m2s[0].psel; 
assign avmm_to_apb_penable  = seq_active  ? seq_penable         : s_apb_m2s[0].penable; 
assign avmm_to_apb_paddr    = seq_active  ? seq_paddr           : s_apb_m2s[0].paddr; 
assign avmm_to_apb_pwrite   = seq_active  ? seq_pwrite          : s_apb_m2s[0].pwrite; 
assign avmm_to_apb_pwdata   = seq_active  ? seq_pwdata          : s_apb_m2s[0].pwdata; 
assign seq_pready           = seq_active  ? avmm_to_apb_pready  : 0;
assign seq_prdata           = seq_active  ? avmm_to_apb_prdata  : 0;
assign seq_pserr            = seq_active  ? avmm_to_apb_pserr   : 0;
assign s_apb_s2m[0].pready  = !seq_active ? avmm_to_apb_pready  : 0;
assign s_apb_s2m[0].prdata  = !seq_active ? avmm_to_apb_prdata  : 0;
assign s_apb_s2m[0].pserr   = !seq_active ? avmm_to_apb_pserr   : 0;

//AVMM to APB Shim module
avmm_to_apb #(
  .AVMM_ADDR_WIDTH              ( 32                                            ),
  .AVMM_DATA_WIDTH              ( 32                                            ),
  .USE_AVMM_READDATAVALID       ( 1                                             ),
  .AVMM_RD_LATENCY              ( 0                                             )
) u_eth_avmm_to_apb (
  .clk                          ( fpga_clk_100                                  ),
  .rst                          ( eth_reset_in                                  ),
  .psel                         ( avmm_to_apb_psel                              ),
  .penable                      ( avmm_to_apb_penable                           ),
  .paddr                        ( avmm_to_apb_paddr                             ),
  .pwrite                       ( avmm_to_apb_pwrite                            ),
  .pwdata                       ( avmm_to_apb_pwdata                            ),
  .pready                       ( avmm_to_apb_pready                            ),
  .prdata                       ( avmm_to_apb_prdata                            ),
  .pserr                        ( avmm_to_apb_pserr                             ),
  .avmm_address                 ( eth_avmm_address                              ),
  .avmm_write                   ( eth_avmm_write                                ),
  .avmm_writedata               ( eth_avmm_writedata                            ),
  .avmm_read                    ( eth_avmm_read                                 ),
  .avmm_readdata                ( eth_avmm_readdata                             ),
  .avmm_readdatavalid           ( eth_avmm_readdatavalid                        ),
  .avmm_waitrequest             ( eth_avmm_waitrequest                          )
);

//-----------------------------------------------------------------------------------------------------
// AVMM Interface Logic
//-----------------------------------------------------------------------------------------------------
//
// There are three AVMM interfaces on the Ethernet IP core:
// 1) Ethernet Reconfiguration Interface: 21-bit address space but only 12-bit used (from Ethernet IP user guide)
// 2) Transceiver Reconfiguration Interface: 19-bit address space per lane (4 lanes)
// 3) RS-FEC Reconfiguration Interface: 11-bits address space
//
// The APB interface at the top of this module is split into two 27-bit address spaces. One 27-bit
// address space will be used to access the three AVMM spaces above. Use upper bits to select the 
// interface to communicate with. The biggest address space is 19-bits (21-bits word aligned) so
// use bits 23:21 for the interface selector:
//
// Address Mapping (word/32-bit aligned):
// Ethernet Reconfig:   0x0000_0000         to    0x001F_FFFF
// XCVR0 Reconfig:      0x0020_0000         to    0x003F_FFFF
// XCVR1 Reconfig:      0x0040_0000         to    0x005F_FFFF
// XCVR2 Reconfig:      0x0060_0000         to    0x007F_FFFF
// XCVR3 Reconfig:      0x0080_0000         to    0x009F_FFFF
// RS-FEC Reconfig:     0x00A0_0000         to    0x00BF_FFFF


//There is one transceiver reconfiguration interface on the IP. The four individual transceiver interfaces are 
//concatenated onto the one interface at IP. 
assign xcvr_rcfg_addr             = {xcvr3_rcfg_addr, xcvr2_rcfg_addr, xcvr1_rcfg_addr, xcvr0_rcfg_addr};
assign xcvr_rcfg_write            = {xcvr3_rcfg_write, xcvr2_rcfg_write, xcvr1_rcfg_write, xcvr0_rcfg_write};
assign xcvr_rcfg_read             = {xcvr3_rcfg_read, xcvr2_rcfg_read, xcvr1_rcfg_read, xcvr0_rcfg_read};
assign xcvr_rcfg_writedata        = {xcvr3_rcfg_writedata, xcvr2_rcfg_writedata, xcvr1_rcfg_writedata, xcvr0_rcfg_writedata};
assign {xcvr3_rcfg_readdata, 
        xcvr2_rcfg_readdata,
        xcvr1_rcfg_readdata,
        xcvr0_rcfg_readdata}      = xcvr_rcfg_readdata;
assign {xcvr3_rcfg_waitrequest,
        xcvr2_rcfg_waitrequest,
        xcvr1_rcfg_waitrequest,
        xcvr0_rcfg_waitrequest}   = xcvr_rcfg_waitrequest;

// Use upper bits of APB/AVMM address to pick the reconfiguration interface to communicate to. 
always_comb begin

  eth_rcfg_addr                   = '0;
  eth_rcfg_write                  = 0;
  eth_rcfg_read                   = 0;
  eth_rcfg_writedata              = '0;  
  xcvr0_rcfg_addr                 = '0;
  xcvr0_rcfg_write                = 0;
  xcvr0_rcfg_read                 = 0;
  xcvr0_rcfg_writedata            = '0;
  xcvr1_rcfg_addr                 = '0;
  xcvr1_rcfg_write                = 0;
  xcvr1_rcfg_read                 = 0;
  xcvr1_rcfg_writedata            = '0;
  xcvr2_rcfg_addr                 = '0;
  xcvr2_rcfg_write                = 0;
  xcvr2_rcfg_read                 = 0;
  xcvr2_rcfg_writedata            = '0;
  xcvr3_rcfg_addr                 = '0;
  xcvr3_rcfg_write                = 0;
  xcvr3_rcfg_read                 = 0;
  xcvr3_rcfg_writedata            = '0;
  rsfec_rcfg_addr                 = '0;
  rsfec_rcfg_write                = 0;
  rsfec_rcfg_read                 = 0;
  rsfec_rcfg_writedata            = '0;
  eth_avmm_waitrequest            = 0;
  eth_avmm_readdata               = '0;
  eth_avmm_readdatavalid          = 0;

  case (eth_avmm_address[23:21])
    3'b000  : begin
      eth_rcfg_addr               = {9'b0, eth_avmm_address[13:2]};                //User guide only specifies 12-bit usable address
      eth_rcfg_write              = eth_avmm_write;
      eth_rcfg_read               = eth_avmm_read;
      eth_rcfg_writedata          = eth_avmm_writedata;                            //32-bit at IP
      eth_avmm_waitrequest        = eth_rcfg_waitrequest;
      eth_avmm_readdata           = eth_rcfg_readdata;                             //32-bit at IP
      eth_avmm_readdatavalid      = eth_rcfg_readdatavalid;
    end
    3'b001  : begin
      xcvr0_rcfg_addr             = eth_avmm_address[20:2];
      xcvr0_rcfg_write            = eth_avmm_write;
      xcvr0_rcfg_read             = eth_avmm_read;
      xcvr0_rcfg_writedata        = eth_avmm_writedata[7:0];
      eth_avmm_waitrequest        = xcvr0_rcfg_waitrequest;
      eth_avmm_readdata           = {24'b0, xcvr0_rcfg_readdata};                   //8-bit at IP
      eth_avmm_readdatavalid      = xcvr0_rcfg_read && !xcvr0_rcfg_waitrequest;     //No read data valid signal so based on waitrequest
    end 
    3'b010  : begin 
      xcvr1_rcfg_addr             = eth_avmm_address[20:2];
      xcvr1_rcfg_write            = eth_avmm_write;
      xcvr1_rcfg_read             = eth_avmm_read;
      xcvr1_rcfg_writedata        = eth_avmm_writedata[7:0];
      eth_avmm_waitrequest        = xcvr1_rcfg_waitrequest;
      eth_avmm_readdata           = {24'b0, xcvr1_rcfg_readdata};                   //8-bit at IP
      eth_avmm_readdatavalid      = xcvr1_rcfg_read && !xcvr1_rcfg_waitrequest;     //No read data valid signal so based on waitrequest
    end 
    3'b011  : begin 
      xcvr2_rcfg_addr             = eth_avmm_address[20:2];
      xcvr2_rcfg_write            = eth_avmm_write;
      xcvr2_rcfg_read             = eth_avmm_read;
      xcvr2_rcfg_writedata        = eth_avmm_writedata[7:0];
      eth_avmm_waitrequest        = xcvr2_rcfg_waitrequest;
      eth_avmm_readdata           = {24'b0, xcvr2_rcfg_readdata};                   //8-bit at IP
      eth_avmm_readdatavalid      = xcvr2_rcfg_read && !xcvr2_rcfg_waitrequest;     //No read data valid signal so based on waitrequest
    end 
    3'b100  : begin 
      xcvr3_rcfg_addr             = eth_avmm_address[20:2];
      xcvr3_rcfg_write            = eth_avmm_write;
      xcvr3_rcfg_read             = eth_avmm_read;
      xcvr3_rcfg_writedata        = eth_avmm_writedata[7:0];
      eth_avmm_waitrequest        = xcvr3_rcfg_waitrequest;
      eth_avmm_readdata           = {24'b0, xcvr3_rcfg_readdata};                   //8-bit at IP
      eth_avmm_readdatavalid      = xcvr3_rcfg_read && !xcvr3_rcfg_waitrequest;     //No read data valid signal so based on waitrequest
    end 
    3'b101  : begin 
      rsfec_rcfg_addr             = eth_avmm_address[12:2];
      rsfec_rcfg_write            = eth_avmm_write;
      rsfec_rcfg_read             = eth_avmm_read;
      rsfec_rcfg_writedata        = eth_avmm_writedata[7:0];
      eth_avmm_waitrequest        = rsfec_rcfg_waitrequest;
      eth_avmm_readdata           = {24'b0, rsfec_rcfg_readdata};                   //8-bit at IP
      eth_avmm_readdatavalid      = rsfec_rcfg_read && !rsfec_rcfg_waitrequest;     //No read data valid signal so based on waitrequest
    end
    default : begin
      eth_rcfg_addr               = '0;
      eth_rcfg_write              = 0;
      eth_rcfg_read               = 0;
      eth_rcfg_writedata          = '0;  
      xcvr0_rcfg_addr             = '0;
      xcvr0_rcfg_write            = 0;
      xcvr0_rcfg_read             = 0;
      xcvr0_rcfg_writedata        = '0;
      xcvr1_rcfg_addr             = '0;
      xcvr1_rcfg_write            = 0;
      xcvr1_rcfg_read             = 0;
      xcvr1_rcfg_writedata        = '0;
      xcvr2_rcfg_addr             = '0;
      xcvr2_rcfg_write            = 0;
      xcvr2_rcfg_read             = 0;
      xcvr2_rcfg_writedata        = '0;
      xcvr3_rcfg_addr             = '0;
      xcvr3_rcfg_write            = 0;
      xcvr3_rcfg_read             = 0;
      xcvr3_rcfg_writedata        = '0;
      rsfec_rcfg_addr             = '0;
      rsfec_rcfg_write            = 0;
      rsfec_rcfg_read             = 0;
      rsfec_rcfg_writedata        = '0;
      eth_avmm_waitrequest        = 0;
      eth_avmm_readdata           = '0;
      eth_avmm_readdatavalid      = 0;
    end
  endcase
end


//Ethernet IP Status signal synchronization
assign eth_status_sync_in   = {o_tx_lanes_stable, o_rx_pcs_ready, o_ehip_ready, o_rx_block_lock, o_rx_am_lock,
                               o_rx_hi_ber, o_local_fault_status, o_remote_fault_status, o_cdr_lock, o_tx_pll_locked};

data_sync #(
  .DATA_WIDTH                 (10) 
) eth_status_sync (
  .clk                        (fpga_clk_100),
  .rst_n                      (!eth_reset_in),
  .sync_in                    (eth_status_sync_in),
  .sync_out                   (eth_status_sync_out)
);

assign {tx_lanes_stable_sync, rx_pcs_ready_sync, ehip_rdy_sync, rx_block_lock_sync, rx_am_lock_sync,
        rx_hi_ber_sync, local_fault_sync, remote_fault_sync, cdr_locked_sync, tx_pll_locked_sync}     = eth_status_sync_out;


//------------------------------------------------------------------------------
// Ethernet Adaptation Sequencing FSM and Related Logic
//------------------------------------------------------------------------------ 

always_ff @(posedge fpga_clk_100) begin
  if (eth_reset_in) begin
    seq_state                   <= SEQ_WAIT_TO_START;
    seq_active                  <= 0;
    seq_step_cnt                <= 0;
    timer                       <= 0;
    start_sequencer             <= 0;
    seq_has_started             <= 0;
    eth_rst_from_seq_n          <= 0;
    seq_start_addr              <= 0;
    seq_end_addr                <= 0;
    seq_attempts                <= 0;
    seq_failed                  <= 0;
    seq_status_last             <= 0;
    seq_time_en                 <= 0;
  end else begin
    
    case (seq_state)
      
      SEQ_WAIT_TO_START  : begin
        if (tx_pll_locked_sync) begin
          seq_state             <= SEQ_PAUSE;
          timer                 <= PAUSE_TIME;
          eth_rst_from_seq_n    <= 1'b1;
          seq_attempts          <= seq_attempts + 1'b1;
          seq_time_en           <= 1'b1;
        end
      end
      
      SEQ_START  : begin
        seq_state               <= SEQ_STEP_0;
        seq_active              <= 1'b1;
        seq_step_cnt            <= 0;
        seq_has_started         <= 1;
      end
      
      SEQ_STEP_0 : begin
        seq_start_addr          <= 8'd0;
        seq_end_addr            <= 8'd30;
        start_sequencer         <= 1'b1;
        seq_state               <= SEQ_WAIT;
      end
      
      SEQ_STEP_1 : begin
        seq_start_addr          <= 8'd31;
        seq_end_addr            <= 8'd116;
        start_sequencer         <= 1'b1;
        seq_state               <= SEQ_WAIT;
      end
      
      SEQ_STEP_2 : begin
        seq_start_addr          <= 8'd149;
        seq_end_addr            <= 8'd186;
        start_sequencer         <= 1'b1;
        seq_state               <= SEQ_WAIT;
      end
      
      SEQ_STEP_3 : begin
        seq_start_addr          <= 8'd0;
        seq_end_addr            <= 8'd30;
        start_sequencer         <= 1'b1;
        seq_state               <= SEQ_WAIT;
      end
      
      SEQ_STEP_4 : begin
        seq_start_addr          <= 8'd31;
        seq_end_addr            <= 8'd52;
        start_sequencer         <= 1'b1;
        seq_state               <= SEQ_WAIT;
      end
      
      SEQ_STEP_5 : begin
        seq_start_addr          <= 8'd117;
        seq_end_addr            <= 8'd153;
        start_sequencer         <= 1'b1;
        seq_state               <= SEQ_WAIT;
      end
      
      SEQ_STEP_6 : begin
        seq_start_addr          <= 8'd186;
        seq_end_addr            <= 8'd186;
        start_sequencer         <= 1'b1;
        seq_state               <= SEQ_WAIT;
      end
      
      SEQ_WAIT  : begin
        if (seq_status[0]) begin
          start_sequencer       <= 1'b0;
          seq_status_last       <= seq_status;
          if (seq_status[1]) begin
            if (seq_attempts <= 5) begin
              seq_state           <= SEQ_ERROR;
              seq_has_started     <= 1'b0;
              timer               <= 20;
              eth_rst_from_seq_n  <= 1'b0;
            end else begin
              seq_state           <= SEQ_DONE;
              seq_failed          <= 1'b1;
            end
          end else begin
          case (seq_step_cnt)
            0: begin
              seq_state         <= SEQ_PAUSE;
              timer             <= PAUSE_TIME;
            end
            1: begin
              seq_state         <= SEQ_STEP_2;
              seq_step_cnt      <= seq_step_cnt + 1'b1;
            end
            2: begin
              seq_state         <= SEQ_STEP_3;
              seq_step_cnt      <= seq_step_cnt + 1'b1;
            end
            3: begin
              seq_state         <= SEQ_PAUSE;
              timer             <= PAUSE_TIME;
            end
            4: begin
              seq_state         <= SEQ_STEP_5;
              seq_step_cnt      <= seq_step_cnt + 1'b1;
            end
            5: begin
              seq_state         <= SEQ_STEP_6;
              seq_step_cnt      <= seq_step_cnt + 1'b1;
            end
            6: begin
              seq_state         <= SEQ_WAIT_STATUS;
              timer             <= STAT_WAIT_TIME;
            end
          endcase
        end
      end
      end
      
      SEQ_PAUSE : begin
        if (!(|timer)) begin
          if (seq_has_started) begin
            seq_step_cnt        <= seq_step_cnt + 1'b1;
            if (seq_step_cnt == 4'd0) begin
              seq_state         <= SEQ_STEP_1;
            end else if (seq_step_cnt == 4'd3) begin
              seq_state         <= SEQ_STEP_4;
            end
          end else begin
            seq_state           <= SEQ_START;
          end
        end else begin
          timer                 <= timer - 1'b1;
        end
      end
      
      SEQ_ERROR : begin
        if (!(|timer)) begin
          seq_state             <= SEQ_WAIT_TO_START;
        end else begin
          timer                 <= timer - 1'b1;
        end
      end
      
      SEQ_WAIT_STATUS : begin
        if (seq_status_good) begin
          seq_state             <= SEQ_DONE;
        end else if (!(|timer)) begin
          seq_state             <= SEQ_ERROR;
          seq_has_started       <= 1'b0;
          timer                 <= 20;
          eth_rst_from_seq_n    <= 1'b0;
        end else begin
          timer                 <= timer - 1'b1;
        end
      end
      
      SEQ_DONE  : begin
        seq_active              <= 1'b0;
        seq_time_en             <= 1'b0;
      end
    endcase
  end
end  

assign seq_status_good = tx_lanes_stable_sync & rx_pcs_ready_sync & rx_block_lock_sync & rx_am_lock_sync & !local_fault_sync & !remote_fault_sync & cdr_locked_sync & tx_pll_locked_sync;

always_ff @(posedge fpga_clk_100) begin
  if (eth_reset_in) begin
    seq_time                    <= 0;
  end else begin
    if (seq_state == SEQ_WAIT_TO_START) begin
      seq_time                  <= 0;
    end else begin
      if (seq_time_en) begin
        seq_time                <= seq_time + 1'b1;
      end
    end
  end
end

//-----------------------------------------------------------------------------------------------------
// Sequencer 
//-----------------------------------------------------------------------------------------------------
eth_adapt_sequencer eth_adapt_sequencer (
  .clk                          ( fpga_clk_100                                  ),
  .rst                          ( eth_reset_in                                  ),
  .start_seq                    ( start_sequencer                               ),
  .seq_start_addr               ( seq_start_addr                                ),
  .seq_end_addr                 ( seq_end_addr                                  ),
  .seq_return_data              (                                               ),
  .seq_last_attr_info           (                                               ),
  .seq_status                   ( seq_status                                    ),
  .psel                         ( seq_psel                                      ),
  .penable                      ( seq_penable                                   ),
  .paddr                        ( seq_paddr                                     ),
  .pwrite                       ( seq_pwrite                                    ),
  .pwdata                       ( seq_pwdata                                    ),
  .pready                       ( seq_pready                                    ),
  .prdata                       ( seq_prdata                                    ),
  .pserr                        ( seq_pserr                                     ),
  .stat_cmd_addr                ( stat_cmd_addr                                 ),
  .stat_cmd_data                ( stat_cmd_data                                 ),
  .stat_cmd_mask                ( stat_cmd_mask                                 ),
  .stat_cmd_dataval             ( stat_cmd_dataval                              ),
  .stat_cmd_type                ( stat_cmd_type                                 ),
  .stat_rom_addr                ( stat_rom_addr                                 ),
  .stat_seq_state               ( stat_seq_state                                )
);

//-----------------------------------------------------------------------------------------------------
// General Purpose Registers
//-----------------------------------------------------------------------------------------------------
//Status Registers
assign stat_reg[0]    = stat_cmd_addr[31:0];
assign stat_reg[1]    = {stat_cmd_data[7:0], stat_cmd_mask[7:0], stat_cmd_dataval[7:0], stat_rom_addr[7:0]};
assign stat_reg[2]    = {25'b0, stat_cmd_type[2:0], stat_seq_state[3:0]};
assign stat_reg[3]    = {8'b0, seq_state[3:0], seq_step_cnt[3:0], 2'b0, seq_status_last[5:0], 1'b0, seq_attempts[2:0], 3'b0, seq_failed};
assign stat_reg[4]    = {22'b0, tx_lanes_stable_sync, rx_pcs_ready_sync, ehip_rdy_sync, rx_block_lock_sync, rx_am_lock_sync, rx_hi_ber_sync, local_fault_sync, remote_fault_sync, cdr_locked_sync, tx_pll_locked_sync}; 
assign stat_reg[5]    = seq_time;

//Register Module
s_apb_reg #(
  .N_CTRL                       ( NUM_CTRL_REG                                  ),
  .N_STAT                       ( NUM_STAT_REG                                  )
) u_eth_gen_purpose_reg (                     
  .i_aclk                       ( fpga_clk_100                                  ), 
  .i_arst                       ( eth_reset_in                                  ),
  .i_apb_m2s                    ( s_apb_m2s[1]                                  ),
  .o_apb_s2m                    ( s_apb_s2m[1]                                  ),
  .i_pclk                       ( fpga_clk_100                                  ), 
  .i_prst                       ( eth_reset_in                                  ),
  .o_ctrl                       ( ctrl_reg                                      ),
  .i_stat                       ( stat_reg                                      )
);

endmodule
