`include "HOLOLINK_def.svh"
///////////////////////////////////////////////////////////////////////////////////////////////////////////
// Top level targeting the AD9986 Evaluation Platform
///////////////////////////////////////////////////////////////////////////////////////////////////////////
module FPGA_top #(
    parameter                           BUILD_REV = 48'h0
) (
  
    //FPGA Clock and Push Button Reset
    input  logic                        FPGA_CLK_IN     ,
    input  logic                        FPGA_PB_RESET_L ,
    
    //Reconfig
    output logic                        FPGA_RECONFIG   ,
    
    //QSFP SERDES
    input  logic    [ 1:0][3:0]         QSFP_RX_P       ,
    output logic    [ 1:0][3:0]         QSFP_TX_P       ,

  //If build, then we can disable JESD and not have pins at top causing issues in Quartus. Else, include for sim.
  `ifdef QUARTUS_BUILD
    `ifdef JESD_SIF_ENABLE
      //FMC SERDES      
      input  logic    [ `JESD_NUM_LANES-1:0]              JESD_RXDP       ,
      input  logic    [ `JESD_NUM_LANES-1:0]              JESD_RXDN       ,
      output logic    [ `JESD_NUM_LANES-1:0]              JESD_TXDP       ,
      output logic    [ `JESD_NUM_LANES-1:0]              JESD_TXDN       ,
    `endif
  `else
    //FMC SERDES      
    input  logic    [ `JESD_NUM_LANES-1:0]              JESD_RXDP       ,
    input  logic    [ `JESD_NUM_LANES-1:0]              JESD_RXDN       ,
    output logic    [ `JESD_NUM_LANES-1:0]              JESD_TXDP       ,
    output logic    [ `JESD_NUM_LANES-1:0]              JESD_TXDN       ,
  `endif
  
    //REFCLKs
    input  logic                        FPGA_REFCLK_EXT ,   //REFCLK0
    input  logic                        FPGA_ETH_REFCLK ,   //REFCLK1
    input  logic                        FMC_GBTCLK0_M2C ,   //REFCLK2
    input  logic                        FMC_GBTCLK1_M2C ,   //REFCLK3
    input  logic                        FMC_GBTCLK2_M2C ,   //REFCLK4
    input  logic                        FMC_GBTCLK3_M2C ,   //REFCLK5
    input  logic                        AD9545_REFCLK   ,   //REFCLK6
    input  logic                        JESD_REFCLK2    ,   //REFCLK7
    input  logic                        ETH_REFCLK0     ,   //REFCLK8
    
    //I2C
    inout tri                           FMC_I2C_SCL_1V8 ,
    inout tri                           FMC_I2C_SDA_1V8 ,
    
    
    
    //Headers, LEDs, and Straps
    output logic                        FPGA_HB_LED     ,
    output logic    [15:0]              DBG_HEADER      ,
    output logic    [15:0]              STX_DBG_LED     ,
    input  logic    [15:0]              DBG_SWITCH      ,
    input  logic    [ 3:0]              STX_STRAP       ,
    
    output logic                        FMC_LA04_N,         //FMC H11 - SPI0_SCLK - AD_SCLK - OUTPUT
    output logic                        FMC_LA04_P,         //FMC H10 - SPI0_MOSI - AD_MOSI - INOUT
    input  logic                        FMC_LA05_N,         //FMC D12 - SPI0_MISO - AD_MISO - INOUT
    output logic                        FMC_LA05_P,         //FMC D11 - SPI0_CSB - AD_CSN - OUTPUT
    output logic                        FMC_LA07_P,         //FMC H13 - RSTB - AD_RST - OUTPUT
    output logic                        FMC_LA11_P,         //                      HMC_SCLK
    inout tri                           FMC_LA12_N,         //FMC G16 - SPI1_SDIO - HMC_SDIO
    output logic                        FMC_LA12_P,         //FMC G15 - SPI1_CSB - HMC_CSNoutput          hmc_csn         ,
    input  logic                        FMC_LA00_CC_TERM_P, //HMC CLK10
    
  
    //QSFP0
//  input  logic                        QSFP0_PWR_GOOD_1V8, //
    output logic                        QSFP0_LPMODE,       //
    output logic                        QSFP0_RESET_L,      //                       
    inout  tri                          QSFP0_SCL,          //
    inout  tri                          QSFP0_SDA,          //
//  input  logic                        QSFP0_INT_L,        //
    output logic                        QSFP0_MODSEL_L,     //
    input  logic                        QSFP0_MODPRS_L,     //
    output logic                        QSFP0_PWR_EN,       //
                                                            
                                                            
    //QSFP1                                                 
//  input  logic                        QSFP1_PWR_GOOD_1V8, //
    output logic                        QSFP1_LPMODE,       //
    output logic                        QSFP1_RESET_L,      //
    inout  tri                          QSFP1_SCL,          //
    inout  tri                          QSFP1_SDA,          //
//  input  logic                        QSFP1_INT_L,        //
    output logic                        QSFP1_MODSEL_L,     //
    input  logic                        QSFP1_MODPRS_L,     //
    output logic                        QSFP1_PWR_EN,       //
    
    
    input  logic                        FMC_CLK0_M2C_TERM_P ,
    input  logic                        FMC_CLK1_M2C_TERM_P ,
    input  logic                        FMC_CLK2_BIDIR_IN_P ,
    
    //FMC Power
    output logic                        FMC_PWR_EN          ,
    output logic                        FMC_LA22_P          ,
    output logic                        FPGA_PG_C2M         ,
    
    //FPGA EEPROM 
    inout  tri                          FPGA_EEPROM_SCL     ,
    inout  tri                          FPGA_EEPROM_SDA     ,
    
    //FPGA EXT Clock OUT
    output logic                        FPGA_EXT_CLK_OUT_P  ,
    //AD9545
    input  logic                        AD9545_CORECLK_P,
    output logic                        AD9545_RESETB,            
    input  logic                        AD9545_M0,                
    input  logic                        AD9545_M1,                
    input  logic                        AD9545_M2,                
    input  logic                        AD9545_M3,                
    input  logic                        AD9545_M4,                
    input  logic                        AD9545_M5,                
    input  logic                        AD9545_M6,                
    inout  tri                          AD9545_SCK,
    inout  tri                          AD9545_SDA  
);

//Always enable IPs in simulation
`ifdef SIMULATION
    `define IP_EN 1
`endif

//Always enable IPs in SPYGLASS
`ifdef SPYGLASS
    `define IP_EN 1
`endif

//Change the JESD Lane Width based on IP flavor
`ifdef JESD_HIGH_PERF
  `define JESD_LANE_DWIDTH 128
`else
  `define JESD_LANE_DWIDTH 64
`endif

localparam                        HALF_SECOND   = 26'h2FAF07F;

//------------------------------------------------------------------------------
// Reset, clocks, and initialization 
//------------------------------------------------------------------------------
logic                             ninit_done;
logic                             fpga_clk_pll_rst;           //FPGA clock PLL reset
logic                             fpga_clk_100;               //100MHz clock from FPGA PLL
logic                             fpga_clk_50;                //50MHz clock from FPGA PLL
logic                             fpga_clk_pll_lock;          //FPGA PLL lock
logic                             rst_sys;                    //System reset - synchronous to 100MHz clock
logic [`HOST_IF_INST-1:0]         eth_rdy;                    //Ethernet ready indication. 
logic                             sys_rst;                    //Hololink Sys Reset
logic [`HOST_IF_INST-1:0]         eth_rst_n;                  //Reset from ethernet module. Synced to eth_clk_div64 from each. 
logic [`HOST_IF_INST-1:0]         eth_clk_div64;              //Ethernet module output clock. 402.83MHz from IP.
logic                             apb_clk;                    //APB clock
logic                             apb_rst;                    //APB reset. From Hololink.
logic                             hif_clk;                    //Hololink host interface clock
logic                             hif_pll_locked;             //Hololink host interface clock PLL lock
logic [`SENSOR_RX_IF_INST-1:0]    sif_rx_clk;                 //Hololink sensor interface clock
logic [`SENSOR_RX_IF_INST-1:0]    sif_rx_rst;                 //Hololink sensor interface reset
logic [`SENSOR_TX_IF_INST-1:0]    sif_tx_clk;                 //Hololink sensor interface clock
logic [`SENSOR_TX_IF_INST-1:0]    sif_tx_rst;                 //Hololink sensor interface reset
logic                             hololink_init_done;         //Hololink initialization done
logic                             pb_rst_debounced;           //Debounced push button
logic                             hl_sw_rst;                  //Hololink software reset
logic                             clk_checker_source;         //Source clock for the clock checker module
logic                             jesd_core_pll_refclk;       //Source clock for the JESD PLL refclk
logic                             jesd_core_clk;              //JESD core clock output from PLL
logic                             jesd_core_dev_clk;          //JESD core device clock
logic                             jesd_pll_locked;            //JESD core PLL locked
logic                             ptp_clk;
logic                             ptp_rst;

//------------------------------------------------------------------------------
// Heartbeat and debug LEDs 
//------------------------------------------------------------------------------
logic                             heartbeat;                  //Heartbeat LED signal
logic   [25:0]                    second_cnt;                 //Counter for heartbeat LED logic
logic   [15:0]                    dbg_led_out;                //Debug LED output

//------------------------------------------------------------------------------
// User APB Drops 
//------------------------------------------------------------------------------
logic [`REG_INST-1:0]             apb_psel;
logic                             apb_penable;
logic [31         :0]             apb_paddr;
logic [31         :0]             apb_pwdata;
logic                             apb_pwrite;
logic [`REG_INST-1:0]             apb_pready;
logic [31         :0]             apb_prdata              [`REG_INST];
logic [`REG_INST-1:0]             apb_pserr;

//------------------------------------------------------------------------------
// Hololink Host Rx/Tx 
//------------------------------------------------------------------------------
logic [`HOST_IF_INST-1:0]         hl_axis_rx_tvalid;
logic [`HOST_WIDTH-1:0]           hl_axis_rx_tdata      [`HOST_IF_INST];
logic [`HOST_IF_INST-1:0]         hl_axis_rx_tlast;
logic [`HOSTUSER_WIDTH-1:0]       hl_axis_rx_tuser      [`HOST_IF_INST];
logic [`HOST_WIDTH/8-1:0]         hl_axis_rx_tkeep      [`HOST_IF_INST];
      
logic [`HOST_IF_INST-1:0]         hl_axis_tx_tvalid;
logic [`HOST_WIDTH-1:0]           hl_axis_tx_tdata      [`HOST_IF_INST];
logic [`HOST_IF_INST-1:0]         hl_axis_tx_tlast;
logic [`HOSTUSER_WIDTH-1:0]       hl_axis_tx_tuser      [`HOST_IF_INST];
logic [`HOST_WIDTH/8-1:0]         hl_axis_tx_tkeep      [`HOST_IF_INST];
logic [`HOST_IF_INST-1:0]         hl_axis_tx_tready;

//------------------------------------------------------------------------------
// Hololink Sensor Rx/Tx 
//------------------------------------------------------------------------------
logic [`SENSOR_RX_IF_INST-1:0]    o_sif_axis_tready;
logic [`DATAPATH_WIDTH-1:0]       sif_rx_tdata              [`SENSOR_RX_IF_INST];
logic [`SENSOR_RX_IF_INST-1:0]    sif_rx_tvalid;
logic [`SENSOR_RX_IF_INST-1:0]    sif_rx_tlast;
logic [`SENSOR_TX_IF_INST-1:0]    sif_tx_tready;
logic [`DATAPATH_WIDTH-1:0]       sif_tx_tdata              [`SENSOR_TX_IF_INST];
logic [`SENSOR_TX_IF_INST-1:0]    sif_tx_tvalid;
logic [`SENSOR_TX_IF_INST-1:0]    sif_tx_tlast;

//------------------------------------------------------------------------------
// I2C
//------------------------------------------------------------------------------
logic [`I2C_INST-1:0]             i2c_scl;
logic [`I2C_INST-1:0]             i2c_sda;
logic [`I2C_INST-1:0]             i2c_scl_en;
logic [`I2C_INST-1:0]             i2c_sda_en;
            
logic                             fpga_eeprom_i2c_scl_sync;
logic                             fpga_eeprom_i2c_sda_sync;
logic                             ad9545_i2c_scl_sync;
logic                             ad9545_i2c_sda_sync;
  
//------------------------------------------------------------------------------
// SPI
//------------------------------------------------------------------------------
logic [`SPI_INST-1:0]             spi_csn;
logic [`SPI_INST-1:0]             spi_sck;
logic [3          :0]             spi_sdio_i [`SPI_INST];
logic [3          :0]             spi_sdio_o [`SPI_INST];
logic [`SPI_INST-1:0]             spi_oen;
          
logic                             hmc_spi_sdio_sync;
logic                             adi_spi_miso_sync;
logic                             adi_mosi;
logic                             adi_miso;
logic                             hmc_sdio;
logic [`GPIO_INST-1:0]            hl_gpio;

//------------------------------------------------------------------------------
// General Purpose Registers
//------------------------------------------------------------------------------
logic                             clk_chk_en;
logic  [31:0]                     clk_target;
logic  [31:0]                     clk_tolerance;
logic                             clk_in_tol;
logic  [31:0]                     clk_count;
logic                             sw_rst_ad9545;
logic                             sw_fpga_reconfig;
logic  [6:0]                      ad9545_mode_pins;

//------------------------------------------------------------------------------
// GPIO Signals needed for MxFE board
//------------------------------------------------------------------------------
logic  [1:0]                      hl_gpio_sync_0;
logic  [1:0]                      hl_gpio_sync_1;
logic  [1:0]                      hl_gpio_sync_2;
logic  [1:0]                      hl_gpio_sync_3;

//------------------------------------------------------------------------------
// PPS
//------------------------------------------------------------------------------
logic                             sys_pps;
logic                             sys_pps_stretch;
logic [17:0]                      timer_cnt;
logic                             timer_done;
logic [47:0]                      ptp_sec;
logic [31:0]                      ptp_nano;

//------------------------------------------------------------------------------
// JESD
//------------------------------------------------------------------------------
logic [`JESD_NUM_LANES*`JESD_LANE_DWIDTH-1:0]   jesd_rx_data;
logic                                           jesd_rx_valid;
logic                                           jesd_rx_last;
logic                                           jesd_tx_rdy;
logic [`JESD_NUM_LANES*`JESD_LANE_DWIDTH-1:0]   jesd_tx_data;
logic                                           jesd_tx_valid;



//------------------------------------------------------------------------------
// Intel Reset Release IP. Indicates when the FPGA has fully configured and 
// user-logic can start to operate.
//------------------------------------------------------------------------------
`ifdef QUARTUS_BUILD
  reset_ip intel_reset_ip (
    .ninit_done                       ( ninit_done       )
  );
`else
  assign ninit_done             = 0;
`endif

//------------------------------------------------------------------------------
// Hold the FPGA clock PLL in reset until the FPGA has completely configured.
//------------------------------------------------------------------------------
assign fpga_clk_pll_rst = ninit_done;

//------------------------------------------------------------------------------
// FPGA PLL - drives the 100MHz clock and an unused 50MHz clock.
//------------------------------------------------------------------------------
`ifdef SIMULATION       //Simulation
  assign fpga_clk_100                       =  FPGA_CLK_IN;
`else                   //Real FPGA build...
  fpga_clk_pll_mfg  fpga_clk_pll (                  
    .rst                            ( fpga_clk_pll_rst    ),
    .refclk                         ( FPGA_CLK_IN         ),
    .locked                         ( fpga_clk_pll_lock   ),   
    .outclk_0                       ( fpga_clk_100        ), 
    .outclk_1                       ( fpga_clk_50         ) 
  );
`endif


//------------------------------------------------------------------------------
// FPGA reset module. Drives an async reset and a sync reset on the 100MHz clock.
//------------------------------------------------------------------------------
logic   hl_sw_rst_sync;
logic   fpga_clk_pll_lock_sync;
logic   hif_pll_locked_sync;


`ifdef SIMULATION       //Simulation
  assign rst_sys                    = !FPGA_PB_RESET_L;
`else                   //Real FPGA build...
  data_sync    #(
    .DATA_WIDTH ( 3                                      )
  ) rst_and_lock_sync (
    .clk        ( fpga_clk_100                           ),
    .rst_n      ( 1'b1                                   ),
    .sync_in    ({hl_sw_rst, fpga_clk_pll_lock, hif_pll_locked}),
    .sync_out   ({hl_sw_rst_sync, fpga_clk_pll_lock_sync, hif_pll_locked_sync})
  );
  
  assign rst_sys                    = !fpga_clk_pll_lock_sync || hl_sw_rst_sync || pb_rst_debounced;
`endif


//------------------------------------------------------------------------------
// Reset Push Button Debouncer
//------------------------------------------------------------------------------
rst_pb_debounce  rst_pb_debounce (
  .clk                            ( fpga_clk_100        ),
  .pb_rst_in                      ( FPGA_PB_RESET_L     ),
  .pb_rst_out                     ( pb_rst_debounced    )
);


//------------------------------------------------------------------------------
// Hololink System Reset
//------------------------------------------------------------------------------
`ifdef SIMULATION
  assign sys_rst                            = rst_sys || !hif_pll_locked;
`else
  assign sys_rst                            = rst_sys || !hif_pll_locked_sync;
`endif


//------------------------------------------------------------------------------
// Heartbeat LED Logic
//------------------------------------------------------------------------------
always_ff @(posedge apb_clk) begin
  if (rst_sys) begin
    second_cnt                            <= 26'b0;
    heartbeat                             <= 1'b0;
  end else begin
    
    if (second_cnt == HALF_SECOND) begin
      second_cnt                          <= 26'b0;
      heartbeat                           <= ~heartbeat;
    end else begin                
      second_cnt                          <= second_cnt + 1'b1;
    end
  end
end

always_ff @(posedge apb_clk) begin
  FPGA_HB_LED                             <= heartbeat;
end


//------------------------------------------------------------------------------
// Debug LED Logic
//------------------------------------------------------------------------------
dbg_led u_dbg_led (
  .clk                            ( apb_clk             ),
  .rst                            ( rst_sys             ),
  .led                            ( dbg_led_out         )
);

always_ff @(posedge apb_clk) begin
  STX_DBG_LED                           <= dbg_led_out;
end

//------------------------------------------------------------------------------
// APB Clock
//------------------------------------------------------------------------------
assign apb_clk                          = fpga_clk_100;

//------------------------------------------------------------------------------
// Sensor Interface
//------------------------------------------------------------------------------
//sif_clk is connected to jesd_core_clk, the output of the JESD PLL. The input to
//the JESD PLL is dependent on JESD_USE_FMC_CLK define. 
generate
  for (genvar i=0; i<`SENSOR_RX_IF_INST; i++) begin : gen_sif_clk
    assign sif_rx_clk[i] = jesd_core_clk;
  end
endgenerate

generate
  for (genvar i=0; i<`SENSOR_TX_IF_INST; i++) begin : gen_sif_tx_clk
    assign sif_tx_clk[i] = jesd_core_clk;
  end
endgenerate

assign sif_rx_tvalid[0]                 = jesd_rx_valid;
assign sif_rx_tdata[0]                  = jesd_rx_data;
assign sif_rx_tlast[0]                  = jesd_rx_last;
assign jesd_tx_valid                    = sif_tx_tvalid[0];
assign jesd_tx_data                     = sif_tx_tdata[0];
assign sif_tx_tready[0]                 = jesd_tx_rdy;


//------------------------------------------------------------------------------
// QSFP GPIO
//------------------------------------------------------------------------------
assign QSFP0_LPMODE                     = 1'b0;
assign QSFP0_RESET_L                    = 1'b1;
assign QSFP0_MODSEL_L                   = 1'b0;
assign QSFP0_PWR_EN                     = 1'b1;
assign QSFP1_LPMODE                     = 1'b0;
assign QSFP1_RESET_L                    = 1'b1;
assign QSFP1_MODSEL_L                   = 1'b0;
assign QSFP1_PWR_EN                     = 1'b1;


//------------------------------------------------------------------------------
// Hololink Host Interface PLL
//------------------------------------------------------------------------------

hif_pll hif_pll (
  .rst                            ( !eth_rst_n[0]       ),      //Reset from ethernet core. Released after TX PLL is locked.
  .refclk                         ( eth_clk_div64[0]    ),      //Ethernet IP output clock - div64 (402.83MHz)
  .locked                         ( hif_pll_locked      ),      //Locked
  .outclk_0                       ( /* x1 clk */        ),      //x1 clock output
  .outclk_1                       ( hif_clk             ),      //div2 clock output 201.415MHz
  .outclk_2                       ( ptp_clk             )
);



//------------------------------------------------------------------------------
// General Purpose Registers
//------------------------------------------------------------------------------

//Mode pins from the AD9545
data_sync    #(
  .DATA_WIDTH ( 7                                                                           )
) ad9545_mode_pins_sync (
  .clk        ( apb_clk                                                                     ),
  .rst_n      ( ~apb_rst                                                                    ),
  .sync_in    ({AD9545_M6, AD9545_M5, AD9545_M4, AD9545_M3, AD9545_M2, AD9545_M1, AD9545_M0}),
  .sync_out   (ad9545_mode_pins                                                             )
);


general_purpose_regs #(
  .N_CTRL_REGS                    (4),
  .N_STAT_REGS                    (3)
) u_gen_purpose_regs (
  .i_apb_clk                      (apb_clk),
  .i_apb_rst                      (apb_rst),
  .i_apb_psel                     (apb_psel[4]),
  .i_apb_penable                  (apb_penable),         
  .i_apb_paddr                    (apb_paddr),           
  .i_apb_pwdata                   (apb_pwdata),          
  .i_apb_pwrite                   (apb_pwrite),          
  .o_apb_pready                   (apb_pready[4]),
  .o_apb_prdata                   (apb_prdata[4]),
  .o_apb_pserr                    (apb_pserr[4]),
  //Control Regs
  .o_sw_ad9545_rst                (sw_rst_ad9545),
  .o_sw_fpga_reconfig             (sw_fpga_reconfig),
  .o_clk_chk_en                   (clk_chk_en),
  .o_clk_chk_target               (clk_target),
  .o_clk_chk_tolerance            (clk_tolerance),
  //Status Regs                   
  .i_clk_chk_in_tolerance         (clk_in_tol),
  .i_clk_chk_count                (clk_count),
  .i_ad9545_mode_pins             (ad9545_mode_pins)
);


//------------------------------------------------------------------------------
// Intel 100Gb Host Interfaces
//------------------------------------------------------------------------------
genvar i;
generate
  for (i=0; i<`HOST_IF_INST; i++) begin: ethernet_100gb
    eth_top eth_top_inst  (
      .hif_clk                    ( hif_clk             ),
      .fpga_clk_100               ( fpga_clk_100        ),
      .eth_reset_in               ( rst_sys             ),
      .eth_rdy                    ( eth_rdy[i]          ),
      .heartbeat                  ( heartbeat           ),
                                    
      .eth_clk                    ( eth_clk_div64[i]    ),
      .eth_rst_n                  ( eth_rst_n[i]        ),
      // 25G IO                     
      .i_clk_ref                  ( FPGA_ETH_REFCLK     ),
      .i_rx_serial                ( QSFP_RX_P[i]        ),
      .o_tx_serial                ( QSFP_TX_P[i]        ),
      //RX                          
      .o_eth_axis_rx_tvalid       ( hl_axis_rx_tvalid[i]),
      .o_eth_axis_rx_tdata        ( hl_axis_rx_tdata[i] ),
      .o_eth_axis_rx_tlast        ( hl_axis_rx_tlast[i] ),
      .o_eth_axis_rx_tuser        ( hl_axis_rx_tuser[i] ),
      .o_eth_axis_rx_tkeep        ( hl_axis_rx_tkeep[i] ),
      //TX                          
      .i_eth_axis_tx_tvalid       ( hl_axis_tx_tvalid[i]),
      .i_eth_axis_tx_tdata        ( hl_axis_tx_tdata[i] ),
      .i_eth_axis_tx_tlast        ( hl_axis_tx_tlast[i] ),
      .i_eth_axis_tx_tuser        ( hl_axis_tx_tuser[i] ),
      .i_eth_axis_tx_tkeep        ( hl_axis_tx_tkeep[i] ),
      .o_eth_axis_tx_tready       ( hl_axis_tx_tready[i]),
      //APB                              
      .i_eth_apb_psel             ( apb_psel[i]         ),
      .i_eth_apb_penable          ( apb_penable         ),
      .i_eth_apb_paddr            ( apb_paddr           ),
      .i_eth_apb_pwdata           ( apb_pwdata          ),
      .i_eth_apb_pwrite           ( apb_pwrite          ),
      .o_eth_apb_pready           ( apb_pready[i]       ),
      .o_eth_apb_prdata           ( apb_prdata[i]       ),
      .o_eth_apb_pserr            ( apb_pserr[i]        )
    );
  end
endgenerate


//------------------------------------------------------------------------------
// Sensor IP - Example of JESD IP Instance. Replace with specific JESD IP or
// other sensor interface IP.
//------------------------------------------------------------------------------
`ifdef SIMULATION
  assign jesd_core_pll_refclk         = FMC_CLK1_M2C_TERM_P;
`elsif JESD_USE_FMC_CLK
  assign jesd_core_pll_refclk         = FMC_CLK1_M2C_TERM_P;
`else
  assign jesd_core_pll_refclk         = AD9545_CORECLK_P;
`endif

jesd_top #(
  .NUM_LANES                      (`JESD_NUM_LANES),
  .LANE_DWIDTH                    (`JESD_LANE_DWIDTH)
) u_jesd_top (                   
  `ifdef SIMULATION
    .i_xcvr_refclk                (FMC_GBTCLK0_M2C),
  `else
    .i_xcvr_refclk                (FPGA_ETH_REFCLK),
  `endif
  .i_core_pll_refclk              (jesd_core_pll_refclk),
  .i_apb_clk                      (apb_clk),
  .i_apb_rst                      (apb_rst),
  .i_apb_psel                     (apb_psel[3]),
  .i_apb_penable                  (apb_penable),
  .i_apb_paddr                    (apb_paddr),
  .i_apb_pwdata                   (apb_pwdata),
  .i_apb_pwrite                   (apb_pwrite),
  .o_apb_pready                   (apb_pready[3]),
  .o_apb_prdata                   (apb_prdata[3]),           
  .o_apb_pserr                    (apb_pserr[3]),
  .i_init_done                    (hololink_init_done),
  .i_sif_rst                      (sif_tx_rst),
  //JESD Clocks, serial data, control                    
  .i_rx_serdes_p                  (JESD_RXDP),
  .i_rx_serdes_n                  (JESD_RXDN),
  .o_jesd_core_clk                (jesd_core_clk),
  .o_tx_serdes_p                  (JESD_TXDP),
  .o_tx_serdes_n                  (JESD_TXDN),
  .i_tx_sysref                    (FMC_CLK0_M2C_TERM_P),
  .i_rx_sysref                    (FMC_CLK0_M2C_TERM_P),
  .o_jesd_tx_intrpt               (),
  .o_jesd_rx_intrpt               (),
  //JESD RX User
  .o_jesd_rx_data                 (jesd_rx_data),
  .o_jesd_rx_valid                (jesd_rx_valid),
  .o_jesd_rx_last                 (jesd_rx_last),
  //JESD TX User
  .i_jesd_tx_data                 (jesd_tx_data),
  .i_jesd_tx_valid                (jesd_tx_valid),
  .o_jesd_tx_rdy                  (jesd_tx_rdy)
);


//------------------------------------------------------------------------------
// Hololink
//------------------------------------------------------------------------------
HOLOLINK_top
#(
  .BUILD_REV                    (BUILD_REV            )
) u_hololink_top (
  .i_sys_rst                    (sys_rst              ),
  // Control Plane
  .i_apb_clk                    (apb_clk              ),
  .o_apb_rst                    (apb_rst              ),
  // APB Register Interface
  .o_apb_psel                   (apb_psel             ),
  .o_apb_penable                (apb_penable          ),
  .o_apb_paddr                  (apb_paddr            ),
  .o_apb_pwdata                 (apb_pwdata           ),
  .o_apb_pwrite                 (apb_pwrite           ),
  .i_apb_pready                 (apb_pready           ),
  .i_apb_prdata                 (apb_prdata           ),
  .i_apb_pserr                  (apb_pserr            ),
//------------------------------------------------------------------------------
// User Auto Initialization Complete
//------------------------------------------------------------------------------
  .o_init_done                  (hololink_init_done   ),  
//------------------------------------------------------------------------------
// Sensor IF
//------------------------------------------------------------------------------
  // Sensor Interface Clock and Reset
  .i_sif_rx_clk                 (sif_rx_clk           ),
  .o_sif_rx_rst                 (sif_rx_rst           ),
  .i_sif_tx_clk                 (sif_tx_clk           ),
  .o_sif_tx_rst                 (sif_tx_rst           ),
  // Sensor Rx Streaming Interface
  .i_sif_axis_tvalid            (sif_rx_tvalid        ),
  .i_sif_axis_tlast             (sif_rx_tlast         ),
  .i_sif_axis_tdata             (sif_rx_tdata         ),
  .i_sif_axis_tkeep             ('{default:'1}        ),
  .i_sif_axis_tuser             ('{default:0}         ),
  .o_sif_axis_tready            (o_sif_axis_tready    ),
  // Sensor Tx Streaming Interface (Unimplemented)
  .o_sif_axis_tvalid            (sif_tx_tvalid        ),
  .o_sif_axis_tlast             (                     ),
  .o_sif_axis_tdata             (sif_tx_tdata         ),
  .o_sif_axis_tkeep             (                     ),
  .o_sif_axis_tuser             (                     ),
  .i_sif_axis_tready            (sif_tx_tready        ),
  // Sensor Event
  .i_sif_event                  ('b0                  ),
//------------------------------------------------------------------------------
// Host IF
//------------------------------------------------------------------------------
  // Host Interface Clock and Reset
  .i_hif_clk                    (hif_clk              ),
  .o_hif_rst                    (                     ),
  // Host Rx Interface
  .i_hif_axis_tvalid            (hl_axis_rx_tvalid    ),
  .i_hif_axis_tlast             (hl_axis_rx_tlast     ),
  .i_hif_axis_tdata             (hl_axis_rx_tdata     ),
  .i_hif_axis_tkeep             (hl_axis_rx_tkeep     ),
  .i_hif_axis_tuser             (hl_axis_rx_tuser     ),
  .o_hif_axis_tready            (                     ),
  // Host Tx Interface
  .o_hif_axis_tvalid            (hl_axis_tx_tvalid    ),
  .o_hif_axis_tlast             (hl_axis_tx_tlast     ),
  .o_hif_axis_tdata             (hl_axis_tx_tdata     ),
  .o_hif_axis_tkeep             (hl_axis_tx_tkeep     ),
  .o_hif_axis_tuser             (hl_axis_tx_tuser     ),
  .i_hif_axis_tready            (hl_axis_tx_tready    ),
//------------------------------------------------------------------------------
// Peripheral IF
//------------------------------------------------------------------------------
  // SPI Interface, QSPI compatable
  .o_spi_csn                    (spi_csn              ),
  .o_spi_sck                    (spi_sck              ),
  .i_spi_sdio                   (spi_sdio_i           ),
  .o_spi_sdio                   (spi_sdio_o           ),
  .o_spi_oen                    (spi_oen              ),
  // I2C Interface
  .i_i2c_scl                    (i2c_scl              ),
  .i_i2c_sda                    (i2c_sda              ),
  .o_i2c_scl_en                 (i2c_scl_en           ),
  .o_i2c_sda_en                 (i2c_sda_en           ),
  // GPIO
  .o_gpio                       (hl_gpio              ),
  .i_gpio                       ('b0                  ),
//------------------------------------------------------------------------------
// sensor reset
//------------------------------------------------------------------------------
  .o_sw_sys_rst                 (hl_sw_rst),
  .o_sw_sen_rst                 (),
//------------------------------------------------------------------------------
// PTP
//------------------------------------------------------------------------------
  .i_ptp_clk                    ( ptp_clk           ),
  .o_ptp_rst                    ( ptp_rst           ),
  .o_ptp_sec                    ( ptp_sec           ),
  .o_ptp_nanosec                ( ptp_nano          ),
  .o_pps                        ( sys_pps           )
);


//------------------------------------------------------------------------------
// I2C Interface
//------------------------------------------------------------------------------

  // Glitch filtering
  glitch_filter  #(
    .DATA_WIDTH   ( 4                                                   ),
    .RESET_VALUE  ( 1'b1                                  ),
    .FILTER_DEPTH ( 8                                     )
  ) i2c_glitch_filter (
    .clk          ( apb_clk                               ),
    .rst_n        (~apb_rst                               ),
    .sync_in      ({FPGA_EEPROM_SDA, FPGA_EEPROM_SCL,             
                    AD9545_SDA,      AD9545_SCK                         }),
    .sync_out     ({fpga_eeprom_i2c_sda_sync, fpga_eeprom_i2c_scl_sync,
                    ad9545_i2c_sda_sync,      ad9545_i2c_scl_sync       })
  );

  assign i2c_scl[0]   = i2c_scl_en[0] ? fpga_eeprom_i2c_scl_sync : 1'b0;
  assign i2c_sda[0]   = i2c_sda_en[0] ? fpga_eeprom_i2c_sda_sync : 1'b0;
  assign i2c_scl[1]   = i2c_scl_en[1] ? ad9545_i2c_scl_sync  : 1'b0;
  assign i2c_sda[1]   = i2c_sda_en[1] ? ad9545_i2c_sda_sync  : 1'b0;
  
  assign FPGA_EEPROM_SCL = i2c_scl_en[0] ? 1'bz : 1'b0;
  assign FPGA_EEPROM_SDA = i2c_sda_en[0] ? 1'bz : 1'b0;
  assign AD9545_SCK       = i2c_scl_en[1] ? 1'bz : 1'b0;
  assign AD9545_SDA       = i2c_sda_en[1] ? 1'bz : 1'b0;
  

//------------------------------------------------------------------------------
// SPI Interface
//------------------------------------------------------------------------------

  data_sync    #(
    .DATA_WIDTH ( 1                                      )
  ) adi_spi_glitch_filter (
    .clk        ( apb_clk                                ),
    .rst_n      ( ~apb_rst                               ),
    .sync_in    (FMC_LA05_N ),
    .sync_out   (adi_spi_miso_sync)
  );
  
  data_sync    #(
    .DATA_WIDTH ( 1                                      )
  ) hmc_spi_glitch_filter (
    .clk        ( apb_clk                                ),
    .rst_n      ( ~apb_rst                               ),
    .sync_in    (FMC_LA12_N ),
    .sync_out   (hmc_spi_sdio_sync)
  );

  // ADI ADC/DAC SPI
  assign adi_miso       = adi_spi_miso_sync;
  assign adi_mosi       = spi_sdio_o[0][0];  
  assign spi_sdio_i[0]  = {1'b0, 1'b0, adi_miso, 1'b0};
  
  // HMC SPI
  assign FMC_LA12_N     = hmc_sdio;
  assign hmc_sdio       = spi_oen   [1] ? spi_sdio_o[1][0] : 1'bz;
  assign spi_sdio_i[1]  = {1'b0, 1'b0, hmc_spi_sdio_sync, hmc_spi_sdio_sync};
  
  always_ff @(posedge apb_clk) begin
    FMC_LA04_N      <= spi_sck   [0];
    FMC_LA05_P      <= spi_csn   [0];
    FMC_LA04_P      <= adi_mosi;
    FMC_LA12_P      <= spi_csn   [1];
    FMC_LA11_P      <= spi_sck   [1];
  end


//------------------------------------------------------------------------------
// Intel Mailbox IP for QSPI configuration/remote system update
//------------------------------------------------------------------------------
mailbox_top u_mailbox_top (
  .apb_clk                        (apb_clk),
  .apb_rst                        (apb_rst),
  .i_mailbox_apb_psel             (apb_psel[2]),   
  .i_mailbox_apb_penable          (apb_penable),
  .i_mailbox_apb_paddr            (apb_paddr),  
  .i_mailbox_apb_pwdata           (apb_pwdata), 
  .i_mailbox_apb_pwrite           (apb_pwrite), 
  .o_mailbox_apb_pready           (apb_pready[2]),           
  .o_mailbox_apb_prdata           (apb_prdata[2]),           
  .o_mailbox_apb_pserr            (apb_pserr[2])
);


//------------------------------------------------------------------------------
// GPIO Signals needed for AD9986 board
//------------------------------------------------------------------------------
logic           [3:0]           hl_gpio_sync_in;
logic           [3:0]           hl_gpio_sync_out;

assign hl_gpio_sync_in      = {hl_gpio[3], hl_gpio[2], hl_gpio[1], hl_gpio[0]};

data_sync #(
  .DATA_WIDTH             (4) 
) hl_gpio_sync (
  .clk                    (apb_clk),
  .rst_n                  (!rst_sys),
  .sync_in                (hl_gpio_sync_in),
  .sync_out               (hl_gpio_sync_out)
);

always_ff @(posedge apb_clk) begin
  FMC_PWR_EN            <= hl_gpio_sync_out[0];
  FPGA_PG_C2M           <= hl_gpio_sync_out[1];
  FMC_LA07_P            <= hl_gpio_sync_out[2];
  FMC_LA22_P            <= hl_gpio_sync_out[3];
end


//------------------------------------------------------------------------------
// Miscellaneous
//------------------------------------------------------------------------------
always_ff @(posedge apb_clk) begin
  FPGA_RECONFIG                   <= sw_fpga_reconfig;
end

always_ff @(posedge ptp_clk) begin
  DBG_HEADER                      <= {15'h7FFF, sys_pps_stretch};
end

//------------------------------------------------------------------------------
// Misc. Logic
//------------------------------------------------------------------------------
always_ff @(posedge apb_clk) begin
  AD9545_RESETB                   <= !sw_rst_ad9545;
end


`ifdef JESD_USE_FMC_CLK
  assign clk_checker_source       = FMC_CLK1_M2C_TERM_P;
`else
  assign clk_checker_source       = AD9545_CORECLK_P;
`endif

clock_checker #(
) u_clk_chk (
  .test_clk                       ( clk_checker_source                                    ),                             
  .rst                            ( apb_rst                                               ),                                  
  .en                             ( clk_chk_en                                            ),
  .test_clk_target                ( clk_target                                            ),
  .test_clk_tol                   ( clk_tolerance                                         ),
  .latch_count                    ( heartbeat                                             ),
  .clk_in_tolerance               ( clk_in_tol                                            ),
  .last_clk_count                 ( clk_count                                             )
);

//Tie off unused APB bus signals. 
assign apb_pserr[7:5]         = 3'b000;
assign apb_pready[7:5]        = 3'b111;
assign apb_prdata[5]          = '{default:0};
assign apb_prdata[6]          = '{default:0};
assign apb_prdata[7]          = '{default:0};

//------------------------------------------------------------------------------
// PPS
//------------------------------------------------------------------------------
  
`ifdef SIMULATION
  assign timer_done = timer_cnt[3];
`else
  assign timer_done = timer_cnt[17];
`endif

  always_ff @ (posedge ptp_clk) begin
    if (ptp_rst) begin 
      sys_pps_stretch <= 1'b0;
      timer_cnt   <= '0;
    end else begin
      if (sys_pps) begin
        sys_pps_stretch <= 1'b1;
      end else if (timer_done) begin
        timer_cnt   <= '0;
        sys_pps_stretch <= 1'b0;
      end else if (sys_pps_stretch) begin
        timer_cnt   <= timer_cnt + 1'b1;
      end
    end
  end

endmodule
