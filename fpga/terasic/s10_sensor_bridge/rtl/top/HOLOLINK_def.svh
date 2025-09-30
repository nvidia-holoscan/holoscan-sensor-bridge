`ifndef HOLOLINK_def
`define HOLOLINK_def

package HOLOLINK_pkg;

//-----------------------------------------------------
// Define FPGA Vendor for Macro Instantiation
//-----------------------------------------------------

// XILINX / ALTERA / LATTICE / MICROSEMI
  `define ALTERA

//-----------------------------------------------------
// Holoscan IP Host Clock Frequency
//
// Used for internal timer calculation
//-----------------------------------------------------

  `define HIF_CLK_FREQ  201416016

//-----------------------------------------------------
// Holoscan IP APB Clock Frequency
//
// Used for I2C clock divider setting
//-----------------------------------------------------

  `define APB_CLK_FREQ  100000000

//-----------------------------------------------------
// Holoscan IP PTP Clock Frequency
//
// Used for internal timer calculation
//-----------------------------------------------------

  `define PTP_CLK_FREQ  100707500

//-----------------------------------------------------
// Board Info Enumeration
//-----------------------------------------------------
  //UUID is used to uniquely identify the board. The UUID is sent over BOOTP.
  `define UUID                 128'h7A37_7BF7_76CB_4756_A4C5_7DDD_AED8_354B

  // Define ENUM_EEPROM if board info is stored in an external EEPROM.
  // Otherwise, soft MAC address and Board Serial Number can be used
  `define ENUM_EEPROM

  `ifdef ENUM_EEPROM
    `define EEPROM_REG_ADDR_BITS 8                //EEPROM Register Address Bits. Valid values: 8, 16
  `else
    `define MAC_ADDR             48'hCAFEC0FFEE00 //Soft MAC Address. Can be passed to Hololink IP I/O
    `define BOARD_SN             56'h0            //Soft Board Serial Number. Can be passed to Hololink IP I/O
    `define BOARD_VER            160'h0
    `define FPGA_CRC             16'h0
    `define MISC                 32'h0
  `endif
//-----------------------------------------------------
// Sensor Interface
//-----------------------------------------------------

  `ifdef JESD_HIGH_PERF
    `define DATAPATH_WIDTH 1024 
  `else
    `define DATAPATH_WIDTH  512                // Sensor interface data width. This should be set to MAX width between SIF RX and TX widths
                                               // Valid values: 8, 16, 64, 128, 512, 1024 
  `endif
  `define DATAKEEP_WIDTH  `DATAPATH_WIDTH/8  // Sensor interface data keep width
  `define DATAUSER_WIDTH  1                  // Sensor interface data user width

//-----------------------------------------------------
// Sensor RX IF
//-----------------------------------------------------

  `define SENSOR_RX_IF_INST  1                   // Number of Sensor RX Interface. Valid values: undefined, 1 - 32
  //----------------------------------------------------------------------------------
  //If no Sensor RX Interfaces are used, then comment out "`define SENSOR_RX_IF_INST" 
  //This will remove Sensor RX IF I/Os from HOLOLINK_top module.
  //The same applies for "SENSOR_TX_IF_INST", "SPI_INST", and "I2C_INST" definitions. 
  //----------------------------------------------------------------------------------

  `ifdef SENSOR_RX_IF_INST
    //`define SENSOR_RX_DATA_GEN             // If defined, Sensor RX Data Generator is instantiated. This can be used for bring-up. 

    localparam integer  SIF_RX_WIDTH        [`SENSOR_RX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH};
    //--------------------------------------------------------------------------------
    // Sensor RX Packetizer Parameters
    // If RX_PACKETIZER_EN is set to 0, then Packetizer is disabled for that Sensor RX interface. 
    // Example of how array index matches to Sensor is:
    //                    {Sensor[1], Sensor[0]}
    // RX_PACKETIZER_EN = {        1,         1}
    //--------------------------------------------------------------------------------
    localparam integer  RX_PACKETIZER_EN    [`SENSOR_RX_IF_INST-1:0] = '{default:'1};
    `ifdef JESD_HIGH_PERF
    localparam integer  SIF_VP_COUNT        [`SENSOR_RX_IF_INST-1:0] = {4};
    localparam integer  SIF_SORT_RESOLUTION [`SENSOR_RX_IF_INST-1:0] = {16};
    localparam integer  SIF_VP_SIZE         [`SENSOR_RX_IF_INST-1:0] = {256};
    localparam integer  SIF_NUM_CYCLES      [`SENSOR_RX_IF_INST-1:0] = {1};
    `else
    localparam integer  SIF_VP_COUNT        [`SENSOR_RX_IF_INST-1:0] = {4};
    localparam integer  SIF_SORT_RESOLUTION [`SENSOR_RX_IF_INST-1:0] = {16};
    localparam integer  SIF_VP_SIZE         [`SENSOR_RX_IF_INST-1:0] = {128};
    localparam integer  SIF_NUM_CYCLES      [`SENSOR_RX_IF_INST-1:0] = {1};
    `endif
  `endif
 
//-----------------------------------------------------
// Sensor TX IF
//-----------------------------------------------------
  `define SENSOR_TX_IF_INST  1                  // Number of Sensor TX Interface. Valid values: undefined, 1 - 32

  `ifdef SENSOR_TX_IF_INST
    localparam integer  SIF_TX_WIDTH        [`SENSOR_TX_IF_INST-1:0] = '{default:`DATAPATH_WIDTH}; // Define width for each interface. 
    localparam integer  SIF_TX_BUF_SIZE     [`SENSOR_TX_IF_INST-1:0] = '{default : 2048};          // Define buffer size for each interface. 
  `endif

//-----------------------------------------------------
// Host IF
//-----------------------------------------------------

  `define HOST_WIDTH      512                // Host interface data width.                     Valid values: 8, 64, 128, 512
  `define HOSTKEEP_WIDTH  `HOST_WIDTH/8      // Host interface data keep width
  `define HOSTUSER_WIDTH  1                  // Host interface data user width
  `define HOST_IF_INST    2                  // Host interface instantiation number.           Valid values: 1 - 32
  `define HOST_MTU       4096                // Maximum Transmission Unit for Ethernet packet. Valid values: 1500, 4096
  
//------------------------------------------------------------------------------
// Peripheral Control
//------------------------------------------------------------------------------

  `define SPI_INST  2   // SPI interface instantiation number. Valid values: undefined, 1 - 8
  `define I2C_INST  2   // I2C interface instantiation number. Valid values: undefined, 1 - 8
  `define GPIO_INST 16  // INOUT GPIO instantiation number.    Valid values: 1 - 255

  localparam [`GPIO_INST-1:0] GPIO_RESET_VALUE = 16'b0000000000001111;

//------------------------------------------------------------------------------
// Register IF
//
// Creates <REG_INST> number of APB register interfaces for user logic access
//------------------------------------------------------------------------------

  `define REG_INST 8

//------------------------------------------------------------------------------
// System Initialization
//
// Initialization for the Host Interface registers so communication can be
// established between the FPGA and the Host
//------------------------------------------------------------------------------

  `define N_INIT_REG 20

  localparam logic [63:0] init_reg [`N_INIT_REG] = '{
    // 32b Addr   | 32b Data
    // Hololink Internal Reg Initialization // TODO To be removed (not required, can be done by sw)
    {32'h0200_0020, 32'h0000_2000}, // inst_dec_0, ecb_udp_port
    {32'h0201_0020, 32'h0000_2000}, // inst_dec_1, ecb_udp_port
    {32'h0200_021C, 32'h0000_2329}, // ctrl_evt_0, ctrl_evt_host_udp_port
    {32'h0200_0220, 32'h0000_2329}, // ctrl_evt_0, ctrl_evt_fpga_udp_port
    {32'h0201_021C, 32'h0000_2329}, // ctrl_evt_1, ctrl_evt_host_udp_port
    {32'h0201_0220, 32'h0000_2329}, // ctrl_evt_1, ctrl_evt_fpga_udp_port
    {32'h0200_0304, 32'h0000_05CE}, // dp_pkt_0  , dp_pkt_len
    {32'h0200_0308, 32'h0000_3000}, // dp_pkt_0  , dp_pkt_fpga_udp_port
    {32'h0201_0304, 32'h0000_05CE}, // dp_pkt_1  , dp_pkt_len
    {32'h0201_0308, 32'h0000_3000}, // dp_pkt_1  , dp_pkt_fpga_udp_port
    {32'h0300_0210, 32'h004C_4B40}, // i2c timeout
    {32'h0200_0108, 32'h0000_0064}, // eth_pkt_0 , Eth pkt data plane priority
    {32'h0201_0108, 32'h0000_0064}, // eth_pkt_1 , Eth pkt data plane priority
    {32'h0200_0024, 32'h0000_12B7}, // inst_dec_0, stx_udp_port
    {32'h0201_0024, 32'h0000_12B7}, // inst_dec_1, stx_udp_port
    //PTP Config
    {32'h0000_0110, 32'h0000_0002}, // ptp, dpll cfg1
    {32'h0000_0114, 32'h0000_0002}, // ptp, dpll cfg2
    {32'h0000_0118, 32'h0000_0003}, // ptp, delay avg factor
    {32'h0000_010C, 32'h0000_0000}, // ptp, delay asymmetry 
    {32'h0000_0104, 32'h0000_0003}  // ptp, dpll en
  };


//------------------------------------------------------------------------------
// Customer Specific Defines
//------------------------------------------------------------------------------
  //------------------------------------------------------------------------------
  // JESD IF
  //------------------------------------------------------------------------------
  `define JESD_NUM_LANES    8

  //------------------------------------------------------------------------------
  // COE - Camera Over Ethernet Enable
  // Disables COE packet generation functionality 
  //------------------------------------------------------------------------------
  `define DISABLE_COE 1

endpackage: HOLOLINK_pkg
`endif
