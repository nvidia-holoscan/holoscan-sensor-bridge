`ifndef HOLOLINK_def
`define HOLOLINK_def

package HOLOLINK_pkg;

//-----------------------------------------------------
// Define FPGA Vendor for Macro Instantiation
//-----------------------------------------------------

// XILINX / ALTERA / LATTICE / MICROCHIP
  `define LATTICE

//-----------------------------------------------------
// Holoscan IP Host Clock Frequency
//
// Used for internal timer calculation
//-----------------------------------------------------

  `define HIF_CLK_FREQ  156250000

//-----------------------------------------------------
// Holoscan IP APB Clock Frequency
//
// Used for I2C clock divider setting
//-----------------------------------------------------

   `define APB_CLK_FREQ  19531250

//-----------------------------------------------------
// Holoscan IP PTP Clock Frequency
//
// Used for internal timer calculation
//-----------------------------------------------------

  `define PTP_CLK_FREQ  100446545

//-----------------------------------------------------
// MAC Address
//
// If MAC address is not defined in an external memory,
// i.e EEPROM, then the soft MAC address can be used
//-----------------------------------------------------

//-----------------------------------------------------
// Board Info Enumeration
//
// Use ENUM_EEPROM if board info is stored in an external
// EEPROM. Otherwise,
//-----------------------------------------------------

  `define ENUM_EEPROM
  `define EEPROM_REG_ADDR_BITS 8
  `define UUID                 128'h889B_7CE3_65A5_4247_8B05_4FF1_904C_3359

`ifndef ENUM_EEPROM
  `define MAC_ADDR  48'hCAFEC0FFEE00
  `define BOARD_VER 160'h0
  `define BOARD_SN  56'h0
  `define FPGA_CRC  16'h0
  `define MISC      32'h0
`endif
//-----------------------------------------------------
// Sensor IF
//-----------------------------------------------------

`ifndef DATAPATH_WIDTH
  `define DATAPATH_WIDTH  64                 // Sensor interface data width
`endif
  `define DATAKEEP_WIDTH  `DATAPATH_WIDTH/8  // Sensor interface data keep width
  `define DATAUSER_WIDTH  2                  // Sensor interface data user width
`ifndef SENSOR_IF_INST
  `define SENSOR_IF_INST  2                  // Sensor interface instantiation number
`endif

  `define SENSOR_TX_ENABLE  1                // Sensor interface TX Path Enable

//-----------------------------------------------------
// Host IF
//-----------------------------------------------------

`ifndef HOST_WIDTH
  `define HOST_WIDTH      64                 // Host interface data width
`endif
  `define HOSTKEEP_WIDTH  `HOST_WIDTH/8      // Host interface data keep width
  `define HOSTUSER_WIDTH  1                  // Host interface data user width
`ifndef HOST_IF_INST
  `define HOST_IF_INST    2                  // Host interface instantiation number
`endif

  `define HOST_MTU       4096

//------------------------------------------------------------------------------
// Peripheral Control
//------------------------------------------------------------------------------

`ifndef SPI_INST
  `define SPI_INST  2   // SPI interface instantiation number
`endif
`ifndef  I2C_INST
  `define I2C_INST  4   // I2C interface instantiation number
`endif
  `define GPIO_INST 31  // INOUT GPIO instantiation number

  localparam [`GPIO_INST-1:0] GPIO_RESET_VALUE ='0;

//------------------------------------------------------------------------------
// Register IF
//
// Creates <REG_INST> number of APB register interfaces for user logic access
//------------------------------------------------------------------------------

  `define REG_INST 8

//------------------------------------------------------------------------------
// Packetizer
//
// Sets the Packetizer Parameters for Each Sensor IF
//------------------------------------------------------------------------------
  `define RX_PACKETIZER_EN 1

  localparam integer  SIF_SORT_RESOLUTION [`SENSOR_IF_INST-1:0] = {2   , 2   };
  localparam integer  SIF_VP_COUNT        [`SENSOR_IF_INST-1:0] = {2   , 2   };
  localparam integer  SIF_VP_SIZE         [`SENSOR_IF_INST-1:0] = {64  , 64  };
  localparam integer  SIF_NUM_CYCLES      [`SENSOR_IF_INST-1:0] = {3   , 3   };
  localparam integer  SIF_DYN_VP          [`SENSOR_IF_INST-1:0] = {0   , 0   };
  localparam integer  SIF_MIXED_VP_SIZE   [`SENSOR_IF_INST-1:0] = {0   , 0   };
//------------------------------------------------------------------------------
// TX Stream Buffer
//
// Sets the Transmit Stream Buffer Size (number of entries) for Each Sensor IF
//------------------------------------------------------------------------------
  localparam integer SIF_TX_BUF_SIZE      [`SENSOR_IF_INST-1:0] = {4096 , 4096};

//------------------------------------------------------------------------------
// System Initialization
//
// Initialization for the Host Interface registers so communication can be
// established between the FPGA and the Host
//------------------------------------------------------------------------------

  `define N_INIT_REG 38


  localparam logic [63:0] init_reg [`N_INIT_REG] = '{
    // 32b Addr   | 32b Data
    {32'h1000_7A74, 32'h0000_0020}, // Lattice pcs 0, pcs_lpbk_ctrl
    {32'h1000_7AD9, 32'h0000_0079}, // Lattice pcs 0, pcs_eqlz_en
    {32'h1000_7AD1, 32'h0000_0065}, // Lattice pcs 0, pcs_preq_gain
    {32'h1000_7AD3, 32'h0000_0061}, // Lattice pcs 0, pcs_poeq_gain
    {32'h1000_7AD5, 32'h0000_0065}, // Lattice pcs 0, pcs_iter_cnt
    {32'h1000_7A80, 32'h0000_0001}, // Lattice pcs 0, pcs_reg_update
    {32'h2000_0000, 32'h0000_0003}, // Lattice mac 0, mac_mode
    {32'h2000_0004, 32'h0000_0000}, // Lattice mac 0, mac_tx_ctl
    {32'h2000_0008, 32'h0000_0061}, // Lattice mac 0, mac_rx_ctl
    {32'h2000_000C, 32'h0000_05E0}, // Lattice mac 0, mac_pkg_len
    {32'h2000_0010, 32'h0000_0010}, // Lattice mac 0, mac_ipg_val
    {32'h3000_7A74, 32'h0000_0020}, // Lattice pcs 1, pcs_lpbk_ctrl
    {32'h3000_7AD9, 32'h0000_0079}, // Lattice pcs 1, pcs_eqlz_en
    {32'h3000_7AD1, 32'h0000_0065}, // Lattice pcs 1, pcs_preq_gain
    {32'h3000_7AD3, 32'h0000_0061}, // Lattice pcs 1, pcs_poeq_gain
    {32'h3000_08D5, 32'h0000_0065}, // Lattice pcs 1, pcs_iter_cnt
    {32'h3000_0880, 32'h0000_0001}, // Lattice pcs 1, pcs_reg_update
    {32'h4000_0000, 32'h0000_0003}, // Lattice mac 1, mac_mode
    {32'h4000_0004, 32'h0000_0000}, // Lattice mac 1, mac_tx_ctl
    {32'h4000_0008, 32'h0000_0061}, // Lattice mac 1, mac_rx_ctl
    {32'h4000_000C, 32'h0000_05E0}, // Lattice mac 1, mac_pkg_len
    {32'h4000_0010, 32'h0000_0010}, // Lattice mac 1, mac_ipg_val
    // Hololink Internal Reg Initialization // TODO To be removed (not required, can be done by sw)
    {32'h0300_0210, 32'h004C_4B40}, // i2c timeout
    {32'h0200_0020, 32'h0000_2000}, // inst_dec_0, ecb_udp_port
    {32'h0201_0020, 32'h0000_2000}, // inst_dec_1, ecb_udp_port
    {32'h0200_0304, 32'h0000_05CE}, // dp_pkt_0  , dp_pkt_len
    {32'h0200_0308, 32'h0000_3000}, // dp_pkt_0  , dp_pkt_fpga_udp_port
    {32'h0201_0304, 32'h0000_05CE}, // dp_pkt_1  , dp_pkt_len
    {32'h0201_0308, 32'h0000_3000}, // dp_pkt_1  , dp_pkt_fpga_udp_port
    {32'h0200_0108, 32'h0000_0064}, // eth_pkt_0 , Eth pkt data plane priority
    {32'h0201_0108, 32'h0000_0064}, // eth_pkt_1 , Eth pkt data plane priority
    {32'h0200_0024, 32'h0000_12b7}, // inst_dec_0, stx_udp_port
    {32'h0201_0024, 32'h0000_12b7}, // inst_dec_1, stx_udp_port
    //PTP Config
    {32'h0000_0110, 32'h0000_0002}, // ptp, dpll cfg1
    {32'h0000_0114, 32'h0000_0002}, // ptp, dpll cfg2
    {32'h0000_0118, 32'h0000_0003}, // ptp, delay avg factor
    {32'h0000_010C, 32'h0000_0038}, // ptp, delay asymmetry 
    {32'h0000_0104, 32'h0000_0003}  // ptp, dpll en
  };

endpackage: HOLOLINK_pkg
`endif





