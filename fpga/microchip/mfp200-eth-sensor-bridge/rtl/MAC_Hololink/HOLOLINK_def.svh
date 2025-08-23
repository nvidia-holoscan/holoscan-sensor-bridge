`ifndef HOLOLINK_def
`define HOLOLINK_def

package HOLOLINK_pkg;

//-----------------------------------------------------
// Define FPGA Vendor for Macro Instantiation
//-----------------------------------------------------

// XILINX / PSG / LATTICE / MICROCHIP
  `define MICROCHIP

//-----------------------------------------------------
// Holoscan IP Host Clock Frequency
//
// Used for internal timer calculation
//-----------------------------------------------------

  `define HIF_CLK_FREQ  156250000
  `define PTP_CLK_FREQ  100000000

//-----------------------------------------------------
// Holoscan IP APB Clock Frequency
//
// Used for I2C clock divider setting
//-----------------------------------------------------

  `define APB_CLK_FREQ  19531000

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
  `define UUID                 128'hED6A_9292_DEBF_40AC_B603_A24E_0253_09C1

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

//`define SENSOR_TX_ENABLE  1                  // Sensor interface TX Path Enable

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

  `define HOST_MTU       1500

//------------------------------------------------------------------------------
// Peripheral Control
//------------------------------------------------------------------------------

`ifndef SPI_INST
  `define SPI_INST  1   // SPI interface instantiation number
`endif
`ifndef  I2C_INST
  `define I2C_INST  3   // I2C interface instantiation number
`endif
  `define GPIO_INST 16  // INOUT GPIO instantiation number

  localparam [`GPIO_INST-1:0] GPIO_RESET_VALUE ='0;

//------------------------------------------------------------------------------
// Register IF
//
// Creates <REG_INST> number of APB register interfaces for user logic access
//------------------------------------------------------------------------------

  `define REG_INST 1

//------------------------------------------------------------------------------
// Packetizer
//
// Sets the Packetizer Parameters for Each Sensor IF
//------------------------------------------------------------------------------

  localparam integer  SIF_SORT_RESOLUTION [`SENSOR_IF_INST-1:0] = {8   , 8   };
  localparam integer  SIF_VP_COUNT        [`SENSOR_IF_INST-1:0] = {0   , 0   };
  localparam integer  SIF_VP_SIZE         [`SENSOR_IF_INST-1:0] = {32  , 32  };
  localparam integer  SIF_NUM_CYCLES      [`SENSOR_IF_INST-1:0] = {1   , 1   };
  localparam integer  SIF_DYN_VP          [`SENSOR_IF_INST-1:0] = {0   , 0   };
  localparam integer  SIF_MIXED_VP_SIZE   [`SENSOR_IF_INST-1:0] = {0   , 0   };

//------------------------------------------------------------------------------
// System Initialization
//
// Initialization for the Host Interface registers so communication can be
// established between the FPGA and the Host
//------------------------------------------------------------------------------

  `define N_INIT_REG 9

  localparam logic [63:0] init_reg [`N_INIT_REG] = '{
    // 32b Addr   | 32b Data
    //-----------------------------------------------
    // Add register writes to Ethernet MAC/PCS here
    //-----------------------------------------------
    // Hololink Internal Reg Initialization // TODO To be removed (not required, can be done by sw)
    {32'h0300_0210, 32'h004C_4B40}, // i2c timeout
    {32'h0200_0020, 32'h0000_2000}, // inst_dec_0, ecb_udp_port
    {32'h0200_0304, 32'h0000_05CE}, // dp_pkt_0  , dp_pkt_len
    {32'h0200_0308, 32'h0000_3000}, // dp_pkt_0  , dp_pkt_fpga_udp_port
    {32'h0200_0108, 32'h0000_0064}, // eth_pkt_0 , Eth pkt data plane priority

    {32'h0201_0020, 32'h0000_2000}, // inst_dec_0, ecb_udp_port
    {32'h0201_0304, 32'h0000_05CE}, // dp_pkt_0  , dp_pkt_len
    {32'h0201_0308, 32'h0000_3000}, // dp_pkt_0  , dp_pkt_fpga_udp_port
    {32'h0201_0108, 32'h0000_0064}  // eth_pkt_0 , Eth pkt data plane priority
  };

endpackage: HOLOLINK_pkg
`endif





