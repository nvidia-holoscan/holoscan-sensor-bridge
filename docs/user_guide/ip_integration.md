# IP Integration

## Top Level Module

The top level module of the Holoscan Sensor Bridge IP is "HOLOLINK_top".

## User Configurability

The Holoscan Sensor Bridge (HSB) IP is designed to be easily configurable to various use
case of the IP, such as number of sensors and Ethernet ports. It’s also designed to be
compatible with multiple FPGA vendors. The following describes the configurations
available to user.

### Macro Definitions

The Holoscan Sensor Bridge definitions file, “HOLOLINK_def.svh”, defines the following
macros.

The macros can be configured to user’s application of the Holoscan Sensor Bridge IP. The
default macro value is the configuration that has been tested and verified.

Table 1

| \*\*Macro                                     | **Tested Values**                                                                                                                                                                                                                                                    | **Description**                                                                                                                           |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| HIF_CLK_FREQ                                  | 156250000 (for DATAPATH_WIDTH=64)<br />201416016 (for DATAPATH_WIDTH=512)                                                                                                                                                                                            | Clock frequency of the Host Interface. Unit is in Hz                                                                                      |
| APB_CLK_FREQ                                  | 19531250 (for DATAPATH_WIDTH=64)<br />100000000 (for DATAPATH_WIDTH=512)                                                                                                                                                                                             | Clock frequency of the APB Interface. Unit is in Hz                                                                                       |
| PTP_CLK_FREQ                                  | 100446545                                                                                                                                                                                                                                                            | Clock frequency of the PTP Interface. Unit is in Hz. Range 95MHz to 105MHz                                                                |
| UUID[127:0]                                   | <ul><li>128'h889B7CE3_65A5_4247_8B05_4FF1904C3359 for Lattice LF-SNSR-ETH-EV</li><li>128'hED6A9292_DEBF_40AC_B603_A24E025309C1 for Microchip MPF200-ETH-SENSOR-BRIDGE</li><li>128'hF1627640_B4DC_48AF_A360_C65B09B3D230 for Leopard Imaging VB1940 Eagle Camera</li> | Universally Unique Identifier are used to identify the board to fetch and remote flash the correct bitfile.                               |
| ENUM_EEPROM                                   | Defined or undefined                                                                                                                                                                                                                                                 | When defined, read the contents of the Enumeration packet from external non-volatile memory. If undefined, use the macros defined below.. |
| EEPROM_REG_ADDR_BITS                          | 8 or 16                                                                                                                                                                                                                                                              | Valid when ENUM_EEPROM is defined. Number for register address bits used in the EEPROM.                                                   |
| DATAPATH_WIDTH                                | 8, 16, 32, 64, 128, 256, 512, 1024                                                                                                                                                                                                                                   | Width of the Sensor AXI Stream TDATA in bits. This number must be byte-aligned. Meaning, it must be a number divisible by 8.              |
| DATAKEEP_WIDTH                                | DATAPATH_WIDTH/8                                                                                                                                                                                                                                                     | Width of the Sensor AXI Stream TKEEP.This should not be changed.                                                                          |
| DATAUSER_WIDTH                                | 1-2                                                                                                                                                                                                                                                                  | Width of the Sensor AXI Stream TUSER signal.                                                                                              |
| SENSOR_RX_IF_INST                             | undefined, 1-32                                                                                                                                                                                                                                                      | Number of Sensor RX Interface.                                                                                                            |
| SIF_RX_WIDTH[SENSOR_RX_IF_INST-1:0]           | 8, 16, 32, 64, 128, 256, 512, 1024                                                                                                                                                                                                                                   | Valid when SENSOR_RX_IF_INST is defined. Each Sensor RX interface width can be individually defined. Max value should be DATAPATH_WIDTH.  |
| SIF_RX_DATA_GEN                               | undefined, defined                                                                                                                                                                                                                                                   | Valid when SENSOR_RX_IF_INST is defined. Instantiates a Data Generator module to test Sensor to Host data path.                           |
| SIF_RX_PACKETIZER_EN[SENSOR_RX_IF_INST-1:0]   | 0, 1                                                                                                                                                                                                                                                                 | Valid when SENSOR_RX_IF_INST is defined. Each Sensor RX interface Packetizer can be individually enabled.                                 |
| SIF_RX_VP_COUNT[SENSOR_RX_IF_INST-1:0]        | DO NOT TOUCH                                                                                                                                                                                                                                                         | Valid when SENSOR_RX_IF_INST and SIF_RX_PACKETIZER_EN is defined. TBD. Do not change.                                                     |
| SIF_RX_SORT_RESOLUTION[SENSOR_RX_IF_INST-1:0] | DO NOT TOUCH                                                                                                                                                                                                                                                         | Valid when SENSOR_RX_IF_INST and SIF_RX_PACKETIZER_EN is defined. TBD. Do not change.                                                     |
| SIF_RX_VP_SIZE[SENSOR_RX_IF_INST-1:0]         | DO NOT TOUCH                                                                                                                                                                                                                                                         | Valid when SENSOR_RX_IF_INST and SIF_RX_PACKETIZER_EN is defined. TBD. Do not change.                                                     |
| SIF_RX_NUM_CYCLES[SENSOR_RX_IF_INST-1:0]      | DO NOT TOUCH                                                                                                                                                                                                                                                         | Valid when SENSOR_RX_IF_INST and SIF_RX_PACKETIZER_EN is defined. TBD. Do not change.                                                     |
| SENSOR_TX_IF_INST                             | undefined, 1-32                                                                                                                                                                                                                                                      | Number of Sensor TX Interface.                                                                                                            |
| SIF_TX_WIDTH[SENSOR_TX_IF_INST-1:0]           | 8, 64, 512                                                                                                                                                                                                                                                           | Valid when SENSOR_TX_IF_INST is defined. Each Sensor TX interface width can be individually defined. Max value should be DATAPATH_WIDTH.  |
| SIF_TX_BUF_SIZE[SENSOR_TX_IF_INST-1:0]        | 1024, 2048, 4096                                                                                                                                                                                                                                                     | Valid when SENSOR_TX_IF_INST is defined. Set the Sensor TX Buffer Size.                                                                   |
| HOST_WIDTH                                    | 8, 16, 32, 64, 128, 256, 512, 1024                                                                                                                                                                                                                                   | Width of the Host AXI Stream TDATA in bits. This number must be byte-aligned. Meaning, it must be a number divisible by 8.                |
| HOSTKEEP_WIDTH                                | HOST_WIDTH/8                                                                                                                                                                                                                                                         | Width of the Host AXI Stream TKEEP.This should not be changed.                                                                            |
| HOSTUSER_WIDTH                                | 1                                                                                                                                                                                                                                                                    | Width of the Host AXI Stream TUSER signal.                                                                                                |
| HOST_IF_INST                                  | 1-32                                                                                                                                                                                                                                                                 | Number of Host interfaces.                                                                                                                |
| HOST_MTU                                      | 1500, 4096                                                                                                                                                                                                                                                           | Size of Ethernet packet in bytes.                                                                                                         |
| SPI_INST                                      | undefined, 1-8                                                                                                                                                                                                                                                       | Number of SPI interfaces.                                                                                                                 |
| I2C_INST                                      | undefined, 1-8                                                                                                                                                                                                                                                       | Number of I2C interfaces.                                                                                                                 |
| GPIO_INST                                     | 0-255                                                                                                                                                                                                                                                                | Number of GPIO Input & Output bits.                                                                                                       |
| GPIO_RESET_VALUE[GPIO_INST-1:0]               | 0                                                                                                                                                                                                                                                                    | Reset value of GPIO bits.                                                                                                                 |
| REG_INST                                      | 1-8                                                                                                                                                                                                                                                                  | Number of user register.                                                                                                                  |
| N_INIT_REG                                    | Integer value                                                                                                                                                                                                                                                        | Number of initialization registers.                                                                                                       |

### Build Revision

Parameter `BUILD_REV[47:0]` can be passed down to the "HOLOLINK_top" module to uniquely
identify the FPGA build revision.

### Sensor Interface Configuration

In applications where the HSB IP connects multiple sensors with varying bandwidths, the
IP can be configured on a per-sensor-interface basis to optimize resource utilization.

The `DATAPATH_WIDTH` parameter defines the AXI Stream vector width for all sensor
interfaces to simplify integration and should be set to match the widest sensor. The
`SIF_RX_WIDTH` and `SIF_TX_WIDTH` parameters can be configured individually to specify
narrower data widths for each sensor interface.

For example, if the first sensor interface has a width of 64 bits and the second has 32
bits: Set `DATAPATH_WIDTH` to 64 and set `SIF_RX_WIDTH` to {32, 64}. For second sensor
interface, only the [31:0] of the AXI Stream TDATA and [3:0] of the AXI Stream TKEEP are
used and the remaining signals should be tied to 0. The LSB [3:0] of AXI Stream TKEEP is
still required to be set to all 1'b1.

Furthermore, the Packetizer function, which is needed when streaming camera to Thor, can
be enabled for each sensor interface. For example, if the first sensor needs the
Packetizer function but the second sensor does not, set `SIF_RX_PACKETIZER_EN` parameter
to {0,1}.

HSB IP supports Virtual Port mapping. For more information, refer to
[Sensor Interface to Virtual Port Mapping](dataplane.md#sensor-interface-to-virtual-port-mapping)

### BOOTP Packet

The HSB IP transmits a broadcast BOOTP Request packet on each Host interface
approximately once per second. BOOTP packets are used to enumerate the HSB to host and
to change the IP address from default in case of multiple board enumeration. Default IP
address of HSB IP is "192.168.0.2" for Host interface 0 and "192.168.0.3" for Host
interface 1 and so on. HSB IP BOOTP packets adhere to RFC-951. The Vendor Field of the
BOOTP packet is used to communicate enumeration info and status of Holoscan Sensor Board
to the host.

Holoscan Sensor Bridge IP BOOTP Vendor field is laid out in below table:

Table 2

| **Byte Number** | **Description**                   | **Value**                                                                     |
| --------------- | --------------------------------- | ----------------------------------------------------------------------------- |
| [0]             | Vendor Tag                        | 0xE0                                                                          |
| [1]             | Tag Length                        | 0x2B                                                                          |
| [5:2]           | ASCII "NVDA"                      | 0x4144564E, in Big Endian format                                              |
| [6]             | Ethernet Port Number              | 0x0 for Port 0 , 0x1 for Port 1                                               |
| [7]             | Enumeration Version               | 0x2                                                                           |
| [9:8]           | Reserved                          | Reserved. Set to 0.                                                           |
| [25:10]         | UUID                              | UUID defined in HOLOLINK_def.svh. Big Endian format                           |
| [29:26]         | Reserved                          | Reserved. Set to 0.                                                           |
| [36:30]         | Board Serial Number               | BOARD_SN fetched from EEPROM or from HSB IP input port. Little Endian format. |
| [38:37]         | Holoscan Sensor Bridge IP Version | *0x2510* for *Holoscan SDK v2.5.0* release. Little Endian format.             |
| [44:39]         | Reserved                          | Reserved. Set to 0.                                                           |
| [54:45]         | PTP Timestamp                     | PTP Timestamp. Seconds and Nanoseconds in PTP v2 format. Big Endian format.   |
| [56:55]         | Packet Sequence Number            | Packet Sequence Number. Increments per BOOTP packet sent. Big Endian format.  |
| [57]            | Status                            | 0x1 when PTP is enabled in HSB and a SYNC PTP packet has been received.       |
| [63:58]         | Reserved                          | Reserved. Set to 0.                                                           |

Each of the macros used for enumeration packet is further explained below.

UUID: This macro defines the unique identifier for the board. Holoscan uses this value
to identify which board it is interfacing and fetch the correct bitfile for remote
flash. For newly developed boards, users can generate and assign a new UUID. This macro
must be defined and is not optionally stored in the non-volatile memory.

ENUM_EEPROM: When this macro is defined, enumeration fields, the MAC address and Board
Serial Number, are retrieved from an on-board, non-volatile memory upon boot-up. The
enumeration fields are retrieved over I2C port 0 with EEPROM 7-bit address 0x50 and must
be stored in specific address. Further details can be provided upon request.

Mass production of boards with Holoscan Sensor Bridge IP requires an external EEPROM to
store unique MAC address and board serial numbers. When using evaluation platforms for
test and bring-up, enumeration fields, MAC address and Board Serial Number, can be
passed to HSB IP via input port.

## System Initialization

The Holoscan Sensor Bridge IP provides system initialization function to write to
registers upon power up. This function is used to initialize the Ethernet block to
establish ethernet connection between the FPGA and the host, and can be used for any
other user function in the top level design.

The list of registers to be initialized is defined in “HOLOLINK_def.svh” as “init_reg”
array. The “init_reg” is an unpacked array, sized [N_INIT_REG] [63:0], where the
N_INIT_REG macro defines the number of registers to be initialized and the 64-bit vector
is used to define the 32- bit address of register at [63:32] and the 32-bit write data
at [31:0].

To give an example of one of the init_reg array entry:

{32'h1000_0020, 32'h0000_00FF} //init_reg[0]

Will write to the User REG_INST_0 block address offset 0x0000_0020 the data 0x0000_00FF.

Once system initialization is complete, the “o_init_done” port will be asserted high in
“i_apb_clk” domain. User can use this signal to gate their logic if it’s dependent on
ethernet block initialization.
