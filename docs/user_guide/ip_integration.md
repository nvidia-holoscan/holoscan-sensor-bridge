# IP Integration

## Top Level Module

The top level module of the Holoscan Sensor Bridge IP is "HOLOLINK_top".

## User Configurability

The Holoscan Sensor Bridge IP is designed to be easily configurable to various use case
of the IP, such as number of sensors and Ethernet ports. It’s also designed to be
compatible with multiple FPGA vendors. The following describes the configurations
available to user.

### Macro Definitions

The Holoscan Sensor Bridge definitions file, “HOLOLINK_def.svh”, defines the following
macros.

The macros can be configured to user’s application of the Holoscan Sensor Bridge IP. The
default macro value is the configuration that has been tested and verified.

Table 13

| \*\*Macro                       | **Tested Values**                                                                                                                                                | **Description**                                                                                                                           |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| FPGA_VENDOR                     | LATTICE, ALTERA, MICROCHIP                                                                                                                                       | Defines the FPGA Vendor used.                                                                                                             |
| HIF_CLK_FREQ                    | 156250000 (for DATAPATH_WIDTH=64)<br />201416016 (for DATAPATH_WIDTH=512)                                                                                        | Clock frequency of the Host Interface. Unit is in Hz                                                                                      |
| APB_CLK_FREQ                    | 19531250 (for DATAPATH_WIDTH=64)<br />100000000 (for DATAPATH_WIDTH=512)                                                                                         | Clock frequency of the APB Interface. Unit is in Hz                                                                                       |
| PTP_CLK_FREQ                    | 100446545                                                                                                                                                        | Clock frequency of the PTP Interface. Unit is in Hz. Range 90MHz to 110MHz                                                                |
| UUID[127:0]                     | 128'h889B7CE3_65A5_4247_8B05_4FF1904C3359 for Lattice LF-SNSR-ETH-EV<br />128'hED6A9292_DEBF_40AC_B603_A24E025309C1 for Microchip MPF200-ETH-SENSOR-BRIDGE<br /> | Universally Unique Identifier are used to identify the board to fetch and remote flash the correct bitfile.                               |
| ENUM_EEPROM                     | Defined or undefined                                                                                                                                             | When defined, read the contents of the Enumeration packet from external non-volatile memory. If undefined, use the macros defined below.. |
| EEPROM_REG_ADDR_BITS            | 8 or 16                                                                                                                                                          | Specify number for register address bits used in the EEPROM.                                                                              |
| MAC_ADDR[47:0]                  | Any value                                                                                                                                                        | Used in BOOTP packet if ENUM_EEPROM is undefined.                                                                                         |
| BOARD_VER[159:0]                | Any value                                                                                                                                                        | Used in BOOTP packet if ENUM_EEPROM is undefined.                                                                                         |
| BOARD_SN[55:0]                  | Any value                                                                                                                                                        | Used in BOOTP packet if ENUM_EEPROM is undefined.                                                                                         |
| FPGA_CRC[15:0]                  | Any value                                                                                                                                                        | Used in BOOTP packet if ENUM_EEPROM is undefined.                                                                                         |
| MISC[31:0]                      | Any value                                                                                                                                                        | Used in BOOTP packet if ENUM_EEPROM is undefined.                                                                                         |
| DATAPATH_WIDTH                  | 8, 64, 512                                                                                                                                                       | Width of the Sensor AXI Stream TDATA in bits. This number must be byte-aligned. Meaning, it must be a number divisible by 8.              |
| DATAKEEP_WIDTH                  | DATAPATH_WIDTH/8                                                                                                                                                 | Width of the Sensor AXI Stream TKEEP.This should not be changed.                                                                          |
| DATAUSER_WIDTH                  | 1-2                                                                                                                                                              | Width of the Sensor AXI Stream TUSER signal.                                                                                              |
| SENSOR_IF_INST                  | 1-2                                                                                                                                                              | Number of Sensor Interface.                                                                                                               |
| SENSOR_TX_ENABLE                | 1 or undefined                                                                                                                                                   | Instantiates modules needed for Sensor TX                                                                                                 |
| HOST_WIDTH                      | 8, 64, 512                                                                                                                                                       | Width of the Host AXI Stream TDATA in bits. This number must be byte-aligned. Meaning, it must be a number divisible by 8.                |
| HOSTKEEP_WIDTH                  | DATAPATH_WIDTH/8                                                                                                                                                 | Width of the Host AXI Stream TKEEP.This should not be changed.                                                                            |
| HOSTUSER_WIDTH                  | 1                                                                                                                                                                | Width of the Host AXI Stream TUSER signal.                                                                                                |
| HOST_IF_INST                    | 1-2                                                                                                                                                              | Number of Host interfaces.                                                                                                                |
| HOST_MTU                        | 4096 (DO NOT CHANGE FOR 10G SYSTEM)                                                                                                                              | Size of Ethernet packet in bytes.                                                                                                         |
| SPI_INST                        | 1-8                                                                                                                                                              | Number of SPI interfaces.                                                                                                                 |
| I2C_INST                        | 1-8                                                                                                                                                              | Number of I2C interfaces.                                                                                                                 |
| GPIO_INST                       | 0-255                                                                                                                                                            | Number of GPIO Input & Output bits.                                                                                                       |
| GPIO_RESET_VALUE[GPIO_INST-1:0] | 0                                                                                                                                                                | Reset value of GPIO bits.                                                                                                                 |
| REG_INST                        | 1-8                                                                                                                                                              | Number of user register.                                                                                                                  |
| RX_PACKETIZER_EN                | 1 or undefined (KEEP AS 1 FOR 10G SYSTEM)                                                                                                                        | Instantiate modules needed for Sensor Packetizer                                                                                          |
| SIF_SORT_RESOLUTION             | DO NOT TOUCH                                                                                                                                                     | TBD. Do not change.                                                                                                                       |
| SIF_VP_COUNT                    | DO NOT TOUCH                                                                                                                                                     | TBD. Do not change.                                                                                                                       |
| SIF_VP_SIZE                     | DO NOT TOUCH                                                                                                                                                     | TBD. Do not change.                                                                                                                       |
| SIF_NUM_CYCLES                  | DO NOT TOUCH                                                                                                                                                     | TBD. Do not change.                                                                                                                       |
| SIF_DYN_VP                      | DO NOT TOUCH                                                                                                                                                     | TBD. Do not change.                                                                                                                       |
| SIF_MIXED_VP_SIZE               | DO NOT TOUCH                                                                                                                                                     | TBD. Do not change.                                                                                                                       |
| SIF_TX_BUF_SIZE                 | DO NOT TOUCH                                                                                                                                                     | TBD. Do not change.                                                                                                                       |
| N_INIT_REG                      | Integer value                                                                                                                                                    | Number of initialization registers.                                                                                                       |

### Build Revision

Parameter `BUILD_REV[47:0]` can be passed down to the "HOLOLINK_top" module to uniquely
identify the FPGA build revision.

### BOOTP Packet

The HSB IP transmits a broadcast BOOTP Request packet approximately once per second.
BOOTP packets are used to enumerate the Holoscan Sensor Bridge to host and to change the
IP address from default in case of multiple board enumeration. Holoscan Sensor Bridge IP
BOOTP packets adhere to RFC-951. The Vendor Field of the BOOTP packet is used to
communicate enumeration info and status of Holoscan Sensor Board to the host.

Holoscan Sensor Bridge IP BOOTP Vendor field is laid out in below table:

| **Byte Number** | **Description**                   | **Value**                                                                          |
| --------------- | --------------------------------- | ---------------------------------------------------------------------------------- |
| [0]             | Vendor Tag                        | 0xE0                                                                               |
| [1]             | Tag Length                        | 0x2B                                                                               |
| [5:2]           | ASCII "NVDA"                      | 0x4144564E, in Big Endian format                                                   |
| [6]             | Ethernet Port Number              | 0x0 for Port 0 , 0x1 for Port 1                                                    |
| [7]             | Enumeration Version               | 0x2                                                                                |
| [9:8]           | Reserved                          | 0x0000                                                                             |
| [25:10]         | UUID                              | UUID defined in HOLOLINK_def.svh. Big Endian format                                |
| [29:26]         | Reserved                          | 0x00000000                                                                         |
| [36:30]         | Board Serial Number               | BOARD_SN defined in HOLOLINK_def.svh or fetched from EEPROM. Little Endian format  |
| [38:37]         | Holoscan Sensor Bridge IP Version | *0x2507* for *Holoscan SDK v2.2.0* release. Little Endian format.                  |
| [40:39]         | FPGA CRC                          | FPGA_CRC defined in HOLOLINK_def.svh or fetched from EEPROM. Little Endian format. |
| [44:41]         | MISC                              | MISC defined in HOLOLINK_def.svh or fetched from EEPROM. Little Endian format.     |
| [54:45]         | PTP Timestamp                     | PTP Timestamp. Seconds and Nanoseconds in PTP v2 format. Big Endian format.        |
| [56:55]         | Packet Sequence Number            | Packet Sequence Number. Increments per BOOTP packet sent. Big Endian format.       |
| [57]            | Status                            | 0x1 when PTP is enabled in HSB and a SYNC PTP packet has been received.            |
| [63:58]         | Reserved                          | Reserved                                                                           |

BOOTP enumeration info can be stored in non-volatile memory or from macros defined in
the “*HOLOLINK_def.svh*”.

Mass production of boards with HOLOLINK IP requires an external EEPROM to store unique
MAC address and board serial numbers. When Using evaluation platforms for test and
bring-up, using the macro for fixed values are acceptable.

Each of the macros used for enumeration packet is further explained below.

UUID: This macro defines the unique identifier for the board. Holoscan uses this value
to identify which board it is interfacing and fetch the correct bitfile for remote
flash. For newly developed boards, users can generate and assign a new UUID. This macro
must be defined and is not optionally stored in the non-volatile memory.

ENUM_EEPROM: When this macro is defined, the contents of the below macros are retrieved
from an on-board, non-volatile memory upon boot-up. The contents of the below macros
must be stored in specific address of the non-volatile memory. Further details can be
provided upon request.

Macros listed below are used only if ENUM_EEPROM macro is undefined.

MAC_ADDR: This macro defines the UDP MAC Address of the FPGA. The MAC Address is
incremented by 1 for each Host interface available. For example, if the MAC_ADDR macro
is defined as 48’hCAFEC0FFEE00, Host Interface 0 will have address 48’hCAFEC0FFEE00 and
Host Interface 1 will have address 48’hCAFEC0FFEE01, and so on.

BOARD_VER: This macro defines the Board Version.

BOARD_SN: This macro defines the Board Serial Number.

FPGA_CRC: This macro defines the CRC of the FPGA bit image and is used only if
ENUM_EEPROM macro is undefined. The intended function is for software to check the
validity of the FPGA bit image by checking it against what the software calculates. This
function is not yet supported.

MISC: This macro defines miscellaneous information.

## System Initialization

The Holoscan Sensor Bridge IP provides system initialization function to write to
registers upon power up. This function is used to initialize the Ethernet block to
establish ethernet connection between the FPGA and the host, and can be used for any
other user function in the top level design.

The list of registers to be initialized is defined in “Hololink_def.svh” as “init_reg”
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
