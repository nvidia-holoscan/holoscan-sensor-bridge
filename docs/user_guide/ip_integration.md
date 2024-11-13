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

| \*\*Macro                         | **Tested Values**                                                                  | **Description**                                                                                                                                |
| --------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| FPGA_VENDOR                       | LATTICE, PSG, MICROCHIP                                                            | Defines the FPGA Vendor used.                                                                                                                  |
| HIF_CLK_FREQ                      | 156250000 (for DATAPATH_WIDTH=64)<br />201416016 (for DATAPATH_WIDTH=512)          | Clock frequency of the Host Interface. Unit is in Hz                                                                                           |
| APB_CLK_FREQ                      | 19531250 (for DATAPATH_WIDTH=64)<br />100000000 (for DATAPATH_WIDTH=512)           | Clock frequency of the APB  Interface. Unit is in Hz                                                                                           |
| BOARD_ID\[7:0\]                   | 8'h02 for Lattice LF-SNSR-ETH-EV<br />8'h04 for Microchip MPF200-ETH-SENSOR-BRIDGE | Used in Enumeration packet                                                                                                                     |
| ENUM_EEPROM                       | Defined or undefined                                                               | When defined, read the contents of the Enumeration packet from from external non-volatile memory. If undefined, use the macros defined below.. |
| MAC_ADDR\[47:0\]                  | Any value                                                                          | Used in Enumeration packet if ENUM_EEPROM is undefined.                                                                                        |
| BOARD_VER\[159:0\]                | Any value                                                                          | Used in Enumeration packet if ENUM_EEPROM is undefined.                                                                                        |
| BOARD_SN\[55:0\]                  | Any value                                                                          | Used in Enumeration packet if ENUM_EEPROM is undefined.                                                                                        |
| FPGA_CRC\[15:0\]                  | Any value                                                                          | Used in Enumeration packet if ENUM_EEPROM is undefined.                                                                                        |
| MISC\[31:0\]                      | Any value                                                                          | Used in Enumeration packet if ENUM_EEPROM is undefined.                                                                                        |
| DATAPATH_WIDTH                    | 64, 512                                                                            | Width of the AXI Stream TDATA in bits. This number must be byte-aligned. Meaning, it must be a number divisible by 8.                          |
| DATAKEEP_WIDTH                    | DATAPATH_WIDTH/8                                                                   | Width of the AXI Stream TKEEP.This should not be changed.                                                                                      |
| DATAUSER_WIDTH                    | 1                                                                                  | Width of the AXI Stream TUSER signal.                                                                                                          |
| SENSOR_IF_INST                    | 1-2                                                                                | Number of Sensor interfaces.                                                                                                                   |
| HOST_IF_INST                      | 1-2                                                                                | Number of Host interfaces.                                                                                                                     |
| HOST_MTU                          | 1500 (DO NOT CHANGE FOR 10G SYSTEM)                                                | Size of Ethernet packet in bytes.                                                                                                              |
| SPI_INST                          | 1-8                                                                                | Number of SPI interfaces.                                                                                                                      |
| I2C_INST                          | 1-8                                                                                | Number of I2C interfaces.                                                                                                                      |
| GPIO_INST                         | 0-255                                                                              | Number of GPIO Input & Output bits.                                                                                                            |
| GPIO_RESET_VALUE\[GPIO_INST-1:0\] | 0                                                                                  | Reset value of GPIO bits.                                                                                                                      |
| REG_INST                          | 1-8                                                                                | Number of user register.                                                                                                                       |
| SIF_SORT_RESOLUTION               | DO NOT TOUCH                                                                       | TBD. Do not change.                                                                                                                            |
| SIF_VP_COUNT                      | DO NOT TOUCH                                                                       | TBD. Do not change.                                                                                                                            |
| SIF_VP_SIZE                       | DO NOT TOUCH                                                                       | TBD. Do not change.                                                                                                                            |
| SIF_NUM_CYCLES                    | DO NOT TOUCH                                                                       | TBD. Do not change.                                                                                                                            |
| SIF_DYN_VP                        | DO NOT TOUCH                                                                       | TBD. Do not change.                                                                                                                            |
| SIF_MIXED_VP_SIZE                 | DO NOT TOUCH                                                                       | TBD. Do not change.                                                                                                                            |
| N_INIT_REG                        | Integer value                                                                      | Number of initialization registers.                                                                                                            |

### Build Revision

Parameter "HOLOLINK_REV" can be passed down to the "HOLOLINK_top" module. This parameter
is used to identify the revision of the FPGA and the HOLOLINK and is sent to the host as
part of the Enumeration Packet.

From the module where "HOLOLINK_top" module is instantiated, the instantiated parameter,
HOLOLINK_REV\[15:0\] must be set to *16'h2407* for *Holoscan SDK v1.1.0* release.

### Enumeration Packet

The Holoscan Sensor Bridge IP transmits a broadcast UDP enumeration packet approximately
once per second. The content of the enumeration packet can be sourced from on-board,
non-volatile memory or from macros defined in the “*HOLOLINK_def.svh*”.

Mass production of boards with HOLOLINK IP requires an external EEPROM to store unique
MAC address and board serial numbers. When Using evaluation platforms for test and
bring-up, using the macro for fixed values are acceptable.

Each of the macros used for enumeration packet is further explained below.

BOARD_ID: This macro defines the Board ID. Holoscan uses this value to identify which
board it is interfacing. This macro must be defined and is not optionally stored in the
non-volatile memory.

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
array. The “init_reg” is an unpacked array, sized \[N_INIT_REG\] \[63:0\], where the
N_INIT_REG macro defines the number of registers to be initialized and the 64-bit vector
is used to define the 32- bit address of register at \[63:32\] and the 32-bit write data
at \[31:0\].

To give an example of one of the init_reg array entry:

{32'h1000_0020, 32'h0000_00FF} //init_reg\[0\]

Will write to the User REG_INST_0 block address offset 0x0000_0020 the data 0x0000_00FF.

Once system initialization is complete, the “o_init_done” port will be asserted high in
“i_apb_clk” domain. User can use this signal to gate their logic if it’s dependent on
ethernet block initialization.
