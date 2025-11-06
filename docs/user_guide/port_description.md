# Port Description

The port descriptions for the Holoscan Sensor Bridge IP are described in the following
sections. For more information on macros defined in "HOLOLINK_def.svh", refer to
[Macro Definitions](ip_integration.md#macro-definition)

## Global Reset

Table 1 Global Reset Port

| **Signal Name** | **Direction** | **Description**                         |
| --------------- | ------------- | --------------------------------------- |
| i_sys_rst       | Input         | Global, Asynchronous, Active High Reset |

## User Register Ports

Table 2 User Register Clock and Reset Ports

| **Signal Name** | **Direction** | **Description**                        |
| --------------- | ------------- | -------------------------------------- |
| i_apb_clk       | Input         | APB Clock. Must be greater than 20MHz. |
| o_apb_rst       | Output        | APB Synchronous, Active High Reset     |

Table 3 User Register APB Ports

| **Signal Name**                       | **Direction** | **Description**     |
| ------------------------------------- | ------------- | ------------------- |
| i_apb_pready [N-1:0]<sup>1<sup>       | Input         | APB Ready           |
| [31:0] i_apb_prdata[N-1:0]<sup>1<sup> | Input         | APB Read Data       |
| i_apb_pserr[N-1:0]<sup>1<sup>         | Input         | APB Completer Error |
| o_apb_psel[N-1:0]<sup>1<sup>          | Output        | APB Select          |
| o_apb_penable                         | Output        | APB Enable          |
| o_apb_paddr[31:0]                     | Output        | APB Address         |
| o_apb_pwdata[31:0]                    | Output        | APB Write Data      |
| o_apb_pwrite                          | Output        | APB Write           |

1. N=`REG_INST`

Table 4 External Enumeration Ports

These ports are only available if `ENUM_EEPROM` is NOT defined in "HOLOLINK_def.svh"

| **Signal Name**                     | **Direction** | **Description**                                                                     |
| ----------------------------------- | ------------- | ----------------------------------------------------------------------------------- |
| [47:0] i_mac_addr[N-1:0]<sup>1<sup> | Input         | MAC Address for each Ethernet interface.                                            |
| i_board_sn[55:0]                    | Input         | Board Serial Number.                                                                |
| i_enum_vld                          | Input         | Enumeration Valid. Once "i_mac_addr" and "i_board_sn" is set, set and keep it to 1. |

1. N=`HOST_IF_INST`

Table 5 User Register System Initialization Ports

| **Signal Name** | **Direction** | **Description**                                                     |
| --------------- | ------------- | ------------------------------------------------------------------- |
| o_init_done     | Output        | System Initialization done. Refer to System Initialization section. |

## Sensor Interface Ports

Table 6 Sensor RX Interface Ports

These ports are only available if `SENSOR_RX_IF_INST` is defined in "HOLOLINK_def.svh"

| **Signal Name**                            | **Direction** | **Description**                                                                          |
| ------------------------------------------ | ------------- | ---------------------------------------------------------------------------------------- |
| i_sif_rx_clk[N-1:0]<sup>1<sup>             | Input         | Sensor RX Interface Clock.                                                               |
| o_sif_rx_rst[N-1:0]<sup>1<sup>             | Output        | Sensor RX Interface Synchronous, Active-High Reset                                       |
| i_sif_axis_tvalid[N-1:0]<sup>1<sup>        | Input         | Sensor RX AXI-Stream Valid                                                               |
| i_sif_axis_tlast[N-1:0]<sup>1<sup>         | Input         | Sensor RX AXI-Stream Last.                                                               |
| [W-1:0] i_sif_axis_tdata[N-1:0]<sup>1<sup> | Input         | Sensor RX AXI-Stream Data.                                                               |
| [X-1:0] i_sif_axis_tkeep[N-1:0]<sup>1<sup> | Input         | Sensor RX AXI-Stream Keep. Currently not supported. See Sensor RX section for more info. |
| [Y-1:0] i_sif_axis_tuser[N-1:0]<sup>1<sup> | Input         | Sensor RX AXI-Stream User.                                                               |
| o_sif_axis_tready[N-1:0]<sup>1<sup>        | Output        | Sensor RX AXI-Stream Ready                                                               |

1. N=`SENSOR_RX_IF_INST`, W=`DATAPATH_WIDTH`, X=`DATAKEEP_WIDTH`, Y=`DATAUSER_WIDTH`

Table 7 Sensor TX Interface

These ports are only available if `SENSOR_TX_IF_INST` is defined in "HOLOLINK_def.svh"

| **Signal Name**                            | **Direction** | **Description**                                    |
| ------------------------------------------ | ------------- | -------------------------------------------------- |
| i_sif_tx_clk[N-1:0]<sup>1<sup>             | Input         | Sensor TX Interface Clock.                         |
| o_sif_tx_rst[N-1:0]<sup>1<sup>             | Output        | Sensor TX Interface Synchronous, Active-High Reset |
| o_sif_axis_tvalid[N-1:0]<sup>1<sup>        | Output        | TBD                                                |
| o_sif_axis_tlast[N-1:0]<sup>1<sup>         | Output        | TBD                                                |
| [W-1:0] o_sif_axis_tdata[N-1:0]<sup>1<sup> | Output        | TBD                                                |
| [X-1:0] o_sif_axis_tkeep[N-1:0]<sup>1<sup> | Output        | TBD                                                |
| [Y-1:0] o_sif_axis_tuser[N-1:0]<sup>1<sup> | Output        | TBD                                                |
| i_sif_axis_tready[N-1:0]<sup>1<sup>        | Input         | TBD                                                |

1. N=`SENSOR_TX_IF_INST`, W=`DATAPATH_WIDTH`, X=`DATAKEEP_WIDTH`, Y=`DATAUSER_WIDTH`

Table 8 Sensor Event Ports

| **Signal Name**    | **Direction** | **Description**                                                                    |
| ------------------ | ------------- | ---------------------------------------------------------------------------------- |
| i_sif_event [15:0] | Input         | Sensor Interface Event. Asynchronous. Refer to Sensor RX section for more details. |

## Host Interface Ports

Table 9 Host Interface Clock and Reset Ports

| **Signal Name** | **Direction** | **Description**                                                   |
| --------------- | ------------- | ----------------------------------------------------------------- |
| i_hif_clk       | Input         | 156.25MHz Host Interface Clock. See clocking section for details. |
| o_hif_rst       | Output        | Host Interface Synchronous, Active-High Reset.                    |

Table 10 Host RX Interface Ports

Connect the Host RX AXI-Streaming ports directly to Ethernet MAC RX AXI-Streaming ports.

| **Signal Name**                           | **Direction** | **Description**  |
| ----------------------------------------- | ------------- | ---------------- |
| i_hif_axis_tvalid[N-1:0]<sup>1<sup>       | Input         | AXI-Stream Valid |
| i_hif_axis_tlast[N-1:0]<sup>1<sup>        | Input         | AXI-Stream Last  |
| [W-1:0]i_hif_axis_tdata[N-1:0]<sup>1<sup> | Input         | AXI-Stream Data  |
| [X-1:0]i_hif_axis_tkeep[N-1:0]<sup>1<sup> | Input         | AXI-Stream Keep  |
| [Y-1:0]i_hif_axis_tuser[N-1:0]<sup>1<sup> | Input         | AXI-Stream User  |
| o_hif_axis_tready[N-1:0]<sup>1<sup>       | Output        | AXI-Stream Ready |

1. N=`HOST_IF_INST`, W=`DATAPATH_WIDTH`, X=`DATAKEEP_WIDTH`, Y=`DATAUSER_WIDTH`

Table 11 Host TX Interface Ports

Connect the Host TX AXI-Streaming ports directly to Ethernet MAC TX AXI-Streaming ports.

| **Signal Name**                           | **Direction** | **Description**  |
| ----------------------------------------- | ------------- | ---------------- |
| o_hif_axis_tvalid[N-1:0]<sup>1<sup>       | Output        | AXI-Stream Valid |
| o_hif_axis_tlast[N-1:0]<sup>1<sup>        | Output        | AXI-Stream Last  |
| [W-1:0]o_hif_axis_tdata[N-1:0]<sup>1<sup> | Output        | AXI-Stream Data  |
| [X-1:0]o_hif_axis_tkeep[N-1:0]<sup>1<sup> | Output        | AXI-Stream Keep  |
| [Y-1:0]o_hif_axis_tuser[N-1:0]<sup>1<sup> | Output        | AXI-Stream User  |
| i_hif_axis_tready[N-1:0]<sup>1<sup>       | Input         | AXI-Stream Read  |

1. N=`HOST_IF_INST`, W=`DATAPATH_WIDTH`, X=`DATAKEEP_WIDTH`, Y=`DATAUSER_WIDTH`

## Peripheral Interface Ports

Table 12 SPI Ports

These ports are only available if `SPI_INST` is defined in "HOLOLINK_def.svh"

| **Signal Name**                   | **Direction** | **Description**          |
| --------------------------------- | ------------- | ------------------------ |
| o_spi_csn[N-1:0]<sup>1<sup>       | Output        | Chip Select (Active Low) |
| o_spi_sck[N-1:0]<sup>1<sup>       | Output        | SPI Clock                |
| o_spi_oen[N-1:0]<sup>1<sup>       | Output        | Output Enable            |
| [3:0]o_spi_sdio[N-1:0]<sup>1<sup> | Output        | SDIO Output              |
| [3:0]i_spi_sdio[N-1:0]<sup>1<sup> | Input         | SDIO Input               |

1. N=`SPI_INST`

Table 13 I2C Ports

These ports are only available if `I2C_INST` is defined in "HOLOLINK_def.svh"

| **Signal Name**                | **Direction** | **Description**         |
| ------------------------------ | ------------- | ----------------------- |
| i_i2c_scl[N-1:0]<sup>1<sup>    | Input         | I2C Clock               |
| i_i2c_sda[N-1:0]<sup>1<sup>    | Input         | I2C Data                |
| o_i2c_scl_en[N-1:0]<sup>1<sup> | Output        | I2C Clock Output Enable |
| o_i2c_sda_en[N-1:0]<sup>1<sup> | Output        | I2C Data Output Enable  |

1. N=`I2C_INST`

Table 14 GPIO Ports

| **Signal Name**          | **Direction** | **Description**                      |
| ------------------------ | ------------- | ------------------------------------ |
| i_gpio[N-1:0]<sup>1<sup> | Input         | GPIO In. Synchronized to “i_apb_clk” |
| o_gpio[N-1:0]<sup>1<sup> | Output        | GPIO Out. Synchronous to “i_apb_clk” |

1. N=`GPIO_INST`

Table 15 Sensor Reset Port

| **Signal Name**     | **Direction** | **Description**                                                                          |
| ------------------- | ------------- | ---------------------------------------------------------------------------------------- |
| o_sw_sen_rst [31:0] | Output        | Register Controlled Reset. Connect to on-board sensor reset pin                          |
| o_sw_sys_rst        | Output        | Register controlled self-clearing reset. Can be used to reset blocks, such as PCS block. |

Table 16 PTP Clock and Reset

| **Signal Name** | **Direction** | **Description** |
| --------------- | ------------- | --------------- |
| i_ptp_clk       | Input         | PTP Clock       |
| o_ptp_rst       | Output        | PTP Reset       |

Table 17 Internal PTP Port

These ports are only available if `EXT_PTP` is NOT defined in "HOLOLINK_def.svh". PTP
module is instantiated within the HSB IP and is used to synchronize and timestamp.

| **Signal Name**      | **Direction** | **Description**                                                          |
| -------------------- | ------------- | ------------------------------------------------------------------------ |
| o_ptp_sec [47:0]     | Output        | PTP Seconds Field per PTP1588-2019 v2 spec. Synchronous to i_ptp_clk     |
| o_ptp_nanosec [31:0] | Output        | PTP Nanoseconds Field per PTP1588-2019 v2 spec. Synchronous to i_ptp_clk |
| o_pps                | Output        | Pulse Per Second. Synchronous to i_ptp_clk                               |

Table 18 External PTP Port

These ports are only available if `EXT_PTP` is defined in "HOLOLINK_def.svh". If the
Ethernet MAC IP has built-in PTP feature, that can be used to pass the timestamp to the
HSB IP.

| **Signal Name**      | **Direction** | **Description**                                                          |
| -------------------- | ------------- | ------------------------------------------------------------------------ |
| i_ptp_sec [47:0]     | Input         | PTP Seconds Field per PTP1588-2019 v2 spec. Synchronous to i_ptp_clk     |
| i_ptp_nanosec [31:0] | Input         | PTP Nanoseconds Field per PTP1588-2019 v2 spec. Synchronous to i_ptp_clk |
