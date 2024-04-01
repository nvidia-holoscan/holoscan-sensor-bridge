# Port Description

The port descriptions for the Holoscan Sensor Bridge IP are described in the following
sections.

## Global Reset

| Signal Name | Direction | Description                             |
| ----------- | --------- | --------------------------------------- |
| i_sys_rst   | Input     | Global, Asynchronous, Active High Reset |

## User Register Ports

Table 1 User Register Clock and Reset Ports

| Signal Name | Direction | Description                            |
| ----------- | --------- | -------------------------------------- |
| i_apb_clk   | Input     | APB Clock. Must be greater than 20MHz. |
| o_apb_rst   | Output    | APB Synchronous, Active High Reset     |

Table 2 User Register APB Ports

| **Signal Name**                           | **Direction** | **Description**     |
| ----------------------------------------- | ------------- | ------------------- |
| i_apb_pready \[N-1:0\]<sup>1<sup>         | Input         | APB Ready           |
| \[31:0\] i_apb_prdata\[0:N-1\]<sup>1<sup> | Input         | APB Read Data       |
| i_apb_pslverr\[N-1:0\]<sup>1<sup>         | Input         | APB Completer Error |
| o_apb_psel\[N-1:0\]<sup>1<sup>            | Output        | APB Select          |
| o_apb_penable                             | Output        | APB Enable          |
| o_apb_paddr\[31:0\]                       | Output        | APB Address         |
| o_apb_pwdata\[31:0\]                      | Output        | APB Write Data      |
| o_apb_pwrite                              | Output        | APB Write           |

1. N=REG_INST. Refer to the Macro Definitions section for details.

Table 3 User Register System Initialization Ports

| **Signal Name** | **Direction** | **Description**                                                     |
| --------------- | ------------- | ------------------------------------------------------------------- |
| o_init_done     | Output        | System Initialization done. Refer to System Initialization section. |

## Sensor Interface Ports

Table 4 Sensor Interface Clock and Reset Ports

| **Signal Name** | **Direction** | **Description**                                 |
| --------------- | ------------- | ----------------------------------------------- |
| i_sif_clk       | Input         | Sensor Interface Clock.                         |
| o_sif_rst       | Output        | Sensor Interface Synchronous, Active-High Reset |

Table 5 Sensor RX Interface Ports

| **Signal Name**                                | **Directi on** | **Description**                                                                |
| ---------------------------------------------- | -------------- | ------------------------------------------------------------------------------ |
| i_sif_axis_tvalid\[N-1:0\]<sup>1<sup>          | Input          | AXI-Stream Valid                                                               |
| i_sif_axis_tlast\[N-1:0\]<sup>1<sup>           | Input          | AXI-Stream Last. Currently not supported. See Sensor RX section for more info. |
| \[W-1:0\] i_sif_axis_tdata\[0:N-1\]<sup>1<sup> | Input          | AXI-Stream Data.                                                               |
| \[X-1:0\] i_sif_axis_tkeep\[0:N-1\]<sup>1<sup> | Input          | AXI-Stream Keep. Currently not supported. See Sensor RX section for more info. |
| \[Y-1:0\] i_sif_axis_tuser\[0:N-1\]<sup>1<sup> | Input          | AXI-Stream User.                                                               |
| o_sif_axis_tready                              | Output         | AXI-Stream Ready                                                               |

1. N=SENSOR_IF_INST, W=DATAPATH_WIDTH, X=DATAKEEP_WIDTH, Y=DATAUSER_WIDTH. See Macro
   Definitions section for details.Table 6 Sensor Event Ports

Table 6 Sensor Event Ports

| **Signal Name**                  | **Direction** | **Description**                                                      |
| -------------------------------- | ------------- | -------------------------------------------------------------------- |
| i_sif_event \[N-1:0\]<sup>1<sup> | Input         | Sensor Interface Event. Refer to Sensor RX section for more details. |

1.N=SENSOR_IF_INST. See Macro Definitions section for details.

## Host Interface Ports

Table 7 Host Interface Clock and Reset Ports

| **Signal Name** | **Direction** | **Description**                                                   |
| --------------- | ------------- | ----------------------------------------------------------------- |
| i_hif_clk       | Input         | 156.25MHz Host Interface Clock. See clocking section for details. |
| o_hif_rst       | Output        | Host Interface Synchronous, Active-High Reset.                    |

Table 8 Host RX Interface Ports

Connect the Host RX AXI-Streaming ports directly to Ethernet MAC TX AXI-Streaming ports.

| **Signal Name**                               | **Direction** | **Description**  |
| --------------------------------------------- | ------------- | ---------------- |
| i_hif_axis_tvalid\[N-1:0\]<sup>1<sup>         | Input         | AXI-Stream Valid |
| i_hif_axis_tlast\[N-1:0\]<sup>1<sup>          | Input         | AXI-Stream Last  |
| \[W-1:0\]i_hif_axis_tdata\[0:N-1\]<sup>1<sup> | Input         | AXI-Stream Data  |
| \[X-1:0\]i_hif_axis_tkeep\[0:N-1\]<sup>1<sup> | Input         | AXI-Stream Keep  |
| \[Y-1:0\]i_hif_axis_tuser\[0:N-1\]<sup>1<sup> | Input         | AXI-Stream User  |
| o_hif_axis_tready                             | Output        | AXI-Stream Ready |

1. N=HOST_IF_INST, W=DATAPATH_WIDTH, X=DATAKEEP_WIDTH, Y=DATAUSER_WIDTH. See Macro
   Definitions section for details.

Table 9 Host TX Interface Ports

Connect the Host TX AXI-Streaming ports directly to Ethernet MAC RX AXI-Streaming ports.

| **Signal Name**                               | **Direction** | **Description**  |
| --------------------------------------------- | ------------- | ---------------- |
| o_hif_axis_tvalid\[N-1:0\]<sup>1<sup>         | Output        | AXI-Stream Valid |
| o_hif_axis_tlast\[N-1:0\]<sup>1<sup>          | Output        | AXI-Stream Last  |
| \[W-1:0\]o_hif_axis_tdata\[0:N-1\]<sup>1<sup> | Output        | AXI-Stream Data  |
| \[X-1:0\]o_hif_axis_tkeep\[0:N-1\]<sup>1<sup> | Output        | AXI-Stream Keep  |
| \[Y-1:0\]o_hif_axis_tuser\[0:N-1\]<sup>1<sup> | Output        | AXI-Stream User  |
| i_hif_axis_tready                             | Input         | AXI-Stream Read  |

1. N=HOST_IF_INST, W=DATAPATH_WIDTH, X=DATAKEEP_WIDTH, Y=DATAUSER_WIDTH. See Macro
   Definitions section for details.

## Peripheral Interface Ports

Table 10 SPI Ports

| **Signal Name**                       | **Direction** | **Description**          |
| ------------------------------------- | ------------- | ------------------------ |
| o_spi_csn\[N-1:0\]<sup>1<sup>         | Output        | Chip Select (Active Low) |
| o_spi_sck\[N-1:0\]<sup>1<sup>         | Output        | SPI Clock                |
| o_spi_oen\[N-1:0\]<sup>1<sup>         | Output        | Output Enable            |
| \[3:0\]o_spi_sdio\[0:N-1\]<sup>1<sup> | Output        | SDIO Output              |
| \[3:0\]i_spi_sdio\[0:N-1\]<sup>1<sup> | Input         | SDIO Input               |

1. N=SPI_INST. See Macro Definitions section for details.

Table 11 I2C Ports

| **Signal Name**                  | **Direction** | **Description**         |
| -------------------------------- | ------------- | ----------------------- |
| i_i2c_scl\[N-1:0\]<sup>1<sup>    | Input         | I2C Clock               |
| i_i2c_sda\[N-1:0\]<sup>1<sup>    | Input         | I2C Data                |
| o_i2c_scl_en\[N-1:0\]<sup>1<sup> | Output        | I2C Clock Output Enable |
| o_i2c_sda_en\[N-1:0\]<sup>1<sup> | Output        | I2C Data Output Enable  |

1. N=I2C_INST. See Macro Definitions section for details.

Table 12 GPIO Ports

| **Signal Name**            | **Direction** | **Description**                      |
| -------------------------- | ------------- | ------------------------------------ |
| i_gpio\[N-1:0\]<sup>1<sup> | Input         | GPIO In. Synchronized to “i_apb_clk” |
| o_gpio\[M-1:0\]<sup>1<sup> | Output        | GPIO Out. Synchronous to “i_apb_clk” |

1. N=GPIO_IN, M=GPIO_OUT. See Macro Definitions section for details.

Table 13 Sensor Reset Port

| **Signal Name**                   | **Direction** | **Description**                                                                          |
| --------------------------------- | ------------- | ---------------------------------------------------------------------------------------- |
| o_sw_sen_rst \[N-1:0\]<sup>1<sup> | Output        | Register Controlled Reset. Connect to on-board sensor reset pin                          |
| o_sw_sys_rst                      | Output        | Register controlled self-clearing reset. Can be used to reset blocks, such as PCS block. |

1. N=SENSOR_IF_INST. See Macro Definitions section for details.
