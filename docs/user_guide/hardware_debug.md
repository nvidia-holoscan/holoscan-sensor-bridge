# Hardware Debug

Few methods are available to debug the Holoscan Sensor Bridge in hardware to narrow down
the issues.

## Data Generator

Internal to the Holoscan Sensor Bridge IP, there's a Sensor Data Generator that drives
the Sensor RX Interface signals. Enabling the Sensor Data Generator tests from the
internal Sensor RX to Host TX Dataplane path to verify correct RoCE/CoE configuration
and Host TX path is integrated correctly. To enable Data Generator:

1. Add **\`define SENSOR_RX_DATA_GEN** to "HOLOLINK_def.svh"
1. Add the below code to the python script and call the function after setting up the
   receiver.

```
def data_gen(sif_index):
    sif_mask = 0x10000 * sif_index
    hololink.write_uint32(0x01000104 + sif_mask,0x00000000) # data_gen     , Disable
    hololink.write_uint32(0x01000108 + sif_mask,0x00000001) # data_gen     , Mode - Count
    hololink.write_uint32(0x01000110 + sif_mask,0x00007FFF) # data_gen     , Output Rate = 0x7FFF/2^16
    hololink.write_uint32(0x01000104 + sif_mask,0x00000003) # data_gen     , Enable + Continuous Mode
```

## Packet Sequence Number (PSN) Registers

Packet Sequence Number registers count the number of packets at various dataplane
traffic of the Holoscan Sensor Bridge.

**Sensor Interface (SIF) TVALID Counter** SIF TVALID Counter register increments for
every clock cycle "i_sif_axis_tvalid" is asserted. If this register is not incrementing,
check the driver and connection to "i_sif_axis_tvalid" port.

**Sensor Interface Start of Frame (SOF) PSN** Start of Frame is considered as the first
cycle "i_sif_axis_tvalid && o_sif_axis_tready" after "i_sif_axis_tlast" is asserted. In
camera application, "i_sif_axis_tlast" is asserted at the end of a frame. If this
register is not incrementing, check the SIF TVALID counter is incrementing and
"i_sif_axis_tlast" is properly asserted. If those signals are properly asserted,
Holoscan Sensor Bridge "o_sif_axis_tready" is stuck low, which could indicate Host
Dataplane traffic is in a stuck state.

**Host Dataplane PSN** Once sufficient sensor data is received to send out an Ethernet
packet sized by configurable register, sensor data is encapsulated by CoE or RoCE
headers and transmitted via Host TX Interface. Each Dataplane Host packets sent are
incrementally numbered. If this register is not incrementing, either no sensor data is
driven into Holoscan Sensor Bridge IP or the CoE/RoCE configuration is not set properly
to transmit the data.

```
def read_psn(sif_index):
    sif_mask = 0x10000 * sif_index
    roce_mask = 0x00040 * sif_index
    hololink.read_uint32(0x01000080 + sif_mask)  # SIF TVALID Counter 
    hololink.read_uint32(0x01000084 + sif_mask)  # SIF SOF PSN
    hololink.read_uint32(0x0000103C + roce_mask) # HIF Dataplane PSN 
```

## Peripheral Debug

The SPI and I2C controllers provide status registers to help debug communication issues.

**Status Register Monitoring** Monitor the STATUS register (offset 0x0300_0080 for SPI
and 0x0300_0280 for I2C) for both controllers:

- **BUSY (bit 0)**: 1 = transaction in progress, 0 = idle
- **FSM_ERR (bit 1)**: Configuration errors (e.g., invalid byte counts)
- **DONE (bit 4)**: Transaction completed successfully

**I2C Additional Status Bits**

- **ARB_LOST (bit 2)**: Arbitration lost (multiple masters)
- **NACK (bit 3)**: Slave sent negative acknowledge

**Common Issues and Debug Steps**

1. **Transaction Not Starting**

   - Check CONTROL.START bit is set
   - Verify BUS_EN register selects correct device
   - Monitor BUSY bit transition from 0 to 1

1. **Transaction Stuck in BUSY State**

   - Check FSM_ERR bit for configuration errors
   - Verify byte count registers are valid
   - For I2C: Check device address and timeout settings
   - For SPI: Check SPI_MODE and PRESCALER configuration

1. **Communication Failures**

   - **I2C NACK**: Verify device address and power
   - **I2C ARB_LOST**: Check for bus contention
   - **SPI Data Issues**: Verify SPI_MODE and clock frequency
   - **Both**: Check signal integrity with oscilloscope
