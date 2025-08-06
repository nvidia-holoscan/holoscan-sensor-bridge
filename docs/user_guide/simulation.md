# Simulation

Simulation bring up allows users to check the HOLOLINK IP instantiation and port
connections. Modifications listed below are needed to bring up the HOLOLINK IP in a
configured state to stream sensor data out of the Host TX interface.

1. Comment out `define ENUM_EEPROM` in "HOLOLINK_def.svh" In hardware, when
   `ENUM_EEPROM` is defined, the Holoscan Sensor Bridge IP reads the external EEPROM via
   I2C to fetch the unique MAC address and meta data. But because this will require a
   Bus Functional Model of the EEPROM in the simulation testbench, users can comment out
   `define ENUM_EEPROM`. When `ENUM_EEPROM` is not defined, the Holoscan Sensor Bridge
   IP will use the `MAC_ADDR` and other defines in the "HOLOLINK_def.svh" as hardcoded
   values for the ethernet interface.

1. Initialize the Holoscan Sensor Bridge IP. In hardware, the software APIs configure
   the Holoscan Sensor Bridge IP for dataplane stream on Host TX interface. In
   simulation, users can use the "init_reg" in "HOLOLINK_def.svh" to configure the
   Holoscan Sensor Bridge IP out of reset.

Below is a list of registers that can be added to "init_reg" to initialize ethernet port
0\. Once the Sensor RX AXI-S interface is driven with 1408 bytes or more, the Holoscan
Sensor Bridge IP will drive the dataplane stream on the Host TX interface.

```
    //Address       Data
    {32'h0200_0304, 32'h0000_000B}, // dp_pkt_0  , dp_pkt_len
    {32'h0200_0308, 32'h0000_12B7}, // dp_pkt_0  , dp_pkt_host_udp_port
    {32'h0200_030C, 32'h0000_0001}, // dp_pkt_0  , dp_pkt_vip_mask
    {32'h0000_1020, 32'h0000_600D}, // sif_0     , dp_pkt_mac_addr_lo
    {32'h0000_1024, 32'h0000_0000}, // sif_0     , dp_pkt_mac_addr_hi
    {32'h0000_1028, 32'h0000_BEEF}, // sif_0     , dp_pkt_ip_addr
    {32'h0000_102C, 32'h0000_3000}, // sif_0     , dp_pkt_fpga_udp_port
    {32'h0000_1000, 32'h0000_0000}, // sif_0     , Destination QP
    {32'h0000_1004, 32'h0000_F00D}, // sif_0     , Remote Key
    {32'h0000_1008, 32'h0000_0000}, // sif_0     , Buffer 0 Virtual Address
    {32'h0000_1018, 32'h0001_0000}, // sif_0     , Bytes per Window
    {32'h0000_101C, 32'h0000_0001}, // sif_0     , Buffer Enable
    {32'h0200_0108, 32'h0000_0064}, // eth_pkt_0 , Eth pkt data plane priority
```

Above example is for 1 data path, for additional data paths

1. Add offset "0x0001_0000" to the dp_pkt register addresses and add offset
   "0x0000_0040" to the sif register addresses.
1. Instantiate 2nd or more data paths in Holoscan Sensor Bridge IP HIF TX/RX Interface
   ports.
1. Increase N_INIT_REG by number of registers added to initialize.
