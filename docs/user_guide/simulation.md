# Simulation

Simulation bring up allows users to check the HOLOLINK IP instantiation and port
connections. Modifications listed below are needed to bring up the HOLOLINK IP in a
configured state to stream sensor data out of the Host TX interface.

1. Comment out "define ENUM_EEPROM" in "HOLOLINK_def.svh" In hardware, when ENUM_EEPROM
   is defined, the HOLOLINK IP reads the external EEPROM via I2C to fetch the unique MAC
   address and meta data. But because this will require a Bus Functional Model of the
   EEPROM in the simulation testbench, users can comment out "define ENUM_EEPROM". When
   "ENUM_EEPROM" is not defined, the HOLOLINK IP will use the "MAC_ADDR" and other
   defines in the "HOLOLINK_def.svh" as hardcoded values for the ethernet interface.

1. Initialize the HOLOLINK IP In hardware, the software APIs configure the HOLOLINK IP
   for dataplane stream on Host TX interface. In simulation, users can use the
   "init_reg" in "HOLOLINK_def.svh" to configure the HOLOLINK IP out of reset.

Below is a list of registers that can be added to "init_reg" to initialize ethernet port
0\. Once the Sensor RX AXI-S interface is driven with 1408 bytes or more, the HOLOLINK IP
will drive the dataplane stream on the Host TX interface.

```
    //Address       Data
    {32'h0200_030C, 32'h0000_05CE}, // dp_pkt_0  , dp_pkt_len
    {32'h0200_0324, 32'h0000_0001}, // dp_pkt_0  , dp_pkt_vip_mask
    {32'h0200_0310, 32'h0000_600D}, // dp_pkt_0  , dp_pkt_mac_addr_lo
    {32'h0200_0314, 32'h0000_0000}, // dp_pkt_0  , dp_pkt_mac_addr_hi
    {32'h0200_0318, 32'h0000_BEEF}, // dp_pkt_0  , dp_pkt_ip_addr
    {32'h0200_031C, 32'h0000_12B7}, // dp_pkt_0  , dp_pkt_host_udp_port
    {32'h0200_0320, 32'h0000_3000}, // dp_pkt_0  , dp_pkt_fpga_udp_port
    {32'h0200_1000, 32'h0000_0000}, // dp_pkt_0  , Destination QP
    {32'h0200_1008, 32'h0000_0000}, // dp_pkt_0  , Start Virtual Address MSB
    {32'h0200_100C, 32'h0000_0000}, // dp_pkt_0  , Start Virtual Address LSB
    {32'h0200_1010, 32'h0000_0000}, // dp_pkt_0  , End Virtual Address MSB
    {32'h0200_1014, 32'h0001_0000}, // dp_pkt_0  , End Virtual Address LSB
    {32'h0200_1004, 32'h0000_F00D}, // dp_pkt_0  , Remote Key
    {32'h0200_0108, 32'h0000_0064}, // eth_pkt_0 , Eth pkt data plane priority
```

Above example is for 1 data path, for additional data paths

1. Add offset "0x0001_0000" to the addresses above
1. Instantiate 2nd or more data paths in HOLOLINK IP HIF TX/RX Interface ports.
1. Increase N_INIT_REG by number of registers added to initialize.