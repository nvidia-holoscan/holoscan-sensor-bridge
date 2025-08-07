# Known IP Limitations

1. The Holoscan Sensor Bridge IP supports 1 outstanding ECB instructions at a time.
   Following instructions can only be sent after the receiving the ack for the previous
   ECB command.
1. The Holoscan Sensor Bridge IP supports single 32bit read or write ECB commands only.
1. The Holoscan Sensor Bridge IP does not support rapid back-to-back ICMP (or PING)
   requests. The time between successive pings should be >= 1ms.
1. The Sensor RX AXI-Stream TKEEP signals are unused. This implies all bytes of TDATA
   are valid when TVALID is high.
1. The Host TX & RX AXI-stream supports TKEEP not equal to all 1’s only when TLAST is
   high. TKEEP must be all 1’s when TLAST is low.
1. `DATAPATH_WIDTH` and `HOST_WIDTH` in "HOLOLINK_def.svh" must be equal in size.
