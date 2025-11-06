# Known IP Limitations

1. The Holoscan Sensor Bridge IP supports single and block 32-bit read or write ECB
   commands.
1. The Holoscan Sensor Bridge IP does not support rapid back-to-back ICMP (or PING)
   requests. The time between successive pings should be >= 1ms.
1. The Sensor RX AXI-Stream TKEEP signals are unused. This implies all bytes of TDATA
   are valid when TVALID is high.
1. The Host TX & RX AXI-stream supports TKEEP not equal to all 1’s only when TLAST is
   high. TKEEP must be all 1’s when TLAST is low.
