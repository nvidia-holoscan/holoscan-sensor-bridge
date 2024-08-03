# Precision Time Protocol (PTP)

HOLOLINK IP supports Precision Time Protocol (PTP) per IEEE1588-2019 specification.

PTP synchronizes the HOLOLINK IP's internal time to the host time. This allows the
HOLOLINK IP to accurately timestamp the incoming sensor data for the host processing and
synchronize multiple boards on the network.

HOLOLINK IP timer operates in the following manner:

1. When the HOLOLINK IP comes out of reset, the timer begins at 0 seconds and 0
   nanoseconds. At each rising clock edge, the timer increments by (1/HIF_CLK_FREQ)
   nanoseconds and 16-bit fractional nanoseconds, where HIF_CLK_FREQ is a parameter
   defined in "HOLOLINK_def.svh" For example, if HIF_CLK_FREQ is set to 156250000Hz in
   10G application, the incremental value per rising clock edge is 6.4ns.
1. When the HOLOLINK IP receives a SYNC (and FOLLOW-UP for 2-step) message from the
   host, the HOLOLINK IP latches the received host timestamp as it's internal time and
   continues to increment as before.
1. When PTP frequency adjustment is enabled through software, the HOLOLINK IP no longer
   latches it's internal time to the received host timestamp. And instead, uses the
   calculated offset measurement to adjust its incremental value. Using frequency
   adjustment compensates for on-board oscillator drift, temperature variation, and
   synchronization with higher accuracy.

## PTP Profile

HOLOLINK IP supports PTP-1588 default profile. Details of what is supported in the
HOLOLINK IP is listed below.

1. Operates as PTP Slave only
1. Transmit and Receive PTP messages over Ethernet L2 Layer
1. Receive Forwardable(0x011B19000000) and Non-forwardable (0x0180C200000E) Multicast
   MAC addresses
1. Support One and Two-Step Sync messages
1. Support End to End Delay Mechanism
1. Transmit Delay Request Messages with Forwardable Multicast MAC address

## PTP Limitations

The HOLOLINK IP PTP currently has these limitations that can be added in future
revisions.

1. Mean link delay is not part of the offset measurement calculation.
1. Correction Field is not part of the offset measurement calculation.
1. Announce messages are ignored.
1. No Best Master Clock Algorithm. It assumes there is only 1 master in the network at a
   given time.

## PTP Performance

The performance of the HOLOLINK IP PTP was tested by comparing the Pulse Per Second
(PPS) between the host and the HOLOLINK IP after frequency adjustment was enabled.

| **Offset** | **End to End Standard Deviation** |
| ---------- | --------------------------------- |
| \< 20 us   | \< 100 ns                         |
