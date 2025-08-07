# Precision Time Protocol (PTP)

HSB IP supports Precision Time Protocol (PTP) per IEEE1588-2019 specification.

PTP synchronizes the HSB IP's internal time to the host time. To enable the host as the
Time Transmitter and send PTP packets, refer to the
[Host Setup](sensor_bridge_hardware_setup.md) page.

## PTP Use Case

Synchronized PTP time can be used to:

1. Timestamp specific Sensor Interface and Host Interface events and passed in the
   Metadata packet.
1. Synchronize multiple HSBs on the network.
1. In camera application, synchronized timestamp can be used to generate a synchronized
   Vertical SYNC strobe to align the exposure across multiple cameras on the network.
   VSYNC is currently supported external to the HSB IP and needs a FPGA GPIO pin that
   routes to the camera sensor VSYNC pin.

## PTP Profile

HSB IP supports modified version of the PTP-1588 default profile. Details of what is
supported in the HSB IP is listed below.

1. Operates as PTP Receiver only
1. Transmit and Receive PTP messages over Ethernet L2 Layer
1. Receive Forwardable(0x011B19000000) Multicast MAC addresses
1. Support One and Two-Step Sync messages
1. Support End to End Delay Mechanism
1. Transmit Delay Request Messages with Forwardable Multicast MAC address

## PTP Limitations

The HSB IP PTP currently has these limitations that can be added in future revisions.

1. Announce messages are ignored.
1. No Best Master Clock Algorithm. It assumes there is only 1 master in the network at a
   given time.

## PTP Timer

PTP block in HSB IP runs on "i_ptp_clk" domain. The PTP clock can be asynchronous to the
"i_hif_clk" domain but for best performance, it is recommended to generate the PTP clock
derived from the Ethernet PCS or MAC clock. For optimal performance, generate the PTP
clock frequency in range 95MHz to 105MHz.

HSB IP timer operates in the following manner:

1. When the HSB IP comes out of reset, timer begins at 0 seconds and 0 nanoseconds. At
   each rising clock edge, the timer increments by (1/`PTP_CLK_FREQ`) nanoseconds and
   24-bit fractional nanoseconds, where `PTP_CLK_FREQ` is a parameter defined in
   "HOLOLINK_def.svh" For example, if `PTP_CLK_FREQ=100446545` in 10G application, the
   incremental value per rising clock edge is 9.955ns.
1. When PTP frequency adjustment is enabled,the HSB IP latches the received host
   timestamp in the SYNC (and FOLLOW-UP for 2-step) message and the timer and continues
   to increment as before.
1. In subsequently received SYNC messages, the HSB IP no longer latches it's internal
   time to the received host timestamp and instead, uses the calculated offset to adjust
   its incremental value. Adjusting the incremental value (inverse of frequency)
   compensates for on-board oscillator drift and temperature variation.

## PTP Configuration

PTP registers can be configured to achieve high accuracy between host and HSB. Below are
descriptions of PTP functionality and the configurable registers.

Frequency Adjustment is calculated from the Offset Measurement (OFM) and applies to the
clock period value. OFM is calculated by taking the time difference between the host
SYNC timestamp and the HSB timestamp at the time SYNC message was received. 2
configurable gain is applied to the OFM, first configurable gain is a coarse gain and
the second configurable gain a fine gain. Both configurable gain values apply a right
shift to the OFM value. The fine gain is accumulated per sample and the coarse gain is
directly added to calculate the Frequency Adjustment value. New Frequency Adjustment
value is calculated and applied per SYNC message. The higher number of SYNC messages per
second, the greater the accuracy. The smaller the value of coarse and fine gain, the
greater the accuracy but could potentially be unstable since the Frequency Adjustment
can oscillate between a large positive and negative number. The greater the value of
coarse and fine gain, the lesser the accuracy and increased settling time, but less
potential to be unstable.

Mean Delay is the averaged value of delay between host to HSB and HSB to host. Mean
Delay is calculated per SYNC message. Mean Delay can be averaged in a moving average to
smooth out outliers.

Delay Asymmetry accounts for asymmetry between the RX and TX path outside of HSB IP,
these delay asymmetry can be vendor specific. For example, MAC RX can have a longer
delay to process than MAC TX path. If these asymmetry values are known (via simulation,
datasheet) it can achieve greater PTP accuracy. Delay Asymmetry register value is in
nanosecond unit and is (RX delay - TX delay) meaning, positive number means RX has
greater delay and negative number means TX has greater delay.

Below lists the configurable PTP registers.

| **Reg Name**          | **Reg Addr** | **Reg Value Range**     | **Notes**                                                         |
| --------------------- | ------------ | ----------------------- | ----------------------------------------------------------------- |
| Gain Enable           | 0x00000104   | 0x0 - 0x3               | Enable Frequency Adjustment Gain. [0]=Coarse Gain, [1]=Fine Gain  |
| Delay Asymmetry       | 0x0000010C   | 0x00000000 - 0xFFFFFFFF | Unit is in nanoseconds.                                           |
| Coarse Gain           | 0x00000110   | 0x0 - 0xF               | Frequency Adjustment Coarse Gain                                  |
| Fine Gain             | 0x00000114   | 0x0 - 0xF               | Frequency Adjustment Fine Gain                                    |
| Mean Delay Avg Factor | 0x00000118   | 0x0 - 0x3               | Averages by factor of 2. So 0x1 = 2 samples, 0x2 = 4 samples, etc |

Example python script to enable PTP is below.

```
  def ptp_enable(hololink)
    hololink.write_uint32(0x00000110, 0x00000002)  # DPLL CFG 1
    hololink.write_uint32(0x00000114, 0x00000002)  # DPLL CFG 2
    hololink.write_uint32(0x00000118, 0x00000003)  # Mean Delay
    hololink.write_uint32(0x00000104, 0x00000003)  # Enable DPLL
```

## PTP Performance

The performance of the HSB IP PTP was tested by comparing the Pulse Per Second (PPS)
between the host and the HSB IP after frequency adjustment was enabled.

| **Offset** | **End to End Standard Deviation** |
| ---------- | --------------------------------- |
| < 10 ns    | < 20 ns                           |

The performance test was done using the following configuration.

| **Parameter or Reg**  | **Value**   |
| --------------------- | ----------- |
| HIF_CLK_FREQ          | 156250000Hz |
| PTP_CLK_FREQ          | 100446545Hz |
| Gain Enable           | 0x3         |
| Delay Asymmetry       | 0x33        |
| Coarse Gain           | 0x2         |
| Fine Gain             | 0x2         |
| Mean Delay Avg Factor | 0x3         |
