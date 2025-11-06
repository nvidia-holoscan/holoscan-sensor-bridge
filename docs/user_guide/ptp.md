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

HSB IP supports the following PTP profiles

1. 1588 profile with End to End Delay Mechanism
1. gPTP profile
1. 1588 profile with Peer to Peer Delay Mechanism

PTP within HSB IP are limited to:

1. Operates as PTP Receiver only
1. Transmit and Receive PTP messages over Ethernet L2 Layer
1. Does not support Announce messages.
1. Does not support Best Master Clock Algorithm. It assumes there is only 1 master in
   the network at a given time.
1. PTP traffic can only occur on Host Interface 0.

## PTP Timer

PTP block in HSB IP runs on "i_ptp_clk" domain. The PTP clock can be asynchronous to the
"i_hif_clk" domain but for best performance, it is recommended to generate the PTP clock
derived from the Ethernet PCS or MAC clock. Use PTP clock frequency in range 95MHz to
105MHz for optimal performance.

HSB IP timer operates in the following manner:

1. Following reset, timer begins at 0 seconds and 0 nanoseconds. At each rising clock
   edge, the timer increments by (1/`PTP_CLK_FREQ`) nanoseconds and 24-bit fractional
   nanoseconds, where `PTP_CLK_FREQ` is a parameter defined in "HOLOLINK_def.svh" For
   example in 10G application, `PTP_CLK_FREQ=100446545` and the incremental value per
   rising clock edge is 9.955ns.
1. When PTP frequency adjustment is enabled,the HSB IP latches the received host
   timestamp from the received SYNC message (and FOLLOW-UP message for 2-step) and the
   timer and continues to increment as before.
1. In subsequently received SYNC messages, the HSB IP no longer latches its internal
   time to the received host timestamp. Instead, it uses the calculated offset to adjust
   its incremental value. Adjusting the incremental value (inverse of frequency)
   compensates for on-board oscillator drift and temperature variation.

## PTP Configuration

PTP registers can be configured to achieve high accuracy between host and HSB. Below are
descriptions of PTP functionality and the configurable registers.

Frequency Adjustment is calculated from the Offset Measurement (OFM) and applied to the
clock period value. OFM is the time difference between the host SYNC timestamp and the
HSB timestamp at the time SYNC message was received. 2 configurable gain is applied to
the OFM, a coarse gain and a fine gain. Both configurable gain values apply a right
shift to the OFM value. The coarse and fine gain implements a Digital PLL (DPLL) to
calculate the Frequency Adjustment value. New Frequency Adjustment value is calculated
and applied per SYNC message. The higher number of SYNC messages per second, higher the
frequency adjustment and hence, higher accuracy. Lower coarse and fine gain values
provide greater accuracy but may cause instability as the Frequency Adjustment can
oscillate between large positive and negative values. Higher gain values reduce accuracy
and increase settling time, but improve stability.

Mean Delay Average Factor takes a moving average of mean delay between host and HSB to
smooth out outliers.

Delay Asymmetry accounts for the vendor specific asymmetry between the RX and TX path
outside of the HSB IP. For example, MAC RX can have a longer delay than the MAC TX path.
If these asymmetry values are known (via simulation or datasheet) it can achieve greater
PTP accuracy. Delay Asymmetry register value is in nanosecond unit and positive number
means RX has a greater delay and negative number means TX has greater delay.

Below lists the configurable PTP registers.

| **Reg Name**          | **Reg Addr** | **Reg Value Range**     | **Notes**                                                         |
| --------------------- | ------------ | ----------------------- | ----------------------------------------------------------------- |
| Gain Enable           | 0x00000104   | 0x0 - 0x3               | Enable Frequency Adjustment Gain. [0]=Coarse Gain, [1]=Fine Gain  |
| PTP Profile           | 0x00000108   | 0x0 - 0x2               | PTP Profiles. [0]=1588 E2E, [1]=gPTP, [2]=1588 P2P                |
| Delay Asymmetry       | 0x0000010C   | 0x00000000 - 0xFFFFFFFF | Ingress Asymmetry. Unit is in nanoseconds.                        |
| Coarse Gain           | 0x00000110   | 0x0 - 0xF               | Frequency Adjustment Coarse Gain                                  |
| Fine Gain             | 0x00000114   | 0x0 - 0xF               | Frequency Adjustment Fine Gain                                    |
| Mean Delay Avg Factor | 0x00000118   | 0x0 - 0x3               | Averages by factor of 2. So 0x1 = 2 samples, 0x2 = 4 samples, etc |

An example python script to reconfigure PTP is shown below. The values used in the
example is the default configuration after reset.

```python
  def ptp_enable(hololink)
    hololink.write_uint32(0x00000108, 0x00000000)  # PTP Profile
    hololink.write_uint32(0x0000010C, 0x00000033)  # Delay Asymmetry
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
| < 25 ns    | < 20 ns                           |

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
