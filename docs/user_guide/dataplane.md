# Dataplane

Dataplane is the flow of data from sensor interface to the host interface and vice
versa. AXI-Stream protocol is used to interface to and from the Sensor and Host to the
Holoscan Sensor Bridge IP. AXI-Stream is an AMBA defined bus protocol to transfer data
between endpoints.

AXI4-Stream protocol specification documented in revision IHI0051B is used in the
Holoscan Sensor Bridge IP.

Bandwidth matching between Sensor and Host is an important consideration of Holoscan
Sensor Bridge integration. Sensor bandwidth must be equal to or less than the Host
bandwidth to avoid backpressure. Currently, the AXIS TDATA width of Sensor Interface
(defined in `DATAPATH_WIDTH`) and Host Interface (defined in `HOST_WIDTH`) must be the
same width. For example, in 10G application, the Host clock runs at 156.25MHz and
`HOST_WIDTH=64`, therefore, `DATAPATH_WIDTH=64`. And to match the Host bandwidth, and
the sensor clock can run at 156.25MHz if AXIS TVALID is high all the time or run at
double the clock frequency and assert AXIS TVALID every other clock cycle. The Sensor
clock can be asynchronous to Host clock but the effective Sensor bandwidth must be equal
to or less than the Host bandwidth.

## Sensor RX (Sensor to FPGA)

In the current version of the Holoscan Sensor Bridge IP, Sensor RX AXI-Stream signals
are constrained to:

The full TDATA bus will be sent to the host for every TVALID cycle.

TLAST high will terminate and transmit UDP packet with the amount of data currently in
the buffer and including the data during the TLAST cycle. TLAST high is an optional
function that can be used to synchronize the start of packet sensor AXI-S to the sensor
window. More information about sensor window is described later in this page.

TUSER signal is used to indicate the type of long packet in camera application, such as
embedded data or end of frame.

TKEEP signals are ignored.

In the future version of the Holoscan Sensor Bridge IP, the expected functionality of
the Sensor RX AXI-Stream signals are:

TKEEP signal will indicate valid bytes on TDATA to be transmitted. TKEEP is only valid
when TLAST is high.

An example timing diagram of the Sensor RX AXI-Stream is depicted below. This example
uses sensor data size of 39 bytes (`DATAPATH_WIDTH=64`). D0, D1, and so on in the
diagram represents 1 byte.

Note in the TLAST clock cycle, the MSB is padded 0. The padded 0 will be transmitted to
the host.

![sensor_timing](sensor_timing.png)

Figure 1 Sensor RX AXI-Streaming Interface

### Sensor Window

In certain sensor applications, a window size serves specific purpose. For example, in
camera sensor application, the number of bytes in 1 frame can be the sensor window size.

Once the end of sensor window is reached, a metadata UDP packet is sent. The metadata
UDP packet alerts the host that sensor data buffered in memory is ready to be processed.
See below in Sensor TX section for more details on metadata UDP packet.

End of sensor window can occur in 2 ways.

1. Number of received sensor data reaches the configured sensor window.
1. "i_sif_axis_tlast" is asserted.

Ideally, the calculated sensor window size matches the received sensor data size. In the
case they're different, asserting "i_sif_axis_tlast" allows for re-synchronization of
sensor window between host and the Holoscan Sensor Board.

If "i_sif_axis_tlast" is asserted earlier than the calculated sensor window size, a
metadata packet is sent to host with the Flags[0] field asserted to indicate early tlast
and the internal sensor window resets to 0.

If "i_sif_axis_tlast" is asserted after the end of window, first, a metadata packet is
sent to host for reaching end of window, and then an another metadata packet once
"i_sif_axis_tlast" is asserted, this is treated as early tlast condition.

In an architecture where the frame size is expected to be dynamic, "i_sif_axis_tlast"
can be used as the primary source of generating the end of frame. In this case, the
window size is to be configured to the largest expected value, allowing
"i_sif_axis_tlast" to always drive the end of window.

In the metadata packet, there's a "Valid Number of Bytes within Buffer" field that can
be used to identify the number of bytes transferred in the current sensor window.

### Camera Streaming

Although the Holoscan Sensor Bridge's sensor ports are agnostic to data format, there
are situations in which a unified data format is preferable. One specific application is
for streaming camera data. For seamless integration with the example IMX274 Holoscan
driver, MIPI CSI-2 camera data must be arranged in the following way. This allows for
the same drivers to work across different HSB devices. Another specific application is
when Camera over Ethernet (CoE) format is used to stream sensor data, the MIPI CSI-2
camera data MUST be arranged in the following way.

Users can adopt a different MIPI CSI-2 format than the one listed below (when not using
CoE format), as long as the software driver is also developed to decode the incoming
MIPI CSI-2 data.

Outside of the use of current software drivers for camera data, there is no restriction
on the format of the i_sif_axis_tdata data bus.

The Sensor AXI-S interface for MIPI CSI-2 camera data for IMX274 example and CoE is as
follows.

1. Short MIPI packets must not streamed to Sensor AXI-S interface.
1. Headers of Long MIPI packets must not streamed to Sensor AXI-S interface.
1. Footers of Long MIPI packets must not streamed to Sensor AXI-S interface.
1. Each Long MIPI packet can be aligned to 8-byte boundary. They can be zero padded if
   needed.
1. Pixel data must remain in the RAW format, for example MIPI RAW10 format.
1. i_sif_axis_tuser[0] can be asserted during cycles containing embedded data (MIPI Data
   Type = 0x12)\*
1. i_sif_axis_tuser[1] can be asserted on the final clock cycle of a long packet. This
   is identified as the Line End signal.\*\*
1. i_sif_axis_tlast can be asserted on the final cycle of a frame (Frame End
   packet).\*\*

\*"i_sif_axis_tuser" signals are used only for high-bandwidth camera with need for
internal Image Signal Processing (ISP). For further details, please contact the NVIDIA
Holoscan team.

\*\*Asserting "i_sif_axis_tlast" on the final cycle of a frame is optional, but is
highly recommended to re-synchronize on frames if the calculated and actual sensor
window is different. To assert "i_sif_axis_tlast" on Frame End packet will require a
buffer to hold the previous long packet, since the Frame End packet comes some time
after the previous long packet.

An example of formatting IMX274 in 1080p to Sensor RX AXI-S with `DATAPATH_WIDTH=8` is
shown below. ![MIPI_packing_8b](MIPI_packing_8b.png)

Figure 3 MIPI to Sensor RX AXI-S Packing (DATAPATH_WIDTH=8)

An example of formatting IMX274 in 1080p to Sensor RX AXI-S with `DATAPATH_WIDTH=64` is
shown below. ![MIPI_packing_64b](MIPI_packing_64b.png)

Figure 4 MIPI to Sensor RX AXI-S Packing (DATAPATH_WIDTH=64)

An example sensor window calculation for IMX274 in 1080p, RAW10 mode with
DATAPATH_WIDTH=64 is shown below.
![sensor_window_calculation](sensor_window_calculation.png)

Figure 5 Sensor Window Calculation

### Sensor Interface to Virtual Port Mapping

The Holoscan Sensor Bridge IP supports mapping multiple sensor interfaces to virtual
ports for data access. This mapping is controlled by the FPGA parameter `SIF_NUM_VP`
which defines the number of virtual ports per sensor interface. When not explicitly
configured, `SIF_NUM_VP` defaults to 1 for all sensors. The virtual port mapping is only
valid when using high bandwidth camera data settings. When high bandwidth camera data
settings are not used, the virtual port setting is undefined and set to 0. Software
interacts with sensor data through these virtual port indices.

The virtual ports are concatenated together in a sequential manner, where each sensor
interface's virtual ports are allocated in order. For example:

- With 2 sensors and `SIF_NUM_VP={1,1}`:
  - Sensor 0's data is accessible on virtual port 0
  - Sensor 1's data is accessible on virtual port 1
- With 2 sensors and `SIF_NUM_VP={2,2}`:
  - Sensor 0's data is accessible on virtual ports 0 and 1
  - Sensor 1's data is accessible on virtual ports 2 and 3

### Sensor Event

A sensor event signal can be used to generate an interrupt message to the host.

For example, the sensor event signal can be connected to the front-end sensor PHY error
signal to notify the host in the event of a PHY error and for the host to perform a soft
reset.

Another example, in camera application, the end of frame can be connected to sensor
event signal for host to change camera setting between frames.

The interrupt message can be generated on either positive or negative or both edges of
sensor event signal that can be configured using software API. Sensor event signal is
crossed into Host clock domain within Holoscan Sensor Bridge IP, so the sensor event
signal duration will need to be asserted for a minimum of 2 Host clock cycles.

## Sensor TX (FPGA to Sensor)

Sensor TX is a preliminary function. Refer to the audio player example for details.

## Host TX (FPGA to Host)

The Sensor RX data are packetized into UDP packets and transmitted on Host TX AXI-Stream
ports. There are several different UDP packet formats available.

1. Camera over Ethernet (CoE)
1. RDMA over Converged Ethernet (RoCE)
1. Linux based UDP sockets

At the end of a sensor window, a Metadata packet is sent to interrupt the host and
notify sensor frame is ready for processing.

### CoE

For details on CoE packets, please contact NVIDIA Holoscan team.

### RoCE

| **Byte Count**           | **Description**                                                                                  | **Byte Size** | **Endianness** | **Notes**                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------ | ------------- | -------------- | ------------------------------------------------------- |
| 0 to 13                  | Ethernet Header                                                                                  | 14            | Big Endian     | Software configured Destination MAC address per sensor  |
| 14 to 33                 | IPv4 Header                                                                                      | 20            | Big Endian     | Software configured Destination IPV4 address per sensor |
| 34 to 41                 | UDP Header                                                                                       | 8             | Big Endian     | Software configured Destination UDP Port per sensor     |
| 42                       | BTH<sup>2</sup>: Opcode                                                                          | 1             | Big Endian     | Hard coded to 0x2A                                      |
| 43                       | BTH: Solicited Event (1b), Migration Request (1b), Pad Count (2b), Transport Header Version (4b) | 1             | Big Endian     | Hard coded to 0x0                                       |
| 44 to 45                 | BTH: Partition Key (P_Key)                                                                       | 2             | Big Endian     | Hard coded to 0xFFFF                                    |
| 46                       | BTH: FECN (1b), BECN (1b), Reserved (6b)                                                         | 1             | Big Endian     | Hard coded to 0x0                                       |
| 47 to 49                 | BTH: Destination QP                                                                              | 3             | Big Endian     | Software configured per sensor                          |
| 50                       | BTH: Acknowledge Request (1b), Reserved (7b)                                                     | 1             | Big Endian     | Hard coded to 0x0                                       |
| 51 to 53                 | BTH: Packet Sequence Number                                                                      | 3             | Big Endian     | Increments per ethernet packet                          |
| 54 to 61                 | RETH<sup>3</sup>: Virtual Address                                                                | 8             | Big Endian     | Software configured base address, plus DMA offset       |
| 62 to 65                 | RETH: R_Key                                                                                      | 4             | Big Endian     | Software configured per sensor                          |
| 66 to 69                 | RETH: DMA Length                                                                                 | 4             | Big Endian     | Software configurable length                            |
| 70 to N+69<sup>1</sup>   | Sensor Data                                                                                      | N             | Little Endian  | RAW data received from sensor AXIS port                 |
| N+70 to N+73<sup>1</sup> | iCRC                                                                                             | 4             | Little Endian  |                                                         |

1. N=Configured Ethernet packet length or remaining end of sensor window size.
1. BTH: Base Transport Header
1. RETH: RDMA Extended Transport Header

### Linux UDP based

RoCE (RDMA over Converged Ethernet) is typically associated with specialized hardware
and switch support for true zero-copy, low-latency networking. However, even in
environments where the network switch or hardware does not natively support RoCE, it is
still possible to leverage the RoCE protocol at the software level. In such cases,
standard Linux sockets can be used to transmit and receive Ethernet frames that follow
the RoCE packet structure. The application or driver can simply parse the relevant RoCE
fields—such as the Base Transport Header (BTH), RETH, and other protocol-specific
headers—directly from the UDP payload. This approach allows for interoperability and
testing of RoCE-based data flows without requiring full hardware offload or switch
support, enabling software-based RDMA emulation, protocol development, or integration
with custom FPGA or sensor platforms that generate RoCE-formatted packets. While this
method does not provide the full performance benefits of hardware-accelerated RDMA, it
enables flexible deployment and development in standard Ethernet environments.

### Metadata RoCE Packet

| **Byte Count** | **Description**                                                                                  | **Byte Size** | **Endianness** | **Notes**                                                                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------ | ------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 to 13        | Ethernet Header                                                                                  | 14            | Big Endian     | Software configured Destination MAC address per sensor                                                                                               |
| 14 to 33       | IPv4 Header                                                                                      | 20            | Big Endian     | Software configured Destination IPV4 address per sensor                                                                                              |
| 34 to 41       | UDP Header                                                                                       | 8             | Big Endian     | Software configured Destination UDP Port per sensor                                                                                                  |
| 42             | BTH: Opcode                                                                                      | 1             | Big Endian     | 0x2B                                                                                                                                                 |
| 43             | BTH: Solicited Event (1b), Migration Request (1b), Pad Count (2b), Transport Header Version (4b) | 1             | Big Endian     | 0x0                                                                                                                                                  |
| 44 to 45       | BTH: Partition Key (P_Key)                                                                       | 2             | Big Endian     | 0xFFFF                                                                                                                                               |
| 46             | BTH: FECN (1b), BECN (1b), Reserved (6b)                                                         | 1             | Big Endian     | 0x0                                                                                                                                                  |
| 47 to 49       | BTH: Destination QP                                                                              | 3             | Big Endian     | Software configured per sensor                                                                                                                       |
| 50             | BTH: Acknowledge Request (1b), Reserved (7b)                                                     | 1             | Big Endian     | 0x0                                                                                                                                                  |
| 51 to 53       | BTH: Packet Sequence Number                                                                      | 3             | Big Endian     | Increments per ethernet packet                                                                                                                       |
| 54 to 61       | RETH: Virtual Address                                                                            | 8             | Big Endian     | Software configured base address, plus DMA offset                                                                                                    |
| 62 to 65       | RETH: R_Key                                                                                      | 4             | Big Endian     | Software configured per sensor                                                                                                                       |
| 66 to 69       | RETH: DMA Length                                                                                 | 4             | Big Endian     | Software configurable length                                                                                                                         |
| 70 to 73       | RETH: Write Immediate                                                                            | 4             | Big Endian     | Includes 3 bytes of BTH: Packet Sequence Number, along with 1 byte to identify sensor buffer                                                         |
| 74 to 77       | Flags                                                                                            | 4             | Little Endian  | Bit \[0\]: High: indicates that sensor sent tlast before expected. Low: sensor’s tlast lines up with expected last roce_buf_len. Bits \[31:1\]: RSVD |
| 78 to 81       | Packet Sequence Number                                                                           | 4             | Little Endian  | BTH: Packet Sequence Number                                                                                                                          |
| 82 to 85       | CRC of Sensor Data in whole Window                                                               | 4             | Little Endian  | CRC of the entirety of the raw data window                                                                                                           |
| 86 to 97       | PTP of First Sensor Data of Window                                                               | 12            | Little Endian  | PTP Timestamp of the first cycle of i_sif_axis_tvalid for the frame                                                                                  |
| 98 to 105      | Valid # of Bytes within Buffer                                                                   | 8             | Little Endian  | Number of valid bytes sent by FPGA within the frame                                                                                                  |
| 106 to 109     | Frame Number                                                                                     | 4             | Little Endian  | 16 bit counter that increments every frame                                                                                                           |
| 110 to 121     | PTP of Metadata Packet Formation                                                                 | 12            | Little Endian  | PTP Timestamp of the cycle when the metadata packet was created in fpga                                                                              |
| 122 to 201     | Reserved                                                                                         | 80            | Little Endian  |                                                                                                                                                      |
| 202 to 205     | iCRC                                                                                             | 4             | Little Endian  |                                                                                                                                                      |

The Ethernet, IPv4, and UDP headers abide by the standard Ethernet header format.

The Metadata Packet is sent at the end of frame of data sent by the FPGA. This packet
includes some metadata status about the transmission of the frame. This packet follows
the RoCE protocol, but speciically uses the 0x2B opcode, which generates an interrupt to
the Host.

Note the Frame Check Sequence (FCS) is the last 4 bytes of the ethernet packet and is
part of the ethernet packet length, but it is NOT calculated and added by the Holoscan
Sensor Bridge IP. The ethernet IP is expected to calculate and add the FCS to the end of
the UDP packet.

The Preamble and Start Frame Delimiter (SFD) are also expected to be added by the
Ethernet IP, these fields are not calculated as part of the ethernet packet length.

### Sensor Data Throughput

Because of UDP overhead, the total sensor data throughput is less than the maximum
ethernet bandwidth available.

Using the reference design ethernet packet length of 1486 bytes, the calculated sensor
data throughput is 9.146G, for a 10G ethernet block.

When considering the Preamble, SFD, and Interpacket Gap added per ethernet packet by the
Ethernet IP, the total sensor throughput will be reduced.

Note the sensor data throughput is dependent on the ethernet packet length. Smaller
ethernet packet length will result in smaller sensor data throughput.

### Sensor Data Latency

The Holoscan Sensor Bridge IP will begin packetizing the sensor data into UDP packets
once enough sensor data is received to construct a full UDP packet. The latency between
the last sensor data of the UDP packet received at the input of the Holoscan Sensor
Bridge IP and the first data of the UDP packet transmitted out of the Holoscan Sensor
Bridge IP is 50 host interface clock cycles.

This is assuming there is no other transmit packets in queue.

## Host RX (Host to FPGA)

The Host RX interface is used for the host to communicate with the FPGA. The host can
send BOOTP Response, PING, and Nvidia-defined Ethernet Control Bus (ECB) to read and
write FPGA registers.

The Host RX AXI-Stream ports must be connected to the MAC IP RX AXI-Stream ports.
Holoscan Sensor Bridge IP Host RX expects the Frame Check Sequence (FCS) to be removed.

Host RX AXI-Stream ports comply to the AMBA AXI-Stream protocol. TUSER signal indicates
a packet error (ex: Frame Check Sequence error). TUSER signal is evaluated only when
TLAST is high.

When TUSER is high on the same cycle as TLAST is high, the Holoscan Sensor Bridge IP
will drop the entire ethernet packet.
