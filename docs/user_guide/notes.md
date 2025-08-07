# Important notes

## RoceReceiverOp

Receiver operators, including RoceReceiverOp and LinuxReceiverOperator specifications
include:

- Always return the most recently received video frame. This operator will never return
  a video frame older than 1 frame time. If the pipeline is busy and is unavailable to
  accept incoming video frames, older frame data is replaced with newer frame data. If
  the frame request occurs while another frame is being received, this operator returns
  the last completed video frame.
- Never return the same video frame twice. If the pipeline is faster than a video
  reception time, the pipeline will block until the next incoming frame is complete.
- PTP timestamps, sent by HSB on both the start-of-frame and the end-of-frame condition,
  are available by looking at received metadata. When PTP is properly configured, this
  provides a reliable mechanism for determining the actual latency for processing or
  this displaying this data. Note that PTP is only supported for IGX configurations.

RoceReceiverOp uses RDMA with special caveats that are likely to change in future
versions:

- There is no protection against rewriting the video buffer while the pipeline is using
  it. The current implementation allocates enough GPU memory to receive two video
  frames, and the transmitter is configured to alternate between these two buffers.
  Video pipelines usually start by copying the data from the receiver buffer into
  another region of memory (via CsiToBayerOp)--this minimizes the time during which an
  overwrite hazard can occur--but scheduling these operations is dependent on CPU
  availability. CudaCRCOp can be used to verify that received data has not been
  corrupted; as long as it computes the same value that's found in the receiver
  metadata, then you've proven that the memory contents have not been updated.

## Holoscan sensor bridge IP address configuration

For systems where the 192.168.0.0/24 network is unavailable, you can use the
`hololink-set-ip` command to reconfigure sensor bridge IP addresses. Sensor bridge
devices transmit enumeration messages based on the BOOTP protocol; the host can reply
with a request to set the IP address to a specific value. Programs in the Holoscan
sensor bridge host software all accept a `--hololink=<IP-address>` command line
parameter to look for the device with the given IP address instead of the default
192.168.0.2.

First, locate the MAC ID and local interface that the sensor bridge port is connected to
with the `hololink-enumerate` command. Within the
[demo container](setup.html#running-holoscan-sensor-bridge-demos-from-source):

```none
$ hololink-enumerate
mac_id=48:B0:2D:EE:84:DA hsb_ip_version=0x2507 fpga_crc=0xffff ip_address=192.168.0.2 fpga_uuid=889b7ce3-65a5-4247-8b05-4ff1904c3359 serial_number=10030032828115 interface=enP5p3s0f0np0 board=hololink-lite
mac_id=48:B0:2D:EE:84:DB hsb_ip_version=0x2507 fpga_crc=0xffff ip_address=192.168.0.3 fpga_uuid=889b7ce3-65a5-4247-8b05-4ff1904c3359 serial_number=10030032828115 interface=enP5p3s0f1np1 board=hololink-lite
mac_id=48:B0:2D:EE:84:DA hsb_ip_version=0x2507 fpga_crc=0xffff ip_address=192.168.0.2 fpga_uuid=889b7ce3-65a5-4247-8b05-4ff1904c3359 serial_number=10030032828115 interface=enP5p3s0f0np0 board=hololink-lite
mac_id=48:B0:2D:EE:84:DB hsb_ip_version=0x2507 fpga_crc=0xffff ip_address=192.168.0.3 fpga_uuid=889b7ce3-65a5-4247-8b05-4ff1904c3359 serial_number=10030032828115 interface=enP5p3s0f1np1 board=hololink-lite
```

This configuration has two network ports, with MAC ID 48:B0:2D:EE:84:DA connected to the
local enP5p3s0f0np0 device; 48:B0:2D:EE:84:DB is connected to enP5p3s0f1np1. These
connect to the same sensor bridge device, as shown by the common serial number. Note
that these messages will be observed by the local system regardless of the IP addresses
of local network devices.

For our example, we'll set up this configuration:

- Host eth0 will be configured to 192.168.200.101/24
- Host eth1 will be configured to 192.168.200.102/24
- Sensor bridge port 48:B0:2D:EE:84:DA will use IP address 192.168.200.2
- Sensor bridge port 48:B0:2D:EE:84:DB will use IP address 192.168.200.3
- Explicit routes are added with enP5p3s0f0np0 to 192.168.200.2 and enP5p3s0f1np1 to
  192.168.200.3

First, follow the [setup instructions](setup.md) to configure the host IP addresses and
routes for your expected configuration-- replace the references to 192.168.0.x with the
addresses you expect to use. Then, use the `hololink-set-ip` command to reconfigure the
sensor bridge device ports. To configure the target IP address examples above, in the
demo container,

```none
$ hololink-set-ip 48:B0:2D:EE:84:DA 192.168.200.2 48:B0:2D:EE:84:DB 192.168.200.3
Setting the following addresses:
  48:B0:2D:EE:84:DB to 192.168.200.3
  48:B0:2D:EE:84:DA to 192.168.200.2
Running in daemon mode; run with '--one-time' to exit after configuration.
Updating mac=48:B0:2D:EE:84:DA from ip=192.168.0.2 to new_ip=192.168.200.2
Updating mac=48:B0:2D:EE:84:DB from ip=192.168.0.3 to new_ip=192.168.200.3
Found mac_id=48:B0:2D:EE:84:DA using peer_ip=192.168.200.2
Found mac_id=48:B0:2D:EE:84:DB using peer_ip=192.168.200.3
```

`hololink-set-ip` accepts a list of MAC ID and IP address pairs. Because the sensor
bridge IP address configuration is not stored in nonvolatile memory, `hololink-set-ip`
runs as a daemon, and must be running whenever this configuration is desired. When
`hololink-set-ip` sees an enumeration message from a device with a listed MAC ID but a
different IP address, it will reply with a request to set the desired IP address-- this
accommodates IP address reverting on power cycle or reset. Following this, pinging the
target IP address now works:

```none
$ ping 192.168.200.2
PING 192.168.200.2 (192.168.200.2) 56(84) bytes of data.
64 bytes from 192.168.200.2: icmp_seq=1 ttl=64 time=0.215 ms
64 bytes from 192.168.200.2: icmp_seq=2 ttl=64 time=0.352 ms
64 bytes from 192.168.200.2: icmp_seq=3 ttl=64 time=0.365 ms
^C
...
$ ping 192.168.200.3
PING 192.168.200.3 (192.168.200.3) 56(84) bytes of data.
64 bytes from 192.168.200.3: icmp_seq=1 ttl=64 time=0.218 ms
64 bytes from 192.168.200.3: icmp_seq=2 ttl=64 time=0.323 ms
64 bytes from 192.168.200.3: icmp_seq=3 ttl=64 time=0.281 ms
^C
```

Note that the routing must be set up correctly for the ping commands to work as
expected; for examples on how to permanently configure routes, see the
[Host setup](setup.md) page.
[Here is more specific information on sensor bridge IP address reconfiguration](architecture.md#DataChannel-enumeration-and-ip-address-configuration).

## Network cables and adapters

Please use one of the following cables to connect the sensor bridge SPF+ to the **IGX
devkit** QSFP port:

1. [Fiber optic cable](https://www.amazon.com/FLYPROFiber-10ft-4pack-Fiber-Length-Options/dp/B089K1J5GG)
   with
   [optick to SFP+](https://www.amazon.com/Multi-Mode-Transceiver-10GBASE-SR-SFP-10G-SR-Supermicro/dp/B01N1H1Z2F?th=1)
   and
   [SFP+ to QSFP](https://www.amazon.com/Converter-CVR-QSFP-SFP10G-Mellanox-MAM1Q00A-QSA-All-Metal/dp/B082V1TLHH?th=1)
   adapters
1. Copper
   [SFP+ to SFP+](https://www.amazon.com/10G-SFP-DAC-Cable-SFP-H10GB-CU2M/dp/B00U8BL09Q/ref=sr_1_4?dib=eyJ2IjoiMSJ9.Cf-3YlRVvPfOvuT9WkBirl136H7mBcMmsk3GZo6CrIgg6twUeFibkg2B33myyuT9gB0QLyjJtTm3HKhnEhweaz73ZteuRh32EQoRms2iNgX8I3HM6_CTTqjm7Pt6x1HMSCNBpbtGP2UjMWH1_LROIHSpFF3SEf53-aG4o0kkVvDVVWeTVvr-bQHiGkMqKCv9EDZCMso3MU8BX9zT_-sZOHfCBMpOPHcU_-uPGAdl47o.SFlNj6GVEC-arkVjg8PX91PbzSpRKD5cWOWvg3hEyPI&dib_tag=se&keywords=sfp%2B%2Bto%2Bsfp%2B%2Bcable&qid=1708561658&sr=8-4&th=1)
   with an additional
   [SFP+ to QSFP](https://www.amazon.com/Converter-CVR-QSFP-SFP10G-Mellanox-MAM1Q00A-QSA-All-Metal/dp/B082V1TLHH?th=1)
   adapter

Please use the following cables to connect the sensor bridge SFP+ to the **Jetson AGX
Orin devkit** 10G Ethernet port:

1. [RJ45 Ethernet cable](https://www.amazon.com/Amazon-Basics-Ethernet-High-Speed-Snagless/dp/B089MGH8W3/ref=sr_1_5?crid=1KJ1COP3OKCV7&dib=eyJ2IjoiMSJ9.awXrUbdN3xPxSw8yHRVmtqoUhU1UJEBgQ7Bt3D1N-o4R66qUmZdXTiq-3z8avmIBca3drzlYJhDUl2a8emDyXxFtjeYRRH6OgEOfqtc1w9-y1SPhRXhFWKwLnC3aFhzNs6uT3x_OYvZRxUgOiadVqR8GAUdJiHgH-2SyzwUS8bM_CMRTnRdrU6y-d59mmKSet0zarNIM5FuTMVdwoBJIs_DecT4gyQQA4UnlgvC9VsXYpIxPlFkLnJGnllhPNGDUtysKngtLL1_WyhiUI5y0Q2lcAqDyHlzCCPCPRmm6Hpg.-xBCBUe3Gj5rNmopY7uoCfHAf0ybNBqeWSgi1ARCvW8&dib_tag=se&keywords=rj45%2Bethernet%2Bcable&qid=1708561933&s=electronics&sprefix=rj45%2B%2Celectronics%2C172&sr=1-5&th=1)
   with
   [RJ45 to SFP+ adapter](https://www.amazon.com/10Gtek-SFP-10G-T-S-Compatible-10GBase-T-Transceiver/dp/B01KFBFL16?pd_rd_w=JvDu0&content-id=amzn1.sym.80b2efcb-1985-4e3a-b8e5-050c8b58b7cf&pf_rd_p=80b2efcb-1985-4e3a-b8e5-050c8b58b7cf&pf_rd_r=0ZFMCGJQJSRGSKQ4G71B&pd_rd_wg=fWzpt&pd_rd_r=d37211e0-40ab-4fe9-807d-0f62cad47c18&pd_rd_i=B01KFBFL16&ref_=pd_bap_d_grid_rp_0_4_i&th=1)

All Cables and adapters are available for purchase online - please note that the links
above are **only for demonstration purposes** and should not be considered as a purchase
recommendation.
