# Release Notes

## 2.0-GA, January 2025

### Dependencies

- IGX: [IGX-SW 1.1 Production Release](https://developer.nvidia.com/igx-downloads)
- AGX: Use [SDK Manager](https://developer.nvidia.com/sdk-manager) to set up JetPack 6.0
  release 2. Note that JetPack 6.1 is not yet supported for HSB.
- Holoscan Sensor Bridge, 10G; FPGA v2412

Be sure and follow the installation instructions included with the release. To generate
documentation, in the host system, run `sh docs/make_docs.sh`, then use your browser to
look at `docs/user_guide/_build/html/index.html`.

### Updates from 1.1-GA

- **HSB 2.0-GA relies on FPGA IP version 2412.** Check the user guide for instructions
  on how to update your configuration. Note that the enumeration data has changed, so
  pre 2.0-GA software will not enumerate boards publishing 1.1 (or earlier) enumeration
  data; and likewise, 1.1 and earlier software will not find the newer 2.0 configuration
  boards. For Lattice-CLNX100-ETH-SENSOR-BRIDGE devices, be sure and include the
  "--force" option when updating the HSB firmware; this way the software uses hardcoded
  enumeration data in the software tree instead of relying on that from the device
  itself. See [the firmware download instructions](sensor_bridge_firmware_setup.md) for
  more details. If you need to revert your FPGA back to the 2407 version, check the FAQ
  below.

- **HSB is updated to work with Holoscan SDK 2.7.** Some older APIs, specifically in the
  [C++ fmt tool](https://github.com/fmtlib/fmt) have been deprecated, so minor code
  adjustments have been applied to keep HSB host code up to date.

- **New HSB features for safety and reliability,** including CRCs, control plane
  sequence number checking, and additional timestamps are included. Timestamps included
  capture the PTP time when the first data in the received frame is observed by the FPGA
  IP block and the time after the last data in the frame is sent. With ConnectX based
  host systems, which support hardware PTP synchronization, these timestamps are within
  a microsecond of the host time, and can be used to accurately measure latency through
  the pipeline. These metadata values are available to pipeline operators via the
  [HSDK application metadata API](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_app.html#dynamic-application-metadata).
  See the user guide for more details. Sequence number checking is enabled for control
  plane transactions, and can provide protection against interaction from several hosts
  to the same HSB unit. The overall CRC of the received data frame is also included, in
  a later release, a high-performance CUDA based CRC checker will be demonstrated
  showing frame-rate CRC validation of ingress data.

- **Multiple sensors over a single network port.** APIs are added to allow applications
  to configure multiple sensors to use the same network port. In the examples directory,
  `single_network_stereo_imx274_player.py` demonstrates how to configure both cameras in
  an IMX274 stereo pair to transmit 1080p video using a single network port. Note that
  4k video streams require about 6.5Gbps each, so using both cameras in this mode over a
  single network port is not supported. See the user guide for more details.

- **Performance and latency measurement tools.** The timestamps included with safety and
  reliability features can be used to accurately measure latency, from the time that
  data arrives to the FPGA IP block, all the way through the end of the pipeline (e.g.
  following visualization). See `examples/imx274_transfer_latency.py` for an example.
  See [latency.md](latency.md) for more details on latency measurement.

- **GammaCorrectionOp is removed.** HSDK 2.3 added support for
  [sRGB space](https://en.wikipedia.org/wiki/SRGB), providing an optimized path
  including Gamma correction in the visualizer. By removing HSB's naive gamma correction
  and using the visualizer instead, pipeline latency is reduced by .5ms. For
  applications that used GammaCorrectionOp, just remove that operator from the pipeline
  and include `framebuffer_srgb=True` in the constructor parameter list for HolovizOp.

- **Support for IMX477 cameras via Microchip MPF200-ETH-SENSOR-BRIDGE.**

### FAQ

- If your application, running in the demo container, halts with a "Failed to initialize
  glfw" message, make sure to grant the application permission to connect with the
  display via the "xhost +" command. This command is not remembered across reboots.

- Reverting from FPGA 2412 to 2407 on Lattice HSB units. If you need to revert a Lattice
  HSB unit from 2412 back to 2407, use the 2.0-GA tree and program with the 2407
  manifest file. From within the demo container:

  ```sh
  hololink program scripts/manifest-2407.yaml
  ```

  After programming and power cycling, the board will no longer be visible to the 2.0-GA
  version of HSB host code. At this time you can go back to using the 1.1-GA release to
  work with the board. The 1.1-GA software will not be able to enumerate boards running
  the 2412 configuration; the newer tree must be used to write the older firmware.

- HSB network receiver operators use
  [APIs provided by the Holoscan SDK](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_app.html#dynamic-application-metadata)
  to share timestamps with later operators in the pipeline. Be sure and call the
  application (C++) `is_metadata_enabled(true)` method or (python)
  `is_metadata_enabled = True` at initialization time; otherwise each operator will only
  see an empty metadata structure. In your operator's `compute` method, if you add
  additional items to the pipeline metadata, be sure and add that metadata before
  calling `(output).emit`. If you have a pipeline that merges two paths, and experience
  a `runtime_error` exception when it fails to merge the metadata from those paths, see
  [the page on Metadata update policies](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_app.html#metadata-update-policies)
  for information on how to manage this.

- It is possible to overrun the bandwidth available on the ethernet, particularly when
  using multiple sensors over a single network connection. For example, a 4k, 60FPS,
  RAW10 video stream requires about 6.5Gbps, and a stereo pair configured this way would
  require something like 13Gbps--which far exceeds a single 10Gbps network port. In this
  case, HSB will drop data, probably within a single network message. When operated
  outside of specification this way, reconstruction of the original sensor data is not
  possible. The software has no concept of available bandwidth, so it is up to the
  developer to ensure that bandwidth limits are not exceeded.

- AGX on-board ethernet supports hardware PTP synchronization, which can be enabled by
  following the same directions given to set up PTP for IGX. Specifically, with the
  appropriate network device name in `$EN0` (e.g. `eth0`), just follow the instructions
  on the host setup page for setting up PTP on IGX.

### Known Anomalies

- Orin AGX running on JetPack 6.1 shows very slow network behavior. Investigation of
  this is underway; for now, on Orin AGX, only JetPack 6.0 r2 is supported.

- Following software-commanded reset, HSB sometimes observes a sequence number or
  control plane transaction failure. When the HSB is commanded to reset, the host system
  observes a loss of network connectivity and may take some time before steady
  communication is available. In some specific Orin AGX systems, `dmesg` shows this
  renegotiation can take more than 60 seconds. During this time, HSB software attempts
  to read the FPGA version ID, and can time out. Investigation of this is underway; for
  now, systems with this behavior can be worked around by putting a 10G ethernet switch
  between the HSB and the host system.

- Orin AGX systems, running with stereo sensor feeds on the same network port, using
  either the multithreaded or event based schedulers, have unreliable operation. For
  now, AGX systems with HSB are only supported with the default (greedy) scheduler.

## 1.1.0-GA, August 2024

### Dependencies

- IGX: [IGX-SW 1.0 Production Release](https://developer.nvidia.com/igx-downloads)
- AGX: Use [SDK Manager](https://developer.nvidia.com/sdk-manager) to set up JetPack 6.0
  release 2.
- Holoscan Sensor Bridge, 10G; FPGA v2407

Be sure and follow the installation instructions included with the release. To generate
documentation, in the host system, run `sh docs/make_docs.sh`, then use your browser to
look at `docs/user_guide/_build/html/index.html`.

### Updates from 1.0-GA

- **Most HSB framework components are now implemented in C++**, supporting applications
  written in C++. For an example HSB application written in C++, see
  `examples/imx274_player.cpp`. Changes that affect application code in both C++ and
  Python include

  - HololinkEnumerator is renamed to Enumerator
  - HololinkDataChannel is now DataChannel
  - RoceReceiverOperator is now RoceReceiverOp

  Functionally, use of these objects is the same as 1.0-GA. Note that sensor drivers
  written in Python are still supported.

- **HSB is updated to work with Holoscan SDK 2.3**. Your deployment system must be
  configured with a compatible software environment (e.g. JetPack 6 for AGX
  configurations).

- There are some small **changes to the host setup instructions**. Changes focused on
  updates to network device names and performance on AGX configurations.

- **Hardware ISP via ArgusIspOp** [Orin in iGPU mode only]. Applications can offload
  image signal processing by using the capabilities built in to the NV ISP device
  present in Orin systems running with iGPU (AGX or IGX without a dGPU). Support is
  provided for 1080p images; contact NVIDIA to get updated libraries with support for 4K
  images. For an example pipeline using this feature, see
  `examples/linux_hwisp_player.py`.

- APIs are available for accessing **GPIOs** on headers on Holoscan sensor bridge. See
  the user guide for details.

### Known Anomalies

- AGX Network linkup problems on "hololink reset".

  In some setups, calls to hololink reset result in a series of messages output to the
  AGX kernel log that look like this:

  ```none
  [ 15.587973] nvethernet 6800000.ethernet: [xpcs_lane_bring_up][477][type:0x4][loga-0x0] PCS block lock SUCCESS
  [ 15.588001] nvethernet 6800000.ethernet eth0: Link is Up - 10Gbps/Full - flow control off
  [ 16.099966] nvethernet 6800000.ethernet: [xpcs_lane_bring_up][477][type:0x4][loga-0x0] PCS block lock SUCCESS
  [ 16.099987] nvethernet 6800000.ethernet eth0: Link is Up - 10Gbps/Full - flow control off
  ```

  While the host network interface is resynchronizing, communication with HSB will be
  unreliable. This resynchronization is complete when these messages stop being added to
  the kernel log.

- iGPU configurations: "Failed to detect NVIDIA driver version" displayed when the
  container is started.

  This message can be ignored. The Holoscan SDK container initialization includes a
  check for the dGPU driver version; in the iGPU configuration, this driver isn't
  loaded, resulting in this message. iGPU operation is unaffected by this and will
  operate as expected.

- AGX running examples/linux_body_pose_estimation.py --camera-mode=0, the first time,
  may cause the video to hang.

  The first time the body-post-estimation app is run, the .onnx file is converted to a
  TRT engine file, which is a step than can take several minutes. Subsequent runs of the
  body pose estimation app will skip the conversion and just load this engine file
  directly. During the conversion, when high bandwidth is in use on the network (via
  "--camera-mode=0"), the kernel stops delivering received UDP messages to the
  application, resulting in no video being displayed. Later runs the same program, after
  the conversion is complete, run as expected.

- HSB does not forward data received on the MIPI interface, resulting in an "Ingress
  frame timeout; ignoring." message.

  A bug in the FPGA MIPI receiver block IP causes data to be dropped before being
  delivered to the FPGA's UDP packetizer; resulting in no sensor data being delivered to
  the host. If you've commanded a camera to send image data, but no data is observed and
  the timeout message is displayed, you can verify that this is the cause by issuing
  these commands within the HSB demo container:

  ```
  hololink read_uint32 0x50000000   # for the first camera
  hololink read_uint32 0x60000000   # for the second camera
  ```

  If a camera is configured to issue data, but a 0 appears in this memory location, then
  this is an indication that the receiver is in this stuck state. `hololink.reset()` is
  able to clear this condition.

- PTP timestamps, published by HSB, aren't synchronized with the host time.

  The
  [user guide hardware setup instructions](https://docs.nvidia.com/holoscan/sensor-bridge/1.1.0/setup.html)
  show how to configure the `phc2sys` tool, which ensures that the NIC time-of-day
  clock, which is published with PTP network messages, is synchronized with the host
  real-time clock. As written, the setup instructions rely on the default ntpdate
  configuration to initialize the system clock to the rest of the world. As written, the
  `phc2sys` startup doesn't properly wait for `ntpdate` to complete, so that when
  `ntpdate` does finish, `phc2sys` sees a large jump in the system time. Because
  `phc2sys` slowly adjusts the NIC clock, the time published by `ptp4l` will not be
  synchronized with the system clock, and could take a very long time to do so. To
  verify you're in this condition, observe the output in the "sys offset" column from
  the command `systemctl status phc2sys-*.service`; large absolute values are an
  indication of this condition. To work around this, run
  `sudo systemctl restart phc2sys-*.service` after ntpdate is synchronized. There is
  some anecdotal evidence that adding "-w" to the `phc2sys` command line may fix this
  problem but the documentation for this option doesn't address the configuration in use
  here.
