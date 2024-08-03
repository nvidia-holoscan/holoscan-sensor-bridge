# Release Notes

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

- **Hardware ISP via ArgusIspOp** \[Orin in iGPU mode only\]. Applications can offload
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
