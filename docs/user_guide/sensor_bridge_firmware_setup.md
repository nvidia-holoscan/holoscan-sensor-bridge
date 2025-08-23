## Holoscan Sensor Bridge FPGA firmware update

Holoscan sensor bridge is implemented using a pair of FPGAs, where images for both
components are programmable and should be updated.

1. **Power cycle** the sensor bridge device and make sure 2 green LEDs are on

1. Follow the [setup instructions](setup.md) to build and run the demo container. All
   the following commands are to be run from within the demo container.

1. Check connectivity with the sensor bridge board with the ping command:

   ```
   $ ping 192.168.0.2
   PING 192.168.0.2 (192.168.0.2) 56(84) bytes of data.
   64 bytes from 192.168.0.2: icmp_seq=1 ttl=64 time=0.143 ms
   64 bytes from 192.168.0.2: icmp_seq=2 ttl=64 time=0.098 ms
   64 bytes from 192.168.0.2: icmp_seq=3 ttl=64 time=0.094 ms
   ```

   If your system uses the 192.168.0.0/24 network for another purpose, see instructions
   for [configuring the IP addresses used by the sensor bridge device.](notes.md) After
   reconfiguring addresses appropriately, be sure you can ping your device at the
   address you expect.

1. These instructions assume that your FPGA was loaded with 0x2412 or newer firmware;
   this is the version included with HSB 2.0 (the previous public release). If your
   configuration is older, follow the
   [instructions to update to 2.0 first](https://docs.nvidia.com/holoscan/sensor-bridge/2.0.0/sensor_bridge_firmware_setup.html).
   To see what version of FPGA IP your HSB is configured with:

   ```sh
   hololink-enumerate
   ```

   This command will display data received from HSB's bootp enumeration message, which
   includes the HSB IP version.

1. For Lattice CPNX100-ETH-SENSOR-BRIDGE devices loaded with 0x2412 or newer firmware,
   `program_lattice_cpnx100` will reprogram it:

   ```sh
   program_lattice_cpnx100 scripts/manifest.yaml
   ```

   If you're using a nonstandard IP address, replace "192.168.200.2" with the address
   you've configured for your device in:

   ```sh
   program_lattice_cpnx100 --hololink=192.168.200.2 scripts/manifest.yaml
   ```

1. For programming the Leopard imaging VB1940 Eagle Camera: make sure you cloned and
   built the `holoscan-sensor-bridge repo` (for instructions please see
   [Thor Host Setup](thor-jp7-setup.md) page).

   ```sh
   cd holoscan-sensor-bridge/build
   sudo ./tools/program_leopard_cpnx100/program_leopard_cpnx100 ~/holoscan-sensor-bridge/scripts/manifest_leopard_cpnx100.yaml
   ```

1. When run this way, the manifest file directs the firmware tool to download the FPGA
   BIT files from
   [NVIDIA Artifactory](https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/hsb)
   with the version validated for use with this software tree. When run on an IGX
   configuration, firmware updates can take up to 5 minutes; when run on AGX, expect a
   run time of as much as 30 minutes. **Do not interrupt the process in the middle.**

1. For the Lattice CPNX100-ETH-SENSOR-BRIDGE, once flashing is complete, **power cycle**
   the device and watch that the sensor bridge powers up with 2 green LEDs on

1. Ping the sensor bridge device at its IP address (e.g. 192.168.0.2) and verify a valid
   ping response
