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

1. If your current configuration is a Lattice CPNX100-ETH-SENSOR-BRIDGE device currently
   loaded with 2412 or newer firmware, `hololink program` will reprogram it:

   ```none
   hololink program scripts/manifest.yaml
   ```

   If you're using a nonstandard IP address, replace "192.168.200.2" with the address
   you've configured for your device in:

   ```none
   hololink --hololink=192.168.200.2 program scripts/manifest.yaml
   ```

1. If your configuration is a Lattice CPNX100-ETH-SENSOR-BRIDGE device with older (e.g.
   2407\) firmware, `hololink program --force` will reprogram it. Note that this only
   works on units that are available at the default IP address of 192.168.0.2.

   ```none
   hololink program --force scripts/manifest.yaml
   ```

   The way that HSB units are enumerated by the host has changed, so systems producing
   2407 enumeration data are not visible to hosts running 2.0-GA software. Including the
   `--force` option tells the host software to use a hard-coded enumeration structure,
   which enables the software to access the flash memory on the older configuration of
   HSB.

1. When run this way, the manifest file directs the firmware tool to download the FPGA
   BIT files from
   [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_sensor_bridge_fpga_ip)
   with the version validated for use with this software tree. When run on an IGX
   configuration, firmware updates can take up to 5 minutes; when run on AGX, expect a
   run time of as much as 30 minutes. **Do not interrupt the process in the middle.**

1. Once flashing is complete, **power cycle** the device and watch that the sensor
   bridge powers up with 2 green LEDs on

1. Ping the sensor bridge device at '192.168.0.2' and verify a valid ping response
