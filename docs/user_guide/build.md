# Build and Test the Holoscan Sensor Bridge demo container

## Building the Holoscan Sensor Bridge container

Holoscan sensor bridge host software includes instructions for building a demo
container. This container is used to run all holoscan tests and examples.

1. Fetch the sensor bridge source code from
   [https://github.com/nvidia-holoscan/holoscan-sensor-bridge](https://github.com/nvidia-holoscan/holoscan-sensor-bridge)

   ```none
   $ git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge
   ```

1. Build the sensor bridge demonstration container. For systems with dGPU,

   ```none
   $ cd holoscan-sensor-bridge
   $ sh docker/build.sh --dgpu
   ```

   For systems with iGPU,

   ```none
   $ cd holoscan-sensor-bridge
   $ sh docker/build.sh --igpu
   ```

   Notes:

   - `--dgpu` requires a system with a dGPU installed (e.g. IGX with A6000 dGPU) and an
     OS installed with appropriate dGPU support (e.g.
     [IGX OS 1.1.2 Production Release](https://developer.nvidia.com/igx-downloads) with
     dGPU).
   - `--igpu` is appropriate for systems running on a system with iGPU (e.g. AGX or IGX
     without a dGPU). This requires an OS installed with iGPU support (e.g. for AGX:
     JetPack 6.2.1; for IGX: IGX OS with iGPU configuration).

## Run tests in the demo container

To run the sensor bridge demonstration container, from a terminal in the GUI,

```none
xhost +
sh docker/demo.sh
```

This brings you to a shell prompt inside the Holoscan sensor bridge demo container.
(Note that iGPU configurations, when starting the demo container, will display the
message "Failed to detect NVIDIA driver version": this can be ignored.) Now you're ready
to run sensor bridge applications.

## Holoscan sensor bridge software loopback tests

Sensor bridge host software includes a test fixture that runs in loopback mode, where no
sensor bridge equipment is necessary. This test works by generating UDP messages and
sending them over the Linux loopback interface.

In the shell in the demo container:

```none
pytest
```

Note that the test fixture intentionally introduces errors into the software stack. As
long as pytest indicates that all test have passed, any error messages published by
individual tests can be ignored.

For systems with a sensor bridge device and IMX274, the test fixture can execute
additional tests that prove that the device and network connections are working as
expected.

First, ensure that the
[sensor bridge firmware is up to date](sensor_bridge_firmware_setup.md).

For IGX configurations,
[connect both SFP+ connections on the sensor bridge device to the QSFP connectors](sensor_bridge_hardware_setup.md#connecting-holoscan-sensor-bridge-to-the-host),
then

```none
sh ./test-igx-cpnx100-imx274.sh
```

For AGX configurations, only one camera is supported, so only
[SFP+ 0](sensor_bridge_hardware_setup.md#connecting-holoscan-sensor-bridge-to-the-host)
is to be connected. Run the device test on AGX this way:

```none
sh ./test-agx-cpnx100-imx274.sh
```

If things are not working as expected, check the
[troubleshooting page](troubleshooting.md).
