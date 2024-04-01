# Host Setup

After the [Holoscan sensor bridge board is set up](sensor_bridge_hardware_setup.md),
follow the directions on the appropriate page below to configure your host system.

- [IGX running IGX OS 1.0 DP, with CX7 SmartNIC](igx_baseos_roce_deployment.md)
- [Jetson AGX Orin, L4T JP6.0, with Linux Sockets](concord_l4t_linux_sock_deployment.md)

## Holoscan sensor bridge software prerequisites

Configure a few prerequisites in order to access and run the sensor bridge container:

1. Grant your user permission to the docker subsystem:

   ```none
   $ sudo usermod -aG docker $USER
   ```

   Reboot the computer to activate this setting.

1. Log in to Nvidia GPU Cloud (NGC) with your developer account:

   - If you don't have a developer account for NGC please register at
     [https://catalog.ngc.nvidia.com/](https://catalog.ngc.nvidia.com/)

   - Create an API key for your account:
     [https://ngc.nvidia.com/setup/api-key](https://ngc.nvidia.com/setup/api-key)

   - Use your API key to log in to nvcr.io:

     ```none
     $ docker login nvcr.io
     Username: $oauthtoken
     Password: <Your token key to NGC>
     WARNING! Your password will be stored unencrypted in /home/<user>/.docker/config.json.
     Configure a credential helper to remove this warning. See
     https://docs.docker.com/engine/reference/commandline/login/#credentials-store

     Login Succeeded
     ```

## Running Holoscan Sensor Bridge demos from source

Holoscan sensor bridge host software includes instructions for building a demo
container. This container is used to run all holoscan tests and examples.

### Build the demo container

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
     [IGX OS 1.0 DP](https://developer.nvidia.com/igx-downloads) with dGPU).
   - `--igpu` is appropriate for systems running on a system with iGPU (e.g. AGX or IGX
     without a dGPU). This requires an OS installed with iGPU support (e.g. for AGX:
     JetPack 6.0; for IGX: IGX OS with iGPU configuration).

### Run the demo container

To run the sensor bridge demonstration container, from a terminal in the GUI,

```none
xhost +
sh docker/demo.sh
```

This brings you to a shell prompt inside the Holoscan sensor bridge demo container.
(Note that iGPU configurations, when starting the demo container, will display the
message "Failed to detect NVIDIA driver version": this can be ignored.) Now you're ready
to run sensor bridge applcations.

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
pytest --udp-server
```

For AGX configurations, only one camera is supported, so only
[SFP+ 0](sensor_bridge_hardware_setup.md#connecting-holoscan-sensor-bridge-to-the-host)
is to be connected. Run the device test on AGX this way:

```none
pytest --udp-server --unaccelerated-only
```

If things are not working as expected, check the
[troubleshooting page](troubleshooting.md).
