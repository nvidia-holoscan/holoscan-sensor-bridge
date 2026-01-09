# HSB Emulator

- [Overview](#overview)
  - [Features](#features)
  - [Environment Compatibility](#environment-compatibility)
- [Building HSB Emulator](#building-hsb-emulator)
  - [Configuring the Emulator-only build](#configuring-the-emulator-only-build)
  - [Build Instructions](#build-instructions)
- [Examples](#examples)
- [Testing](#testing)
- [Developing with HSB Emulator](#developing-with-hsb-emulator)
  - [Primary Classes](#primary-classes)
    - [HSBEmulator](doxygen_HSBEmulator)
    - [DataPlane](doxygen_DataPlane)
    - [Transmitters](doxygen_BaseTransmitter)
    - [I2CPeripheral](doxygen_I2CPeripheral)
      - [Vb1940Emulator](doxygen_Vb1940Emulator)
  - [Structure of an HSB Emulator application](#structure-of-an-hsb-emulator-application)
- [Troubleshooting](#troubleshooting)
- [Clean-Up](#clean-up)

## Overview

The holoscan-sensor-bridge (HSB) repository includes a minimally featured emulation of
the HSB framework's IP for use in testing and development within a hosted environment -
the "HSB Emulator". Though it is called like a single object, it is a collection of
objects that can be used to configure a real endpoint and data source for HSB host-side
applications. This allows for simulation, testing, or development of a host application
when the hardware is not fully developed, in testing, or a full software-in-loop/CI
environment.

An outline of the typical communication pipelines between and HSB Host application and
an HSB Emulator application is shown in the diagram below. This graphic is but one
particular topology and user client code may piece together the components so that an
HSB Emulator application may meet the needs of their particular Host application(s)'s
code. Note also that the two applications may exist either on different device, within
the same device in different processes, or even within different threads of the same
process (through transport options may be limited). The details of the specific objects
in the diagram are described in
[Developing with HSB Emulator](#developing-with-hsb-emulator)

<img src="HSBEmulatorApplicationCommunicationFlow.png" alt="Communication Pipelines between HSB Host and Emulator Applications" width="100%"/>

The HSB Emulator as provided is *not* meant to be a full implementation of an MCU or
FPGA HSB device nor provide the full performance benefits of those implementations. Many
features of the HSB IP are not fully implemented in that the existing emulator will
accept and reply to requests, e.g. ptp, but may be "no-ops" internally so that apps that
currently depend on that interface being present can proceed but leave the
implementation detail to applications or expected behavior. The features that are
implemented are only the ones necessary for a host application to configure and accept
sensor data through the HSB transport pipelines.

### Features

The features that are supported in provided code are

1. Sensor agnostic
   - HSB Emulator itself has no built-in knowledge of any given sensor.
   - Sensors and drivers that do not depend on state change queries, e.g. the HSB
     example "stateless" imx drivers, are compatible by default
1. `LinuxTransmitter` for RoCEv2 UDP-based transport
   - Name is to be consistent with the rest of HSB where the implementation is via Linux
     interface access and sockets
1. `COETransmitter` for IEEE 1722B link layer transport (Camera-over-Ethernet, CoE)
1. `BaseTransmitter` and `DataPlane` interfaces for adding experimental transport
   protocols
1. Underlying C++ implementation + Python API
1. CUDA buffer support
   - `DLPack` array interface format - compatible with contiguous arrays from any Python
     package that supports `DLPack`
   - sensor data buffers may be provided on host or CUDA GPU device
1. Configuration of "stateful" sensors via an `I2CPeripheral` interface
   - SPI and GPIO are not fully implemented
1. Isolated build from host
   - HSB Emulator may be built without linking to either HSB or Holoscan SDK to minimize
     environmental and package dependencies
1. Loopback testing ability
   - The HSB Emulator can be used to test both accelerated and unaccelerated transport
     on the same device the host is running on though for accelerated transport, e.g.
     `RoceReceiverOp` and `SIPLCaptureOp`, a physical connection and network isolation
     would be needed

### Limitations

The HSB Emulator is meant to be as hardware agnostic as possible and may run on the same
device as host applications where it cannot change many settings. To that end, it is not
fully emulating any particular FPGA or MCU implementation of HSB IP and therefore some
features are not or will not be implemented.

1. Only one instance of HSBEmulator is likely to work on any single device:
   - This is a current limitation in the APIs to be consistent with real HSB IP. While
     only one HSBEmulator may run, the number of sensor `DataPlane`s that may attach to
     the instance is limited only by the number of IP addresses that can be assigned and
     configurable sensor interfaces (SIFs) for the emulated device (\<= 32).
   - Running in isolated network namespaces will free up this restriction for
     connections other than a SW loopback, but requires application & system maintenance
1. The HSB Emulator may not be reprogrammed via the host side IP tools like a real HSB:
   - set_ip
   - p2p
   - reset
   - any reprogramming

### Environment Compatibility

Python and C++ APIs are provided and designed to work in any recent Linux environment -
x86-64 or arm64. They can be extended to other operating systems or environments (e.g.
32-bit) by suitably changing/re-implementing `src/hololink/emulation/net.cpp` where most
of the host-specific functionality has been isolated, though POSIX APIs have been
generally been assumed elsewhere. Other than OS facilities, requirements have been kept
to a minimum and are described in more detail in the
[Building HSB Emulator](#building-hsb-emulator) section. HSB Emulator has been tested
with host applications in the following environments:

1. x86 Linux Desktop - Ubuntu 22.04+ - Emulator applications only
1. Jetson Orin Nano Developer Kit - Jetpack 6.1+ - Emulator applications only
1. AGX Orin Developer Kit - Jetpack 6.1+ - Host or Emulator applications
1. IGX Orin Developer Kit - IGX BaseOS 1.0+ - Host or Emulator applications
1. AGX Thor Developer Kit - Jetpack 7.1+ - Host or Emulator applications

The [Examples](#examples) provided are tested mostly with imx274- (where applicable) and
the vb1940-based HSB camera applications. The `serve_linux_file` and `serve_coe_file`
have been used with the HSB-provided imx477 and imx715 sensor drivers as well.

## Building HSB Emulator

There are 2 ways to build the HSB Emulator

- Full
  - This is a normal build of the HSB repository that will build the emulator and
    include it within the normal hololink Python module
- Emulator-only
  - Minimal build dependencies. An optional hololink Python module is provided within a
    virtual environment for the Python API

### Build instructions

For all build types, clone and enter the repository

```
git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
cd holoscan-sensor-bridge
```

::::{tab-set}

:::{tab-item} Full

Follow the appropriate [](HostSetupTarget) instructions first.

```
mkdir build
cmake -S . -B build
cmake --build build -j
```

:::

:::{tab-item} Emulator-only

Install build dependencies. It's assumed an appropriate
[CUDA toolkit](https://developer.nvidia.com/cuda-downloads) has been installed for the
target system. Follow platform instructions to ensure version, gpu, and driver
compatibility:

```
sudo apt-get update
sudo apt-get install -y cmake zlib1g-dev 
```

If building the Python bindings as well

```
sudo apt-get install -y python3-pybind11 python3-venv python3-pip

```

Additional dependencies for the Python bindings are required for the provided examples
(`numpy` v1.23+ and `cupy`), but the virtual environment that is created will
automatically populate them. The versions required are not necessarily compatible with
the system Python, especially on Ubuntu 22.04 and comparable or earlier

```
mkdir build
cmake -S src/hololink/emulation -B build
cmake --build build -j
```

:::

::::

### Configuring the Emulator-only build

Several `cmake` flags are available to configure the "Emulator-only" build if
prerequisites are not met or not in expected locations

| dependency                                                                  | cmake configuration                                      | default setting           |
| --------------------------------------------------------------------------- | -------------------------------------------------------- | ------------------------- |
| Python unavailable/not needed                                               | `-DHSB_EMULATOR_BUILD_PYTHON=OFF`                        | `ON`                      |
| Python virtual environment location <br> (ignored if Python build is `OFF`) | `-DHSB_EMULATOR_PYTHON_VENV=path/to/virtual_environment` | `${CMAKE_BINARY_DIR}/env` |

Note for cmake version < 3.24 (`apt` default available in Ubuntu 22.04 or earlier),
cmake does not allow setting a native architecture and a default
`CMAKE_CUDA_ARCHITECTURES` of 80 86 87 compute capability is given to ensure target Orin
platform compatibility. For cmake >= 3.24, native is selected by default.

## Examples

To facilitate testing and development of HSB Emulator applications, several examples
applications in both C++ and Python are provided to demonstrate different possible
configurations and implementations. The source code for all the examples are located in
`src/hololink/emulation/examples`. The C++ applications by default will be built to
`<build directory>/examples/`.

For all examples, `--help` will provide invocation and flag details. All current
transport capabilities and therefore the examples will require running as `root` or
under `sudo`.

When building as "Emulator-only" with Python bindings, the Python examples should be
invoked with the Python interpreter in the provided virtual environment. For example, to
run the `serve_coe_stereo_vb1940_frames` example in Python, the full invocation would be

```
sudo build/env/bin/python3 src/hololink/emulation/examples/serve_coe_stereo_vb1940_frames.py 192.168.0.2
```

| example | sample invocation | brief description | HSB sample camera <br> driver compatibility | Compatible HSB receivers | Notes |
| ------- | ----------------- | ----------------- | ------------------------------------------- | ------------------------ | ----- |
| `serve_linux_file` | `./serve_linux_file 192.168.0.2 imx_single_raw_frame.dat`| serve frames from a file | imx cameras |`LinuxReceiver`, `RoceReceiver` |  |
| `serve_coe_file` | `./serve_coe_file 192.168.0.2 imx_single_raw_frame.dat` | serve frames from a file | imx cameras | `LinuxCoeReceiver` |  |
| `serve_coe_vb1940_file` | `./serve_coe_vb1940_file 192.168.0.2 imx_single_raw_frame.dat` | serve frames from a file | VB1940 | `LinuxCoeReceiver`, `SIPLCaptureOp` | Compatible with AGX Thor platform |
| `serve_linux_single_vb1940_frames` | `./serve_linux_single_vb1940_frames 192.168.0.2` | serve test RGB frames from single VB1940 camera | VB1940 | `LinuxReceiver`, `RoceReceiver` |  |
| `serve_linux_stereo_vb1940_frames` | `./serve_linux_stereo_vb1940_frames 192.168.0.2` | serve test RGB frames from stereo VB1940 camera | VB1940 | `LinuxReceiver`, `RoceReceiver` |  |
| `serve_coe_single_vb1940_frames` | `./serve_coe_single_vb1940_frames 192.168.0.2` | serve test RGB frames from single VB1940 camera | VB1940 | `LinuxCoeReceiver`, `SIPLCaptureOp` | Compatible with AGX Thor platform |
| `serve_coe_stereo_vb1940_frames` | `./serve_coe_stereo_vb1940_frames 192.168.0.2` | serve test RGB frames from stereo VB1940 camera | VB1940 | `LinuxCoeReceiver`, `SIPLCaptureOp` | Compatible with AGX Thor platform |

For the `*_file` examples, it is crucial that image frame sizes are set up
appropriately, especially on Jetson AGX Thor platform. <b>You will need to know the correct
CSI image frame size that needs to be sent.</b> These are based on Host application 
driver/operator requirements which are in turn dependent on the camera and HSB hardware. 
As examples,

- For imx274 capture in mode 1 (1920X1080 RAW10 - 4 pixels/5 bytes, 175 header bytes, 8 
  lines of optical black, 8-byte line alignment and 8-byte overall alignment) the CSI 
  frame size is (175 + (((1920 + 3) / 4 * 5 + 7) / 8 * 8) * (1080 + 8) + 7) / 8 * 8 = 
  2611376 bytes, so the input file should be at least 2611376 bytes in size and each 
  slice of that size will be transmitted as one frame from 192.168.0.2 to an 
  `*imx274_player` application with the following call:
  - `serve_linux_file -s 2611376 192.168.0.2 my_file.dat`
- For vb1940 capture in mode 0 (2560X1984 RAW10 - 4 pixels/5 bytes, 1 line of prefix 
  status, 2 lines of postfix status, 8-byte line alignment and 8-byte overall alignment) 
  the CSI frame size is ((((2560 + 3) / 4 * 5 + 7) / 8 * 8) * (1984 + 3) + 7) / 8 * 8 = 
  6458400 bytes, but on Jetson AGX Thor, it additionally goes through a 
  swizzle/packetizer step to a raw10 format (3 pixels/4 bytes, 64-byte line alignment) 
  such that the CSI frame size is effectively 
  ((((2560 + 2) / 3 * 4 + 63) / 64 * 64) * (1984 + 3) + 7) / 8 * 8 = 6867072 bytes. 
  Therefore on Jetson AGX Thor, a similar call would be:
  - `serve_coe_vb1940_file -s 6867072 192.168.0.2 my_file.dat`

For further details on other modes see the provided drivers in `python/hololink/sensors/`.

By default the `*file` examples will assume the whole file is one image frame. To set 
the image frame size, the `-s` flag is available in each of the examples. When the file 
size is greater than the value supplied to the `-s` option, it will assume multiple 
frames are in the file and cycle through them, advancing by `-s` bytes after each image 
frame and reset to the start of the file when a full frame can no longer be sent. For 
example, if `X` is provided to the `-s` option, `-s X`, and the file size is `N * X + Y` 
bytes where `N > 0`, and `0 <= Y < X`, the examples will send `N` image frames, reset to 
the beginning of the file, and then continue.

## Testing

Full-loop testing of the HSB Emulator is provided within the pytest framework in the docker
container. To run the test, launch the docker container and run:

```
pytest
```

the same as described in [Run tests in the demo container](RunningTestsTarget). The default pytest configuration
will test vb1940 camera modes with linux sockets in roce and coe packet formats for gpu or cpu inputs as well as
linux roce with imx477 camera modes. To run an extended sweep of parameters including also imx715 and imx274 camera mode, run:

```
pytest --emulator
```

to get the more comprehensive testing parameters. 

### Hardware Loopback

The tests above for vb1940 are all executed by default over software loopback. To run them over-the-wire using available network interfaces, they may be run in hardware loopback mode by attaching an Ethernet cable between 2 Ethernet-compatible ports on the target test device. Note the interfaces `IF1` and `IF2` (from e.g. `ip addr sh`) that correspond to the attached ports. Then run the tests with the following option:

```
pytest --hw-loopback IF1,IF2
```

This will run the same tests for the HSB Emulator, but isolate `IF1` from `IF2` and launch the relevant the emulator process on `IF1` and the test host application process targeting `IF2`. 

The scripts `nsisolate.sh`, `nsexec.sh`, `nspid.sh`, and `nsjoin.sh` in the `scripts/` directory are provided to facilitate development of additional hardware loopback tests. The `--hw-loopback IF1,IF2` switch in `pytest` is roughly equivalent to:

```
# setup environment
scripts/nsisolate.sh IF1 192.168.0.2/24
# for each test
scripts/nsexec.sh IF1 <emulator example command + args>
# teardown environment
scripts/nsjoin.sh IF1 <old IP address to reset>
```

The `nspid.sh` script facilitates finding the relevant `pid` with which to control a process within the network namespace created by `nsisolate.sh` since you cannot simply control/kill the `nsexec.sh` command without creating an orphaned process.

### Accelerated Networking on Hardware Loopback

On platforms that offer accelerated networking such as Jetson AGX Thor (SIPL COE over MGBE), IGX Orin (RoCEv2 over CX-7), and DGX Spark (RoCEv2 over CX-7), the network namespace scripts above also provide a simple way to test accelerated networking HSB applications. 

Under normal conditions, the Linux operating systems used on those target devices will route connections between two interfaces through an internal software/loopback interface. Isolating one of the connected interfaces in a network namespace will prevent the software loopback rerouting and force use of the accelerating hardware. The HSB Emulator tests for RoCE will already handle this and on relevant platforms will automatically configure a network namespace for testing when provided with the `--hw-loopback` parameters. On AGX Thor with SIPL, the MGBE interface needs to be managed independently but the tests will still run over the isolated interface if provided. As examples:

```
# interface names may vary
pytest --hw-loopback enP5p3s0f1np1,enP5p3s0f0np0
```

will run the linux, coe transport examples over standard linux sockets and also run the RoCE examples for vb1940 through the CX-7 ports.

```
scripts/nsisolate.sh enP2p1s0 192.168.0.2/24
# this next line run outside the container if mgbe0_0 is "DOWN" after isolating
sudo ip link set mgbe0_0 down && sudo ip link set mgbe0_0 up
# back in the container
pytest --hw-loopback enP2p1s0,mgbe0_0 --json-config my_single_vb1940_sensor_config.json
scripts/nsjoin.sh enP2p1s0
```

will run the linux, coe transport examples over standard linux sockets and also run the SIPL example for vb1940 through the MGBE port with COE offloading.

## Developing with HSB Emulator

### Primary Classes

For all the classes below, the C++ namespace is assumed `hololink::emulation` and the
Python module is loaded as `import hololink.emulation as hemu`. Python-equivalent
examples or definitions are given for clarity for non-trivial signatures. For further
details including private and protected members of classes for development purposes, see
the source code in `src/hololink/emulation`.

:::{dropdown} HSBEmulator

The HSB Emulator class is used for command and control communication between the Emulator
device and the Host HSB Application

`HSBEmulator` \<--> `Hololink` in host HSB application

```{eval-rst}
.. _doxygen_HSBEmulator:
.. doxygenclass:: hololink::emulation::HSBEmulator
   :members: start, stop, is_running, write, read, get_i2c, HSBEmulator
```

```{eval-rst}
.. doxygenclass:: hololink::emulation::MemRegister
   :members: MemRegister, write, read, write_many, read_many, write_range, read_range
```

`I2CController` is a component in `HSBEmulator` that is only needed to develop a sensor
emulator/driver bridge for I2C sensors.

```{eval-rst}
.. doxygenclass:: hololink::emulation::I2CController
   :members: attach_i2c_peripheral
```

:::

:::{dropdown} DataPlanes

`DataPlane`s are the object with which client applications of the HSB Emulator send data
to the Host application and how receivers identify unique sensor sources

`DataPlane` --> appropriate subclass of BaseReceiverOp in host HSB Application

```{eval-rst}
.. _doxygen_DataPlane:
.. doxygenclass:: hololink::emulation::DataPlane
   :members: DataPlane, start, stop, stop_bootp, is_running, send, get_sensor_id, packetizer_enabled
```

The `IPAddress` and the utility function `IPAddress_from_string` are provided to
encapsulate and simplify construction of the network interface properties required for
the `DataPlane` object in the hosted environment

```{eval-rst}
.. doxygenstruct:: hololink::emulation::IPAddress
   :members:
```

```{eval-rst}
.. doxygenfunction:: hololink::emulation::IPAddress_from_string

```

Two implementations of the `DataPlane` interface provided are `LinuxDataPlane` and
`COEDataPlane`

RoCEv2 UDP-based transport implementation.

```{eval-rst}
.. doxygenclass:: hololink::emulation::LinuxDataPlane
   :members: LinuxDataPlane
```

```{eval-rst}
.. doxygenclass:: hololink::emulation::COEDataPlane
   :members: COEDataPlane
```

:::

:::{dropdown} Transmitters - For transport development

Transmitters represent the actual transport implementation that is used by the DataPlane
to send the data.

For development of new/modified transport only. HSBEmulator applications currently do
not interact directly with these objects

```{eval-rst}
.. _doxygen_BaseTransmitter:
.. doxygenclass:: hololink::emulation::BaseTransmitter
   :members: send
```

Two implementations of the `BaseTransmitter` interface provided are `LinuxTransmitter`
and `COETransmitter`

```{eval-rst}
.. doxygenclass:: hololink::emulation::LinuxTransmitter
   :members: LinuxTransmitter, send
```

```{eval-rst}
.. doxygenclass:: hololink::emulation::COETransmitter
   :members: COETransmitter, send
```

:::

:::{dropdown} I2CPeripheral - For virtual sensor or driver bridge development

For development of new sensor bridge drivers (bridge HSBEmulator to a real device) or
emulating a sensor.

```{eval-rst}
.. _doxygen_I2CPeripheral:
.. doxygenclass:: hololink::emulation::I2CPeripheral
   :members: start, attach_to_i2c, stop, i2c_transaction
```

The primary example of implementing an I2CPeripheral for an HSB Emulator application is
the STM VB1940 sensor for emulating the
[Leopard Eagle HSB from Leopard Imaging](https://leopardimaging.com/sensor-bridge/).

```{eval-rst}
.. _doxygen_Vb1940Emulator:
.. doxygenclass:: hololink::emulation::sensors::Vb1940Emulator
   :members: start, attach_to_i2c, stop, i2c_transaction, is_streaming, get_pixel_width, get_pixel_height, get_bytes_per_line, get_image_start_byte, get_csi_length, get_pixel_bits
```

:::

### Structure of an HSB Emulator application

The [Examples](#examples) in the next section and the `main` functions in their source
code show typical workflows for starting up an HSBEmulator instance, but the general
outline in C++ and Python is given below for the case of RoCEv2 transmission over Linux
sockets.

::::{tab-set-code}

:::{code-block} cpp

// including the appropriate header for the implementation of the
// DataPlane is sufficient for a minimal application 
#include "hololink/emulation/linux_data_plane.hpp"

// OPTIONAL: import the target sensor emulator 
#include "hololink/emulation/sensors/vb1940_emulator.hpp"

// Declare a target HSBEmulator instance. In this case, the example is explicitly
// emulating a Leopard Eagle HSB configuration (needed for VB1940). It is not yet
// ready to receive commands from a host application until a subsequent call to its
// `start()` method. 
// NOTE: If multiple instances of HSBEmulator exist on a single device
// (even across // processes), they can receive and emit independent ECB packets, which 
// may conflict with the host application. 
HSBEmulator hsb(HSB_LEOPARD_EAGLE_CONFIG);

// Create an implementation of the DataPlane interface which will be used to send data
// as if from a *single* sensor. For this, you need to set a sensor_id to identify the
// sensor interface the channel will emulate and IPAddress + data_plane_id that identify
// NOTE: multiple DataPlanes with IPAddress or data_plane_id will result in multiple
// BOOTP broadcast message sources 
// NOTE: if multiple DataPlanes have the same (data_plane_id, sensor_id) combination, 
// their configurations may override each other
uint8_t data_plane_id = 0; 
uint8_t sensor_id = 0; 
LinuxDataPlane linux_data_plane(hsb, IPAddress_from_string(ip_address), 
   data_plane_id, sensor_id);

// Optional for stateful sensor emulation or driver bridge declare the sensor instance 
sensors::Vb1940Emulator vb1940; 
// Attach it to an appropriate I2CController and bus. On Leopard Eagle HSB, the i2c bus 
// address is the sensor_id offset from CAM_I2C_BUS 
vb1940.attach_to_i2c(hsb.get_i2c(hololink::I2C_CTRL), hololink::CAM_I2C_BUS
   + sensor_id);

// Start the emulator. After this, the HSBEmulator instance can receive and respond
// to control commands from the host application. All DataPlane instances which have
// been declared will start emitting BOOTP broadcast messages. All I2CPeripherals 
// will receive a call to their start() methods 
hsb.start();

// client code in a loop. Passing the DataPlane implementation gives the loop access
// to send data to the host application and sending the sensor instance (if needed)
// provides access to sensor-specific/state data
application_specific_data_loop(LinuxDataPlane, vb1940, ...user configuration data)

// OPTIONAL: Stop the emulator if the application needed to be able to restart the 
// HSBEmulator instance without exiting. All registered elements will receive a call
// to their `stop()` methods for cleanup. The emulator and all registered elements 
// can be re-started with a subsequent call to // `start()`. 
hsb.stop(); 

:::

:::{code-block} python

# importing the appropriate module for the implementation of the DataPlane is sufficient for
# a minimal application
import hololink.emulation as hemu

# import the hololink module for access to some HSB constants.
# NOTE: this is very minimal in the "Emulator-only" build and does not include any of the
# other hololink libraries
import hololink as hololink_module

# Declare a target HSBEmulator instance. In this case, the example is explicitly emulating a
# Leopard Eagle HSB configuration (needed for VB1940). It is not yet ready to receive
# commands from a host application until a subsequent call to
# its `start()` method.
# NOTE: If multiple instances of HSBEmulator exist on a single device (even across
# processes), they can receive and emit independent ECB packets, which may conflict with the
# host application.
hsb = hemu.HSBEmulator(hemu.HSB_LEOPARD_EAGLE_CONFIG)

# Create an implementation of the DataPlane interface which will be used to send data as if
# from a *single* sensor. For this, you need to set a sensor_id to identify the sensor
# interface the channel will emulate and IPAddress + data_plane_id that identify
# NOTE: multiple DataPlanes with IPAddress or data_plane_id will result in multiple BOOTP
# broadcast message sources
# NOTE: if multiple DataPlanes have the same (data_plane_id, sensor_id) combination, their
# configurations may override each other
data_plane_id = 0
sensor_id = 0
data_plane = hemu.LinuxDataPlane( hsb, hemu.IPAddress(args.ip_address), 
   data_plane_id, sensor_id)

# Optional for stateful sensor emulation or driver bridge declare the sensor instance 
vb1940 = hemu.sensors.Vb1940Emulator()
# Attach it to an appropriate I2CController and bus. On Leopard Eagle HSB, the i2c bus 
# address is the sensor_id offset from CAM_I2C_BUS 
vb1940.attach_to_i2c(
   hsb.get_i2c(hololink_module.I2C_CTRL), hololink_module.CAM_I2C_BUS + sensor_id
)

# Start the emulator.
# After this, the HSBEmulator instance can receive and respond to control commands from the
# host application. All DataPlane instances which have been declared will start emitting
# BOOTP broadcast messages. All I2CPeripherals will receive a call to their start() methods
hsb.start()

# sensor client code in a loop. Passing the DataPlane implementation gives the loop access to
# send data to the host application
# and sending the sensor instance (if needed) provides access to sensor-specific/state data
application_specific_data_loop(LinuxDataPlane, vb1940, ...user configuration data)

# OPTIONAL: Stop the emulator if the application needed to be able to restart the HSBEmulator
# instance without exiting. All registered elements will receive a call to their `stop()`
# methods for cleanup. The emulator and all registered elements can be re-started with a
# subsequent call to `start()`.
hsb.stop() 

:::

::::

For stereo/dual sensor setups, see the `*stereo_vb1940_frames` examples. The specific
loops used to stream the files or test frames across all the examples can be found in
the `emulator_utils` files.

## Troubleshooting

| Problem Description | Root Cause | Corrective Action (s) |
| ------------------- | ---------- | --------------------- |
| "Permission Denied"/"Operation not permitted" in socket creation or bind | UDP/port combination is protected by [capabilities](https://man7.org/linux/man-pages/man7/capabilities.7.html) privilege | - Run under `sudo` <br> - Check firewall status and if necessary ensure the ports are open for UDP. <br> Example (Ubuntu 24.04 UDP to/from IGX on port 192.168.0.101): <br> `sudo ufw status # check for UDP access, specifically on port 8192`<br>`sudo ufw allow from 192.168.0.101 to any port 8192`<br> - use `sudo setcap cap_net_raw+ep /path/to/executable` to remove raw socket restrictions. WARNING: allowing this on the Python interpreter has severe security implications |
| white streaks/lost packets in frame visible on receiver end | on the receiver host device, make sure your have sufficient network receiver buffer size. Common with LinuxReceiverOp and LinuxCoeReceiverOp, esp on AGX devices. | `sudo sysctl -w net.core.rmem_max=16777216 && sudo sysctl -w net.core.rmem_default=16777216` |
| running with Python code: <br>AttributeError: 'numpy.ndarray' object has no attribute '\_\_dlpack\_\_' <br>or similar with data types other than 'numpy.ndarray' | data type (in this case numpy array) does not support [Python array API](https://data-apis.org/array-api/latest/design_topics/data_interchange.html) DLPack data interchange format | - change to a data type that supports DLPack <br> - upgrade environment to version of package that does support DLPack (numpy requires 1.23+) |
| frequent I2C failures with emulator sending to Thor host device | failed packet receipt on Thor mGbE interface with no retries in JP 7.0 with SIPL rel 38.2.1 | upgrade to JP 7.1 EA rel 38.4 |

## Clean-up

`rm -rf build`

