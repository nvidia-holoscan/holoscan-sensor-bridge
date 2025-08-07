# HSB Emulation

The emulation directory of the Holoscan Sensor Bridge (HSB) project encapsulates a
minimal interface to develop or test applications that require an HSB device in
production without physically having an HSB device present. This functionality is
provided as a static or dynamic library (the "HSB Emulator") as well as a python module
within a hololink package.

HSB Emulator provides an interface in C++ and Python to send data (stored on any\*
device in contiguous memory) as a DLTensor satisfying the
[DLPack](https://dmlc.github.io/dlpack/latest/) interface. In the Python API, this means
any contiguous array object that satisfies the Python array API standard with DLPack
[data interchange mechanisms](https://data-apis.org/array-api/latest/design_topics/data_interchange.html#data-interchange).
This offers a stable C/C++ ABI and in Python has wide adoption in Python but
specifically excludes sparse and distributed arrays (and certain array views in numpy) -
those must be converted to contiguous arrays before transmission.

\*any in this case means any host cpu or CUDA GPU memory

There are 2 types of builds and build environments provided

- Emulator-only
  - This is a build that uses code/sources within the HSB source tree and create a
    static library, dynamic library, and bare bones python module installed within a
    python virtual environment without linking against hololink or holoscan libraries.
    Runtime dependencies are minimized and C++ and Python APIs are provided.
- Full
  - This is a build done within the standard HSB cmake build structure. It includes the
    libraries from the isolated build and will integrate the python interface within the
    hololink python package and provide example operators within the holoscan SDK
    framework.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building](#building)
  - [Emulator-only](#emulator-only-build)
  - [Full](#full-build)
- [Environment Configuration](#environment-configuration)
- [Running Examples](#running-examples)
  - [Emulator-only](#run-emulator-only-examples)
- [Clean-Up](#clean-up)
  - [Emulator-only](#emulator-only-clean-up)
  - [Full](#full-clean-up)
- [Troubleshooting](#troubleshooting)
- [Known issues](#known-issues)

## Prerequisites

The emulator builds have been tested with the requirements below on the following
platforms

- AGX Orin DevKit with Jetson 6.1 (Ubuntu 22.04)
- IGX Orin DevKit with IGX BaseOs 1.0 (Ubuntu 22.04)
- x86 Desktop with Ubuntu 24.04

<table>
  <tbody>
    <tr>
      <td> build description </td>
      <td> API </td>
      <td> build requirements </td>
      <td> runtime requirements </td>
      <td> notes </td>
    </tr>
    <tr>
      <td rowspan="2"> Emulator-only </td>
      <td> C++ or Python </td>
      <td> posix utilities (ar, grep, awk, sed, etc) <br> GNU make <br> pkg-config <br> gcc <br> curl <br> dlpack (v0.6+) </td>
      <td> CUDA runtime (Toolkit v12)  <br> zlib <br> posix networking headers </td>
      <td> CUDA runtime including nvcc in standard install directories or with discoverable pkg-config file
    </tr>
    <tr>
      <td> Python </td>
      <td> python (v3.10+) <br> pybind11 (C++ headers and python module) <br> venv </td>
      <td> python (v3.10+) <br> pybind11 <br> numpy (v1.22+) <br> CuPy(optional, CUDA 12.x compatible)</td>
      <td> numpy requirement and optional CuPy only explicitly for examples but some python packages may rely on numpy DLPack compatibility internally. Note the system numpy for Ubuntu-22.04 or earlier (when installed via <code>apt</code>) is not compatible with HSB Emulator. A virtual environment is created in the build to facilitate use. If you install numpy via <code>pip</code> before <code>apt</code>, system python might capture a compatible numpy version </td>
    </tr>
    <tr>
      <td> Full </td>
      <td> C++ or Python </td>
      <td> docker <br> cmake (v3.20+) <br> pybind11 <br> python (v3.10+) </td>
      <td> docker <br> Holoscan SDK (3.2.0+) <br> python (v3.10+) </td>
      <td> For full build and runtime requirements see <a href="https://docs.nvidia.com/holoscan/sensor-bridge/latest/introduction.html#host-requirements">Host Requirements</a></td>
    </tr>
  </tbody>
</table>

An internet connection is required to download the `dlpack.h` requirement unless
provided manually. The
[`dlpack.h`](https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h) file need
only be present in the `src/hololink/emulation/dlpack` directory.

Versions of requirements listed below are minimally tested, not necessarily minimal
required.

## Building

### Emulator-only build

```
$ sudo apt-get install python3-pip python3-venv
$ git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
$ cd holoscan-sensor-bridge/src/hololink/emulation
$ make -j
```

#### Configuring the Emulator-only build

Several `make` flags are available to configure the "Emulator-only" build if
prerequisites are not met or not in expected locations

<table>
  <tbody>
    <tr>
      <td> dependency </td>
      <td> make/environmental variables </td>
    </tr>
    <tr>
      <td> make CuPy available in python example </td>
      <td> Set <code>PY_GPU</code> to any non-empty string
    </tr>
    <tr>
      <td> python unavailable/not needed </td>
      <td> Set <code>NO_PYTHON</code> to any non-empty string </td>
    </tr>
    <tr>
      <td> pkg-config missing </td>
      <td> <code>ZLIB_CFLAGS</code> set to the include directory and any compile flags required for local version of zlib <bc> <code>ZLIB_LFLAGS</code> set to the linker library paths and libraries for local version of zlib. Similar for <code>CUDA_CFLAGS</code> and <code>CUDA_LFLAGS</code> for the CUDA runtime library </td>
    </tr>
    <tr>
      <td> non-standard cuda runtime installation directory or missing/misplaced cudart-*.pc file </td>
      <td> environmental variable <code>PKG_CONFIG_PATH</code> set to the location of the cudart-*.pc file if outside the pkg-config search path. If necessary <code>CUDA_PKG_CONFIG</code> set to the cudart-*.pc file name </td>
    </tr>
  </tbody>
</table>

### Full build

Follow the top level environment setup and docker container build instructions for
[holoscan-sensor-bridge](https://docs.nvidia.com/holoscan/sensor-bridge/latest/build.html)

Once docker container is built, from the git repository directory:

```
$ sh docker/demo.sh
$ mkdir build
$ cmake -S . -B build
$ cmake --build build -j [--target specify emulator specific targets]
```

## Environment Configuration

Set up Ethernet adapter to be on same subnet as target HSDK host (e.g. QSFP ports
192.168.0.101/102 on IGX)

`$ sudo ip addr add 192.168.0.XXX/24 dev <Adapter connected to IGX>`

Example programs may assume a default IP address of `192.168.0.221`.

If the adapter already or automatically has an assigned address, you may have to delete
it.

## Running Examples

In all examples below, the emulator software is run on a host with an ethernet adapter
configured to send/receive on `192.168.0.221`.

### Run Emulator-only examples

Run all examples below from `/path/to/src/hololink/emulation/`.

It is assumed that the user has an appropriately formatted binary data file "data.dat"
at `/path/to/data.dat`

#### C++

```
$ sudo examples/serve_frames 192.168.0.221 /path/to/data.dat
```

#### Python

```
sudo env/bin/python3 examples/serve_frames.py 192.168.0.221 /path/to/data.dat
```

## Troubleshooting

<table>
  <tbody>
    <tr>
      <td> Problem Description </td>
      <td> Root Cause </td>
      <td> Corrective Action(s) </td>
    </tr>
    <tr>
      <td> Permission Denied/Operation not permitted in socket creation or bind </td>
      <td> UDP/port combination is protected by <a href="https://man7.org/linux/man-pages/man7/capabilities.7.html">capabilities</a> privilege </td>
      <td> 
        - Run under <code>sudo</code> <br> 
        - Check firewall status and if necessary ensure the ports are open for UDP. <br> 
        Example (Ubuntu 24.04 UDP to/from IGX on port 192.168.0.101): <br> 
        <code>sudo ufw status # check for UDP access, specifically on port 8192<br>sudo ufw allow from 192.168.0.101 to any port 8192</code><br>
        - use <code>sudo setcap cap_net_raw+ep /path/to/executable</code> to remove raw socket restrictions. WARNING: allowing this on the python interpreter has severe security implications</td>
    </tr>
    <tr>
      <td>running with python code: <br>AttributeError: 'numpy.ndarray' object has no attribute '__dlpack__' <br>or similar with data types other than 'numpy.ndarray'</td>
      <td>data type does not support <a href=https://data-apis.org/array-api/latest/design_topics/data_interchange.html>Python array API</a> DLPack data interchange format</td>
      <td>- change to a data type that supports DLPack <br> - upgrade environment to version of package that does support DLPack</td>
    </tr>
  </tbody>
</table>

## Clean-up

### Emulator-only clean-up

For rebuild: `$ make reset`

For full reset including `dlpack.h`: `$ make clean`

### Full clean-up

For rebuild: `$ rm -rf build`

For full reset include `dlpack.h`: `$ rm -f src/hololink/emulation/dlpack.h`

### Known Issues
