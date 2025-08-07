# Running Holoscan Sensor Bridge examples

Holoscan sensor bridge Python example applications are located under the `examples`
directory.

Below are instructions for running the applications on the IGX and the Jetson AGX
platforms.

- Examples starting with the word "linux\_" in the filename use the unaccelerated Linux
  Sockets API network receiver operator. These examples work on both IGX and AGX
  systems.
- Examples without "linux\_" in the filename use the accelerated network receiver
  operator and require ConnectX SmartNIC controllers, like those on IGX. AGX systems
  cannot run these examples.
- These examples all work on both iGPU and dGPU configurations. If the underlying OS and
  Holoscan sensor bridge container are built with the appropriate iGPU or dGPU setting,
  the application code itself does not change.

Most examples have both the accelerated and an unaccelerated Linux Sockets API version.

## IMX274 player example

To run the high-speed video player with IMX274, in the demo container with a ConnectX
accelerated network controller,

`````{tab-set}
````{tab-item} Python
```sh
$ python3 examples/imx274_player.py
```

or, for unaccelerated configurations (e.g. AGX),

```sh
$ python3 examples/linux_imx274_player.py
```

````
````{tab-item} C++

The C++ examples need to be built first using these commands; this leaves the resulting
executables in /tmp/build/examples.

```sh
$ export BUILD_DIR=/tmp/build
$ cmake -S . -B $BUILD_DIR -G Ninja -DHOLOLINK_BUILD_PYTHON=OFF
$ cmake --build $BUILD_DIR -j $(nproc)
```

After examples are built, you can run the `imx274_player`:

```sh
$ $BUILD_DIR/examples/imx274_player
```

Note that only the C++ example is only supported with the accelerated network receiver.

````
`````

Documentation breaking down the source code for the IMX274 player application is
[available here](applications.md#imx274_player); this example illustrates the basic
sensor bridge workflow which is described in the
[architecture documentation](architecture.md). Press Control/C to stop the video player.

## Leopard imaging VB1940 Eagle player example

This example is similar to the IMX274 player example above, using an Li VB1940 Eagle
camera instead of IMX274. To run the high-speed video player with Li VB1940 Eagle, in
the demo container with a ConnectX accelerated network controller,

```sh
$ python3 examples/vb1940_player.py
```

or, for unaccelerated configurations (e.g. AGX),

```sh
$ python3 examples/linux_vb1940_player.py
```

## Running the IMX274 TAO PeopleNet example

The tao-peoplenet example demonstrates running inference on a live video feed.
[Tao PeopleNet](https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/cv_models/peoplenet.html)
provides a model that given an image can detect persons, bags, and faces. In this
example, when those items are detected, bounding boxes are shown as an overlay over the
live video.

**Prerequisite**: Download the PeopleNet ONNX model from the NGC website:

```sh
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=resnet34_peoplenet_int8.onnx' -O examples/resnet34_peoplenet_int8.onnx
```

For systems with accelerated network interfaces,

```sh
$ python3 examples/tao_peoplenet.py
```

or unaccelerated configurations,

```sh
$ python3 examples/linux_tao_peoplenet.py
```

This will bring up the Holoscan visualizer on the GUI showing the live video feed from
the IMX274 device as well as red/green box overlays when a person image is captured.
Press Ctrl/C to exit. More information about this application can be found
[here](applications.md#tao_peoplenet).

## Running the IMX274 body pose example

**Prerequisite**: Download the YOLOv8 ONNX model from the YOLOv8 website and generate
the body pose ONNX model. Within the Holoscan sensor bridge demo container:

From the repo base directory `holoscan-sensor-bridge`:

```sh
apt-get update && apt-get install -y ffmpeg
pip3 install ultralytics onnx
cd examples
yolo export model=yolov8n-pose.pt format=onnx
trtexec --onnx=yolov8n-pose.onnx --saveEngine=yolov8n-pose.engine.fp32
cd -
```

Note that this conversion step only needs to be executed once; the
`yolov8n-pose.engine.fp32` file contains the converted model and is all that's needed
for the demo to run. The installed components will be forgotten when the container is
exited; those do not need to be present in future runs of the demo.

For systems with accelerated network interfaces, within the sensor bridge demo
container, launch the Body Pose estimation:

```sh
$ python3 examples/body_pose_estimation.py
```

For unaccelerated configurations (e.g. AGX), launch the Body Pose estimation example
within the demo container this way:

```sh
$ python3 examples/linux_body_pose_estimation.py
```

This will bring up the Holoscan visualizer on the GUI showing the live video feed from
the IMX274 device, along with a green overlay showing keypoints found by the body pose
net model. For more information about this application, look
[here](applications.md#body_pose_estimation).

Press Ctrl/C to exit.

## Running the Stereo IMX274 and Leopard imaging VB1940 Eagle examples

For IGX, `examples/stereo_imx274_player.py` shows an example with two independent
pipelines, one for each camera on the dual-camera module. Accelerated networking is used
to provide real time access to the pair of 4k image streams. Make sure that
[both network ports are connected](sensor_bridge_hardware_setup.md#connecting-holoscan-sensor-bridge-to-the-host)
between the IGX and the Holoscan sensor bridge unit.

```sh
$ python3 examples/stereo_imx274_player.py
```

This brings up a visualizer display with two frames, one for the left channel and the
other for the right.

For the purpose of aggregating lower bandwidth streams, you can observe the following
examples aggregating both cameras to a single network port:

IGX Orin with IMX274:

```sh
$ python3 examples/single_network_stereo_imx274_player.py
```

AGX Orin with IMX274:

```sh
$ python3 examples/linux_single_network_stereo_imx274_player.py
```

IGX Orin with Li VB1940 Eagle:

```sh
$ python3 examples/single_network_stereo_vb1940_player.py
```

AGX Orin with Li VB1940 Eagle:

```sh
$ python3 examples/linux_single_network_stereo_vb1940_player.py
```

Applications wishing to map sensors to specific data channels can do so using the
`use_sensor` API, which is demonstrated in these examples. The AGX network interface is
limited to 10Gbps so support is only provided for observing stereo video in 1080p mode.

## Running the GPIO example

`examples/gpio_example_app.py` is a simple example of using the GPIO interface of the
sensor bridge to set GPIO directions, read input values from GPIO pins and write output
values to GPIO pins. To run the application:

```sh
$ python3 examples/gpio_example_app.py
```

This brings up a textual display which cycles over different pre-set pin configurations
and allows time between different settings of the pins to measure or readback pins
values. Please refer to the application structure section to read more about the
[GPIO example application](applications.md#gpio-example-application).

## Running the NVIDIA ISP with live capture example

`examples/linux_hwisp_player.py` shows an example of NVIDIA ISP unit processing the
Bayer frame captured live using IMX274. The ISP unit currently is available on Jetson
Orin AGX and IGX Orin in iGPU configuration.

Before starting the docker run, setup the `nvargus-daemon` with the flag
`enableRawReprocessing=1`. This enables us to run the ISP with the Bayer frame capture
using Holoscan sensor bridge unit and this change persists through even restart. In the
host system:

```sh
sudo su
pkill nvargus-daemon
export enableRawReprocessing=1
nvargus-daemon
exit
```

To run the example, within the demo container:

```sh
$ python3 examples/linux_hwisp_player.py
```

This will run the application with visualizer display showing the live capture.

Note if user wishes to undo running the `nvargus-daemon` with flag
`enableRawReprocessing=1`, then please execute the following command.

```sh
sudo su
pkill nvargus-daemon
unset enableRawReprocessing
nvargus-daemon
exit
```

## Running the Latency for IMX274 example

For IGX systems, `examples/imx274_latency.py` shows an example of how to use timestamp
to profile hardware and software pipeline. This example demonstrates recording
timestamps received from the FPGA when data is acquired and timestamps measured in the
host at various points in frame reception and pipeline execution. At the end of the run,
the application will provide a duration and latency report with average, minimum, and
maximum values.

Before running the app, make sure the PTP sync has been enabled on the setup and then
use the following commands to run the example.

```sh
$ python3 examples/imx274_latency.py
```

Running the latency example application on AGX Orin systems:

```sh
$ python3 examples/linux_imx274_latency.py
```

## Running the ECam0M30ToF Player Application

The `ecam0m30tof_player.py` application demonstrates how to capture and display depth
and/or IR data from the ECam0M30ToF time-of-flight camera using RoCE (RDMA over
Converged Ethernet) for high-performance data transmission, utilizing Holoviz operator's
`DEPTH_MAP` rendering for enhanced depth visualization. This application can be modified
to run on Jetson AGX by changing the receiver operator from `RoceReceiverOp` to
`LinuxReceiverOperator`.

**Prerequisites**: ECam0M30ToF camera connected to the Hololink device

To run the application use following command.

```bash
python3 examples/ecam0m30tof_player.py --hololink 192.168.0.2 --camera-mode=<0|1|2>
```

**Camera Configuration**:

- `--camera-mode`: Select camera mode (0: `DEPTH_IR`, 1: `DEPTH`, 2: `IR`)

## Running the frame validation example for IMX274

For AGX systems, `examples/linux_imx274_frame_validation.py` shows an example of how to
access frame metadata in order to detect missing frames, frame timestamp misalignment
and frame CRC errors. This example demonstrates recording timestamps, frame numbers and
CRC32 received from the FPGA when data is acquired. During the run, missing frames,
timestamp misalignment and CRC32 errors are detected and prompted out to the console. At
the end of the run, the application will provide a duration and latency report with
average, minimum, and maximum values same as the Latency example application. These
values are collected during the application run to assess the impact of various
detection mechanisms on the latency of the pipeline.

Before running the app, enable PTP sync on your setup, then use the following commands
to run the example. Running the frame validation example on AGX Orin systems:

```sh
$ python3 examples/linux_imx274_frame_validation.py
```

Since the CRC32 calculation in this example is done by CPU, trying to detect CRC32 error
using the example as is will trigger frame loss errors. For that reason CRC32 error
detection is not enabled by default. To enable CRC32 detection every N frames use the
`--crc-frame-check` option:

```sh
$ python3 examples/linux_imx274_frame_validation.py --crc-frame-check 50
```

In this example, the application will check for CRC32 frame errors every 50 frames.

While there is no equivalent example for IGX systems, this example can be quickly
adapted by users for IGX.
