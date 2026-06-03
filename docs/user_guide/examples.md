# Running Holoscan Sensor Bridge examples

Holoscan sensor bridge Python example applications are located under the `examples`
directory.

Below are instructions for running the applications on the AGX Thor, DGX Spark, IGX and
the Jetson AGX Orin platforms. For each IGX camera example, an appropriately configured
DGX Spark may also be used.

- Examples starting with the word "linux\_" in the filename use the unaccelerated Linux
  Sockets API network receiver operator. These examples work on DGX Spark and both IGX
  and AGX Orin/Thor systems.
- Examples without "linux\_" in the filename use the accelerated network receiver
  operator and require ConnectX SmartNIC controllers, like those on IGX and DGX Spark.
  AGX Orin and AGX Thor systems cannot run these examples.
- These examples all work on both iGPU and dGPU configurations. If the underlying OS and
  Holoscan sensor bridge container are built with the appropriate iGPU or dGPU setting,
  the application code itself does not change.
- Examples starting with "sipl\_" use the
  [SIPL](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/CameraDevelopment/SIPLFramework/Introduction-to-SIPL.html)
  accelerated network receiver operator and require MGBE SmartNIC controller and are
  unique to AGX Thor.
- For accelerated networking examples on IGX Thor, `CUDA_VISIBLE_DEVICES` environment
  variable will have to be set to the matching graphics GPU configuration, which
  defaults to iGPU. For example, if dGPU graphics are configured, `imx274_player` must
  have `CUDA_VISIBLE_DEVICES` set to `0` to select RDMA and graphics to the dGPU.

Most examples have both the accelerated and an unaccelerated Linux Sockets API version.

## IMX274 player example

To run the high-speed video player with IMX274, in the demo container with a ConnectX
accelerated network controller,

`````{tab-set}
````{tab-item} Python
```sh
$ python3 examples/imx274_player.py
```

or, for unaccelerated configurations (e.g. AGX Orin, AGX Thor),

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

Note that only the C++ example is only supported with the accelerated network receiver 
with ConnectX SmartNIC controllers.

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

for unaccelerated configurations (e.g. AGX Orin, AGX Thor),

```sh
$ python3 examples/linux_vb1940_player.py
```

lastly, running SIPL accelerated network python example on AGX Thor:

```sh
$ python3 ./examples/sipl_player.py --json-config ./examples/sipl_config/vb1940_single.json
```

## Running the TAO PeopleNet example

The tao-peoplenet example demonstrates running inference on a live video feed.
[Tao PeopleNet](https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/cv_models/peoplenet.html)
provides a model that given an image can detect persons, bags, and faces. In this
example, when those items are detected, bounding boxes are shown as an overlay over the
live video.

Note that metadata passing has an
[open issue](https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v4.2.0) when
used with InferenceOp and is explicitly disabled in the Application in these examples.

**Prerequisite**: Download the PeopleNet ONNX model from the NGC website:

```sh
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=resnet34_peoplenet_int8.onnx' -O examples/resnet34_peoplenet_int8.onnx
```

For systems with a ConnectX accelerated network controller interfaces with IMX274
camera,

```sh
$ python3 examples/tao_peoplenet.py
```

for unaccelerated configurations (e.g. AGX Orin, AGX Thor) with IMX274 camera,

```sh
$ python3 examples/linux_tao_peoplenet.py
```

lastly, running SIPL accelerated network python example on AGX Thor with LI VB1940
camera:

```sh
$ python3 ./examples/sipl_tao_peoplenet.py --json-config ./examples/sipl_config/vb1940_single.json
```

This will bring up the Holoscan visualizer on the GUI showing the live video feed from
the IMX274/Li VB1940 device as well as red/green box overlays when a person image is
captured. Press Ctrl/C to exit. More information about this application can be found
[here](applications.md#tao_peoplenet).

## Running the body pose example

**Prerequisite**: Download the YOLOv8 ONNX model from the YOLOv8 website and generate
the body pose ONNX model. Within the Holoscan sensor bridge demo container:

**NOTE:** The stock AGX Orin Development Kits running Jetpack 7.2 may not have
sufficient disk space to install dependencies. Prior to installing the python
dependencies below, ensure you have plenty of disk space by clearing out unused packages
or the docker builder cache

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
container, launch the Body Pose estimation with IMX274 camera:

```sh
$ python3 examples/body_pose_estimation.py
```

for unaccelerated configurations (e.g. AGX Orin, AGX Thor), launch the Body Pose
estimation example within the demo container this way:

```sh
$ python3 examples/linux_body_pose_estimation.py
```

lastly, running SIPL accelerated network python example on AGX Thor with Li VB1940
camera:

```sh
$ python3 ./examples/sipl_body_pose_estimation.py --json-config ./examples/sipl_config/vb1940_single.json
```

This will bring up the Holoscan visualizer on the GUI showing the live video feed from
the IMX274/Li VB1940 device, along with a green overlay showing keypoints found by the
body pose net model. For more information about this application, look
[here](applications.md#body_pose_estimation).

Press Ctrl/C to exit.

## Running the Stereo IMX274 and Leopard imaging Li VB1940 Eagle examples

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

AGX Orin or AGX Thor with IMX274:

```sh
$ python3 examples/linux_single_network_stereo_imx274_player.py
```

IGX Orin with Li VB1940 Eagle:

```sh
$ python3 examples/single_network_stereo_vb1940_player.py
```

AGX Orin or AGX Thor with Li VB1940 Eagle:

```sh
$ python3 examples/linux_single_network_stereo_vb1940_player.py
```

SIPL accelerated network python example on AGX Thor with Li VB1940 camera:

```sh
$ python3 ./examples/sipl_player.py --json-config ./examples/sipl_config/vb1940_dual.json
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
`enableRawReprocessing=1` and `rawReprocessModulePartName="A6V26"`. This enables us to
run the ISP with the Bayer frame capture from imx274 using Holoscan sensor bridge unit
and this change persists through even restart. In the host system:

```sh
sudo su
pkill nvargus-daemon
export enableRawReprocessing=1
export rawReprocessModulePartName="A6V26"
nvargus-daemon
exit
```

To run the example, within the demo container:

```sh
$ python3 examples/linux_hwisp_player.py
```

This will run the application with visualizer display showing the live capture. Note
that the default camera mode has changed since Jetpack 6 and now defaults to 4k frames.
This example is compatible with Jetpack 7.2+.

Note if user wishes to undo running the `nvargus-daemon` with flag
`enableRawReprocessing=1`, then please execute the following command.

```sh
sudo su
pkill nvargus-daemon
unset enableRawReprocessing
unset rawReprocessModulePartName
nvargus-daemon
exit
```

**This example will not run on AGX Thor**

## Running the Latency Application for IMX274 example

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

**This example will not run on AGX Thor**

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

**This example will not run on AGX Thor**

## Running CPU and GPU frame validation examples

Frame validation examples demonstrate how to access frame metadata in order to detect
missing frames, frame timestamp misalignment and frame CRC errors. These examples record
timestamps, frame numbers and CRC32 data received from the FPGA when data is acquired.
During the run, missing frames, timestamp misalignment and CRC32 errors are detected and
reported. At the end of the run, the application provides a duration and latency report
with average, minimum, and maximum values. These values are collected during the
application run to assess the impact of various detection mechanisms on the latency of
the pipeline.

### IMX274

#### Linux Receiver

For AGX systems (or unaccelerated configurations),
`examples/linux_imx274_frame_validation.py` uses standard Linux sockets for network
communication with CPU-based CRC validation.

Before running the app,
[enable PTP sync](https://docs.nvidia.com/holoscan/sensor-bridge/latest/setup.html#) on
your setup, then use the following commands to run the example. Running the frame
validation example on AGX Orin systems:

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

#### RoCE Receiver

For systems with RDMA-capable network hardware (ConnectX NICs) such as IGX Orin,
`examples/imx274_frame_validation.py` provides high-performance frame validation with
GPU-accelerated CRC checking using nvCOMP 5.0. This example uses the accelerated network
receiver operator and requires ConnectX SmartNIC controllers.

Before running the app,
[enable PTP sync](https://docs.nvidia.com/holoscan/sensor-bridge/latest/setup.html#) on
your setup, then use the following command:

```sh
$ python3 examples/imx274_frame_validation.py
```

Unlike the CPU-based CRC validation in the Linux version,
[GPU-based CRC using nvCOMP 5.0](https://docs.nvidia.com/cuda/nvcomp/crc32.html) is fast
enough to validate every frame by default. CRC validation is enabled by default with
`--crc-frame-check 1`. To disable CRC validation entirely, use `--crc-frame-check 0`.

At the end of execution, the application provides a CRC validation report showing total
frames processed, CRC errors detected, and success rate, followed by detailed
performance metrics including frame time, transfer latency, operator latency, and
processing time.

To validate the stereo camera configuration:

```sh
$ python3 examples/stereo_imx274_frame_validation.py
```

#### Performance

The nvCOMP CRC calculation performance on IGX-dGPU for single-camera configuration with
4K resolution (measured over 1000 frames):

```text
Minimum: 275 us
Maximum: 390 us
Average: 295 us
```

**Note on Startup Performance:** Runtime performance of an HSDK pipeline at startup can
be unpredictable, usually due to GPU kernel initialization. This is likely to lead to
CRC failures: when the pipeline is slower than the camera frame rate, the receiver
buffer can be overwritten with new data, which triggers the failure that CRC checking is
looking for. Once the pipeline is fully initialized and can keep up with the received
data, these errors would no longer be expected. We have primarily observed this issue in
the stereo camera case. For this reason, our testing (see
`tests/test_imx274_pattern.py`) skips CRC checking at the beginning, e.g. only after 15
frames have been received. User applications would likely use a similar startup state to
avoid misleading errors occurring due to this known condition.

**This example will not run on AGX Thor**

### Li VB1940 Eagle

#### FUSA CoE Receiver (nvCOMP)

For systems with FUSA (Functional Safety) CoE (Camera over Ethernet) support such as AGX
Thor, `examples/vb1940_fusa_nvcomp_crc_validation.py` provides high-performance frame
validation with GPU-accelerated CRC checking using nvCOMP 5.0. This example uses the
FUSA CoE capture operator for accelerated network data transfer.

Before running the app,
[enable PTP sync](https://docs.nvidia.com/holoscan/sensor-bridge/latest/setup.html#) on
your setup, then use the following command:

```sh
$ python3 examples/vb1940_fusa_nvcomp_crc_validation.py
```

[GPU-based CRC using nvCOMP 5.0](https://docs.nvidia.com/cuda/nvcomp/crc32.html) is fast
enough to validate every frame by default. The application computes CRC on the full CSI
frame (including CSI header and trailing bytes) to match the camera's FPGA CRC
computation.

At the end of execution, the application provides a CRC validation report showing total
frames processed, CRC errors detected, and success rate, followed by detailed
performance metrics including frame time, FUSA capture latency, operator latency, and
processing time.

#### Performance

The nvCOMP CRC calculation performance on AGX Thor for Li VB1940 Eagle camera
configuration (measured over 1000 frames):

```text
Minimum: 554.4 us
Maximum: 631.0 us
Average: 614.3 us
```

## Running PVA frame validation examples

PVA CRC validation examples demonstrate hardware-accelerated CRC computation using
NVIDIA PVA (Programmable Vision Accelerator) to validate camera frames. These examples
compare PVA-computed CRC values against camera FPGA-embedded CRC values to detect data
corruption.

For requirements and build instructions, see the
[PVA CRC README](../../python/hololink/operators/pva_crc/README.md).

To run the PVA CRC validation examples, first set the `LD_LIBRARY_PATH` environment
variable (required before running any application). Within the demo container:

```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra/:/opt/nvidia/pva-sdk-2.9/lib/aarch64-linux-gnu/:/usr/lib/aarch64-linux-gnu/nvidia
```

To run the IMX274 PVA CRC validation example (tested on IGX Orin with dGPU):

```sh
$ python3 examples/imx274_pva_crc_validation.py --frame-limit 100
```

To run the Li VB1940 Eagle PVA CRC validation example (tested on AGX Thor):

```sh
$ python3 examples/vb1940_fusa_pva_crc_validation.py --frame-limit 100
```

### Performance

#### IMX274

The PVA CRC calculation performance on IGX-dGPU for single-camera configuration with 4K
resolution (measured over 1000 frames):

```text
Minimum: 412 us
Maximum: 442 us
Average: 414 us
```

#### Li VB1940 Eagle

The PVA CRC calculation performance on AGX Thor for single-camera configuration with
1080p resolution (measured over 1000 frames):

```text
Minimum: 106 us
Maximum: 137 us
Average: 111 us
```

## Running the UART dual-board example

`examples/uart_dual_board_loopback.py` tests UART between two boards.

**Hardware setup**: Connect Board 1 GPIO 10 (TX) to Board 2 GPIO 11 (RX) and Board 2
GPIO 10 (TX) to Board 1 GPIO 11 (RX). See
[Lattice Bajoran Board GPIO Pin Locations](https://docscontent.nvidia.com/dims4/default/133096d/2147483647/strip/true/crop/2400x1016+0+0/resize/1440x610!/format/webp/quality/90/?url=https%3A%2F%2Fk3-prod-nvidia-docs.s3.us-west-2.amazonaws.com%2Fbrightspot%2Fsphinx%2F0000019b-b3d8-d33f-abbb-fbda31f40000%2Fholoscan%2Fsensor-bridge%2Flatest%2F_images%2FLattice-Bajoran-Board-front.png)
to identify the pins.

Two modes are supported: **one-way** (`tx`/`rx`), with one board sending and the other
receiving, and **dual**, where both boards send and receive.

**One-way (tx/rx)** — one board transmits, one receives:

```sh
# Terminal 1 (receiver):
$ python3 examples/uart_dual_board_loopback.py --mode rx --hololink 192.168.2.2

# Terminal 2 (transmitter):
$ python3 examples/uart_dual_board_loopback.py --mode tx --hololink 192.168.0.2
```

**Dual mode** — both boards transmit and receive:

```sh
# Terminal 1 (board 1):
$ python3 examples/uart_dual_board_loopback.py --mode dual --hololink 192.168.0.2 --test-string "HELLO" --expected-rx-string "WORLD"

# Terminal 2 (board 2):
$ python3 examples/uart_dual_board_loopback.py --mode dual --hololink 192.168.2.2 --test-string "WORLD" --expected-rx-string "HELLO"
```

Pass `--mode` (`tx`, `rx`, or `dual`), `--hololink` for each board's IP, and optionally
`--test-string`, `--expected-rx-string`, and `--flow-control` in dual mode. Run with
`--help` for all options.

The UART FIFO size is 256 bytes. Both applications chunk data accordingly and verify
transmitted vs. received data.

## Sub-Frame Processing Examples

Sub-frame processing allows high-resolution sensor data frames to be processed and
displayed incrementally as they arrive, reducing latency and enabling progressive
display. See the
[Sub-Frame Processing Applications](applications.md#sub-frame-processing-applications)
section for detailed information about how sub-frame processing works.

### IMX274 Sub-Frame Player Example

The sub-frame IMX274 player example demonstrates processing and displaying camera frames
as sub-frames rather than complete frames. This enables lower latency visualization for
high-resolution cameras.

`````{tab-set}
````{tab-item} Python
```sh
# With ConnectX accelerated network controller
$ python3 examples/sub_frame_imx274_player.py --sub-frame-rows 540

# For unaccelerated configurations (e.g. AGX Orin, AGX Thor)
$ python3 examples/linux_sub_frame_imx274_player.py --sub-frame-rows 540
```

The `--sub-frame-rows` parameter specifies the number of rows per sub-frame.

**NOTE:** `sub_frame_rows` must evenly divide the frame height. For example, a valid value would be
`--sub-frame-rows 540` (2160 ÷ 540 = 4 sub-frames).

````
````{tab-item} C++

The C++ examples need to be built first using these commands:

```sh
$ export BUILD_DIR=$(pwd)/build
$ cd $BUILD_DIR
$ cmake ..
$ make -j
```

After examples are built, you can run the sub-frame IMX274 player:

```sh
# With ConnectX accelerated network controller
$ $BUILD_DIR/examples/sub_frame_imx274_player --sub-frame-rows 540

# For unaccelerated configurations
$ $BUILD_DIR/examples/linux_sub_frame_imx274_player --sub-frame-rows 540
```

````
`````

### VB1940 Sub-Frame Visualizer Example

The sub-frame VB1940 player example demonstrates sub-frame processing with
display-synchronized capture using the `SubFrameVisualizerOp`. Sub-frames are visualized
incrementally as they arrive. Each sub-frame is composited into its position in the
display frame without waiting for the full frame to complete. Capture timing is
synchronized to the display refresh using the FPGA PTP/PPS output, minimizing end-to-end
latency.

```sh
$ $BUILD_DIR/examples/sub_frame_vb1940_player --sub-frame-rows 248
```

The `--sub-frame-rows` parameter specifies the number of rows per sub-frame. The VB1940
default mode is 2560×1984 at 60fps.

**NOTE:** `sub_frame_rows` must evenly divide the frame height. For 1984 rows, valid
values include 248 (8 sub-frames), 496 (4 sub-frames), and 992 (2 sub-frames).

Additional options:

- **`--fullscreen`**: Run in fullscreen mode (default: true)
- **`--exclusive-display`**: Use exclusive display mode for lower latency
- **`--camera-mode`**: Select VB1940 sensor mode (default:
  `VB1940_MODE_2560X1984_60FPS`)
- **`--use-sensor`**: Select sensor index when two sensors are connected (0 or 1,
  default: 0)
- **`--frame-limit`**: Exit after receiving this many frames (default: run indefinitely)

See [SubFrameVisualizerOp](applications.md#subframevisualizierop) in the applications
guide for a detailed description of how display-synchronized capture works.

### Sub-Frame Processing Modes

Sub-frame examples support two display modes:

- **Sub-Frame Combiner Mode** (IMX274 examples): Sub-frames are accumulated by
  `SubFrameCombinerOp` into a complete frame buffer before being passed to `HolovizOp`
  for display. Capture timing is independent of the display refresh rate.

- **Sub-Frame Visualizer Mode** (VB1940 example): Sub-frames are composited directly
  onto the display by `SubFrameVisualizerOp` as they arrive. Capture is synchronized to
  the display FPO (First Pixel Out) event via PTP/PPS, so each display refresh shows the
  latest available data with minimum latency.

### Sub-Frame Configuration Parameters

- **`sub_frame_rows`**: Number of rows per sub-frame. Set to 0 to disable sub-frame
  processing (full-frame mode). **Must evenly divide the frame height.**

### Performance Considerations

- **Memory**: Smaller sub-frames reduce memory requirements but increase processing
  overhead. The `SubFrameCombinerOp` waits for all sub-frames to arrive before emitting
  a complete frame, or emits an incomplete frame if a sub-frame is dropped and the next
  frame starts.

- **Network**: Sub-frame size should align with network packet sizes to minimize partial
  sub-frames and improve efficiency.
