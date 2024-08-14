# Running Holoscan Sensor Bridge examples

Holoscan sensor bridge Python example applications are located under the `examples`
directory.

The C++ examples need to be build first using these commands

```sh
$ export BUILD_DIR=/tmp/build
$ cmake -S . -B $BUILD_DIR -G Ninja -DHOLOLINK_BUILD_PYTHON=OFF
$ cmake --build $BUILD_DIR -j $(nproc)
```

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
````
````{tab-item} C++
```sh
$ $BUILD_DIR/examples/imx274_player
```
````
`````

or, for unaccelerated configurations (e.g. AGX),

```sh
$ python3 examples/linux_imx274_player.py
```

Documentation breaking down the source code for the IMX274 player application is
[available here](applications.md#imx274_player); this example illustrates the basic
sensor bridge workflow which is described in the
[architecture documentation](architecture.md). Press Control/C to stop the video player.

## Running the TAO PeopleNet example

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

## Running the body pose example

**Prerequisite**: Download the YOLOv8 ONNX model from the YOLOv8 website and generate
the body pose ONNX model. Within the Holoscan sensor bridge demo container:

From the repo base directory `holoscan-sensor-bridge`:

```sh
apt-get update && apt-get install -y ffmpeg
pip3 install ultralytics onnx
cd examples
yolo export model=yolov8n-pose.pt format=onnx
cd -
```

Note that this conversion step only needs to be executed once; the `yolov8n-pose.onnx`
file contains the converted model and is all that's needed for the demo to run. The
installed components will be forgotten when the container is exited; those do not need
to be present in future runs of the demo.

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
net model. The first time the body pose example is run, the model is converted to an
fp32 file, which can take several minutes. These conversion results are cached in a
local file and reused on subsequent runs of the example program. For more information
about this application, look [here](applications.md#body_pose_estimation).

Press Ctrl/C to exit.

## Running the Stereo IMX274 example

`examples/stereo_imx274_player.py` shows an example with two independent pipelines, one
for each camera on the dual-camera module. Only an accelerated version is included, and
[both network ports must be connected](sensor_bridge_hardware_setup.md#connecting-holoscan-sensor-bridge-to-the-host)
between the IGX and the Holoscan sensor bridge unit.

```sh
$ python3 examples/stereo_imx274_player.py
```

This brings up a visualizer display with two frames, one for the left channel and the
other for the right.

## Running the GPIO example

`examples/gpio_example_app.py` is a simple example of using the GPIO interface of the
sensor bridge to set GPIO directions, read input values from GPIO pins and write output
values to GPIO pins. To run the appliction:

```sh
$ python3 examples/gpio_example_app.py
```

This brings up a textual display which cycles over different pre-set pin configurations
and alows time between different settings of the pins to measure or readback pins
values. Please refer to the application structure section to read more about the
[GPIO example application](applications.md#gpio-example-application).

## Running the NVIDIA ISP with live capture example

`examples/linux_hwisp_player.py` shows an example of NVIDIA ISP unit processing the
Bayer frame captured live using IMX274. The ISP unit currently is available on Jetson
Orin AGX and IGX Orin in iGPU configuration.

Before starting the docker run, setup the `nvargus-daemon` with the flag
`enableRawReprocessing=1`. This enables us to run the ISP with the Bayer frame capture
using Holoscan sensor bridge unit and this change persists through even restart.

```sh
sudo su
pkill nvargus-daemon
export enableRawReprocessing=1
nvargus-daemon
exit
```

Now use following commands to run the example.

```sh
$ export DISPLAY=:1
$ xhost +
$ sh ./docker/demo.sh
$ python3 examples/linux_hwisp_player.py
```

This will run the application with visualizer display showing the live capture.

Note if user wishes to undo running the `nvargus-daemon` with
flag`enableRawReprocessing=1`, then please execute following command.

```sh
sudo su
pkill nvargus-daemon
unset enableRawReprocessing
nvargus-daemon
exit
```
