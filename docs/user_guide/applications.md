# Application structure

[Holoscan](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_core.html)
applications are built by specifying sequences of operators. Connecting the output of
one operator to the input of another operator (via the `add_flow` API) configures
Holoscan's pipeline and specifies when individual operators can run.

Holoscan sensor bridge leverages this framework by providing operators and objects that
send and receive data in Holoscan applications. There are additional operators for
converting application specific data (e.g. CSI-2 formatted video data) into formats that
are acceptable inputs for other standard Holoscan operators. To see how a sensor bridge
application works, we'll step through the example IMX274 player.

## imx274_player

The application in examples/imx274_player.py configures the following pipeline. When a
loop through the pipeline finishes, execution restarts at the top, where new data is
acquired and processed.

```{mermaid}
:align: center
:caption: IMX274 Player

%%{init: {"theme": "base", "themeVariables": { }} }%%

graph
    r[RoceReceiverOp] --> c[CsiToBayerOp]
    c --> i[ImageProcessorOp]
    i --> d[BayerDemosaicOp]
    d --> v[HolovizOp]
```

- `RoceReceiverOp` wakes up when an end-of-frame UDP message is received. When it
  finishes, the received frame data is available in GPU memory, along with metadata
  which is published to the application layer. Holoscan sensor bridge uses
  [RoCE v2](https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet) to transmit data
  plane traffic over UDP; this is why the receiver is called `RoceReceiverOp`.
- `CsiToBayerOp` is aware that the received data is a CSI-2 RAW10 image, which it
  translates into a [bayer video frame](https://en.wikipedia.org/wiki/Bayer_filter).
  Each pixel color component in this image is decoded and stored as a uint16 value. For
  more information about RAW10, see the
  [MIPI CSI-2 specification](https://www.mipi.org/specifications/csi-2).
- `ImageProcessorOp` adjusts the received bayer image color and brightness to make it
  acceptable for display.
- `BayerDemosaicOp` converts the bayer image data into RGBA.
- `HolovizOp` displays the RGBA image on the GUI.

For each step in the pipeline, the image data is stored in a buffer in GPU memory.
Pointers to that data are passed between each element in the pipeline, avoiding
expensive memory copies between host and GPU memory. GPU acceleration is used to perform
each operator's function, resulting in very low latency operation.

The Python `imx274_player.py` and C++ `imx274_player.cpp` files initialize the sensor
bridge device, camera, and pipeline in this way. To enhance readability, some details
are skipped--be sure and check the actual example code for more details.

`````{tab-set}
````{tab-item} Python
```python
import hololink as hololink_module

def main():
    # Get handles to GPU
    cuda.cuInit(0)
    cu_device_ordinal = 0
    cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)

    # Look for sensor bridge enumeration messages; return only the one we're looking for
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip="192.168.0.2")
    # Use that enumeration data to instantiate a data receiver object
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    # Now that we can communicate, create the camera controller
    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(hololink_channel, ...)

    # Set up our Holoscan pipeline
    application = HoloscanApplication(cu_context, cu_device_ordinal, camera, hololink_channel, ...)
    application.config(...)

    # Connect and initialize the sensor bridge device
    hololink = hololink_channel.hololink()
    hololink.start()  # Establish a connection to the sensor bridge device
    hololink.reset()  # Drive the sensor bridge to a known state

    # Configure the camera for 4k at 60 frames per second
    camera_mode = imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS
    camera.setup_clock()
    camera.configure(camera_mode)

    # Run our Holoscan pipeline
    application.run()  # we don't usually return from this call.
    hololink.stop()
```
````
````{tab-item} C++
```cpp
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>

int main(int argc, char** argv)
{
  // Get handles to GPU
  cuInit(0);
  int cu_device_ordinal = 0;
  CUdevice cu_device;
  cuDeviceGet(&cu_device, cu_device_ordinal);
  CUcontext cu_context;
  cuCtxCreate(&cu_context, 0, cu_device);

  // Look for sensor bridge enumeration messages; return only the one we're looking for
  hololink::Metadata channel_metadata = hololink::Enumerator::find_channel(hololink_ip);
  // Use that enumeration data to instantiate a data receiver object
  hololink::DataChannel hololink_channel(channel_metadata);

  // Import the IMX274 sensor module and the IMX274 mode
  py::module_ imx274 = py::module_::import("hololink.sensors.imx274");
  py::object Imx274Cam = imx274.attr("dual_imx274").attr("Imx274Cam");

  // Now that we can communicate, create the camera controller
  py::object camera = Imx274Cam("hololink_channel"_a = hololink_channel, ...);

  // Set up our Holoscan pipeline
  auto application = holoscan::make_application<HoloscanApplication>(...)
  application->config(...)

  // Connect and initialize the sensor bridge device
  std::shared_ptr<hololink::Hololink> hololink = hololink_channel.hololink();
  hololink->start(); // Establish a connection to the sensor bridge device
  hololink->reset(); // Drive the sensor bridge to a known state

  // Configure the camera for 4k at 60 frames per second
  camera.attr("setup_clock")();
  camera.attr("configure")(Imx274_Mode(0));

  // Run our Holoscan pipeline
  application->run(); // we don't usually return from this call.
  hololink->stop();
}
```
````
`````

Important details:

- `Enumerator.find_channel` blocks the caller until an enumeration message that matches
  the given criteria is found. If no matching device is found, this method will time out
  (default 20 seconds) and raise an exception. Holoscan sensor bridge enumeration
  messages are sent once per second.
- Holoscan sensor bridge devices transmit enumeration messages for each data plane
  controller, which currently correspond directly with each sensor bridge Ethernet
  interface. If both interfaces on a device are connected to a host, the host will
  receive a pair of distinct enumeration messages, one for each data port, from the same
  sensor bridge device.
- Enumeration messages are sent to the local broadcast address, and routers are not
  allowed to forward these local broadcast messages to other networks. You must have a
  local connection between the host and the sensor bridge device in order to enumerate
  it.
- `Enumerator.find_channel` returns a dictionary of name/value pairs containing
  identifying information about the data port being discovered, including MAC ID, IP
  address, versions of all the programmable components within the device, device serial
  number, and which specific instance this data port controller is within the device.
  While the IP address may change, the MAC ID, serial number, and data plane controller
  instance are constant. The host does not need to request any of the data included in
  this dictionary; its all broadcast by the sensor bridge device.
- `DataChannel` is the local controller for a data plane on the sensor bridge device. It
  contains the APIs for configuring the target addresses for packets transmitted on that
  data plane-- this is used by the receiver operator, described below.
- In this example, the `camera` object provides most of the APIs that the application
  layer would access. When the application configures the camera, the camera object
  knows how to work with the various sensor bridge controller objects to properly
  configure `DataChannel`.
- Usually there are multiple `DataChannel` instances on a single `Hololink` sensor
  bridge device, and many APIs on the `Hololink` device will affect all the
  `DataChannel` objects on that same device. In this example, calling `hololink.reset`
  will reset all the data channels on this device; and in the stereo IMX274
  configuration, calling `camera.setup_clock` sets the clock that is shared between both
  cameras. For this reason, it's important that the application is careful about calling
  `camera.setup_clock`--resetting the clock (e.g. on the second image sensor) while the
  first camera is running can lead to undefined states.

Holoscan, on the call to `application.run`, invokes the application's `compose` method,
which includes this:

`````{tab-set}
````{tab-item} Python
```python
class HoloscanApplication(holoscan.core.Application):
    def __init__(self, ..., camera, hololink_channel, ...):
        ...
        self._camera = camera
        self._hololink_channel = hololink_channel
        ...
    def compose(self):
        ...
        # Create the CSI to bayer converter.
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(...)

        # The call to camera.configure(...) earlier set our image dimensions
        # and bytes per pixel.  This call asks the camera to configure the
        # converter accordingly.
        self._camera.configure_converter(csi_to_bayer_operator)

        # csi_to_bayer_operator now knows the image dimensions and bytes per pixel,
        # and can compute the overall size of the received image data.
        frame_size = csi_to_bayer_operator.get_csi_length()

        # Create a receiver object that fills out our frame buffer.  The receiver
        # operator knows how to configure hololink_channel to send its data
        # to us and to provide an end-of-frame indication at the right time.
        receiver_operator = hololink_module.operators.RoceReceiverOp(
            hololink_channel,
            frame_size, ...)
        ...
        # Use add_flow to connect the operators together:
        ...
        #   receiver_operator.compute() will be followed by csi_to_bayer_operator.compute()
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        ...
```
````
````{tab-item} C++
```cpp
class HoloscanApplication : public holoscan::Application {
public:
    explicit HoloscanApplication(..., py::object camera, hololink::DataChannel& hololink_channel, ...)
        : ...
        , camera_(camera)
        , hololink_channel_(hololink_channel)
        ...
    {
    }

    void compose() override
    {
        ...
        // Create the CSI to bayer converter.
        auto csi_to_bayer_operator = make_operator<hololink::operators::CsiToBayerOp>(...);

        // The call to camera.attr("configure")(...) earlier set our image dimensions
        // and bytes per pixel.  This call asks the camera to configure the
        // converter accordingly.
        camera_.attr("configure_converter")(csi_to_bayer_operator);

        // csi_to_bayer_operator now knows the image dimensions and bytes per pixel,
        // and can compute the overall size of the received image data.
        const size_t frame_size = csi_to_bayer_operator->get_csi_length();

        // Create a receiver object that fills out our frame buffer.  The receiver
        // operator knows how to configure hololink_channel to send its data
        // to us and to provide an end-of-frame indication at the right time.
        auto receiver_operator = make_operator<hololink::operators::RoceReceiverOp>(
            holoscan::Arg("hololink_channel", &hololink_channel_),
            holoscan::Arg("frame_size", frame_size), ...);
        ...
        // Use add_flow to connect the operators together:
        ...
        //   receiver_operator.compute() will be followed by csi_to_bayer_operator.compute()
        add_flow(receiver_operator, csi_to_bayer_operator, { { "output", "input" } });
        ...
    }

private:
    const py::object camera_;
    hololink::DataChannel& hololink_channel_;
};
```
````
`````

Some key points:

- `receiver_operator` has no idea it is dealing with video data. It's just informed of
  the memory region(s) to fill and the size of a block of data. When a complete block of
  data is received, the CPU will be notified so that pipeline processing can continue.
- Given an expected frame size, the receiver buffer will allocate GPU memory large
  enough for the received data plus additional metadata; that memory is allocated in a
  way that meets hardware and subsequent operator requirements.
- `csi_to_bayer_operator` is aware of memory layout for CSI-2 formatted image data. Our
  call to `camera.configure_converter` allows the camera to communicate the image
  dimensions and pixel depth; with that knowledge, the call to
  `csi_to_bayer_operator.get_csi_length` can return the size of the memory block
  necessary to manage these images. This memory size includes not only the image data
  itself, but CSI-2 metadata, and GPU memory alignment requirements. Because
  CsiToBayerOp is a GPU accelerated function, it may have special memory requirements
  that the camera sensor object is not aware of.
- `receiver_operator` coordinates with `holoscan_channel` to configure the sensor bridge
  data plane. Configuration automatically handles setting the sensor bridge device with
  our host Ethernet and IP addresses, destination memory addresses, security keys, and
  frame size information.
- the sensor bridge device, following configuration by the `holoscan_channel` object,
  will start forwarding all received sensor data to the configured receiver. We haven't
  instructed the camera to start streaming data yet, but at this point, we're ready to
  receive it.
- `receiver_operator` keeps track of a `device` parameter, which in this application is
  our camera. When `receiver_operator.start` is called, it will call `device.start,`
  which in our IMX274 implementation, will instruct the camera to begin streaming data.

In this example, `receiver_operator` is a `RoceReceiverOp` instance, which takes
advantage of the RDMA acceleration features present in the ConnectX firmware. With
`RoceReceiverOp`, the CPU only sees an interrupt when the last packet for the frame is
received--all frame data sent before that is written to GPU memory in the background. In
systems without ConnectX devices, `LinuxReceiverOperator` provides the same
functionality but uses the host CPU and Linux kernel to receive the ingress UDP
requests; and the CPU writes that payload data to GPU memory. This provides the same
functionality as `RoceReceiverOp` but at considerably lower performance.

## tao_peoplenet

A demonstration application where inference is used to generate a video overlay is
included in examples/tao_peoplenet.py. The
[Tao PeopleNet](https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/cv_models/peoplenet.html)
is used to determine the locations of persons, bags, and faces in the live video stream.
This example program draws bounding boxes on an overlay illustrating where those objects
are detected in the video frame.

Pipeline structure:

```{mermaid}
:align: center
:caption: IMX274 Player with Inference

%%{init: {"theme": "base", "themeVariables": { }} }%%

graph
    r[RoceReceiverOp] --> c[CsiToBayerOp]
    c --> i[ImageProcessorOp]
    i --> d[BayerDemosaicOp]
    d --> s[ImageShiftToUint8Operator]
    s --> p[FormatConverterOp]
    p --> fi[FormatInferenceInputOp]
    fi --> in[InferenceOp]
    in --> pf[PostprocessorOp]
    s -- live video --> v[HolovizOp]
    pf -- overlay --> v
```

Adding inference to the video pipeline is easy: just add the appropriate operators and
data flows. In our case, we use the video mixer built in to `HolovizOp` to display the
overlay generated by inference. `RoceReceiverOp` is specified to always provide the most
recently received video frame, so if a pipeline takes more than one frame time to
complete, the next iteration through the loop will always work on the most recently
received video frame.

## body_pose_estimation

The Body Pose Estimation application takes input from a live video, performs inference
using YOLOv8 pose model, and then shows keypoints overlaid onto the original video. The
keypoints are:

\[nose, left eye, right eye, left ear, right ear, left shoulder, right shoulder, left
elbow, right elbow, left wrist, right wrist, left hip, right hip, left knee, right knee,
left ankle, right ankle\]

This application's pipeline is the same as the People Detection application:

```{mermaid}
:align: center
:caption: Body Pose Estimation

%%{init: {"theme": "base", "themeVariables": { }} }%%

graph
    r[RoceReceiverOp] --> c[CsiToBayerOp]
    c --> i[ImageProcessorOp]
    i --> d[BayerDemosaicOp]
    d --> s[ImageShiftToUint8Operator]
    s --> p[FormatConverterOp]
    p --> fi[FormatInferenceInputOp]
    fi --> in[InferenceOp]
    in --> pf[PostprocessorOp]
    s -- live video --> v[HolovizOp]
    pf -- overlay --> v

```

The difference is that InferenceOp uses the YOLOv8 pose model and the post processor
operator has logic that performs postprocessing specific to the YOLOv8 model.
Specifically, it takes the output of Inference and filters out detections that have low
scores and applies non-max suppression (nms) before sending the output to `HolovizOp`.

## IMX274 Stereo live video demonstration

Multiple receiver operators can be instantiated to support data feeds from multiple
cameras. In `examples/stereo_imx274_player.py`, the same pipeline for live video feed is
presented, except that it is instantiated twice, once for each camera on the IMX274
stereo camera board. In this case, Holoscan cycles between each pipeline, providing two
separate windows (one for each visualizer) on the display. Each `receiver_operator`
instance is independent and runs simultaneously.

For systems with only a single network connection, Holoscan Sensor Bridge can be
configured to transmit both cameras data over the same network connection. The 10Gbps
network port on HSB doesn't have the bandwidth to support two 4K 60FPS video streams, so
support is limited to cameras in 1080p mode. See
`examples/single_network_stereo_imx274_player.py` for an example showing how to
configure HSB to work in this way. As before, each `receiver_operator` is independent,
even when using the same network interface.

## GPIO Example application

This application demonstrates how to utilize the hololink GPIO interface and can be
found under the `hololink/examples` folder.

The hololink GPIO interface supports 16 GPIOs numbered 0..15. These GPIOs can be set as
either **input** or **output** and have 2 logical values:

- **High** - 3.3V can be measured on the GPIO pin
- **Low** - 0V can b measured on the GPIO pin

The following image maps the GPIO and ground pins on the hololink board:

<img src="sensor_bridge_board_gpios.png" alt="GPIO Interface" width="100%"/>

As can be observed from the image:

- The 4 pins in the corners of the connector are ground pins (marked 'G' in the image
  above)
- The lower set of pins between the 2 lower ground pins are GPIO pins numbered 0 to 7
- The upper set of pins between the 2 upper ground pins are GPIO pins numbered 8 to 15

### GPIO Example application flow

The GPIO Example is a simple application made up of 2 operators as can be seen in the
following diagram:

```{mermaid}
:align: center
:caption: GPIO Example Flow

%%{init: {"theme": "base", "themeVariables": { }} }%%

graph
    w[GpioSetOp] --> r[GpioReadOp]
```

- **GPIO Set Operator** - this operator sweeps through the 16 GPIO pins setting them one
  by one to values and directions defined per 5 different pin configurations:

1. **ALL_OUT_L**- All pins output low
1. **ALL_OUT_H**- All pins output high
1. **ALL_IN** - All pins input
1. **ODD_OUT_H**- Odd pins output high, even pins input
1. **EVEN_OUT_H** -Even pins output high, odd pins input

Each cycle of this operator configures one pin to a direction and value and sends the
last changed pin number and the current running configuration to the GPIO read
operator.\
Once all 16 pins are set per the currently running configuration,the operator will move
on the next cycle to the next configuration.

- **GPIO Read Operator** - This operator reads and displays the current value of the
  last configured pin. It delays 10 seconds to allow the user to validate the pin level
  and direction with an external measurement device like a multimeter or scope.

### GPIO Software interface

The GPIO interface is a class defined within the hololink module. It exports the
following GPIO interface:

1. **get_gpio()** - gets a GPIO interface instance from the hololink module
1. **set_direction( pin, direction )** - sets the pin direction as input or output
1. **get_direction( pin )** - gets the direction set for the pin
1. **set_value( pin, value )** - for pins set as direction **output**, set the value of
   the pin to high or low.
1. **get_value( pin )** - for pins set as direction **input**, reads the value of the
   pin (high or low).

- **pin numbers** - range between 0 to 15.
- **pin direction** - enumerated values: IN-1,OUT-0
- **pin values** - enumerated values: HIGH-1,LOW-0

## NVIDIA ISP for live capture

Jetson boards have built in support for ISP (Image Signal Processing) unit for
processing Bayer images and outputting images in standard color space(s). The ISP is
operational in Jetson Orin AGX and Orin IGX in iGPU configuration.

A sample ISP application presented in `examples/linux_hwisp_player.py` configures the
following pipeline. When a loop through the pipeline finishes, execution restarts at the
top, where new data is acquired and processed.

```{mermaid}
:align: center
:caption: Linux ISP Player

%%{init: {"theme": "base", "themeVariables": { }} }%%

graph

    r[LinuxReceiverOperator] --> c[CsiToBayerOp]
    c --> i[ArgusIspOp]
    i --> v[HolovizOp]
```

`ArgusIspOp` allows the users to access the ISP via Argus API. This operator takes in
Bayer uncompressed image of uint16 per pixel (MSB aligned) and outputs RGB888 image. It
is available as C++ operator with Python bindings.

The `ArgusIspOp` can be configured using following required parameters at the
application level. Below is a snippet from an existing python based example.

`````{tab-set}
````{tab-item} Python
```python
    argus_isp = hololink_module.operators.ArgusIspOp(
        self,
        name="argus_isp",
        bayer_format=bayer_format.value, # RGGB or other Bayer format
        exposure_time_ms=16.67,          # Exposure time in milliseconds. 60fps is 16.67ms
        analog_gain=10.0,                # Minimum Analog Gain
        pixel_bit_depth=10,              # Effective bit depth of input per pixel
        pool=isp_pool,
    )

```
````
`````

The input to the `ArgusIspOp` is a Bayer Image uncompressed to uint16 per pixel. The
values in uint16 should be MSB aligned. For example, if the camera sensor produces
Raw10, the 10bits should be MSB aligned in 16bits. Currently supported output from
`ArgusIspOp` is RGB888 in Rec 709 standard color space and gamma corrected.

The glass to display latency of the pipeline mentioned above on Jetson Orin AGX is 37ms
for a resolution of 1920x1080 at 60 fps.

Please reach out to NVIDIA for further questions on ISP capabilities and usage.

## ECam0M30ToF Player

The `ecam0m30tof_player.py` application demonstrates the use of ECam0M30ToF
time-of-flight camera and showcasing handling of depth and IR data in the pipeline. This
application uses RoCE for high-performance data transmission and supports multiple
camera modes for different use cases.

```{mermaid}
:align: center
:caption: ECam0M30ToF Player

%%{init: {"theme": "base", "themeVariables": { }} }%%

graph
    r[RoceReceiverOp] --> c[CsiToBayerOp]
    c --> i[ImageShiftAndProcessingOperator]
    i --> v[HolovizOp]
```

The `ImageShiftAndProcessingOperator` is a new operator designed to process depth and IR
data. It does following operations:

- Converts depth data to grayscale for visualization
- Handles dual-plane data (depth + IR) when in combined mode
- Performs data format conversion and normalization

The `HolovizOp` provides rendering for both Active IR and depth data, with depth data
visualized using the `DEPTH_MAP` option for 3D rendering.
