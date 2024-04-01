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
    r[RoceReceiverOperator] --> c[CsiToBayerOp]
    c --> i[ImageProcessorOp]
    i --> d[BayerDemosaicOp]
    d --> g[GammaCorrectionOp]
    g --> v[HolovizOp]
```

- `RoceReceiverOperator` blocks until an end-of-frame UDP message is received. On
  return, the received frame data is available in GPU memory, along with metadata which
  is published to the application layer. Holoscan sensor bridge uses
  [RoCE v2](https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet) to transmit data
  plane traffic over UDP; this is why the receiver is called `RoceReceiverOperator`.
- `CsiToBayerOp` is aware that the received data is a CSI-2 RAW10 image, which it
  translates into a [bayer video frame](https://en.wikipedia.org/wiki/Bayer_filter).
  Each pixel color component in this image is decoded and stored as a uint16 value. For
  more information about RAW10, see the
  [MIPI CSI-2 specification](https://www.mipi.org/specifications/csi-2).
- `ImageProcessorOp` adjusts the received bayer image color and brightness to make it
  acceptable for display.
- `BayerDemosaicOp` converts the bayer image data into RGBA.
- `GammaCorrectionOp`
  [adjusts the luminance](https://en.wikipedia.org/wiki/Gamma_correction) of the RGBA
  image to improve human perception.
- `HolovizOp` displays the RGBA image on the GUI.

For each step in the pipeline, the image data is stored in a buffer in GPU memory.
Pointers to that data are passed between each element in the pipeline, avoiding
expensive memory copies between host and GPU memory. GPU acceleration is used to perform
each operator's function, resulting in very low latency operation.

imx274_player.py initializes the sensor bridge device, camera, and pipeline in this way.
To enhance readability, some details are skipped--be sure and check the actual example
code for more details.

```python

  import hololink as hololink_module

  def main():
      # Get handles to GPU
      cuda.cuInit(0)
      cu_device_ordinal = 0
      cu_device = cuda.cuDeviceGet(cu_device_ordinal)
      cu_context = cuda.cuCtxCreate(0, cu_device)

      # Look for sensor bridge enumeration messages; return only the one we're looking for
      channel_metadata = hololink_module.HololinkEnumerator.find_channel(channel_ip="192.168.0.2")
      # Use that enumeration data to instantiate a data receiver object
      hololink_channel = hololink_module.HololinkDataChannel(channel_metadata)

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

Important details:

- `HololinkEnumerator.find_channel` blocks the caller until an enumeration message that
  matches the given criteria is found. If no matching device is found, this method will
  time out (default 20 seconds) and raise an exception. Holoscan sensor bridge
  enumeration messages are sent once per second.
- Holoscan sensor bridge devices transmit enumeration messages for each data plane
  controller, which currently correspond directly with each sensor bridge Ethernet
  interface. If both interfaces on a device are connected to a host, the host will
  receive a pair of distinct enumeration messages, one for each data port, from the same
  sensor bridge device.
- Enumeration messages are sent to the local broadcast address, and routers are not
  allowed to forward these local broadcast messages to other networks. You must have a
  local connection between the host and the sensor bridge device in order to enumerate
  it.
- `HololinkEnumerator.find_channel` returns a dictionary of name/value pairs containing
  identifying information about the data port being discovered, including MAC ID, IP
  address, versions of all the programmable components within the device, device serial
  number, and which specific instance this data port controller is within the device.
  While the IP address may change, the MAC ID, serial number, and data plane controller
  instance are constant. The host does not need to request any of the data included in
  this dictionary; its all broadcast by the sensor bridge device.
- `HololinkDataChannel` is the local controller for a data plane on the sensor bridge
  device. It contains the APIs for configuring the target addresses for packets
  transmitted on that data plane-- this is used by the receiver operator, described
  below.
- In this example, the `camera` object provides most of the APIs that the application
  layer would access. When the application configures the camera, the camera object
  knows how to work with the various sensor bridge controller objects to properly
  configure `HololinkDataChannel`.
- Usually there are multiple `HololinkDataChannel` instances on a single `Hololink`
  sensor bridge device, and many APIs on the `Hololink` device will affect all the
  `HololinkDataChannel` objects on that same device. In this example, calling
  `hololink.reset` will reset all the data channels on this device; and in the stereo
  IMX274 configuration, calling `camera.setup_clock` sets the clock that is shared
  between both cameras. For this reason, it's important that the application is careful
  about calling `camera.setup_clock`--resetting the clock while a camera is running can
  lead to undefined states.

Holoscan, on the call to `application.run`, invokes the application's `compose` method,
which includes this:

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
        receiver_operator = hololink_module.operators.RoceReceiverOperator(
            hololink_channel,
            frame_size, ...)
        ...
        # Use add_flow to connect the operators together:
        ...
        #   receiver_operator.compute() will be followed by csi_to_bayer_operator.compute()
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        ...
```

Some key points:

- `receiver_operator` has no idea it is dealing with video data. It's just informed of
  the memory region(s) to fill and the size of a block of data. When a complete block of
  data is received, the CPU will be notified so that pipeline processing can continue.
- Applications can pass in a frame buffer device memory pointer to the constructor for
  `receiver_opearator`, or the receiver will allocate one for you. When it allocates a
  buffer, it can take into account special requirements for various configurations of
  GPU and RDMA controller.
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
- To support RDMA, the receiver operator is given a single block of memory (instead of a
  memory pool to allocate from). The peripheral component is granted access to this
  region only, and that region does not change throughout the life of the sensor bridge
  application.
- the sensor bridge device, following configuration by the `holoscan_channel` object,
  will start forwarding all received sensor data to the configured receiver. We haven't
  instructed the camera to start streaming data yet, but at this point, we're ready to
  receive it.
- `receiver_operator` keeps track of a `device` parameter, which in this application is
  our camera. When `receiver_operator.start` is called, it will call `device.start,`
  which in our IMX274 implementation, will instruct the camera to begin streaming data.

In this example, `receiver_operator` is a `RoceReceiverOperator` instance, which takes
advantage of the RDMA acceleration features present in the ConnectX firmware. With
`RoceReceiverOperator`, the CPU only sees an interrupt when the last packet for the
frame is received--all frame data sent before that is written to GPU memory in the
background. In systems without ConnectX devices, `LinuxReceiverOperator` provides the
same functionality but uses the host CPU and Linux kernel to receive the ingress UDP
requests; and the CPU writes that payload data to GPU memory. This provides the same
functionality as `RoceReceiverOperator` but at considerably lower performance.

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
    r[RoceReceiverOperator] --> c[CsiToBayerOp]
    c --> i[ImageProcessorOp]
    i --> d[BayerDemosaicOp]
    d --> g[GammaCorrectionOp]
    g --> s[ImageShiftToUint8Operator]
    s --> p[FormatConverterOp]
    p --> fi[FormatInferenceInputOp]
    fi --> in[InferenceOp]
    in --> pf[PostprocessorOp]
    s -- live video --> v[HolovizOp]
    pf -- overlay --> v
```

Adding inference to the video pipeline is easy: just add the appropriate operators and
data flows. In our case, we use the video mixer built in to `HolovizOp` to display the
overlay generated by inference. `RoceReceiverOperator` is specified to always provide
the most recently received video frame, so if a pipeline takes more than one frame time
to complete, the next iteration through the loop will always work on the most recently
received video frame.

## body_pose_estimation

The Body Pose Estimation application takes input from a live video, performs inference
using YOLOv8 pose model, and then shows keypoints overlayed onto the original video. The
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
    r[RoceReceiverOperator] --> c[CsiToBayerOp]
    c --> i[ImageProcessorOp]
    i --> d[BayerDemosaicOp]
    d --> g[GammaCorrectionOp]
    g --> s[ImageShiftToUint8Operator]
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
cameras. In examples/stereo_imx274_player.py, the same pipeline for live video feed is
presented, except that it is instantiated twice, once for each camera on the IMX274
stereo camera board. In this case, Holoscan cycles between each pipeline, providing two
separate windows (one for each visualizer) on the display. Each `receiver_operator`
instance is independent and runs simultaneously.
