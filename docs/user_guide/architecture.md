# Holoscan sensor bridge software architecture

Holoscan sensor bridge devices provide a high-speed interface between sensor equipment
and GPU accelerated Holoscan applications. Control of peripherals connected to a sensor
bridge device are provided through I2C, SPI, or local bus interfaces on the sensor
bridge board. Interactions with these peripherals take place using network messages that
are referred to as the _control plane_. Data acquired from high-speed sensors is
gathered by the FPGA and forwarded via UDP back to the host, using messages referred to
as the _data plane_.

Holoscan sensor bridge host software provides objects that manage the control plane
messages used to control I2C, SPI, and local bus transactions. Other sensor bridge host
software objects provide network receiver operators which configure and direct data
received by data plane traffic. For systems with ConnectX SmartNIC devices, received
data plane traffic can be transparently written to GPU memory using RDMA. For systems
without ConnectX devices, there is a network receiver object which uses the Linux
Sockets API to provide the same functionality but without the high performance that
offloading packet reception offers.

Holoscan sensor bridge software also includes additional operators for image format
conversion and image signal processing.

## Example application-sensor workflow: Live video

As an introduction to the architecture in the sensor bridge host software, we'll step
through the major aspects of examples/imx274_player.py. In this example, an IMX274
stereo camera unit provides a live video feed for 4k or 1080p video running at 60FPS.

### Applications interact with APIs on sensor objects

Applications typically instantiate and use sensor objects which provide device specific
APIs. For example, a camera object can provide an API for setting its exposure:

`````{tab-set}
````{tab-item} Python
```python
camera.set_exposure(1000)
```
````
````{tab-item} C++
```cpp
camera->set_exposure(1000);
```
````
`````

To instantiate a camera object, application code will typically

- Use `Enumerator.find_channel` to enumerate the sensor bridge devices visible to the
  local system. `find_channel` accepts arguments that filter received messages; when an
  enumeration message that matches the given criteria is found, a dict is returned with
  metadata about the enumerated device.

  `````{tab-set}
  ````{tab-item} Python
  ```python
  channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
  ```
  ````
  ````{tab-item} C++
  ```cpp
  hololink::Metadata channel_metadata = hololink::Enumerator::find_channel(hololink_ip);
  ```
  ````
  `````

  When enumeration data is observed from the given IP address, information about the
  found device is returned into the `channel_metadata` variable.

- Construct a `DataChannel` object using `channel_metadata`. This object connects
  received data with a GPU memory buffer.

  `````{tab-set}
  ````{tab-item} Python
  ```python
  hololink_channel = hololink_module.DataChannel(channel_metadata)
  ```
  ````
  ````{tab-item} C++
  ```cpp
  hololink::DataChannel hololink_channel(channel_metadata);
  ```
  ````
  `````

- Construct our camera sensor object using `hololink_channel`:

  `````{tab-set}
  ````{tab-item} Python
  ```python
  camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(hololink_channel, ...)
  ```
  ````
  ````{tab-item} C++
  Note that in this case the IMX274 sensor is implemented using Python, the code below shows how to create the Python object in C++.
  ```cpp
  py::module_ imx274 = py::module_::import("hololink.sensors.imx274");
  py::object Imx274Cam = imx274.attr("dual_imx274").attr("Imx274Cam");
  py::object camera = Imx274Cam("hololink_channel"_a = hololink_channel, ...);
  ```
  ````
  `````

  Constructing the camera instance does not actually interact with the sensor bridge
  device-- we just store device communication information for later use. The camera
  instance is necessary for our HoloscanApplication's constructor to run; so we're
  motivated to create this object relatively early.

- A single sensor bridge device usually has multiple `DataChannel` instances; many APIs
  affect all data channel instances associated with a specific sensor bridge device. In
  order to reset the sensor bridge device--the only way to guarantee that the device is
  in a known state--we'll get a handle to the underlying `Hololink` instance. All
  `DataChannel` instances on this board will return the same `Hololink` instance here.
  The call to `hololink.reset` will reset all attached data channel instances.

  `````{tab-set}
  ````{tab-item} Python
  ```python
  hololink = hololink_channel.hololink()
  hololink.reset()
  ```
  ````
  ````{tab-item} C++
  ```cpp
  std::shared_ptr<hololink::Hololink> hololink = hololink_channel.hololink();
  hololink->reset();
  ```
  ````
  `````

- In our sample application, we need initialize the camera clock and configure the image
  format the camera will transmit. Note that in the IMX274 stereo camera, the same clock
  is used to drive both camera devices, so care must be taken to ensure you don't
  initialize the camera while the other is in use.

  `````{tab-set}
  ````{tab-item} Python
  ```python
  camera.setup_clock()
  camera_mode = imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS
  camera.configure(camera_mode)
  ```
  ````
  ````{tab-item} C++
  ```cpp
  camera.attr("setup_clock")();
  py::object camera_mode = Imx274_Mode(0);
  camera.attr("configure")(camera_mode);
  ```
  ````
  `````

  In our IMX274 demo, `camera.setup_clock` and `camera.configure` call the sensor bridge
  device's I2C controller objects to write the proper set of device registers.

- Now we're ready to start our application pipeline.

  `````{tab-set}
  ````{tab-item} Python
  ```python
  application.run()
  ```
  ````
  ````{tab-item} C++
  ```cpp
  application->run();
  ```
  ````
  `````

  Unless the pipeline is explicitly stopped, this call to `application.run` will never
  return.

- `application.run` starts with a call to each operator's `start` method.

  The `start` method of both `RoceReceiverOp` and `LinuxReceiverOperator` will call
  `camera.start` (where camera is the device object passed into the constructor's
  `device` parameter). The camera, on a call to `start`, will be configured to start
  sending video data.

- `application.run` then goes into a loop, executing the pipeline, calling each
  operator's `compute` method. The network receiver operator `compute` method blocks
  until a whole data frame is received into the memory block it was initialized with.

Our camera object works with device registers by reads and writes on an I2C bus present
in the sensor bridge device. Suppose our camera is connected to the sensor bridge I2C
controller when bus enable==0. (We'll store that address in the constant
`hololink_module.CAM_I2C_BUS`.) The camera object can fetch a handle to an
`Hololink.I2c` object, with APIs for generating I2C transactions, by calling
`hololink.get_i2c`. If the camera itself responds to an I2C bus address of 0x34 (which
we'll call `CAM_I2C_ADDRESS`), then it can support a `camera.set_register` method this
way:

```python
class Imx274Cam:
    def __init__(
        self,
        hololink_channel,
        i2c_bus=hololink_module.CAM_I2C_BUS,
        ...
    ):
        self._hololink = hololink_channel.hololink()
        self._i2c = self._hololink.get_i2c(i2c_bus)
    ...
    def set_register(self, register, value):
        ...
        self._i2c.i2c_transaction(
            CAM_I2C_ADDRESS,
            write_data,
            read_byte_count,
        )
    def set_exposure(self, value):
        self.set_register(..., value)
```

Sensor objects of all types can be supported this way: interfaces for I2C, SPI, or
sensor bridge local bus can all be controlled by APIs present on the `Hololink`
instance.

### DataChannel enumeration and IP address configuration

Once per second, each _data plane_ instance in a sensor bridge device sends out UDP
enumeration packets; the host uses these to locate accessible devices.
`Enumerator.find_channel` method gathers and decodes these messages and uses that to
generate the dictionary passed back as `channel_metadata`. Holoscan sensor bridge sends
these packets using the local broadcast MAC ID (FF:FF:FF:FF:FF:FF). Routers are not
allowed to forward these messages to other networks, so only locally connected hosts
will receive these. Your host must be connected to the same network as the sensor bridge
device in order to communicate.

Holoscan sensor bridge enumeration messages are based on the BOOTP protocol, and like
BOOTP, they provide a mechanism to reconfigure the IP address of that HSB device. If the
host wishes to reconfigure the IP address of the device, it sends a reply message with a
new IP address to be assigned to that data plane controller. The sensor bridge demo
container includes a command line tool called `hololink` that can be used to assign new
IP addresses to sensor bridge devices:

```none
$ hololink-set-ip b0:4f:13:e0:20:4c 192.168.100.250
```

Your MAC-ID and IP addresses will be different; a list with any number of mac-id and
ip-address pairs can be given. By default, this starts a process that runs forever; on
receipt of an enumeration request with any IP address other than the configured value, a
reply is sent that assigns the configured address to that data channel. Running this as
a daemon is important when resetting the sensor bridge device:

- Application code establishes a connection at the new IP address
- Application executes `hololink.reset`
- The device resets and reverts back to the default IP address
- Application code sees enumeration with the default IP address--which it ignores
- When `hololink-set-ip` sees the enumeration packet with something besides the new IP
  address, it'll send a reply with the new IP address configuration
- Holoscan sensor bridge updates its IP address. Enumeration data will now be sent using
  that new address
- Application code then sees enumeration at the new IP address
- Application reconnects and completes the reset request

Enumeration request and reply packets follow the specification given in
[RFC951](https://datatracker.ietf.org/doc/html/rfc951) with the exception that
enumeration requests are sent by the sensor bridge device on UDP port 12267 and replies
are sent by the host to UDP port 12268.

[Specific information about host network configuration can be found here.](notes.md#holoscan-sensor-bridge-ip-address-configuration)

### Holoscan sensor bridge data channel uses RoCE v2 RDMA write and RDMA write immediate requests

ConnectX SmartNIC firmware has support for handling authenticated
[RoCE v2](https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet) requests without
CPU intervention. Sensor bridge devices leverage this by generating
[RDMA write and RDMA write with immediate requests](https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17/available+communication+operations)
to send data plane content to the host. DataChannel configures the sensor bridge device
with target network addressing, authentication keys, and individual-packet and overall
data-frame sizes. Once configured, the sensor bridge device will send received sensor
data in RDMA write requests with a payload size given by the individual-packet size
value. These requests, on receipt by ConnectX, are written directly into GPU or system
memory--these writes are completely offloaded from the CPU. After the total number of
received bytes reaches the data-frame size, a special metadata packet is sent using an
RDMA write-immediate request. This write-immediate request schedules an interrupt for
the CPU, which is used to flag that the received data is ready for further processing--
this interrupt is what `RoceReceiverOp.compute` waits for on a call to `get_next_frame.`

### Holoscan SDK metadata and HSB

HSB devices send a block of metadata following each received data frame. See the
[description of HSB latency](latency.md) for details on the contents and uses of this
data.
