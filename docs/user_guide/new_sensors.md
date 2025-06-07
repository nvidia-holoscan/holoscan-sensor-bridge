# Adapting new sensors

Hololink software provides tools necessary to control sensors connected to a Hololink
device. Supporting a new sensor is usually a matter of creating an object which has
methods for the device functions you need, then using those APIs in your application's
operators. Let's do this with an example `MyCamera` object, which has a register (0x100)
that we use to read the device version.

```python
import logging

import hololink as hololink_module


class MyCamera:
    CAMERA_I2C_BUS_ADDRESS = 0x34

    def __init__(self, hololink_channel, hololink_i2c_controller_address):
        # Get handles to these controllers but don't actually talk to them yet
        self._hololink = hololink_channel.hololink()
        self._i2c = self._hololink.get_i2c(hololink_i2c_controller_address)

    def get_version(self):
        VERSION = 0x100
        return self.get_register(VERSION)

    def get_register(self, register):
        # write_buffer will contain the big-endian 2-byte address
        # of the register we're reading.
        write_buffer = bytearray(10)  # must be at least 2
        serializer = hololink_module.Serializer(write_buffer)
        serializer.append_uint16_be(register)
        # send write_buffer to the peripheral device,
        # and return data read back from it.  reply will
        # be a 4 byte buffer, or None if there's a problem
        read_byte_count = 4
        reply = self._i2c.i2c_transaction(
            self.CAMERA_I2C_BUS_ADDRESS,
            serializer.data(),  # same as write_buffer[:serializer.length()]
            read_byte_count
        )
        # deserializer fetches data from reply; this
        # raises an exception if reply is None
        deserializer = hololink_module.Deserializer(reply)
        # Fetch an unsigned 32-bit value stored in big-endian format
        r = deserializer.next_u32_be()
        return r
```

With this, we can create a simple program that reads this version register:

```python
def main():
    # Get a handle to the Hololink port we're connected to.
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip="192.168.0.2")
    hololink_channel = hololink_module.DataChannel(channel_metadata)
    # Instantiate the camera itself; CAM_I2C_BUS is the appropriate bus enable setting
    # for the I2C controller our camera is attached to
    camera = MyCamera(hololink_channel, hololink_module.CAM_I2C_BUS)
    # Establish a connection to the hololink device
    hololink = hololink_channel.hololink()
    hololink.start()
    # Fetch the device version.
    version = camera.get_version()
    logging.info(f"{version=}")
```

Following the call to `hololink.start`, the network control plane is available for
communication. Sensor objects should follow this pattern:

- A `configure` method which uses the control plane to set up the data produced by the
  sensor. This method is called by the application code.
- A `start` method that the data receiver operator calls in its startup, which
  configures the sensor to begin producing data.
- A `stop` method, called when the data receiver is shut down, that stops the sensor
  data flow.
- A `configure_converter` method, which allows the sensor object to configure the next
  element in the application pipeline. This method is called by the application layer
  when the pipeline is being set up.

```python
class MyCamera:
    ...
    def configure(self, mode, ...):
        # Configure the camera for our useful mode by putting appropriate values
        # in the camera registers.
        self.set_register(...)
        self.set_register(...)
        ...

    def start(self):
        # Tell the camera to start streaming data out.
        self.set_register(...)

    def stop(self):
        # Tell the camera to stop streaming data.
        self.set_register(...)

    def configure_converter(self, converter):
        converter.configure(self._width * self._height ...)
    ...
```

The `configure_converter` method allows the sensor to coordinate with the next layer of
the pipeline, where the raw data from the sensor is handled. For example, in a CSI-2
video application, the raw video data is framed by CSI-2 metadata and stored in an
encoded format (e.g. RAW10). Because the converter is a GPU accelerated process, it
likely has additional considerations that must be included when memory is allocated for
the received data (e.g. inclusion of GPU memory cache alignment). In our video
application, the sensor would know what the raw image dimensions would be, and the
converter can add to that the space necessary for CSI-2 framing data. Following that,
the converter object now knows how to best allocate memory for the network receiver.

The application layer, following the call to `sensor.configure_converter`, can now ask
the converter for help in allocating GPU memory. This memory is then passed to the
network receiver operator.

With the camera now configured to send traffic over the data plane, the application can
instantiate a `RoceReceiverOp` (or `LinuxReceiverOperator`) to receive data plane
traffic to the specific region of GPU memory. Finally, when `application.run` finishes
configuration, and calls our receiver operator's `start` method, it will call
`camera.start`, which updates the camera to start sending video data.
