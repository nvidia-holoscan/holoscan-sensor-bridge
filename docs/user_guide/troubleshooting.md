# Troubleshooting

Additional troubleshooting notes can be found on the
[release notes page](RELEASE_NOTES.md).

## Segmentation fault from Holoscan Visualizer

If the Holoscan visualizer is not able to access the host display, the program will
usually crash with a segmentation fault. Make sure that `xhost +local:docker` is
executed on the host system before running the Holoscan application and make sure the
`DISPLAY` environment variable is set properly in the container where the application is
run.

## Unable to connect to the sensor bridge device

The `hololink enumerate` command, in the demo container, can be used to monitor
enumeration messages sent by the sensor bridge device. If no messages appear, then check
for power to the sensor bridge device, physical connections to the device, and
appropriate network configurations as listed above. `ping 192.168.0.2` and
`ping 192.168.0.3` can be used to check for connectivity. If an HSB device is running an
incompatible FPGA image (e.g. FPGA is 2407 while the host software requires 2412), ping
would be successful but no enumeration data would appear. Firmware version problems can
be solved by [reprogramming your device](sensor_bridge_firmware_setup.md).

## Visualizer display is completely white

If there are no error messages on the application console, then it indicates that the
control plane is able to connect but there is no data being received on the data plane.
For unaccelerated network connections, `tcpdump` can be used to determine if traffic is
being sent from the sensor bridge device. In accelerated network configurations, the
ConnectX NIC hides the data plane traffic from the CPU, so `tcpdump` will not report it.
Instead, you can check the packet receiver counter this way:

```none
cat /sys/class/infiniband/roceP5p3s0f0/ports/1/hw_counters/rx_write_requests
```

or, to see all counters published by the ConnectX driver,

```none
for i in /sys/class/infiniband/roceP5p3s0f0/ports/1/counters/*; do
echo -n $i
echo -n ": "
cat $i
done
```

Use the appropriate value where `roceP5p3s0f0` is shown here. When no data plane
requests are received, be sure and check that the sensor is properly connected to the
sensor bridge board.

## Sensor Bridge LED indications

The Holoscan Sensor Bridge board has two leds that depending on their state have the
following indications:

1. **Both leds are off** - The Holoscan Sensor Bridge Board is not powered.
1. **Both leds are on with green color** - The Holoscan Sensor Bridge Board is powered
   and ready.
1. **One green led or green led blinking** - The Holoscan Sensor Bridge Board is powered
   with incorrect power supply that does not meet the minimum 12V/2A requirements.
1. **One green led and one red led** - FPGA flashing failed, FPGA might need to be
   flashed with a FTDI cable.
