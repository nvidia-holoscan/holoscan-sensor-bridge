# Troubleshooting

## Segmentation fault from Holoscan Visualizer

If the Holoscan visualizer is not able to access the host display, the program will
usually crash with a segmentation fault. Make sure that `xhost +` is executed on the
host system before running the Holoscan application and make sure the `DISPLAY`
environment variable is set properly in the container where the application is run.

## Unable to connect to the sensor bridge device

The `hololink enumerate` command, in the demo container, can be used to monitor
enumeration messages sent by the sensor bridge device. If no messages appear, then check
for power to the sensor bridge device, physical connections to the device, and
appropriate network configurations as listed above. `ping 192.168.0.2` and
`ping 192.168.0.3` can also be used to check for connectivity.

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
