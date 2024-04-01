## Installation: IGX system running IGX OS 1.0 DP with CX7 SmartNIC

Here are instructions to set up an IGX system running
[IGX OS 1.0 DP](https://developer.nvidia.com/igx-downloads) with CX7 SmartNIC. While
sensor bridge applications normally run in a container, these settings are all to be
configured in the IGX system directly--not within the container. These instructions are
remembered across power cycles and therefore only need to be set up once.

1. Determine the name of the network device associated with the first CX7 port. This is
   the rightmost QSFP port when looking at the back of the IGX unit.

   <img src="igx-rear-qsfp0.png" alt="IGX QSFP0" width="75%"/>

   ```none
   $ ls /sys/class/infiniband
   mlx5_0 mlx5_1
   ```

   This will produce a list of CX7 ports; your device names may vary. The lowest
   numbered one, in this case `mlx5_0`, is the first CX7 port. Next, determine which
   host ethernet port is associated with that device.

   ```none
   $ ls /sys/class/infiniband/mlx5_0/device/net
   eth0
   ```

   This indicates that the host network interface associated with mlx5_0 is `eth0`; as
   before, your system may produce a different name.

1. IGX OS uses NetworkManager to configure network interfaces. By default, the sensor
   bridge device uses the address 192.168.0.2 for the first port. Set up your first
   ethernet device (e.g. `eth0`) to use the address 192.168.0.101 with a permanent route
   to 192.168.0.2 using the following commands. Replace `eth0` with your network
   interface as necessary.
   ([Here](notes.md#holoscan-sensor-bridge-ip-address-configuration) is more information
   about configuring your system if you cannot use the 192.168.0.0/24 network in this
   way.)

   ```none
   $ sudo nmcli con add con-name hololink-eth0 ifname eth0 type ethernet ip4 192.168.0.101/24
   $ sudo nmcli connection modify hololink-eth0 +ipv4.routes "192.168.0.2/32 192.168.0.101"
   $ sudo nmcli connection up hololink-eth0
   ```

   Apply power to the sensor bridge device, ensure that it's properly connected, then
   `ping 192.168.0.2` to check connectivity:

   ```none
   $ ping 192.168.0.2
   PING 192.168.0.2 (192.168.0.2) 56(84) bytes of data.
   64 bytes from 192.168.0.2: icmp_seq=1 ttl=64 time=0.225 ms
   64 bytes from 192.168.0.2: icmp_seq=2 ttl=64 time=0.081 ms
   64 bytes from 192.168.0.2: icmp_seq=3 ttl=64 time=0.088 ms
   64 bytes from 192.168.0.2: icmp_seq=4 ttl=64 time=0.132 ms
   ^C
   --- 192.168.0.2 ping statistics ---
   4 packets transmitted, 4 received, 0% packet loss, time 3057ms
   rtt min/avg/max/mdev = 0.081/0.131/0.225/0.057 ms
   ```

1. The second SFP+ connector on the sensor bridge device is used to transmit data
   acquired from the second camera on a stereo camera module (like the IMX274). By
   default, the sensor bridge device uses the address 192.168.0.3 for that second port.
   Connect the second IGX QSFP port (indicated with the red arrow below) to the second
   SFP+ port on the sensor bridge device. Use the process above to determine the host
   Ethernet device on the second port (e.g. `mlx5_1` might be associated with `eth1`).

   <img src="igx-rear-qsfp1.png" alt="IGX QSFP1" width="75%"/>

   Use `nmcli` to configure the second QSFP network port with an appropriate address and
   permanent route. Replace `eth1` with your network interface as necessary.

   ```none
   $ sudo nmcli con add con-name hololink-eth1 ifname eth1 type ethernet ip4 192.168.0.102/24
   $ sudo nmcli connection modify hololink-eth1 +ipv4.routes "192.168.0.3/32 192.168.0.102"
   $ sudo nmcli connection up hololink-eth1
   ```

   Now test the second connection with `ping 192.168.0.3`:

   ```none
   $ ping 192.168.0.3
   PING 192.168.0.3 (192.168.0.3) 56(84) bytes of data.
   64 bytes from 192.168.0.3: icmp_seq=1 ttl=64 time=0.210 ms
   64 bytes from 192.168.0.3: icmp_seq=2 ttl=64 time=0.271 ms
   64 bytes from 192.168.0.3: icmp_seq=3 ttl=64 time=0.181 ms
   64 bytes from 192.168.0.3: icmp_seq=4 ttl=64 time=0.310 ms
   64 bytes from 192.168.0.3: icmp_seq=5 ttl=64 time=0.258 ms
   ^C
   --- 192.168.0.3 ping statistics ---
   5 packets transmitted, 5 received, 0% packet loss, time 4102ms
   rtt min/avg/max/mdev = 0.181/0.246/0.310/0.045 ms
   ```

   When the second port is configured, the first port should continue to respond to
   pings as appropriate.

1. Install [git-lfs](https://git-lfs.com)

   [IGX OS 1.0 DP](https://developer.nvidia.com/igx-downloads) does not come with
   `git-lfs` installed; but some data files in the Holoscan sensor bridge source
   repository use LFS. If you plan on using git to fetch the sensor bridge host
   software, you'll need to install git-lfs:

   ```none
   sudo apt-get update
   sudo apt-get install -y git-lfs
   ```

1. For IGX with iGPU only: Install
   [NVIDIA DLA](https://developer.nvidia.com/deep-learning-accelerator) compiler.
   Applications using inference need this at initialization time; the IGX OS image for
   iGPU doesn't include it.

   ```none
   sudo apt update && sudo apt install -y nvidia-l4t-dla-compiler
   ```

To run example sensor bridge applications, continue with instructions on the
[setup page](setup.md).
