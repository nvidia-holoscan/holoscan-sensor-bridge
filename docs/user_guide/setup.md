# Host Setup

Holoscan sensor bridge is supported on the following configurations:

- IGX systems configured with
  [IGX OS 1.1.2 Production Release](https://developer.nvidia.com/igx-downloads) with CX7
  SmartNIC devices.
- AGX Orin systems running [JP6.2.1](https://developer.nvidia.com/embedded/jetpack). In
  this configuration, the on-board Ethernet controller is used with the Linux kernel
  network stack for data I/O; all network I/O is performed by the CPU without network
  acceleration.
- Thor systems running [JP7.0.0](https://developer.nvidia.com/embedded/jetpack) with
  MGBE SmartNIC device and CoE transport. JP7.0.0 release currently supports only the
  [Leopard imaging VB1940 Eagle Camera](sensor_bridge_hardware_setup.md).

After the [Holoscan sensor bridge board is set up](sensor_bridge_hardware_setup.md),
configure a few prerequisites in your host system. While holoscan sensor bridge
demonstration applications usually run in a container, these commands are all to be
executed outside the container, on the host system directly. These configurations are
remembered across power cycles and therefore only need to be set up once.

- Install [git-lfs](https://git-lfs.com)

  Some data files in the Holoscan sensor bridge source repository use GIT LFS.

  ```none
  sudo apt-get update
  sudo apt-get install -y git-lfs
  ```

- Grant your user permission to the docker subsystem:

  ```none
  $ sudo usermod -aG docker $USER
  ```

  Reboot the computer to activate this setting.

Next, follow the directions on the appropriate tab below to configure your Orin host
system.

**For Thor host setup, please follow the instructions on the
[Thor Host Setup](thor-jp7-setup.md) page.**

`````{tab-set}
````{tab-item} IGX

- Determine the name of the network device associated with the first CX7 port. This is
  the rightmost QSFP port when looking at the back of the IGX unit.

  <img src="igx-rear-qsfp0.png" alt="IGX QSFP0" width="75%"/>

  ```none
  $ ls /sys/class/infiniband
  roceP5p3s0f0 roceP5p3s0f1
  ```

  This will produce a list of CX7 ports; your device names may vary. The lowest
  numbered one, in this case `roceP5p3s0f0`, is the first CX7 port.  Let's assign
  that name to the variable `$IN0`.

  ```none
  $ IN=(/sys/class/infiniband/*)
  $ IN0=`basename ${IN[0]}`
  $ echo $IN0
  roceP5p3s0f0
  ```

  Next, determine which host ethernet port is associated with that device, and assign
  that to the variable `$EN0`, which we'll use later during network configuration.

  ```none
  $ EN0=`basename /sys/class/infiniband/$IN0/device/net/*`
  $ echo $EN0
  enP5p3s0f0np0
  ```

  In summary, the host network interface associated with `$IN0` (`roceP5p3s0f0`) is
  `$EN0` (`enP5p3s0f0np0`); your specific device names may vary.

- IGX OS uses NetworkManager to configure network interfaces. By default, the sensor
  bridge device uses the address 192.168.0.2 for the first port. Set up your first
  ethernet device (`$EN0`) to use the address 192.168.0.101 with a permanent route
  to 192.168.0.2: ([Here](notes.md#holoscan-sensor-bridge-ip-address-configuration) is more information
  about configuring your system if you cannot use the 192.168.0.0/24 network in this
  way.)

  ```none
  $ sudo nmcli con add con-name hololink-$EN0 ifname $EN0 type ethernet ip4 192.168.0.101/24
  $ sudo nmcli connection modify hololink-$EN0 +ipv4.routes 192.168.0.2/32
  $ sudo nmcli connection modify hololink-$EN0 ethtool.ring-rx 4096
  $ sudo nmcli connection up hololink-$EN0
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

- The second SFP+ connector on the sensor bridge device is used to transmit data
  acquired from the second camera on a stereo camera module (like the IMX274). By
  default, the sensor bridge device uses the address 192.168.0.3 for that second port.
  Connect the second IGX QSFP port (indicated with the red arrow below) to the second
  SFP+ port on the sensor bridge device.

  <img src="igx-rear-qsfp1.png" alt="IGX QSFP1" width="75%"/>

  Let's refer to these as `$IN1` and `$EN1`.  Given the commands to assign `$IN0` and
  `$EN0` above,

  ```none
  $ IN1=`basename ${IN[1]}`
  $ echo $IN1
  roceP5p3s0f1
  $ EN1=`basename /sys/class/infiniband/$IN1/device/net/*`
  $ echo $EN1
  enP5p3s0f1np1
  ```

  As above, your device names may be different.  Configure the second QSFP network port
  with an appropriate address and permanent route:

  ```none
  $ sudo nmcli con add con-name hololink-$EN1 ifname $EN1 type ethernet ip4 192.168.0.102/24
  $ sudo nmcli connection modify hololink-$EN1 +ipv4.routes 192.168.0.3/32
  $ sudo nmcli connection modify hololink-$EN1 ethtool.ring-rx 4096
  $ sudo nmcli connection up hololink-$EN1
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

````
````{tab-item} AGX

Demos and examples in this package assume a sensor bridge device is connected to `eno1`,
which is the RJ45 connector on the AGX Orin.

- Linux sockets require a larger network receiver buffer.

  Most sensor bridge self-tests use Linux's loopback interface; if the kernel starts
  dropping packets due to out-of-buffer space then these tests will fail.

  ```none
  echo 'net.core.rmem_max = 31326208' | sudo tee /etc/sysctl.d/52-hololink-rmem_max.conf
  sudo sysctl -p /etc/sysctl.d/52-hololink-rmem_max.conf
  ```

- Configure a static IP address of 192.168.0.101 on the on-board network port.

  L4T uses NetworkManager to configure interfaces; by default interfaces are configured
  as DHCP clients. Use the following command to update the IP address to 192.168.0.101.
  ([Here](notes.md#holoscan-sensor-bridge-ip-address-configuration) is more information
  about configuring your system if you cannot use the 192.168.0.0/24 network in this
  way.)

  Note that for AGX running JP6.2.1, the on-board ethernet device is `eno1`; if you're
  running a different configuration, use the appropriate name for the variable EN0:

  ```none
  EN0=eno1
  sudo nmcli con add con-name hololink-$EN0 ifname $EN0 type ethernet ip4 192.168.0.101/24
  sudo nmcli connection up hololink-$EN0
  ```

  Apply power to the sensor bridge device, ensure that it's properly connected, then
  `ping 192.168.0.2` to check connectivity.

- For the Linux socket based examples, isolating a processor core from Linux kernel is
  recommended. For high bandwidth applications, like 4k video acquisition, isolation of
  the network receiver core is required. When an example program runs with processor
  affinity set to that isolated core, performance is improved and latency is reduced.
  By default, sensor bridge software runs the time-critical background network receiver
  process on the third processor core. If that core is isolated from Linux scheduling,
  no processes will be scheduled on that core without an explicit request from the
  user, and reliability and performance is greatly improved.

  Isolating that core from Linux can be achieved by editing
  `/boot/extlinux/extlinux.conf`. Add the setting `isolcpus=2` to the end of the line
  that starts with `APPEND`. Your file should look like something like this:

  ```none
  TIMEOUT 30
  DEFAULT primary

  MENU TITLE L4T boot options

  LABEL primary
        MENU LABEL primary kernel
        LINUX /boot/Image
        ...
        APPEND ${cbootargs} ...<other-settings>... isolcpus=2

  ```

  Sensor bridge applications can run the network receiver process on another core by
  setting the environment variable `HOLOLINK_AFFINITY` to the core it should run on.
  For example, to run on the first processor core,

  ```none
  HOLOLINK_AFFINITY=0 python3 examples/linux_imx274_player.py
  ```

  Setting `HOLOLINK_AFFINITY` to blank will skip any core affinity settings in the
  sensor bridge code.

- Run the "jetson_clocks" tool on startup, to set the core clocks to their maximum.

  ```none
  JETSON_CLOCKS_SERVICE=/etc/systemd/system/jetson_clocks.service
  cat <<EOF | sudo tee $JETSON_CLOCKS_SERVICE >/dev/null
  [Unit]
  Description=Jetson Clocks Startup
  After=nvpmodel.service

  [Service]
  Type=oneshot
  ExecStart=/usr/bin/jetson_clocks

  [Install]
  WantedBy=multi-user.target
  EOF
  sudo chmod u+x $JETSON_CLOCKS_SERVICE
  sudo systemctl enable jetson_clocks.service
  ```

- Set the AGX Orin power mode to 'MAXN' for optimal performance. The setting can be
  changed via L4T power drop down setting found on the upper left corner of the screen:

  <img src="concord_maxn.png" alt="AGX Orin MAXN" width="50%"/>

- Restart the AGX Orin. This allows core isolation and performance settings to take
  effect. If configuring for 'MAXN' performance doesn't request that you reset the
  unit, then execute the reboot command manually:

  ```none
  reboot
  ```


````
`````

Now, for all configurations,

- Enable PTP on $EN0. This synchronizes the timestamps reported with received data with
  the host time.

  Run the `phc2sys` tool at boot time. This synchronizes the clock in $EN0 with the
  system clock. First, install the `linuxptp` tool.

  ```none
  sudo apt update && sudo apt install -y linuxptp
  ```

  Next, set up a systemd service file that will run `phc2sys`.

  ```none
  PHC2SYS_SERVICE=/etc/systemd/system/phc2sys-$EN0.service
  cat <<EOF | sudo tee $PHC2SYS_SERVICE >/dev/null
  [Unit]
  Description=Copy system time to $EN0
  Requires=NetworkManager.service
  After=NetworkManager.service
  After=timemaster.service

  [Service]
  Type=simple
  ExecStartPre=timeout 3m bash -c "until [ \"\$(nmcli -g GENERAL.STATE device show $EN0)\" = \"100 (connected)\" ]; do sleep 1; done"
  ExecStart=/usr/sbin/phc2sys -c $EN0 -s CLOCK_REALTIME -O 0 -S 0.0001

  [Install]
  WantedBy=multi-user.target
  EOF
  ```

  Configure it for execution at startup, and start it now.

  ```none
  sudo chmod u+x $PHC2SYS_SERVICE
  sudo systemctl enable phc2sys-$EN0.service
  sudo systemctl start phc2sys-$EN0.service
  ```

  Next, run `ptp4l` to send PTP SYNC messages to $EN0.

  ```none
  cat <<EOF | sudo tee /etc/linuxptp/hsb-ptp.conf >/dev/null
  # This configuration is appropriate for NVIDIA Holoscan sensor bridge
  # applications, where PTP messages are sent over L2 and a 1/2 second interval.
  [global]
  logSyncInterval -1
  logMinDelayReqInterval -1
  network_transport L2
  EOF
  ```

  Set up a systemd service file for this.

  ```none
  PTP4L_SERVICE=/etc/systemd/system/ptp4l-$EN0.service
  cat <<EOF | sudo tee $PTP4L_SERVICE >/dev/null
  [Unit]
  Description=Send PTP SYNC messages to $EN0
  After=phc2sys-$EN0.service

  [Service]
  Type=simple
  ExecStart=/usr/sbin/ptp4l -i $EN0 -f /etc/linuxptp/hsb-ptp.conf

  [Install]
  WantedBy=multi-user.target
  EOF
  ```

  Finally, run it.

  ```none
  sudo chmod u+x $PTP4L_SERVICE
  sudo systemctl enable ptp4l-$EN0.service
  sudo systemctl start ptp4l-$EN0.service
  ```

- For IGX with iGPU only: Install
  [NVIDIA DLA](https://developer.nvidia.com/deep-learning-accelerator) compiler.
  Applications using inference need this at initialization time; the IGX OS image for
  iGPU doesn't include it.

  ```none
  sudo apt update && sudo apt install -y nvidia-l4t-dla-compiler
  ```

- Log in to Nvidia GPU Cloud (NGC) with your developer account:

  - If you don't have a developer account for NGC please register at
    [https://catalog.ngc.nvidia.com/](https://catalog.ngc.nvidia.com/)

  - Create an API key for your account:
    [https://ngc.nvidia.com/setup/api-key](https://ngc.nvidia.com/setup/api-key)

  - Use your API key to log in to nvcr.io:

    ```none
    $ docker login nvcr.io
    Username: $oauthtoken
    Password: <Your token key to NGC>
    WARNING! Your password will be stored unencrypted in /home/<user>/.docker/config.json.
    Configure a credential helper to remove this warning. See
    https://docs.docker.com/engine/reference/commandline/login/#credentials-store

    Login Succeeded
    ```

Now proceed to [build and test the Holoscan Sensor Bridge container](build.md).
