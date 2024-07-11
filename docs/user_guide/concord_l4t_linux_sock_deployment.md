## <a name="Agx-l4t-sock">Installation: AGX Orin system running Ubuntu or L4T with Linux Sockets</a>

Here are instructions to set up an AGX Orin system running JP6.0 L4T, for use with Linux
sockets. While Holoscan sensor bridge applications normally run in a container, these
settings are all to be configured in the AGX Orin system directly-not within the
container. These instructions are remembered across power cycles and therefore only need
to be set up once. Demos and examples in this package assume a sensor bridge device is
connected to eth0, which is the RJ45 connector on the AGX Orin.

1. Install [git-lfs](https://git-lfs.com).

   L4T does not come with `git-lfs` installed; but some data files in the sensor bridge
   project use LFS. Fetching the sensor bridge host software from git will require that
   git-lfs is installed.

   ```none
   sudo apt-get update
   sudo apt-get install -y git-lfs
   ```

1. Linux sockets require a larger network receiver buffer.

   Most sensor bridge self-tests use Linux's loopback interface; if the kernel starts
   dropping packets due to out-of-buffer space then these tests will fail.

   ```none
   echo 'net.core.rmem_max = 31326208' | sudo tee /etc/sysctl.d/52-hololink-rmem_max.conf
   sudo sysctl -p /etc/sysctl.d/52-hololink-rmem_max.conf
   ```

1. Configure eth0 for a static IP address of 192.168.0.101.

   L4T uses NetworkManager to configure interfaces; by default interfaces are configured
   as DHCP clients. Use the following command to update the IP address to 192.168.0.101.
   ([Here](notes.md#holoscan-sensor-bridge-ip-address-configuration) is more information
   about configuring your system if you cannot use the 192.168.0.0/24 network in this
   way.)

   ```none
   sudo nmcli con add con-name hololink-eth0 ifname eth0 type ethernet ip4 192.168.0.101/24
   sudo nmcli connection up hololink-eth0
   ```

   Apply power to the sensor bridge device, ensure that it's properly connected, then
   `ping 192.168.0.2` to check connectivity.

1. For the Linux socket based examples, isolating a processor core from Linux kernel is
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
         FDT /boot/dtb/kernel_tegra234-p3701-0000-p3737-0000.dtb
         INITRD /boot/initrd
         APPEND ${cbootargs} root=/dev/mmcblk0p1 rw rootwait ...<other-settings>... isolcpus=2

   ```

   Sensor bridge applications can run the network receiver process on another core by
   setting the environment variable `HOLOLINK_AFFINITY` to the core it should run on.
   For example, to run on the first processor core,

   ```none
   HOLOLINK_AFFINITY=0 python3 examples/linux_imx274_player.py
   ```

   Setting `HOLOLINK_AFFINITY` to blank will skip any core affinity settings in the
   sensor bridge code.

1. Run the "jetson_clocks" tool on startup, to set the core clocks to their maximum.

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

1. Set the AGX Orin power mode to 'MAXN' for optimal performance. The setting can be
   changed via L4T power drop down setting found on the upper left corner of the screen:

   <img src="concord_maxn.png" alt="AGX Orin MAXN" width="50%"/>

1. Restart the AGX Orin. This allows core isolation and performance settings to take
   effect. If configuring for 'MAXN' performance doesn't request that you reset the
   unit, then execute the reboot command manually:

   ```none
   reboot
   ```
