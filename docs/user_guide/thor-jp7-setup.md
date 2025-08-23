# Thor host setup

After installing JetPack 7.0.0 on the Thor devkit, please follow these steps to set up
your Thor for running Holoscan sensor bridge examples.

- Install Holoscan SDK v3.5.1

```none
  echo "deb https://repo.download.nvidia.com/jetson/jetson-4fed1671 r38.1 main" | sudo tee /etc/apt/sources.list.d/nvidia-l4t-apt-source-jpea.list
  sudo apt update
  sudo apt install holoscan
```

- Install other Holoscan sensor bridge dependencies:

```none
  sudo apt install -y git-lfs cmake libfmt-dev libssl-dev libcurlpp-dev libyaml-cpp-dev libibverbs-dev python3-dev
```

- Obtain the Holoscan sensor bridge repository:

```none
  git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
```

- Build Holoscan sensor bridge, inside the `holoscan-sensor-bridge` directory:

```none
  mkdir build && cd build
  cmake -DCCCL_DIR:PATH="/usr/local/cuda/targets/sbsa-linux/lib/cmake/cccl" -DHOLOLINK_BUILD_SIPL=1 ..
  make -j
```

- Enable the network interface and ensure that the camera enumerates. Note that this
  documentation assumes a camera IP address of 192.168.0.2.

```none
  EN0=mgbe0_0
  sudo nmcli con add con-name hololink-$EN0 ifname $EN0 type ethernet ip4 192.168.0.101/24
  sudo nmcli connection modify hololink-$EN0 +ipv4.routes 192.168.0.2/32
  sudo nmcli connection modify hololink-$EN0 ethtool.ring-rx 4096
  sudo nmcli connection up hololink-$EN0
```

- Retrieve your camera's MAC ID with the `hololink-enumerate` command:

```none
  ./tools/enumerate/hololink-enumerate
```

You should see a response from the camera similar to this one:

```none
  mac_id=8C:1F:64:6D:70:03 hsb_ip_version=0x2506 fpga_crc=0xffff ip_address=192.168.0.2 fpga_uuid=f1627640-b4dc-48af-a360-c55b09b3d230 serial_number=ffffffffffffff interface=mgbe0_0 board=Leopard Eagle
```

Make sure to set all of the `ip_address` and `mac_address` fields in the configuration
files (there are multiple instances in each configuration file)

```
  ../examples/sipl_config/vb1940_single.json
  ../examples/sipl_config/vb1940_dual.json
```

Give permission for root to access the X display, Note that the DISPLAY environment
variable needs to be set first if being run over SSH.

```none
  xhost +
```

Run the `sipl_player` application using either the single or dual camera configurations
contained in the `examples/sipl_config` directory. To run them using HW ISP capture
mode, use the following:

```none
  sudo ./examples/sipl_player --json-config ../examples/sipl_config/vb1940_single.json
  sudo ./examples/sipl_player --json-config ../examples/sipl_config/vb1940_dual.json
```

To run the examples for RAW capture mode, add the --raw argument. Note that the image
quality will not be very good due to the lack of proper ISP processing (the image may be
extremely dark).

```none
  sudo ./examples/sipl_player --json-config ../examples/sipl_config/vb1940_single.json --raw
  sudo ./examples/sipl_player --json-config ../examples/sipl_config/vb1940_dual.json --raw
```

JP7.0.0 release currently supports only the
[Leopard imaging VB1940 Eagle Camera](sensor_bridge_hardware_setup.md).
