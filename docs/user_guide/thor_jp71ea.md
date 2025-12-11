# Thor JetPack 7.1 EA

The following provides additional documentation for using Holoscan Sensor Bridge with
the JetPack 7.1 Early Access release on a Thor AGX Devkit.

## Host Setup

After installing JetPack 7.1 EA on the Thor devkit (following the instructions provided
in the release notes of the JetPack 7.1 EA package), please follow these steps to setup
Holoscan Sensor Bridge.

- Install and configure Docker to use the NVIDIA container runtime:

  ```none
  sudo apt install -y docker-buildx
  sudo nvidia-ctk runtime configure --runtime=docker
  ```

- Grant your user permission to the docker subsystem:

  ```none
  sudo usermod -aG docker $USER
  ```

- Add the CUDA runtime path environment variables for your user:

  ```none
  cat << EOF >> ~/.bashrc
  export PATH=/usr/local/cuda-13.0/bin:\$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:\$LD_LIBRARY_PATH
  EOF
  ```

- Reboot the computer to apply the settings above.<br/><br/>

- Copy the `Jetson_Multimedia_API_R38.3.0_aarch64.tbz2` and
  `Jetson_SIPL_API_R38.3.0_aarch64.tbz2` files to the Thor Devkit from the
  `generic_release_aarch64` folder of the extracted JetPack 7.1 EA package, then install
  them:

  ```none
  sudo tar xjf Jetson_Multimedia_API_R38.3.0_aarch64.tbz2 -C /
  sudo tar xjf Jetson_SIPL_API_R38.3.0_aarch64.tbz2 -C /
  ```

- Install Holoscan SDK and other Holoscan Sensor Bridge dependencies:

  ```none
  sudo apt install -y holoscan git-lfs cmake libfmt-dev libssl-dev libcurlpp-dev libyaml-cpp-dev libibverbs-dev python3-dev
  ```

- Obtain the Holoscan Sensor Bridge repository:

  ```none
  git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
  ```

- Build Holoscan Sensor Bridge, inside the `holoscan-sensor-bridge` directory:

  ```none
  mkdir build && cd build
  cmake -DHOLOLINK_BUILD_SIPL=1 -DHOLOLINK_BUILD_FUSA=1 ..
  make -j
  ```

## Running the CoE-Accelerated Examples

Thor's hardware-accelerated CoE capabilities can be leveraged by Holoscan Sensor Bridge
using one of two different paths outlined below.

### SIPL

[SIPL](https://docs.nvidia.com/jetson/archives/r38.2.1/DeveloperGuide/SD/CameraDevelopment/CoECameraDevelopment/SIPL-for-L4T/Introduction-to-SIPL.html)
is a modular, extensible framework for image sensor control and image processing that
exposes the full hardware capabilities of Thor including CoE and ISP hardware
acceleration. SIPL-enabled sensor drivers are written using the
[Unified Device Driver Framework (UDDF)](uddf_drivers.md), and reference VB1940 UDDF
drivers are included with JetPack 7.1 EA.

Use the following to run the SIPL-based CoE example applications for the VB1940 sensor:

- Retrieve your camera's MAC ID:

  ```none
  ./tools/enumerate/hololink-enumerate
  ```

  Example output:

  ```none
  mac_id=8C:1F:64:6D:70:03 hsb_ip_version=0x2510 fpga_crc=0xffff ip_address=192.168.0.2 fpga_uuid=f1627640-b4dc-48af-a360-c55b09b3d230 serial_number=ffffffffffffff interface=mgbe0_0 board=Leopard Eagle
  ```

- Update the `ip_address` and `mac_address` fields in these configuration files
  (multiple instances in each file):

  ```none
  ../examples/sipl_config/vb1940_single.json
  ../examples/sipl_config/vb1940_dual.json
  ```

- Run the `sipl_player` application with hardware ISP enabled:

  ```none
  ./examples/sipl_player --json-config ../examples/sipl_config/vb1940_single.json
  ./examples/sipl_player --json-config ../examples/sipl_config/vb1940_dual.json
  ```

- For RAW capture mode (ISP disabled), use the following. Note that image quality will
  be poor due to the lack of ISP processing.

  ```none
  ./examples/sipl_player --json-config ../examples/sipl_config/vb1940_single.json --raw
  ./examples/sipl_player --json-config ../examples/sipl_config/vb1940_dual.json --raw
  ```

### FuSa

FuSa is a new API included with JetPack 7.1 which exposes access to Thor's CoE data
capture path without providing the additional camera control and ISP access that is
offered by SIPL. This allows applications direct control of the Holoscan Sensor Bridge
and attached sensors in a CoE-accelerated environment, bypassing the need for SIPL and
its UDDF driver implementations. This enables applications to follow a more traditional
Holoscan Sensor Bridge implementation where the sensor control is managed directly by
the application instead of by external drivers. Because of this, FuSa example
applications exist for both the IMX274 and VB1940 sensors using the existing reference
drivers provided by Holoscan Sensor Bridge.

A number of FuSa-based example applications are included for the IMX274 and VB1940 using
the `fusa-coe` prefix. The C++ sample applications can be run natively (not using a
container) and are built by the host setup instructions above, while the Python variants
must be run using the [Holoscan Sensor Bridge container](build.md).

For example, to run the C++ VB1940 player example, run the following command with the IP
address replaced with the IP address of the device:

```none
./examples/fusa_coe_vb1940_player --hololink 192.168.0.2
```
