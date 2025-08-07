# Holoscan Sensor Bridge VM-Based Device Configuration

Holoscan Sensor Bridge (HSB) is an extremely flexible device that offers high-speed
connectivity for virtually any application. Any hardware connected to HSB will require
drivers to configure the device, but writing these drivers is generally not too
difficult using the SPI or I2C userspace interfaces that HSB provides.

In many cases existing drivers already exist outside of HSB, so adding support to HSB is
often a simple matter of copying and then modifying the drivers to use the HSB SPI or
I2C userspace APIs instead of the previous APIs it was using.

In some cases, however, the existing drivers are not just complex but they are also
written as kernel drivers that rely on large chunks of the kernel itself in order to
function properly. In these situations, converting the drivers to be userspace HSB
drivers may not be an acceptable solution.

To support cases like this, HSB provides a solution that makes use of a sidecar virtual
machine that is capable of running kernel drivers natively as kernel code within the VM,
and then SPI communication is forwarded to the HSB and connected device via the
userspace Holoscan application.

> **Note:** Currently only SPI kernel drivers are supported. I2C driver support may be
> added in a later release.

This support has been designed around one particular example included with HSB, which is
a signal generation application that passes data between HSB and an Analog Devices
AD9986 mixed signal front end (MxFE) via the JESD interface. Because of this, the code
and instructions provided are tailored to this use case and will require modifications
to work with other use cases.

## Building a Virtual Machine Image

Since the signal generator example uses an Analog Devices MxFE, the virtual machine runs
a [Yocto] image that is built using the [Analog Devices Kernel] along with extra
Holoscan-provided drivers and device tree configuration that connects the MxFE
(specifically, an AD9986 + HMC7044) to HSB. The `meta-hsb` directory defines a Yocto
meta layer that provides all the code and recipes that are needed to build the
HSB-enabled sidecar VM image using following instructions.

In the `hololink/configurator` directory:

1. Clone the required Yocto layers:
   ```sh
   git clone -b scarthgap-5.0.9 git://git.yoctoproject.org/poky
   git clone -b scarthgap https://github.com/openembedded/meta-openembedded.git
   ```
1. Setup the build environment (this must be done in every shell that will be used to
   perform `bitbake` commands):
   ```sh
   source poky/oe-init-build-env
   ```
   Note that when this is called it will automatically create an empty project in the
   `build` directory and will change your path to that directory.
1. Add the absolute paths to the `meta-openembedded/meta-oe` and `meta-hsb` directories
   to the `conf/bblayers.conf` in the newly created project directory. The paths should
   be added to the end, after `meta-yocto-bsp`, and need to be in that order (`meta-oe`
   first followed by `meta-hsb`. For example (change `workspace` with the path that
   contains your `hololink` directory):
   ```
   BBLAYERS ?= " \
     /workspace/hololink/configurator/poky/meta \
     /workspace/hololink/configurator/poky/meta-poky \
     /workspace/hololink/configurator/poky/meta-yocto-bsp \
     /workspace/hololink/configurator/meta-openembedded/meta-oe \
     /workspace/hololink/configurator/meta-hsb \
     "
   ```
1. Set the `MACHINE` variable in `conf/local.conf` to `hsb-ad9986`:
   ```sh
   echo 'MACHINE="hsb-ad9986"' >> conf/local.conf
   ```
1. Build the `core-image-minimal` image:
   ```sh
   bitbake core-image-minimal
   ```

## Using the Image

Running the VM is done automatically by the `demo.sh` script in this directory. Doing so
requires the image and device tree files that were output from the build step, above.
The script will launch the Hololink docker container along with a configuration VM in
the background:

```sh
./demo.sh
```

The process ID of the background VM will be written to `spi-vm.pid` and its console
output will be written to `spi-vm.log`. When the container terminates, the script will
also automatically terminate the VM.

Now that the configuration VM is available, see the Holoscan Sensor Bridge SDK
documentation for details on how to use configuration from the VM.

> **Note**: This script will use the image that was built in
> `hololink/configurator/build` by default, expecting the image files to be in the
> default Yocto output directory (`tmp/deploy/images/hsb-ad9986/`). If the path is
> different, for example if the image was built on another machine and then copied to
> the current system, the `--spi-vm` argument can be used to specify the image root.

> **Note:** The configuration VM currently only supports configuring devices once per
> boot. If the application is to be run a second time, the container and VM must be
> restarted.

[analog devices kernel]: https://github.com/analogdevicesinc/linux
[yocto]: https://www.yoctoproject.org/
