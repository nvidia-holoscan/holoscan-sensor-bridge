# UDDF (SIPL) Drivers

When accessing Holoscan Sensor Bridge via
[SIPL](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/CameraDevelopment/CoECameraDevelopment/SIPL-for-L4T/Introduction-to-SIPL.html),
such as when using the `SIPLCapture` operator and/or running the `sipl_player` example
application on a supported AGX Thor platform, the sensor drivers that are used are
provided by external driver libraries written against the
[Unified Device Driver Framework (UDDF)](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/CameraDevelopment/CoECameraDevelopment/SIPL-for-L4T/HSL-UDDF-Overview.html);
these drivers are not provided by the HSB repo.

NVIDIA's reference UDDF drivers are installed with JetPack as prebuilt binaries, and the
source code for these drivers are also included with the SIPL Camera SDK package. For
example, for the `vb1940` reference driver included with JetPack 7.1:

- The prebuilt driver library is installed to
  `/usr/lib/nvsipl_uddf/libnvuddf_eagle_library.so`
- The source code for this library is available in
  `/usr/src/jetson_sipl_api/sipl/uddf/samples/drivers/eagleAIO`

Note that if the SIPL Camera SDK (and the above source code) is not installed, download
and extract the `Camera SIPL` package from the
[JetPack Downloads](https://developer.nvidia.com/embedded/jetpack/downloads) page.

The `sipl_player` example application includes JSON config files (in
`examples/sipl_config/`) that configure the application to use the installed UDDF driver
libraries, and these will work out of the box with a new JetPack installation. See
[Leopard imaging VB1940 Eagle player example](examples.md#leopard-imaging-vb1940-eagle-player-example)
for an example of how to run this SIPL application using these configs and corresponding
drivers.

## Updating UDDF Drivers

If the reference UDDF drivers need to be updated -- either to change the sensor
programming or to update the Hololink code that is statically linked into the driver --
the drivers can be recompiled and installed from the driver source code:

```bash
$ mkdir eagle_uddf && cd eagle_uddf
$ cmake /usr/src/jetson_sipl_api/sipl/uddf/samples/drivers/eagleAIO/
$ make -j
$ sudo make install
```

As noted above, these UDDF drivers interact with HSB using a statically linked copy of
the Hololink core source code (see
`/usr/src/jetson_sipl_api/sipl/uddf/samples/drivers/eagleAIO/hololink/core`). If the
Hololink code of a UDDF driver needs to be updated, this can be done by overwriting the
UDDF driver's version of Hololink core with the latest in the HSB repo (e.g. from
`src/hololink/core`) and then rebuilding/installing using the commands above.

Note that Hololink compatibility is only guaranteed with the Hololink tag corresponding
to the JetPack release that is being used (i.e. `JetPack_7.1`); that is the version of
the code as it was originally copied to the UDDF driver code. Newer versions of Hololink
may not be compatible as-is and may require other updates to be made to the UDDF driver
code.

## Adding new UDDF Drivers

As the HSB repo does not contain any UDDF drivers, adding new UDDF drivers is also not
within the scope of this documentation. The Eagle VB1940 reference driver discussed and
used above is meant to provide a sample UDDF driver implementation, and should be used
as the basis for any new driver implementation.

It will be noted, however, that the `HsbTransportDriver` class included with the UDDF
driver acts as the Hololink client for the UDDF driver, and is responsible for all
interactions with HSB. Any updates to `hololink::core` within the driver may result in
corresponding changes to `HsbTransportDriver`.

For more information on SIPL and how to write and use UDDF drivers, see the
[Camera Development using CoE](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/CameraDevelopment/CoECameraDevelopment.html)
section of the
[Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/index.html).
