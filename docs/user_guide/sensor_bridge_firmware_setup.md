## Holoscan Sensor Bridge FPGA firmware update

The Holoscan Sensor Bridge (HSB) FPGA firmware can be remotely flashed over Ethernet.

1. Follow the [setup instructions](setup.md) to build and run the demo container. All
   the following commands are to be run from within the demo container.

1. Check connectivity with the sensor bridge board with the ping command:

   ```
   $ ping 192.168.0.2
   PING 192.168.0.2 (192.168.0.2) 56(84) bytes of data.
   64 bytes from 192.168.0.2: icmp_seq=1 ttl=64 time=0.143 ms
   64 bytes from 192.168.0.2: icmp_seq=2 ttl=64 time=0.098 ms
   64 bytes from 192.168.0.2: icmp_seq=3 ttl=64 time=0.094 ms
   ```

   If your system uses the 192.168.0.0/24 network for another purpose, see instructions
   for [configuring the IP addresses used by the sensor bridge device.](notes.md) After
   reconfiguring addresses appropriately, be sure you can ping your device at the
   address you expect.

1. These instructions assume that your FPGA was loaded with 0x2412 or newer firmware;
   this is the version included with HSB 2.0 (the previous public release). If your
   configuration is older, follow the
   [instructions to update to 2.0 first](https://docs.nvidia.com/holoscan/sensor-bridge/2.0.0/sensor_bridge_firmware_setup.html).
   To see what version of FPGA IP your HSB is configured with:

   ```sh
   hololink-enumerate
   ```

   This command will display data received from HSB's bootp enumeration message, which
   includes the HSB IP version.

## HSB Flasher

The `hsb_flasher` tool provides a generic entry point to providing board specific flash
mechanisms, for firmware, to HSB devices over ethernet. It aggregates known tools such
as (`program_lattice_cpnx100`, and `program_leopard_cpnx100`) into a single command that
automatically identifies the connected device and selects the appropriate flash
strategy.

### Supported Devices

1. Lattice CPNX100-ETH-SENSOR-BRIDGE

1. Leopard imaging VB1940 Eagle Camera

### Quick Start

1. Verify network connectivity with the sensor bridge:

   ```sh
   ping 192.168.0.2
   ```

1. Flash the device to the desired version (hexadecimal, without `0x` prefix):

   ```sh
   hsb_flasher --hololink 192.168.0.2 --target-version 2603
   ```

   If the device is already running the target version, the tool will exit with a
   message indicating no update is needed.

1. Once flashing is complete, **power cycle** the device.

1. Verify the update by pinging the device and running `hololink-enumerate` again to
   confirm the new firmware version.

### Detailed Description

In normal usage as shown in [Quick Start](#quick-start), `hsb_flasher` performs the
following steps:

1. **Device discovery** — The tool listens for BOOTP enumeration messages. Of the
   enumeration data, current firmware version (`hsb_ip_version`), and FPGA UUID
   (`fpga_uuid`) are the most relevant. If the device is already at the target version,
   the tool exits early.

1. **Manifest selection** — The FPGA UUID from the enumeration response is used to find
   the matching YAML manifest. The tool scans all `.yaml` files in the
   `firmware_information/` directory and selects the one whose `fpga_uuid` field matches
   the discovered device. Each YAML manifest describes a specific board type and lists
   available firmware versions with their download URLs, expected file sizes, and MD5
   checksums.

1. **Firmware fetch** — The target version entry is looked up in the matched YAML
   manifest to determine the firmware files needed (CLNX and/or CPNX depending on the
   board). If the firmware files are hosted remotely, they are downloaded over HTTPS and
   verified against the expected size and MD5 checksum. Local paths can also be
   specified in the YAML.

1. **Flash strategy selection** — The tool scans Python modules in the
   `firmware_flash_strategies/` directory and queries each one by calling its
   `supports(fpga_uuid, version)` function with the device's FPGA UUID and current
   firmware version. The first module that returns `true` is selected. Each strategy
   module maps the device version to a specific C++ flash routine (exposed via pybind11)
   that knows how to program that board.

1. **Flash execution** — The selected strategy's `do_flash()` function is invoked with
   the firmware file paths. The underlying C++ flash routine connects to the device via
   the Hololink core library and conducts the flash.

After flashing completes, the device must be **power cycled** for the new firmware to
take effect.

### Direct Mode

Direct flash mode allows flashing firmware from local files, bypassing the built-in YAML
manifest lookup and firmware download. Device discovery and flash strategy selection
still operate normally: the tool connects to the device, reads its FPGA UUID and
firmware version, and uses these to find a matching Python module in
`firmware_flash_strategies/`. The only difference from standard mode is that the
firmware files come from your local paths instead of being resolved and fetched through
a YAML manifest.

To flash with a CPNX image only (e.g. HSB Leopard):

```sh
hsb_flasher -H 192.168.0.2 --cpnx /path/to/cpnx.bit
```

To flash with both CLNX and CPNX images (e.g. HSB Lite):

```sh
hsb_flasher -H 192.168.0.2 --clnx /path/to/clnx.bit --cpnx /path/to/cpnx.bit
```

By default, the flash strategy is selected using the firmware version reported by the
device during discovery. To override this (e.g. to force a specific flash routine when
the device is running an unrecognized version), use the optional `--flash-version`
argument with a hexadecimal version string (without `0x` prefix):

```sh
hsb_flasher -H 192.168.0.2 --cpnx /path/to/cpnx.bit --flash-version 2507
```

### Custom Images

Support for a new FPGA board can be added by providing two files: a YAML firmware
manifest and a Python flash strategy module. At runtime, `hsb_flasher` discovers both
automatically by scanning the `firmware_information/` and `firmware_flash_strategies/`
directories alongside the executable.

#### Firmware Manifest (YAML)

Create a new `.yaml` file in the `firmware_information/` directory. The file must
contain the FPGA UUID of the target board and a list of firmware versions with their
download locations, MD5 checksums, and file sizes.

```yaml
fpga_uuid:
  - <your-fpga-uuid>

firmware_versions:
  - version: 0x2603
    cpnx:
      - location: https://example.com/firmware/cpnx_v2603.bit
        md5: <md5-checksum>
        size: <file-size-in-bytes>
    clnx:
      - location: https://example.com/firmware/clnx_v2603.bit
        md5: <md5-checksum>
        size: <file-size-in-bytes>
```

The `fpga_uuid` field is a YAML sequence; use a one-item list for a single UUID. Include
a `clnx` entry only if the board requires it; boards that only need CPNX firmware (like
HSB Leopard) can omit the `clnx` field entirely.

The `location` field can be an HTTPS URL (firmware will be downloaded and cached) or a
local file path.

#### Flash Strategy Module (Python)

Create a new `.py` file in the `firmware_flash_strategies/` directory. The module must
expose two top-level functions that `hsb_flasher` calls:

- **`supports(fpga_uuid, version)`** — Return `True` if this module can flash the given
  FPGA UUID at the given firmware version.

- **`do_flash(fpga_uuid, version, mac_address, ip_address, clnx_path, cpnx_path)`** —
  Perform the flash operation. Return `True` on success, `False` on failure.

A minimal flash strategy module follows this pattern:

```python
from abc import ABC, abstractmethod
from typing import Optional


class MyBoardFlasherBase(ABC):
    FPGA_UUID = "<your-fpga-uuid>"
    VERSION: int = 0

    def __init__(self, ip_address: str, mac_address: str):
        self.ip_address = ip_address
        self.mac_address = mac_address

    @classmethod
    def supports(cls, fpga_uuid: str, version: int) -> bool:
        return fpga_uuid == cls.FPGA_UUID and version == cls.VERSION

    @abstractmethod
    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        pass


class MyBoardFlasher2603(MyBoardFlasherBase):
    VERSION = 0x2603

    def flash(self, clnx_path: str, cpnx_path: str) -> bool:
        # Implement board-specific flash logic here.
        # Typically calls a pybind11 C++ module to perform SPI operations.
        ...


FLASH_STRATEGIES = [
    MyBoardFlasher2603,
]


def supports(fpga_uuid: str, version: int) -> bool:
    return any(cls.supports(fpga_uuid, version) for cls in FLASH_STRATEGIES)


def get_flasher(fpga_uuid: str, version: int,
                ip_address: str, mac_address: str) -> Optional[MyBoardFlasherBase]:
    for cls in FLASH_STRATEGIES:
        if cls.supports(fpga_uuid, version):
            return cls(ip_address=ip_address, mac_address=mac_address)
    return None


def do_flash(fpga_uuid: str, version: int, mac_address: str, ip_address: str,
             clnx_path: str, cpnx_path: str) -> bool:
    flasher = get_flasher(fpga_uuid, version, ip_address, mac_address)
    if flasher is None:
        return False
    return flasher.flash(clnx_path, cpnx_path)
```

The `flash()` method in each version-specific class contains the board-specific
programming logic. The existing implementations use pybind11 C++ modules that perform
SPI erase/program/verify operations through the Hololink core library. See
`flash_tools/hsb_lite/` and `flash_tools/hsb_leopard/` for reference implementations.

## Microchip MPF200-ETH-SENSOR-BRIDGE

1. For Microchip MPF200-ETH-SENSOR-BRIDGE devices

   ```sh
   polarfire_esb --flash scripts/mchp_manifest.yaml
   ```

   Use "--force" command switch when FPGA is running older version of bit file like 2407
   or 2412.

   ```sh
   polarfire_esb --force --flash scripts/mchp_manifest.yaml
   ```

For all Holoscan sensor bridges, once flashing is complete, **power cycle** the device.

For the Lattice CPNX100-ETH-SENSOR-BRIDGE, watch that the sensor bridge powers up with 2
green LEDs on

Ping the sensor bridge device at its IP address (e.g. 192.168.0.2) and verify a valid
ping response

## Legacy Scripts

The following scripts are the underlying tools used by `hsb_flasher` internally for
Lattice and Leopard imaging devices. They are available in the repository for reference
or advanced use:

- `program_lattice_cpnx100` — located in `tools/program_lattice_cpnx100/`
- `program_leopard_cpnx100` — located in `tools/program_leopard_cpnx100/`
