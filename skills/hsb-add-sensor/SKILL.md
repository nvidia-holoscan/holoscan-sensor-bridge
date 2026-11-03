---
name: hsb-add-sensor
description: Add a new Bayer RAW camera sensor to the hololink repo. Reads a datasheet PDF and register CSV, generates Python and C++ sensor modules following the IMX274 pattern, generates RoCE and Linux socket player examples, and verifies with a clean build. This skill writes and modifies source files in the repository, runs docker builds, and executes commands on a remote devkit over SSH. Each phase requires explicit user confirmation before proceeding.
license: "Apache-2.0"
version: "1.0.0"
tags:
  - holoscan-sensor-bridge
  - hsb
  - sensor
  - camera
  - driver
tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - Bash
disable-model-invocation: true
allowed-tools: Read,Write,Edit,MultiEdit,Grep,Glob,Bash
metadata:
  author: "Holoscan Team <holoscan-team@nvidia.com>"
  team: holoscan
  tags:
    - holoscan-sensor-bridge
    - hsb
    - sensor
    - camera
    - driver
  agents:
    - claude-code
---

# HSB Add Sensor

Use this skill when the user wants to integrate a new Bayer RAW camera sensor into the hololink repository, generating all required Python and C++ source files and a runnable player example.

## Prerequisites

Verify these programmatically before collecting inputs. Use the same `SSH_TARGET`, `REMOTE_ROOT`, and `REMOTE_SSH_OPTS` environment variables as the `hsb-setup` and `hsb-app` skills.

1. **SSH connectivity** — confirm the devkit is reachable:
   ```bash
   ssh -o ConnectTimeout=10 -o BatchMode=yes $SSH_TARGET "echo ok"
   ```
   If SSH fails, follow the same SSH key remediation flow used in `hsb-setup` (verify key-based authentication is configured on the devkit).

2. **HSB board reachable** — ping the FPGA board at `192.168.0.2`:
   ```bash
   ssh -o BatchMode=yes $SSH_TARGET "ping -c 4 -W 2 192.168.0.2"
   ```
   If ping fails, stop and tell the user to check board power and network connection.

3. **HSB setup is complete** — verify the demo container image exists on the devkit:
   ```bash
   ssh -o BatchMode=yes $SSH_TARGET bash -s <<'REMOTE'
   cd "${REMOTE_ROOT:-/home/ubuntu}"
   for dir in holoscan-sensor-bridge*; do
       if [ -f "$dir/VERSION" ]; then
           VERSION=$(cat "$dir/VERSION")
           if docker image inspect "hololink-demo:$VERSION" >/dev/null 2>&1; then
               echo "SETUP_OK=yes REPO_DIR=$dir VERSION=$VERSION"
               break
           fi
       fi
   done
   REMOTE
   ```
   If no demo container is found, stop and instruct the user to run `/hsb-setup-skill` first.

4. **Camera sensor physically connected** — the user is responsible for connecting the new camera sensor to the Lattice HSB FPGA board. Ask the user to confirm this before proceeding.

Display prerequisite results before continuing:
```
Prerequisites:
- SSH to $SSH_TARGET:           OK
- HSB board ping (192.168.0.2): OK  (4/4 packets)
- Demo container:               hololink-demo:X.X.X found
- Camera connected:             confirmed by user
```

If any check fails, stop and provide specific remediation instructions.

## Sensor constraints

This skill only supports:
- Quad Bayer RAW format image streams (RGGB, BGGR, GRBG, or GBRG)
- Bit depth: 8, 10, or 12 bpp
- I2C transport (same wiring as IMX274)
- Single mode: the highest resolution + frame rate extracted from the datasheet

If the sensor does not meet these constraints, inform the user and stop.

## Naming conventions

For a sensor named `imx274`, derive all identifiers as follows:

| Variable | Example | Rule |
|---|---|---|
| `SENSOR_NAME` | `imx274` | lowercase, no spaces |
| `SENSOR_UPPER` | `IMX274` | uppercase of SENSOR_NAME |
| `SENSOR_CLASS` | `Imx274` | Title-case first letter of each word |
| `SENSOR_CLASS_UPPER` | `IMX274` | Same as SENSOR_UPPER (used in include guards) |
| `MODE_ENUM_PREFIX` | `IMX274_MODE` | `SENSOR_UPPER + "_MODE"` |
| `DRIVER_NAME` | `"IMX274-NATIVE"` | `SENSOR_UPPER + "-NATIVE"` |
| `PYTHON_CLASS` | `Imx274Cam` | `SENSOR_CLASS + "Cam"` |
| `CPP_SENSOR_CLASS` | `NativeImx274Sensor` | `"Native" + SENSOR_CLASS + "Sensor"` |
| `CPP_FRAME_FORMAT_CLASS` | `Imx274FrameFormat` | `SENSOR_CLASS + "FrameFormat"` |
| `CPP_LIB_NAME` | `native_imx274_camera_sensor` | `"native_" + SENSOR_NAME + "_camera_sensor"` |

## Phase overview

Show the full plan at the start:

- Phase 0: Collect inputs — sensor name, datasheet PDF, register CSV, repo path, build target
- Phase 1: Parse datasheet PDF — extract mode parameters and register tables
- Phase 2: Reconcile registers — cross-check CSV against datasheet, resolve mismatches
- Phase 3: Generate Python module — `python/hololink/sensors/<sensor>/`
- Phase 4: Generate C++ module — `src/hololink/sensors/camera/<sensor>/`
- Phase 5: Generate player examples — `examples/<sensor>_player.py` (RoCE) and `examples/linux_<sensor>_player.py` (Linux socket / AGX Orin)
- Phase 6: Build verification — `sh ./docker/build [--igpu/--dgpu]`, offer to run via `/hsb-app-skill`

After each phase, show a summary and ask `Proceed to Phase <N+1>? Type 'yes' to continue or 'no' to stop:` before continuing.

---

## Phase 0 — Collect inputs

Ask the user for:

1. **Sensor name** — lowercase, no spaces (e.g., `imx274`). Derive all naming variables from this.
2. **Datasheet PDF path** — absolute path on the local machine. Confirm the file is readable with the Read tool.
3. **Register CSV path** — required. Absolute path to a CSV with columns `address,value` (hex or decimal). The CSV is the primary source for register values; the datasheet is used for cross-validation and to fill in sequences not present in the CSV.
4. **Repo path** — absolute path to the hololink repo. Default: `~/hololink`. Confirm it exists by checking for `python/hololink/sensors/__init__.py`.
5. **Build target** — `--igpu` or `--dgpu`.
6. **MIPI lane count** — how many MIPI data lanes does the sensor use? (1, 2, or 4). HSB defaults to 4-lane; if the sensor uses fewer, `program_mipi_phy()` must be called in the player before streaming. Ask explicitly — do not assume 4 without confirmation.
6b. **Sensor interface type** — is this a **direct-MIPI** sensor (connects directly to the HSB FPGA MIPI PHY, like IMX274/IMX681) or a **GMSL** sensor (connects via a GMSL SerDes pair like MAX96716A deser + MAX96719B ser, like OX08D10)? This determines whether li_i2c_expander is used and how I2C is routed. If uncertain, ask.
7. **I2C expander jack** — which jack on the LI I2C expander is the sensor connected to? Answer is `J1` (OUTPUT_1, `expander_configuration=0`) or `J2` (OUTPUT_2, `expander_configuration=1`). Only ask for **direct-MIPI sensors** (IMX274/IMX681 wiring pattern). **Do not ask for GMSL sensors** — I2C reaches the sensor through the GMSL tunnel, not through the LI I2C expander. Skip this question entirely for GMSL cameras. If uncertain whether the sensor is GMSL or direct-MIPI, ask the user.

Display confirmed settings:

```
Inputs confirmed:
- Sensor name:    imx274
- Datasheet:      /path/to/imx274_datasheet.pdf
- Register CSV:   /path/to/imx274_regs.csv
- Repo path:      /path/to/hololink
- Build target:   --igpu
- MIPI lanes:     2
- Sensor type:    direct-MIPI / GMSL
- I2C expander:   J1 (OUTPUT_1, expander_configuration=0)  [direct-MIPI only]
  -- OR --
- GMSL link:      A / B  [GMSL only]
```

---

## Phase 1 — Parse datasheet PDF

Read the datasheet PDF using the Read tool. Extract the following information:

### 1.1 Mode parameters (highest resolution + frame rate)

Extract:
- **Width** and **Height** (pixels) — use the highest resolution available
- **Frame rate** (fps) — at the highest resolution
- **Bit depth** — 8, 10, or 12 bpp. Map to `PixelFormat`:
  - 8 bpp → `hololink.sensors.csi.PixelFormat.RAW_8`
  - 10 bpp → `hololink.sensors.csi.PixelFormat.RAW_10`
  - 12 bpp → `hololink.sensors.csi.PixelFormat.RAW_12`
- **Bayer format** — RGGB, BGGR, GRBG, or GBRG. Map to:
  - RGGB → `hololink_module.sensors.csi.BayerFormat.RGGB` (Python) / `hololink::csi::BayerFormat::RGGB` (C++)
  - BGGR → `BGGR`
  - GRBG → `GRBG`
  - GBRG → `GBRG`
- **I2C address** — 7-bit address in binary or hex (e.g., `0b00110100`, `0x34`)
- **Metadata lines** — number of embedded/metadata lines prepended to image data. Search the datasheet for the embedded data or EBD section and read the exact count from there. Never guess or assume a value — if it cannot be confirmed from the datasheet, default to 0 and flag with a `# TODO: verify embedded line count from datasheet`.
- **MIPI lane count** — confirm from the datasheet (look for CSI_LANE_MODE or equivalent register, and the supported lane configurations table). Verify it matches what the user provided in Phase 0. Flag any discrepancy.

### 1.2 Register tables

Extract:
- **Mode init table** — the full register write sequence to configure the sensor for the selected mode. Each entry is `(register_address, value)` in hex. Include any wait/delay entries (typically written as a delay-sentinel followed by milliseconds).
- **Start sequence** — register writes to begin streaming (e.g., enable streaming bit).
- **Stop sequence** — register writes to stop streaming.
- **Test pattern registers** — registers to enable/disable test pattern output. Mark as `# TODO: verify from datasheet` if not found.

If start/stop sequences or test pattern registers cannot be found in the datasheet, generate TODO placeholders and tell the user:
```
⚠ Could not find start/stop sequences in the datasheet.
  Placeholders have been generated. Review and fill in:
  python/hololink/sensors/<sensor>/<sensor>_mode.py
```

### 1.3 Wait/delay sentinel

The Python code uses a string sentinel (e.g., `"imx274-table-wait-ms"`) to indicate delays in register tables. The C++ code uses `0xFFFF` as the sentinel (a `uint16_t` value that is not a valid register address).

Derive the Python sentinel as: `"<SENSOR_NAME>-table-wait-ms"`

### 1.4 Display extraction summary

Fill in the actual values extracted from the datasheet. Register counts will vary by sensor.

```
Datasheet extraction results:
- Resolution:     <WIDTH> × <HEIGHT> @ <FPS> fps
- Bit depth:      <N> bpp  →  RAW_<N>
- Bayer format:   <RGGB|BGGR|GRBG|GBRG>
- I2C address:    <binary> (<hex>)
- Metadata lines: <N>
- Mode registers: <N> entries
- Start sequence: <N> entries  (or: not found — placeholder generated)
- Stop sequence:  <N> entries  (or: not found — placeholder generated)
- Test pattern:   found (<N> registers) / not found (placeholder generated)
```

Ask the user to confirm before proceeding. If any critical field (resolution, bit depth, I2C address) could not be extracted, stop and ask the user to provide it manually.

---

## Phase 2 — Reconcile registers

Read the register CSV and cross-check it against the datasheet-extracted tables.

### CSV format

Expected: one register per line, format `address,value` where both are hex (with or without `0x` prefix) or decimal integers.

```csv
address,value
0x3000,0x01
0x3001,0xFF
```

### Cross-check rules

1. For every register in the CSV, check if the same address appears in the datasheet mode table.
2. If the address matches but the value differs, flag it as a **mismatch**.
3. Registers in the CSV that are not in the datasheet mode table are **CSV-only** (may be additional tuning).
4. Registers in the datasheet that are not in the CSV are **datasheet-only** (will use datasheet value).

### Present mismatches

```
Register cross-check results:
- Matched:        45 registers
- Mismatches:     2 registers (CSV value vs datasheet value):
    0x300A: CSV=0x10  vs  Datasheet=0x0C
    0x301F: CSV=0xFF  vs  Datasheet=0x00
- CSV-only:       3 registers (will be appended to mode table)
- Datasheet-only: 7 registers (datasheet value used)

For each mismatch, which value should be used?
  [1] 0x300A — use CSV value (0x10) or datasheet value (0x0C)?
  [2] 0x301F — use CSV value (0xFF) or datasheet value (0x00)?
```

Wait for the user to resolve each mismatch before continuing.

### Final register table

After reconciliation, produce the final merged table:
- Start with the datasheet mode init table
- Apply resolved mismatch values
- Append CSV-only registers at the end

---

## Phase 3 — Generate Python module

Create the directory `<REPO_PATH>/python/hololink/sensors/<SENSOR_NAME>/` and generate these files:

### 3.1 `<sensor>_mode.py`

Follow the IMX274 pattern exactly. Key elements:

```python
"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
...
"""

from collections import namedtuple
from enum import Enum

import hololink

# values are on hex number system to be consistent with rest of the list
<SENSOR_UPPER>_TABLE_WAIT_MS = "<SENSOR_NAME>-table-wait-ms"
<SENSOR_UPPER>_WAIT_MS = 0x01
<SENSOR_UPPER>_WAIT_MS_START = 0x0F

# Register addresses for camera properties. They only accept 8bits of value.
# (add register address constants extracted from datasheet, e.g.:)
# REG_EXPOSURE = 0xXXXX
# REG_GAIN     = 0xXXXX

<sensor_name>_start = [
    # TODO: fill from datasheet start sequence
    (<SENSOR_UPPER>_TABLE_WAIT_MS, <SENSOR_UPPER>_WAIT_MS_START),
]

<sensor_name>_stop = [
    (<SENSOR_UPPER>_TABLE_WAIT_MS, <SENSOR_UPPER>_WAIT_MS),
    # TODO: fill from datasheet stop sequence
]

# Mode: <WIDTH>x<HEIGHT> <BIT_DEPTH> bits <FPS>fps
<sensor_name>_mode_<WIDTH>x<HEIGHT>_<FPS>fps = [
    # (register_address, value) pairs in hex
    # ... (generated from reconciled register table)
    (<SENSOR_UPPER>_TABLE_WAIT_MS, <SENSOR_UPPER>_WAIT_MS),
]


class <SENSOR_CLASS>_Mode(Enum):
    <MODE_ENUM_PREFIX>_<WIDTH>X<HEIGHT>_<FPS>FPS = 0
    Unknown = 1


frame_format = namedtuple(
    "FrameFormat", ["width", "height", "framerate", "pixel_format"]
)

<sensor_name>_frame_format = []
<sensor_name>_frame_format.insert(
    <SENSOR_CLASS>_Mode.<MODE_ENUM_PREFIX>_<WIDTH>X<HEIGHT>_<FPS>FPS.value,
    frame_format(<WIDTH>, <HEIGHT>, <FPS>, hololink.sensors.csi.PixelFormat.<RAW_N>),
)
```

Replace all `<PLACEHOLDER>` tokens with the extracted values. Use the reconciled register table for the mode table. All register addresses and values must be in hex (`0xXXXX` format).

### 3.2 `<sensor>_cam.py`

The implementation differs between direct-MIPI sensors (IMX274/IMX681 pattern) and GMSL sensors (OX08D10 pattern). Determine the sensor type before generating this file.

**Common rules (all sensor types):**
- Replace `CAM_I2C_ADDRESS` with the extracted I2C address
- Replace `DRIVER_NAME` with `"<SENSOR_UPPER>"`
- Import `hololink as hololink_module` at the top (needed for csi enum and renesas access)
- Use `logging.debug(...)` / `logging.info(...)` directly — do NOT create a named logger with `logging.getLogger(__name__)`. Hololink installs a custom formatter that requires a `log_timestamp_s` field; named loggers bypass it and raise `ValueError: Formatting field not found in record: 'log_timestamp_s'` at runtime.
- `setup_clock` must access renesas via `hololink_module` — there is NO `from hololink.sensors import renesas_bajoran_lite_ts1` import. The module lives at `hololink.renesas_bajoran_lite_ts1`, accessed as:
  ```python
  def setup_clock(self):
      self._hololink.setup_clock(
          hololink_module.renesas_bajoran_lite_ts1.device_configuration()
      )
  ```
- In `set_mode()`, store `self._pixel_format` as a `csi.PixelFormat` enum:
  ```python
  self._pixel_format = hololink_module.sensors.csi.PixelFormat.RAW_<N>
  ```
- Add a `pixel_format()` method that returns `self._pixel_format` (the enum, not an integer)
- `bayer_format()` must return a `csi.BayerFormat` enum, NOT a plain string:
  ```python
  def bayer_format(self):
      return hololink_module.sensors.csi.BayerFormat.<RGGB|BGGR|GRBG|GBRG>
  ```
- In `configure_converter`, pass `self._pixel_format` (enum) to `converter.transmitted_line_bytes()` — not a raw integer like `10`:
  ```python
  transmitted_line_bytes = converter.transmitted_line_bytes(self._pixel_format, self._width)
  ```
  Also accumulate metadata line offset using the extracted count:
  ```python
  embedded_line_bytes = converter.received_line_bytes(
      converter.transmitted_line_bytes(self._pixel_format, self._width)
  )
  start_byte += embedded_line_bytes * <METADATA_LINES>
  ```
- In `configure_camera`, dispatch on `<SENSOR_CLASS>_Mode` values
- Keep `set_exposure_reg`, `set_digital_gain_reg`, `set_analog_gain_reg` as implementations with real register addresses extracted from the datasheet. If addresses could not be found, generate TODO stubs and add `# TODO: fill register address` comment.

**Direct-MIPI sensors only (IMX274/IMX681 pattern — NOT for GMSL cameras):**
- Import `from . import <sensor>_mode, li_i2c_expander` (see 3.2a below)
- `__init__` must accept both `i2c_bus` and `expander_configuration`, and set up the I2C expander:
  ```python
  def __init__(
      self,
      hololink_channel,
      i2c_bus=hololink_module.CAM_I2C_BUS,
      expander_configuration=0,
  ):
      self._hololink = hololink_channel.hololink()
      self._i2c = self._hololink.get_i2c(i2c_bus)
      self._i2c_expander = li_i2c_expander.LII2CExpander(self._hololink, i2c_bus)
      if expander_configuration == 1:
          self._i2c_expander_configuration = li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_2
      else:
          self._i2c_expander_configuration = li_i2c_expander.I2C_Expander_Output_EN.OUTPUT_1
  ```
- `_write_reg` and `_read_reg` must acquire `i2c_lock` and call `self._i2c_expander.configure()` before every transaction — exactly as `dual_imx274.py` does:
  ```python
  def _write_reg(self, reg, val):
      write_bytes = bytearray(100)
      serializer = hololink_module.Serializer(write_bytes)
      serializer.append_uint16_be(reg)
      serializer.append_uint8(val)
      i2c_lock = self._hololink.i2c_lock()
      i2c_lock.lock()
      try:
          self._i2c_expander.configure(self._i2c_expander_configuration.value)
          self._i2c.i2c_transaction(CAM_I2C_ADDRESS, write_bytes[:serializer.length()], 0)
      finally:
          i2c_lock.unlock()
  ```
  Apply the same lock+expander pattern to `_read_reg`.

**GMSL sensors only (OX08D10 pattern — omit li_i2c_expander entirely):**
- Import `from . import <sensor>_mode` only — do NOT import `li_i2c_expander`
- `__init__` only needs `i2c_bus` (no `expander_configuration`):
  ```python
  def __init__(
      self,
      hololink_channel,
      i2c_bus=hololink_module.CAM_I2C_BUS,
  ):
      self._hololink = hololink_channel.hololink()
      self._i2c = self._hololink.get_i2c(i2c_bus)
  ```
- `_write_reg` and `_read_reg` use bare `i2c_transaction()` — no `i2c_lock`, no expander:
  ```python
  def _write_reg(self, reg, val):
      write_bytes = bytearray(100)
      serializer = hololink_module.Serializer(write_bytes)
      serializer.append_uint16_be(reg)
      serializer.append_uint8(val)
      self._i2c.i2c_transaction(CAM_I2C_ADDRESS, write_bytes[:serializer.length()], 0)
  ```
- I2C to the SerDes chips uses the same bare `i2c_transaction()` with the SerDes I2C address (see "GMSL register I/O" in the GMSL section below).

### 3.2a `li_i2c_expander.py` — direct-MIPI sensors only

**Skip this step for GMSL sensors.** GMSL cameras tunnel I2C through the GMSL link and do not use the LI I2C expander at all.

For **direct-MIPI sensors** only: each sensor directory carries its own copy of `li_i2c_expander.py`. Copy it from `python/hololink/sensors/imx274/li_i2c_expander.py` into `python/hololink/sensors/<sensor>/li_i2c_expander.py` — do not import it from a sibling sensor package.

### 3.3 `__init__.py`

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ... (license header)

from . import <sensor>_cam, <sensor>_mode

__all__ = [
    "<sensor>_cam",
    "<sensor>_mode",
]
```

### 3.4 `CMakeLists.txt`

```cmake
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ... (license header)

copy_hololink_python_files(
    TARGET_NAME copy_hololink_sensors_<sensor>
    DESTINATION sensors/<sensor>
    FILES
        __init__.py
        <sensor>_cam.py
        <sensor>_mode.py
        # li_i2c_expander.py   -- include this line for direct-MIPI sensors only (IMX274/IMX681 pattern)
        #                         omit for GMSL sensors (OX08D10 pattern)
)
```

### 3.5 Update parent files

**`python/hololink/sensors/CMakeLists.txt`**: Add `add_subdirectory(<sensor>)` after the existing `add_subdirectory` lines.

**`python/hololink/sensors/__init__.py`**: Add `"<sensor>"` to the `_MODULES` list. This enables lazy loading via `hololink.sensors.<sensor>`.

---

## Phase 4 — Generate C++ module

Create the directory `<REPO_PATH>/src/hololink/sensors/camera/<SENSOR_NAME>/` and generate these files:

### 4.1 `<sensor>_mode.hpp`

Follow `imx274_mode.hpp` exactly. Key customizations:

```cpp
// Include guard: SENSORS_CAMERA_<SENSOR_UPPER>_<SENSOR_UPPER>_MODE_HPP
// namespace hololink::sensors
// namespace <sensor_name>_mode { ... }

// Mode enum
enum Mode {
    <MODE_ENUM_PREFIX>_<WIDTH>X<HEIGHT>_<FPS>FPS = 0,
    <MODE_ENUM_PREFIX>_COUNT = 1,
};

// Register address constants (from datasheet, use TABLE_WAIT_MS = 0xFFFF)
static constexpr uint16_t <SENSOR_UPPER>_TABLE_WAIT_MS = 0xFFFF;
static constexpr uint8_t  <SENSOR_UPPER>_WAIT_MS = 0x01;
static constexpr uint8_t  <SENSOR_UPPER>_WAIT_MS_START = 0x0F;

// Array size constants — count entries in each table and set these exactly
constexpr size_t START_SEQUENCE_SIZE = <N>;
constexpr size_t STOP_SEQUENCE_SIZE  = <N>;
constexpr size_t MODE_<WIDTH>X<HEIGHT>_<FPS>FPS_SEQUENCE_SIZE = <N>;

// constexpr std::array<std::pair<uint16_t, uint8_t>, ...> for each table
// Use 0xFFFF as TABLE_WAIT_MS sentinel (not the string sentinel from Python)
// clang-format off / clang-format on wrapping
// static_assert size checks after each array
```

**IMPORTANT for array sizes**: Count every entry in the table including `TABLE_WAIT_MS` sentinel entries. The `static_assert` will catch miscounts at compile time — do not guess.

The `Imx274FrameFormat` class becomes `<SENSOR_CLASS>FrameFormat` and follows the same pattern (constructor + `format()` override).

### 4.2 `<sensor>_mode.cpp`

```cpp
// SPDX header
#include "<sensor>_mode.hpp"

namespace hololink::sensors {

<SENSOR_CLASS>FrameFormat::<SENSOR_CLASS>FrameFormat(CameraMode mode_id, const std::string& mode_name,
    int64_t width, int64_t height, double frame_rate,
    csi::PixelFormat pixel_format)
    : CameraFrameFormat(mode_id, mode_name, width, height, frame_rate, pixel_format)
{
}

const <SENSOR_CLASS>FrameFormat::Format& <SENSOR_CLASS>FrameFormat::format() const
{
    return format_;
}

} // namespace hololink::sensors
```

### 4.3 `native_<sensor>_sensor.hpp`

Follow `native_imx274_sensor.hpp` exactly. Key customizations:
- Include guard: `SENSORS_CAMERA_<SENSOR_UPPER>_NATIVE_<SENSOR_UPPER>_SENSOR_HPP`
- Include `"<sensor>_mode.hpp"`
- Class name: `Native<SENSOR_CLASS>Sensor` extending `CameraSensor`
- `DRIVER_NAME = "<SENSOR_UPPER>-NATIVE"`
- `I2C_ADDRESS = <extracted_i2c_address>`
- Member: `std::unordered_map<int, std::shared_ptr<<SENSOR_CLASS>FrameFormat>> mode_frame_formats_`
- Expose same public interface: `configure`, `set_mode`, `start`, `stop`, `configure_converter`, `setup_clock`, `get_version`, `test_pattern`, `set_exposure_reg`, `set_analog_gain_reg`, `set_digital_gain_reg`

### 4.4 `native_<sensor>_sensor.cpp`

Follow `native_imx274_sensor.cpp` exactly. Key customizations:
- Replace all `imx274_mode::` namespace with `<sensor_name>_mode::`
- Replace all `Imx274` class references with `Native<SENSOR_CLASS>Sensor`
- Replace `IMX274_MODE_COUNT` with `<MODE_ENUM_PREFIX>_COUNT`
- In `set_mode`, set `bayer_format_` to the extracted Bayer constant
- In `configure_converter`:
  - `start_byte += converter->received_line_bytes(<METADATA_LINES>)` for metadata
  - Handle optical black lines per bit depth if applicable (from datasheet; use 0 if unknown)
- `initialize_supported_modes`: populate `mode_frame_formats_` for the single mode using `<SENSOR_CLASS>FrameFormat`
- `setup_clock`: keep the `renesas_bajoran_lite_ts1` call (same clock hardware). Include the header with a relative path — it lives in `imx274/`, not alongside the new sensor: `#include "../imx274/renesas_bajoran_lite_ts1.hpp"`
- `test_pattern`: add register writes from datasheet; use TODO stubs if not found

### 4.5 `CMakeLists.txt`

```cmake
# SPDX header

add_library(native_<sensor>_camera_sensor STATIC
  native_<sensor>_sensor.cpp
  <sensor>_mode.cpp
  )

set_property(TARGET native_<sensor>_camera_sensor PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::sensors::native_<sensor>_camera_sensor ALIAS native_<sensor>_camera_sensor)

target_include_directories(native_<sensor>_camera_sensor
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(native_<sensor>_camera_sensor
  PUBLIC
    hololink_camera_sensor
  )

set_target_properties(native_<sensor>_camera_sensor PROPERTIES
  EXPORT_NAME sensors::native_<sensor>_camera_sensor)

install(TARGETS native_<sensor>_camera_sensor
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-sensors)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/native_<sensor>_sensor.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/<sensor>_mode.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/sensors/camera/<sensor>
        COMPONENT hololink-sensors)
```

### 4.6 Update parent file

**`src/hololink/sensors/camera/CMakeLists.txt`**: Add `add_subdirectory(<sensor>)` after the existing `add_subdirectory` lines.

---

## Phase 5 — Generate player examples

Generate **two** player files for every sensor:

### 5.1 `examples/<sensor>_player.py` — RoCE receiver (InfiniBand platforms)

Follow `examples/imx274_player.py` exactly. Key customizations:
- Replace all `imx274` module references with `<sensor>`
- Replace `Imx274Cam` with `<SENSOR_CLASS>Cam`
- Replace `Imx274_Mode` with `<SENSOR_CLASS>_Mode`
- Replace `imx274_mode` with `<sensor>_mode`
- Replace the `optical_black` value in `ImageProcessorOp` — look for `DT_PEDESTAL` or equivalent pedestal/offset register in the datasheet. For Sony sensors this is often documented in the output format section. Use `0` if not found (add `# TODO: confirm optical_black from datasheet`)
- The `--expander-configuration` argument maps to the I2C expander for sensor port selection — **include for direct-MIPI sensors only** (IMX274/IMX681 pattern). **Omit for GMSL sensors** — replace it with `--link` (values: `a` or `b`, corresponding to GMSL link A and link B).
- Adjust the `--camera-mode` help string to show the new sensor's modes
- Keep `--pattern`, `--ptp-sync`, `--skip-reset`, `--headless`, `--fullscreen`, `--frame-limit`, `--ibv-name`, `--ibv-port` arguments unchanged
- `bayer_format()` now returns a `csi.BayerFormat` enum, so use `bayer_grid_pos=bayer_format.value` in `BayerDemosaicOp`
- Pipeline: `RoceReceiverOp -> CsiToBayerOp -> ImageProcessorOp -> BayerDemosaicOp -> HolovizOp`

### 5.2 `examples/linux_<sensor>_player.py` — Linux socket receiver (AGX Orin and other platforms without RoCE)

Follow `examples/linux_imx715_player.py` as the template. Key differences from the RoCE player:

**Receiver operator**: Replace `RoceReceiverOp` with `LinuxReceiverOperator`. The Linux receiver takes `hololink_channel` and `device` instead of `ibv_name`/`ibv_port`:
```python
receiver_operator = hololink_module.operators.LinuxReceiverOperator(
    self,
    condition,
    name="receiver",
    frame_size=frame_size,
    frame_context=frame_context,
    hololink_channel=self._hololink_channel,
    device=self._camera,
)
```

**`ImageProcessorOp`**: The linux player passes format info explicitly to `ImageProcessorOp`:
```python
pixel_format = self._camera.pixel_format()
bayer_format = self._camera.bayer_format()
image_processor_operator = hololink_module.operators.ImageProcessorOp(
    self,
    name="image_processor",
    optical_black=<VALUE>,       # from datasheet DT_PEDESTAL or equivalent
    bayer_format=bayer_format.value,
    pixel_format=pixel_format.value,
)
```
This requires `pixel_format()` and `bayer_format()` to return csi enums (see Phase 3.2).

**`BayerDemosaicOp`**: Use `generate_alpha=True, alpha_value=65535` (the linux pattern):
```python
demosaic = holoscan.operators.BayerDemosaicOp(
    self,
    pool=bayer_pool,
    generate_alpha=True,
    alpha_value=65535,
    bayer_grid_pos=bayer_format.value,
    interpolation_mode=0,
)
```

**Window size args**: Add `--window-height`, `--window-width`, `--title`. Default window size to `<WIDTH> // 8` x `<HEIGHT> // 8`:
```python
parser.add_argument("--window-height", type=int, default=<HEIGHT> // 8)
parser.add_argument("--window-width",  type=int, default=<WIDTH>  // 8)
parser.add_argument("--title", help="Window title")
```

**No InfiniBand args**: Remove `--ibv-name` and `--ibv-port`.

**Keep**: `--camera-mode`, `--headless`, `--fullscreen`, `--frame-limit`, `--configuration`, `--hololink`, `--log-level`, `--pattern`, `--ptp-sync`, `--skip-reset`. For **direct-MIPI sensors**, also keep `--expander-configuration`. For **GMSL sensors**, replace `--expander-configuration` with `--link` (values: `a` or `b`, for GMSL link A and link B).

**MIPI lane configuration**: If the sensor uses fewer than 4 MIPI lanes (from Phase 0 input), the FPGA MIPI receiver must be reconfigured before streaming — HSB defaults to 4-lane. Add the following helper functions (copied verbatim from `examples/ecam0m30tof_player.py`) and call them in `main()` after `hololink.reset()` and before `camera.setup_clock()`:

```python
def _write_lmmi_register(camera_idx, data, addr, hololink):
    slow_spi = hololink.get_spi(
        hololink_module.CLNX_SPI_BUS,
        chip_select=0, cpol=0, cpha=1, width=1, prescaler=0xF,
    )
    request = [0x11, 0x0C]
    while 1:
        rdval = slow_spi.spi_transaction(bytearray(), bytearray(request), 1)
        if int.from_bytes(bytes(rdval), byteorder="little") & 0x4:
            break
    for reg_addr, val in [(0x0D, data), (0x0F, addr), (0x0C, (camera_idx << 4) + 0x1)]:
        slow_spi.spi_transaction(bytearray(), bytearray([1, reg_addr, val]), 0)


def _reconfigurable_mipi(camera_idx, lane_prog, clk_settle, data_settle, hololink):
    reg0A = ((clk_settle & 0x1) << 3) + (lane_prog << 1)
    _write_lmmi_register(camera_idx, reg0A,            0x0A, hololink)
    _write_lmmi_register(camera_idx, clk_settle >> 1,  0x0B, hololink)
    _write_lmmi_register(camera_idx, (clk_settle >> 2) & 0x80, 0x0C, hololink)
    _write_lmmi_register(camera_idx, (data_settle & 0x3) << 2, 0x0F, hololink)
    _write_lmmi_register(camera_idx, (data_settle >> 2) & 0xF, 0x10, hololink)


def program_mipi_phy(camera_idx, num_lanes, clock_freq_mhz, hololink):
    # clock_freq_mhz = MIPI D-PHY clock in MHz = bit_rate_mbps / 2
    lane_prog = num_lanes - 1
    if   clock_freq_mhz >= 1000: data_settle = 6
    elif clock_freq_mhz >= 350:  data_settle = 7
    elif clock_freq_mhz >= 200:  data_settle = 8
    elif clock_freq_mhz >= 150:  data_settle = 9
    else:                        data_settle = 11
    _reconfigurable_mipi(camera_idx, lane_prog, 9, data_settle, hololink)
```

Call it for **both** FPGA PHY indices (0 and 1) regardless of which jack the sensor is on — the ecam0m30tof reference confirms both must be configured:

```python
# In main(), after hololink.reset() and before camera.setup_clock():
program_mipi_phy(0, <NUM_LANES>, <CLOCK_FREQ_MHZ>, hololink)
program_mipi_phy(1, <NUM_LANES>, <CLOCK_FREQ_MHZ>, hololink)
```

Compute `<CLOCK_FREQ_MHZ>` from the sensor's IOP PLL settings: `IOPCK_MHz / 2` (D-PHY is DDR). For example, IMX681 with IOPCK = 2000 MHz → `clock_freq_mhz = 1000`. Derive from the IOP PLL registers in the register CSV; verify against the datasheet PLL formula. If cannot be determined, add a `# TODO: verify clock_freq_mhz from IOP PLL registers` comment.

Do **not** add `program_mipi_phy` if the sensor uses 4 lanes — HSB already defaults to 4-lane and no reconfiguration is needed.

Display both generated file paths and a one-line summary for each.

---

## Phase 6 — Build verification

### Step 1 — Run the build

Change to the repo root and run:

```bash
cd <REPO_PATH>
sh ./docker/build <BUILD_TARGET>
```

Stream output to the user. The build may take several minutes.

### Step 2 — Analyze result

**On success** (exit code 0, no errors):

```
Build: PASSED ✓
- All new sensor files compiled successfully
- docker/build <BUILD_TARGET> exited with code 0

Next step: Run the new sensor's player app to validate on hardware.
Would you like to run /hsb-app-skill to launch <SENSOR_NAME>_player.py? (yes/no)
```

**On failure** — do not dump raw build output. Summarize:

```
Build: FAILED
- Command: sh ./docker/build <BUILD_TARGET>
- Likely cause: <analysis>
- Failed file: <file>:<line> (if identifiable)
- Suggested fix: <action>
```

Common failure categories:

| Error | Cause | Fix |
|-------|-------|-----|
| `fatal error: renesas_bajoran_lite_ts1.hpp: No such file or directory` | Header lives in `imx274/`, not in the new sensor's directory | Use `#include "../imx274/renesas_bajoran_lite_ts1.hpp"` in `native_<sensor>_sensor.cpp` |
| `array size mismatch` / `static_assert` | Array size constant doesn't match actual entries | Recount entries in `<sensor>_mode.hpp` and update the `constexpr size_t` constant |
| `undeclared identifier` / `no member named` | Wrong class name, namespace, or mode enum reference | Check naming convention table and fix references |
| `error: no such file` in `#include` | Wrong include path | Verify include paths in `.cpp` files |
| `undefined reference to ...` | Missing `add_subdirectory` in parent CMakeLists | Add `add_subdirectory(<sensor>)` to the correct parent CMakeLists |
| Python `ModuleNotFoundError` | Missing `add_subdirectory` or missing `_MODULES` entry | Check `python/hololink/sensors/CMakeLists.txt` and `__init__.py` |

After diagnosing, show the user the specific files that will be changed and the proposed fix. Require explicit user confirmation before applying any fix and re-running the build. Allow up to 3 fix attempts; after each failed retry, ask the user for guidance before continuing.

### Step 3 — Session summary

After build success (or after max retries):

```
**Phase 6 — Build verification**
- Files generated:
    python/hololink/sensors/<sensor>/__init__.py
    python/hololink/sensors/<sensor>/<sensor>_mode.py
    python/hololink/sensors/<sensor>/<sensor>_cam.py
    python/hololink/sensors/<sensor>/CMakeLists.txt
    src/hololink/sensors/camera/<sensor>/native_<sensor>_sensor.hpp
    src/hololink/sensors/camera/<sensor>/native_<sensor>_sensor.cpp
    src/hololink/sensors/camera/<sensor>/<sensor>_mode.hpp
    src/hololink/sensors/camera/<sensor>/<sensor>_mode.cpp
    src/hololink/sensors/camera/<sensor>/CMakeLists.txt
    examples/<sensor>_player.py
    examples/linux_<sensor>_player.py
- Files modified:
    python/hololink/sensors/__init__.py  (added <sensor> to _MODULES)
    python/hololink/sensors/CMakeLists.txt  (added add_subdirectory)
    src/hololink/sensors/camera/CMakeLists.txt  (added add_subdirectory)
- Build: PASSED / FAILED
- TODOs remaining: <list any placeholder TODOs in the generated files>

To test on real hardware, run: /hsb-app-skill
```

---

## Important rules

1. **Never copy IMX274 register values verbatim** — all register addresses and values must come from the extracted datasheet or the reconciled CSV. The IMX274 register tables are sensor-specific and wrong for any other sensor.

2. **Use placeholders with `# TODO` markers** for anything that cannot be extracted from the datasheet (start/stop sequences, test pattern registers, optical black count, gain/exposure addresses). Never silently guess a value.

3. **All register values in hex** — use `0xXX` format for all addresses and values, consistent with the IMX274 pattern.

4. **C++ array sizes must be exact** — count every entry in the `constexpr std::array`, including `TABLE_WAIT_MS` sentinels. A wrong count will produce a compile-time `static_assert` failure.

5. **License headers** — every generated file must include the Apache-2.0 SPDX license header matching the year and format used in other files in the same directory.

6. **Write to user's repo** — use the `REPO_PATH` the user confirmed in Phase 0. If the repo is on the devkit, transfer files via SSH (`scp` or `ssh` + heredoc). If local, write directly.

7. **Coding style should be consistent** -- use the 'REPO_PATH' to verify the coding style and naming conventions.

8. **Never assume Python and C++ APIs mirror each other** — always read the actual Python source before writing Python code, even when the C++ equivalent is already known. Names, signatures, and patterns frequently differ (e.g., a C++ `const` array may be a Python function call; a C++ include path may not correspond to any Python import path). Use `grep` or `Read` on the relevant `.py` files to verify before writing.

9. **Always verify register values against the datasheet** — never use, modify, or derive register values (including timing registers like LINE_LENGTH_PCK and FRM_LENGTH_LINES) without first reading the relevant section of the datasheet to confirm the value, its constraints, and the formula governing its effect. If the datasheet is not available or does not cover a register, flag it with a `# TODO: verify from datasheet` comment.

10. **Add inline comments on every register** — every `(address, value)` entry in a register table must have an inline comment explaining what the register does and what the value means (e.g., register name, key bit field description, or functional effect). Do not leave bare hex tuples without explanation. Look up the register in the sensor datasheet or SerDes datasheet before writing the comment. If the register is not documented, write what functional block it belongs to and mark it `# TODO: verify register meaning`. Example:
    ```python
    (0x0301, 0xC8),  # PLL1_CTRL1: PLL1 pre-divider and charge pump
    (0x380E, 0x05),  # VTS_H: total rows per frame high byte
    (0x4319, 0x33),  # HDR combine: keep both bitwidth groups enabled
    ```

11. **No em dashes in generated files** — do not use em dashes (`—`) in any generated source code, comments, or docstrings. Use a regular hyphen (`-`) instead.

12. **Python cam class API contract** — these are required by `LinuxReceiverOperator` and the `ImageProcessorOp` linux-player API. Violating them causes silent runtime failures or `AttributeError`:
   - `bayer_format()` must return `hololink_module.sensors.csi.BayerFormat.<X>` (enum), never a plain string like `"GBRG"`.
   - `pixel_format()` must exist and return `hololink_module.sensors.csi.PixelFormat.<X>` (enum).
   - `configure_converter()` must pass `self._pixel_format` (enum) to `converter.transmitted_line_bytes()`, not a raw integer depth like `10`.
   - `setup_clock` calls `hololink_module.renesas_bajoran_lite_ts1.device_configuration()` (a function, not a constant). Do NOT add `from hololink.sensors import renesas_bajoran_lite_ts1` — that module lives at `hololink.renesas_bajoran_lite_ts1`, not under `hololink.sensors`, and the import will raise `ImportError` at runtime.
   - **Direct-MIPI sensors only** (IMX274/IMX681 pattern):
     - `__init__` must accept both `i2c_bus=hololink_module.CAM_I2C_BUS` and `expander_configuration=0` and set up `li_i2c_expander` — the player passes `expander_configuration` as a keyword argument.
     - `_write_reg`/`_read_reg` must use `i2c_lock` + `self._i2c_expander.configure()` before every transaction (copy the pattern from `dual_imx274.py`). A bare `i2c_transaction()` call without the expander will silently address the wrong sensor port.
     - `li_i2c_expander.py` must be copied into the new sensor's directory and listed in its `CMakeLists.txt` — it is not importable from a sibling sensor package.
   - **GMSL sensors only** (OX08D10 pattern):
     - Do NOT include `expander_configuration` in `__init__` — use `--link` (values: `a`/`b` for GMSL link A and B) instead.
     - Do NOT copy or import `li_i2c_expander.py` — it is not used. I2C reaches the sensor through the GMSL tunnel.
     - `_write_reg`/`_read_reg` use bare `i2c_transaction()` — no `i2c_lock`, no expander call needed.

---

## Troubleshooting guide

When the sensor streams corrupt frames, shows no image, or behaves differently from a known-good reference driver, work through these checks in order before changing any other code.

### 1. Register write delays are too short (`TABLE_WAIT_MS` value)

**Symptom:** Sensor initializes without errors but image is corrupt, blank, or shows wrong colors. Test pattern works correctly (confirming the pipeline is functional) but live image does not.

**Cause:** The inter-register write delay (`TABLE_WAIT_MS` sentinel value) is set too short. Many Sony sensors require 50 ms between register table writes during mode initialization. Using 1 ms causes I2C writes to fail silently — the register is not written, but no error is returned.

**Fix:** Always take the `TABLE_WAIT_MS` value directly from the vendor-supplied register settings file, not from another sensor's driver. For Sony sensors this is typically `0x32` (50 ms). Do not reduce it without explicit vendor confirmation.

**IMX681 reference values (from Sony Register Settings file):**
- `IMX681_WAIT_MS = 0x32` (50 ms, inter-register delay)
- `IMX681_WAIT_MS_START = 0x1E` (30 ms, post-stream-on delay)

### 2. Post-stream-on delay is wrong (`WAIT_MS_START` value)

**Symptom:** Sensor starts streaming but first few frames are corrupt or missing.

**Cause:** The delay inserted after the stream-on register write (`0x0100 = 0x01`) is too short or uses the wrong constant. Sony sensors typically need 30 ms after stream-on before the first valid frame arrives.

**Fix:** Read the post-stream-on delay from the vendor register settings file. It is separate from the inter-register delay and often has a different value.

### 3. MIPI data rate mismatch between sensor and FPGA PHY

**Symptom:** No data received, or data is garbled. `program_mipi_phy` is called with a clock that does not match the sensor's IOP PLL.

**Cause:** The `clock_freq_mhz` argument to `program_mipi_phy` must equal `IOPCK / 2` (D-PHY is DDR). If IOPCK is read from the register settings file (e.g., 030D/030E/030F for Sony sensors), compute: `IOPCK = INCK * PLL_MPY / PRE_DIV`, then `clock_freq_mhz = IOPCK / 2`.

**Fix:** Compute `clock_freq_mhz` from the IOP PLL registers actually written in the mode table. If multiple modes use the same IOPCK (as IMX681 does for both 15fps and 30fps), use the same value for all modes. Never guess or copy from another sensor.

**IMX681 example:** Both 15fps and 30fps modes use IOPCK = 2200 MHz (030D=0x03, 030E=0x01, 030F=0x13) → `clock_freq_mhz = 1100`.

### 4. MIPI lane count mismatch

**Symptom:** No data received. HSB defaults to 4-lane MIPI; sensors that use 2 lanes require `program_mipi_phy` to be called before streaming.

**Fix:** Call `program_mipi_phy(0, <num_lanes>, <clock_freq_mhz>, hololink)` and `program_mipi_phy(1, <num_lanes>, <clock_freq_mhz>, hololink)` for both FPGA PHY indices after `hololink.reset()` and before `camera.setup_clock()`.

### 5. Wrong Bayer format

**Symptom:** Image streams but colors are wrong (e.g., sky appears green, skin tones are purple).

**Cause:** The Bayer pattern order (RGGB, BGGR, GRBG, GBRG) is incorrect. This must be read from the datasheet's pixel array orientation section, not guessed from the image.

**Fix:** Check the datasheet for the pixel order register (e.g., `PIX_ORDER` or equivalent). Test pattern output can confirm the pipeline is correct independently of Bayer format.

### 6. Embedded/EBD line count is wrong

**Symptom:** Image is shifted vertically, or the top N lines are corrupted even when test pattern is correct.

**Cause:** The embedded data (EBD) line count offset in `configure_converter` is wrong. EBD lines use a different MIPI data type (not RAW10) and are filtered by the HSB receiver -- they should not be counted. Set embedded line count to 0 unless the datasheet explicitly states that EBD lines are delivered as image data.

**Fix:** Set `<METADATA_LINES> = 0` and verify with the datasheet's embedded data section. Do not count EBD lines that use a separate MIPI DT from the image data.

---

## GMSL sensor bring-up

GMSL cameras (e.g., OX08D10 via Aumovio SAT301DP17 module) have a fundamentally different bring-up path from direct MIPI sensors like IMX274. This section documents what the skill cannot derive from the sensor datasheet alone, based on the OX08D10 bring-up experience.

### What makes GMSL different

A direct MIPI sensor connects like this:
```
Host -> Hololink FPGA -> MIPI PHY -> Sensor
I2C:  Host -> Hololink FPGA -> I2C expander -> Sensor
```

A GMSL camera adds two chips between the FPGA and the sensor:
```
Host -> Hololink FPGA -> [MAX96716A deserializer] --(GMSL2 coax)--> [MAX96719B serializer] -> Sensor
I2C:  tunneled through the GMSL link -- NOT through the LI I2C expander
```

This means every aspect of sensor bring-up changes: I2C transport, clock delivery, reset control, frame format routing, and MIPI PHY configuration all go through the SerDes chips rather than directly.

### What the sensor datasheet cannot tell you

| What you need | Where it actually comes from |
|---|---|
| Serializer I2C address (e.g., MAX96719B = 0x40) | Camera module interface spec (e.g., SAT301DP17, section 4) |
| Deserializer I2C address (e.g., MAX96716A = 0x28) | Taurotech/board schematic |
| GMSL link speed must be 6Gbps (not 3Gbps default) | MAX96716A datasheet + trial/error; wrong speed = GMSL never locks |
| Serializer MIPI PHY mode (C-PHY vs D-PHY, lane count) | Module spec + MAX96719B datasheet; vendor default is often C-PHY 2-lane |
| Sensor clock (RCLKOUT) delivered via serializer GPIO | Module schematic (GPIO6 = RCLKOUT, GPIO3 = sensor reset) |
| Sensor reset released via serializer GPIO | Module schematic; sensor cannot respond to I2C until reset is released |
| GMSL link lock polling register and bit | MAX96716A deserializer datasheet |
| Deserializer video pipe routing (pipe_sel, dphy_dst) | Board design + trial/error with vendor reference |
| Virtual channel mapping register (0x4813) | Vendor reference driver; not in OX08D10 public datasheet |
| Vendor HSL sequences (system defaults, PWL, HDR combine) | Proprietary OX08DHsl.py from sensor vendor (OmniVision) |
| Voltage monitor thresholds (0x4583-0x458E) | Camera module vendor; module-specific supply design |
| ASIL/safety config (0x1035, 0x4Fxx) | Sensor vendor; safety register map not published |

### What the datasheet CAN tell you (and the skill still needs to ask)

These are reliably extractable from the sensor datasheet:
- Sensor I2C address (7-bit)
- Active resolution and Bayer format
- EBD/embedded line count
- Start/stop streaming register (usually 0x0100)
- Exposure and gain register addresses
- Test pattern register addresses

### Pixel format warning

GMSL HDR sensors often have a mismatch between the datasheet "default" output format and the format the GMSL module actually streams. For the OX08D10:

- Datasheet register map default: RAW_8/RAW_10 (power-on reset value)
- GMSL module actual output: RAW_12 (forced by serializer tunnel mode and 5 output group registers)

**Never use the register-map default bit depth for a GMSL sensor.** Always confirm with the vendor reference driver or module interface spec.

### Received frame height warning

GMSL serializers can append non-image rows to the MIPI frame. For the OX08D10 via MAX96719B:

- Active image: 2160 lines
- Received by FPGA: 2176 lines (2160 + 16 trailing rows from serializer)

If you size the receive buffer for the active height only, the buffer overflows and frames are dropped. The frame format table must use `received_height` for buffer sizing, and `configure_converter` must pass `trailing_bytes` to skip the non-image rows.

**Pattern for configure_converter with trailing rows:**
```python
ACTIVE_HEIGHT = 2160
FRONT_EMBEDDED_LINES = 2

def configure_converter(self, converter):
    start_byte = converter.receiver_start_byte()
    transmitted_line_bytes = converter.transmitted_line_bytes(self._pixel_format, self._width)
    received_line_bytes = converter.received_line_bytes(transmitted_line_bytes)
    front_embedded_bytes = received_line_bytes * FRONT_EMBEDDED_LINES
    trailing_lines = self._received_height - FRONT_EMBEDDED_LINES - ACTIVE_HEIGHT
    trailing_bytes = received_line_bytes * trailing_lines
    converter.configure(
        start_byte + front_embedded_bytes,
        received_line_bytes,
        self._width,
        ACTIVE_HEIGHT,
        self._pixel_format,
        trailing_bytes,
    )
```

### No LI I2C expander for GMSL cameras

IMX274/IMX681 use the LI I2C expander (`li_i2c_expander.py`) to select the camera port on a shared I2C bus. GMSL cameras do NOT use the expander -- I2C reaches the sensor through the GMSL tunnel. Omit `li_i2c_expander` from the driver entirely. Use bare `i2c_transaction()` calls for both the sensor and the SerDes chips.

### Required documents for GMSL bring-up

When the skill encounters a GMSL sensor, it should ask the user to provide:

1. **Camera module interface specification** -- contains serializer I2C address, sensor I2C address, EEPROM address, GPIO assignments for reset and clock
2. **MAX96719B (or equivalent serializer) datasheet** -- MIPI PHY configuration registers, RCLKOUT enable, reset GPIO control
3. **MAX96716A (or equivalent deserializer) datasheet** -- link speed registers, pipe routing registers, tunnel mode enable, lock status bit
4. **Vendor reference driver** -- contains the working GMSL init sequence, pipe routing values (pipe_sel/dphy_dst), D-PHY vs C-PHY configuration, and any proprietary vendor HSL sequences

Without items 1-4, the skill can generate the sensor register driver (I2C read/write, streaming control, exposure/gain) but must stub out GMSL bring-up with `raise NotImplementedError`.

### GMSL bring-up sequence (for the driver cam file)

The bring-up must happen in this exact order:

```python
def configure_vendor_tunnel_path(self, mode, link):
    # 1. Configure deserializer control path: set 6Gbps speed, select link
    #    CRITICAL: Default link speed is 3Gbps; MAX96719B is 6Gbps-only.
    #    Wrong speed = GMSL link never locks, all I2C to sensor times out.
    self._configure_deserializer_control(link)

    # 2. Poll for GMSL link lock (deserializer lock bit, 10s timeout)
    self._wait_for_link_lock(link)

    # 3. Configure deserializer video pipe (tunnel mode, data type = RAW12)
    self._configure_deserializer_video(link)

    # 4. Configure serializer: MIPI PHY, RCLKOUT, packet-detect timer
    self._configure_serializer()

    # 5. Release sensor reset via serializer GPIO; wait 500ms for sensor boot
    self._release_sensor_reset()
    time.sleep(0.5)

    # 6. Clear serializer loss-of-lock flag
    self.set_max929x_register(SER_ADDR, 0x1024, 0x20)

    # 7. Validate sensor identity (I2C now live via GMSL tunnel)
    self.validate_sensor_identity()

    # 8. Write sensor mode registers
    for reg, val in self._mode_table(mode):
        self._write_table_entry(reg, val)

    # 9. Set virtual channel mapping
    self._set_virtual_channel(link)
```

### GMSL register I/O: two separate I2C helpers

GMSL drivers need two I2C helper pairs -- one for the sensor and one for the SerDes chips:

```python
# Sensor registers (16-bit address, 8-bit data, SCCB)
def get_register(self, register): ...
def set_register(self, register, value): ...

# Serializer/deserializer registers (also 16-bit address, 8-bit data)
# Called with the SerDes I2C address as the first argument
def get_max929x_register(self, i2c_address, register): ...
def set_max929x_register(self, i2c_address, register, value): ...
```

Both use bare `i2c_transaction()` -- no I2C expander, no i2c_lock needed for GMSL cameras.

### Always consult existing working GMSL drivers first

When any question arises during GMSL bring-up -- GMSL link not locking, unexpected register readback, I2C timeout, frame corruption -- **the first action is to find an existing working GMSL driver in the repo that uses the same SerDes chips and read its bring-up code line by line.** Do not guess values, invent workarounds, or reason from first principles. The existing drivers encode hard-won bring-up knowledge that is not in any public datasheet.

**This rule applies at every stage of driver generation, not just when something breaks.** Before writing any SerDes init code, find a working reference and copy its structure. Only deviate where the new sensor's datasheet or module spec explicitly requires it.

Search for existing GMSL drivers before writing any code:
```bash
grep -rl "configure_deserializer\|set_max929x_register\|MAX96716\|MAX96719" python/hololink/sensors/
```

Read every matching file. For each function that corresponds to something you need to implement (deserializer init, link lock poll, serializer config, sensor reset, virtual channel mapping), read the reference implementation carefully before writing the new one. Match the structure, register write order, wait times, and lock check logic from the working driver unless the SerDes datasheet or module spec says otherwise.

### Mode register tables: 45fps vs 30fps

For the OX08D10 specifically (and HDR sensors in general), the skill learned:

- **45fps**: Sensor runs at 45fps from power-on by default. Only 5 registers needed to force RAW12 output groups (0x4319, 0x431A, 0x431C, 0x431D, 0x431F). No PLL changes required.
- **30fps**: Requires full PLL reconfiguration with vendor-validated values. VTS/HTS cannot be computed from the datasheet without first knowing the validated PLL output frequency. Additionally requires voltage monitor thresholds, ASIL safety config, and 8 vendor HSL sequences from a proprietary file.

**General rule**: For HDR GMSL sensors, the lower frame rate modes require substantially more sensor init than higher frame rate modes. The default power-on state is the highest frame rate. Going slower requires PLL reprogramming and vendor SDK sequences.
