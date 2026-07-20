# ADI ToF Camera Player (`adcam_player`)

A Holoscan-based application for capturing, processing, and visualizing depth
data from an ADI ADTF3175 Time-of-Flight camera connected via the Holoscan
Sensor Bridge (HSB).

---

## Table of Contents

- [Overview](#overview)
- [End-to-End Data Flow](#end-to-end-data-flow)
- [Prerequisites](#prerequisites)
- [Getting the Source](#getting-the-source)
- [Build](#build)
- [Usage](#usage)
- [Command-Line Options](#command-line-options)
- [Execution Flow](#execution-flow)
- [Operator Graph](#operator-graph)
- [ADTFUnpackOp — Frame Unpacking](#adtfunpackop--frame-unpacking)
- [Firmware Update](#firmware-update)
- [Troubleshooting](#troubleshooting)

---

## Overview

`adcam_player` connects to the HSB over the network, initializes the ADTF3175
ToF sensor via I2C, receives MIPI/CSI frames, unpacks the ADI 5-byte-per-pixel
format, and displays three output planes side-by-side using Holoviz:

| Panel | Content |
|-------|---------|
| Left (0 – 33%) | Depth |
| Center (33 – 66%) | Active Brightness |
| Right (66 – 100%) | Confidence |

---

## End-to-End Data Flow

```
[ ToF Sensor (ADSD3100 / ADTF3066) ]
    │
    │  Control: I2C
    │    - Exposure / modulation / modes
    │
    │  Data: MIPI CSI-2 (RAW phase / amplitude)
    ▼
[ ADSD3500 Dual Depth Processor ]
    │
    │  Processing:
    │   - Phase → Depth conversion
    │   - Calibration
    │   - Filtering / noise reduction
    │
    │  Output Streams:
    │   - Depth (16-bit)
    │   - Active Brightness (AB, 16-bit)
    │   - Confidence (8-bit)
    │
    │  Interface:
    │   - MIPI CSI-2 (processed frames)
    ▼
═══════════════════════════════════════
[ HSB FPGA (Sensor Bridge) ]
═══════════════════════════════════════
    │
    │  1. MIPI RX (CSI-2 Capture)
    │     - Lane alignment
    │     - Frame/line decoding
    │
    │  2. Frame Assembly
    │     - Line buffers → complete frame
    │     - MIPI line synchronization
    │
    │  3. Packetization
    │     - Frame → UDP packets
    │     - Header fields added:
    │         • Frame ID
    │         • Packet sequence #
    │         • Stream ID (frame stream)
    │         • Timestamp
    │
    │  4. Ethernet Transmission
    │     - 10/25GbE MAC → PHY
    │     - Optical module (SFP+/QSFP)
    │     - Jumbo frames (~9000 MTU)
    │
    │  5. Control Plane (Parallel)
    │     - I2C tunneling from host → sensor
    ▼
═══════════════════════════════════════
🔗 Direct Optical Ethernet Link
(HSB ↔ AGX Thor / IGX Orin)
═══════════════════════════════════════
    │
    │  Key Characteristics:
    │   - Point-to-point (no switch)
    │   - Very low latency
    │   - No routing / minimal packet loss (if tuned)
    │
    ▼
[ NVIDIA Host NIC (Thor / IGX Orin) ]
    │
    │  Hardware Layer:
    │   - RX queues (RSS / multi-queue)
    │   - Ring buffers
    │   - Interrupt moderation
    │
    │  Data Path Options:
    │   (A) UDP Socket → CPU → GPU
    │   (B) RDMA / GPUDirect → GPU (preferred)
    ▼
═══════════════════════════════════════
[ Holoscan Sensor Bridge Receiver Operator ]
═══════════════════════════════════════
    │
    │  1. Packet Receive
    │     - Pull from NIC (socket or RDMA)
    │
    │  2. Packet Handling
    │     - Sequence validation
    │     - Reordering
    │     - Loss detection
    │
    │  3. Frame Reassembly
    │     - Packets → full image
    │     - Frame completeness check
    │
    │  4. Memory Placement
    │     - Pinned host memory OR
    │     - Direct GPU buffer (GPUDirect)
    ▼
═══════════════════════════════════════
🧠 Holoscan Application (ADCAM Player App)
═══════════════════════════════════════
    │
    │  Pipeline Graph:
    │
    │   RoceReceiverOp / LinuxReceiverOp
    │        ↓
    │   CsiToBayerOp
    │        ↓
    │   ADTFUnpackOp  (GPU: unpack + colorize)
    │        ↓
    │   HolovizOp
    ▼
[ Final Output ]
    - Holoviz display (Depth / ActiveBrightness / Confidence)
    - Saved frames
```

---

## Prerequisites

- NVIDIA Holoscan SDK
- Holoscan Sensor Bridge connected at `192.168.0.2` (default)
- ADI ADTF3175 sensor module
- CUDA-capable GPU
- (Optional) InfiniBand/ROCE NIC for high-bandwidth reception

---

## Getting the Source

```bash
git clone https://github.com/nvidia-holoscan/holoscan-sensor-bridge.git
cd holoscan-sensor-bridge
```

---

## Build

### 1. Build the container and start it (from the repo root on the devkit)

```bash
sh ./docker/build.sh

export DISPLAY=$DISPLAY

xhost +

sh ./docker/demo.sh
```

### 2. Inside the container — configure

```bash
export LD_LIBRARY_PATH=/opt/nvidia/holoscan/lib:$LD_LIBRARY_PATH
cmake -S . -B build
```

### 3. Build

```bash
cmake --build build -j$(nproc)
```

### 4. Source directory

```bash
ls examples/aditof/
# cpp/                — C++ sources and CMakeLists.txt
# python/             — Python helper scripts
# adi_manifest.yaml   — Firmware download manifest (URL, size, MD5 for ADCAM_Fw_Dual_Update_X.Y.Z.bin)
# README.md           — This file
```

### 5. Output binary

```
./build/examples/aditof/cpp/adcam_player
```

---

## Usage

### C++ player

```bash
./build/examples/aditof/cpp/adcam_player [options]
```

### Python player

```bash
python3 examples/aditof/python/adcam_player.py [options]
```

### Examples

#### Reset and capture (first run / cold start)

```bash
# C++    — full power-on reset, then capture
./build/examples/aditof/cpp/adcam_player --resetAdcam 1 --capture 1

# Python — full power-on reset, then capture
python3 examples/aditof/python/adcam_player.py --resetAdcam 1 --capture 1
```

#### Capture only (device already running)

```bash
# C++
./build/examples/aditof/cpp/adcam_player --capture 1

# Python
python3 examples/aditof/python/adcam_player.py --capture 1
```

#### ADSD3100 — QMP modes (512×512, 1 Gbps MIPI)

```bash
# C++    — mode 3: QMP, 512×512, ab_averaging on
./build/examples/aditof/cpp/adcam_player --captureMode 3 --capture 1

# Python — mode 3
python3 examples/aditof/python/adcam_player.py --captureMode 3 --capture 1

# C++    — mode 6 (default): QMP, 512×512, ab_averaging on
./build/examples/aditof/cpp/adcam_player --captureMode 6 --capture 1

# Python — mode 6 (default)
python3 examples/aditof/python/adcam_player.py --captureMode 6 --capture 1
```

#### ADTF3066 — VGA modes (512×640, 1 Gbps MIPI)

```bash
# C++    — mode 0: VGA, 512×640
./build/examples/aditof/cpp/adcam_player --captureMode 0 --capture 1

# Python — mode 0
python3 examples/aditof/python/adcam_player.py --captureMode 0 --capture 1

# C++    — mode 7: VGA, 512×640
./build/examples/aditof/cpp/adcam_player --captureMode 7 --capture 1

# Python — mode 7
python3 examples/aditof/python/adcam_player.py --captureMode 7 --capture 1
```

#### ADTF3066 — QVGA modes (256×320, 1 Gbps MIPI)

```bash
# C++    — mode 3: QVGA, 256×320
./build/examples/aditof/cpp/adcam_player --captureMode 3 --capture 1

# Python — mode 3
python3 examples/aditof/python/adcam_player.py --captureMode 3 --capture 1

# C++    — mode 6 (default): QVGA, 256×320
./build/examples/aditof/cpp/adcam_player --captureMode 6 --capture 1

# Python — mode 6 (default)
python3 examples/aditof/python/adcam_player.py --captureMode 6 --capture 1

# C++    — mode 8: QVGA, 256×320
./build/examples/aditof/cpp/adcam_player --captureMode 8 --capture 1

# Python — mode 8
python3 examples/aditof/python/adcam_player.py --captureMode 8 --capture 1
```

#### Firmware update

```bash
# C++
./build/examples/aditof/cpp/adcam_player --firmwareUpdate adi_manifest.yaml

# Python
python3 examples/aditof/python/adcam_player.py --firmwareUpdate adi_manifest.yaml
```

---

## Command-Line Options

### Common options (C++ and Python)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--resetAdcam <0\|1>` | int | `0` | Full power-on reset sequence (power rails + GPIO); prints firmware versions after reset |
| `--resetPin <0-31>` | int | `0` | GPIO pin number used for camera reset |
| `--captureMode <n>` | int | `6` | Capture mode (0–9); valid modes depend on detected imager type |
| `--capture <0\|1>` | int | `0` | `1` = start capture pipeline |
| `--firmwareUpdate <file>` | string | — | Path to firmware manifest YAML |
| `--frame-limit <n>` | int | `300` | Stop after N frames (`0` = unlimited) |
| `--ibv-name <dev>` | string | auto-detected | InfiniBand/network device name |
| `--ibv-port <n>` | int | `1` | InfiniBand port number |
| `--log-level <level>` | string | `info` | Log verbosity: `trace` `debug` `info` `warn` `error` |
| `-h`, `--help` | flag | — | Print usage |

### C++ only options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--hololink <ip>` | string | `192.168.0.2` | Override HSB board IP address |
| `--headless` | flag | off | Run without display window (no HolovizOp GUI) |
| `--fullscreen` | flag | off | Run Holoviz in fullscreen mode |

### Python only options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--resetOnly <0\|1>` / `-RO` | int | `0` | GPIO-only soft reset (no power cycle) |
| `--getStatus <0\|1>` / `-gs` | int | `0` | Read and log chip status registers |
| `--force` | flag | off | Allow firmware downgrade (requires `--firmwareUpdate`) |

**Capture mode → frame geometry and imager settings**

The correct mode table is selected automatically at runtime via `get_imager_type_and_ccb_version()`
after the sensor is detected. Two tables are defined in `adcam_lib.hpp`:

#### `adsd3100_standardModes` — ADSD3100

| Mode Number| MIPI width (bytes) | MIPI height | Pixel dims | Type | Dump Type | Range |
|------|-------------------|-------------|------------|------|-----------|-------|
| 0 | 3072 | 1707 | 1024 × 1024 | MP | Native | Short |
| 1 | 3072 | 1707 | 1024 × 1024 | MP | Native | Long |
| 2 | 2560 | 512 | 512 × 512 | QMP | 2x2 analog | Short |
| 3 | 2560 | 512 | 512 × 512 | QMP | 2x2 analog | Long |
| 5 | 2560 | 512 | 512 × 512 | QMP | mixed bin | Long |
| **6** (default) | **2560** | **512** | **512 × 512** | QMP | mixed bin | Short |

All modes: `phase_depth_bits`=6 (16-bit), `ab_bits`=6 (16-bit), `confidence_bits`=2 (8-bit), `depth_enable`=1, `output_mipi`=2.
MP modes require 1.5 Gbps MIPI; QMP modes require 1 Gbps MIPI.

#### `adtf3066_standardModes` — ADTF3066

| Mode Number| MIPI width (bytes) | MIPI height | Pixel dims | Type | Dump Type | Range |
|------|-------------------|-------------|------------|------|-----------|-------|
| 0 | 2560 | 640 | 512 × 640 | VGA | native | Short |
| 1 | 2560 | 640 | 512 × 640 | VGA | native | Long |
| 7 | 2560 | 640 | 512 × 640 | VGA | native | Long |
| 3 | 1280 | 320 | 256 × 320 | QVGA | 2x2 analog | Long |
| **6** (default) | **1280** | **320** | **256 × 320** | QVGA | mixed bin | Short |
| 8 | 1280 | 320 | 256 × 320 | QVGA | mixed bin | Long |

All ADTF3066 modes: `phase_depth_bits`=6 (16-bit), `ab_bits`=6 (16-bit), `confidence_bits`=2 (8-bit), `ab_averaging`=1, `depth_enable`=1, `output_mipi`=2, 1 Gbps MIPI.

**Imager settings field encoding:**

| Field | Valid values | Encoding |
|-------|-------------|----------|
| `phase_depth_bits` | 0, 2, 3, 4, 5, 6 | `0`=0-bit, `2`=8-bit, `3`=10-bit, `4`=12-bit, `5`=14-bit, `6`=16-bit |
| `ab_bits` | 0, 2, 3, 4, 5, 6 | `0`=0-bit, `2`=8-bit, `3`=10-bit, `4`=12-bit, `5`=14-bit, `6`=16-bit |
| `confidence_bits` | 0, 1, 2 | `0`=off, `1`=4-bit, `2`=8-bit |
| `ab_averaging` | 0, 1 | bool: `0`=off, `1`=on |
| `depth_enable` | 0, 1 | bool: `0`=off, `1`=on |
| `output_mipi` | 0, 1, 2 | number of MIPI output lanes |

**ADSD3500 Set Imager Mode command** — `set_mode()` in `adcam_lib.cpp`:

`set_mode()` sends a two-word I²C command. The correct mode table (`adsd3100_standardModes`
or `adtf3066_standardModes`) is selected from `imager_type_` (set by `get_imager_type_and_ccb_version()`).
Both words are built dynamically via `adcam_make_mode_settings()`:

```
Word 1: 0xDAXX   — XX = mode number (e.g. mode 6 → 0xDA06)
Word 2: 0xYYYY   — bit-packed imager settings:

  Bit  0      : depth_enable
  Bit  1      : data_interleaving  (always 1)
  Bit  2      : ab_enable          (always 1)
  Bit  3      : ab_averaging
  Bits [6:4]  : phase_depth_bits   encoded as (6 − enum_val)
  Bits [9:7]  : ab_bits            encoded as (6 − enum_val)
  Bits [11:10]: confidence_bits    (direct: 0=off, 1=4-bit, 2=8-bit)
  Bits [13:12]: output_mipi        (direct: 0, 1, or 2 lanes)
```

Example computed values:

| Modes (ADSD3100) | `phase_depth_bits` | `ab_bits` | `confidence_bits` | `ab_averaging` | Word 2 |
|------------------|--------------------|-----------|-------------------|---------------|--------|
| 0, 1 | 6 | 6 | 2 | 0 | `0x2807` |
| 2, 3, 5, 6 | 6 | 6 | 2 | 1 | `0x280F` |

---

## Execution Flow

The following describes the complete call sequence from `main()`:

### 1. Initialization

```
main()
 ├─ holoscan::set_log_level()         Set Holoscan + HSB log level
 ├─ cuInit() / cuDeviceGet()          Initialize CUDA device
 ├─ cuDevicePrimaryCtxRetain()        Acquire CUDA primary context
 ├─ Enumerator::find_channel()        Discover HSB on the network
 ├─ DataChannel()                     Open the data channel
 └─ Adcam()                           Construct sensor instance (I2C + GPIO)
```

### 2. Reset (if `--resetAdcam 1`)

```
adcam_reset_power_on()
 ├─ configure_reset_low()             Assert GPIO reset pin LOW
 ├─ expander0_.set_register() ×N      Power rail sequencing via I2C expanders
 ├─ configure_reset_high()            Release GPIO reset pin HIGH
 └─ sleep(5s)                         Wait for ADTF3175 boot

// C++ player: immediately after reset, firmware versions are read and printed
switch_from_standard_to_burst()
get_fw_version_burst_mode(GET_MASTER_FIRMWARE_COMMAND)  → prints "Master Firmware version = X.Y.Z.W"
get_fw_version_burst_mode(GET_SLAVE_FIRMWARE_COMMAND)   → prints "Slave Firmware version = X.Y.Z.W"
switch_from_burst_to_standard()
```

### 3. Firmware Update (if `--firmwareUpdate <manifest>`)

```
Programmer::fetch_manifest()          Parse manifest YAML
Programmer::check_eula()              Display and accept license terms
Programmer::check_images()            Download/verify firmware images (MD5 + size)
Programmer::program_and_verify_images()
 └─ Adsd3500Flash::adsd3500_flash()   Flash master + slave firmware
       ├─ get_ChipID()                Verify chip responds
       ├─ switch_from_standard_to_burst_mode()
       ├─ get_fw_version_burst_mode() Read current version
       ├─ write firmware pages
       └─ get_fw_version_burst_mode() Verify new version
```
After a successful update the process exits; `--capture` is not required.

### 4. Sensor Probe and Imager Detection

```
probe_adcam_adtf3175()                      Read 0x0112; check ID == {0x59, 0x31}
 ├─ returns 1 → ADTF3175 confirmed present
 └─ returns 0 → auto adcam_reset_power_on() + retry
                 └─ still 0 → log error and exit
get_imager_type_and_ccb_version()           Read register 0x0032 (ADSD3500_CMD_GET_CHIP_INFO)
 └─ resp[0] (bits [15:8]) = Imager Type     1=ADSD3100, 2=ADTF3066
 └─ resp[1] (bits  [7:0]) = CCB Version     1=Ver0, 2=Ver1, 3=Ver2, 4=Ver3
 └─ Re-initializes width/height/pixel_width/pixel_height from correct mode table
    ADSD3100 → adsd3100_standardModes
    ADTF3066 → adtf3066_standardModes
```

### 5. Capture Pipeline (if `--capture 1`)

`compose()` is an override of `holoscan::Application::compose()`. It is **never
called directly** — the Holoscan framework calls it once inside `application->run()` to
build the operator graph before the scheduler starts.

```
main()
 └─ holoscan::make_application<HoloscanApplication>(headless, ..., adcam_inst, frame_limit)
      └─ application->run()
           │
           ├─ HoloscanApplication::compose()   ← called once by the framework
           │     │
           │     ├─ Step 1: make_condition<CountCondition | BooleanCondition>
           │     │           frame limit or run-forever condition
           │     │
           │     ├─ Step 2: make_resource<BlockMemoryPool>("csi_to_bayer_pool")
           │     │           device memory pool (2 blocks, uint16) for CSI→Bayer
           │     │
           │     ├─ Step 3: make_operator<CsiToBayerOp>("csi_to_bayer")
           │     │           allocator=csi_to_bayer_pool, cuda_device_ordinal
           │     │
           │     ├─ Step 4: Camera initialization and configuration
           │     │           probe_adcam_adtf3175()     — confirm sensor reachable
           │     │           configure_converter()      — set CSI frame geometry (width × height)
           │     │           set_mipi()                 — configure MIPI lane speed + deskew
           │     │           set_mode()                 — send Set Imager Mode command:
           │     │                                         Word 1: 0xDA00 | adcam_mode_
           │     │                                         Word 2: adcam_make_mode_settings()
           │     │                                                  (table selected by imager_type_)
           │     │           get_csi_length()           — compute frame_size for receiver
           │     │
           │     ├─ Step 5: make_operator<RoceReceiverOp | LinuxReceiverOp>("receiver")
           │     │           device_start → Adcam::start()
           │     │           device_stop  → Adcam::stop()
           │     │
           │     ├─ Step 6: make_resource<BlockMemoryPool>("ADTF_output_pool")
           │     │           device memory pool (8 blocks, uint16) for ADTFUnpackOp
           │     │
           │     ├─ Step 7: make_operator<ADTFUnpackOp>("ADIToF_data")
           │     │           width=get_pixel_width(), height=get_pixel_height(), num_planes=3
           │     │
           │     ├─ Step 8: make_operator<HolovizOp>("holoviz")
           │     │           Depth (left) / ActiveBrightness (center) / Conf (right)
           │     │
           │     └─ Step 9: add_flow × 3  — wire the operator graph
           │                 receiver → csi_to_bayer → ADIToF_data → holoviz
           │
           └─ GXF Scheduler starts
                └─ each frame: ADTFUnpackOp::compute() is called automatically
```

Stream start/stop callbacks:
- `device_start` → `Adcam::start()` — enable MIPI clock continuous mode, start streaming
- `device_stop`  → `Adcam::stop()`  — stop streaming

---

## Operator Graph

```
[RoceReceiverOp | LinuxReceiverOp]
         │ output → input
    [CsiToBayerOp]
         │ output → input
     [ADTFUnpackOp]
         │ output → receivers
      [HolovizOp]
    ┌────┴────┬─────────┐
  Depth    ActiveBR   Confidence
```

---

## ADTFUnpackOp — Frame Unpacking

`ADTFUnpackOp` is implemented across two source files and a shared header, all located under `cpp/`:

| File | Compiler | Role |
|---|---|---|
| `cpp/adcam_unpack_op.hpp` | — | Shared declarations: CUDA kernel launcher prototypes + operator class definition |
| `cpp/adcam_unpack_op.cu` | `nvcc` | GPU side — `__global__` CUDA kernels + launch wrapper functions |
| `cpp/adcam_unpack_op.cpp` | `g++` | CPU side — Holoscan operator lifecycle + GXF memory management; calls the launch wrappers from the `.cu` |

Both object files are linked together into the `adcam_player` binary.

---

### `adcam_unpack_op.cu` — GPU kernels

Contains all CUDA device code. Each kernel is exposed via a C-callable launcher
declared in the header and called from `adcam_unpack_op.cpp`:

| Kernel | Launcher | Purpose |
|---|---|---|
| `shift_and_cast_kernel` | `shift_and_cast_kernel(..., cudaStream_t)` | Converts `uint16_t` → `uint8_t` by right-shifting 8 bits (CSI buffer arrives as 16-bit words); launcher is a C++ overload of the same name with an extra `cudaStream_t` parameter |
| `unpack_kernel` | `unpack_kernel_launch()` | Splits 5 B/px packed stream → separate `depth[]`, `conf[]`, `ab[]` `uint16_t` arrays — one thread per pixel |
| `jet_kernel` | `jet_kernel_launch()` | Maps `depth[]` → RGB using a 256-entry Jet LUT stored in CUDA `__constant__` memory; depth normalized to 0–4000 mm |
| `grayscale_kernel` | `grayscale_kernel_launch(..., max_val)` | Maps `ab[]` / `conf[]` → grayscale RGB; `max_val=4096` for AB (12-bit), `max_val=255` for Confidence (8-bit) |

The Jet LUT is placed in `__constant__` memory for cached broadcast reads across all threads.

#### Kernel purpose details

**`shift_and_cast_kernel`** — Type conversion: `uint16_t` → `uint8_t`

The CSI/MIPI buffer from `CsiToBayerOp` arrives as 16-bit words. The sensor packs its
5-byte/pixel data into those words, but the actual payload byte values fit in 8 bits.
This kernel right-shifts each 16-bit word by 8 bits (`>> 8`) and truncates to `uint8_t`,
recovering the original raw bytes. Without this step the subsequent unpack kernel would
read garbage.

**`unpack_kernel`** — Demultiplex v8.1.0+ two-subframe frame into 3 planes (all modes)

The same two-subframe layout applies to all capture modes — MP, QMP, VGA, and QVGA.
`N = width × height` pixels; only the frame dimensions differ per mode:

| Imager | Mode | Pixel dims (W × H) | N pixels | Total frame bytes (5 × N) |
|--------|------|-------------------|----------|--------------------------|
| ADSD3100 | 0, 1 (MP) | 1024 × 1024 | 1,048,576 | 5,242,880 |
| ADSD3100 | 2, 3, 5, 6 (QMP) | 512 × 512 | 262,144 | 1,310,720 |
| ADTF3066 | 0, 1, 7 (VGA) | 512 × 640 | 327,680 | 1,638,400 |
| ADTF3066 | 3, 6, 8 (QVGA) | 256 × 320 | 81,920 | 409,600 |

One CUDA thread per pixel reads from two subframe regions:

```
// Subframe 1: Depth + Confidence interleaved (3 bytes/pixel)
sf1_base   = idx * 3
depth[idx] = raw[sf1_base] | (raw[sf1_base + 1] << 8)   // uint16 LE
conf[idx]  = raw[sf1_base + 2]                           // uint8

// Subframe 2: Active Brightness (2 bytes/pixel), after subframe 1
sf2_base   = N * 3 + idx * 2
ab[idx]    = raw[sf2_base] | (raw[sf2_base + 1] << 8)   // uint16 LE
```

Inputs/outputs (all device memory):
- `raw`   — `uint8_t*`  packed input (5 bytes × width × height)
- `depth` — `uint16_t*` unpacked depth plane
- `conf`  — `uint16_t*` unpacked confidence plane
- `ab`    — `uint16_t*` unpacked active brightness plane

Output: three separate `uint16_t` device arrays (`depth[]`, `conf[]`, `ab[]`),
each W×H — one value per pixel.

> **Firmware v8.1.0+ — Two-subframe frame layout (all modes)**
>
> Starting with firmware v8.1.0 the frame is split into **two subframes** instead
> of a single 5-byte/pixel interleaved stream. The structure is identical for all
> imager modes; only the pixel count N = W × H differs:
>
> | Subframe | Content | Bytes/pixel | Total bytes |
> |----------|---------|-------------|-------------|
> | 1 | Depth (16-bit) + Confidence (8-bit) interleaved per pixel | 3 | 3 × N |
> | 2 | Active Brightness (16-bit) for all pixels | 2 | 2 × N |
>
> Full stream layout:
>
> ```
> ┌─────────────────────────────────────────────────────┐
> │  Subframe 1 — Depth + Confidence (interleaved)      │
> │  [ D1_L | D1_H | C1 | D2_L | D2_H | C2 | ... ]      │
> │  3 bytes × N pixels  (N = W × H)                    │
> ├─────────────────────────────────────────────────────┤
> │  Subframe 2 — Active Brightness                     │
> │  [ AB1_L | AB1_H | AB2_L | AB2_H | ... ]            │
> │  2 bytes × N pixels                                 │
> └─────────────────────────────────────────────────────┘
> ```
>
> Per-pixel extraction (v8.1.0+, all modes):
> ```
> depth[i] = subframe1[i*3 + 0] | (subframe1[i*3 + 1] << 8)  → uint16
> conf[i]  = subframe1[i*3 + 2]                               → uint8
> ab[i]    = subframe2[i*2 + 0] | (subframe2[i*2 + 1] << 8)  → uint16
> ```
>
> The total frame size is always 5 × N bytes.
> The 5 bytes per pixel are fully consumed — there are **no padding or don't-care bytes**:
>
> | Subframe | Bytes/pixel | Role | Running total |
> |----------|-------------|------|---------------|
> | SF1 | 3 | Depth (2 B) + Confidence (1 B) | 3 × N |
> | SF2 | 2 | Active Brightness (2 B) | 2 × N |
> | **Total** | **5** | | **5 × N** |
>
> `3N + 2N = 5N` — the same total as the previous single-interleaved format.
> `unpack_kernel` handles this two-subframe layout for all modes (MP, QMP, VGA, QVGA).

**`jet_kernel`** — Depth → false-color RGB (Jet colormap)

Maps each `uint16_t` depth value to an RGB triplet using a precomputed 256-entry
Jet LUT stored in CUDA `__constant__` memory. Normalization: depth is divided by
4000 mm and scaled to 0–255. Near = blue, mid = green/yellow, far = red. Output:
`uint8_t` RGB image for the **Depth** panel in Holoviz.

**`grayscale_kernel`** — AB / Confidence → grayscale RGB

Maps each `uint16_t` value to a grayscale intensity using a `max_val` parameter,
then writes the same value to all three RGB channels. Used twice per frame with
different normalization ranges:

| Channel | `max_val` | Range | Reason |
|---------|-----------|-------|---------|
| Active Brightness | `4096.0` | 12-bit | AB is a 12-bit ADC value |
| Confidence | `255.0` | 8-bit | Conf is a direct uint8 value (v8.1.0+ subframe 1) |

---

### `adcam_unpack_op.cpp` — Holoscan operator logic

Contains the Holoscan lifecycle methods and GXF memory management:

| Method | What it does |
|---|---|
| `setup()` | Registers input/output ports and parameters with the Holoscan framework |
| `start()` | Computes `frame_size_ = width × height` once at pipeline startup |
| `compute()` | Called every frame — full pipeline orchestration (see below) |

**How `compute()` is invoked — framework dispatch**

`compute()` is **never called directly** from `adcam_player.cpp`. The Holoscan
framework calls it automatically on every incoming frame:

```
adcam_player.cpp — HoloscanApplication::compose()
 │
 ├─ make_operator<ADTFUnpackOp>("ADIToF_data", ...)   // register operator
 │
 ├─ add_flow(csi_to_bayer_operator, ADIToF_data,      // wire input
 │           {{"output", "input"}})
 │
 └─ add_flow(ADIToF_data, visualizer,                 // wire output
             {{"output", "receivers"}})

app.run()
 └─ GXF Scheduler
      └─ (each frame, triggered when CsiToBayerOp emits)
           └─ ADTFUnpackOp::compute(op_input, op_output, context)
```

`adcam_player.cpp` only declares *what* to run and *how operators connect* —
the GXF scheduler handles *when* `compute()` is called.

**`compute()` step-by-step per frame:**

```
 1. Receive GXF entity from CsiToBayerOp
 2. Extract CUDA stream from the message
 3. Get raw input tensor (uint16, 5 B/px packed); validate device storage
 4. Get BlockMemoryPool allocator handle
 5. Allocate output GXF tensors: "Depth", "ActiveBrightness", "Conf"  (uint8 RGB, device)
 6. Allocate scratch uint16 tensors: depthraw, confraw, abraw
 7. call shift_and_cast_kernel()   → uint8* raw          [from .cu]
 8. call unpack_kernel_launch()    → depth/conf/ab uint16 [from .cu]
 9. call jet_kernel_launch()       → depth_rgb uint8 RGB  [from .cu]
10. call grayscale_kernel_launch() → ab_rgb, conf_rgb     [from .cu]
11. Emit output GXF entity to HolovizOp
```

---


### Parameters (as configured in `adcam_player.cpp`)

| Parameter | Value | Description |
|---|---|---|
| `num_planes` | `3` | Depth + Active Brightness + Confidence |
| `width` | `get_pixel_width()` | Frame width in pixels (set from mode table after imager detection) |
| `height` | `get_pixel_height()` | Frame height in pixels (set from mode table after imager detection) |
| `allocator` | `BlockMemoryPool` (8 blocks, device) | GPU memory pool for output tensors |
| `in_tensor_name` | `""` | Unnamed input tensor from `CsiToBayerOp` |
| `out_tensor_name` | `"output"` | Output port name |

### Full data flow — Pre-unpack to Display

```
cpp/adcam_unpack_op.hpp    ← shared: kernel launcher prototypes + operator class
        │
        ├── cpp/adcam_unpack_op.cu    (nvcc)    cpp/adcam_unpack_op.cpp (g++)
        │                                        ADTFUnpackOp::compute()
        │                                          │
        │   [INPUT]                                │  step 1: receive GXF entity
        │   CsiToBayerOp output                    │  step 2: extract CUDA stream
        │   (uint16 words, 5 B/px packed)          │  step 3: get input tensor (uint16)
        │                                          │           validate device storage
        │                                          │  step 4: get BlockMemoryPool handle
        │                                          │  step 5: allocate output GXF tensors
        │                                          │           "Depth" / "ActiveBrightness"
        │                                          │           "Conf"  (uint8 RGB, device)
        │                                          │  step 6: allocate scratch uint16 tensors
        │                                          │           depthraw / confraw / abraw
        │                                          │
        │   shift_and_cast_kernel  ◄───────────────┤  step 7
        │     uint16 → uint8 (>>8)                 │
        │     output: uint8* raw                   │
        │                                          │
        │   unpack_kernel          ◄───────────────┤  step 8
        │     5 B/px → 3 planes                    │
        │     depth[]  uint16                      │
        │     conf[]   uint16                      │
        │     ab[]     uint16                      │
        │                                          │
        │   jet_kernel             ◄───────────────┤  step 9
        │     depth → Jet RGB                      │  GXF tensor "Depth" allocated
        │     (0–4000 mm, near=blue, far=red)      │
        │                                          │
        │   grayscale_kernel ×2    ◄───────────────┤  step 10
        │     ab   → grayscale RGB                 │  GXF tensor "ActiveBrightness"
        │     conf → grayscale RGB                 │  GXF tensor "Conf"
        │                                          │
        │                                          │  step 11: op_output.emit(out_message)
        │                                          │           push 3 tensors downstream
        │
        └── HolovizOp::compute()
              "Depth"            → left viewport   (0.00–0.33)  Jet color
              "ActiveBrightness" → center viewport (0.33–0.66)  Grayscale
              "Conf"             → right viewport  (0.66–1.00)  Grayscale
                    │
              GPU → display window "ADI ToF Player"
```

> All GPU work (unpack + colorization + rendering) stays on the same CUDA stream —
> no host–device synchronization between steps.

### Functions called per step — `ADTFUnpackOp::compute()` (steps 1–11)

All steps run inside `ADTFUnpackOp::compute()` in `adcam_unpack_op.cpp`. Steps 1–6 are CPU-side setup; steps 7–10 launch CUDA kernels; step 11 emits the result downstream.

| Step | Description | Functions called |
|------|-------------|-----------------|
| **1** | Receive GXF entity | `op_input.receive<holoscan::gxf::Entity>("input")` |
| **2** | Extract CUDA stream | `cuda_stream_handler_.from_message()`, `cuda_stream_handler_.get_cuda_stream()` |
| **3** | Get input tensor + validate storage | `entity.get<nvidia::gxf::Tensor>(...)`, `input_tensor->storage_type()`, `input_tensor->size()` |
| **4** | Get allocator handle | `nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(...)` |
| **5** | Allocate output GXF tensors (`Depth`, `ActiveBrightness`, `Conf`) | `nvidia::gxf::Entity::New()`, `out_message.add<nvidia::gxf::Tensor>(...)`, `tensor->reshape<uint8_t>(...)`, `tensor->data<uint8_t>()` |
| **6** | Allocate scratch `uint16` tensors (`depthraw`, `confraw`, `abraw`) | `scratch_entity.add<nvidia::gxf::Tensor>(...)`, `tensor->reshape<uint16_t>(...)`, `tensor->data<uint16_t>()` |
| **7** | uint16 → uint8 shift (`>>8`) | `shift_and_cast_kernel(raw_u16, raw, size * 5, stream)` |
| **8** | Unpack 5 B/px → depth/conf/ab uint16 planes | `unpack_kernel_launch(raw, depth, conf, ab, width, height, stream)` |
| **9** | depth uint16 → Jet RGB colormap | `jet_kernel_launch(depth, depth_rgb_ptr, size, stream)` |
| **10** | ab/conf uint16 → grayscale RGB | `grayscale_kernel_launch(ab, ab_rgb_ptr, size, stream)`, `grayscale_kernel_launch(conf, conf_rgb_ptr, size, stream)` |
| **11** | Emit output entity with 3 tensors | `op_output.emit(out_message)` |

---

## Firmware Update

### Steps to use the manifest YAML

1. Obtain the firmware binary (`.bin`) for the ADSD3500 sensor.
   The default manifest (`adi_manifest.yaml`) references `ADCAM_Fw_Dual_Update_X.Y.Z.bin`
   and will download it automatically from the ADI download server.
2. Compute the file size and MD5 checksum (if using a custom binary):
   ```bash
   wc -c ADCAM_Fw_Dual_Update_X.Y.Z.bin   # file size in bytes
   md5sum ADCAM_Fw_Dual_Update_X.Y.Z.bin  # MD5 hash
   ```
3. Fill in the values in the manifest YAML (`adi_manifest.yaml`) already provided:
   - `filename` — absolute or relative path to the firmware binary (or `url:` for remote fetch)
   - `size` — byte count from step 2
   - `md5` — MD5 hash from step 2
4. Run the updater:
   ```bash
   ./adcam_player --firmwareUpdate adi_manifest.yaml
   ```

The updater will:
1. Parse and validate the manifest
2. Prompt for EULA acceptance (unless `--accept-eula` is set in `Programmer::Args`)
3. Download or read the firmware binary and verify size + MD5
4. Flash master sensor, verify version
5. Flash slave sensor, verify version
6. Exit — power cycle the device before resuming capture

---

## Troubleshooting

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| `ADTF3175 NOT Found` | Sensor not initialized or powered off | Auto-reset is attempted once; if it persists, check sensor power and HSB connection |
| No frames received | MIPI not streaming | Verify `--captureMode` and `--resetPin` values |
| `Firmware flash failed` | Invalid binary or I2C error | Check manifest MD5/size and sensor power |
| Black/frozen Holoviz window | CUDA or IBV issue | Check `--ibv-name` and CUDA device availability |
| `Imager Type: Unknown (raw=...)` | `get_imager_type_and_ccb_version()` returned unexpected value | Check byte order: `resp[0]`=Imager Type, `resp[1]`=CCB Version |
| Wrong pixel dimensions | Imager not yet detected before `compose()` | Ensure `get_imager_type_and_ccb_version()` is called before `application->run()` |
