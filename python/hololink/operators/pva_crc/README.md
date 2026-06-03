# PVA CRC Operator

Hardware-accelerated CRC validation using NVIDIA PVA (Programmable Vision Accelerator).

## Requirements

- NVIDIA platform with PVA hardware
  - Refer to
    [Supported Platforms](https://docs.nvidia.com/pva/sdk/2.8.1/frequently-asked-questions.html#which-platforms-support-pva)
    for more details.
- PVA runtime version 2.9
  - If you are using AGX Thor, update to JetPack 7.1
  - If you are using IGX Orin-dGPU, update to BaseOS 2.0
  - These versions come with PVA runtime v2.9 installed at `/opt/nvidia/pva-sdk-2.9`
- Hololink demo container
  - Refer to
    [Build and Test Holoscan Sensor Bridge demo container](https://docs.nvidia.com/holoscan/sensor-bridge/latest/build.html)

## PVA CRC Files

**C++ implementation** (in `src/hololink/operators/pva_crc/`):

- `pva_crc.hpp` - C++ header defining the PvaCrc class API
- `libpva_2.9_crc.a` - Static library with PVA-accelerated CRC32 (downloaded at build
  time, or placed in `src/hololink/operators/pva_crc/lib/`)
- `cupva_allowlist_pva_crc_2.9` - cuPVA allowlist file (deploy on host; see
  [Copy allowlist and enable](#copy-allowlist-and-enable))

**Python layer** (in `python/hololink/operators/pva_crc/`):

- `pva_crc_op.py` - Holoscan operators (ComputePvaCrcOp, CheckPvaCrcOp) that wrap the
  C++ library
- `cpp/pva_crc_pybind.cpp` - PyBind11 wrapper exposing C++ API to Python
- `cpp/CMakeLists.txt` - CMake build configuration for the `pva_crc` Python extension
- `cpp/PreparePVADependencies.cmake` - Downloads PVA CRC static library when needed

**Build integration:**

- `PreparePVADependencies.cmake` (top-level in this operator) - Optional copy from
  `src/.../lib/` to build dir
- `CMakeLists.txt` - Python package installation configuration

## Build

From the hololink repo root, build the prototype and demo images (use `--dgpu` for IGX
Orin dGPU or `--igpu` for AGX Thor). **Build on a host that has
`/opt/nvidia/pva-sdk-2.9`** so the Python wheel includes the `pva_crc` C++ extension
(`hololink/operators/pva_crc/cpp/`); otherwise the demo container will have no PVA CRC
module and you will see `ModuleNotFoundError: No module named 'pva_crc'`.

```bash
sh docker/build.sh --dgpu
# or
sh docker/build.sh --igpu
```

## Copy allowlist and enable

On the **host** (outside the container), run once per system:

```bash
curl -f -L -o /tmp/cupva_allowlist_pva_crc_2.9 \
  "https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/pva/crc/v2.9/cupva_allowlist_pva_crc_2.9"
sudo cp /tmp/cupva_allowlist_pva_crc_2.9 /etc/pva/allow.d/cupva_allowlist_pva_crc_2.9
sudo nvidia-pva-allow update
sudo nvidia-pva-allow enable
```

## Launch container

```bash
sh docker/demo.sh
```

## Export library path

Inside the container:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra/:/opt/nvidia/pva-sdk-2.9/lib/aarch64-linux-gnu/:/usr/lib/aarch64-linux-gnu/nvidia
```

## Run

- **IGX Orin dGPU (IMX274):**

  `python3 examples/imx274_pva_crc_validation.py --frame-limit 100`

- **AGX Thor (VB1940):**

  `python3 examples/vb1940_fusa_pva_crc_validation.py --frame-limit 100`

**Expected output**: `SUCCESS! PVA matches camera perfectly (100%)`

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'pva_crc'`

**Cause**: C++ module not built (wheel was built without PVA SDK, or you are not using
the demo image).

**Solution**: Rebuild with [Build](#build) on a host that has `/opt/nvidia/pva-sdk-2.9`
so the wheel includes the PVA CRC extension.

### Error: `libnvscibuf.so: cannot open shared object file`

**Cause**: PVA SDK libraries not in `LD_LIBRARY_PATH`. You must explicitly set
`LD_LIBRARY_PATH` before running the application.

**Solution**: Set `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/tegra/:/opt/nvidia/pva-sdk-2.9/lib/aarch64-linux-gnu/:/usr/lib/aarch64-linux-gnu/nvidia
```

### Error: `PvaError_PVAVPUAuthFailed`

**Cause**: Allowlist is not deployed

**Solution**: On the **host** (outside the container), deploy the allowlist per
[Copy allowlist and enable](#copy-allowlist-and-enable).
