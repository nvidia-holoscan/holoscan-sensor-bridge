# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# See README.md for detailed information.

import distutils.command.build
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import cmake_build_extension
import setuptools

with open("../VERSION") as f:
    s = f.read()
    original_version = s.strip()
    m = re.match(r"[0-9\.]+", original_version)
    VERSION = m.group(0)


# Setuptools validates packages=[...] against the source tree before
# the build_ext (CMake) command runs, and rejects package_dir paths
# that escape the project root with "..". The hololink_module Python
# sources live outside python/ (under hololink_module/host/python,
# hololink_module/host/operators/python, hololink_module/module/<m>/
# python, hololink_module/python/sensors). Mirror them into a stable
# temp directory and point setuptools at it via absolute-path
# package_dir entries; CMake's install step later writes the same .py
# files plus the compiled pybind extensions into the wheel directly.
def _mirror_adapter_python_sources():
    here = Path(__file__).parent.resolve()
    mirror_root = (
        Path(tempfile.gettempdir()) / "hololink-module-py-mirror" / "hololink_module"
    )
    if mirror_root.exists():
        shutil.rmtree(mirror_root)
    mappings = [
        ("../hololink_module/host/python", ""),
        ("../hololink_module/host/operators/python", "operators"),
        ("../hololink_module/module/hsb_lite/python", "hsb_lite"),
        ("../hololink_module/python/sensors", "sensors"),
        ("../hololink_module/python/sensors/imx274", "sensors/imx274"),
        ("../hololink_module/host/sensors/vb1940/python", "sensors/vb1940"),
        ("../hololink_module/python/sensors/ar0234", "sensors/ar0234"),
        ("../hololink_module/python/sensors/deserializers", "sensors/deserializers"),
        (
            "../hololink_module/python/sensors/deserializers/max96716a",
            "sensors/deserializers/max96716a",
        ),
        ("../hololink_module/python/sensors/serializers", "sensors/serializers"),
        (
            "../hololink_module/python/sensors/serializers/max9295d",
            "sensors/serializers/max9295d",
        ),
        ("../hololink_module/python/sensors/hawk", "sensors/hawk"),
        ("../hololink_module/module/taurotech_da326/python", "taurotech_da326"),
    ]
    for src_rel, dst_rel in mappings:
        src = (here / src_rel).resolve()
        dst = mirror_root / dst_rel
        dst.mkdir(parents=True, exist_ok=True)
        for py in src.glob("*.py"):
            shutil.copy(py, dst / py.name)
    return mirror_root


_ADAPTER_MIRROR_ROOT = _mirror_adapter_python_sources()


# Override build command so that CMake's binary dir lives in a known
# location across runs:
# - If HOLOLINK_BUILD_DIR is set (e.g. "/tmp/hololink-build"), use it
#   verbatim. Successive `pip install` runs reuse the same CMake
#   binary dir, so unchanged compile units are skipped and incremental
#   builds work.
# - Otherwise, fall back to a per-invocation TemporaryDirectory so CI
#   (gitlab-runner) doesn't trip over leftover CMake-produced files
#   it can't delete.
class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        hololink_build_dir = os.environ.get("HOLOLINK_BUILD_DIR")
        if hololink_build_dir:
            os.makedirs(hololink_build_dir, exist_ok=True)
            self.build_base = hololink_build_dir
            self._tmp_dir = None
        else:
            self._tmp_dir = TemporaryDirectory()
            self.build_base = self._tmp_dir.name


def has_coe_offload():
    if os.path.isdir("/dev/") and "coe-chan-0" in os.listdir("/dev"):
        return True
    return "COE_OFFLOAD" in os.environ and os.environ["COE_OFFLOAD"] == "1"


def get_l4t_version():
    # /sys/class/dmi/id/bios_version reports the UEFI firmware version, which on
    # IGX may carry a leading "v" (e.g. "v36.4.3"). Extract the dotted numeric
    # portion so int() never chokes on a "v" prefix or a trailing suffix, then
    # normalize to a fixed-length triple so version comparisons are reliable.
    with open("/sys/class/dmi/id/bios_version", "r") as f:
        bios_version = f.read().strip()
    m = re.search(r"\d+(?:\.\d+)*", bios_version)
    if not m:
        return (0, 0, 0)
    parts = [int(x) for x in m.group(0).split(".")]
    # Normalize to (major, minor, patch). A short match like "39" would yield
    # (39,), and in Python (39,) >= (39, 0, 0) is False -- which would wrongly
    # disable the argus-ISP build on an L4T-39 platform. Padding fixes that.
    return tuple((parts + [0, 0, 0])[:3])


setuptools.setup(
    name="hololink",
    version=VERSION,
    # The hololink package and tools/ live next to setup.py, so they
    # use the default (project-root) package_dir; the hololink_module
    # tree was mirrored above and is reached by absolute path.
    package_dir={
        "hololink_module": str(_ADAPTER_MIRROR_ROOT),
        "hololink_module.operators": str(_ADAPTER_MIRROR_ROOT / "operators"),
        "hololink_module.hsb_lite": str(_ADAPTER_MIRROR_ROOT / "hsb_lite"),
        "hololink_module.sensors": str(_ADAPTER_MIRROR_ROOT / "sensors"),
        "hololink_module.sensors.imx274": str(
            _ADAPTER_MIRROR_ROOT / "sensors" / "imx274"
        ),
        "hololink_module.sensors.vb1940": str(
            _ADAPTER_MIRROR_ROOT / "sensors" / "vb1940"
        ),
        "hololink_module.sensors.ar0234": str(
            _ADAPTER_MIRROR_ROOT / "sensors" / "ar0234"
        ),
        "hololink_module.sensors.deserializers": str(
            _ADAPTER_MIRROR_ROOT / "sensors" / "deserializers"
        ),
        "hololink_module.sensors.deserializers.max96716a": str(
            _ADAPTER_MIRROR_ROOT / "sensors" / "deserializers" / "max96716a"
        ),
        "hololink_module.sensors.serializers": str(
            _ADAPTER_MIRROR_ROOT / "sensors" / "serializers"
        ),
        "hololink_module.sensors.serializers.max9295d": str(
            _ADAPTER_MIRROR_ROOT / "sensors" / "serializers" / "max9295d"
        ),
        "hololink_module.sensors.hawk": str(_ADAPTER_MIRROR_ROOT / "sensors" / "hawk"),
        "hololink_module.taurotech_da326": str(
            _ADAPTER_MIRROR_ROOT / "taurotech_da326"
        ),
    },
    packages=[
        "hololink",
        "hololink/hololink_core",
        "hololink/emulation",
        "hololink/emulation/sensors",
        "hololink/operators",
        "hololink/sensors",
        "hololink/sensors/camera",
        "hololink/sensors/camera/imx274",
        "hololink/sensors/ecam0m30tof",
        "hololink/sensors/imx274",
        "hololink/sensors/imx715",
        "hololink/sensors/vb1940",
        "hololink/sensors/ecam0m30tof",
        "hololink_module",
        "hololink_module.operators",
        "hololink_module.hsb_lite",
        "hololink_module.sensors",
        "hololink_module.sensors.imx274",
        "hololink_module.sensors.vb1940",
        "hololink_module.sensors.ar0234",
        "hololink_module.sensors.deserializers",
        "hololink_module.sensors.deserializers.max96716a",
        "hololink_module.sensors.serializers",
        "hololink_module.sensors.serializers.max9295d",
        "hololink_module.sensors.hawk",
        "hololink_module.taurotech_da326",
        "tools",
    ],
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="_hololink",
            # Name of the resulting package name (import hololink)
            install_prefix="",
            # Selects the folder where the main CMakeLists.txt is stored
            source_dir="..",
            cmake_configure_options=[
                # This option points CMake to the right Python interpreter, and helps
                # the logic of FindPython3.cmake to find the active version
                f"-DPython3_ROOT_DIR={Path(sys.prefix)}",
                "-DBUILD_SHARED_LIBS:BOOL=OFF",
                # argus has a breaking change between l4t 36 (Jetpack 6.X) and l4t 39 (Jetpack 7.2+) for which this is compatible.
                f"-DHOLOLINK_BUILD_ARGUS_ISP={'ON' if os.path.isdir('/usr/src/jetson_multimedia_api/argus') and get_l4t_version() >= (39, 0, 0) else 'OFF'}",
                f"-DHOLOLINK_BUILD_SIPL={'ON' if has_coe_offload() else 'OFF'}",
                f"-DHOLOLINK_BUILD_FUSA={'ON' if has_coe_offload() else 'OFF'}",
                f"-DHOLOLINK_BUILD_PVA={'ON' if os.path.isdir('/opt/nvidia/pva-sdk-2.9') else 'OFF'}",
                # build roce only if the infiniband device driver & devices are present. It is not enough to check for the directory presence, e.g. AGX Thor
                f"-DHOLOLINK_BUILD_ROCE={'ON' if os.path.isdir('/sys/class/infiniband') and any(os.scandir('/sys/class/infiniband')) else 'OFF'}",
                f"-DHOLOLINK_ROCE_USE_GPU_VRAM={os.environ.get('HOLOLINK_ROCE_USE_GPU_VRAM', 'ON')}",
                # Add this to debug the code
                # "-DCMAKE_BUILD_TYPE=Debug",
                "-DHOLOLINK_PYTHON_INSTALL_DIR:PATH=.",
                # Use the lld linker when available (much faster than GNU BFD ld at
                # static-linking the example/tool binaries and the pybind extension).
                # Requires CMake >= 3.29; lld is installed in docker/Dockerfile. Guard
                # on presence so native builds without lld fall back instead of erroring.
                # (-fuse-ld=lld resolves the linker as ld.lld.)
                *(["-DCMAKE_LINKER_TYPE=LLD"] if shutil.which("ld.lld") else []),
                # Route compilation through sccache when it is available (e.g. the
                # docker/Dockerfile.demo build), so unchanged translation units are
                # served from the shared object cache instead of recompiled. Opt out
                # by setting HOLOLINK_USE_SCCACHE=OFF (docker/build.sh --disable-sccache).
                *(
                    [
                        "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                        "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                        "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
                    ]
                    if os.environ.get("HOLOLINK_USE_SCCACHE", "ON") != "OFF"
                    and shutil.which("sccache")
                    else []
                ),
            ],
        ),
    ],
    cmdclass={
        "build": BuildCommand,
        "build_ext": cmake_build_extension.BuildExtension,
    },
    install_requires=[
        "cuda-python",
        "nvtx",
    ],
    entry_points={
        "console_scripts": [
            "hololink=tools.hololink:main",
            "imx274=tools.imx274:main",
            "polarfire_esb=tools.polarfire_esb:main",
        ],
    },
)
