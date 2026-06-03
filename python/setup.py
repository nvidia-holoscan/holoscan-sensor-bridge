# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cmake_build_extension
import setuptools

with open("../VERSION") as f:
    s = f.read()
    original_version = s.strip()
    m = re.match(r"[0-9\.]+", original_version)
    VERSION = m.group(0)


# Override build command to create a temporary directory where the build files are stored
# to avoid that files produced by CMake can't be deleted by gitlab-runner
class BuildCommand(distutils.command.build.build):
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self._tmp_dir = TemporaryDirectory()
        self.build_base = self._tmp_dir.name


def has_coe_offload():
    if os.path.isdir("/dev/") and "coe-chan-0" in os.listdir("/dev"):
        return True
    return "COE_OFFLOAD" in os.environ and os.environ["COE_OFFLOAD"] == "1"


def get_l4t_version():
    with open("/sys/class/dmi/id/bios_version", "r") as f:
        bios_version = f.read().strip()
    return tuple(int(x) for x in bios_version.split("-")[0].split("."))


setuptools.setup(
    name="hololink",
    version=VERSION,
    packages=[
        "hololink",
        "hololink/hololink_core",
        "hololink/emulation",
        "hololink/emulation/sensors",
        "hololink/operators",
        "hololink/sensors",
        "hololink/sensors/camera",
        "hololink/sensors/camera/imx274",
        "hololink/sensors/imx274",
        "hololink/sensors/imx715",
        "hololink/sensors/vb1940",
        "hololink/sensors/ecam0m30tof",
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
