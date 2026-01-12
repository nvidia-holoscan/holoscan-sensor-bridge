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
        "hololink/sensors/d555",
        "tools",
    ],
    ext_modules=[
        cmake_build_extension.CMakeExtension(
            name="_hololink",
            # Name of the resulting package name (import hololink)
            install_prefix="hololink",
            # Selects the folder where the main CMakeLists.txt is stored
            source_dir="..",
            cmake_configure_options=[
                # This option points CMake to the right Python interpreter, and helps
                # the logic of FindPython3.cmake to find the active version
                f"-DPython3_ROOT_DIR={Path(sys.prefix)}",
                "-DBUILD_SHARED_LIBS:BOOL=OFF",
                f"-DHOLOLINK_BUILD_ARGUS_ISP={'ON' if os.path.isdir('/usr/src/jetson_multimedia_api/argus') else 'OFF'}",
                f"-DHOLOLINK_BUILD_SIPL={'ON' if os.path.isdir('/usr/src/jetson_sipl_api') else 'OFF'}",
                f"-DHOLOLINK_BUILD_FUSA={'ON' if os.path.isdir('/usr/src/jetson_sipl_api/sipl/fusa') else 'OFF'}",
                # Add this to debug the code
                # "-DCMAKE_BUILD_TYPE=Debug",
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
