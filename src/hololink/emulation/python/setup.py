"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
See README.md for detailed information.
"""

import re

from setuptools import setup


def get_version():
    with open("../../../../VERSION", "r") as f:
        version = re.match(r"[0-9\.]+", f.read().strip()).group(0)
    return version


setup(
    name="hololink",
    version=get_version(),
    description="Holoscan Sensor Bridge Emulation",
    url="https://github.com/nvidia-holoscan/holoscan-sensor-bridge",
    packages=["hololink", "hololink.emulation", "hololink.emulation.sensors"],
    package_dir={"hololink": "hololink"},
    package_data={
        "hololink.emulation": ["*.so"],
        "hololink.emulation.sensors": ["*.so"],
    },
    include_package_data=True,
)
