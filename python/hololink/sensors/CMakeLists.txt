# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# camera/, the top-level imx274/, and the hololink_sensor module below (sensors.cpp, providing
# Sensor/I2CExpanderOutputEN/LII2CExpander) depend on the legacy pybind11_add_hololink_module()
# macro -- broken against the installed HSDK5 holoscan-sdk (see python/hololink/CMakeLists.txt's
# own comment), so none of them are built here.
# ecam0m30tof/, imx715/, ox08d10/, and vb1940/ have no such dependency (pure Python, plain
# copy_hololink_python_files() only -- see each one's own CMakeLists.txt) and are wired in
# below. imx715/, ox08d10/, and (2026-10-02) vb1940/ were each investigated separately: their own
# CMakeLists.txt files already used the modern pattern (vb1940/'s has no .cpp at all -- imu.py,
# vb1940.py, and vb1940_mode.py are all pure Python, confirmed by reading them, not assumed from
# an older comment here claiming otherwise), this parent file's own add_subdirectory() call was
# just never added -- the same gap ecam0m30tof had before it was fixed.
include(copy_hololink_python_files)

copy_hololink_python_files(
    TARGET_NAME copy_hololink_sensors_python
    DESTINATION sensors
    FILES
        __init__.py
        csi.py
        imx477.py
        v4l2_ids.py
)

add_subdirectory(ecam0m30tof)
# pure Python (mode tables, the HSDK4-era driver and I2C expander helpers), used by the tests'
# camera_example_conf and importable without the unported pybind11 sensor modules
add_subdirectory(imx274)
add_subdirectory(imx715)
add_subdirectory(ox08d10)
add_subdirectory(vb1940)
