# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# See README.md for detailed information.

include(pybind11_add_hololink_module)

set(EMULATION_SENSORS_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/hololink/emulation/python/hololink/emulation/sensors)
        
pybind11_add_hololink_module(
    CPP_CMAKE_TARGET emulation_sensors
    CLASS_NAME "_emulation_sensors"
    SOURCES ${EMULATION_SENSORS_SOURCE_DIR}/_sensors.cpp
)
add_custom_command(TARGET emulation_sensors_python PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${EMULATION_SENSORS_SOURCE_DIR}/__init__.py ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Copying __init__.py to hololink/emulation/sensors/"
)