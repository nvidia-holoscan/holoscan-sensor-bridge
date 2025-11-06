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

add_library(emulation_sensors STATIC
  vb1940_emulator.cpp
)
target_compile_options(emulation_sensors PRIVATE -DHOLOSCAN_SDK_VERSION=${HOLOSCAN_SDK_VERSION} -g3)
set_property(TARGET emulation_sensors PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::emulation::sensors ALIAS emulation_sensors)

find_package(CUDAToolkit REQUIRED)

target_include_directories(emulation_sensors
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../>
    $<INSTALL_INTERFACE:src>
)

target_link_libraries(emulation_sensors
  PUBLIC
    hololink::emulation
)

install(TARGETS emulation_sensors
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-emulation-sensors)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/vb1940_emulator.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/emulation/sensors
        COMPONENT hololink-emulation-sensors)

