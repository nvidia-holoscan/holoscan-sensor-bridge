# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_library(emulation_host STATIC
    data_plane.cpp
    hsb_emulator.cpp
    register_memory.cpp
    net.cpp)

target_include_directories(emulation_host
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation>
)

target_link_libraries(emulation_host
    PUBLIC
        emulation_common
        dlpack::dlpack
)

set_property(TARGET emulation_host PROPERTY POSITION_INDEPENDENT_CODE ON)

##### Transport target handling

add_library(emulationcoe STATIC
    coe_data_plane.cpp
    coe_transmitter.cpp
)
target_include_directories(emulationcoe 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation>
)
target_link_libraries(emulationcoe 
    PUBLIC
        hololink::emulation 
        CUDA::cudart 
)
set_property(TARGET emulationcoe PROPERTY POSITION_INDEPENDENT_CODE ON)
add_library(hololink::emulation::coe ALIAS emulationcoe)

install(TARGETS emulationcoe
    EXPORT HololinkTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT hololink-emulation
)

add_library(emulationroce STATIC
    rocev2_data_plane.cpp
    rocev2_transmitter.cpp
)

target_include_directories(emulationroce 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../../>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation>
)
target_link_libraries(emulationroce 
    PUBLIC 
        hololink::emulation 
        ${ZLIB_LIBRARIES} 
        CUDA::cudart 
)
set_property(TARGET emulationroce PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::emulation::roce ALIAS emulationroce)

install(TARGETS emulationroce
    EXPORT HololinkTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT hololink-emulation)
