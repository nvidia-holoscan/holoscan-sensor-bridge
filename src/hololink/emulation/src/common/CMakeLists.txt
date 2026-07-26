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

add_library(emulation_common STATIC
    hsb_emulator.cpp
    hsb_config.cpp
    base_transmitter.cpp
    data_plane.cpp
    net.cpp
    utils.cpp
    apb_events.cpp
    register_memory.cpp
    i2c.cpp)

# Transport-specific common sources are NOT compiled into emulation_common.
# Instead they are published to PARENT_SCOPE and assembled into per-transport
# libraries (emulationroce, emulationcoe, ...) up in ../CMakeLists.txt, alongside
# the matching platform sources from src/${HSB_EMULATOR_PLATFORM}/.
set(EMULATIONROCE_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/rocev2_data_plane.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocev2_transmitter.cpp
    PARENT_SCOPE)
set(EMULATIONCOE_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/coe_data_plane.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/coe_transmitter.cpp
    PARENT_SCOPE)
set(EMULATIONUDP_COMMON_SOURCES
    PARENT_SCOPE)

target_sources(emulation_common
    PRIVATE
        $<$<BOOL:${ZLIB_PATH}>:${ZLIB_PATH}/crc32.c>
)

target_include_directories(emulation_common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../${HSB_EMULATOR_PLATFORM}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation>
    PRIVATE
        $<BUILD_INTERFACE:$<$<BOOL:${ZLIB_PATH}>:${ZLIB_PATH}>>
)

target_compile_definitions(emulation_common PUBLIC ${HSB_HAL_DEFINES})

target_link_libraries(emulation_common
    PUBLIC
        dlpack::dlpack
    PRIVATE
        # Picks up platform-specific include dirs/defines published by the
        # platform subdir (src/CMakeLists.txt processes ${HSB_EMULATOR_PLATFORM}/
        # before common/). PRIVATE so consumers of emulation_common don't inherit
        # STM32 paths; only common's own .cpp files do.
        hsb_platform
        $<$<BOOL:${ZLIB_FOUND}>:ZLIB::ZLIB>
)

set_property(TARGET emulation_common PROPERTY POSITION_INDEPENDENT_CODE ON)

install(TARGETS emulation_common
        EXPORT HololinkTargets
        COMPONENT hololink-emulation)
