# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# TEMPLATE platform — single-source layout. All platform-side stubs (DataPlane,
# HSBEmulator, I2CController, COEDataPlane, RoCEv2DataPlane subclass bodies, the
# wire-emit hooks, and the net-module helpers) live in HSBTemplate.cpp. The
# transport libraries (emulationcoe / emulationroce) pull their platform symbols
# in transitively via emulation_host.

add_library(emulation_host STATIC HSBTemplate.cpp)

target_include_directories(emulation_host
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation>
)

# Publish this directory on hsb_platform so common/*.cpp can resolve
# "data_plane.hpp", "hsb_emulator.hpp", etc. to the TEMPLATE shims. Also
# publish ZLIB_PATH (set by cmake/deps/zlib_crc32.cmake in TEMPLATE.cmake) so
# common/rocev2_transmitter.cpp can `#include <zlib.h>` for the software iCRC.
target_include_directories(hsb_platform INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${ZLIB_PATH}>
)

target_link_libraries(emulation_host
    PUBLIC
        emulation_common
        dlpack::dlpack
)

set_property(TARGET emulation_host PROPERTY POSITION_INDEPENDENT_CODE ON)

##### Transport sources — published to PARENT_SCOPE
#
# The transport-specific platform symbols (COETransmitter / RoCEv2Transmitter
# globals, the matching DataPlane subclass ctors/dtors, and the send_*_packet
# wire-emit hooks) live in HSBTemplate_coe.cpp and HSBTemplate_rocev2.cpp so they
# sit in the same archive as the common transport sources that supply the
# COETransmitter / RoCEv2Transmitter vtables (emulationcoe.a / emulationroce.a).
# Keeping them out of HSBTemplate.cpp avoids a circular link dependency between
# emulation_host.a and the transport archives.

set(EMULATIONCOE_PLATFORM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/HSBTemplate_coe.cpp
    PARENT_SCOPE)
set(EMULATIONROCE_PLATFORM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/HSBTemplate_rocev2.cpp
    PARENT_SCOPE)
set(EMULATIONUDP_PLATFORM_SOURCES PARENT_SCOPE)
