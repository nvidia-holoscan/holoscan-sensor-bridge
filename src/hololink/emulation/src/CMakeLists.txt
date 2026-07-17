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

# INTERFACE library that the platform subdir attaches its includes/defines to.
# common/ then links against it PRIVATE so platform-specific include paths
# reach common .cpp files without leaking to common's downstream consumers.
add_library(hsb_platform INTERFACE)

# Order: platform first (so hsb_platform is populated and EMULATION*_PLATFORM_SOURCES
# are published into our scope), then common (which consumes hsb_platform PRIVATE
# and publishes EMULATION*_COMMON_SOURCES).
add_subdirectory(${HSB_EMULATOR_PLATFORM})
add_subdirectory(common)

# INTERFACE library: consumers link to emulation_host and emulation_common (both STATIC).
# Order: host then common so the linker resolves host->common refs (e.g. validate_configuration).
# Use --whole-archive for emulation_host so all objects (e.g. stm32f7xx_it.c interrupt handlers)
# and strong referenced symbols are linked without explicit need for user application level; 
# they are only referenced from the vector table in the same static lib.
add_library(emulation INTERFACE)

# need --whole-archive to link all the objects in the static libraries. prevents erroneous dead code elimination from library link orders
target_link_libraries(emulation INTERFACE
    -Wl,--whole-archive
    emulation_host
    emulation_common
    -Wl,--no-whole-archive
    dlpack::dlpack
)

# Expose the project's src/ dir (BUILD) and ${CMAKE_INSTALL_INCLUDEDIR} (INSTALL)
# so consumers can use the prefixed include style #include "hololink/emulation/...".
# emulation_host / emulation_common / transport libs only publish up to src/hololink/
# (good enough for bare-name includes), but downstream code such as the pybind
# binding _emulation.cpp and the examples reference headers as
# hololink/emulation/<foo>.hpp, which needs src/ at build time and include/ after
# install. Headers are installed under ${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation/
# (see ../CMakeLists.txt), so include/ is the correct INSTALL_INTERFACE root.
target_include_directories(emulation INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# embedded targets will have a lower level hal driver that needs to get propagated
if (TARGET embedded_hal_driver)
    target_link_libraries(emulation INTERFACE embedded_hal_driver)
endif()

add_library(hololink::emulation ALIAS emulation)

##### Transport libraries — sources from common/ + ${HSB_EMULATOR_PLATFORM}/
#
# Source lists (EMULATION*_COMMON_SOURCES, EMULATION*_PLATFORM_SOURCES) are
# published by the subdirectories above via PARENT_SCOPE. We assemble the
# per-transport static libraries here so each one can mix common and
# platform-specific code without any subdir owning the library definition.

# Helper: define a transport library named <name> from the supplied sources,
# wire up the standard include directories (platform dir FIRST so e.g.
# linux/net.hpp wins over emulation/net.hpp for `#include "net.hpp"` from
# common/*.cpp), link it to the standard set of dependencies, install it,
# and create the hololink::emulation::<alias> alias.
function(_hsb_add_transport_library name alias)
    set(_sources ${ARGN})
    if(NOT _sources)
        return()
    endif()

    add_library(${name} STATIC ${_sources})

    target_include_directories(${name}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${HSB_EMULATOR_PLATFORM}>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation>)

    target_link_libraries(${name}
        PUBLIC
            hsb_platform
            emulation_host
            emulation_common
            dlpack::dlpack)

    if(TARGET embedded_hal_driver)
        target_link_libraries(${name} PUBLIC embedded_hal_driver)
    endif()
    if(TARGET CUDA::cudart)
        target_link_libraries(${name} PUBLIC CUDA::cudart)
    endif()

    set_property(TARGET ${name} PROPERTY POSITION_INDEPENDENT_CODE ON)

    add_library(hololink::emulation::${alias} ALIAS ${name})

    install(TARGETS ${name}
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-emulation)
endfunction()

_hsb_add_transport_library(emulationroce roce
    ${EMULATIONROCE_COMMON_SOURCES}
    ${EMULATIONROCE_PLATFORM_SOURCES})

# RoCE iCRC needs zlib's crc32 unless the platform provides a CRC peripheral
# (CRC_OFFLOAD). On the linux target ZLIB_FOUND is set by find_package(ZLIB).
if(TARGET emulationroce AND ZLIB_FOUND)
    target_link_libraries(emulationroce PUBLIC ${ZLIB_LIBRARIES})
endif()

_hsb_add_transport_library(emulationcoe coe
    ${EMULATIONCOE_COMMON_SOURCES}
    ${EMULATIONCOE_PLATFORM_SOURCES})

# hsb_platform must be in the export set because the transport static libs
# (emulationroce/coe) and emulation_common link to it: STATIC archives don't
# bake in their deps, so CMake propagates them through the link interface even
# when the link was PRIVATE. INTERFACE targets have no artifact; LIBRARY/ARCHIVE
# DESTINATIONs are ignored for it but kept here to share the install() call.
install(TARGETS emulation hsb_platform
        EXPORT HololinkTargets
        COMPONENT hololink-emulation)
