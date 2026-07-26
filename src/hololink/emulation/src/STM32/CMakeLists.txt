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

# Enable CMake support for ASM language
enable_language(ASM)

string(TOLOWER ${STM32_FAMILY} STM32_FAMILY_LOWER)
string(TOLOWER ${STM32_SERIES} STM32_SERIES_LOWER)

set(STM32_CMSIS_DEVICE_STARTUP
    ${STM32_CMSIS_DEVICE_PATH}/Source/Templates/${MCU_TOOLCHAIN}/startup_${STM32_FAMILY_LOWER}xx.s)
message(STATUS "STM32_CMSIS_DEVICE_STARTUP = ${STM32_CMSIS_DEVICE_STARTUP}")

# build HAL driver target
set(STM32_HAL_DEFINES ${STM32_HAL_DEFINES}
	USE_HAL_DRIVER
    ${STM32_FAMILY}xx
    $<$<CONFIG:Debug>:DEBUG>
)
message(STATUS "STM32_HAL_DEFINES = ${STM32_HAL_DEFINES}")

set(STM32_HAL_SRC_DIR
    ${STM32_MCU_PKG_PATH}/Drivers/${STM32_HAL}/Src)
message(STATUS "STM32_HAL_SRC_DIR = ${STM32_HAL_SRC_DIR}")

file(GLOB STM32_HAL_SRCS "${STM32_HAL_SRC_DIR}/*")
# the STM32 MCU packages include source files in the HAL driver src folder that do not intend to actually be compiled with the HAL driver. Exclude them
list(FILTER STM32_HAL_SRCS EXCLUDE REGEX "template\\.c$")
list(APPEND STM32_HAL_SRCS ${STM32_CMSIS_DEVICE_SRC})

add_library(stm32_hal_driver STATIC
    ${STM32_HAL_SRCS}
    ${STM32_CMSIS_DEVICE_PATH}/Source/Templates/system_${STM32_SERIES_LOWER}xx.c)
set_property(TARGET stm32_hal_driver PROPERTY CMAKE_C_STANDARD 11)
set_property(TARGET stm32_hal_driver PROPERTY CMAKE_C_STANDARD_REQUIRED ON)
set_property(TARGET stm32_hal_driver PROPERTY CMAKE_C_EXTENSIONS ON)
target_include_directories(stm32_hal_driver 
    PUBLIC 
        ${STM32_HAL_INCLUDE_DIR} 
        ${STM32_CMSIS_INCLUDE_DIR} 
        ${STM32_BOARD_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/src/${HSB_EMULATOR_PLATFORM}/${HSB_EMULATOR_TARGET})
target_compile_definitions(stm32_hal_driver PUBLIC ${STM32_HAL_DEFINES})
add_library(embedded_hal_driver ALIAS stm32_hal_driver)

add_library(emulation_host STATIC
    ${STM32_STARTUP_FILE}
    hsb_emulator.cpp
    data_plane.cpp
    gpio.cpp
    i2c.cpp
    spi.cpp
    net.cpp
    tim.c
    stm32_system.c
    ${HSB_EMULATOR_TARGET}/board.c
)

target_include_directories(emulation_host
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${HSB_EMULATOR_TARGET}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation>
)

# When ON, the STM32 hardware CRC peripheral is used for the RoCEv2 iCRC and the
# zlib software fallback (and its include path) is dropped. When OFF, the build
# pulls in ZLIB_PATH so common/rocev2_transmitter.cpp can #include <zlib.h>.
option(CRC_OFFLOAD "Use the STM32 hardware CRC peripheral instead of zlib's software crc32" OFF)

# Publish platform-specific include paths and compile definitions for emulation_common
# to consume. processed in src/CMakeLists.txt: this subdir runs before common/.
# BUILD_INTERFACE only: STM32 toolchain headers and our build-tree paths must not
# leak into the exported install interface (no STM32 install target exists, and
# absolute build paths break relocatability).
#
# IMPORTANT: each include path is wrapped in its own $<BUILD_INTERFACE:...>. A
# single $<BUILD_INTERFACE:${LIST_VAR}> with a multi-element CMake list (e.g.
# STM32_CMSIS_INCLUDE_DIR contains both CMSIS/Include and CMSIS/Device/.../Include)
# triggers a CMake list-in-genex pitfall: the inner ';' is split during property
# evaluation, the leading '/' on items after the first is dropped by
# target_include_directories' relative-path handling, and the first item ends up
# concatenated under CMAKE_CURRENT_SOURCE_DIR of the declaring directory.
set(_hsb_platform_iface_includes
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/${HSB_EMULATOR_TARGET}
    ${STM32_HAL_INCLUDE_DIR}
    ${STM32_CMSIS_INCLUDE_DIR}
    ${STM32_BOARD_INCLUDE_DIR})
if(NOT CRC_OFFLOAD)
    list(APPEND _hsb_platform_iface_includes ${ZLIB_PATH})
endif()
foreach(_dir IN LISTS _hsb_platform_iface_includes)
    target_include_directories(hsb_platform INTERFACE
        $<BUILD_INTERFACE:${_dir}>)
endforeach()
unset(_hsb_platform_iface_includes)
target_compile_definitions(hsb_platform INTERFACE
    $<BUILD_INTERFACE:${STM32_HAL_DEFINES}>
    $<BUILD_INTERFACE:${HSB_HAL_DEFINES}>
    $<BUILD_INTERFACE:$<$<BOOL:${CRC_OFFLOAD}>:CRC_OFFLOAD>>
)

target_compile_definitions(emulation_host PUBLIC ${HSB_HAL_DEFINES})

target_link_libraries(emulation_host
    PUBLIC
        emulation_common
        stm32_hal_driver
        dlpack::dlpack
)

set_property(TARGET emulation_host PROPERTY POSITION_INDEPENDENT_CODE ON)

##### Transport sources — published to PARENT_SCOPE
#
# The actual emulationroce / emulationcoe libraries are assembled
# in ../CMakeLists.txt from these lists plus the matching common/ sources.

set(EMULATIONROCE_PLATFORM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/rocev2_data_plane.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocev2_transmitter.cpp
    PARENT_SCOPE)

set(EMULATIONCOE_PLATFORM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/coe_data_plane.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/coe_transmitter.cpp
    PARENT_SCOPE)

install(TARGETS emulation_host stm32_hal_driver
    EXPORT HololinkTargets
    COMPONENT hololink-emulation)
