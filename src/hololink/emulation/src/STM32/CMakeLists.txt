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
    apb_events.cpp
    stm32_system.c
)

target_include_directories(emulation_host
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${HSB_EMULATOR_TARGET}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation>
)

target_compile_definitions(emulation_host PUBLIC ${HSB_HAL_DEFINES})

target_link_libraries(emulation_host
    PUBLIC
        emulation_common
        stm32_hal_driver
        dlpack::dlpack
)

set_property(TARGET emulation_host PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(emulationroce STATIC
    rocev2_data_plane.cpp
    rocev2_transmitter.cpp
    ${ZLIB_PATH}/crc32.c
)

target_include_directories(emulationroce 
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation>
    PRIVATE
        ${ZLIB_PATH}
)
target_link_libraries(emulationroce 
    PUBLIC 
        emulation_host
        emulation_common
        stm32_hal_driver
        dlpack::dlpack
)
#target_compile_definitions(emulationroce PRIVATE "-Dlocal")
set_property(TARGET emulationroce PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::emulation::roce ALIAS emulationroce)