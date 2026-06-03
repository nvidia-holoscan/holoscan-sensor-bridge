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

cmake_minimum_required(VERSION 3.22)
include_guard(GLOBAL)
if (TARGET stm32mcu::hal)
    return()
endif()

# Validate that STM32CubeMX code is compatible with C standard
if((CMAKE_C_STANDARD EQUAL 90) OR (CMAKE_C_STANDARD EQUAL 99))
    message(FATAL_ERROR "Generated code requires C11 or higher")
endif()

# find MCU FW Package for the target STM32 Family
unset(STM32_MCU_PKG_PATH CACHE)
find_path(STM32_MCU_PKG_PATH
  NAMES ${STM32_MCU_PKG}
  PATHS ${STM32_PATH}
  NO_DEFAULT_PATH
  DOC "Path to STM32 MCU Package"
)

if (STM32_MCU_PKG_PATH)
    set(STM32_MCU_PKG_PATH "${STM32_MCU_PKG_PATH}/${STM32_MCU_PKG}")
    # globs need absolute path
    get_filename_component(STM32_MCU_PKG_PATH "${STM32_MCU_PKG_PATH}" ABSOLUTE)
    message(STATUS "Found ${STM32_MCU_PKG} at ${STM32_MCU_PKG_PATH}")
else()
    message(FATAL_ERROR "Failed to find required ${STM32_MCU_PKG} package in path ${STM32_PATH}")
endif()

# HAL PATHS
set(STM32_HAL_INCLUDE_DIR
    ${STM32_MCU_PKG_PATH}/Drivers/${STM32_HAL}/Inc)
message(STATUS "STM32_HAL_INCLUDE_DIR = ${STM32_HAL_INCLUDE_DIR}")
set(STM32_CMSIS_CORE_INCLUDE_DIR
    ${STM32_MCU_PKG_PATH}/Core/Include)
message(STATUS "STM32_CMSIS_CORE_INCLUDE_DIR = ${STM32_CMSIS_CORE_INCLUDE_DIR}")

file(GLOB _cmsis_device_paths "${STM32_MCU_PKG_PATH}/Drivers/CMSIS/Device/ST/*")
if(_cmsis_device_paths)
    list(GET _cmsis_device_paths 0 STM32_CMSIS_DEVICE_PATH)
endif()
message(STATUS "STM32_CMSIS_DEVICE_PATH = ${STM32_CMSIS_DEVICE_PATH}")

set(STM32_CMSIS_INCLUDE_DIR
    ${STM32_MCU_PKG_PATH}/Drivers/CMSIS/Include
    ${STM32_CMSIS_DEVICE_PATH}/Include)
message(STATUS "STM32_CMSIS_INCLUDE_DIR = ${STM32_CMSIS_INCLUDE_DIR}")

file(GLOB STM32_CMSIS_DEVICE_SRC "${STM32_CMSIS_DEVICE_PATH}/Source/Templates/*.c")
message(STATUS "device system source = ${STM32_CMSIS_DEVICE_SRC}")
