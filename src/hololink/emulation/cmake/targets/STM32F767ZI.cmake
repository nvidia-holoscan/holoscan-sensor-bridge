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

message(STATUS "configuring for stm32f767zi")

if(NOT DEFINED STM32_PATH OR STM32_PATH STREQUAL "")
  message(WARNING "STM32_PATH is not set - searching ~/STM32. If not found, set it to your STM32CubeF7 (or SDK) root for STM32F767ZI, e.g. -DSTM32_PATH=/path/to/STM32CubeF7")
  set(STM32_PATH "$ENV{HOME}/STM32")
endif()

# CMake does not expand ~ in file(GLOB)/find_path; use $ENV{HOME} and make absolute
if(STM32_PATH MATCHES "^~")
  string(REPLACE "~" "$ENV{HOME}" STM32_PATH "${STM32_PATH}")
endif()
get_filename_component(STM32_PATH "${STM32_PATH}" ABSOLUTE)

# set cmake configuration variables
set(HSB_EMULATOR_PLATFORM "STM32")
set(STM32_SERIES "STM32F7")
set(STM32_FAMILY "STM32F767")
set(STM32_MCU_PKG "STM32CubeF7")
set(STM32_MCU_PKG_PATH ${STM32_PATH})
set(STM32_HAL "${STM32_SERIES}xx_HAL_Driver")
set(MCU_TOOLCHAIN "gcc")
set(STM32_BOARD_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/src/stm32/${HSB_EMULATOR_TARGET})
set(STM32_HAL_DEFINES
  STM32
  ${STM32_SERIES}
  ${STM32_FAMILY}
  STM32F767ZI)

set(HSB_HAL_DEFINES
  "MAX_DATA_PLANES=1")

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -T \"${CMAKE_SOURCE_DIR}/src/${HSB_EMULATOR_PLATFORM}/${HSB_EMULATOR_TARGET}/${STM32_FAMILY}XX_FLASH.ld\"")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Map=${HSB_EMULATOR_TARGET}.map -Wl,--gc-sections")

# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# check dependencies
include("${HOLOLINK_REL_PATH}/cmake/hololink_deps/dlpack.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/deps/stm32mcu.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/deps/zlib_crc32.cmake")


set(STM32_STARTUP_FILE "${STM32_CMSIS_DEVICE_PATH}/Source/Templates/gcc/startup_stm32f767xx.s")
message(STATUS "startup file = ${STM32_STARTUP_FILE}")

list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)
