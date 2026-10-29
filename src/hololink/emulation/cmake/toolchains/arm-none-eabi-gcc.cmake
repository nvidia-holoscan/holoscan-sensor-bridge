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

# requires arm-none-eabi-gcc to be in path and of version 11 or higher

set(CMAKE_SYSTEM_NAME               Generic)
set(CMAKE_SYSTEM_PROCESSOR          arm)

set(CMAKE_C_COMPILER_ID GNU)
set(CMAKE_CXX_COMPILER_ID GNU)

# Some default GCC settings
# arm-none-eabi- must be part of path environment
set(TOOLCHAIN_PREFIX                arm-none-eabi-)

set(CMAKE_C_COMPILER                ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_ASM_COMPILER              ${CMAKE_C_COMPILER})
set(CMAKE_CXX_COMPILER              ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_LINKER                    ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_OBJCOPY                   ${TOOLCHAIN_PREFIX}objcopy)
set(CMAKE_SIZE                      ${TOOLCHAIN_PREFIX}size)

set(CMAKE_EXECUTABLE_SUFFIX_ASM     ".elf")
set(CMAKE_EXECUTABLE_SUFFIX_C       ".elf")
set(CMAKE_EXECUTABLE_SUFFIX_CXX     ".elf")

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# MCU specific flags
set(TARGET_FLAGS "-mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard ")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${TARGET_FLAGS}")
set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -x assembler-with-cpp -MMD -MP")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fdata-sections -ffunction-sections")

set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_C_FLAGS_RELEASE "-Os -g0")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-Os -g0")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_C_FLAGS} -fno-rtti -fno-exceptions -fno-threadsafe-statics")

set(CMAKE_EXE_LINKER_FLAGS "${TARGET_FLAGS} --specs=nano.specs -Wl,--print-memory-usage -Wl,--gc-sections")

# WHOLE_ARCHIVE feature for the $<LINK_LIBRARY:...> generator expression.
#
# CMake only defines this feature after it identifies the linker, which happens
# during ABI detection's link step. CMAKE_TRY_COMPILE_TARGET_TYPE is
# STATIC_LIBRARY here (bare-metal has no linkable default), so nothing is ever
# linked, CMAKE_<LANG>_COMPILER_LINKER_ID comes back empty, and
# Platform/Linker/GNU.cmake is never included. Without it, src/CMakeLists.txt's
# "$<LINK_LIBRARY:WHOLE_ARCHIVE,emulation_host,emulation_common>" fails to
# configure twice over: the feature is reported as unsupported for the link
# language, and -- because the OVERRIDE=DEFAULT attribute is missing -- the
# plain occurrence of emulation_common that emulation_host propagates through
# its PUBLIC link interface is treated as an unresolvable feature conflict.
#
# Setting CMAKE_<LANG>_COMPILER_LINKER_ID here would not help: CMake resolves
# the feature through Platform/Linker/<system>-<linker>-<lang>.cmake, and it
# ships no such module for CMAKE_SYSTEM_NAME=Generic. So replicate what
# Platform/Linker/GNU.cmake would have set. Plain --whole-archive /
# --no-whole-archive rather than --push-state / --pop-state so this works with
# older arm-none-eabi binutils.
foreach(_hsb_lang IN ITEMS C CXX ASM)
    set(CMAKE_${_hsb_lang}_LINK_LIBRARY_USING_WHOLE_ARCHIVE
        "LINKER:--whole-archive"
        "<LINK_ITEM>"
        "LINKER:--no-whole-archive")
    set(CMAKE_${_hsb_lang}_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED TRUE)
    set(CMAKE_${_hsb_lang}_LINK_LIBRARY_WHOLE_ARCHIVE_ATTRIBUTES
        LIBRARY_TYPE=STATIC DEDUPLICATION=YES OVERRIDE=DEFAULT)
endforeach()
unset(_hsb_lang)
