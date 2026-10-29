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

# ===========================================================================
# hololink_pcie kernel module -- CMake wrapper around the kbuild Makefile
#
# kbuild writes its objects next to the sources it is pointed at, so rather
# than invoking `make -C src/ehsb/kmod` in place (which litters the source
# tree with .o/.ko/.mod files) the sources are staged into the build tree
# first and kbuild runs there. The staged layout reproduces this directory,
# because hololink_pcie_drv.c includes the ABI headers as "include/...":
#
#   ${CMAKE_BINARY_DIR}/ehsb/kmod/         Makefile + hololink_pcie_drv.c
#   ${CMAKE_BINARY_DIR}/ehsb/kmod/include/ hololink_pcie_uapi.h, hololink_pcie_regs.h,
#                                          hololink_pcie_bootp.h
#
# `make -C src/ehsb/kmod` from a shell still works for anyone who prefers it.
# ===========================================================================

option(HOLOLINK_BUILD_KERNEL_MODULE
    "Build the hololink_pcie kernel module via its kbuild Makefile" ON)

# ---------------------------------------------------------------------------
# The uapi header is the ioctl ABI. Userspace carries a copy (the installed
# hololink_module header) so that it need not reach into the kernel tree; it
# must be byte-identical to this one or the ioctl numbers / struct sizes
# silently disagree at runtime.
# Checked at configure time regardless of whether the module itself is built.
# ---------------------------------------------------------------------------
function(_hololink_kmod_check_mirror _canonical)
    file(READ "${_canonical}" _canonical_text)
    foreach(_mirror IN LISTS ARGN)
        if(EXISTS "${_mirror}")
            file(READ "${_mirror}" _mirror_text)
            if(NOT _canonical_text STREQUAL _mirror_text)
                message(FATAL_ERROR
                    "hololink_pcie header mirror out of date: ${_mirror} differs from ${_canonical}. "
                    "Copy the kmod header over it; the kernel/userspace contract must be identical on both sides.")
            endif()
        endif()
    endforeach()
endfunction()
_hololink_kmod_check_mirror("${CMAKE_CURRENT_SOURCE_DIR}/include/hololink_pcie_uapi.h"
    "${CMAKE_SOURCE_DIR}/hololink_module/module/hsb_pcie/hololink_pcie_uapi.h")
# The bootp announcement layout: the userspace copy pins it to the host parser in a test.
_hololink_kmod_check_mirror("${CMAKE_CURRENT_SOURCE_DIR}/include/hololink_pcie_bootp.h"
    "${CMAKE_SOURCE_DIR}/hololink_module/module/hsb_pcie/hololink_pcie_bootp.h")

execute_process(COMMAND uname -r
    OUTPUT_VARIABLE _hololink_uname_r
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(KERNEL_DIR "/lib/modules/${_hololink_uname_r}/build"
    CACHE PATH "kernel build tree used to compile the hololink_pcie module")

set(_build_kmod TRUE)
if(NOT HOLOLINK_BUILD_KERNEL_MODULE)
    message(STATUS "hololink_pcie kernel module: disabled (HOLOLINK_BUILD_KERNEL_MODULE=OFF)")
    set(_build_kmod FALSE)
elseif(NOT EXISTS "${KERNEL_DIR}")
    message(WARNING
        "hololink_pcie kernel module: skipped -- kernel build tree '${KERNEL_DIR}' not found. "
        "Install the matching linux-headers package or pass -DKERNEL_DIR=<path>, "
        "or silence this with -DHOLOLINK_BUILD_KERNEL_MODULE=OFF.")
    set(_build_kmod FALSE)
endif()

if(_build_kmod)
    # kbuild is GNU-make-only; drive it with make regardless of this project's
    # generator (Ninja included) rather than with CMAKE_MAKE_PROGRAM.
    find_program(HOLOLINK_KMOD_MAKE_PROGRAM NAMES gmake make REQUIRED)

    set(_kmod_stage "${CMAKE_BINARY_DIR}/ehsb/kmod")
    set(_kmod_include_dir "${_kmod_stage}/include")
    set(_kmod_ko "${_kmod_stage}/hololink_pcie_drv.ko")

    # Created at configure time as well as in the custom command: the command's
    # WORKING_DIRECTORY has to already exist when the build tool spawns it.
    file(MAKE_DIRECTORY "${_kmod_stage}" "${_kmod_include_dir}")

    set(_kmod_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/Makefile
        ${CMAKE_CURRENT_SOURCE_DIR}/hololink_pcie_drv.c
    )
    set(_kmod_uapi_headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/hololink_pcie_uapi.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/hololink_pcie_regs.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/hololink_pcie_bootp.h
    )

    add_custom_command(
        OUTPUT ${_kmod_ko}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_kmod_stage} ${_kmod_include_dir}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_kmod_sources} ${_kmod_stage}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_kmod_uapi_headers} ${_kmod_include_dir}
        COMMAND ${HOLOLINK_KMOD_MAKE_PROGRAM} -C ${_kmod_stage} KDIR=${KERNEL_DIR}
                HL_PCIE_DRV_VERSION=${HOLOLINK_VERSION_NUM}
        DEPENDS ${_kmod_sources} ${_kmod_uapi_headers}
        WORKING_DIRECTORY ${_kmod_stage}
        COMMENT "Building hololink_pcie_drv.ko against ${KERNEL_DIR}"
        VERBATIM)

    add_custom_target(hololink_pcie_drv ALL DEPENDS ${_kmod_ko})

    # Not installed into /lib/modules -- loading is an explicit, privileged
    # step (see README.md). Kept apart from lib/hololink/modules, which holds
    # the dlopen'd hololink_module .so files.
    install(FILES ${_kmod_ko}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/hololink/kmod
        COMPONENT hololink-kernel-module)
endif()
