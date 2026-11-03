# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# PVA CRC native extension package. The C++ backend (cpp/) is built here when
# HOLOLINK_BUILD_PVA=ON (PVA SDK present at build time); only this __init__.py copy runs
# otherwise, matching every other hololink package directory's own COPYONLY pattern.

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    ${HOLOLINK_PYTHON_MODULE_OUT_DIR}/operators/pva_crc/__init__.py
    COPYONLY
)

# HOLOLINK_BUILD_PVA is checked by our caller (python/hololink/operators/CMakeLists.txt), which
# only add_subdirectory()s this directory at all when it's on -- mirrors
# src/hololink/operators/CMakeLists.txt's own if(HOLOLINK_BUILD_PVA) gate for the header-only
# target. Build and place the pva_crc C++ extension only on aarch64 (Tegra) in addition:
# NvSciBuf is Tegra-specific; on x86_64 the extension is skipped so the rest of the package
# still builds.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    add_subdirectory(cpp)
    set(PVA_CRC_CPP_DEST "${HOLOLINK_PYTHON_MODULE_OUT_DIR}/operators/pva_crc/cpp")
    add_custom_command(
        OUTPUT ${PVA_CRC_CPP_DEST}/.pva_crc_copied
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PVA_CRC_CPP_DEST}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pva_crc> ${PVA_CRC_CPP_DEST}/
        COMMAND ${CMAKE_COMMAND} -E touch ${PVA_CRC_CPP_DEST}/.pva_crc_copied
        DEPENDS pva_crc
    )
    add_custom_target(copy_pva_crc_module ALL DEPENDS ${PVA_CRC_CPP_DEST}/.pva_crc_copied)
    add_dependencies(hololink_python_all copy_pva_crc_module)
endif()
