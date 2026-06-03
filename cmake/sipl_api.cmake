# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include_guard(GLOBAL)

# Detect SIPL API version from the L4T major version.
# Check L4T_MAJ_VER env var first (set in Docker builds); fall back to
# parsing /sys/class/dmi/id/bios_version on the host.
if(DEFINED ENV{L4T_MAJ_VER})
    set(_l4t_major_raw "$ENV{L4T_MAJ_VER}")
    message(STATUS "L4T major version (from environment): ${_l4t_major_raw}")
else()
    file(READ "/sys/class/dmi/id/bios_version" _bios_version)
    string(REGEX MATCH "^([0-9]+)\\." _match "${_bios_version}")
    if(NOT _match)
        message(FATAL_ERROR "Could not determine L4T major version: L4T_MAJ_VER is not set and /sys/class/dmi/id/bios_version could not be parsed (${_bios_version})")
    endif()
    set(_l4t_major_raw "${CMAKE_MATCH_1}")
    message(STATUS "L4T major version (from bios_version): ${_l4t_major_raw}")
endif()
if(NOT _l4t_major_raw MATCHES "^[0-9]+$")
    message(FATAL_ERROR "L4T major version must be numeric, got '${_l4t_major_raw}'")
endif()
math(EXPR L4T_MAJ_VER "${_l4t_major_raw}")
if(L4T_MAJ_VER EQUAL 38)
    set(_nvsipl_api_major 1)
elseif(L4T_MAJ_VER EQUAL 39)
    set(_nvsipl_api_major 2)
else()
    message(FATAL_ERROR "Unsupported L4T major version '${L4T_MAJ_VER}' for SIPL API detection (expected 38 or 39)")
endif()
# Cache so the value is visible in every subdirectory scope. include_guard(GLOBAL)
# above means the file body only runs on the first include; a plain set() would
# only populate the first scope that includes this file, leaving sibling
# subdirectories with the variable undefined.
set(L4T_MAJ_VER "${L4T_MAJ_VER}" CACHE INTERNAL
    "Detected L4T major version (e.g. 38, 39); compare with EQUAL/GREATER/LESS")
set(NVSIPL_API_MAJOR_VERSION "${_nvsipl_api_major}" CACHE INTERNAL
    "Detected SIPL API major version (1 for L4T 38, 2 for L4T 39)")
message(STATUS "L4T major version: ${L4T_MAJ_VER}, SIPL API major version: ${NVSIPL_API_MAJOR_VERSION}")
