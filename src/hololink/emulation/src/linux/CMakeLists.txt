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

add_library(emulation_host STATIC
    data_plane.cpp
    hsb_emulator.cpp
    net.cpp
    utils.cpp)

target_include_directories(emulation_host
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/hololink/emulation>
)

# Publish the linux/ source dir so emulation_common can see linux-only headers
# (e.g. linux/data_plane_ctxt.hpp) if it ever needs to. BUILD_INTERFACE only:
# these headers aren't installed and consumers of the exported targets must not
# pick up build-tree paths post-install.
target_include_directories(hsb_platform INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(emulation_host
    PUBLIC
        emulation_common
        dlpack::dlpack
        # DataPlane::send(DLTensor) cudaMemcpy's GPU-resident inputs into a host bounce
        # buffer (LinuxDataPlaneCtxt::double_buffer) before forwarding to the transmitter.
        CUDA::cudart
)

set_property(TARGET emulation_host PROPERTY POSITION_INDEPENDENT_CODE ON)

##### Transport sources — published to PARENT_SCOPE
#
# The actual emulationroce / emulationcoe libraries are assembled in
# ../CMakeLists.txt from these lists plus the matching common/ sources.

set(EMULATIONROCE_PLATFORM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/rocev2_data_plane.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocev2_transmitter.cpp
    PARENT_SCOPE)

set(EMULATIONCOE_PLATFORM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/coe_data_plane.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/coe_transmitter.cpp
    PARENT_SCOPE)

install(TARGETS emulation_host
    EXPORT HololinkTargets
    COMPONENT hololink-emulation)
