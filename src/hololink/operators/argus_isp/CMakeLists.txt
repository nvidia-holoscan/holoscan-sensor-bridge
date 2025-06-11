# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# we don't expose the C++ interface of the operator yet, therefore link
# statically

add_library(argus_isp STATIC
  argus_impl.cpp
  argus_isp.cpp
  camera_provider.cpp
  )

set_property(TARGET argus_isp PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::argus_isp ALIAS argus_isp)

target_include_directories(argus_isp
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
  )

target_include_directories(argus_isp
  PRIVATE
    /usr/src/jetson_multimedia_api/argus/include
  )

# Define a variable to store the directory path
set(ARGUS_LIBRARY_DIR "/usr/lib/aarch64-linux-gnu/nvidia")

target_link_libraries(argus_isp
  PRIVATE
    hololink
    holoscan::core
    CUDA::nvrtc
    CUDA::nppicc
    ${ARGUS_LIBRARY_DIR}/libnvargus.so
    ${ARGUS_LIBRARY_DIR}/libnvargus_socketclient.so
    ${ARGUS_LIBRARY_DIR}/libnvargus_socketserver.so
    ${ARGUS_LIBRARY_DIR}/libnvfusacap.so
    ${ARGUS_LIBRARY_DIR}/libnvscf.so
    ${ARGUS_LIBRARY_DIR}/libnvodm_imager.so
  )

# Installation of the argus_isp operator
install(TARGETS argus_isp
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/argus_isp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/camera_provider.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/argus_impl.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/argus_isp
        COMPONENT hololink-operators)
