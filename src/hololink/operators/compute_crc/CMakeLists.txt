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

find_package(nvcomp REQUIRED
    PATHS /usr/lib/sbsa-linux-gnu/cmake/nvcomp)

# we don't expose the C++ interface of the operator yet, therefore link
# statically
add_library(compute_crc STATIC
  compute_crc.cpp
  )

set_property(TARGET compute_crc PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::compute_crc ALIAS compute_crc)

target_include_directories(compute_crc
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
  PUBLIC
    ${nvcomp_INCLUDE_DIR}
  )

target_link_libraries(compute_crc
  PRIVATE
    holoscan::core
    hololink::core
    nvcomp::nvcomp
    nvcomp::nvcomp_cpu
  )

install(TARGETS compute_crc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)
