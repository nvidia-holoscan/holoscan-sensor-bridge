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

# we don't expose the C++ interface of the operator yet, therefore link
# statically
add_library(packed_format_converter STATIC
  packed_format_converter.cpp
  )

set_property(TARGET packed_format_converter PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::packed_format_converter ALIAS packed_format_converter)

target_include_directories(packed_format_converter
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
  )

target_link_libraries(packed_format_converter
  PRIVATE
    hololink
    holoscan::core
  )

# Installation of the packed_format_converter operator
install(TARGETS packed_format_converter
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packed_format_converter.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/packed_format_converter
        COMPONENT hololink-operators)
