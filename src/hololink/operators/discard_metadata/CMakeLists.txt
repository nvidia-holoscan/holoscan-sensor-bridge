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

# we don't expose the C++ interface of the operator yet, therefore link
# statically
add_library(discard_metadata STATIC
  discard_metadata.cpp
  )

set_property(TARGET discard_metadata PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::discard_metadata ALIAS discard_metadata)

add_dependencies(discard_metadata capture_metadata_schema_generated)

target_include_directories(discard_metadata
  PUBLIC
    $<BUILD_INTERFACE:${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}>
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(discard_metadata
  PUBLIC
    holoscan::schema
  PRIVATE
    holoscan::core
  )

# Installation of the discard_metadata operator
set_target_properties(discard_metadata PROPERTIES EXPORT_NAME operators::discard_metadata)
install(TARGETS discard_metadata
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/discard_metadata.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/discard_metadata
        COMPONENT hololink-operators)
