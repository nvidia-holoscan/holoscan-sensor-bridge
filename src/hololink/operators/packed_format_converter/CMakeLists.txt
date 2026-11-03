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

add_dependencies(packed_format_converter capture_metadata_schema_generated)

target_include_directories(packed_format_converter
  PUBLIC
    $<BUILD_INTERFACE:${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}>
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(packed_format_converter
  PUBLIC
    # packed_format_converter.hpp includes the generated capture_metadata_traits.hpp,
    # which uses holoscan::schema types. capture_metadata_schema_generated alone may
    # not carry holoscan::schema's own usage requirements to an installed consumer
    # (unlike within this build tree); link it explicitly, matching base_receiver_op's
    # INTERFACE link of the same (src/hololink/operators/CMakeLists.txt).
    holoscan::schema
    # packed_format_converter.hpp's output port is now schema::ImageT (the image-domain
    # payload), which pulls in holoscan/sensor_io/sensor_schema_package.hpp for schema::ImageT
    # itself. holoscan::schema alone does not carry sensor_io's own usage requirements to an
    # installed consumer; link it explicitly, same reasoning as holoscan::schema above.
    holoscan::sensor_io
  PRIVATE
    hololink
    holoscan::core
    holoscan::schema
  )

# Installation of the packed_format_converter operator
set_target_properties(packed_format_converter PROPERTIES EXPORT_NAME operators::packed_format_converter)
install(TARGETS packed_format_converter
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/packed_format_converter.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/packed_format_converter
        COMPONENT hololink-operators)
