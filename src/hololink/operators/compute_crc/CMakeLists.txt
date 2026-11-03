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

add_dependencies(compute_crc capture_metadata_schema_generated)

target_include_directories(compute_crc
  PUBLIC
    $<BUILD_INTERFACE:${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}>
    ${nvcomp_INCLUDE_DIR}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
  )

target_link_libraries(compute_crc
  PUBLIC
    # compute_crc.hpp includes the generated capture_metadata_traits.hpp, which
    # uses holoscan::schema types. capture_metadata_schema_generated alone may not
    # carry holoscan::schema's own usage requirements to an installed consumer
    # (unlike within this build tree); link it explicitly, matching
    # base_receiver_op's INTERFACE link of the same
    # (src/hololink/operators/CMakeLists.txt).
    holoscan::schema
    # compute_crc.hpp also instantiates ComputeCrcOp for holoscan::schema::ImageT
    # (the image-domain payload alongside plain holoscan::Tensor), which pulls in
    # holoscan/sensor_io/sensor_schema_package.hpp for schema::ImageT itself. holoscan::schema
    # alone does not carry sensor_io's own usage requirements to an installed consumer; link it
    # explicitly, same reasoning as holoscan::schema above.
    holoscan::sensor_io
  PRIVATE
    holoscan::core
    holoscan::schema
    hololink::core
    nvcomp::nvcomp
    nvcomp::nvcomp_cpu
  )

set_target_properties(compute_crc PROPERTIES EXPORT_NAME operators::compute_crc)
install(TARGETS compute_crc
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)
