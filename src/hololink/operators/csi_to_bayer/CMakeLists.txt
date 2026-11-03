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
add_library(csi_to_bayer STATIC
  csi_to_bayer.cpp
  )

set_property(TARGET csi_to_bayer PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::csi_to_bayer ALIAS csi_to_bayer)

add_dependencies(csi_to_bayer capture_metadata_schema_generated)

target_include_directories(csi_to_bayer
  PUBLIC
    $<BUILD_INTERFACE:${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}>
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(csi_to_bayer
  PUBLIC
    # csi_to_bayer.hpp includes the generated capture_metadata_traits.hpp, which
    # uses holoscan::schema types. capture_metadata_schema_generated alone may not
    # carry holoscan::schema's own usage requirements to an installed consumer
    # (unlike within this build tree); link it explicitly, matching
    # base_receiver_op's INTERFACE link of the same
    # (src/hololink/operators/CMakeLists.txt).
    holoscan::schema
    # csi_to_bayer.hpp's output port is now schema::ImageT (the image-domain payload), which
    # pulls in holoscan/sensor_io/sensor_schema_package.hpp for schema::ImageT itself.
    # holoscan::schema alone does not carry sensor_io's own usage requirements to an installed
    # consumer; link it explicitly, same reasoning as holoscan::schema above.
    holoscan::sensor_io
  PRIVATE
    hololink
    holoscan::core
  )

# Installation of the csi_to_bayer operator
set_target_properties(csi_to_bayer PROPERTIES EXPORT_NAME operators::csi_to_bayer)
install(TARGETS csi_to_bayer
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/csi_to_bayer.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/sub_frame_bands.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/csi_to_bayer
        COMPONENT hololink-operators)
