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

add_library(sub_frame_combiner STATIC
    sub_frame_combiner.cpp
)

set_property(TARGET sub_frame_combiner PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::sub_frame_combiner ALIAS sub_frame_combiner)

add_dependencies(sub_frame_combiner capture_metadata_schema_generated)

target_include_directories(sub_frame_combiner
  PUBLIC
    $<BUILD_INTERFACE:${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}>
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(sub_frame_combiner
  PUBLIC
    # sub_frame_combiner.hpp includes the generated capture_metadata_traits.hpp,
    # which uses holoscan::schema types. capture_metadata_schema_generated alone
    # may not carry holoscan::schema's own usage requirements to an installed
    # consumer (unlike within this build tree); link it explicitly, matching
    # base_receiver_op's INTERFACE link of the same
    # (src/hololink/operators/CMakeLists.txt).
    holoscan::schema
    # sub_frame_combiner.hpp's input/output ports are now schema::ImageT (the image-domain
    # payload), which pulls in holoscan/sensor_io/sensor_schema_package.hpp for schema::ImageT
    # itself. holoscan::schema alone does not carry sensor_io's own usage requirements to an
    # installed consumer; link it explicitly, same reasoning as holoscan::schema above.
    holoscan::sensor_io
  PRIVATE
    hololink
    holoscan::core
    holoscan::schema
    CUDA::cuda_driver
)

# Installation of the sub_frame_combiner operator
set_target_properties(sub_frame_combiner PROPERTIES EXPORT_NAME operators::sub_frame_combiner)
install(TARGETS sub_frame_combiner
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/sub_frame_combiner.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/sub_frame_combiner
        COMPONENT hololink-operators)
