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

add_library(sub_frame_visualizer STATIC
    sub_frame_visualizer.cpp
)

set_property(TARGET sub_frame_visualizer PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::sub_frame_visualizer ALIAS sub_frame_visualizer)

add_dependencies(sub_frame_visualizer capture_metadata_schema_generated)

# The band shader is compiled to SPIR-V at build time with the SDK's HoloViz helper. It is
# installed beside holoscan-config.cmake and included by it when the visualization component is
# requested; fall back to the file itself when holoscan was found without components.
if(NOT COMMAND holoviz_generate_shader_header)
  include("${holoscan_DIR}/HolovizShaderHeader.cmake")
endif()
holoviz_generate_shader_header(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/sub_frame_visualizer.slang.hpp"
  SLANG shader.slang
  NAMESPACE hololink::operators::sub_frame_visualizer_shader
  PAIRS main=vertex_main:fragment_main
)
target_sources(sub_frame_visualizer PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated/sub_frame_visualizer.slang.hpp")

target_include_directories(sub_frame_visualizer
  PUBLIC
    $<BUILD_INTERFACE:${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}>
  PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/holoviz_generated"
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(sub_frame_visualizer
  PUBLIC
    # sub_frame_visualizer.hpp includes the generated capture_metadata_traits.hpp,
    # which uses holoscan::schema types. capture_metadata_schema_generated alone
    # may not carry holoscan::schema's own usage requirements to an installed
    # consumer (unlike within this build tree); link it explicitly, matching
    # base_receiver_op's INTERFACE link of the same
    # (src/hololink/operators/CMakeLists.txt).
    holoscan::schema
    # sub_frame_visualizer.hpp's input port is now schema::ImageT (the image-domain payload),
    # which pulls in holoscan/sensor_io/sensor_schema_package.hpp for schema::ImageT itself.
    # holoscan::schema alone does not carry sensor_io's own usage requirements to an installed
    # consumer; link it explicitly, same reasoning as holoscan::schema above.
    holoscan::sensor_io
    # HSDK5 HoloViz RHI (rhi::Device/Swapchain/Image/ExternalSemaphore), from the visualization
    # module; see the gate in ../CMakeLists.txt. Public: sub_frame_visualizer.hpp includes the
    # holoviz headers and holds holoviz objects by value.
    holoscan::visualization::holoviz
  PRIVATE
    hololink
    holoscan::core
    CUDA::cuda_driver
)

# Installation of the sub_frame_visualizer operator
set_target_properties(sub_frame_visualizer PROPERTIES EXPORT_NAME operators::sub_frame_visualizer)
install(TARGETS sub_frame_visualizer
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/sub_frame_visualizer.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/sub_frame_visualizer
        COMPONENT hololink-operators)
