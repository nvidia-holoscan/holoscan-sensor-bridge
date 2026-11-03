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

# get the L4T_MAJ_VER from the sipl_api.cmake where it is exposed
include(sipl_api)
if(L4T_MAJ_VER LESS 39)
    message(FATAL_ERROR
        "ArgusIspOp is not compatible with L4T ${L4T_MAJ_VER}. Reconfigure with -DHOLOLINK_BUILD_ARGUS_ISP=OFF")
endif()

add_library(argus_isp STATIC
  argus_impl.cpp
  argus_isp.cpp
  camera_provider.cpp
  )

set_property(TARGET argus_isp PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::argus_isp ALIAS argus_isp)

add_dependencies(argus_isp capture_metadata_schema_generated)

target_include_directories(argus_isp
  PUBLIC
    $<BUILD_INTERFACE:${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}>
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_include_directories(argus_isp
  PRIVATE
    /usr/src/jetson_multimedia_api/argus/include
  )

# Define a variable to store the directory path
set(ARGUS_LIBRARY_DIR "/usr/lib/aarch64-linux-gnu/nvidia")

target_link_libraries(argus_isp
  PUBLIC
    # argus_isp.hpp includes the generated capture_metadata_traits.hpp, which uses
    # holoscan::schema types. capture_metadata_schema_generated alone may not carry
    # holoscan::schema's own usage requirements to an installed consumer (unlike
    # within this build tree); link it explicitly, matching base_receiver_op's
    # INTERFACE link of the same (src/hololink/operators/CMakeLists.txt).
    holoscan::schema
    # argus_isp.hpp's input/output ports are now schema::ImageT (the image-domain payload),
    # which pulls in holoscan/sensor_io/sensor_schema_package.hpp for schema::ImageT itself.
    # holoscan::schema alone does not carry sensor_io's own usage requirements to an installed
    # consumer; link it explicitly, same reasoning as holoscan::schema above.
    holoscan::sensor_io
  PRIVATE
    hololink
    holoscan::core
    holoscan::schema
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
set_target_properties(argus_isp PROPERTIES EXPORT_NAME operators::argus_isp)
install(TARGETS argus_isp
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/argus_isp.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/camera_provider.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/argus_impl.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/argus_isp
        COMPONENT hololink-operators)
