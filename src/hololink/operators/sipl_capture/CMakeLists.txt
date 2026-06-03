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

include(sipl_api)

add_library(sipl_capture STATIC
    sipl_capture_service.cpp
    sipl_camera_output.cpp
)

set_property(TARGET sipl_capture PROPERTY POSITION_INDEPENDENT_CODE ON)

# add_compile_definitions() is directory-scoped and does not accept a target;
# use target_compile_definitions so NVSIPL_API_MAJOR_VERSION reaches this library
# and PUBLIC consumers of hololink::operators::sipl_capture.
if(NOT DEFINED NVSIPL_API_MAJOR_VERSION OR NVSIPL_API_MAJOR_VERSION STREQUAL "")
  message(FATAL_ERROR "NVSIPL_API_MAJOR_VERSION is not set after include(sipl_api)")
endif()
target_compile_definitions(sipl_capture PUBLIC
  "NVSIPL_API_MAJOR_VERSION=${NVSIPL_API_MAJOR_VERSION}"
)

# In SIPL v2 NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW10TP / _RAW12TP are guarded by
# `#if defined(NV_EMBEDDED_L4T)` in /usr/src/jetson_sipl_api/sipl/include/NvSIPLCapStructs.h;
# mirror the default from SIPL's own samples/camera/CMakeLists.txt so those
# enum values are visible for CoE. v1 handles RAW10TP unconditionally, so leave
# v1 builds untouched.
if(NVSIPL_API_MAJOR_VERSION GREATER_EQUAL 2)
  target_compile_definitions(sipl_capture PUBLIC "NV_EMBEDDED_L4T")
endif()

add_library(hololink::operators::sipl_capture ALIAS sipl_capture)

target_link_directories(sipl_capture
  PUBLIC
    /usr/lib/aarch64-linux-gnu/nvidia
)

target_link_libraries(sipl_capture
  PUBLIC
    hololink::core
    holoscan::core
  PRIVATE
    GXF::multimedia
    nvbufsurface
    nvfusacapinterface
    nvrm_surface
    nvscibuf
    nvscisync
    nvsipl
    nvsipl_query
)

target_include_directories(sipl_capture
  PUBLIC
    /usr/src/jetson_sipl_api/sipl/include
    /usr/src/jetson_sipl_api/sipl/include/nvsci
    /usr/src/jetson_sipl_api/sipl/include/query/include
)

# Installation of the sipl_capture operator
set_target_properties(sipl_capture PROPERTIES EXPORT_NAME operators::sipl_capture)
install(TARGETS sipl_capture
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/sipl_capture.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/sipl_capture_service.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/sipl_camera_output.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/sipl_compat.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/sipl_fmt.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators/sipl_capture
        COMPONENT hololink-operators)
