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

include_guard(GLOBAL)
include(CMakePackageConfigHelpers)

if(NOT TARGET hololink_transport_roce)
  message(FATAL_ERROR
    "HololinkRoce export requires the hololink_transport_roce target")
endif()

set(HOLOLINK_ROCE_INSTALL_CMAKEDIR
  "${CMAKE_INSTALL_LIBDIR}/cmake/HololinkRoce")

install(FILES "${CMAKE_CURRENT_LIST_DIR}/Findibverbs.cmake"
  DESTINATION "${HOLOLINK_ROCE_INSTALL_CMAKEDIR}"
  COMPONENT hololink-roce-transport
)

install(EXPORT HololinkRoceTransportTargets
  NAMESPACE hololink::
  FILE HololinkRoceTransportTargets.cmake
  DESTINATION "${HOLOLINK_ROCE_INSTALL_CMAKEDIR}"
  COMPONENT hololink-roce-transport
)

configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/HololinkRoceConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/HololinkRoceConfig.cmake"
  INSTALL_DESTINATION "${HOLOLINK_ROCE_INSTALL_CMAKEDIR}"
  NO_SET_AND_CHECK_MACRO
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/HololinkRoceConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/HololinkRoceConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/HololinkRoceConfigVersion.cmake"
  DESTINATION "${HOLOLINK_ROCE_INSTALL_CMAKEDIR}"
  COMPONENT hololink-roce-transport
)

# Named project binary dirs stay correct if this leaf later calls
# project() while still add_subdirectory()'d. PROJECT_BINARY_DIR here
# would follow that nested project; hololink_BINARY_DIR would not.
# The umbrella consumer test looks at ${hololink_BINARY_DIR}/HololinkRoce.
if(HOLOLINK_ROCE_TRANSPORT_STANDALONE)
  set(_hololink_roce_package_root "${CMAKE_BINARY_DIR}")
else()
  set(_hololink_roce_package_root "${hololink_BINARY_DIR}")
endif()
set(_hololink_roce_build_cmakedir
  "${_hololink_roce_package_root}/HololinkRoce")
file(MAKE_DIRECTORY "${_hololink_roce_build_cmakedir}")

export(EXPORT HololinkRoceTransportTargets
  NAMESPACE hololink::
  FILE "${_hololink_roce_build_cmakedir}/HololinkRoceTransportTargets.cmake"
)

configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/HololinkRoceConfig.cmake.in"
  "${_hololink_roce_build_cmakedir}/HololinkRoceConfig.cmake"
  INSTALL_DESTINATION "HololinkRoce"
  INSTALL_PREFIX "${_hololink_roce_package_root}"
  NO_SET_AND_CHECK_MACRO
)

write_basic_package_version_file(
  "${_hololink_roce_build_cmakedir}/HololinkRoceConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/Findibverbs.cmake"
  "${_hololink_roce_build_cmakedir}/Findibverbs.cmake"
  COPYONLY
)

unset(_hololink_roce_build_cmakedir)
unset(_hololink_roce_package_root)
