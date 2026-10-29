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

if(NOT DEFINED BUILD_DIR OR NOT DEFINED INSTALL_PREFIX)
  message(FATAL_ERROR "BUILD_DIR and INSTALL_PREFIX are required")
endif()

file(REMOVE_RECURSE "${INSTALL_PREFIX}")

set(_install_command
    "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${INSTALL_PREFIX}")
if(DEFINED BUILD_CONFIG AND NOT BUILD_CONFIG STREQUAL "")
  list(APPEND _install_command --config "${BUILD_CONFIG}")
endif()
if(DEFINED COMPONENT AND NOT COMPONENT STREQUAL "")
  list(APPEND _install_command --component "${COMPONENT}")
endif()

execute_process(
  COMMAND ${_install_command}
  RESULT_VARIABLE _result
  COMMAND_ECHO STDOUT
)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "Failed to stage the RoCE transport install tree")
endif()
