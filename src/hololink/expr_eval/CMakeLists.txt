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

add_library(expr_eval STATIC
  expr_eval.cu
)

set_property(TARGET expr_eval PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::expr_eval ALIAS expr_eval)

target_include_directories(expr_eval
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
  )

set_target_properties(expr_eval PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(expr_eval PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                      --generate-line-info
                      --use_fast_math
                      --relocatable-device-code=true
                      >)

target_link_libraries(expr_eval
  PRIVATE
    hololink
    holoscan::core
)

# Installation of the expr_eval library
install(TARGETS expr_eval
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)
