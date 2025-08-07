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

add_library(sig_gen STATIC
  sig_gen_op.cpp
  sig_gen_op.cu
  sig_gen_op_gui.cpp
)

set_property(TARGET sig_gen PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::sig_gen ALIAS sig_gen)

target_include_directories(sig_gen
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
  )

set_target_properties(sig_gen PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

find_package(CUDAToolkit)

target_compile_options(sig_gen PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                      --generate-line-info
                      --use_fast_math
                      --relocatable-device-code=true
                      >)

# If CUDA has multiple include dirs, use the first.
list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_TOOLKIT_INCLUDE_PATH)
target_compile_definitions(sig_gen PUBLIC CUDA_TOOLKIT_INCLUDE_PATH="${CUDA_TOOLKIT_INCLUDE_PATH}")

target_link_libraries(sig_gen
  PRIVATE
    hololink
    hololink::expr_eval
    holoscan::core
    holoscan::viz::imgui
)

# Installation of the sig_gen operator
install(TARGETS sig_gen
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sig_gen_op.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/sig_gen_op_gui.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/sig_gen
        COMPONENT hololink-operators)
