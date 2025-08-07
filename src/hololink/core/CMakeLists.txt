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

add_library(hololink_core STATIC
  arp_wrapper.cpp
  data_channel.cpp
  enumerator.cpp
  hololink.cpp
  jesd.cpp
  logging.cpp
  metadata.cpp
  networking.cpp
  timeout.cpp
  tools.cpp
  traditional_spi.cpp
  traditional_i2c.cpp
  )
add_library(hololink::core ALIAS hololink_core)
set_source_files_properties(enumerator.cpp PROPERTIES COMPILE_FLAGS "-g3")
set_source_files_properties(networking.cpp PROPERTIES COMPILE_FLAGS "-g3")

set_property(TARGET hololink_core PROPERTY POSITION_INDEPENDENT_CODE ON)

if (NOT HOLOLINK_BUILD_ONLY_NATIVE)
  # If we're building all of the Hololink components, including those that depend on
  # Holoscan, we link against holoscan::core. This is a workaround for the fact that
  # Holoscan provides its own copy of the fmtlib includes and we need to ensure that
  # this core library remains compatible with other Hololink libraries or applications
  # which use the fmtlib formatting functions within hololink_core.
  target_link_libraries(hololink_core
    PRIVATE
      holoscan::core
    )
else()
  target_link_libraries(hololink_core
    PRIVATE
      fmt
    )

  target_compile_definitions(hololink_core
    PRIVATE
      FMT_HEADER_ONLY=1
    )
endif()

target_include_directories(hololink_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:src>
  )

# Installation of the hololink_core library
install(TARGETS hololink_core
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-core)

install(FILES
        arp_wrapper.hpp
        csi_controller.hpp
        csi_formats.hpp
        data_channel.hpp
        deserializer.hpp
        enumerator.hpp
        hololink.hpp
        jesd.hpp
        logging.hpp
        # Note that logging_internal.hpp is not included in the public release.
        # logging_internal.hpp
        metadata.hpp
        networking.hpp
        nullable_pointer.hpp
        nvtx_trace.hpp
        serializer.hpp
        smart_object_pool.hpp
        timeout.hpp
        tools.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/core
        COMPONENT hololink-core)
