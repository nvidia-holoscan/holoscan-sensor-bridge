# SPDX-License-Identifier: Apache-2.0
add_library(image_decoder STATIC
  image_decoder.cpp
)

set_property(TARGET image_decoder PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::image_decoder ALIAS image_decoder)

target_include_directories(image_decoder
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:src>
)

target_link_libraries(image_decoder
  PUBLIC
    hololink::native
    realsense2
  PRIVATE
    holoscan::core
)
