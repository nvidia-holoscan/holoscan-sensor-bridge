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
    hololink::core
    # realsense2
    hololink
  PRIVATE
    holoscan::core
)

# Installation of the image_decoder operator
install(TARGETS image_decoder
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/image_decoder.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/operators/image_decoder
        COMPONENT hololink-operators)