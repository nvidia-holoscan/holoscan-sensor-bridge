# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Nested CMake consumer tests for HololinkRoce. These are not gtests; they
# reconfigure a tiny consumer against the build-tree, standalone leaf, and
# installed package.

# Relay this configure's generator/toolchain/compiler selection into the
# nested consumer configures below so they build with the same
# generator/toolset/toolchain/compiler as this outer build, rather than
# silently falling back to the host default. Each is only added when the
# outer configure actually set it.
set(_roce_consumer_toolchain_args)
if(DEFINED CMAKE_GENERATOR AND NOT CMAKE_GENERATOR STREQUAL "")
  list(APPEND _roce_consumer_toolchain_args
    "-DPARENT_GENERATOR=${CMAKE_GENERATOR}")
endif()
if(DEFINED CMAKE_GENERATOR_PLATFORM AND NOT CMAKE_GENERATOR_PLATFORM STREQUAL "")
  list(APPEND _roce_consumer_toolchain_args
    "-DPARENT_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}")
endif()
if(DEFINED CMAKE_GENERATOR_TOOLSET AND NOT CMAKE_GENERATOR_TOOLSET STREQUAL "")
  list(APPEND _roce_consumer_toolchain_args
    "-DPARENT_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}")
endif()
if(DEFINED CMAKE_TOOLCHAIN_FILE AND NOT CMAKE_TOOLCHAIN_FILE STREQUAL "")
  list(APPEND _roce_consumer_toolchain_args
    "-DPARENT_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
endif()
if(DEFINED CMAKE_C_COMPILER AND NOT CMAKE_C_COMPILER STREQUAL "")
  list(APPEND _roce_consumer_toolchain_args
    "-DPARENT_C_COMPILER=${CMAKE_C_COMPILER}")
endif()
if(DEFINED CMAKE_CXX_COMPILER AND NOT CMAKE_CXX_COMPILER STREQUAL "")
  list(APPEND _roce_consumer_toolchain_args
    "-DPARENT_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
endif()

add_test(
  NAME hololink_roce_transport_build_tree_consumer_test
  COMMAND "${CMAKE_COMMAND}"
          "-DBINARY_DIR=${hololink_BINARY_DIR}/roce-transport-build-tree-consumer-build"
          "-DBUILD_CONFIG=$<CONFIG>"
          "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/transport_consumer"
          "-DHololinkRoce_DIR=${hololink_BINARY_DIR}/HololinkRoce"
          ${_roce_consumer_toolchain_args}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/ConfigureAndBuildConsumer.cmake")

add_test(
  NAME hololink_roce_transport_standalone_consumer_test
  COMMAND "${CMAKE_COMMAND}"
          "-DBUILD_CONFIG=$<CONFIG>"
          "-DCONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/transport_consumer"
          "-DPRODUCER_BINARY_DIR=${hololink_BINARY_DIR}/roce-standalone-build"
          "-DPRODUCER_SOURCE_DIR=${hololink_SOURCE_DIR}/src/hololink/transport/roce"
          "-DWORK_DIR=${hololink_BINARY_DIR}/roce-standalone-consumer"
          ${_roce_consumer_toolchain_args}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/ConfigureStandaloneRoce.cmake")

set(_install_prefix
    "${hololink_BINARY_DIR}/roce-transport-consumer-install-prefix")
add_test(
  NAME hololink_roce_transport_installed_consumer_setup
  COMMAND "${CMAKE_COMMAND}"
          "-DBUILD_CONFIG=$<CONFIG>"
          "-DBUILD_DIR=${hololink_BINARY_DIR}"
          -DCOMPONENT=hololink-roce-transport
          "-DINSTALL_PREFIX=${_install_prefix}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/InstallConsumerFixture.cmake")
set_tests_properties(
  hololink_roce_transport_installed_consumer_setup
  PROPERTIES FIXTURES_SETUP hololink_roce_transport_installed_consumer)

add_test(
  NAME hololink_roce_transport_installed_consumer_test
  COMMAND "${CMAKE_COMMAND}"
          "-DBINARY_DIR=${hololink_BINARY_DIR}/roce-transport-consumer-build"
          "-DBUILD_CONFIG=$<CONFIG>"
          "-DINSTALL_PREFIX=${_install_prefix}"
          "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/transport_consumer"
          ${_roce_consumer_toolchain_args}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/ConfigureAndBuildConsumer.cmake")
set_tests_properties(
  hololink_roce_transport_installed_consumer_test
  PROPERTIES FIXTURES_REQUIRED hololink_roce_transport_installed_consumer)
