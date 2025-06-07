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

# Enable testing for the current directory and below
include(CTest) # it calls 'enable_testing()' internally

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

# This function takes in a test name, test sources and test dependencies (link libraries)
# and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTest)
  cmake_parse_arguments("CMAKE_TEST"        # PREFIX
    ""                                      # OPTIONS
    "NAME"                                  # ONEVAL
    "SOURCES;LINK_LIBRARIES"                # MULTIVAL
    ${ARGN}
  )

  # The first parameter can implicitly be the NAME 
  if(NOT CMAKE_TEST_NAME)
    list(GET ARGN 0 CMAKE_TEST_NAME)
  endif()

  add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SOURCES})

  set(BIN_DIR ${hololink_BINARY_DIR})

  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${BIN_DIR}/gtests>"
  )

  target_link_libraries(${CMAKE_TEST_NAME}
    PRIVATE
      holoscan::core
      hololink
      GTest::gmock_main
      GTest::gtest_main
      ${CMAKE_TEST_LINK_LIBRARIES}
  )

  # Run the test from the main bin directory to access data as needed
  add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_property(TEST ${CMAKE_TEST_NAME} PROPERTY ENVIRONMENT
    "LD_LIBRARY_PATH=${BIN_DIR}/lib:$ENV{LD_LIBRARY_PATH}")

endfunction()

# ##################################################################################################
# * core tests ----------------------------------------------------------------------------------
ConfigureTest(
    iq_encoder_decoder_test
  SOURCES
    iq_encoder_decoder_test.cpp
  LINK_LIBRARIES
    hololink::operators::iq_dec
    hololink::operators::iq_enc
)

ConfigureTest(
    signal_generator_test
  SOURCES
    signal_generator_test.cpp
  LINK_LIBRARIES
    hololink::operators::sig_gen
)

ConfigureTest(
    roce_transmitter_test
  SOURCES
  roce_transmitter_test.cpp
  LINK_LIBRARIES
    hololink::operators::roce_transmitter
)
