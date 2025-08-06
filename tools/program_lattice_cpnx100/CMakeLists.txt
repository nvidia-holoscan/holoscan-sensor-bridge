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

# Find required packages
find_package(yaml-cpp REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

# Create executable
add_executable(program_lattice_cpnx100
    main.cpp
    winbond_w25q128jw.cpp
    clnx_flash.cpp
    cpnx_flash.cpp
    programmer.cpp
)

# Set C++17 standard
set_target_properties(program_lattice_cpnx100 PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Set compile definitions
target_compile_definitions(program_lattice_cpnx100 PRIVATE
    FMT_HEADER_ONLY=1
)

# Link libraries
if (NOT HOLOLINK_BUILD_ONLY_NATIVE)
    # When building with Holoscan, fmt is provided by holoscan::core
    target_link_libraries(program_lattice_cpnx100
        PRIVATE
            hololink::core
            yaml-cpp
            CURL::libcurl
            OpenSSL::SSL
            OpenSSL::Crypto
    )
else()
    # When building only native components, we need to link fmt directly
    find_package(fmt REQUIRED)
    target_link_libraries(program_lattice_cpnx100
        PRIVATE
            hololink::core
            yaml-cpp
            CURL::libcurl
            OpenSSL::SSL
            OpenSSL::Crypto
            fmt
    )
endif()

# Include directories
target_include_directories(program_lattice_cpnx100
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Install executable
install(TARGETS program_lattice_cpnx100
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT hololink-tools
) 
