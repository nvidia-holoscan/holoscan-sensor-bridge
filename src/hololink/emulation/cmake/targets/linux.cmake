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
#
# See README.md for detailed information.

set(HSB_EMULATOR_PLATFORM "linux")
set(HSB_HAL_DEFINES "")

# FrameMetadata transport policy:
#   0 = embed FrameMetadata in the final payload packet by padding with zeros
#       (the host RDMA receiver lands it at virtual_address + metadata_offset)
#   1 = send FrameMetadata as its own packet (no zero-padding the payload packet)
# Linux and STM32 default to 0. Override at configure time with
# -DSEPARATE_FRAMEMETADATA_PACKET=<0|1>. Note that the tests/test_emulator_serve_file_receiver_sequences will currently fail if set to 1 as it is specifically testing the -DSEPARATE_FRAMEMETADATA_PACKET=0 case
set(SEPARATE_FRAMEMETADATA_PACKET 0 CACHE STRING
    "0=embed FrameMetadata in last payload packet (zero-pad); 1=send FrameMetadata as a separate packet")
add_compile_definitions(SEPARATE_FRAMEMETADATA_PACKET=${SEPARATE_FRAMEMETADATA_PACKET})

enable_language(CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 80 86 87) # covers all of Orin
    endif()
endif()

# check dependencies
find_package(ZLIB)
find_package(CUDAToolkit)
include("${HOLOLINK_REL_PATH}/cmake/hololink_deps/dlpack.cmake")

##### Standalone python environment handling

if (HSB_EMULATOR_BUILD_PYTHON AND NOT HSB_EMULATOR_TOT_BUILD)
    # we only need to find CUDAToolkit above, but if Python bindings are required, it will be required
    find_package(CUDAToolkit REQUIRED)

    # The Python module is one nanobind extension, `_emulation`, built from the same source
    # and with the same nanobind as the full hololink build (python/hololink/emulation):
    # hololink_deps/nanobind.cmake takes an installed nanobind 3.0.1 (`pip install
    # nanobind==3.0.1` + -Dnanobind_DIR=$(python3 -m nanobind --cmake_dir)) and otherwise
    # fetches it. NB_STATIC/STABLE_ABI match the full build, so the module file is
    # _emulation.abi3.so and works with any Python >= 3.12.
    include("${HOLOLINK_REL_PATH}/cmake/hololink_deps/nanobind.cmake")
    nanobind_add_module(_emulation NB_STATIC STABLE_ABI
        python/hololink/emulation/_emulation.cpp)
    target_include_directories(_emulation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_link_libraries(_emulation
        PRIVATE
            hololink::emulation
            hololink::emulation::coe
            hololink::emulation::sensors
    )
    if (ZLIB_FOUND)
        target_link_libraries(_emulation PRIVATE hololink::emulation::roce)
    endif()

    add_custom_command(TARGET _emulation POST_BUILD
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build_pyvenv.sh $<TARGET_FILE:_emulation> "${CMAKE_CURRENT_SOURCE_DIR}/python" "${CMAKE_BINARY_DIR}/${HSB_EMULATOR_PYTHON_VENV}" ${CUDAToolkit_VERSION_MAJOR}
        COMMENT "Building standalone hololink python environment"
    )

endif()