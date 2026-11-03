# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# hololink.emulation: the HSB emulator's Python module, built from the sources that live
# with the emulator (src/hololink/emulation/python/hololink/emulation) so that the
# Emulator-only build (src/hololink/emulation/cmake/targets/linux.cmake) ships the very
# same package. One nanobind extension, `_emulation`, carries the core classes and the
# `sensors` submodule; the two __init__.py files are copied next to it.

set(EMULATION_PYTHON_SOURCE_DIR
    ${hololink_SOURCE_DIR}/src/hololink/emulation/python/hololink/emulation)

configure_file(${EMULATION_PYTHON_SOURCE_DIR}/__init__.py
    "${HOLOLINK_PYTHON_MODULE_OUT_DIR}/emulation/__init__.py" COPYONLY)
configure_file(${EMULATION_PYTHON_SOURCE_DIR}/sensors/__init__.py
    "${HOLOLINK_PYTHON_MODULE_OUT_DIR}/emulation/sensors/__init__.py" COPYONLY)

include(hololink_deps/nanobind)
nanobind_add_module(emulation_python NB_STATIC STABLE_ABI
    ${EMULATION_PYTHON_SOURCE_DIR}/_emulation.cpp)
target_link_libraries(emulation_python
    PRIVATE
        hololink::emulation
        hololink::emulation::coe
        hololink::emulation::sensors
)
# The RoCEv2 data plane needs zlib (CRC); the emulator tree only defines the target when
# zlib was found.
if(TARGET hololink::emulation::roce)
    target_link_libraries(emulation_python PRIVATE hololink::emulation::roce)
endif()
# The emulator libraries are static, so the module has no hololink shared-library
# dependencies and needs no rpath (same as _hololink).
set_target_properties(emulation_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${HOLOLINK_PYTHON_MODULE_OUT_DIR}/emulation"
    OUTPUT_NAME _emulation
)
add_dependencies(hololink_python_all emulation_python)
