# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# HSDK5 rewrite: a single nanobind module (_hololink.cpp) replacing the ~21 separate
# per-operator pybind11 modules the HSDK4-era tree built, plus the old top-level
# hololink.cpp/hololink_core/ pybind11 modules (all deleted). Most operators need zero
# operator-specific Python code at all: they're registered in
# src/hololink/operators/operator_catalog.cpp, and hololink.operators's generic
# auto-discovery (operators/__init__.py) exposes every registered row.
#
# Parts of sensors/ (camera/, imx715/, ox08d10/, the pybind11 half of vb1940/, and
# sensors/CMakeLists.txt's own pybind11_add_hololink_module(... sensors.cpp) call) are NOT built
# here: those use the legacy pybind11_add_hololink_module() macro, which links every module
# against holoscan::pybind11 -- a target the installed HSDK5 holoscan-sdk no longer exports
# (HSDK5 dropped pybind11 for nanobind, which is what this rewrite itself is switching to).
# They're broken independently of this change; porting them is tracked as follow-up work.
# sensors/ecam0m30tof *is* built (see sensors/CMakeLists.txt): it's pure Python, with no
# pybind11 dependency of its own. emulation/ is ported (one nanobind module, see
# emulation/CMakeLists.txt) and built when HOLOLINK_BUILD_EMULATOR is ON.

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include(hololink_deps/nanobind)

configure_file(__init__.py "${HOLOLINK_PYTHON_MODULE_OUT_DIR}/__init__.py" COPYONLY)
configure_file(renesas_bajoran_lite_ts1.py "${HOLOLINK_PYTHON_MODULE_OUT_DIR}/renesas_bajoran_lite_ts1.py" COPYONLY)

add_subdirectory(operators)
add_subdirectory(sensors)
if(HOLOLINK_BUILD_EMULATOR)
  add_subdirectory(emulation)
else()
  # A build tree configured with the option ON earlier still holds the staged module, and the
  # install(DIRECTORY) of ${HOLOLINK_PYTHON_MODULE_OUT_DIR} below would package it.
  file(REMOVE_RECURSE "${HOLOLINK_PYTHON_MODULE_OUT_DIR}/emulation")
endif()

nanobind_add_module(hololink_python NB_STATIC STABLE_ABI _hololink.cpp)

target_link_libraries(hololink_python
  PRIVATE
    holoscan::core
    hololink::hsdk5_compat
    hololink
    hololink::core
    hololink::operators::operator_catalog
    # the raw receivers, bound for hosts that receive without an operator graph
    hololink::operators::linux_receiver
    hololink::operators::linux_coe_receiver
    hololink::sensors::native_vb1940_camera_sensor
    hololink::sensors::native_imx274_camera_sensor
)

if(HOLOLINK_BUILD_FUSA)
  target_link_libraries(hololink_python PRIVATE hololink::operators::fusa_coe_capture)
endif()

set_target_properties(hololink_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${HOLOLINK_PYTHON_MODULE_OUT_DIR}"
    OUTPUT_NAME _hololink
)

add_dependencies(hololink_python_all hololink_python)

# install compiled HoloLink Python module
install(
    DIRECTORY "${HOLOLINK_PYTHON_MODULE_OUT_DIR}/"
    DESTINATION "${HOLOLINK_PYTHON_INSTALL_DIR}/hololink"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT "hololink-python_libs"
    PATTERN "__pycache__" EXCLUDE
)
