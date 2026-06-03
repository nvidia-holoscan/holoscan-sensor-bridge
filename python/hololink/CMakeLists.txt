# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(pybind11_add_hololink_module)

copy_hololink_python_files(
    TARGET_NAME copy_hololink_python_files
    FILES
        __init__.py
        hsb_controller.py
        renesas_bajoran_lite_ts1.py
        renesas_bajoran_lite_ts2.py
)

add_subdirectory(hololink_core)
add_subdirectory(operators)
add_subdirectory(sensors)
add_subdirectory(emulation)

pybind11_add_hololink_module(
    CPP_CMAKE_TARGET hololink
    CLASS_NAME "Timeout"
    SOURCES hololink.cpp
)

# Note: Link against _hololink_core.so is handled automatically by pybind11_add_hololink_module
# since the hololink target depends on hololink_core

# install compiled HoloLink Python module
install(
    DIRECTORY "${HOLOLINK_PYTHON_MODULE_OUT_DIR}/"
    DESTINATION "${HOLOLINK_PYTHON_INSTALL_DIR}/hololink"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT "hololink-python_libs"
    PATTERN "__pycache__" EXCLUDE
)
