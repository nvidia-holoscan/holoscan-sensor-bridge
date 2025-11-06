# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

pybind11_add_hololink_module(
    CPP_CMAKE_TARGET hololink_core
    CLASS_NAME "ArpWrapper,Deserializer,Reactor,Serializer"
    SOURCES hololink_core.cpp
)

# Export symbols from _hololink_core.so so that _hololink.so can link against it
# By default, pybind11 modules hide symbols, but we need to export C++ symbols
# for use by other Python modules
set_property(TARGET hololink_core_python PROPERTY CXX_VISIBILITY_PRESET default)
set_property(TARGET hololink_core_python PROPERTY VISIBILITY_INLINES_HIDDEN OFF)

# Mark this as a shared library that can be linked against by other modules
target_link_options(hololink_core_python PRIVATE
    "LINKER:--export-dynamic"
)
