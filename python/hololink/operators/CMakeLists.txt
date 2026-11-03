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

# Every hololink operator is registered in src/hololink/operators/operator_catalog.cpp
# and exposed here via this package's __init__.py generic registry auto-discovery --
# no per-operator Python or CMake code needed. pva_crc/ is the one exception: it houses a
# standalone native extension (not a registered Operator), only built when a precompiled PVA
# SDK is present -- see pva_crc/CMakeLists.txt's own comment.

configure_file(__init__.py "${HOLOLINK_PYTHON_MODULE_OUT_DIR}/operators/__init__.py" COPYONLY)

if(HOLOLINK_BUILD_PVA)
    add_subdirectory(pva_crc)
endif()
