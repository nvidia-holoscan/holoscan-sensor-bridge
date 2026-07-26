# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Pure-Python pieces of the hololink_module tree. The pybind extension
# itself is built under host/CMakeLists.txt; this subtree stages the .py
# files that live next to it so PYTHONPATH=${BUILD}/python python3 -c
# "import hololink_module.sensors.imx274" just works.
if(HOLOLINK_BUILD_PYTHON)
    add_subdirectory(sensors)
endif()
