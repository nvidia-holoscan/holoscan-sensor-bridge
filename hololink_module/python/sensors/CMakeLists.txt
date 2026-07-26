# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

set(_sensors_out
    "${CMAKE_BINARY_DIR}/python/hololink_module/sensors")
file(MAKE_DIRECTORY "${_sensors_out}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
    "${_sensors_out}/__init__.py" COPYONLY)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/csi.py"
    "${_sensors_out}/csi.py" COPYONLY)

add_subdirectory(imx274)
add_subdirectory(ar0234)
add_subdirectory(deserializers)
add_subdirectory(serializers)
add_subdirectory(hawk)
