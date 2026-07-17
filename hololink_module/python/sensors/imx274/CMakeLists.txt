# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

set(_imx274_out
    "${CMAKE_BINARY_DIR}/python/hololink_module/sensors/imx274")
file(MAKE_DIRECTORY "${_imx274_out}")

foreach(_pyfile
    __init__.py
    imx274_cam.py
    imx274_mode.py
    li_i2c_expander.py)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/${_pyfile}"
        "${_imx274_out}/${_pyfile}" COPYONLY)
endforeach()
