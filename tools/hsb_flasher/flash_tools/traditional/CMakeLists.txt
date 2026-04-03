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

add_library(traditional_peripherals STATIC
    traditional_spi.cpp
    traditional_i2c.cpp
)

set_property(TARGET traditional_peripherals PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(traditional_peripherals
    PUBLIC
    hololink_core
)

# pybind11 module exposing traditional SPI/I2C to Python
pybind11_add_module(traditional_peripherals_py
    traditional_bindings.cpp
)

target_link_libraries(traditional_peripherals_py
    PRIVATE
    traditional_peripherals
    hololink_core
    fmt::fmt-header-only
)

# Copy traditional_peripherals_py module to hsb_flasher output directory
add_custom_command(TARGET traditional_peripherals_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:traditional_peripherals_py>
        $<TARGET_FILE_DIR:hsb_flasher>/$<TARGET_FILE_NAME:traditional_peripherals_py>
    COMMENT "Copying traditional_peripherals_py module to hsb_flasher directory"
)
