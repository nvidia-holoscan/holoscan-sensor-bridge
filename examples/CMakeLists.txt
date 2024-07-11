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

add_executable(imx274_player
  imx274_player.cpp
  )

add_test(NAME imx274_player_test
          COMMAND imx274_player --frame-limit=10 --headless
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(imx274_player_test PROPERTIES
                      FAIL_REGULAR_EXPRESSION "ERROR;\\[error\\]")

target_link_libraries(imx274_player
  PRIVATE
    hololink
    hololink::native
    hololink::operators::csi_to_bayer
    hololink::operators::image_processor
    hololink::operators::gamma_correction
    hololink::operators::roce_receiver
    holoscan::core
    holoscan::ops::bayer_demosaic
    holoscan::ops::holoviz
    pybind11::embed
  )
