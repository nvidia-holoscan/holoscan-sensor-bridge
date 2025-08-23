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

set(EXAMPLE_INSTALL_TARGETS)
if(NOT HOLOLINK_BUILD_ONLY_NATIVE)
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
    hololink::operators::csi_to_bayer
    hololink::operators::image_processor
    hololink::operators::roce_receiver
    holoscan::core
    holoscan::ops::bayer_demosaic
    holoscan::ops::holoviz
    holoscan::pybind11
    pybind11::embed
  )

list(APPEND EXAMPLE_INSTALL_TARGETS imx274_player)

add_executable(vb1940_player
  vb1940_player.cpp
  )

target_link_libraries(vb1940_player
  PRIVATE
    hololink
    hololink::operators::csi_to_bayer
    hololink::operators::image_processor
    hololink::operators::roce_receiver
    hololink::sensors::native_vb1940_camera_sensor
    holoscan::core
    holoscan::ops::bayer_demosaic
    holoscan::ops::holoviz
  )

list(APPEND EXAMPLE_INSTALL_TARGETS vb1940_player)

add_executable(single_network_stereo_vb1940_player
  single_network_stereo_vb1940_player.cpp
  )

target_link_libraries(single_network_stereo_vb1940_player
  PRIVATE
    hololink
    hololink::operators::csi_to_bayer
    hololink::operators::image_processor
    hololink::operators::roce_receiver
    hololink::sensors::native_vb1940_camera_sensor
    holoscan::core
    holoscan::ops::bayer_demosaic
    holoscan::ops::holoviz
  )

list(APPEND EXAMPLE_INSTALL_TARGETS single_network_stereo_vb1940_player)

# Signal Generator sample app
add_executable(signal_generator
  signal_generator.cpp
)

target_link_libraries(signal_generator
  PRIVATE
    hololink
    hololink::core
    hololink::operators::iq_dec
    hololink::operators::iq_enc
    hololink::operators::roce_receiver
    hololink::operators::roce_transmitter
    hololink::operators::sig_gen
    hololink::operators::sig_viewer
    hololink::operators::udp_transmitter
    holoscan::core
)

list(APPEND EXAMPLE_INSTALL_TARGETS signal_generator)

if(HOLOLINK_BUILD_SIPL)
  add_executable(sipl_player
    sipl_player.cpp
  )

  target_link_libraries(sipl_player
    PRIVATE
      hololink
      hololink::operators::image_processor
      hololink::operators::packed_format_converter
      hololink::operators::sipl_capture
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
      gxf_multimedia
  )

  list(APPEND EXAMPLE_INSTALL_TARGETS sipl_player)
endif() # HOLOLINK_BUILD_SIPL

endif() # NOT HOLOLINK_BUILD_ONLY_NATIVE

# Installation of the examples
install(TARGETS ${EXAMPLE_INSTALL_TARGETS}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT hololink-examples)

set(EXAMPLE_INSTALL_FILES "")

if(HOLOLINK_BUILD_SIPL)
list(APPEND EXAMPLE_INSTALL_FILES
     sipl_config/vb1940_dual.json
     sipl_config/vb1940_single.json)
endif()

if(NOT HOLOLINK_BUILD_ONLY_NATIVE)
list(APPEND EXAMPLE_INSTALL_FILES
    body_pose_estimation.py
    distributed_imx274_player.py
    distributed_tao_peoplenet.py
    ecam0m30tof_player.py
    gpio_example_app.py
    imx274_latency.py
    imx274_player.py
    imx477_player.py
    imx715_player.py
    linux_audio_player.py
    linux_body_pose_estimation_imx477.py
    linux_body_pose_estimation.py
    linux_coe_imx274_player.py
    linux_hwisp_player.py
    linux_imx274_player.py
    linux_imx477_player.py
    linux_imx715_player.py
    linux_single_network_stereo_imx274_player.py
    linux_single_network_stereo_imx477_player.py
    linux_single_network_stereo_vb1940_player.py
    linux_tao_peoplenet_imx477.py
    linux_tao_peoplenet.py
    linux_vb1940_player.py
    native_imx274_player.py
    signal_generator.py
    single_network_stereo_imx274_player.py
    single_network_stereo_vb1940_player.py
    stereo_imx274_player.py
    stereo_imx477_player.py
    tao_peoplenet.py
    vb1940_player.py)
if(HOLOLINK_BUILD_SIPL)
  list(APPEND EXAMPLE_INSTALL_FILES sipl_player.py)
endif() # HOLOLINK_BUILD_SIPL
endif() # NOT HOLOLINK_BUILD_ONLY_NATIVE

install(FILES ${EXAMPLE_INSTALL_FILES}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT hololink-examples)
