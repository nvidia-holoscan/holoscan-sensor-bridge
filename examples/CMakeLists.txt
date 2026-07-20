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

set(EXAMPLE_INSTALL_TARGETS)

if(NOT HOLOLINK_BUILD_ONLY_NATIVE)
if(HOLOLINK_BUILD_ROCE)
  add_executable(imx274_player
    imx274_player.cpp
    )

  if(HOLOLINK_BUILD_TESTS)
    add_test(NAME imx274_player_test
              COMMAND imx274_player --frame-limit=10 --headless
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(imx274_player_test PROPERTIES
                          FAIL_REGULAR_EXPRESSION "ERROR;\\[error\\]")
  endif()

  target_link_libraries(imx274_player
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators::roce_receiver
      hololink::sensors::native_imx274_camera_sensor
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS imx274_player)

  add_executable(module_imx274_player module_imx274_player.cpp)
  target_link_libraries(module_imx274_player
    PRIVATE
      hololink::module
      hololink::hsb_lite::headers
      hololink::operators
      hololink::module::sensors::imx274
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )
  list(APPEND EXAMPLE_INSTALL_TARGETS module_imx274_player)

  if(HOLOLINK_MODULE_BUILD_OPERATORS)
    add_executable(imx274_module_tutorial imx274_module_tutorial.cpp)
    target_link_libraries(imx274_module_tutorial
      PRIVATE
        hololink::module
        hololink::hsb_lite::headers
        hololink::operators
        hololink::module::sensors::imx274
        holoscan::core
        holoscan::ops::bayer_demosaic
        holoscan::ops::holoviz
      )
    list(APPEND EXAMPLE_INSTALL_TARGETS imx274_module_tutorial)
  endif() # HOLOLINK_MODULE_BUILD_OPERATORS

  add_executable(module_quad_imx274_player module_quad_imx274_player.cpp)
  target_link_libraries(module_quad_imx274_player
    PRIVATE
      hololink::module
      hololink::hsb_lite::headers
      hololink::operators
      hololink::module::sensors::imx274
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )
  list(APPEND EXAMPLE_INSTALL_TARGETS module_quad_imx274_player)
endif() # HOLOLINK_BUILD_ROCE

# Status-LED companion for the Device Module Tutorial. It only touches the
# control plane and the bespoke LED service, so it needs the host module
# library and the Tutorial module's public header — no RoCE or Holoscan.
add_executable(example_module_tutorial example_module_tutorial.cpp)
target_link_libraries(example_module_tutorial
  PRIVATE
    hololink::module
    hololink::tutorial::headers
  )
list(APPEND EXAMPLE_INSTALL_TARGETS example_module_tutorial)

if(HOLOLINK_BUILD_SOCKET)
  add_executable(linux_imx274_player
    linux_imx274_player.cpp
    )

  if(HOLOLINK_BUILD_TESTS)
    add_test(NAME linux_imx274_player_test
              COMMAND linux_imx274_player --frame-limit=10 --headless
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    set_tests_properties(linux_imx274_player_test PROPERTIES
                          FAIL_REGULAR_EXPRESSION "ERROR;\\[error\\]")
  endif()

  target_link_libraries(linux_imx274_player
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators
      hololink::operators::linux_receiver_op
      hololink::sensors::native_imx274_camera_sensor
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS linux_imx274_player)

  add_executable(linux_max96716a_player
    linux_max96716a_player.cpp
    )

  target_link_libraries(linux_max96716a_player
    PRIVATE
      hololink
      hololink::operators
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      holoscan::core
      holoscan::ops::holoviz
    )

  target_include_directories(linux_max96716a_player PRIVATE
      ${CMAKE_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS linux_max96716a_player)

  add_executable(linux_hawk_single_link_player
    linux_hawk_single_link_player.cpp
    )

  target_link_libraries(linux_hawk_single_link_player
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      hololink::module::sensors::hawk
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  target_include_directories(linux_hawk_single_link_player PRIVATE
      ${CMAKE_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS linux_hawk_single_link_player)

  add_executable(linux_hawk_dual_link_player
    linux_hawk_dual_link_player.cpp
    )

  target_link_libraries(linux_hawk_dual_link_player
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      hololink::module::sensors::hawk
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  target_include_directories(linux_hawk_dual_link_player PRIVATE
      ${CMAKE_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS linux_hawk_dual_link_player)

  add_executable(linux_single_network_stereo_hawk_player
    linux_single_network_stereo_hawk_player.cpp
    )

  target_link_libraries(linux_single_network_stereo_hawk_player
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      hololink::module::sensors::hawk
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  target_include_directories(linux_single_network_stereo_hawk_player PRIVATE
      ${CMAKE_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS linux_single_network_stereo_hawk_player)
endif() # HOLOLINK_BUILD_SOCKET

if(HOLOLINK_BUILD_ROCE)
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

  add_executable(module_vb1940_player module_vb1940_player.cpp)
  target_link_libraries(module_vb1940_player
    PRIVATE
      hololink::module
      hololink::leopard_vb1940::headers
      hololink::operators
      hololink::module::sensors::vb1940
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )
  list(APPEND EXAMPLE_INSTALL_TARGETS module_vb1940_player)

  add_executable(module_stereo_vb1940_player module_stereo_vb1940_player.cpp)
  target_link_libraries(module_stereo_vb1940_player
    PRIVATE
      hololink::module
      hololink::leopard_vb1940::headers
      hololink::operators
      hololink::module::sensors::vb1940
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )
  list(APPEND EXAMPLE_INSTALL_TARGETS module_stereo_vb1940_player)

  add_executable(hawk_single_link_player
    hawk_single_link_player.cpp
    )

  target_link_libraries(hawk_single_link_player
    PRIVATE
      hololink
      hololink::module
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      hololink::module::sensors::hawk
      hololink::module
      hololink::taurotech_da326::headers
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS hawk_single_link_player)

  add_executable(hawk_dual_link_player
    hawk_dual_link_player.cpp
    )

  target_link_libraries(hawk_dual_link_player
    PRIVATE
      hololink
      hololink::module
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      hololink::module::sensors::hawk
      hololink::module
      hololink::taurotech_da326::headers
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS hawk_dual_link_player)

  add_executable(max96716a_player
    max96716a_player.cpp
    )

  target_link_libraries(max96716a_player
    PRIVATE
      hololink
      hololink::module
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS max96716a_player)

  add_executable(single_network_stereo_hawk_player
    single_network_stereo_hawk_player.cpp
    )

  target_link_libraries(single_network_stereo_hawk_player
    PRIVATE
      hololink
      hololink::module
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      hololink::module::sensors::hawk
      hololink::module
      hololink::taurotech_da326::headers
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS single_network_stereo_hawk_player)
endif() # HOLOLINK_BUILD_ROCE

# Software (Linux) receiver players. These use the module
# LinuxReceiverOp, which needs no ibverbs, so they are gated on the
# operators tree (Holoscan) rather than HOLOLINK_BUILD_ROCE — they build
# and run where the RoCE module players above do not. NOT
# HOLOLINK_BUILD_ONLY_NATIVE because the module (and thus the
# hololink::operators / hololink::module::sensors targets these link)
# is not built in native-only configurations.
if(HOLOLINK_MODULE_BUILD_OPERATORS AND NOT HOLOLINK_BUILD_ONLY_NATIVE)
  add_executable(module_linux_imx274_player module_linux_imx274_player.cpp)
  target_link_libraries(module_linux_imx274_player
    PRIVATE
      hololink::module
      hololink::hsb_lite::headers
      hololink::operators
      hololink::module::sensors::imx274
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )
  list(APPEND EXAMPLE_INSTALL_TARGETS module_linux_imx274_player)

  add_executable(module_linux_vb1940_player module_linux_vb1940_player.cpp)
  target_link_libraries(module_linux_vb1940_player
    PRIVATE
      hololink::module
      hololink::leopard_vb1940::headers
      hololink::operators
      hololink::module::sensors::vb1940
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )
  list(APPEND EXAMPLE_INSTALL_TARGETS module_linux_vb1940_player)
endif() # HOLOLINK_MODULE_BUILD_OPERATORS

if(HOLOLINK_BUILD_FUSA)
  add_executable(fusa_coe_imx274_player
    fusa_coe_imx274_player.cpp
    )

  target_link_libraries(fusa_coe_imx274_player
    PRIVATE
      hololink
      hololink::operators::fusa_coe_capture
      hololink::operators::image_processor
      hololink::operators::packed_format_converter
      hololink::sensors::native_imx274_camera_sensor
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_imx274_player)

  add_executable(fusa_coe_vb1940_player
    fusa_coe_vb1940_player.cpp
    )

  target_link_libraries(fusa_coe_vb1940_player
    PRIVATE
      hololink
      hololink::operators::fusa_coe_capture
      hololink::operators::image_processor
      hololink::operators::packed_format_converter
      hololink::sensors::native_vb1940_camera_sensor
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_vb1940_player)

  add_executable(module_fusa_coe_vb1940_player
    module_fusa_coe_vb1940_player.cpp
    )

  target_link_libraries(module_fusa_coe_vb1940_player
    PRIVATE
      hololink::module
      hololink::leopard_vb1940::headers
      hololink::operators
      hololink::module::sensors::vb1940
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS module_fusa_coe_vb1940_player)

  add_executable(fusa_coe_test_sensor_receiver
    fusa_coe_test_sensor_receiver.cpp
    )

  target_link_libraries(fusa_coe_test_sensor_receiver
    PRIVATE
      hololink
      hololink::operators::fusa_coe_capture
      holoscan::core
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_test_sensor_receiver)

  add_executable(fusa_coe_max96716a_player
    fusa_coe_max96716a_player.cpp
    )

  target_link_libraries(fusa_coe_max96716a_player
    PRIVATE
      hololink
      hololink_operators
      hololink::operators::fusa_coe_capture
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      holoscan::core
      holoscan::ops::holoviz
    )

  target_include_directories(fusa_coe_max96716a_player PRIVATE
      ${CMAKE_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_max96716a_player)

  add_executable(fusa_coe_hawk_single_link_player
    fusa_coe_hawk_single_link_player.cpp
    )

  target_link_libraries(fusa_coe_hawk_single_link_player
    PRIVATE
      hololink
      hololink_operators
      hololink::operators::fusa_coe_capture
      hololink::operators::image_processor
      hololink::operators::packed_format_converter
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      hololink::module::sensors::hawk
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  target_include_directories(fusa_coe_hawk_single_link_player PRIVATE
      ${CMAKE_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_hawk_single_link_player)

  add_executable(fusa_coe_hawk_dual_link_player
    fusa_coe_hawk_dual_link_player.cpp
    )

  target_link_libraries(fusa_coe_hawk_dual_link_player
    PRIVATE
      hololink
      hololink_operators
      hololink::operators::fusa_coe_capture
      hololink::operators::image_processor
      hololink::operators::packed_format_converter
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      hololink::module::sensors::hawk
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  target_include_directories(fusa_coe_hawk_dual_link_player PRIVATE
      ${CMAKE_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_hawk_dual_link_player)

  add_executable(fusa_coe_single_network_stereo_hawk_player
    fusa_coe_single_network_stereo_hawk_player.cpp
    )

  target_link_libraries(fusa_coe_single_network_stereo_hawk_player
    PRIVATE
      hololink
      hololink_operators
      hololink::operators::fusa_coe_capture
      hololink::operators::image_processor
      hololink::operators::packed_format_converter
      hololink::module::sensors::max96716a
      hololink::module
      hololink::taurotech_da326::headers
      hololink::module::sensors::hawk
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  target_include_directories(fusa_coe_single_network_stereo_hawk_player PRIVATE
      ${CMAKE_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_single_network_stereo_hawk_player)
endif() # HOLOLINK_BUILD_FUSA

if(HOLOLINK_BUILD_ROCE)
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
endif() # HOLOLINK_BUILD_ROCE

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
  )

  list(APPEND EXAMPLE_INSTALL_TARGETS sipl_player)
endif() # HOLOLINK_BUILD_SIPL

if(HOLOLINK_BUILD_SOCKET)
  add_executable(linux_single_network_stereo_imx274_player
    linux_single_network_stereo_imx274_player.cpp
    )

  target_link_libraries(linux_single_network_stereo_imx274_player
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators::linux_receiver_op
      hololink::sensors::native_imx274_camera_sensor
      holoscan::core
      holoscan::ops::bayer_demosaic
      holoscan::ops::holoviz
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS linux_single_network_stereo_imx274_player)

  if(HOLOLINK_BUILD_PYTHON)
    include(hololink_deps/pybind11)
    add_executable(linux_sub_frame_imx274_player
      linux_sub_frame_imx274_player.cpp
      )

    if(HOLOLINK_BUILD_TESTS)
      add_test(NAME linux_sub_frame_imx274_player_test
                COMMAND linux_sub_frame_imx274_player --frame-limit=10 --headless --sub-frame-rows=1080
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      set_tests_properties(linux_sub_frame_imx274_player_test PROPERTIES
                            FAIL_REGULAR_EXPRESSION "ERROR;\\[error\\]")
    endif()

    target_link_libraries(linux_sub_frame_imx274_player
      PRIVATE
        hololink
        hololink::operators::csi_to_bayer
        hololink::operators::image_processor
        hololink::operators::linux_receiver_op
        hololink::operators::sub_frame_combiner
        holoscan::core
        holoscan::ops::bayer_demosaic
        holoscan::ops::holoviz
        holoscan::pybind11
        pybind11::embed
      )

    list(APPEND EXAMPLE_INSTALL_TARGETS linux_sub_frame_imx274_player)
  endif() # HOLOLINK_BUILD_PYTHON
endif() # HOLOLINK_BUILD_SOCKET

if(HOLOLINK_BUILD_ROCE)
  if(HOLOLINK_BUILD_PYTHON)
    add_executable(sub_frame_imx274_player
      sub_frame_imx274_player.cpp
      )

    if(HOLOLINK_BUILD_TESTS)
      add_test(NAME sub_frame_imx274_player_test
                COMMAND sub_frame_imx274_player --frame-limit=10 --headless --sub-frame-rows=1080
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      set_tests_properties(sub_frame_imx274_player_test PROPERTIES
                            FAIL_REGULAR_EXPRESSION "ERROR;\\[error\\]")
    endif()

    target_link_libraries(sub_frame_imx274_player
      PRIVATE
        hololink
        hololink::operators::csi_to_bayer
        hololink::operators::image_processor
        hololink::operators::roce_receiver
        hololink::operators::sub_frame_combiner
        holoscan::core
        holoscan::ops::bayer_demosaic
        holoscan::ops::holoviz
        holoscan::pybind11
        pybind11::embed
      )

    list(APPEND EXAMPLE_INSTALL_TARGETS sub_frame_imx274_player)
  endif() # HOLOLINK_BUILD_PYTHON

endif() # HOLOLINK_BUILD_ROCE

if(HOLOLINK_BUILD_ROCE)
  if(HOLOLINK_BUILD_PYTHON)
    add_executable(sub_frame_vb1940_player
      sub_frame_vb1940_player.cpp
      )

    if(HOLOLINK_BUILD_TESTS)
      # Note: No "--headless" support, requires display
      add_test(NAME sub_frame_vb1940_player_test
                COMMAND sub_frame_vb1940_player --frame-limit=10 --sub-frame-rows=1080
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
      set_tests_properties(sub_frame_vb1940_player_test PROPERTIES
                            FAIL_REGULAR_EXPRESSION "ERROR;\\[error\\]")
    endif()

    target_link_libraries(sub_frame_vb1940_player
      PRIVATE
        hololink
        hololink::operators::csi_to_bayer
        hololink::operators::image_processor
        hololink::operators::roce_receiver
        hololink::operators::sub_frame_combiner
        hololink::sensors::native_vb1940_camera_sensor
        hololink::operators::sub_frame_visualizer
        holoscan::core
        holoscan::ops::bayer_demosaic
        holoscan::ops::holoviz
     )

    list(APPEND EXAMPLE_INSTALL_TARGETS sub_frame_vb1940_player)
    endif() # HOLOLINK_BUILD_PYTHON
endif() # HOLOLINK_BUILD_ROCE

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

if(HOLOLINK_BUILD_PYTHON)
list(APPEND EXAMPLE_INSTALL_FILES
    body_pose_estimation.py
    distributed_imx274_player.py
    distributed_tao_peoplenet.py
    ecam0m30tof_player.py
    gpio_example_app.py
    hawk_dual_link_player.py
    hawk_single_link_player.py
    imx274_latency.py
    imx274_module_tutorial.py
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
    max96716a_player.py
    module_linux_imx274_player.py
    module_linux_vb1940_player.py
    module_quad_imx274_player.py
    native_imx274_player.py
    signal_generator.py
    single_network_stereo_hawk_player.py
    single_network_stereo_imx274_player.py
    single_network_stereo_vb1940_player.py
    stereo_imx274_player.py
    stereo_imx477_player.py
    tao_peoplenet.py
    vb1940_player.py

    body_pose_estimation.yaml
    example_configuration.yaml
    tao_peoplenet.yaml
)
if(HOLOLINK_BUILD_FUSA)
  list(APPEND EXAMPLE_INSTALL_FILES
    fusa_coe_imx274_player.py
    fusa_coe_vb1940_player.py
    fusa_coe_max96716a_player.py
    fusa_coe_hawk_single_link_player.py
    fusa_coe_hawk_dual_link_player.py
    fusa_coe_single_network_stereo_hawk_player.py)
endif() # HOLOLINK_BUILD_FUSA
if(HOLOLINK_BUILD_SIPL)
  list(APPEND EXAMPLE_INSTALL_FILES sipl_player.py)
endif() # HOLOLINK_BUILD_SIPL
endif() # HOLOLINK_BUILD_PYTHON

install(FILES ${EXAMPLE_INSTALL_FILES}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT hololink-examples)
add_subdirectory(aditof)        
