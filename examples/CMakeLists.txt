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

# Adds an opt-in hardware run of a player: CTest reports it skipped (the command exits 77) unless
# the environment sets <gate>=1, as for the board-driving tests in tests/CMakeLists.txt. The run
# fails on error output, as the HSDK4 tests did. Tests that drive the board at the default address
# share its lock, so they never run at the same time.
function(hololink_add_hardware_test name gate label board)
  list(JOIN ARGN " " command)
  add_test(NAME ${name}
           COMMAND bash -c "if [ \"\${${gate}:-0}\" = \"1\" ]; then exec ${command}; else echo \"SKIPPED: set ${gate}=1 to run against ${board}\"; exit 77; fi"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(${name} PROPERTIES
    LABELS "${label}"
    SKIP_RETURN_CODE 77
    TIMEOUT 120
    RESOURCE_LOCK hsb_192_168_0_2
    FAIL_REGULAR_EXPRESSION "ERROR;\\[error\\]")
endfunction()

if(NOT HOLOLINK_BUILD_ONLY_NATIVE)
# Fully ported to HSDK5 (Graph/Operator<Params>): the module RoCE receiver
# (hololink::module::operators::RoceReceiverOp) and module's own CsiToBayerOp/ImageProcessorOp
# (hololink::module::operators::hsdk5 -- Imx274Cam::configure_converter() needs module's own
# CsiConverterV1, unlike the Hawk players below which bridge to native's converters instead;
# see hololink::module::operators::CsiToBayerOp::make_geometry_adapter()). Gated on the same
# HSDK5 viz/vision availability flags as the native/Hawk RoCE players.
if(HOLOLINK_BUILD_ROCE AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
  add_executable(module_imx274_player module_imx274_player.cpp)
  target_link_libraries(module_imx274_player
    PRIVATE
      hololink
      hololink::module
      hololink::hsb_lite::headers
      hololink::module::operators::hsdk5
      hololink::module::sensors::imx274
      holoscan::core
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cudart
    )
  target_include_directories(module_imx274_player PRIVATE
      ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS module_imx274_player)
endif() # HOLOLINK_BUILD_ROCE AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

# Fully ported to HSDK5 (Graph/Operator<Params>) -- same shape as module_imx274_player above.
if(HOLOLINK_BUILD_ROCE AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
  add_executable(imx274_module_tutorial imx274_module_tutorial.cpp)
  target_link_libraries(imx274_module_tutorial
    PRIVATE
      hololink
      hololink::module
      hololink::hsb_lite::headers
      hololink::module::operators::hsdk5
      hololink::module::sensors::imx274
      holoscan::core
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cudart
    )
  target_include_directories(imx274_module_tutorial PRIVATE
      ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS imx274_module_tutorial)
endif() # HOLOLINK_BUILD_ROCE AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

# Fully ported to HSDK5 (Graph/Operator<Params>) -- same shape as module_imx274_player above,
# generalized to 4 independent legs tiled 2x2 into one HolovizOp.
if(HOLOLINK_BUILD_ROCE AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
  add_executable(module_quad_imx274_player module_quad_imx274_player.cpp)
  target_link_libraries(module_quad_imx274_player
    PRIVATE
      hololink
      hololink::module
      hololink::hsb_lite::headers
      hololink::module::operators::hsdk5
      hololink::module::sensors::imx274
      holoscan::core
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cudart
    )
  target_include_directories(module_quad_imx274_player PRIVATE
      ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS module_quad_imx274_player)
endif() # HOLOLINK_BUILD_ROCE AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

# Status-LED companion for the Device Module Tutorial. It only touches the
# control plane and the bespoke LED service, so it needs the host module
# library and the Tutorial module's public header — no RoCE or Holoscan.
if(HOLOLINK_BUILD_MODULE)
  add_executable(example_module_tutorial example_module_tutorial.cpp)
  target_link_libraries(example_module_tutorial
    PRIVATE
      hololink::module
      hololink::tutorial::headers
    )
  list(APPEND EXAMPLE_INSTALL_TARGETS example_module_tutorial)
endif() # HOLOLINK_BUILD_MODULE

# linux_max96716a_player moved to the HSDK5-native gating alongside
# linux_single_network_stereo_imx274_player / linux_hawk_single_link_player -- see that block.

# Fully ported to HSDK5 (Graph/Operator<Params>) -- same shape as module_imx274_player above;
# see that block's comment for why this links hololink::module::operators::hsdk5 (module's own
# CsiToBayerOp/ImageProcessorOp) rather than native's.
if(HOLOLINK_BUILD_ROCE AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
  add_executable(module_vb1940_player module_vb1940_player.cpp)
  target_link_libraries(module_vb1940_player
    PRIVATE
      hololink
      hololink::module
      hololink::leopard_vb1940::headers
      hololink::module::operators::hsdk5
      hololink::module::sensors::vb1940
      holoscan::core
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cudart
    )
  target_include_directories(module_vb1940_player PRIVATE
      ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS module_vb1940_player)
endif() # HOLOLINK_BUILD_ROCE AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

# Fully ported to HSDK5 (Graph/Operator<Params>) -- same shape as module_vb1940_player above
# plus VsyncStartOp (examples/vsync_start_op.hpp, also ported).
if(HOLOLINK_BUILD_ROCE AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
  add_executable(module_stereo_vb1940_player module_stereo_vb1940_player.cpp)
  target_link_libraries(module_stereo_vb1940_player
    PRIVATE
      hololink
      hololink::module
      hololink::leopard_vb1940::headers
      hololink::module::operators::hsdk5
      hololink::module::sensors::vb1940
      holoscan::core
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cudart
    )
  target_include_directories(module_stereo_vb1940_player PRIVATE
      ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS module_stereo_vb1940_player)
endif() # HOLOLINK_BUILD_ROCE AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

# hawk_single_link_player / hawk_dual_link_player / single_network_stereo_hawk_player /
# max96716a_player moved to the HSDK5-native gating alongside imx274_player/vb1940_player --
# see that block for the receiver operator (hololink::module::operators::RoceReceiverOp,
# ported to HSDK5's Operator<Params> shape) and converter (native
# hololink::operators::CsiToBayerOp/ImageProcessorOp) targets the Hawk players now link.

# Software (Linux) receiver players. These use the module
# LinuxReceiverOp, which needs no ibverbs, so they are gated on the
# operators tree (Holoscan) rather than HOLOLINK_BUILD_ROCE — they build
# and run where the RoCE module players above do not. NOT
# HOLOLINK_BUILD_ONLY_NATIVE because the module (and thus the
# hololink::module::operators::hsdk5 / hololink::module::sensors targets these link)
# is not built in native-only configurations. Fully ported to HSDK5 (Graph/Operator<Params>)
# -- same shape as module_imx274_player/module_vb1940_player above; additionally gated on
# HOLOLINK_HAS_VIZ/HOLOLINK_HAS_BAYER_DEMOSAIC like every other HSDK5 viz-consuming app.
if(HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND NOT HOLOLINK_BUILD_ONLY_NATIVE AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
  add_executable(module_linux_imx274_player module_linux_imx274_player.cpp)
  target_link_libraries(module_linux_imx274_player
    PRIVATE
      hololink
      hololink::module
      hololink::hsb_lite::headers
      hololink::module::operators::hsdk5
      hololink::module::sensors::imx274
      holoscan::core
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cudart
    )
  target_include_directories(module_linux_imx274_player PRIVATE
      ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS module_linux_imx274_player)

  add_executable(module_linux_vb1940_player module_linux_vb1940_player.cpp)
  target_link_libraries(module_linux_vb1940_player
    PRIVATE
      hololink
      hololink::module
      hololink::leopard_vb1940::headers
      hololink::module::operators::hsdk5
      hololink::module::sensors::vb1940
      holoscan::core
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cudart
    )
  target_include_directories(module_linux_vb1940_player PRIVATE
      ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS module_linux_vb1940_player)
# module_pcie_imx258_player (HOLOLINK_BUILD_PCIE) is not built: it drives the hololink_module
# PcieReceiverOp, which is not yet ported to HSDK5.
endif() # HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND NOT HOLOLINK_BUILD_ONLY_NATIVE AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

if(HOLOLINK_BUILD_FUSA)
  # The FuSa CoE players display through holoscan::visualization::HolovizOp, after
  # holoscan::vision::BayerDemosaicOp except for the RGB fusa_coe_max96716a_player.
  # fusa_coe_imx274_player and fusa_coe_vb1940_player use only the native operators
  # (hololink::operators::*); the Hawk, module and MAX96716A players also need the
  # hololink_module operators (hololink::module::operators::hsdk5), so they are built only with
  # HOLOLINK_BUILD_MODULE and HOLOLINK_MODULE_BUILD_OPERATORS.
  if(HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
    add_executable(fusa_coe_imx274_player
      fusa_coe_imx274_player.cpp
      )

    target_link_libraries(fusa_coe_imx274_player
      PRIVATE
        hololink
        hololink::operators::csi_to_bayer
        hololink::operators::fusa_coe_capture
        hololink::operators::image_processor
        hololink::operators::packed_format_converter
        hololink::sensors::native_imx274_camera_sensor
        holoscan::core
        holoscan::visualization::op_holoviz
        holoscan::vision
        CUDA::cudart
      )

    list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_imx274_player)

    # Hawk stereo-camera players: module Hawk/Ar0234/Max9295d/Max96716a/
    # TauroTechDa326InterfaceV1 sensor driver (unchanged) paired with the module FuSa CoE
    # receiver (hololink::module::operators::FusaCoeCaptureOp, ported to HSDK5's
    # Operator<Params> shape -- see hololink::module::operators::hsdk5 below) and the
    # already-ported native converters. Also links native hololink::operators::
    # fusa_coe_capture for its static make_geometry_adapter() helper (pure geometry math,
    # shared by both the native and module FusaCoeCaptureOp shells -- see
    # fusa_coe_hawk_single_link_player.cpp's file comment). Needs HOLOLINK_BUILD_MODULE for
    # the sensor driver and the ported receiver target, in addition to the HSDK5
    # viz/vision gating above.
    if(HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS)
      set(HAWK_FUSA_PLAYERS
        fusa_coe_hawk_single_link_player
        fusa_coe_hawk_dual_link_player
        fusa_coe_single_network_stereo_hawk_player)
      foreach(hawk_player ${HAWK_FUSA_PLAYERS})
        add_executable(${hawk_player} ${hawk_player}.cpp)
        target_link_libraries(${hawk_player}
          PRIVATE
            hololink
            hololink::module
            hololink::module::operators::hsdk5
            hololink::module::sensors::max96716a
            hololink::taurotech_da326::headers
            hololink::module::sensors::hawk
            hololink::operators::csi_to_bayer
            hololink::operators::fusa_coe_capture
            hololink::operators::image_processor
            hololink::operators::packed_format_converter
            holoscan::core
            holoscan::visualization::op_holoviz
            holoscan::vision
            CUDA::cudart
          )
        target_include_directories(${hawk_player} PRIVATE
            ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
        list(APPEND EXAMPLE_INSTALL_TARGETS ${hawk_player})
      endforeach()
    endif() # HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS
  endif() # HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

  # Fully ported to HSDK5 (Graph/Operator<Params>) -- same shape as
  # module_fusa_coe_imx274_player below (see that block's comment).
  if(HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
    add_executable(module_fusa_coe_vb1940_player
      module_fusa_coe_vb1940_player.cpp
      )

    target_link_libraries(module_fusa_coe_vb1940_player
      PRIVATE
        hololink
        hololink::module
        hololink::leopard_vb1940::headers
        hololink::module::operators::hsdk5
        hololink::module::sensors::vb1940
        holoscan::core
        holoscan::visualization::op_holoviz
        holoscan::vision
        CUDA::cudart
      )
    target_include_directories(module_fusa_coe_vb1940_player PRIVATE
        ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)

    list(APPEND EXAMPLE_INSTALL_TARGETS module_fusa_coe_vb1940_player)
  endif() # HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

  # Fully ported to HSDK5 (Graph/Operator<Params>) -- close to a literal blueprint of
  # fusa_coe_imx274_player.cpp (native, already ported) with module types swapped in; see
  # that file for the packetized-vs-unpacketized branch and N-layer ViewTransform pattern.
  if(HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
    add_executable(module_fusa_coe_imx274_player
      module_fusa_coe_imx274_player.cpp
      )

    target_link_libraries(module_fusa_coe_imx274_player
      PRIVATE
        hololink
        hololink::module
        hololink::hsb_lite::headers
        hololink::module::operators::hsdk5
        hololink::module::sensors::imx274
        holoscan::core
        holoscan::visualization::op_holoviz
        holoscan::vision
        CUDA::cudart
      )
    target_include_directories(module_fusa_coe_imx274_player PRIVATE
        ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)

    list(APPEND EXAMPLE_INSTALL_TARGETS module_fusa_coe_imx274_player)
  endif() # HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

  add_executable(fusa_coe_frame_saver
    fusa_coe_frame_saver.cpp
    )

  target_link_libraries(fusa_coe_frame_saver
    PRIVATE
      hololink
      hololink::operators::fusa_coe_capture
      hololink::sensors::native_vb1940_camera_sensor
      holoscan::core
      CUDA::cudart
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_frame_saver)

  # HSDK5 live viewer: FusaCoeCaptureOp -> PackedFormatConverterOp or CsiToBayerOp ->
  # ImageProcessorOp -> BayerDemosaicOp -> HolovizOp. Gated on the Holoscan Visualization
  # module actually being present (see HOLOLINK_HAS_VIZ in the top-level CMakeLists.txt). Also
  # needs holoscan::vision's BayerDemosaicOp for full-color display (HOLOLINK_HAS_BAYER_DEMOSAIC).
  if(HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
    add_executable(fusa_coe_vb1940_player
      fusa_coe_vb1940_player.cpp
      )

    target_link_libraries(fusa_coe_vb1940_player
      PRIVATE
        hololink
        hololink::operators::fusa_coe_capture
        hololink::operators::csi_to_bayer
        hololink::operators::image_processor
        hololink::operators::packed_format_converter
        hololink::sensors::native_vb1940_camera_sensor
        holoscan::core
        holoscan::visualization::op_holoviz
        holoscan::vision
        CUDA::cudart
      )

    list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_vb1940_player)
  endif() # HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

  add_executable(fusa_coe_test_sensor_receiver
    fusa_coe_test_sensor_receiver.cpp
    )

  target_link_libraries(fusa_coe_test_sensor_receiver
    PRIVATE
      hololink
      hololink::operators::fusa_coe_capture
      holoscan::core
      CUDA::cudart
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_test_sensor_receiver)

  # Fully ported to HSDK5 (Graph/Operator<Params>) -- near-verbatim copy of the already-ported
  # max96716a_player/linux_max96716a_player (same RgbReshapeOp, same VPG/TauroTechDa326 setup)
  # with hololink::module::operators::FusaCoeCaptureOp swapped in for the receiver. No
  # CSI/Bayer stage (the VPG's built-in test pattern is RGB888 already, captured via
  # FusaCoeCaptureOp's opaque byte-blob mode), so -- unlike the Hawk FuSa players above --
  # this needs neither csi_to_bayer/image_processor nor native's fusa_coe_capture
  # (make_geometry_adapter() is never called here); matches max96716a_player's own simpler
  # link list and gating: HOLOLINK_HAS_VIZ without HOLOLINK_HAS_BAYER_DEMOSAIC, as there is no
  # demosaic stage.
  if(HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ)
    add_executable(fusa_coe_max96716a_player
      fusa_coe_max96716a_player.cpp
      )

    target_link_libraries(fusa_coe_max96716a_player
      PRIVATE
        hololink
        hololink::module
        hololink::module::operators::hsdk5
        hololink::module::sensors::max96716a
        hololink::taurotech_da326::headers
        holoscan::core
        holoscan::visualization::op_holoviz
      )

    target_include_directories(fusa_coe_max96716a_player PRIVATE
        ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
    list(APPEND EXAMPLE_INSTALL_TARGETS fusa_coe_max96716a_player)

    # fusa_coe_hawk_single_link_player / fusa_coe_hawk_dual_link_player /
    # fusa_coe_single_network_stereo_hawk_player moved to the HSDK5-native gating alongside
    # fusa_coe_imx274_player -- see that block.
  endif() # HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS AND HOLOLINK_HAS_VIZ
endif() # HOLOLINK_BUILD_FUSA

if(HOLOLINK_BUILD_ROCE)
  add_executable(cpu_roce_loopback cpu_roce_loopback.cpp)
  target_link_libraries(cpu_roce_loopback
    PRIVATE
      hololink
      hololink::operators::cpu_roce_transceiver
      holoscan::core
      CUDA::cuda_driver
  )
  list(APPEND EXAMPLE_INSTALL_TARGETS cpu_roce_loopback)
  if(HOLOLINK_BUILD_TESTS)
    add_test(NAME cpu_roce_loopback_help_test
             COMMAND cpu_roce_loopback --help
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  # Fully ported to HSDK5-native operators (hololink::operators::*, holoscan::vision::
  # BayerDemosaicOp, holoscan::visualization::HolovizOp) -- gated on the same HSDK5
  # viz/vision availability flags as fusa_coe_vb1940_player/fusa_coe_imx274_player.
  if(HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
    # C++ port of the HSDK4 tool imx274_latency.py.
    add_executable(roce_imx274_latency
      roce_imx274_latency.cpp
      )

    target_link_libraries(roce_imx274_latency
      PRIVATE
        hololink
        hololink::operators::csi_to_bayer
        hololink::operators::image_processor
        hololink::operators::roce_receiver
        hololink::sensors::native_imx274_camera_sensor
        holoscan::core
        holoscan::visualization::op_holoviz
        holoscan::vision
        CUDA::cuda_driver
      )

    list(APPEND EXAMPLE_INSTALL_TARGETS roce_imx274_latency)
    if(HOLOLINK_BUILD_TESTS)
      add_test(NAME roce_imx274_latency_help_test
               COMMAND roce_imx274_latency --help
               WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endif()

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
        holoscan::visualization::op_holoviz
        holoscan::vision
        CUDA::cudart
      )

    list(APPEND EXAMPLE_INSTALL_TARGETS single_network_stereo_vb1940_player)

    add_executable(imx274_player
      imx274_player.cpp
      )

    target_link_libraries(imx274_player
      PRIVATE
        hololink
        hololink::operators::csi_to_bayer
        hololink::operators::image_processor
        hololink::operators::roce_receiver
        hololink::sensors::native_imx274_camera_sensor
        holoscan::core
        holoscan::visualization::op_holoviz
        holoscan::vision
        CUDA::cudart
      )

    if(HOLOLINK_BUILD_TESTS)
      hololink_add_hardware_test(imx274_player_test HOLOLINK_TEST_IMX274 imx274
        "an HSB with an IMX274 camera attached"
        $<TARGET_FILE:imx274_player> --frame-limit=10 --headless)
    endif()
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
        holoscan::visualization::op_holoviz
        holoscan::vision
        CUDA::cudart
      )

    list(APPEND EXAMPLE_INSTALL_TARGETS vb1940_player)

    # Hawk stereo-camera players: module Hawk/Ar0234/Max9295d/Max96716a/
    # TauroTechDa326InterfaceV1 sensor driver (unchanged) paired with the module RoCE
    # receiver (hololink::module::operators::RoceReceiverOp, ported to HSDK5's
    # Operator<Params> shape -- see hololink::module::operators::hsdk5 below) and the
    # already-ported native converters. Needs HOLOLINK_BUILD_MODULE for the sensor driver
    # and the ported receiver target, in addition to the HSDK5 viz/vision gating above.
    if(HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS)
      set(HAWK_ROCE_PLAYERS
        hawk_single_link_player
        hawk_dual_link_player
        single_network_stereo_hawk_player)
      foreach(hawk_player ${HAWK_ROCE_PLAYERS})
        add_executable(${hawk_player} ${hawk_player}.cpp)
        target_link_libraries(${hawk_player}
          PRIVATE
            hololink
            hololink::module
            hololink::module::operators::hsdk5
            hololink::module::sensors::max96716a
            hololink::taurotech_da326::headers
            hololink::module::sensors::hawk
            hololink::operators::csi_to_bayer
            hololink::operators::image_processor
            holoscan::core
            holoscan::visualization::op_holoviz
            holoscan::vision
            CUDA::cudart
          )
        target_include_directories(${hawk_player} PRIVATE
            ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
        list(APPEND EXAMPLE_INSTALL_TARGETS ${hawk_player})
      endforeach()
    endif() # HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS
  endif() # HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

  # MAX96716A internal video pattern generator (VPG) player: no Hawk/Ar0234 sensor, no
  # CSI/Bayer conversion -- the deserializer's built-in test pattern is RGB888 already, so
  # only the module RoCE receiver (hololink::module::operators::RoceReceiverOp) and the
  # native HolovizOp are needed, without holoscan::vision.
  if(HOLOLINK_HAS_VIZ AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS)
    add_executable(max96716a_player max96716a_player.cpp)
    target_link_libraries(max96716a_player
      PRIVATE
        hololink
        hololink::module
        hololink::module::operators::hsdk5
        hololink::module::sensors::max96716a
        hololink::taurotech_da326::headers
        holoscan::core
        holoscan::visualization::op_holoviz
      )
    target_include_directories(max96716a_player PRIVATE
        ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
    list(APPEND EXAMPLE_INSTALL_TARGETS max96716a_player)
  endif() # HOLOLINK_HAS_VIZ AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS

  # Signal Generator sample app -- its operators' GUIs need hololink::ImGuiRenderer (see
  # HOLOLINK_HAS_IMGUI in the top-level CMakeLists.txt).
  if(HOLOLINK_HAS_IMGUI)
    add_executable(signal_generator
      signal_generator.cpp
    )

    target_link_libraries(signal_generator
      PRIVATE
        hololink
        hololink::core
        hololink::gui_renderer
        hololink::operators::discard_metadata
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
  endif()
endif() # HOLOLINK_BUILD_ROCE

# Fully ported to HSDK5-native operators (hololink::operators::*, holoscan::vision::
# BayerDemosaicOp, holoscan::visualization::HolovizOp) -- gated on the same HSDK5
# viz/vision availability flags as fusa_coe_vb1940_player/fusa_coe_imx274_player, not on
# the legacy holoscan::ops::bayer_demosaic target.
if(HOLOLINK_BUILD_SOCKET AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
  add_executable(linux_imx274_player
    linux_imx274_player.cpp
    )

  target_link_libraries(linux_imx274_player
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators::linux_receiver_op
      hololink::sensors::native_imx274_camera_sensor
      holoscan::core
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cuda_driver
    )

  if(HOLOLINK_BUILD_TESTS)
    hololink_add_hardware_test(linux_imx274_player_test HOLOLINK_TEST_IMX274 imx274
      "an HSB with an IMX274 camera attached"
      $<TARGET_FILE:linux_imx274_player> --frame-limit=10 --headless)
  endif()
  list(APPEND EXAMPLE_INSTALL_TARGETS linux_imx274_player)

  # C++ port of the HSDK4 tool linux_imx274_latency.py.
  add_executable(linux_imx274_latency
    linux_imx274_latency.cpp
    )

  target_link_libraries(linux_imx274_latency
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators::linux_receiver_op
      hololink::sensors::native_imx274_camera_sensor
      holoscan::core
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cuda_driver
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS linux_imx274_latency)

  if(HOLOLINK_BUILD_TESTS)
    add_test(NAME linux_imx274_latency_help_test
             COMMAND linux_imx274_latency --help
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()

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
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cudart
    )

  list(APPEND EXAMPLE_INSTALL_TARGETS linux_single_network_stereo_imx274_player)

  # Hawk stereo-camera players: module Hawk/Ar0234/Max9295d/Max96716a/
  # TauroTechDa326InterfaceV1 sensor driver (unchanged) paired with the module software
  # (Linux) receiver (hololink::module::operators::LinuxReceiverOp, ported to HSDK5's
  # Operator<Params> shape -- see hololink::module::operators::hsdk5 below) and the
  # already-ported native converters. Needs HOLOLINK_BUILD_MODULE for the sensor driver
  # and the ported receiver target, in addition to the HSDK5 viz/vision gating above.
  if(HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS)
    set(HAWK_LINUX_PLAYERS
      linux_hawk_single_link_player
      linux_hawk_dual_link_player
      linux_single_network_stereo_hawk_player)
    foreach(hawk_player ${HAWK_LINUX_PLAYERS})
      add_executable(${hawk_player} ${hawk_player}.cpp)
      target_link_libraries(${hawk_player}
        PRIVATE
          hololink
          hololink::module
          hololink::module::operators::hsdk5
          hololink::module::sensors::max96716a
          hololink::taurotech_da326::headers
          hololink::module::sensors::hawk
          hololink::operators::csi_to_bayer
          hololink::operators::image_processor
          holoscan::core
          holoscan::visualization::op_holoviz
          holoscan::vision
          CUDA::cudart
        )
      target_include_directories(${hawk_player} PRIVATE
          ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
      list(APPEND EXAMPLE_INSTALL_TARGETS ${hawk_player})
    endforeach()
  endif() # HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS
endif() # HOLOLINK_BUILD_SOCKET AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

# MAX96716A internal video pattern generator (VPG) player: no Hawk/Ar0234 sensor, no CSI/Bayer
# conversion -- the deserializer's built-in test pattern is RGB888 already, so only the module
# software (Linux) receiver (hololink::module::operators::LinuxReceiverOp) and the native
# HolovizOp are needed, without holoscan::vision.
if(HOLOLINK_BUILD_SOCKET AND HOLOLINK_HAS_VIZ AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS)
  add_executable(linux_max96716a_player linux_max96716a_player.cpp)
  target_link_libraries(linux_max96716a_player
    PRIVATE
      hololink
      hololink::module
      hololink::module::operators::hsdk5
      hololink::module::sensors::max96716a
      hololink::taurotech_da326::headers
      holoscan::core
      holoscan::visualization::op_holoviz
    )
  target_include_directories(linux_max96716a_player PRIVATE
      ${hololink_SOURCE_DIR}/hololink_module/host/operators/include)
  list(APPEND EXAMPLE_INSTALL_TARGETS linux_max96716a_player)
endif() # HOLOLINK_BUILD_SOCKET AND HOLOLINK_HAS_VIZ AND HOLOLINK_BUILD_MODULE AND HOLOLINK_MODULE_BUILD_OPERATORS

# Sub-frame players: the sensor frame travels and is processed as horizontal bands of
# --sub-frame-rows rows (CsiToBayerOp's sub-frame mode). SubFrameCombinerOp reassembles full frames
# for HolovizOp, or SubFrameVisualizerOp draws each band as it lands. The --help tests need no
# board or display: they prove each player links and starts.
if(HOLOLINK_BUILD_SOCKET AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
  add_executable(linux_sub_frame_imx274_player
    linux_sub_frame_imx274_player.cpp
    )

  target_link_libraries(linux_sub_frame_imx274_player
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators::linux_receiver_op
      hololink::operators::sub_frame_combiner
      hololink::sensors::native_imx274_camera_sensor
      holoscan::core
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cuda_driver
    )

  if(HOLOLINK_BUILD_TESTS)
    add_test(NAME linux_sub_frame_imx274_player_help_test
             COMMAND linux_sub_frame_imx274_player --help
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    hololink_add_hardware_test(linux_sub_frame_imx274_player_test HOLOLINK_TEST_IMX274 imx274
      "an HSB with an IMX274 camera attached"
      $<TARGET_FILE:linux_sub_frame_imx274_player> --frame-limit=10 --headless --sub-frame-rows=1080)
  endif()

  list(APPEND EXAMPLE_INSTALL_TARGETS linux_sub_frame_imx274_player)
endif() # HOLOLINK_BUILD_SOCKET AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

# Grayscale (MONO16) display; --sub-frame-display draws each band through SubFrameVisualizerOp.
if(HOLOLINK_BUILD_SOCKET AND HOLOLINK_HAS_VIZ)
  add_executable(linux_sub_frame_imx274_hsdk5_player
    linux_sub_frame_imx274_hsdk5_player.cpp
    )

  target_link_libraries(linux_sub_frame_imx274_hsdk5_player
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators::linux_receiver_op
      hololink::operators::sub_frame_combiner
      hololink::sensors::native_imx274_camera_sensor
      holoscan::core
      holoscan::visualization::op_holoviz
      CUDA::cudart
    )

  if(HOLOLINK_BUILD_TESTS)
    add_test(NAME linux_sub_frame_imx274_hsdk5_player_help_test
             COMMAND linux_sub_frame_imx274_hsdk5_player --help
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  # --sub-frame-display: render each band as it arrives through SubFrameVisualizerOp
  # (needs the HoloViz library, holoscan::visualization::holoviz).
  if(TARGET hololink::operators::sub_frame_visualizer)
    target_link_libraries(linux_sub_frame_imx274_hsdk5_player
      PRIVATE hololink::operators::sub_frame_visualizer)
    target_compile_definitions(linux_sub_frame_imx274_hsdk5_player
      PRIVATE HOLOLINK_HAS_SUB_FRAME_VISUALIZER=1)
  endif()

  list(APPEND EXAMPLE_INSTALL_TARGETS linux_sub_frame_imx274_hsdk5_player)
endif() # HOLOLINK_BUILD_SOCKET AND HOLOLINK_HAS_VIZ

if(HOLOLINK_BUILD_ROCE AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC)
  add_executable(sub_frame_imx274_player
    sub_frame_imx274_player.cpp
    )

  target_link_libraries(sub_frame_imx274_player
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators::roce_receiver
      hololink::operators::sub_frame_combiner
      hololink::sensors::native_imx274_camera_sensor
      holoscan::core
      holoscan::visualization::op_holoviz
      holoscan::vision
      CUDA::cuda_driver
    )

  if(HOLOLINK_BUILD_TESTS)
    add_test(NAME sub_frame_imx274_player_help_test
             COMMAND sub_frame_imx274_player --help
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    hololink_add_hardware_test(sub_frame_imx274_player_test HOLOLINK_TEST_IMX274 imx274
      "an HSB with an IMX274 camera attached"
      $<TARGET_FILE:sub_frame_imx274_player> --frame-limit=10 --headless --sub-frame-rows=1080)
  endif()

  list(APPEND EXAMPLE_INSTALL_TARGETS sub_frame_imx274_player)
endif() # HOLOLINK_BUILD_ROCE AND HOLOLINK_HAS_VIZ AND HOLOLINK_HAS_BAYER_DEMOSAIC

# Draws each band through SubFrameVisualizerOp, which is built when the HoloViz library is found.
if(HOLOLINK_BUILD_ROCE AND HOLOLINK_HAS_BAYER_DEMOSAIC AND TARGET hololink::operators::sub_frame_visualizer)
  add_executable(sub_frame_vb1940_player
    sub_frame_vb1940_player.cpp
    )

  target_link_libraries(sub_frame_vb1940_player
    PRIVATE
      hololink
      hololink::operators::csi_to_bayer
      hololink::operators::image_processor
      hololink::operators::roce_receiver
      hololink::operators::sub_frame_visualizer
      hololink::sensors::native_vb1940_camera_sensor
      holoscan::core
      holoscan::vision
      CUDA::cuda_driver
    )

  if(HOLOLINK_BUILD_TESTS)
    add_test(NAME sub_frame_vb1940_player_help_test
             COMMAND sub_frame_vb1940_player --help
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    # No --headless: SubFrameVisualizerOp needs a display. 992 rows: two bands of the default
    # 2560x1984 mode.
    hololink_add_hardware_test(sub_frame_vb1940_player_test HOLOLINK_TEST_VB1940 vb1940
      "a Leopard VB1940 board"
      $<TARGET_FILE:sub_frame_vb1940_player> --frame-limit=10 --sub-frame-rows=992)
  endif()

  list(APPEND EXAMPLE_INSTALL_TARGETS sub_frame_vb1940_player)
endif() # HOLOLINK_BUILD_ROCE AND HOLOLINK_HAS_BAYER_DEMOSAIC AND TARGET hololink::operators::sub_frame_visualizer

endif() # NOT HOLOLINK_BUILD_ONLY_NATIVE


# Installation of the examples
install(TARGETS ${EXAMPLE_INSTALL_TARGETS}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT hololink-examples)

set(EXAMPLE_INSTALL_FILES "")

if(HOLOLINK_BUILD_PYTHON)
list(APPEND EXAMPLE_INSTALL_FILES
    body_pose_estimation.py
    e2e_imx274_latency.py
    e2e_vb1940_latency.py
    ecam0m30tof_player.py
    gpio_example_app.py
    hawk_dual_link_player.py
    hawk_single_link_player.py
    imx274_frame_validation.py
    imx274_latency.py
    imx274_module_tutorial.py
    imx274_player.py
    imx477_player.py
    imx715_player.py
    linux_audio_player.py
    linux_body_pose_estimation_imx477.py
    linux_body_pose_estimation.py
    linux_coe_imx274_player.py
    linux_imx274_frame_validation.py
    linux_imx274_latency.py
    linux_imx274_player.py
    linux_imx477_player.py
    linux_imx715_player.py
    linux_single_network_stereo_imx274_player.py
    linux_single_network_stereo_imx477_player.py
    linux_single_network_stereo_vb1940_player.py
    linux_tao_peoplenet_imx477.py
    linux_tao_peoplenet.py
    linux_tsn_imx274_player.py
    linux_vb1940_player.py
    max96716a_player.py
    module_linux_imx274_player.py
    module_linux_vb1940_player.py
    module_quad_imx274_player.py
    native_imx274_player.py
    single_network_stereo_hawk_player.py
    single_network_stereo_imx274_player.py
    single_network_stereo_vb1940_player.py
    stereo_imx274_frame_validation.py
    stereo_imx274_player.py
    stereo_imx477_player.py
    tao_peoplenet.py
    tsn_imx274_player.py
    vb1940_player.py

    body_pose_estimation.yaml
    example_configuration.yaml
    tao_peoplenet.yaml
)
if(HOLOLINK_BUILD_FUSA)
  list(APPEND EXAMPLE_INSTALL_FILES
    fusa_coe_imx274_player.py
    fusa_coe_vb1940_player.py
    fusa_coe_vb1940_latency.py
    fusa_coe_max96716a_player.py
    fusa_coe_hawk_single_link_player.py
    fusa_coe_hawk_dual_link_player.py
    fusa_coe_single_network_stereo_hawk_player.py
    vb1940_fusa_nvcomp_crc_validation.py)
endif() # HOLOLINK_BUILD_FUSA
if(HOLOLINK_BUILD_ARGUS_ISP)
  list(APPEND EXAMPLE_INSTALL_FILES
    linux_hwisp_player.py)
endif() # HOLOLINK_BUILD_ARGUS_ISP
if(HOLOLINK_HAS_IMGUI)
  list(APPEND EXAMPLE_INSTALL_FILES
    signal_generator.py)
endif() # HOLOLINK_HAS_IMGUI
endif() # HOLOLINK_BUILD_PYTHON

install(FILES ${EXAMPLE_INSTALL_FILES}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT hololink-examples)
