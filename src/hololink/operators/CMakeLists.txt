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

# ---------------------------------------------------------------------------
# CaptureMetadata FlatBuffer schema — per-capture hardware metadata, carried
# on its own port alongside (not embedded in) an operator's Tensor/ImageT
# payload. See capture_metadata.fbs for why it's CaptureMetadata rather than
# HololinkMetadata/FrameMetadata.
# ---------------------------------------------------------------------------
set(CAPTURE_METADATA_SCHEMA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

# PYTHON (conditional on HOLOLINK_BUILD_PYTHON, not unconditional -- this flag pulls in a
# nanobind/Python3-dev build step, which a pure-C++ build shouldn't pay for or require) emits a
# Python dataclass module (capture_metadata.CaptureMetadata) that auto-registers the same
# "hololink.CaptureMetadataT" payload identity the C++ side uses (both derive it from the same
# embedded BFBS reflection), so a native Python operator can type a port on it and receive
# hololink::CaptureMetadataT samples from an already-bound C++ operator (e.g.
# ImageProcessorOp.metadata) with no further glue. See operator_catalog.cpp's top-of-file
# comment for the resulting duplicate-registration fix this required there.
set(_capture_metadata_python_args)
if(HOLOLINK_BUILD_PYTHON)
  # Places the generated capture_metadata.py/.pyi/_capture_metadata_holoscan.abi3.so directly
  # alongside the hololink package itself (${CMAKE_BINARY_DIR}/python/${CMAKE_INSTALL_LIBDIR},
  # the same directory python/hololink/CMakeLists.txt's HOLOLINK_PYTHON_MODULE_OUT_DIR is a
  # "hololink" subdirectory of) rather than this target's own default (next to its C++-only
  # outputs under build/src/...) -- that directory is already a PYTHONPATH prerequisite for
  # `import hololink` to work at all, so no separate install-time copy step is needed; bare
  # `import capture_metadata` just works wherever `import hololink` already does. Reconstructed
  # directly (not via HOLOLINK_PYTHON_MODULE_OUT_DIR itself) because python/ is processed after
  # src/ in the top-level CMakeLists, so that variable isn't set yet at this point.
  list(APPEND _capture_metadata_python_args
    PYTHON
    PYTHON_OUTPUT_DIR "${CMAKE_BINARY_DIR}/python/${CMAKE_INSTALL_LIBDIR}"
    PYTHON_INSTALL_COMPONENT "hololink-python_libs"
  )
  # Also install the generated capture_metadata module next to the hololink and
  # hololink_module packages, so bare `import capture_metadata` (done by both packages'
  # __init__.py) resolves after `cmake --install` without relying on where the schema helper's
  # own PYTHON install rule puts it. Matches only the capture_metadata files; the sibling
  # `hololink` directory is installed by python/hololink/CMakeLists.txt.
  set(HOLOLINK_PYTHON_INSTALL_DIR "python"
      CACHE PATH "Output installation directory for hololink python modules")
  install(
    DIRECTORY "${CMAKE_BINARY_DIR}/python/${CMAKE_INSTALL_LIBDIR}/"
    DESTINATION "${HOLOLINK_PYTHON_INSTALL_DIR}"
    COMPONENT "hololink-python_libs"
    FILES_MATCHING
      PATTERN "capture_metadata*"
      PATTERN "_capture_metadata*"
    PATTERN "__pycache__" EXCLUDE
    PATTERN "hololink" EXCLUDE
  )
endif()

holoscan_add_flatbuffer_schema(
  TARGET capture_metadata_schema_generated
  SCHEMA capture_metadata.fbs
  INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}" "${holoscan_SCHEMA_DIR}"
  OUTPUT_DIR "${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}"
  BFBS_EMBED
  TRAITS_HEADER capture_metadata_traits.hpp
  MAX_SERIALIZED_SIZE 1024
  ${_capture_metadata_python_args}
)

install(TARGETS capture_metadata_schema_generated
        EXPORT HololinkTargets
        COMPONENT hololink-operators)

# The generated headers themselves: the operators' public headers include them by bare name
# (<capture_metadata_traits.hpp>), which an installed consumer resolves through the operator
# targets' ${CMAKE_INSTALL_INCLUDEDIR} include directory. The schema helper only exposes
# OUTPUT_DIR to this build tree.
install(FILES
          ${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}/capture_metadata_generated.h
          ${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}/capture_metadata_bfbs_generated.h
          ${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}/capture_metadata_traits.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT hololink-operators)

# ---------------------------------------------------------------------------
# BaseReceiverOp — CRTP template base for Linux and RoCE receiver operators
#
# All template method bodies live in base_receiver_op.inl, which is included
# by each derived-class .cpp alongside its explicit instantiation.  This target
# is therefore an INTERFACE library: it carries no compiled code of its own but
# propagates include paths, schema generation, and link dependencies to every
# derived operator that links against it.
# ---------------------------------------------------------------------------
add_library(base_receiver_op INTERFACE)

add_dependencies(base_receiver_op capture_metadata_schema_generated)

add_library(hololink::operators::base_receiver_op ALIAS base_receiver_op)

target_include_directories(base_receiver_op
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<BUILD_INTERFACE:${CAPTURE_METADATA_SCHEMA_GENERATED_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(base_receiver_op
  INTERFACE
    holoscan::core
    holoscan::schema
    hololink
    hololink::sensor
    CUDA::cuda_driver
    CUDA::cudart
  )

# Installation of the base_receiver_op operator
set_target_properties(base_receiver_op PROPERTIES EXPORT_NAME operators::base_receiver)
install(TARGETS base_receiver_op
        EXPORT HololinkTargets
        COMPONENT hololink-operators)

install(FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/base_receiver_op.hpp
          ${CMAKE_CURRENT_SOURCE_DIR}/base_receiver_op.inl
          ${CMAKE_CURRENT_SOURCE_DIR}/capture_metadata.fbs
          ${CMAKE_CURRENT_SOURCE_DIR}/capture_metadata_utils.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators
        COMPONENT hololink-operators)

if(HOLOLINK_BUILD_ARGUS_ISP)
  add_subdirectory(argus_isp)
endif()
# AudioPacketizerOp is not yet ported to HSDK5 -- it still emits a legacy
# holoscan::gxf::Entity/nvidia::gxf::Tensor, which HSDK5 headers no longer provide.
# Excluded from the default build; opt in with -DHOLOLINK_BUILD_UNPORTED_OPERATORS=ON
# once it's ported.
if(HOLOLINK_BUILD_UNPORTED_OPERATORS)
  add_subdirectory(audio_packetizer)
endif()
add_subdirectory(compute_crc)
add_subdirectory(csi_to_bayer)
add_subdirectory(discard_metadata)
if(HOLOLINK_BUILD_FUSA)
  add_subdirectory(fusa_coe_capture)
endif()
add_subdirectory(image_processor)
add_subdirectory(linux_coe_receiver)
add_subdirectory(linux_receiver)
add_subdirectory(packed_format_converter)
if(HOLOLINK_BUILD_PVA)
  add_subdirectory(pva_crc)
endif()
if (HOLOLINK_BUILD_ROCE)
  add_subdirectory(roce_receiver)
  add_subdirectory(roce_transmitter)
  add_subdirectory(cpu_roce_transceiver)
  if(HOLOLINK_BUILD_GPUNETIO_ROCE)
    add_subdirectory(gpu_roce_transceiver)
  endif()
endif() # HOLOLINK_BUILD_ROCE
add_subdirectory(sub_frame_combiner)
add_subdirectory(udp_transmitter)

# The signal-generator operators draw their GUIs into hololink::ImGuiRenderer, which needs the
# HoloViz library (see HOLOLINK_HAS_IMGUI in the top-level CMakeLists.txt).
if(HOLOLINK_HAS_IMGUI)
  add_subdirectory(iq_dec)
  add_subdirectory(iq_enc)
  add_subdirectory(sig_gen)
  add_subdirectory(sig_viewer)
endif() # HOLOLINK_HAS_IMGUI

# SubFrameVisualizerOp drives the HSDK5 HoloViz library directly
# (holoscan::visualization::holoviz, from the SDK's visualization component).
if(TARGET holoscan::visualization::holoviz)
  add_subdirectory(sub_frame_visualizer)
else()
  message(STATUS "holoscan::visualization::holoviz not found -- sub_frame_visualizer (HSDK5) will be skipped")
endif()

# ---------------------------------------------------------------------------
# operator_catalog — installs the HSDK5 authoring-registry rows (see
# operator_catalog.cpp) for the operators exposed to Python. Grows as more
# operators land manual/automatic registration thunks.
# ---------------------------------------------------------------------------
add_library(operator_catalog STATIC
  operator_catalog.cpp
  )

set_property(TARGET operator_catalog PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(hololink::operators::operator_catalog ALIAS operator_catalog)

target_include_directories(operator_catalog
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

target_link_libraries(operator_catalog
  PUBLIC
    hololink::operators::csi_to_bayer
    hololink::operators::image_processor
    hololink::operators::udp_transmitter
    hololink::operators::packed_format_converter
    hololink::operators::sub_frame_combiner
    hololink::operators::discard_metadata
    hololink::operators::linux_receiver_op
    hololink::operators::linux_coe_receiver_op
    hololink::operators::compute_crc
  PRIVATE
    holoscan::core
    fmt::fmt-header-only
    CUDA::cuda_driver
  )

# FusaCoeCaptureOp's registration (operator_catalog.cpp) is compiled in only when the operator
# itself is built, matching hololink_module's own precedent for this exact flag
# (hololink_module/host/operators/CMakeLists.txt). PUBLIC (not PRIVATE): tests/operator_catalog_test.cpp
# needs to see this too, to guard its own HOLOLINK_BUILD_ROCE-only RoceReceiverOp test cases --
# a PRIVATE define doesn't propagate to a separate translation unit that merely links this
# target, so #ifdef HOLOLINK_BUILD_ROCE there would silently never be true regardless of the
# actual build configuration (caught by those tests unexpectedly passing-by-skipping in a
# HOLOLINK_BUILD_ROCE=OFF build during this operator's own development).
if(HOLOLINK_BUILD_FUSA)
  target_link_libraries(operator_catalog PUBLIC hololink::operators::fusa_coe_capture)
  target_compile_definitions(operator_catalog PUBLIC HOLOLINK_BUILD_FUSA)
endif()

# RoceReceiverOp's and RoceTransmitterOp's registrations (operator_catalog.cpp) are compiled in
# only when the operators themselves are built -- roce_receiver/roce_transmitter (unlike
# linux_receiver_op) are gated on HOLOLINK_BUILD_ROCE above (find_package(ibverbs REQUIRED), not
# available on every host). PUBLIC: see the HOLOLINK_BUILD_FUSA comment just above.
if(HOLOLINK_BUILD_ROCE)
  target_link_libraries(operator_catalog PUBLIC hololink::operators::roce_receiver
    hololink::operators::roce_transmitter)
  target_compile_definitions(operator_catalog PUBLIC HOLOLINK_BUILD_ROCE)
endif()

# ArgusIspOp's registration (operator_catalog.cpp) is compiled in only when the operator itself
# is built -- argus_isp is itself gated on HOLOLINK_BUILD_ARGUS_ISP above (add_subdirectory,
# requires libnvargus/jetson_multimedia_api, not available on every host). PUBLIC: see the
# HOLOLINK_BUILD_FUSA comment above.
if(HOLOLINK_BUILD_ARGUS_ISP)
  target_link_libraries(operator_catalog PUBLIC hololink::operators::argus_isp)
  target_compile_definitions(operator_catalog PUBLIC HOLOLINK_BUILD_ARGUS_ISP)
endif()

# The signal-generator operators' registrations, compiled in when they're built (see
# HOLOLINK_HAS_IMGUI above). PUBLIC: see the HOLOLINK_BUILD_FUSA comment above.
if(HOLOLINK_HAS_IMGUI)
  target_link_libraries(operator_catalog PUBLIC
    hololink::gui_renderer
    hololink::operators::iq_dec
    hololink::operators::iq_enc
    hololink::operators::sig_gen
    hololink::operators::sig_viewer)
  target_compile_definitions(operator_catalog PUBLIC HOLOLINK_HAS_IMGUI)
endif()

set_target_properties(operator_catalog PROPERTIES EXPORT_NAME operators::operator_catalog)
install(TARGETS operator_catalog
        EXPORT HololinkTargets
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT hololink-operators)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/operator_catalog.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hololink/operators
        COMPONENT hololink-operators)
