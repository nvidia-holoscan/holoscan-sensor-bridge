# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# V1-native VB1940 camera driver. Public surface is:
#   hololink::module::sensors::vb1940::Vb1940Cam
# Constructed against a shared_ptr<HololinkInterfaceV1> and a
# shared_ptr<OscillatorInterfaceV1> (the Leopard supplement publishes
# the Bajoran TS2-backed oscillator). Uses the module's V1 I2C /
# I2cLock surface for register I/O. The VB1940 register tables + firmware
# blobs are vendored (byte-for-byte) into vb1940_cam.cpp and the CSI
# converter is the module-native CsiConverterV1, so this driver links no
# legacy hololink::core / hololink::sensors::*.

add_library(hololink_module_sensors_vb1940 STATIC
    vb1940_cam.cpp)
add_library(hololink::module::sensors::vb1940 ALIAS
    hololink_module_sensors_vb1940)

target_compile_features(hololink_module_sensors_vb1940 PUBLIC cxx_std_17)
target_include_directories(hololink_module_sensors_vb1940
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(hololink_module_sensors_vb1940
    PUBLIC
        hololink::module_headers
        # vb1940_cam.hpp #includes the LeopardVb1940InterfaceV1
        # declaration (the supplement's public header), so the path
        # must propagate to consumers (examples, tests).
        hololink::leopard_vb1940::headers)

set_target_properties(hololink_module_sensors_vb1940 PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON)

# Export the driver for external applications: find_package(Hololink) then
# target_link_libraries(app hololink::module::sensors::vb1940). The exported
# interface pulls hololink::leopard_vb1940::headers (the supplement's public
# LeopardVb1940InterfaceV1 declaration), which is itself added to
# HololinkTargets in module/leopard_vb1940/CMakeLists.txt.
include(GNUInstallDirs)
set_target_properties(hololink_module_sensors_vb1940 PROPERTIES
    EXPORT_NAME module::sensors::vb1940)
install(TARGETS hololink_module_sensors_vb1940
    EXPORT HololinkTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT hololink-module)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/hololink
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT hololink-module)

# Python bindings: Vb1940Cam is now bound in python/hololink_module/_hololink_module.cpp,
# consolidated with the other hololink_module pybind11 extensions into one nanobind translation
# unit -- see hololink_module/host/CMakeLists.txt's own Python comment for where that target is
# defined and why. hololink_module.sensors.vb1940.Vb1940Cam still works the same way from the
# application's side; only the extension layout changed.
if(HOLOLINK_BUILD_PYTHON)
    # The compiled extension itself (hololink_module_python target, linking this directory's
    # hololink_module_sensors_vb1940 library) is defined in hololink_module/host/CMakeLists.txt;
    # this just stages this sub-package's own __init__.py next to it, preserving the old
    # per-extension output layout exactly.
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/python/__init__.py"
        "${CMAKE_BINARY_DIR}/python/hololink_module/sensors/vb1940/__init__.py"
        COPYONLY)
endif()
