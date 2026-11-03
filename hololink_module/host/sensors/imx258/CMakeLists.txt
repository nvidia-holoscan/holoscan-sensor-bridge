# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# V1-native IMX258 camera driver (the sensor on the CertusPro-NX Versa PCIe board).
# Public surface is:
#   hololink::module::sensors::imx258::Imx258Cam
# Constructed against a shared_ptr<HololinkInterfaceV1> and uses the
# module's V1 I2C / I2cLock surface for register I/O. The IMX258
# register table is vendored (byte-for-byte, from the legacy
# camera_hdmi_setup.py) into imx258_mode.hpp and the CSI
# converter is the module-native CsiConverterV1, so this driver links
# no legacy hololink::core / hololink::sensors::*. Structure and
# install/export pattern mirror ../imx274/CMakeLists.txt.

add_library(hololink_module_sensors_imx258 STATIC
    imx258_cam.cpp)
add_library(hololink::module::sensors::imx258 ALIAS
    hololink_module_sensors_imx258)

target_compile_features(hololink_module_sensors_imx258 PUBLIC cxx_std_17)
target_include_directories(hololink_module_sensors_imx258
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(hololink_module_sensors_imx258
    PUBLIC
        hololink::module_headers)

set_target_properties(hololink_module_sensors_imx258 PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON)

# Export the driver for external applications: find_package(Hololink) then
# target_link_libraries(app hololink::module::sensors::imx258). Joins the
# project-wide HololinkTargets set (cmake/HololinkExport.cmake), namespaced
# the same way as the legacy operator/sensor targets. The public headers
# install under include/hololink/module/sensors/imx258/, matching the
# namespace-mirroring layout consumers #include.
include(GNUInstallDirs)
set_target_properties(hololink_module_sensors_imx258 PROPERTIES
    EXPORT_NAME module::sensors::imx258)
install(TARGETS hololink_module_sensors_imx258
    EXPORT HololinkTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT hololink-module)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/hololink
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT hololink-module)
