# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# V1-native IMX274 camera driver. Public surface is:
#   hololink::module::sensors::imx274::Imx274Cam
# Constructed against a shared_ptr<HololinkInterfaceV1> and uses the
# module's V1 I2C / I2cLock surface for register I/O. The IMX274
# register tables are vendored (byte-for-byte) into imx274_mode.hpp and
# the CSI converter is the module-native CsiConverterV1, so this driver
# links no legacy hololink::core / hololink::sensors::*.

add_library(hololink_module_sensors_imx274 STATIC
    imx274_cam.cpp)
add_library(hololink::module::sensors::imx274 ALIAS
    hololink_module_sensors_imx274)

target_compile_features(hololink_module_sensors_imx274 PUBLIC cxx_std_17)
target_include_directories(hololink_module_sensors_imx274
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(hololink_module_sensors_imx274
    PUBLIC
        hololink::module_headers)

set_target_properties(hololink_module_sensors_imx274 PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON)

# Export the driver for external applications: find_package(Hololink) then
# target_link_libraries(app hololink::module::sensors::imx274). Joins the
# project-wide HololinkTargets set (cmake/HololinkExport.cmake), namespaced
# the same way as the legacy operator/sensor targets. The public headers
# install under include/hololink/module/sensors/imx274/, matching the
# namespace-mirroring layout consumers #include.
include(GNUInstallDirs)
set_target_properties(hololink_module_sensors_imx274 PROPERTIES
    EXPORT_NAME module::sensors::imx274)
install(TARGETS hololink_module_sensors_imx274
    EXPORT HololinkTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT hololink-module)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/hololink
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT hololink-module)
