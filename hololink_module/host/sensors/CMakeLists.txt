# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(imx274)
# The IMX258 is the sensor on the CertusPro-NX Versa PCIe board; built with the PCIe transport.
if(HOLOLINK_BUILD_PCIE)
  add_subdirectory(imx258)
endif()
add_subdirectory(vb1940)
add_subdirectory(ar0234)
add_subdirectory(deserializers)
add_subdirectory(serializers)
add_subdirectory(hawk)
