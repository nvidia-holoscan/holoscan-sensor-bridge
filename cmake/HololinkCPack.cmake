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


# CPACK config
set(CPACK_PACKAGE_NAME hololink CACHE STRING "Holoscan Sensor Bridge")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Holoscan Sensor Bridge"
    CACHE STRING "Package description for Holoscan Sensor Bridge"
)
set(CPACK_PACKAGE_VENDOR "NVIDIA")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})

set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/nvidia/hololink")

set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Julien Jomier")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

# Sets the package name as debian format
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

# Just one package will all the components
set(CPACK_DEB_COMPONENT_INSTALL 1)
set(CPACK_COMPONENTS_GROUPING ONE_PER_GROUP)

set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "")

set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS 1)

set(CPACK_DEBIAN_CORE_PACKAGE_DEPENDS "libfmt-dev" )
set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "hololink-core" )
set(CPACK_DEBIAN_ROCE_PACKAGE_DEPENDS "hololink-core ibverbs" )

include(CPack)

# Add the components for hololink-core
cpack_add_component(hololink-core GROUP core)
cpack_add_component(hololink-tools GROUP core)

# Add the components for hsb-dev
cpack_add_component(hololink-common GROUP dev)
cpack_add_component(hololink-examples GROUP dev)
cpack_add_component(hololink-scripts GROUP dev)
cpack_add_component(hololink-operators GROUP dev)
cpack_add_component(hololink-python_libs GROUP dev)
cpack_add_component(hololink-python_tools GROUP dev)

# Add the components for hsb-roce
cpack_add_component(hololink-roce-operators GROUP roce)
