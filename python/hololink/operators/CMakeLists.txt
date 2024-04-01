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

# Define and create a directory to store python operators under the hololink python module
set(HOLOLINK_PYTHON_MODULE_OUT_DIR ${CMAKE_BINARY_DIR}/python/${CMAKE_INSTALL_LIBDIR}/hololink/operators)
file(MAKE_DIRECTORY ${HOLOLINK_PYTHON_MODULE_OUT_DIR})

# include the operators
add_subdirectory(csi_to_bayer)
add_subdirectory(image_processor)
add_subdirectory(gamma_correction)
add_subdirectory(linux_receiver)
add_subdirectory(roce_receiver)

# install
install(
    DIRECTORY "${HOLOLINK_PYTHON_MODULE_OUT_DIR}"
    DESTINATION "."
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    DIRECTORY_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT "hololink-python_libs"
    PATTERN "__pycache__" EXCLUDE
)
