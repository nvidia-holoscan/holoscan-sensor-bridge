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

# Helper function to generate pybind11 operator modules
function(pybind11_add_hololink_module)
    cmake_parse_arguments(MODULE                # PREFIX
        ""                                      # OPTIONS
        "CPP_CMAKE_TARGET;CLASS_NAME;IMPORT"    # ONEVAL
        "SOURCES"                               # MULTIVAL
        ${ARGN}
    )

    set(MODULE_NAME ${MODULE_CPP_CMAKE_TARGET})
    set(target_name ${MODULE_NAME}_python)
    pybind11_add_module(${target_name} MODULE ${MODULE_SOURCES})

    target_include_directories(${target_name}
        PUBLIC ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/pydoc
    )

    target_link_libraries(${target_name}
        PRIVATE
            holoscan::core
            holoscan::pybind11
            ${MODULE_CPP_CMAKE_TARGET}
    )

    # Sets the rpath of the module
    file(RELATIVE_PATH install_lib_relative_path
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}
    )
    list(APPEND _rpath
        "\$ORIGIN/${install_lib_relative_path}" # in our install tree (same layout as src)
        "\$ORIGIN/../lib" # in our python wheel"
    )
    list(JOIN _rpath ":" _rpath)
    set_property(TARGET ${target_name}
        APPEND PROPERTY BUILD_RPATH ${_rpath}
    )
    unset(_rpath)

    # Define and create a directory to store python bindings the hololink python module
    file(RELATIVE_PATH module_relative_path
        ${CMAKE_SOURCE_DIR}/python/hololink
        ${CMAKE_CURRENT_LIST_DIR}
    )
    set(module_out_dir ${HOLOLINK_PYTHON_MODULE_OUT_DIR}/${module_relative_path})
    file(MAKE_DIRECTORY ${module_out_dir})

    # if there is no __init__.py, generate one
    if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/__init__.py")
        # custom target to ensure the module's __init__.py file is copied
        configure_file(
            ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/pybind11/__init__.py
            ${module_out_dir}/__init__.py
        )
    else()
       configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/__init__.py"
        ${module_out_dir}/__init__.py
       )
    endif()

    # Note: OUTPUT_NAME filename (_${MODULE_NAME}) must match the module name in the PYBIND11_MODULE macro
    set_target_properties(${target_name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${module_out_dir}
        OUTPUT_NAME _${MODULE_NAME}
    )

endfunction()
