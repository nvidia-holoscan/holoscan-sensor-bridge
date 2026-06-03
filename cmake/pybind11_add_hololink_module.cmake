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

# Find python-dev and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include(hololink_deps/pybind11)

# Helper function to copy pure Python files to the module output directory
# Usage:
#   copy_hololink_python_files(
#       TARGET_NAME <name>           # Name for the custom target (required)
#       DESTINATION <subdir>         # Relative build subdirectory (e.g. "sensors/vb1940", or empty for module root)
#       FILES <file1> <file2> ...    # List of Python files to copy. Note that all files are placed in the same
#                                    # output destination folder, without respect for input subfolder structures.
#   )
function(copy_hololink_python_files)
    cmake_parse_arguments(COPY                  # PREFIX
        ""                                      # OPTIONS
        "TARGET_NAME;DESTINATION"               # ONEVAL
        "FILES"                                 # MULTIVAL
        ${ARGN}
    )

    if(NOT COPY_TARGET_NAME)
        message(FATAL_ERROR "copy_hololink_python_files: TARGET_NAME is required")
    endif()
    if(NOT COPY_FILES)
        message(FATAL_ERROR "copy_hololink_python_files: FILES is required")
    endif()

    # Build destination path, handling empty DESTINATION for root-level files
    if(COPY_DESTINATION)
        # Make COPY_DESTINATION absolute relative to HOLOLINK_PYTHON_MODULE_OUT_DIR
        cmake_path(ABSOLUTE_PATH COPY_DESTINATION BASE_DIRECTORY "${HOLOLINK_PYTHON_MODULE_OUT_DIR}" OUTPUT_VARIABLE _dest_dir)
    else()
        set(_dest_dir "${HOLOLINK_PYTHON_MODULE_OUT_DIR}")
    endif()
    # Ensure destination directory hierarchy exists
    file(MAKE_DIRECTORY "${_dest_dir}")

    set(_output_files)
    foreach(_source_file ${COPY_FILES})
        # Convert to absolute path (handles both absolute and relative paths)
        set(_source_path "${_source_file}")
        cmake_path(ABSOLUTE_PATH _source_path BASE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

        # Extract just the filename for the destination
        cmake_path(GET _source_file FILENAME _filename)
        set(_output_file "${_dest_dir}/${_filename}")

        list(APPEND _output_files "${_output_file}")
        add_custom_command(
            OUTPUT "${_output_file}"
            COMMAND ${CMAKE_COMMAND} -E copy
                "${_source_path}" "${_output_file}"
            DEPENDS "${_source_path}"
            COMMENT "Copying ${_filename} to ${_output_file}"
        )
    endforeach()

    add_custom_target(${COPY_TARGET_NAME} ALL
        DEPENDS ${_output_files}
    )
    add_dependencies(hololink_python_all ${COPY_TARGET_NAME})
endfunction()

# Helper function to generate pybind11 operator modules
function(pybind11_add_hololink_module)
    cmake_parse_arguments(MODULE                # PREFIX
        ""                                      # OPTIONS
        "CPP_CMAKE_TARGET;CLASS_NAME;IMPORT;NAME"    # ONEVAL
        "SOURCES"                               # MULTIVAL
        ${ARGN}
    )

    if(NOT MODULE_NAME)
        set(MODULE_NAME ${MODULE_CPP_CMAKE_TARGET})
    endif()

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

    # Check if this module depends on hololink_core and if hololink_core_python exists
    # If so, configure it to use symbols from _hololink_core.so at runtime instead of embedding them
    if(TARGET hololink_core_python AND NOT MODULE_NAME STREQUAL "hololink_core")
        get_target_property(link_libs ${MODULE_CPP_CMAKE_TARGET} LINK_LIBRARIES)
        if(link_libs)
            # Check if hololink_core or hololink::core is in the dependencies (direct or transitive)
            list(FIND link_libs "hololink_core" has_hololink_core)
            list(FIND link_libs "hololink::core" has_hololink_core_alias)
            if(NOT ${has_hololink_core} EQUAL -1 OR NOT ${has_hololink_core_alias} EQUAL -1)
                # The module will still link against the static hololink_core library,
                # but at runtime, Python will load _hololink_core.so with RTLD_GLOBAL first,
                # making its symbols take precedence over the statically linked ones.
                # This ensures only one copy of the symbols is active in memory.
                
                # Add RPATH to find hololink_core module at runtime
                file(RELATIVE_PATH core_relative_path
                    ${CMAKE_CURRENT_LIST_DIR}
                    ${hololink_SOURCE_DIR}/python/hololink/hololink_core
                )
                set_property(TARGET ${target_name}
                    APPEND PROPERTY BUILD_RPATH "\$ORIGIN/${core_relative_path}"
                )
                set_property(TARGET ${target_name}
                    APPEND PROPERTY INSTALL_RPATH "\$ORIGIN/${core_relative_path}"
                )
                message(STATUS "${target_name} will use hololink_core symbols from _hololink_core.so at runtime (via RTLD_GLOBAL)")
            endif()
        endif()
    endif()

    # Sets the rpath of the module
    file(RELATIVE_PATH install_lib_relative_path
        ${CMAKE_CURRENT_LIST_DIR}
        ${hololink_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}
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

    file(RELATIVE_PATH module_relative_path
        ${hololink_SOURCE_DIR}/python/hololink
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

    add_dependencies(hololink_python_all ${target_name})

endfunction()
