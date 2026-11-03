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

include_guard(GLOBAL)

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

        # Also stage at configure time so the file exists before any
        # build step runs. Without this, pybind11_add_hololink_module's
        # `configure_file` template (also called at configure time) wins
        # the race for `__init__.py` because the build-time custom_command
        # above only re-runs when its OUTPUT is older than DEPENDS — and
        # the freshly-templated OUTPUT is already newer than the source.
        configure_file("${_source_path}" "${_output_file}" COPYONLY)

        # Mark `__init__.py` destinations so pybind11_add_hololink_module
        # knows another caller has claimed staging for this directory and
        # skips its fallback template generation.
        if(_filename STREQUAL "__init__.py")
            set_property(GLOBAL PROPERTY
                "HOLOLINK_INIT_PY_STAGED_AT::${_dest_dir}" 1)
        endif()
    endforeach()

    add_custom_target(${COPY_TARGET_NAME} ALL
        DEPENDS ${_output_files}
    )
    add_dependencies(hololink_python_all ${COPY_TARGET_NAME})
endfunction()
