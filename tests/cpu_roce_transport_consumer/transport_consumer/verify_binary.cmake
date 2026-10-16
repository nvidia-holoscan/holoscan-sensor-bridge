# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT BINARY OR NOT EXISTS "${BINARY}")
  message(FATAL_ERROR "Transport consumer binary was not built: ${BINARY}")
endif()

if(READELF AND EXISTS "${READELF}")
  set(_readelf_executable "${READELF}")
else()
  find_program(_readelf_executable NAMES readelf llvm-readelf)
endif()
if(NOT _readelf_executable)
  message(FATAL_ERROR
    "readelf or llvm-readelf is required to inspect ${BINARY} dependencies")
endif()

execute_process(
  COMMAND "${_readelf_executable}" -d "${BINARY}"
  RESULT_VARIABLE readelf_result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE readelf_error
)
if(NOT readelf_result EQUAL 0)
  message(FATAL_ERROR "readelf failed for ${BINARY}: ${readelf_error}")
endif()
if(dynamic_section MATCHES
   "NEEDED[^\\n]*(libcuda|libholoscan|libhololink|hololink_module)")
  message(FATAL_ERROR
    "Transport consumer acquired a forbidden DT_NEEDED dependency:\n"
    "${dynamic_section}")
endif()
