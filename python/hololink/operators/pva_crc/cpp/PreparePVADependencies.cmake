# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prepare PVA CRC artifacts: download library if missing, then verify.
# Invoked from CMakeLists.txt with -D PVA_CRC_LIB_DEST=... -D CUPVA_ALLOWLIST_DEST=...
# and -D PVA_CRC_LIB_URL=... (optional; used to download library when missing).
# Allowlist is not downloaded here; deploy on host per README.

# Download library if missing
if(NOT EXISTS "${PVA_CRC_LIB_DEST}" AND DEFINED PVA_CRC_LIB_URL)
  get_filename_component(LIB_DIR "${PVA_CRC_LIB_DEST}" DIRECTORY)
  file(MAKE_DIRECTORY "${LIB_DIR}")
  message(STATUS "Downloading PVA CRC library to ${PVA_CRC_LIB_DEST}")
  file(DOWNLOAD "${PVA_CRC_LIB_URL}" "${PVA_CRC_LIB_DEST}" STATUS LIB_STATUS SHOW_PROGRESS)
  list(GET LIB_STATUS 0 LIB_STATUS_CODE)
  if(NOT LIB_STATUS_CODE EQUAL 0)
    message(FATAL_ERROR "Failed to download PVA CRC library from ${PVA_CRC_LIB_URL}")
  endif()
endif()

if(NOT EXISTS "${PVA_CRC_LIB_DEST}")
  get_filename_component(LIB_NAME "${PVA_CRC_LIB_DEST}" NAME)
  message(FATAL_ERROR
    "PVA CRC library not found: ${PVA_CRC_LIB_DEST}\n"
    "Expected library ${LIB_NAME}. Ensure PVA_CRC_LIB_URL is set for download.")
endif()

get_filename_component(LIB_NAME "${PVA_CRC_LIB_DEST}" NAME)
message(STATUS "Using ${LIB_NAME} from build directory")

# Allowlist is not required for build; deploy on host (see README).
if(DEFINED CUPVA_ALLOWLIST_DEST AND EXISTS "${CUPVA_ALLOWLIST_DEST}")
  get_filename_component(ALLOWLIST_NAME "${CUPVA_ALLOWLIST_DEST}" NAME)
  message(STATUS "Using ${ALLOWLIST_NAME} from build directory")
endif()
