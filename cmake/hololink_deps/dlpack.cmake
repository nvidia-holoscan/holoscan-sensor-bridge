# dlpack dependency setup (header-only).
#
# Ensures the dlpack::dlpack target is available by either finding a pre-installed
# version (config or path) or downloading it as a fallback.

include_guard(GLOBAL)
set(HOLOLINK_DLPACK_VERSION "1.0")

# Already exists?
if(TARGET dlpack::dlpack)
  return()
endif()

# Find with cmake config
find_package(dlpack ${HOLOLINK_DLPACK_VERSION} QUIET CONFIG)
if(dlpack_FOUND)
  message(STATUS "Found dlpack: (version ${HOLOLINK_DLPACK_VERSION})")
  return()
endif()

# Find via path discovery
set(_dlpack_known_include_suffixes
  include
  include/3rdparty
)
set(_dlpack_hint_paths
  /opt/nvidia/holoscan
  ${CMAKE_PREFIX_PATH}
  $ENV{CMAKE_PREFIX_PATH}
  /usr
  /usr/local
)
unset(DLPACK_INCLUDE_DIR CACHE)
find_path(DLPACK_INCLUDE_DIR
  NAMES dlpack/dlpack.h
  PATHS ${_dlpack_hint_paths}
  PATH_SUFFIXES ${_dlpack_known_include_suffixes}
  DOC "Path to directory containing dlpack/dlpack.h"
)

# Create target if found via path
if(DLPACK_INCLUDE_DIR)
  message(STATUS "Found dlpack: ${DLPACK_INCLUDE_DIR}")
  add_library(dlpack::dlpack INTERFACE IMPORTED)
  set_target_properties(dlpack::dlpack PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${DLPACK_INCLUDE_DIR}"
  )
  return()
endif()

# Fallback: download single header into build tree
set(_dlpack_build_inc "${CMAKE_BINARY_DIR}/_deps/dlpack/include")
set(_dlpack_header_dir "${_dlpack_build_inc}/dlpack")
set(_dlpack_header_path "${_dlpack_header_dir}/dlpack.h")
set(_dlpack_url "https://raw.githubusercontent.com/dmlc/dlpack/v${HOLOLINK_DLPACK_VERSION}/include/dlpack/dlpack.h")
message(STATUS "dlpack: Downloading v${HOLOLINK_DLPACK_VERSION} → ${_dlpack_header_path}")
file(MAKE_DIRECTORY "${_dlpack_header_dir}")
file(DOWNLOAD "${_dlpack_url}" "${_dlpack_header_path}"
      SHOW_PROGRESS STATUS _dlpack_dl_status)
list(GET _dlpack_dl_status 0 _dlpack_status_code)

# Create target if download was successful
if(_dlpack_status_code EQUAL 0 AND EXISTS "${_dlpack_header_path}")
  add_library(dlpack::dlpack INTERFACE IMPORTED)
  set_target_properties(dlpack::dlpack PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_dlpack_build_inc}"
  )
  return()
endif()

message(FATAL_ERROR "dlpack not found and download failed. Provide a dlpack config package, install Holoscan, set CMAKE_PREFIX_PATH, or allow header download.")