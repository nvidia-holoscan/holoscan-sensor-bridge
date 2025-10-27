# fmt dependency setup (header-only).
#
# Ensures the fmt::fmt-header-only target is available by finding a pre-installed
# version (either via CMake config or header discovery).

include_guard(GLOBAL)
set(HOLOLINK_FMT_VERSION "10.1.1")

# Already exists?
if(TARGET fmt::fmt-header-only)
  return()
endif()

# Try CMake config package
find_package(fmt ${HOLOLINK_FMT_VERSION} QUIET CONFIG)
if(fmt_FOUND)
  message(STATUS "Found fmt: (version ${HOLOLINK_FMT_VERSION})")
  return()
endif()

# Find via path discovery
set(_fmt_hint_paths
  /opt/nvidia/holoscan
  ${CMAKE_PREFIX_PATH}
  $ENV{CMAKE_PREFIX_PATH}
  /usr
  /usr/local
)
unset(FMT_INCLUDE_DIR CACHE)
find_path(FMT_INCLUDE_DIR
  NAMES fmt/format.h
  PATHS ${_fmt_hint_paths}
  PATH_SUFFIXES include
  DOC "Path to directory containing fmt/format.h"
)

# Create imported interface target from path if found
if(FMT_INCLUDE_DIR)
  message(STATUS "Found fmt: ${FMT_INCLUDE_DIR}")
  add_library(fmt::fmt-header-only INTERFACE IMPORTED)
  set_target_properties(fmt::fmt-header-only PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${FMT_INCLUDE_DIR}"
    INTERFACE_COMPILE_DEFINITIONS "FMT_HEADER_ONLY=1"
  )
  return()
endif()

# Fetch
message(STATUS "fmt: Fetching v${HOLOLINK_FMT_VERSION}")
include(FetchContent)
FetchContent_Declare(
  fmt_src
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        ${HOLOLINK_FMT_VERSION}
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(fmt_src)
