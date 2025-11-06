# pybind11 dependency setup.
#
# Ensures the pybind11::pybind11 target is available by either finding a
# pre-installed version or fetching it via FetchContent.

include_guard(GLOBAL)
set(HOLOLINK_PYBIND11_VERSION "2.13.6")

# Already exists?
if(TARGET pybind11::pybind11)
  return()
endif()

# Find via cmake config
find_package(pybind11 ${HOLOLINK_PYBIND11_VERSION} QUIET CONFIG)
if(pybind11_FOUND)
  message(STATUS "Found pybind11: (version ${HOLOLINK_PYBIND11_VERSION})")
  return()
endif()

# Fetch
message(STATUS "pybind11: Fetching v${HOLOLINK_PYBIND11_VERSION}")
include(FetchContent)
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v${HOLOLINK_PYBIND11_VERSION}
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(pybind11)
