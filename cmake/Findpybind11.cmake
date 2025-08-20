# Pybind11 dependency setup.

# We need the same version as the Holoscan SDK which is not
# necessarily available on all the platforms.
set(HOLOLINK_PYBIND11_VERSION 2.13.6)

# Already exists?
if(TARGET pybind11)
  set(pybind11_FOUND TRUE)
  return()
endif()

# Find in path
unset(pybind11_FOUND CACHE)
find_package(pybind11 ${HOLOLINK_PYBIND11_VERSION} QUIET CONFIG)
if(pybind11_FOUND)
  return()
endif()

# Fetch
include(FetchContent)
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v${HOLOLINK_PYBIND11_VERSION}
  GIT_SHALLOW TRUE
)

# Build
FetchContent_MakeAvailable(pybind11)

set(pybind11_FOUND TRUE)