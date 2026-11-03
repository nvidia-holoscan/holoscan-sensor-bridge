# nanobind dependency setup.
#
# Ensures the nanobind_add_module() CMake function is available by either
# finding a pre-installed version or fetching it via FetchContent. Matches
# the version holoscan-sdk itself builds its own Python bindings with
# (public/cmake/deps/nanobind.cmake) -- not load-bearing for correctness
# (hololink's _hololink module only interacts with holoscan's own modules at
# the ordinary Python level, not via shared C++ objects across the module
# boundary), but keeping them in step avoids an unnecessary variable.

# nanobind_add_module() decides on STABLE_ABI from `TARGET Python::SABIModule`, and reads
# Python_INCLUDE_DIRS & co., in the *calling* directory scope; find_package()'s imported
# targets and result variables are directory-scoped. So Python is looked up here, before the
# include guard, in every directory that includes this file. It is cheap (the results are
# cached), and it is the only way each nanobind_add_module() caller (python/hololink,
# python/hololink/emulation, hololink_module/host: sibling subtrees, none an ancestor of
# another) sees the targets. With the lookup inside the guard, whichever subtree included
# this file second built a CPython-tagged module instead of a stable-ABI one.
# Development.SABIModule is unknown to CMake before 3.26 (an unknown component is an error),
# and the Emulator-only build accepts CMake 3.18: ask for it only where it exists.
set(_hololink_python_sabi_component)
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.26)
  set(_hololink_python_sabi_component OPTIONAL_COMPONENTS Development.SABIModule)
endif()
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
  ${_hololink_python_sabi_component})
unset(_hololink_python_sabi_component)

# Development.SABIModule (and the Python::SABIModule target) is new in CMake 3.26; the full
# build requires only 3.24 and the Emulator-only build 3.18. Without the target
# nanobind_add_module(... STABLE_ABI)
# silently builds a module for this interpreter (_x.cpython-3NN-*.so) instead of a
# stable-ABI one (_x.abi3.so). Say so once, so a CPython-tagged module is not a surprise.
if(NOT TARGET Python::SABIModule)
  get_property(_hololink_nanobind_sabi_noted GLOBAL PROPERTY HOLOLINK_NANOBIND_SABI_NOTED)
  if(NOT _hololink_nanobind_sabi_noted)
    set_property(GLOBAL PROPERTY HOLOLINK_NANOBIND_SABI_NOTED TRUE)
    if(CMAKE_VERSION VERSION_LESS 3.26)
      message(STATUS "nanobind: CMake ${CMAKE_VERSION} has no Development.SABIModule "
        "(added in 3.26); the Python modules are built for Python ${Python_VERSION} "
        "only, not for the stable ABI")
    elseif(Python_VERSION VERSION_LESS 3.12)
      message(STATUS "nanobind: Python ${Python_VERSION} is older than 3.12; the Python "
        "modules are built for this interpreter only, not for the stable ABI")
    else()
      message(STATUS "nanobind: Python::SABIModule not found for Python ${Python_VERSION}; "
        "the Python modules are built for this interpreter only, not for the stable ABI")
    endif()
  endif()
endif()

include_guard(GLOBAL)
set(HOLOLINK_NANOBIND_VERSION "3.0.1")

# Already exists?
if(COMMAND nanobind_add_module)
  return()
endif()

# nanobind's own CMakeLists.txt only runs its internal find_package(Python...)
# when neither Python::Module nor Python::Interpreter already exists (see its
# `if (NOT TARGET Python::Module AND NOT TARGET Python::Interpreter)` guard),
# and otherwise silently relies on whatever `Python_INCLUDE_DIRS` happens to be
# set in its parent scope. This repo's own Python3 (not Python) detection
# doesn't satisfy that guard, so nanobind-static ends up compiling without
# Python.h on its include path. Do nanobind's own recommended find_package()
# call ourselves, before fetching it, so the guard is satisfied with correctly
# populated results.

# Find via cmake config
find_package(nanobind ${HOLOLINK_NANOBIND_VERSION} QUIET CONFIG)
if(nanobind_FOUND)
  message(STATUS "Found nanobind: (version ${HOLOLINK_NANOBIND_VERSION})")
  return()
endif()

# Fetch
message(STATUS "nanobind: Fetching v${HOLOLINK_NANOBIND_VERSION}")
set(NB_CREATE_INSTALL_RULES OFF CACHE BOOL "" FORCE)
set(NB_TEST OFF CACHE BOOL "" FORCE)
include(FetchContent)
FetchContent_Declare(nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind
  GIT_TAG v${HOLOLINK_NANOBIND_VERSION}
  GIT_SHALLOW TRUE
  GIT_SUBMODULES_RECURSE TRUE
)
FetchContent_MakeAvailable(nanobind)
