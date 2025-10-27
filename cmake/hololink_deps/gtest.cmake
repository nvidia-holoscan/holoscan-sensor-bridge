# GoogleTest dependency setup.
#
# Ensures the GTest::gtest target is available by either finding a pre-installed
# version or fetching it via FetchContent.

include_guard(GLOBAL)
set(HOLOLINK_GTEST_VERSION "1.17.0")

# Already exists?
if(TARGET GTest::gtest)
  return()
endif()

# Find via cmake config
find_package(GTest ${HOLOLINK_GTEST_VERSION} QUIET CONFIG)
if(GTest_FOUND)
  message(STATUS "Found GTest: (version ${HOLOLINK_GTEST_VERSION})")
  return()
endif()

# Fetch
message(STATUS "GTest: Fetching v${HOLOLINK_GTEST_VERSION}")
include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v${HOLOLINK_GTEST_VERSION}
  GIT_SHALLOW TRUE
)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
