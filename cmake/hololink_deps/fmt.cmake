# fmt dependency setup (header-only).
#
# Ensures the fmt::fmt-header-only target is available by finding a pre-installed
# version (either via CMake config or header discovery).

include_guard(GLOBAL)

# in case the include is moved before finding HSDK, try to find it, but not required and no need to configure
if (NOT holoscan_FOUND)
  find_package(holoscan 4.0 PATHS "/opt/nvidia/holoscan")
endif()

# try to find HSDK's version first.
find_package(fmt 11 QUIET)
if (NOT fmt_FOUND) 
  # fallback to the system version
  find_package(fmt 8 QUIET)
endif()

if (NOT fmt_FOUND)
  message(FATAL_ERROR "fmt targets not found. Make sure either HSDK 4.0+ or libfmt-dev is installed in environment")
else()
  message(STATUS "Found fmt: ${fmt_DIR} (found version \"${fmt_VERSION}\")")
endif()