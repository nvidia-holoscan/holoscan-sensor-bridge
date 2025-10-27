# Package config export
include(CMakePackageConfigHelpers)
set(HOLOLINK_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/hololink")

install(EXPORT HololinkTargets
  NAMESPACE hololink::
  FILE HololinkTargets.cmake
  DESTINATION ${HOLOLINK_INSTALL_CMAKEDIR}
)

configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/HololinkConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/HololinkConfig.cmake
  INSTALL_DESTINATION ${HOLOLINK_INSTALL_CMAKEDIR}
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/HololinkConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/HololinkConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/HololinkConfigVersion.cmake
  DESTINATION ${HOLOLINK_INSTALL_CMAKEDIR}
)
