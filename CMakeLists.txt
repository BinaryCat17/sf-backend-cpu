cmake_minimum_required(VERSION 3.15)
project(sf-backend-cpu LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(sf-spec REQUIRED)

add_subdirectory(ops)
add_subdirectory(backend_cpu)

# --- Installation & Export ---

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install targets
set(INSTALL_TARGETS ops backend_cpu)

install(TARGETS ${INSTALL_TARGETS}
    EXPORT SionFlowBackendCPUTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY ops/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY backend_cpu/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Generate and install the export file
install(EXPORT SionFlowBackendCPUTargets
    FILE SionFlowBackendCPUTargets.cmake
    NAMESPACE SionFlow::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-backend-cpu
)

# Create the config file
configure_package_config_file(
    cmake/sf-backend-cpu-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sf-backend-cpu-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-backend-cpu
)

# Install the config file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sf-backend-cpu-config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-backend-cpu
)