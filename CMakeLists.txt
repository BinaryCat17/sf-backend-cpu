cmake_minimum_required(VERSION 3.25)
project(sf-backend-cpu LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Dependencies
if(NOT TARGET isa)
    find_package(sf-spec REQUIRED)
endif()

# Modules
add_subdirectory(ops)
add_subdirectory(backend_cpu)

# --- Installation & Export ---
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ops backend_cpu
    EXPORT SionFlowBackendCpuTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

export(EXPORT SionFlowBackendCpuTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/SionFlowBackendCpuTargets.cmake"
    NAMESPACE SionFlow::
)

install(DIRECTORY backend_cpu/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ops/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT SionFlowBackendCpuTargets
    FILE SionFlowBackendCpuTargets.cmake
    NAMESPACE SionFlow::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-backend-cpu
)

configure_package_config_file(
    cmake/sf-backend-cpu-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sf-backend-cpu-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-backend-cpu
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sf-backend-cpu-config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sf-backend-cpu
)
