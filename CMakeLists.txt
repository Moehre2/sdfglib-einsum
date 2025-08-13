cmake_minimum_required(VERSION 3.16)

project(sdfglib-einsum VERSION 0.0.1 DESCRIPTION "Einsum Plugin for SDFG library in C++")
include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT TARGET sdfglib::sdfglib)
    find_package(sdfglib CONFIG REQUIRED)
endif()

set(SOURCE_FILES
    src/blas/blas_dispatcher_axpy.cpp
    src/blas/blas_dispatcher_copy.cpp
    src/blas/blas_dispatcher_dot.cpp
    src/blas/blas_dispatcher_gemm.cpp
    src/blas/blas_dispatcher_gemv.cpp
    src/blas/blas_dispatcher_scal.cpp
    src/blas/blas_node_axpy.cpp
    src/blas/blas_node_copy.cpp
    src/blas/blas_node_dot.cpp
    src/blas/blas_node_gemm.cpp
    src/blas/blas_node_gemv.cpp
    src/blas/blas_node_scal.cpp
    src/blas/blas_node.cpp
    src/einsum/einsum_dispatcher.cpp
    src/einsum/einsum_node.cpp
    src/einsum/einsum_serializer.cpp
    src/transformations/einsum_expand.cpp
    src/transformations/einsum_lift.cpp
    src/transformations/einsum2blas_axpy.cpp
    src/transformations/einsum2blas_copy.cpp
    src/transformations/einsum2blas_dot.cpp
    src/transformations/einsum2blas_gemm.cpp
    src/transformations/einsum2blas_gemv.cpp
    src/transformations/einsum2blas_scal.cpp
    src/transformations/einsum2blas.cpp
)

add_library(sdfglib-einsum
    ${SOURCE_FILES}
)
target_include_directories(sdfglib-einsum
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_options(sdfglib-einsum PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -Wno-unused-private-field -Wno-switch -Wno-deprecated-declarations)
target_link_libraries(sdfglib-einsum PUBLIC sdfglib::sdfglib)

set_target_properties(sdfglib-einsum PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

## INSTALLATION

install(TARGETS sdfglib-einsum
    EXPORT sdfglibEinsumTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/sdfg DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

## CMAKE CONFIGURATION

install(EXPORT sdfglibEinsumTargets
    NAMESPACE sdfglib::
    FILE sdfglibEinsumTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfglib
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibEinsumConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/sdfglibEinsumConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibEinsumConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfglib
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibEinsumConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/sdfglibEinsumConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sdfglib
)

add_subdirectory(tests)
