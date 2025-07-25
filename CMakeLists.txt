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
    src/blas/blas_dispatcher_scal.cpp
    src/blas/blas_node_axpy.cpp
    src/blas/blas_node_copy.cpp
    src/blas/blas_node_dot.cpp
    src/blas/blas_node_gemm.cpp
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
    src/transformations/einsum2blas_scal.cpp
    src/transformations/einsum2blas.cpp
)

add_library(sdfglib-einsum
    ${SOURCE_FILES}
)
target_include_directories(sdfglib-einsum
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(sdfglib-einsum PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter -Wno-unused-private-field -Wno-switch -Wno-deprecated-declarations)
target_link_libraries(sdfglib-einsum PUBLIC sdfglib::sdfglib)

set_target_properties(sdfglib-einsum PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

install(TARGETS sdfglib-einsum
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

add_subdirectory(tests)
