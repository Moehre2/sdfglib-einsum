include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_FILES
    blas/blas_dispatcher_axpy_test.cpp
    blas/blas_dispatcher_copy_test.cpp
    blas/blas_dispatcher_dot_test.cpp
    blas/blas_dispatcher_gemm_test.cpp
    blas/blas_dispatcher_gemv_test.cpp
    blas/blas_dispatcher_ger_test.cpp
    blas/blas_dispatcher_symm_test.cpp
    blas/blas_dispatcher_symv_test.cpp
    blas/blas_dispatcher_syr_test.cpp
    blas/blas_dispatcher_syrk_test.cpp
    blas/blas_node_axpy_test.cpp
    blas/blas_node_copy_test.cpp
    blas/blas_node_dot_test.cpp
    blas/blas_node_gemm_test.cpp
    blas/blas_node_gemv_test.cpp
    blas/blas_node_ger_test.cpp
    blas/blas_node_symm_test.cpp
    blas/blas_node_symv_test.cpp
    blas/blas_node_syr_test.cpp
    blas/blas_node_syrk_test.cpp
    einsum/einsum_dispatcher_test.cpp
    einsum/einsum_node_test.cpp
    transformations/einsum_expand_fail_test.cpp
    transformations/einsum_expand_test.cpp
    transformations/einsum_lift_fail_test.cpp
    transformations/einsum_lift_test.cpp
    transformations/einsum2blas_axpy_test.cpp
    transformations/einsum2blas_copy_test.cpp
    transformations/einsum2blas_dot_test.cpp
    transformations/einsum2blas_gemm_test.cpp
    transformations/einsum2blas_gemv_test.cpp
    transformations/einsum2blas_symv_test.cpp
    test.cpp
)

# Now simply link against gtest or gtest_main as needed. Eg
add_executable(sdfglib-einsum_test ${TEST_FILES})
target_include_directories(sdfglib-einsum_test PRIVATE ./)
target_link_libraries(sdfglib-einsum_test gtest_main sdfglib-einsum)

add_test(NAME sdfglib-einsum_test COMMAND sdfglib-einsum_test)

include(GoogleTest)
gtest_discover_tests(sdfglib-einsum_test)
