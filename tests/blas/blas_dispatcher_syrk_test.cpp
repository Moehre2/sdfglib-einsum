#include <gtest/gtest.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/code_generators/c_code_generator.h>
#include <sdfg/function.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <string>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_syrk.h"

using namespace sdfg;

inline void syrk_test(const types::PrimitiveType type1, const blas::BLASType type2,
                      const blas::BLASTriangular uplo, const blas::BLASTranspose trans,
                      const std::string expected_func_def, const std::string expected_main) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);
    builder.add_container("k", sym_desc, true);

    types::Scalar base_desc(type1);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("C", desc2, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<blas::BLASNodeSyrk, const blas::BLASType, blas::BLASTriangular,
                                 blas::BLASTranspose, symbolic::Expression, symbolic::Expression,
                                 std::string, std::string, std::string>(
            block, DebugInfo(), type2, uplo, trans, symbolic::symbol("n"), symbolic::symbol("k"),
            "_alpha", "_A", "_C");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, A, "void", libnode, "_A", {});
    builder.add_memlet(block, C1, "void", libnode, "_C", {});
    builder.add_memlet(block, libnode, "_C", C2, "void", {});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    ASSERT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(), expected_func_def);
    EXPECT_EQ(generator.main().str(), expected_main);
}

TEST(BLASDispatcherSyrk, ssyrkLN) {
    syrk_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Lower,
              blas::BLASTranspose_No,
              "extern void sdfg_1(unsigned long long n, unsigned long long k, float alpha, float "
              "**A, float **C)",
              R"(    {
        float _alpha = alpha;
        float **_A = A;
        float **_C = C;

        cblas_ssyrk(CblasRowMajor, CblasLower, CblasNoTrans, n, k, _alpha, _A, k, 1.0f, _C, n);
    }
)");
}

TEST(BLASDispatcherSyrk, ssyrkUN) {
    syrk_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Upper,
              blas::BLASTranspose_No,
              "extern void sdfg_1(unsigned long long n, unsigned long long k, float alpha, float "
              "**A, float **C)",
              R"(    {
        float _alpha = alpha;
        float **_A = A;
        float **_C = C;

        cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, k, _alpha, _A, k, 1.0f, _C, n);
    }
)");
}

TEST(BLASDispatcherSyrk, ssyrkLT) {
    syrk_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Lower,
              blas::BLASTranspose_Transpose,
              "extern void sdfg_1(unsigned long long n, unsigned long long k, float alpha, float "
              "**A, float **C)",
              R"(    {
        float _alpha = alpha;
        float **_A = A;
        float **_C = C;

        cblas_ssyrk(CblasRowMajor, CblasLower, CblasTrans, n, k, _alpha, _A, n, 1.0f, _C, n);
    }
)");
}

TEST(BLASDispatcherSyrk, ssyrkUT) {
    syrk_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Upper,
              blas::BLASTranspose_Transpose,
              "extern void sdfg_1(unsigned long long n, unsigned long long k, float alpha, float "
              "**A, float **C)",
              R"(    {
        float _alpha = alpha;
        float **_A = A;
        float **_C = C;

        cblas_ssyrk(CblasRowMajor, CblasUpper, CblasTrans, n, k, _alpha, _A, n, 1.0f, _C, n);
    }
)");
}

TEST(BLASDispatcherSyrk, dsyrkLN) {
    syrk_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Lower,
              blas::BLASTranspose_No,
              "extern void sdfg_1(unsigned long long n, unsigned long long k, double alpha, double "
              "**A, double **C)",
              R"(    {
        double _alpha = alpha;
        double **_A = A;
        double **_C = C;

        cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans, n, k, _alpha, _A, k, 1.0, _C, n);
    }
)");
}

TEST(BLASDispatcherSyrk, dsyrkUN) {
    syrk_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Upper,
              blas::BLASTranspose_No,
              "extern void sdfg_1(unsigned long long n, unsigned long long k, double alpha, double "
              "**A, double **C)",
              R"(    {
        double _alpha = alpha;
        double **_A = A;
        double **_C = C;

        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, n, k, _alpha, _A, k, 1.0, _C, n);
    }
)");
}

TEST(BLASDispatcherSyrk, dsyrkLT) {
    syrk_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Lower,
              blas::BLASTranspose_Transpose,
              "extern void sdfg_1(unsigned long long n, unsigned long long k, double alpha, double "
              "**A, double **C)",
              R"(    {
        double _alpha = alpha;
        double **_A = A;
        double **_C = C;

        cblas_dsyrk(CblasRowMajor, CblasLower, CblasTrans, n, k, _alpha, _A, n, 1.0, _C, n);
    }
)");
}

TEST(BLASDispatcherSyrk, dsyrkUT) {
    syrk_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Upper,
              blas::BLASTranspose_Transpose,
              "extern void sdfg_1(unsigned long long n, unsigned long long k, double alpha, double "
              "**A, double **C)",
              R"(    {
        double _alpha = alpha;
        double **_A = A;
        double **_C = C;

        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, n, k, _alpha, _A, n, 1.0, _C, n);
    }
)");
}