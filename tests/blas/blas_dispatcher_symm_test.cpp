#include <gtest/gtest.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/code_generators/c_code_generator.h>
#include <sdfg/element.h>
#include <sdfg/function.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <string>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_symm.h"

using namespace sdfg;

inline void symm_test(const types::PrimitiveType type1, const blas::BLASType type2,
                      const blas::BLASSide side, const blas::BLASTriangular uplo,
                      const std::string expected_func_def, const std::string expected_main) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(type1);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<blas::BLASNodeSymm, const blas::BLASType, blas::BLASSide,
                                 blas::BLASTriangular, symbolic::Expression, symbolic::Expression,
                                 std::string, std::string, std::string, std::string>(
            block, DebugInfo(), type2, side, uplo, symbolic::symbol("m"), symbolic::symbol("n"),
            "_alpha", "_A", "_B", "_C");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, A, "void", libnode, "_A", {});
    builder.add_memlet(block, B, "void", libnode, "_B", {});
    builder.add_memlet(block, C1, "void", libnode, "_C", {});
    builder.add_memlet(block, libnode, "_C", C2, "void", {});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    ASSERT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(), expected_func_def);
    EXPECT_EQ(generator.main().str(), expected_main);
}

TEST(BLASDispatcherSymm, ssymmLL) {
    symm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASSide_Left,
              blas::BLASTriangular_Lower,
              "extern void sdfg_1(unsigned long long m, unsigned long long n, float alpha, float "
              "**A, float **B, float **C)",
              R"(    {
        float _alpha = alpha;
        float **_A = A;
        float **_B = B;
        float **_C = C;

        cblas_ssymm(CblasRowMajor, CblasLeft, CblasLower, m, n, _alpha, _A, m, _B, m, 1.0f, _C, m);
    }
)");
}

TEST(BLASDispatcherSymm, ssymmRL) {
    symm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASSide_Right,
              blas::BLASTriangular_Lower,
              "extern void sdfg_1(unsigned long long m, unsigned long long n, float alpha, float "
              "**A, float **B, float **C)",
              R"(    {
        float _alpha = alpha;
        float **_A = A;
        float **_B = B;
        float **_C = C;

        cblas_ssymm(CblasRowMajor, CblasRight, CblasLower, m, n, _alpha, _A, n, _B, m, 1.0f, _C, m);
    }
)");
}

TEST(BLASDispatcherSymm, ssymmLU) {
    symm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASSide_Left,
              blas::BLASTriangular_Upper,
              "extern void sdfg_1(unsigned long long m, unsigned long long n, float alpha, float "
              "**A, float **B, float **C)",
              R"(    {
        float _alpha = alpha;
        float **_A = A;
        float **_B = B;
        float **_C = C;

        cblas_ssymm(CblasRowMajor, CblasLeft, CblasUpper, m, n, _alpha, _A, m, _B, m, 1.0f, _C, m);
    }
)");
}

TEST(BLASDispatcherSymm, ssymmRU) {
    symm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASSide_Right,
              blas::BLASTriangular_Upper,
              "extern void sdfg_1(unsigned long long m, unsigned long long n, float alpha, float "
              "**A, float **B, float **C)",
              R"(    {
        float _alpha = alpha;
        float **_A = A;
        float **_B = B;
        float **_C = C;

        cblas_ssymm(CblasRowMajor, CblasRight, CblasUpper, m, n, _alpha, _A, n, _B, m, 1.0f, _C, m);
    }
)");
}

TEST(BLASDispatcherSymm, dsymmLL) {
    symm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASSide_Left,
              blas::BLASTriangular_Lower,
              "extern void sdfg_1(unsigned long long m, unsigned long long n, double alpha, double "
              "**A, double **B, double **C)",
              R"(    {
        double _alpha = alpha;
        double **_A = A;
        double **_B = B;
        double **_C = C;

        cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, m, n, _alpha, _A, m, _B, m, 1.0, _C, m);
    }
)");
}

TEST(BLASDispatcherSymm, dsymmRL) {
    symm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASSide_Right,
              blas::BLASTriangular_Lower,
              "extern void sdfg_1(unsigned long long m, unsigned long long n, double alpha, double "
              "**A, double **B, double **C)",
              R"(    {
        double _alpha = alpha;
        double **_A = A;
        double **_B = B;
        double **_C = C;

        cblas_dsymm(CblasRowMajor, CblasRight, CblasLower, m, n, _alpha, _A, n, _B, m, 1.0, _C, m);
    }
)");
}

TEST(BLASDispatcherSymm, dsymmLU) {
    symm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASSide_Left,
              blas::BLASTriangular_Upper,
              "extern void sdfg_1(unsigned long long m, unsigned long long n, double alpha, double "
              "**A, double **B, double **C)",
              R"(    {
        double _alpha = alpha;
        double **_A = A;
        double **_B = B;
        double **_C = C;

        cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, m, n, _alpha, _A, m, _B, m, 1.0, _C, m);
    }
)");
}

TEST(BLASDispatcherSymm, dsymmRU) {
    symm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASSide_Right,
              blas::BLASTriangular_Upper,
              "extern void sdfg_1(unsigned long long m, unsigned long long n, double alpha, double "
              "**A, double **B, double **C)",
              R"(    {
        double _alpha = alpha;
        double **_A = A;
        double **_B = B;
        double **_C = C;

        cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper, m, n, _alpha, _A, n, _B, m, 1.0, _C, m);
    }
)");
}