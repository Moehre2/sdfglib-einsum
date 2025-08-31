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
#include "sdfg/blas/blas_node_symv.h"

using namespace sdfg;

inline void symv_test(const types::PrimitiveType type1, const blas::BLASType type2,
                      const blas::BLASTriangular uplo, const std::string expected_func_def,
                      const std::string expected_main) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(type1);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& x = builder.add_access(block, "x");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& libnode = builder.add_library_node<blas::BLASNodeSymv, const blas::BLASType,
                                             blas::BLASTriangular, symbolic::Expression,
                                             std::string, std::string, std::string, std::string>(
        block, DebugInfo(), type2, uplo, symbolic::symbol("n"), "_alpha", "_A", "_x", "_y");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, A, "void", libnode, "_A", {});
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, y1, "void", libnode, "_y", {});
    builder.add_memlet(block, libnode, "_y", y2, "void", {});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    ASSERT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(), expected_func_def);
    EXPECT_EQ(generator.main().str(), expected_main);
}

TEST(BLASDispatcherSymv, ssymvL) {
    symv_test(
        types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Lower,
        "extern void sdfg_1(unsigned long long n, float alpha, float **A, float *x, float *y)",
        R"(    {
        float _alpha = alpha;
        float **_A = A;
        float *_x = x;
        float *_y = y;

        cblas_ssymv(CblasRowMajor, CblasLower, n, _alpha, _A, n, _x, 1, 1.0f, _y, 1);
    }
)");
}

TEST(BLASDispatcherSymv, ssymvU) {
    symv_test(
        types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Upper,
        "extern void sdfg_1(unsigned long long n, float alpha, float **A, float *x, float *y)",
        R"(    {
        float _alpha = alpha;
        float **_A = A;
        float *_x = x;
        float *_y = y;

        cblas_ssymv(CblasRowMajor, CblasUpper, n, _alpha, _A, n, _x, 1, 1.0f, _y, 1);
    }
)");
}

TEST(BLASDispatcherSymv, dsymvL) {
    symv_test(
        types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Lower,
        "extern void sdfg_1(unsigned long long n, double alpha, double **A, double *x, double *y)",
        R"(    {
        double _alpha = alpha;
        double **_A = A;
        double *_x = x;
        double *_y = y;

        cblas_dsymv(CblasRowMajor, CblasLower, n, _alpha, _A, n, _x, 1, 1.0, _y, 1);
    }
)");
}

TEST(BLASDispatcherSymv, dsymvU) {
    symv_test(
        types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Upper,
        "extern void sdfg_1(unsigned long long n, double alpha, double **A, double *x, double *y)",
        R"(    {
        double _alpha = alpha;
        double **_A = A;
        double *_x = x;
        double *_y = y;

        cblas_dsymv(CblasRowMajor, CblasUpper, n, _alpha, _A, n, _x, 1, 1.0, _y, 1);
    }
)");
}