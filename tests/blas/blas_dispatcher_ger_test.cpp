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
#include "sdfg/blas/blas_node_ger.h"

using namespace sdfg;

inline void ger_test(const types::PrimitiveType type1, const blas::BLASType type2,
                     const std::string expected_func_def, const std::string expected_main) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(type1);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);
    builder.add_container("A", desc2, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& x = builder.add_access(block, "x");
    auto& y = builder.add_access(block, "y");
    auto& A1 = builder.add_access(block, "A");
    auto& A2 = builder.add_access(block, "A");
    auto& libnode =
        builder.add_library_node<blas::BLASNodeGer, const blas::BLASType, symbolic::Expression,
                                 symbolic::Expression, std::string, std::string, std::string,
                                 std::string>(block, DebugInfo(), type2, symbolic::symbol("m"),
                                              symbolic::symbol("n"), "_alpha", "_x", "_y", "_A");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, y, "void", libnode, "_y", {});
    builder.add_memlet(block, A1, "void", libnode, "_A", {});
    builder.add_memlet(block, libnode, "_A", A2, "void", {});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    ASSERT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(), expected_func_def);
    EXPECT_EQ(generator.main().str(), expected_main);
}

TEST(BLASDispatcherGer, sger) {
    ger_test(types::PrimitiveType::Float, blas::BLASType_real,
             "extern void sdfg_1(unsigned long long m, unsigned long long n, float alpha, float "
             "*x, float *y, float **A)",
             R"(    {
        float _alpha = alpha;
        float *_x = x;
        float *_y = y;
        float **_A = A;

        cblas_sger(CblasRowMajor, m, n, _alpha, _x, 1, _y, 1, _A, n);
    }
)");
}

TEST(BLASDispatcherGer, dger) {
    ger_test(types::PrimitiveType::Double, blas::BLASType_double,
             "extern void sdfg_1(unsigned long long m, unsigned long long n, double alpha, double "
             "*x, double *y, double **A)",
             R"(    {
        double _alpha = alpha;
        double *_x = x;
        double *_y = y;
        double **_A = A;

        cblas_dger(CblasRowMajor, m, n, _alpha, _x, 1, _y, 1, _A, n);
    }
)");
}