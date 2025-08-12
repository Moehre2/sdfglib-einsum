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
#include "sdfg/blas/blas_node_scal.h"

using namespace sdfg;

TEST(BLASDispatcherScal, sscal) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("x", desc, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& x1 = builder.add_access(block, "x");
    auto& x2 = builder.add_access(block, "x");
    auto& libnode = builder.add_library_node<blas::BLASNodeScal, const blas::BLASType,
                                             symbolic::Expression, std::string, std::string>(
        block, DebugInfo(), blas::BLASType_real, symbolic::symbol("n"), "_alpha", "_x");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, x1, "void", libnode, "_x", {});
    builder.add_memlet(block, libnode, "_x", x2, "void", {});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    ASSERT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long n, float alpha, float *x)");
    EXPECT_EQ(generator.main().str(), R"(    {
        float _alpha = alpha;
        float *_x = x;

        cblas_sscal(n, _alpha, _x, 1);
    }
)");
}

TEST(BLASDispatcherScal, dscal) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("x", desc, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& x1 = builder.add_access(block, "x");
    auto& x2 = builder.add_access(block, "x");
    auto& libnode = builder.add_library_node<blas::BLASNodeScal, const blas::BLASType,
                                             symbolic::Expression, std::string, std::string>(
        block, DebugInfo(), blas::BLASType_double, symbolic::symbol("n"), "_alpha", "_x");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, x1, "void", libnode, "_x", {});
    builder.add_memlet(block, libnode, "_x", x2, "void", {});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    ASSERT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long n, double alpha, double *x)");
    EXPECT_EQ(generator.main().str(), R"(    {
        double _alpha = alpha;
        double *_x = x;

        cblas_dscal(n, _alpha, _x, 1);
    }
)");
}