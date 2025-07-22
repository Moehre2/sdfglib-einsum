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
#include "sdfg/blas/blas_node_dot.h"

using namespace sdfg;

TEST(BLASDispatcherDot, sdot) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("result", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& result1 = builder.add_access(block, "result");
    auto& result2 = builder.add_access(block, "result");
    auto& x = builder.add_access(block, "x");
    auto& y = builder.add_access(block, "y");
    auto& libnode = builder.add_library_node<blas::BLASNodeDot, std::string, const blas::BLASType,
                                             symbolic::Expression, std::string, std::string>(
        block, DebugInfo(), "_result", blas::BLASType_real, symbolic::symbol("n"), "_x", "_y");
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, y, "void", libnode, "_y", {});
    builder.add_memlet(block, result1, "void", libnode, "_result", {symbolic::zero()});
    builder.add_memlet(block, libnode, "_result", result2, "void", {symbolic::zero()});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    ASSERT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long n, float *result, float *x, float *y)");
    EXPECT_EQ(generator.main().str(), R"(    {
        float *_x = x;
        float *_y = y;
        float _result = result[0];

        _result = _result + cblas_sdot(n, _x, 1, _y, 1);

        result[0] = _result;
    }
)");
}

TEST(BLASDispatcherDot, ddot) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    builder.add_container("result", desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& result1 = builder.add_access(block, "result");
    auto& result2 = builder.add_access(block, "result");
    auto& x = builder.add_access(block, "x");
    auto& y = builder.add_access(block, "y");
    auto& libnode = builder.add_library_node<blas::BLASNodeDot, std::string, const blas::BLASType,
                                             symbolic::Expression, std::string, std::string>(
        block, DebugInfo(), "_result", blas::BLASType_double, symbolic::symbol("n"), "_x", "_y");
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, y, "void", libnode, "_y", {});
    builder.add_memlet(block, result1, "void", libnode, "_result", {symbolic::zero()});
    builder.add_memlet(block, libnode, "_result", result2, "void", {symbolic::zero()});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    ASSERT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long n, double *result, double *x, double *y)");
    EXPECT_EQ(generator.main().str(), R"(    {
        double *_x = x;
        double *_y = y;
        double _result = result[0];

        _result = _result + cblas_ddot(n, _x, 1, _y, 1);

        result[0] = _result;
    }
)");
}