#include "sdfg/blas/blas_node_scal.h"

#include <gtest/gtest.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/element.h>
#include <sdfg/function.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <string>

#include "sdfg/blas/blas_node.h"

using namespace sdfg;

TEST(BLASNodeScal, sscal) {
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

    auto* blas_node = dynamic_cast<blas::BLASNodeScal*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), "sscal(n, _alpha, _x, 1)");
}

TEST(BLASNodeScal, dscal) {
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

    auto* blas_node = dynamic_cast<blas::BLASNodeScal*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), "dscal(n, _alpha, _x, 1)");
}