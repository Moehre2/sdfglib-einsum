#include "sdfg/blas/blas_node_copy.h"

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

TEST(BLASNodeCopy, scopy) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& x = builder.add_access(block, "x");
    auto& y = builder.add_access(block, "y");
    auto& libnode = builder.add_library_node<blas::BLASNodeCopy, const blas::BLASType,
                                             symbolic::Expression, std::string, std::string>(
        block, DebugInfo(), blas::BLASType_real, symbolic::symbol("n"), "_x", "_y");
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, libnode, "_y", y, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeCopy*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), "scopy(n, _x, 1, _y, 1)");
}

TEST(BLASNodeCopy, dcopy) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& x = builder.add_access(block, "x");
    auto& y = builder.add_access(block, "y");
    auto& libnode = builder.add_library_node<blas::BLASNodeCopy, const blas::BLASType,
                                             symbolic::Expression, std::string, std::string>(
        block, DebugInfo(), blas::BLASType_double, symbolic::symbol("n"), "_x", "_y");
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, libnode, "_y", y, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeCopy*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), "dcopy(n, _x, 1, _y, 1)");
}