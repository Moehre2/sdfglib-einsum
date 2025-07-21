#include "sdfg/blas/blas_node_axpy.h"

#include <gtest/gtest.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/element.h>
#include <sdfg/function.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <string>
#include <vector>

#include "sdfg/blas/blas_node.h"

using namespace sdfg;

TEST(BLASNodeAxpy, saxpy) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& x = builder.add_access(block, "x");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& libnode = builder.add_library_node<blas::BLASNodeAxpy, const std::vector<std::string>&,
                                             const std::vector<std::string>&, const blas::BLASType,
                                             symbolic::Expression>(
        block, DebugInfo(), {"_y"}, {"_alpha", "_x", "_y"}, blas::BLASType_real,
        symbolic::symbol("n"));
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, y1, "void", libnode, "_y", {});
    builder.add_memlet(block, libnode, "_y", y2, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeAxpy*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), "_y = saxpy(n, _alpha, _x, 1, _y, 1)");
}

TEST(BLASNodeAxpy, daxpy) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    builder.add_container("alpha", base_desc, true);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& x = builder.add_access(block, "x");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& libnode = builder.add_library_node<blas::BLASNodeAxpy, const std::vector<std::string>&,
                                             const std::vector<std::string>&, const blas::BLASType,
                                             symbolic::Expression>(
        block, DebugInfo(), {"_y"}, {"_alpha", "_x", "_y"}, blas::BLASType_double,
        symbolic::symbol("n"));
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, y1, "void", libnode, "_y", {});
    builder.add_memlet(block, libnode, "_y", y2, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeAxpy*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), "_y = daxpy(n, _alpha, _x, 1, _y, 1)");
}