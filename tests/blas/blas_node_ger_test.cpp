#include "sdfg/blas/blas_node_ger.h"

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

inline void ger_test(const types::PrimitiveType type1, const blas::BLASType type2,
                     const std::string expected) {
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

    auto* blas_node = dynamic_cast<blas::BLASNodeGer*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), expected);
}

TEST(BLASNodeGer, sger) {
    ger_test(types::PrimitiveType::Float, blas::BLASType_real,
             "sger(m, n, _alpha, _x, 1, _y, 1, _A, n)");
}

TEST(BLASNodeGer, dger) {
    ger_test(types::PrimitiveType::Double, blas::BLASType_double,
             "dger(m, n, _alpha, _x, 1, _y, 1, _A, n)");
}