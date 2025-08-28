#include "sdfg/blas/blas_node_gemv.h"

#include <gtest/gtest.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/function.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <string>

#include "sdfg/blas/blas_node.h"

using namespace sdfg;

TEST(BLASNodeGemv, sgemvN) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
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
    auto& libnode =
        builder.add_library_node<blas::BLASNodeGemv, const blas::BLASType, blas::BLASTranspose,
                                 symbolic::Expression, symbolic::Expression, std::string,
                                 std::string, std::string, std::string>(
            block, DebugInfo(), blas::BLASType_real, blas::BLASTranspose_No, symbolic::symbol("m"),
            symbolic::symbol("n"), "_alpha", "_A", "_x", "_y");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, A, "void", libnode, "_A", {});
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, y1, "void", libnode, "_y", {});
    builder.add_memlet(block, libnode, "_y", y2, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeGemv*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), "sgemv('N', m, n, _alpha, _A, n, _x, 1, 1.0, _y, 1)");
}

TEST(BLASNodeGemv, sgemvT) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
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
    auto& libnode =
        builder.add_library_node<blas::BLASNodeGemv, const blas::BLASType, blas::BLASTranspose,
                                 symbolic::Expression, symbolic::Expression, std::string,
                                 std::string, std::string, std::string>(
            block, DebugInfo(), blas::BLASType_real, blas::BLASTranspose_Transpose,
            symbolic::symbol("m"), symbolic::symbol("n"), "_alpha", "_A", "_x", "_y");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, A, "void", libnode, "_A", {});
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, y1, "void", libnode, "_y", {});
    builder.add_memlet(block, libnode, "_y", y2, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeGemv*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), "sgemv('T', m, n, _alpha, _A, n, _x, 1, 1.0, _y, 1)");
}

TEST(BLASNodeGemv, dgemvN) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
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
    auto& libnode =
        builder.add_library_node<blas::BLASNodeGemv, const blas::BLASType, blas::BLASTranspose,
                                 symbolic::Expression, symbolic::Expression, std::string,
                                 std::string, std::string, std::string>(
            block, DebugInfo(), blas::BLASType_double, blas::BLASTranspose_No,
            symbolic::symbol("m"), symbolic::symbol("n"), "_alpha", "_A", "_x", "_y");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, A, "void", libnode, "_A", {});
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, y1, "void", libnode, "_y", {});
    builder.add_memlet(block, libnode, "_y", y2, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeGemv*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), "dgemv('N', m, n, _alpha, _A, n, _x, 1, 1.0, _y, 1)");
}

TEST(BLASNodeGemv, dgemvT) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
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
    auto& libnode =
        builder.add_library_node<blas::BLASNodeGemv, const blas::BLASType, blas::BLASTranspose,
                                 symbolic::Expression, symbolic::Expression, std::string,
                                 std::string, std::string, std::string>(
            block, DebugInfo(), blas::BLASType_double, blas::BLASTranspose_Transpose,
            symbolic::symbol("m"), symbolic::symbol("n"), "_alpha", "_A", "_x", "_y");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, A, "void", libnode, "_A", {});
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, y1, "void", libnode, "_y", {});
    builder.add_memlet(block, libnode, "_y", y2, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeGemv*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), "dgemv('T', m, n, _alpha, _A, n, _x, 1, 1.0, _y, 1)");
}