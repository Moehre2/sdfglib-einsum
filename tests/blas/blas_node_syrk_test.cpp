#include "sdfg/blas/blas_node_syrk.h"

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

inline void syrk_test(const types::PrimitiveType type1, const blas::BLASType type2,
                      const blas::BLASTriangular uplo, const blas::BLASTranspose trans,
                      const std::string expected) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);
    builder.add_container("k", sym_desc, true);

    types::Scalar base_desc(type1);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("C", desc2, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<blas::BLASNodeSyrk, const blas::BLASType, blas::BLASTriangular,
                                 blas::BLASTranspose, symbolic::Expression, symbolic::Expression,
                                 std::string, std::string, std::string>(
            block, DebugInfo(), type2, uplo, trans, symbolic::symbol("n"), symbolic::symbol("k"),
            "_alpha", "_A", "_C");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, A, "void", libnode, "_A", {});
    builder.add_memlet(block, C1, "void", libnode, "_C", {});
    builder.add_memlet(block, libnode, "_C", C2, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeSyrk*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), expected);
}

TEST(BLASNodeSyrk, ssyrkLN) {
    syrk_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Lower,
              blas::BLASTranspose_No, "ssyrk('L', 'N', n, k, _alpha, _A, k, 1.0, _C, n)");
}

TEST(BLASNodeSyrk, ssyrkUN) {
    syrk_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Upper,
              blas::BLASTranspose_No, "ssyrk('U', 'N', n, k, _alpha, _A, k, 1.0, _C, n)");
}

TEST(BLASNodeSyrk, ssyrkLT) {
    syrk_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Lower,
              blas::BLASTranspose_Transpose, "ssyrk('L', 'T', n, k, _alpha, _A, n, 1.0, _C, n)");
}

TEST(BLASNodeSyrk, ssyrkUT) {
    syrk_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Upper,
              blas::BLASTranspose_Transpose, "ssyrk('U', 'T', n, k, _alpha, _A, n, 1.0, _C, n)");
}

TEST(BLASNodeSyrk, dsyrkLN) {
    syrk_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Lower,
              blas::BLASTranspose_No, "dsyrk('L', 'N', n, k, _alpha, _A, k, 1.0, _C, n)");
}

TEST(BLASNodeSyrk, dsyrkUN) {
    syrk_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Upper,
              blas::BLASTranspose_No, "dsyrk('U', 'N', n, k, _alpha, _A, k, 1.0, _C, n)");
}

TEST(BLASNodeSyrk, dsyrkLT) {
    syrk_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Lower,
              blas::BLASTranspose_Transpose, "dsyrk('L', 'T', n, k, _alpha, _A, n, 1.0, _C, n)");
}

TEST(BLASNodeSyrk, dsyrkUT) {
    syrk_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Upper,
              blas::BLASTranspose_Transpose, "dsyrk('U', 'T', n, k, _alpha, _A, n, 1.0, _C, n)");
}