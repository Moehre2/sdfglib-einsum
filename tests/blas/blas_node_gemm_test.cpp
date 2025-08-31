#include "sdfg/blas/blas_node_gemm.h"

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

inline void gemm_test(const types::PrimitiveType type1, const blas::BLASType type2,
                      const blas::BLASTranspose transA, const blas::BLASTranspose transB,
                      const std::string expected) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);
    builder.add_container("k", sym_desc, true);

    types::Scalar base_desc(type1);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<blas::BLASNodeGemm, const blas::BLASType, blas::BLASTranspose,
                                 blas::BLASTranspose, symbolic::Expression, symbolic::Expression,
                                 symbolic::Expression, std::string, std::string, std::string,
                                 std::string>(block, DebugInfo(), type2, transA, transB,
                                              symbolic::symbol("m"), symbolic::symbol("n"),
                                              symbolic::symbol("k"), "_alpha", "_A", "_B", "_C");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, A, "void", libnode, "_A", {});
    builder.add_memlet(block, B, "void", libnode, "_B", {});
    builder.add_memlet(block, C1, "void", libnode, "_C", {});
    builder.add_memlet(block, libnode, "_C", C2, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), expected);
}

TEST(BLASNodeGemm, sgemmNN) {
    gemm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTranspose_No,
              blas::BLASTranspose_No, "sgemm('N', 'N', m, n, k, _alpha, _A, k, _B, n, 1.0, _C, n)");
}

TEST(BLASNodeGemm, sgemmTN) {
    gemm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTranspose_Transpose,
              blas::BLASTranspose_No, "sgemm('T', 'N', m, n, k, _alpha, _A, m, _B, n, 1.0, _C, n)");
}

TEST(BLASNodeGemm, sgemmNT) {
    gemm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTranspose_No,
              blas::BLASTranspose_Transpose,
              "sgemm('N', 'T', m, n, k, _alpha, _A, k, _B, k, 1.0, _C, n)");
}

TEST(BLASNodeGemm, sgemmTT) {
    gemm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTranspose_Transpose,
              blas::BLASTranspose_Transpose,
              "sgemm('T', 'T', m, n, k, _alpha, _A, m, _B, k, 1.0, _C, n)");
}

TEST(BLASNodeGemm, dgemmNN) {
    gemm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTranspose_No,
              blas::BLASTranspose_No, "dgemm('N', 'N', m, n, k, _alpha, _A, k, _B, n, 1.0, _C, n)");
}

TEST(BLASNodeGemm, dgemmTN) {
    gemm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTranspose_Transpose,
              blas::BLASTranspose_No, "dgemm('T', 'N', m, n, k, _alpha, _A, m, _B, n, 1.0, _C, n)");
}

TEST(BLASNodeGemm, dgemmNT) {
    gemm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTranspose_No,
              blas::BLASTranspose_Transpose,
              "dgemm('N', 'T', m, n, k, _alpha, _A, k, _B, k, 1.0, _C, n)");
}

TEST(BLASNodeGemm, dgemmTT) {
    gemm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTranspose_Transpose,
              blas::BLASTranspose_Transpose,
              "dgemm('T', 'T', m, n, k, _alpha, _A, m, _B, k, 1.0, _C, n)");
}