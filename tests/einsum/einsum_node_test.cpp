#include "sdfg/einsum/einsum_node.h"

#include <gtest/gtest.h>

#include "fixtures/einsum.h"

using namespace sdfg;

TEST(EinsumNode, MatrixMatrixMultiplication) {
    auto sdfg_and_node = matrix_matrix_mult();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    EXPECT_EQ(node->toStr(),
              "_out[i,k] = _in1[i,j] * _in2[j,k] for i = 0:I for j = 0:J for k = 0:K");
}

TEST(EinsumNode, TensorContraction3D) {
    auto sdfg_and_node = tensor_contraction_3d();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    EXPECT_EQ(node->toStr(),
              "_out[i,j,k] = _in1[l,j,m] * _in2[i,l,n] * _in3[n,m,k] for i = 0:I for j = 0:J for k "
              "= 0:K for l = 0:L for m = 0:M for n = 0:N");
}

TEST(EinsumNode, MatrixVectorMultiplication) {
    auto sdfg_and_node = matrix_vector_mult();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    EXPECT_EQ(node->toStr(), "_out[i] = _in1[i,j] * _in2[j] for i = 0:I for j = 0:J");
}

TEST(EinsumNode, DiagonalExtraction) {
    auto sdfg_and_node = diagonal_extraction();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    EXPECT_EQ(node->toStr(), "_out[i] = _in[i,i] for i = 0:I");
}

TEST(EinsumNode, MatrixTrace) {
    auto sdfg_and_node = matrix_trace();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    EXPECT_EQ(node->toStr(), "_out = _in[i,i] for i = 0:I");
}

TEST(EinsumNode, MatrixCopy) {
    auto sdfg_and_node = matrix_copy();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    EXPECT_EQ(node->toStr(), "_out[i,j] = _in[i,j] for i = 0:I for j = 0:J");
}

TEST(EinsumNode, MatrixTranspose) {
    auto sdfg_and_node = matrix_transpose();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    EXPECT_EQ(node->toStr(), "_out[j,i] = _in[i,j] for i = 0:I for j = 0:J");
}

TEST(EinsumNode, DotProduct) {
    auto sdfg_and_node = dot_product();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    EXPECT_EQ(node->toStr(), "_out[i] = _in1[i] * _in2[i] for i = 0:I");
}

TEST(EinsumNode, MatrixElementwiseMultiplication) {
    auto sdfg_and_node = matrix_elementwise_mult();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    EXPECT_EQ(node->toStr(),
              "_out[i,j] = _in1[i,j] * _in2[i,j] * _in3[i,j] for i = 0:I for j = 0:J");
}

TEST(EinsumNode, VectorScaling) {
    auto sdfg_and_node = vector_scaling();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    EXPECT_EQ(node->toStr(), "_out[i] = _in1[i] * _in2 * _in3 * _in4 for i = 0:I");
}
