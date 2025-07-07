#include <gtest/gtest.h>
#include <sdfg/codegen/code_generators/c_code_generator.h>

#include "fixtures/einsum.h"

TEST(EinsumDispatcher, MatrixMatrixMultiplication) {
    auto sdfg_and_node = matrix_matrix_mult();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, unsigned long long J, unsigned long long "
              "K, float **A, float **B, float **C)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long k;
unsigned long long j;
unsigned long long i;
    {
        float **_in1 = A;
        float **_in2 = B;

        for (i = 0; i < I; i++)
        {
            for (k = 0; k < K; k++)
            {
                float _out = C[i][k];

                for (j = 0; j < J; j++)
                {
                    _out = _out + _in1[i][j] * _in2[j][k];
                }

                C[i][k] = _out;
            }
        }
    }
)");
}

TEST(EinsumDispatcher, MatrixMatrixMultiplication_partial1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");

    auto& root = builder.subject().root();

    auto& for_i = builder.add_for(root, i, symbolic::Lt(i, symbolic::symbol("I")), symbolic::zero(),
                                  symbolic::add(i, symbolic::one()));
    auto& body_i = for_i.root();

    auto& block = builder.add_block(body_i);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, einsum::LibraryNodeType_Einsum, {"_out"}, {"_out", "_in1", "_in2"}, false,
            DebugInfo(), {{j, symbolic::symbol("J")}, {k, symbolic::symbol("K")}}, {i, k},
            {{i, k}, {i, j}, {j, k}});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, A, "void", libnode, "_in1", {});
    builder.add_memlet(block, B, "void", libnode, "_in2", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, unsigned long long J, unsigned long long "
              "K, float **A, float **B, float **C)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long k;
unsigned long long j;
unsigned long long i;
    for(i = 0;i < I;i = 1 + i)
    {
            {
                float **_in1 = A;
                float **_in2 = B;

                for (k = 0; k < K; k++)
                {
                    float _out = C[i][k];

                    for (j = 0; j < J; j++)
                    {
                        _out = _out + _in1[i][j] * _in2[j][k];
                    }

                    C[i][k] = _out;
                }
            }
    }
)");
}

TEST(EinsumDispatcher, MatrixMatrixMultiplication_partial2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");

    auto& root = builder.subject().root();

    auto& for_i = builder.add_for(root, i, symbolic::Lt(i, symbolic::symbol("I")), symbolic::zero(),
                                  symbolic::add(i, symbolic::one()));
    auto& body_i = for_i.root();

    auto& for_k = builder.add_for(body_i, k, symbolic::Lt(k, symbolic::symbol("K")),
                                  symbolic::zero(), symbolic::add(k, symbolic::one()));
    auto& body_k = for_k.root();

    auto& block = builder.add_block(body_k);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, einsum::LibraryNodeType_Einsum, {"_out"}, {"_out", "_in1", "_in2"}, false,
            DebugInfo(), {{j, symbolic::symbol("J")}}, {i, k}, {{i, k}, {i, j}, {j, k}});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, A, "void", libnode, "_in1", {});
    builder.add_memlet(block, B, "void", libnode, "_in2", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, unsigned long long J, unsigned long long "
              "K, float **A, float **B, float **C)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long k;
unsigned long long j;
unsigned long long i;
    for(i = 0;i < I;i = 1 + i)
    {
            for(k = 0;k < K;k = 1 + k)
            {
                    {
                        float **_in1 = A;
                        float **_in2 = B;

                        float _out = C[i][k];

                        for (j = 0; j < J; j++)
                        {
                            _out = _out + _in1[i][j] * _in2[j][k];
                        }

                        C[i][k] = _out;
                    }
            }
    }
)");
}

TEST(EinsumDispatcher, TensorContraction3D) {
    auto sdfg_and_node = tensor_contraction_3d();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, unsigned long long J, unsigned long long "
              "K, unsigned long long L, unsigned long long M, unsigned long long N, float ***A, "
              "float ***B, float ***C, float ***D)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long i;
unsigned long long j;
unsigned long long l;
unsigned long long n;
unsigned long long k;
unsigned long long m;
    {
        float ***_in1 = A;
        float ***_in2 = B;
        float ***_in3 = C;

        for (i = 0; i < I; i++)
        {
            for (j = 0; j < J; j++)
            {
                for (k = 0; k < K; k++)
                {
                    float _out = D[i][j][k];

                    for (l = 0; l < L; l++)
                    {
                        for (m = 0; m < M; m++)
                        {
                            for (n = 0; n < N; n++)
                            {
                                _out = _out + _in1[l][j][m] * _in2[i][l][n] * _in3[n][m][k];
                            }
                        }
                    }

                    D[i][j][k] = _out;
                }
            }
        }
    }
)");
}

TEST(EinsumDispatcher, MatrixVectorMultiplication) {
    auto sdfg_and_node = matrix_vector_mult();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, unsigned long long J, float **A, float *b, "
              "float *c)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long j;
unsigned long long i;
    {
        float **_in1 = A;
        float *_in2 = b;

        for (i = 0; i < I; i++)
        {
            float _out = c[i];

            for (j = 0; j < J; j++)
            {
                _out = _out + _in1[i][j] * _in2[j];
            }

            c[i] = _out;
        }
    }
)");
}

TEST(EinsumDispatcher, DiagonalExtraction) {
    auto sdfg_and_node = diagonal_extraction();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, float **A, float *b)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long i;
    {
        float **_in = A;

        for (i = 0; i < I; i++)
        {
            float _out;

            _out = _in[i][i];

            b[i] = _out;
        }
    }
)");
}

TEST(EinsumDispatcher, MatrixTrace) {
    auto sdfg_and_node = matrix_trace();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, float **A, float *b)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long i;
    {
        float **_in = A;

        float _out = *b;

        for (i = 0; i < I; i++)
        {
            _out = _out + _in[i][i];
        }

        *b = _out;
    }
)");
}

TEST(EinsumDispatcher, MatrixCopy) {
    auto sdfg_and_node = matrix_copy();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(
        generator.function_definition(),
        "extern void sdfg_1(unsigned long long I, unsigned long long J, float **A, float **B)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long j;
unsigned long long i;
    {
        float **_in = A;

        for (i = 0; i < I; i++)
        {
            for (j = 0; j < J; j++)
            {
                float _out;

                _out = _in[i][j];

                B[i][j] = _out;
            }
        }
    }
)");
}

TEST(EinsumDispatcher, MatrixTranspose) {
    auto sdfg_and_node = matrix_transpose();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(
        generator.function_definition(),
        "extern void sdfg_1(unsigned long long I, unsigned long long J, float **A, float **B)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long j;
unsigned long long i;
    {
        float **_in = A;

        for (j = 0; j < J; j++)
        {
            for (i = 0; i < I; i++)
            {
                float _out;

                _out = _in[i][j];

                B[j][i] = _out;
            }
        }
    }
)");
}

TEST(EinsumDispatcher, DotProduct) {
    auto sdfg_and_node = dot_product();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, float *a, float *b, float *c)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long i;
    {
        float *_in1 = a;
        float *_in2 = b;

        float _out = *c;

        for (i = 0; i < I; i++)
        {
            _out = _out + _in1[i] * _in2[i];
        }

        *c = _out;
    }
)");
}

TEST(EinsumDispatcher, MatrixElementwiseMultiplication) {
    auto sdfg_and_node = matrix_elementwise_mult();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, unsigned long long J, float **A, float "
              "**B, float **C, float **D)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long j;
unsigned long long i;
    {
        float **_in1 = A;
        float **_in2 = B;
        float **_in3 = C;

        for (i = 0; i < I; i++)
        {
            for (j = 0; j < J; j++)
            {
                float _out;

                _out = _in1[i][j] * _in2[i][j] * _in3[i][j];

                D[i][j] = _out;
            }
        }
    }
)");
}

TEST(EinsumDispatcher, VectorScaling) {
    auto sdfg_and_node = vector_scaling();
    auto sdfg = std::move(sdfg_and_node.first);
    auto* node = sdfg_and_node.second;

    EXPECT_TRUE(node);

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());

    EXPECT_EQ(generator.function_definition(),
              "extern void sdfg_1(unsigned long long I, float *a, float b, float c, float d, "
              "float *e)");
    EXPECT_EQ(generator.main().str(), R"(unsigned long long i;
    {
        float *_in1 = a;
        float _in2 = b;
        float _in3 = c;
        float _in4 = d;

        for (i = 0; i < I; i++)
        {
            float _out;

            _out = _in1[i] * _in2 * _in3 * _in4;

            e[i] = _out;
        }
    }
)");
}
