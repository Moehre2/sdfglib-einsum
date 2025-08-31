#include "sdfg/transformations/einsum2blas_gemm.h"

#include <gtest/gtest.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/code_generators/c_code_generator.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/element.h>
#include <sdfg/function.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <string>
#include <utility>
#include <vector>

#include "helper.h"
#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_gemm.h"
#include "sdfg/einsum/einsum_node.h"

using namespace sdfg;

TEST(Einsum2BLASGemm, sgemmNN_1) {
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

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_i, indvar_k}, {indvar_k, indvar_j}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_real);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->alpha(), "1.0f");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, sgemmNN_2) {
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
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_in2", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_i, indvar_k}, {indvar_k, indvar_j}, {}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, alpha, "void", libnode, "_in2", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_real);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->alpha(), "_in2");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, sgemmTN_1) {
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

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_k, indvar_i}, {indvar_k, indvar_j}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_real);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->alpha(), "1.0f");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, sgemmTN_2) {
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
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_in2", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_k, indvar_i}, {indvar_k, indvar_j}, {}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, alpha, "void", libnode, "_in2", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_real);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->alpha(), "_in2");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, sgemmNT_1) {
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

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_i, indvar_k}, {indvar_j, indvar_k}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_real);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->alpha(), "1.0f");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, sgemmNT_2) {
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
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_in2", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_i, indvar_k}, {indvar_j, indvar_k}, {}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, alpha, "void", libnode, "_in2", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_real);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->alpha(), "_in2");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, sgemmTT_1) {
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

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_k, indvar_i}, {indvar_j, indvar_k}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_real);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->alpha(), "1.0f");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, sgemmTT_2) {
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
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_in2", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_k, indvar_i}, {indvar_j, indvar_k}, {}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, alpha, "void", libnode, "_in2", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_real);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->alpha(), "_in2");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, dgemmNN_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_i, indvar_k}, {indvar_k, indvar_j}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_double);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->alpha(), "1.0");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, dgemmNN_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_in2", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_i, indvar_k}, {indvar_k, indvar_j}, {}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, alpha, "void", libnode, "_in2", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_double);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->alpha(), "_in2");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, dgemmTN_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_k, indvar_i}, {indvar_k, indvar_j}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_double);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->alpha(), "1.0");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, dgemmTN_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_in2", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_k, indvar_i}, {indvar_k, indvar_j}, {}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, alpha, "void", libnode, "_in2", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_double);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->alpha(), "_in2");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, dgemmNT_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_i, indvar_k}, {indvar_j, indvar_k}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_double);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->alpha(), "1.0");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, dgemmNT_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_in2", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_i, indvar_k}, {indvar_j, indvar_k}, {}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, alpha, "void", libnode, "_in2", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_double);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_No);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->alpha(), "_in2");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, dgemmTT_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_k, indvar_i}, {indvar_j, indvar_k}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_double);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->alpha(), "1.0");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}

TEST(Einsum2BLASGemm, dgemmTT_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    auto indvar_k = symbolic::symbol("k");
    auto bound_k = symbolic::symbol("K");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0", "_in1", "_in2", "_out"},
            {{indvar_i, bound_i}, {indvar_j, bound_j}, {indvar_k, bound_k}}, {indvar_i, indvar_j},
            {{indvar_k, indvar_i}, {indvar_j, indvar_k}, {}, {indvar_i, indvar_j}});
    builder.add_memlet(block, A, "void", libnode, "_in0", {});
    builder.add_memlet(block, B, "void", libnode, "_in1", {});
    builder.add_memlet(block, alpha, "void", libnode, "_in2", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASGemm transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeGemm*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_double);
    EXPECT_EQ(blas_node->transA(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->transB(), blas::BLASTranspose_Transpose);
    EXPECT_EQ(blas_node->alpha(), "_in2");
    EXPECT_EQ(blas_node->A(), "_in0");
    EXPECT_EQ(blas_node->B(), "_in1");
    EXPECT_EQ(blas_node->C(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->m(), bound_i));
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_j));
    EXPECT_TRUE(symbolic::eq(blas_node->k(), bound_k));
}