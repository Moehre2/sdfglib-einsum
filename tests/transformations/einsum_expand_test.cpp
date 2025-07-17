#include "sdfg/transformations/einsum_expand.h"

#include <gtest/gtest.h>
#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/code_generators/c_code_generator.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/element.h>
#include <sdfg/function.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/for.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "helper.h"
#include "sdfg/einsum/einsum_node.h"

using namespace sdfg;

TEST(EinsumExpand, MatrixMatrixMultiplication_1) {
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

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(k, K, body_i);

    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"B", {indvar_j, indvar_k}}, {"C", {indvar_i, indvar_k}}};

    auto& block1 = builder.add_block(body_k);
    auto& C1 = builder.add_access(block1, "C");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", C1, "void", subsets.at("C"));

    auto& block2 = builder.add_block(body_k);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& C2 = builder.add_access(block2, "C");
    auto& C3 = builder.add_access(block2, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block2, DebugInfo(), {"_out"}, {"_in0", "_in1", "_out"}, {{indvar_j, bound_j}},
            subsets.at("C"), {subsets.at("A"), subsets.at("B"), subsets.at("C")});
    builder.add_memlet(block2, A, "void", libnode, "_in0", {});
    builder.add_memlet(block2, B, "void", libnode, "_in1", {});
    builder.add_memlet(block2, C2, "void", libnode, "_out", {});
    builder.add_memlet(block2, libnode, "_out", C3, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    ASSERT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_k, *einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 2);
    auto* for_k_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_k_opt);
    EXPECT_EQ(for_k_opt, &for_k);
    AT_LEAST(for_k_opt->root().size(), 1);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_k_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* einsum_node_opt = dynamic_cast<einsum::EinsumNode*>(libnode_opt);
    ASSERT_TRUE(einsum_node_opt);
    EXPECT_EQ(einsum_node_opt->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node_opt->inputs(), std::vector<std::string>({"_in0", "_in1", "_out"}));
    AT_LEAST(einsum_node_opt->maps().size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(0), indvar_j));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(0), bound_j));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(1), indvar_k));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(1), bound_k));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->out_indices(), subsets.at("C")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode_opt);
    AT_LEAST(einsum_node_opt->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(2), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumExpand, MatrixMatrixMultiplication_2) {
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

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(k, K, body_i);

    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"B", {indvar_j, indvar_k}}, {"C", {indvar_i, indvar_k}}};

    auto& block1 = builder.add_block(body_k);
    auto& A = builder.add_access(block1, "A");
    auto& B = builder.add_access(block1, "B");
    auto& C1 = builder.add_access(block1, "C");
    auto& C2 = builder.add_access(block1, "C");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", C1, "void", subsets.at("C"));
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block1, DebugInfo(), {"_out"}, {"_in0", "_in1", "_out"}, {{indvar_j, bound_j}},
            subsets.at("C"), {subsets.at("A"), subsets.at("B"), subsets.at("C")});
    builder.add_memlet(block1, A, "void", libnode, "_in0", {});
    builder.add_memlet(block1, B, "void", libnode, "_in1", {});
    builder.add_memlet(block1, C1, "void", libnode, "_out", {});
    builder.add_memlet(block1, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_k, *einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 2);
    auto* for_k_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_k_opt);
    EXPECT_EQ(for_k_opt, &for_k);
    AT_LEAST(for_k_opt->root().size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&for_k_opt->root().at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt->dataflow().nodes().size(), 2);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* einsum_node_opt = dynamic_cast<einsum::EinsumNode*>(libnode_opt);
    ASSERT_TRUE(einsum_node_opt);
    EXPECT_EQ(einsum_node_opt->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node_opt->inputs(), std::vector<std::string>({"_in0", "_in1", "_out"}));
    AT_LEAST(einsum_node_opt->maps().size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(0), indvar_j));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(0), bound_j));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(1), indvar_k));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(1), bound_k));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->out_indices(), subsets.at("C")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode_opt);
    AT_LEAST(einsum_node_opt->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(2), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumExpand, MatrixMatrixMultiplication_3) {
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

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(k, K, body_i);

    auto indvar_j = symbolic::symbol("j");
    auto bound_j = symbolic::symbol("J");
    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"B", {indvar_j, indvar_k}}, {"C", {indvar_i, indvar_k}}};

    auto& block1 = builder.add_block(body_k);
    auto& C1 = builder.add_access(block1, "C");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", C1, "void", subsets.at("C"));

    auto& block2 = builder.add_block(body_i);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& C2 = builder.add_access(block2, "C");
    auto& C3 = builder.add_access(block2, "C");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block2, DebugInfo(), {"_out"}, {"_in0", "_in1", "_out"},
            {{indvar_j, bound_j}, {indvar_k, bound_k}}, subsets.at("C"),
            {subsets.at("A"), subsets.at("B"), subsets.at("C")});
    builder.add_memlet(block2, A, "void", libnode, "_in0", {});
    builder.add_memlet(block2, B, "void", libnode, "_in1", {});
    builder.add_memlet(block2, C2, "void", libnode, "_out", {});
    builder.add_memlet(block2, libnode, "_out", C3, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_i, *einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 2);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_k_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_k_opt);
    EXPECT_EQ(for_k_opt, &for_k);
    AT_LEAST(for_k_opt->root().size(), 1);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_k_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* einsum_node_opt = dynamic_cast<einsum::EinsumNode*>(libnode_opt);
    ASSERT_TRUE(einsum_node_opt);
    EXPECT_EQ(einsum_node_opt->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node_opt->inputs(), std::vector<std::string>({"_in0", "_in1", "_out"}));
    AT_LEAST(einsum_node_opt->maps().size(), 3);
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(0), indvar_j));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(0), bound_j));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(1), indvar_k));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(1), bound_k));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(2), indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(2), bound_i));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->out_indices(), subsets.at("C")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode_opt);
    AT_LEAST(einsum_node_opt->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(2), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumExpand, DiagonalExtraction_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("b", desc, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {{"A", {indvar_i, indvar_i}},
                                                                        {"b", {indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& b1 = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", b1, "void", subsets.at("b"));

    auto& block2 = builder.add_block(body_i);
    auto& A = builder.add_access(block2, "A");
    auto& b2 = builder.add_access(block2, "b");
    auto& b3 = builder.add_access(block2, "b");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block2, DebugInfo(), {"_out"}, {"_in0", "_out"}, {}, subsets.at("b"),
            {subsets.at("A"), subsets.at("b")});
    builder.add_memlet(block2, A, "void", libnode, "_in0", {});
    builder.add_memlet(block2, b2, "void", libnode, "_out", {});
    builder.add_memlet(block2, libnode, "_out", b3, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_i, *einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 2);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 4);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* einsum_node_opt = dynamic_cast<einsum::EinsumNode*>(libnode_opt);
    ASSERT_TRUE(einsum_node_opt);
    EXPECT_EQ(einsum_node_opt->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node_opt->inputs(), std::vector<std::string>({"_in0", "_out"}));
    AT_LEAST(einsum_node_opt->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(0), indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(0), bound_i));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->out_indices(), subsets.at("b")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode_opt);
    AT_LEAST(einsum_node_opt->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(1), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumExpand, DiagonalExtraction_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("b", desc, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {{"A", {indvar_i, indvar_i}},
                                                                        {"b", {indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& A = builder.add_access(block1, "A");
    auto& b1 = builder.add_access(block1, "b");
    auto& b2 = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", b1, "void", subsets.at("b"));
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block1, DebugInfo(), {"_out"}, {"_in0", "_out"}, {}, subsets.at("b"),
            {subsets.at("A"), subsets.at("b")});
    builder.add_memlet(block1, A, "void", libnode, "_in0", {});
    builder.add_memlet(block1, b1, "void", libnode, "_out", {});
    builder.add_memlet(block1, libnode, "_out", b2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_i, *einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 2);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt->dataflow().nodes().size(), 2);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 4);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* einsum_node_opt = dynamic_cast<einsum::EinsumNode*>(libnode_opt);
    ASSERT_TRUE(einsum_node_opt);
    EXPECT_EQ(einsum_node_opt->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node_opt->inputs(), std::vector<std::string>({"_in0", "_out"}));
    AT_LEAST(einsum_node_opt->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(0), indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(0), bound_i));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->out_indices(), subsets.at("b")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode_opt);
    AT_LEAST(einsum_node_opt->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(1), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumExpand, DiagonalExtraction_3) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("b", desc, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {{"A", {indvar_i, indvar_i}},
                                                                        {"b", {indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& A = builder.add_access(block1, "A");
    auto& b = builder.add_access(block1, "b");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block1, DebugInfo(), {"_out"}, {"_in0"}, {}, subsets.at("b"), {subsets.at("A")});
    builder.add_memlet(block1, A, "void", libnode, "_in0", {});
    builder.add_memlet(block1, libnode, "_out", b, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_i, *einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 3);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* einsum_node_opt = dynamic_cast<einsum::EinsumNode*>(libnode_opt);
    ASSERT_TRUE(einsum_node_opt);
    EXPECT_EQ(einsum_node_opt->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node_opt->inputs(), std::vector<std::string>({"_in0"}));
    AT_LEAST(einsum_node_opt->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(0), indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(0), bound_i));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->out_indices(), subsets.at("b")));
    AT_LEAST(einsum_node_opt->in_indices().size(), 1);
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(0), subsets.at("A")));
}

TEST(EinsumExpand, Means_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("b", desc, true);

    auto& root = builder.subject().root();

    gen_for(j, J, root);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    const std::unordered_map<std::string, data_flow::Subset> subsets = {{"A", {indvar_i, indvar_j}},
                                                                        {"b", {indvar_j}}};

    auto& block1 = builder.add_block(body_j);
    auto& b1 = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", b1, "void", {indvar_j});

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& b2 = builder.add_access(block2, "b");
    auto& b3 = builder.add_access(block2, "b");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block2, DebugInfo(), {"_out"}, {"_in0", "_out"}, {{indvar_i, bound_i}}, subsets.at("b"),
            {subsets.at("A"), subsets.at("b")});
    builder.add_memlet(block2, A, "void", libnode, "_in0", {});
    builder.add_memlet(block2, b2, "void", libnode, "_out", {});
    builder.add_memlet(block2, libnode, "_out", b3, "void", {});

    auto& block3 = builder.add_block(body_j);
    auto& b4 = builder.add_access(block3, "b");
    auto& b5 = builder.add_access(block3, "b");
    auto& I = builder.add_access(block3, "I");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::div, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", sym_desc}});
    builder.add_memlet(block3, b4, "void", tasklet3, "_in1", subsets.at("b"));
    builder.add_memlet(block3, I, "void", tasklet3, "_in2", {});
    builder.add_memlet(block3, tasklet3, "_out", b5, "void", subsets.at("b"));

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_j, *einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 3);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 4);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* einsum_node_opt = dynamic_cast<einsum::EinsumNode*>(libnode_opt);
    ASSERT_TRUE(einsum_node_opt);
    EXPECT_EQ(einsum_node_opt->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node_opt->inputs(), std::vector<std::string>({"_in0", "_out"}));
    AT_LEAST(einsum_node_opt->maps().size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(0), indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(0), bound_i));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(1), indvar_j));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(1), bound_j));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->out_indices(), subsets.at("b")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode_opt);
    AT_LEAST(einsum_node_opt->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(1), subsets.at(conn2cont.at("_out"))));
    auto* for_j_opt2 = dynamic_cast<structured_control_flow::For*>(&root_opt.at(2).first);
    ASSERT_TRUE(for_j_opt2);
    EXPECT_NE(for_j_opt2, &for_j);
    AT_LEAST(for_j_opt2->root().size(), 1);
    auto* block3_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt2->root().at(0).first);
    ASSERT_TRUE(block3_opt);
    EXPECT_EQ(block3_opt->dataflow().nodes().size(), 4);
}

TEST(EinsumExpand, Means_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("b", desc, true);

    auto& root = builder.subject().root();

    gen_for(j, J, root);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    const std::unordered_map<std::string, data_flow::Subset> subsets = {{"A", {indvar_i, indvar_j}},
                                                                        {"b", {indvar_j}}};

    auto& block1 = builder.add_block(body_j);
    auto& A = builder.add_access(block1, "A");
    auto& b1 = builder.add_access(block1, "b");
    auto& b2 = builder.add_access(block1, "b");
    auto& b3 = builder.add_access(block1, "b");
    auto& I = builder.add_access(block1, "I");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", b1, "void", {indvar_j});
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block1, DebugInfo(), {"_out"}, {"_in0", "_out"}, {{indvar_i, bound_i}}, subsets.at("b"),
            {subsets.at("A"), subsets.at("b")});
    builder.add_memlet(block1, A, "void", libnode, "_in0", {});
    builder.add_memlet(block1, b1, "void", libnode, "_out", {});
    builder.add_memlet(block1, libnode, "_out", b2, "void", {});
    auto& tasklet3 = builder.add_tasklet(block1, data_flow::TaskletCode::div, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", sym_desc}});
    builder.add_memlet(block1, b2, "void", tasklet3, "_in1", subsets.at("b"));
    builder.add_memlet(block1, I, "void", tasklet3, "_in2", {});
    builder.add_memlet(block1, tasklet3, "_out", b3, "void", subsets.at("b"));

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_j, *einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 3);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt->dataflow().nodes().size(), 2);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 4);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* einsum_node_opt = dynamic_cast<einsum::EinsumNode*>(libnode_opt);
    ASSERT_TRUE(einsum_node_opt);
    EXPECT_EQ(einsum_node_opt->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node_opt->inputs(), std::vector<std::string>({"_in0", "_out"}));
    AT_LEAST(einsum_node_opt->maps().size(), 2);
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(0), indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(0), bound_i));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->indvar(1), indvar_j));
    EXPECT_TRUE(symbolic::eq(einsum_node_opt->num_iteration(1), bound_j));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->out_indices(), subsets.at("b")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode_opt);
    AT_LEAST(einsum_node_opt->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node_opt->in_indices(1), subsets.at(conn2cont.at("_out"))));
    auto* for_j_opt2 = dynamic_cast<structured_control_flow::For*>(&root_opt.at(2).first);
    ASSERT_TRUE(for_j_opt2);
    EXPECT_NE(for_j_opt2, &for_j);
    AT_LEAST(for_j_opt2->root().size(), 1);
    auto* block3_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt2->root().at(0).first);
    ASSERT_TRUE(block3_opt);
    EXPECT_EQ(block3_opt->dataflow().nodes().size(), 4);
}
