#include "sdfg/transformations/einsum_lift.h"

#include <gtest/gtest.h>
#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/function.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/for.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <string>
#include <unordered_map>

#include "helper.h"
#include "sdfg/einsum/einsum_node.h"

using namespace sdfg;

TEST(EinsumLift, MatrixMatrixMultiplication_1) {
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
    builder.add_container("tmp", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(k, K, body_i);

    auto& block1 = builder.add_block(body_k);
    auto& C1 = builder.add_access(block1, "C");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", C1, "void", {indvar_i, indvar_k});

    gen_for(j, J, body_k);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"B", {indvar_j, indvar_k}}, {"C", {indvar_i, indvar_k}}};

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& C2 = builder.add_access(block2, "C");
    auto& C3 = builder.add_access(block2, "C");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, C2, "void", tasklet3, "_in1", subsets.at("C"));
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in2", {});
    builder.add_memlet(block2, tasklet3, "_out", C3, "void", subsets.at("C"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_j}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_k_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_k_opt);
    EXPECT_EQ(for_k_opt, &for_k);
    AT_LEAST(for_k_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_k_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_k_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_j));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_j));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("C")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, MatrixMatrixMultiplication_2) {
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

    auto& block1 = builder.add_block(body_k);
    auto& C1 = builder.add_access(block1, "C");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", C1, "void", {indvar_i, indvar_k});

    gen_for(j, J, body_k);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"B", {indvar_j, indvar_k}}, {"C", {indvar_i, indvar_k}}};

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& C2 = builder.add_access(block2, "C");
    auto& C3 = builder.add_access(block2, "C");
    auto& tasklet2 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block2, C2, "void", tasklet2, "_in3", subsets.at("C"));
    builder.add_memlet(block2, tasklet2, "_out", C3, "void", subsets.at("C"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_j}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_k_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_k_opt);
    EXPECT_EQ(for_k_opt, &for_k);
    AT_LEAST(for_k_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_k_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_k_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_j));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_j));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("C")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, TensorContraction3D_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);
    builder.add_container("l", sym_desc);
    builder.add_container("L", sym_desc, true);
    builder.add_container("m", sym_desc);
    builder.add_container("M", sym_desc, true);
    builder.add_container("n", sym_desc);
    builder.add_container("N", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    types::Pointer desc3(*desc2.clone());
    builder.add_container("A", desc3, true);
    builder.add_container("B", desc3, true);
    builder.add_container("C", desc3, true);
    builder.add_container("D", desc3, true);
    builder.add_container("tmp1", base_desc);
    builder.add_container("tmp2", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    gen_for(k, K, body_j);

    auto& block1 = builder.add_block(body_k);
    auto& D1 = builder.add_access(block1, "D");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", D1, "void", {indvar_i, indvar_j, indvar_k});

    gen_for(l, L, body_k);

    gen_for(m, M, body_l);

    gen_for(n, N, body_m);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_l, indvar_j, indvar_m}},
        {"B", {indvar_i, indvar_l, indvar_n}},
        {"C", {indvar_n, indvar_m, indvar_k}},
        {"D", {indvar_i, indvar_j, indvar_k}}};

    auto& block2 = builder.add_block(body_n);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& C = builder.add_access(block2, "C");
    auto& D2 = builder.add_access(block2, "D");
    auto& D3 = builder.add_access(block2, "D");
    auto& tmp1 = builder.add_access(block2, "tmp1");
    auto& tmp2 = builder.add_access(block2, "tmp2");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block2, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, C, "void", tasklet3, "_in2", subsets.at("C"));
    builder.add_memlet(block2, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, D2, "void", tasklet4, "_in1", subsets.at("D"));
    builder.add_memlet(block2, tmp2, "void", tasklet4, "_in2", {});
    builder.add_memlet(block2, tasklet4, "_out", D3, "void", subsets.at("D"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_l, for_m, for_n}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* for_k_opt = dynamic_cast<structured_control_flow::For*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(for_k_opt);
    EXPECT_EQ(for_k_opt, &for_k);
    AT_LEAST(for_k_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_k_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_k_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_in2", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 3);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_l));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_l));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(1).first, indvar_m));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(1).second, bound_m));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(2).first, indvar_n));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(2).second, bound_n));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("D")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 4);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(3), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, TensorContraction3D_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);
    builder.add_container("k", sym_desc);
    builder.add_container("K", sym_desc, true);
    builder.add_container("l", sym_desc);
    builder.add_container("L", sym_desc, true);
    builder.add_container("m", sym_desc);
    builder.add_container("M", sym_desc, true);
    builder.add_container("n", sym_desc);
    builder.add_container("N", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    types::Pointer desc3(*desc2.clone());
    builder.add_container("A", desc3, true);
    builder.add_container("B", desc3, true);
    builder.add_container("C", desc3, true);
    builder.add_container("D", desc3, true);
    builder.add_container("tmp", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    gen_for(k, K, body_j);

    auto& block1 = builder.add_block(body_k);
    auto& D1 = builder.add_access(block1, "D");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", D1, "void", {indvar_i, indvar_j, indvar_k});

    gen_for(l, L, body_k);

    gen_for(m, M, body_l);

    gen_for(n, N, body_m);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_l, indvar_j, indvar_m}},
        {"B", {indvar_i, indvar_l, indvar_n}},
        {"C", {indvar_n, indvar_m, indvar_k}},
        {"D", {indvar_i, indvar_j, indvar_k}}};

    auto& block2 = builder.add_block(body_n);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& C = builder.add_access(block2, "C");
    auto& D2 = builder.add_access(block2, "D");
    auto& D3 = builder.add_access(block2, "D");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, C, "void", tasklet3, "_in2", subsets.at("C"));
    builder.add_memlet(block2, D2, "void", tasklet3, "_in3", subsets.at("D"));
    builder.add_memlet(block2, tasklet3, "_out", D3, "void", subsets.at("D"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_l, for_m, for_n}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* for_k_opt = dynamic_cast<structured_control_flow::For*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(for_k_opt);
    EXPECT_EQ(for_k_opt, &for_k);
    AT_LEAST(for_k_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_k_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_k_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_in2", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 3);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_l));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_l));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(1).first, indvar_m));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(1).second, bound_m));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(2).first, indvar_n));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(2).second, bound_n));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("D")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 4);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(3), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, MatrixVectorMultiplication_1) {
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
    builder.add_container("c", desc, true);
    builder.add_container("tmp", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    auto& block1 = builder.add_block(body_i);
    auto& c1 = builder.add_access(block1, "c");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", c1, "void", {indvar_i});

    gen_for(j, J, body_i);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"b", {indvar_j}}, {"c", {indvar_i}}};

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, c2, "void", tasklet3, "_in1", subsets.at("c"));
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in2", {});
    builder.add_memlet(block2, tasklet3, "_out", c3, "void", subsets.at("c"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_j}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_j));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_j));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("c")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, MatrixVectorMultiplication_2) {
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
    builder.add_container("c", desc, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    auto& block1 = builder.add_block(body_i);
    auto& c1 = builder.add_access(block1, "c");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", c1, "void", {indvar_i});

    gen_for(j, J, body_i);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"b", {indvar_j}}, {"c", {indvar_i}}};

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tasklet2 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block2, c2, "void", tasklet2, "_in3", subsets.at("c"));
    builder.add_memlet(block2, tasklet2, "_out", c3, "void", subsets.at("c"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_j}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_j));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_j));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("c")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, DiagonalExtraction_1) {
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

    auto& block1 = builder.add_block(body_i);
    auto& b1 = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", b1, "void", {indvar_i});

    const std::unordered_map<std::string, data_flow::Subset> subsets = {{"A", {indvar_i, indvar_i}},
                                                                        {"b", {indvar_i}}};

    auto& block2 = builder.add_block(body_i);
    auto& A = builder.add_access(block2, "A");
    auto& b2 = builder.add_access(block2, "b");
    auto& b3 = builder.add_access(block2, "b");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, b2, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block2, tasklet2, "_out", b3, "void", subsets.at("b"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 4);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("b")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, DiagonalExtraction_2) {
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
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block1, b1, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block1, tasklet2, "_out", b2, "void", subsets.at("b"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 3);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("b")));
    AT_LEAST(einsum_node->in_indices().size(), 1);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at("A")));
}

TEST(EinsumLift, DiagonalExtraction_3) {
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
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet1, "_in", subsets.at("A"));
    builder.add_memlet(block1, tasklet1, "_out", b, "void", subsets.at("b"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 3);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("b")));
    AT_LEAST(einsum_node->in_indices().size(), 1);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at("A")));
}

TEST(EinsumLift, MatrixTrace_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("b", base_desc, true);

    auto& root = builder.subject().root();

    auto& block1 = builder.add_block(root);
    auto& b1 = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", b1, "void", {});

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {{"A", {indvar_i, indvar_i}},
                                                                        {"b", {}}};

    auto& block2 = builder.add_block(body_i);
    auto& A = builder.add_access(block2, "A");
    auto& b2 = builder.add_access(block2, "b");
    auto& b3 = builder.add_access(block2, "b");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, b2, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block2, tasklet2, "_out", b3, "void", subsets.at("b"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 2);
    auto* block1_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 4);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_i));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("b")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, MatrixTrace_2) {
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

    auto& block1 = builder.add_block(root);
    auto& b1 = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", b1, "void", {symbolic::zero()});

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {{"A", {indvar_i, indvar_i}},
                                                                        {"b", {symbolic::zero()}}};

    auto& block2 = builder.add_block(body_i);
    auto& A = builder.add_access(block2, "A");
    auto& b2 = builder.add_access(block2, "b");
    auto& b3 = builder.add_access(block2, "b");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, b2, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block2, tasklet2, "_out", b3, "void", subsets.at("b"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 2);
    auto* block1_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 4);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_i));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("b")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, MatrixCopy_1) {
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
    builder.add_container("B", desc2, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    auto& block1 = builder.add_block(body_j);
    auto& B1 = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", B1, "void", {indvar_i, indvar_j});

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"B", {indvar_i, indvar_j}}};

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& B2 = builder.add_access(block2, "B");
    auto& B3 = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, B2, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block2, tasklet2, "_out", B3, "void", subsets.at("B"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 4);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("B")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, MatrixCopy_2) {
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
    builder.add_container("B", desc2, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"B", {indvar_i, indvar_j}}};

    auto& block1 = builder.add_block(body_j);
    auto& A = builder.add_access(block1, "A");
    auto& B1 = builder.add_access(block1, "B");
    auto& B2 = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", B1, "void", subsets.at("B"));
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block1, B1, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block1, tasklet2, "_out", B2, "void", subsets.at("B"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 3);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("B")));
    AT_LEAST(einsum_node->in_indices().size(), 1);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at("A")));
}

TEST(EinsumLift, MatrixCopy_3) {
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
    builder.add_container("B", desc2, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"B", {indvar_i, indvar_j}}};

    auto& block1 = builder.add_block(body_j);
    auto& A = builder.add_access(block1, "A");
    auto& B = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet1, "_in", subsets.at("A"));
    builder.add_memlet(block1, tasklet1, "_out", B, "void", subsets.at("B"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 3);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("B")));
    AT_LEAST(einsum_node->in_indices().size(), 1);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at("A")));
}

TEST(EinsumLift, MatrixTranspose_1) {
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
    builder.add_container("B", desc2, true);

    auto& root = builder.subject().root();

    gen_for(j, J, root);

    gen_for(i, I, body_j);

    auto& block1 = builder.add_block(body_i);
    auto& B1 = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", B1, "void", {indvar_j, indvar_i});

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"B", {indvar_j, indvar_i}}};

    auto& block2 = builder.add_block(body_i);
    auto& A = builder.add_access(block2, "A");
    auto& B2 = builder.add_access(block2, "B");
    auto& B3 = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, B2, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block2, tasklet2, "_out", B3, "void", subsets.at("B"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 4);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("B")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, MatrixTranspose_2) {
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
    builder.add_container("B", desc2, true);

    auto& root = builder.subject().root();

    gen_for(j, J, root);

    gen_for(i, I, body_j);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"B", {indvar_j, indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& A = builder.add_access(block1, "A");
    auto& B1 = builder.add_access(block1, "B");
    auto& B2 = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", B1, "void", subsets.at("B"));
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block1, B1, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block1, tasklet2, "_out", B2, "void", subsets.at("B"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 3);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("B")));
    AT_LEAST(einsum_node->in_indices().size(), 1);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at("A")));
}

TEST(EinsumLift, MatrixTranspose_3) {
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
    builder.add_container("B", desc2, true);

    auto& root = builder.subject().root();

    gen_for(j, J, root);

    gen_for(i, I, body_j);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}}, {"B", {indvar_j, indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& A = builder.add_access(block1, "A");
    auto& B = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet1, "_in", subsets.at("A"));
    builder.add_memlet(block1, tasklet1, "_out", B, "void", subsets.at("B"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 3);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("B")));
    AT_LEAST(einsum_node->in_indices().size(), 1);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at("A")));
}

TEST(EinsumLift, DotProduct_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    builder.add_container("c", base_desc, true);
    builder.add_container("tmp", base_desc);

    auto& root = builder.subject().root();

    auto& block1 = builder.add_block(root);
    auto& c1 = builder.add_access(block1, "c");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", c1, "void", {});

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {indvar_i}}, {"c", {}}};

    auto& block2 = builder.add_block(body_i);
    auto& a = builder.add_access(block2, "a");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", subsets.at("a"));
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, c2, "void", tasklet3, "_in1", subsets.at("c"));
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in2", {});
    builder.add_memlet(block2, tasklet3, "_out", c3, "void", subsets.at("c"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 2);
    auto* block1_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_i));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("c")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, DotProduct_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    builder.add_container("c", desc, true);
    builder.add_container("tmp", base_desc);

    auto& root = builder.subject().root();

    auto& block1 = builder.add_block(root);
    auto& c1 = builder.add_access(block1, "c");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", c1, "void", {symbolic::zero()});

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {indvar_i}}, {"c", {symbolic::zero()}}};

    auto& block2 = builder.add_block(body_i);
    auto& a = builder.add_access(block2, "a");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", subsets.at("a"));
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, c2, "void", tasklet3, "_in1", subsets.at("c"));
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in2", {});
    builder.add_memlet(block2, tasklet3, "_out", c3, "void", subsets.at("c"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 2);
    auto* block1_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_i));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("c")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, DotProduct_3) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    builder.add_container("c", base_desc, true);

    auto& root = builder.subject().root();

    auto& block1 = builder.add_block(root);
    auto& c1 = builder.add_access(block1, "c");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", c1, "void", {});

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {indvar_i}}, {"c", {}}};

    auto& block2 = builder.add_block(body_i);
    auto& a = builder.add_access(block2, "a");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tasklet2 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", subsets.at("a"));
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block2, c2, "void", tasklet2, "_in3", subsets.at("c"));
    builder.add_memlet(block2, tasklet2, "_out", c3, "void", subsets.at("c"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 2);
    auto* block1_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_i));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("c")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, DotProduct_4) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    builder.add_container("c", desc, true);

    auto& root = builder.subject().root();

    auto& block1 = builder.add_block(root);
    auto& c1 = builder.add_access(block1, "c");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", c1, "void", {symbolic::zero()});

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {indvar_i}}, {"c", {symbolic::zero()}}};

    auto& block2 = builder.add_block(body_i);
    auto& a = builder.add_access(block2, "a");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tasklet2 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", subsets.at("a"));
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block2, c2, "void", tasklet2, "_in3", subsets.at("c"));
    builder.add_memlet(block2, tasklet2, "_out", c3, "void", subsets.at("c"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 2);
    auto* block1_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_i));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("c")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, MatrixElementwiseMultiplication_1) {
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
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);
    builder.add_container("D", desc2, true);
    builder.add_container("tmp1", base_desc);
    builder.add_container("tmp2", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    auto& block1 = builder.add_block(body_j);
    auto& D1 = builder.add_access(block1, "D");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", D1, "void", {indvar_i, indvar_j});

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}},
        {"B", {indvar_i, indvar_j}},
        {"C", {indvar_i, indvar_j}},
        {"D", {indvar_i, indvar_j}}};

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& C = builder.add_access(block2, "C");
    auto& D2 = builder.add_access(block2, "D");
    auto& D3 = builder.add_access(block2, "D");
    auto& tmp1 = builder.add_access(block2, "tmp1");
    auto& tmp2 = builder.add_access(block2, "tmp2");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block2, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, C, "void", tasklet3, "_in2", subsets.at("C"));
    builder.add_memlet(block2, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, D2, "void", tasklet4, "_in1", subsets.at("D"));
    builder.add_memlet(block2, tmp2, "void", tasklet4, "_in2", {});
    builder.add_memlet(block2, tasklet4, "_out", D3, "void", subsets.at("D"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_in2", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("D")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 4);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(3), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, MatrixElementwiseMultiplication_2) {
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
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);
    builder.add_container("D", desc2, true);
    builder.add_container("tmp1", base_desc);
    builder.add_container("tmp2", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}},
        {"B", {indvar_i, indvar_j}},
        {"C", {indvar_i, indvar_j}},
        {"D", {indvar_i, indvar_j}}};

    auto& block1 = builder.add_block(body_j);
    auto& A = builder.add_access(block1, "A");
    auto& B = builder.add_access(block1, "B");
    auto& C = builder.add_access(block1, "C");
    auto& D1 = builder.add_access(block1, "D");
    auto& D2 = builder.add_access(block1, "D");
    auto& tmp1 = builder.add_access(block1, "tmp1");
    auto& tmp2 = builder.add_access(block1, "tmp2");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", D1, "void", subsets.at("D"));
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block1, B, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block1, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block1, C, "void", tasklet3, "_in2", subsets.at("C"));
    builder.add_memlet(block1, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, D1, "void", tasklet4, "_in1", subsets.at("D"));
    builder.add_memlet(block1, tmp2, "void", tasklet4, "_in2", {});
    builder.add_memlet(block1, tasklet4, "_out", D2, "void", subsets.at("D"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_in2"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("B")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
}

TEST(EinsumLift, MatrixElementwiseMultiplication_3) {
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
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);
    builder.add_container("D", desc2, true);
    builder.add_container("tmp", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    auto& block1 = builder.add_block(body_j);
    auto& D1 = builder.add_access(block1, "D");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", D1, "void", {indvar_i, indvar_j});

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}},
        {"B", {indvar_i, indvar_j}},
        {"C", {indvar_i, indvar_j}},
        {"D", {indvar_i, indvar_j}}};

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& C = builder.add_access(block2, "C");
    auto& D2 = builder.add_access(block2, "D");
    auto& D3 = builder.add_access(block2, "D");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, C, "void", tasklet3, "_in2", subsets.at("C"));
    builder.add_memlet(block2, D2, "void", tasklet3, "_in3", subsets.at("D"));
    builder.add_memlet(block2, tasklet3, "_out", D3, "void", subsets.at("D"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_in2", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("D")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 4);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(3), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, MatrixElementwiseMultiplication_4) {
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
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);
    builder.add_container("D", desc2, true);
    builder.add_container("tmp", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}},
        {"B", {indvar_i, indvar_j}},
        {"C", {indvar_i, indvar_j}},
        {"D", {indvar_i, indvar_j}}};

    auto& block1 = builder.add_block(body_j);
    auto& A = builder.add_access(block1, "A");
    auto& B = builder.add_access(block1, "B");
    auto& C = builder.add_access(block1, "C");
    auto& D1 = builder.add_access(block1, "D");
    auto& D2 = builder.add_access(block1, "D");
    auto& tmp = builder.add_access(block1, "tmp");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", D1, "void", subsets.at("D"));
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet2, "_in1", subsets.at("A"));
    builder.add_memlet(block1, B, "void", tasklet2, "_in2", subsets.at("B"));
    builder.add_memlet(block1, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 =
        builder.add_tasklet(block1, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block1, tmp, "void", tasklet3, "_in1", {});
    builder.add_memlet(block1, C, "void", tasklet3, "_in2", subsets.at("C"));
    builder.add_memlet(block1, D1, "void", tasklet3, "_in3", subsets.at("D"));
    builder.add_memlet(block1, tasklet3, "_out", D2, "void", subsets.at("D"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_in2"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("B")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
}

TEST(EinsumLift, MatrixElementwiseMultiplication_5) {
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
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);
    builder.add_container("D", desc2, true);
    builder.add_container("tmp", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"A", {indvar_i, indvar_j}},
        {"B", {indvar_i, indvar_j}},
        {"C", {indvar_i, indvar_j}},
        {"D", {indvar_i, indvar_j}}};

    auto& block1 = builder.add_block(body_j);
    auto& A = builder.add_access(block1, "A");
    auto& B = builder.add_access(block1, "B");
    auto& C = builder.add_access(block1, "C");
    auto& D = builder.add_access(block1, "D");
    auto& tmp = builder.add_access(block1, "tmp");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet1, "_in1", subsets.at("A"));
    builder.add_memlet(block1, B, "void", tasklet1, "_in2", subsets.at("B"));
    builder.add_memlet(block1, tasklet1, "_out", tmp, "void", {});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp, "void", tasklet2, "_in1", {});
    builder.add_memlet(block1, C, "void", tasklet2, "_in2", subsets.at("C"));
    builder.add_memlet(block1, tasklet2, "_out", D, "void", subsets.at("D"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 5);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_in2"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("B")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
}

TEST(EinsumLift, VectorScaling_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", base_desc, true);
    builder.add_container("c", base_desc, true);
    builder.add_container("d", base_desc, true);
    builder.add_container("e", desc, true);
    builder.add_container("tmp1", base_desc);
    builder.add_container("tmp2", base_desc);
    builder.add_container("tmp3", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    auto& block1 = builder.add_block(body_i);
    auto& e1 = builder.add_access(block1, "e");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", e1, "void", {indvar_i});

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {}}, {"c", {}}, {"d", {}}, {"e", {indvar_i}}};

    auto& block2 = builder.add_block(body_i);
    auto& a = builder.add_access(block2, "a");
    auto& b = builder.add_access(block2, "b");
    auto& c = builder.add_access(block2, "c");
    auto& d = builder.add_access(block2, "d");
    auto& e2 = builder.add_access(block2, "e");
    auto& e3 = builder.add_access(block2, "e");
    auto& tmp1 = builder.add_access(block2, "tmp1");
    auto& tmp2 = builder.add_access(block2, "tmp2");
    auto& tmp3 = builder.add_access(block2, "tmp3");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", subsets.at("a"));
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block2, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, c, "void", tasklet3, "_in2", subsets.at("c"));
    builder.add_memlet(block2, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, tmp2, "void", tasklet4, "_in1", {});
    builder.add_memlet(block2, d, "void", tasklet4, "_in2", subsets.at("d"));
    builder.add_memlet(block2, tasklet4, "_out", tmp3, "void", {});
    auto& tasklet5 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, e2, "void", tasklet5, "_in1", subsets.at("e"));
    builder.add_memlet(block2, tmp3, "void", tasklet5, "_in2", {});
    builder.add_memlet(block2, tasklet5, "_out", e3, "void", subsets.at("e"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 7);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(),
              std::vector<std::string>({"_in0", "_in1", "_in2", "_in3", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("e")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 5);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(3), subsets.at(conn2cont.at("_in3"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(4), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, VectorScaling_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", base_desc, true);
    builder.add_container("c", base_desc, true);
    builder.add_container("d", base_desc, true);
    builder.add_container("e", desc, true);
    builder.add_container("tmp1", base_desc);
    builder.add_container("tmp2", base_desc);
    builder.add_container("tmp3", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {}}, {"c", {}}, {"d", {}}, {"e", {indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& a = builder.add_access(block1, "a");
    auto& b = builder.add_access(block1, "b");
    auto& c = builder.add_access(block1, "c");
    auto& d = builder.add_access(block1, "d");
    auto& e1 = builder.add_access(block1, "e");
    auto& e2 = builder.add_access(block1, "e");
    auto& tmp1 = builder.add_access(block1, "tmp1");
    auto& tmp2 = builder.add_access(block1, "tmp2");
    auto& tmp3 = builder.add_access(block1, "tmp3");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", e1, "void", subsets.at("e"));
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, a, "void", tasklet2, "_in1", subsets.at("a"));
    builder.add_memlet(block1, b, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block1, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block1, c, "void", tasklet3, "_in2", subsets.at("c"));
    builder.add_memlet(block1, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp2, "void", tasklet4, "_in1", {});
    builder.add_memlet(block1, d, "void", tasklet4, "_in2", subsets.at("d"));
    builder.add_memlet(block1, tasklet4, "_out", tmp3, "void", {});
    auto& tasklet5 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, e1, "void", tasklet5, "_in1", subsets.at("e"));
    builder.add_memlet(block1, tmp3, "void", tasklet5, "_in2", {});
    builder.add_memlet(block1, tasklet5, "_out", e2, "void", subsets.at("e"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_in2", "_in3"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("e")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 4);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(3), subsets.at(conn2cont.at("_in3"))));
}

TEST(EinsumLift, VectorScaling_3) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", base_desc, true);
    builder.add_container("c", base_desc, true);
    builder.add_container("d", base_desc, true);
    builder.add_container("e", desc, true);
    builder.add_container("tmp1", base_desc);
    builder.add_container("tmp2", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    auto& block1 = builder.add_block(body_i);
    auto& e1 = builder.add_access(block1, "e");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", e1, "void", {indvar_i});

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {}}, {"c", {}}, {"d", {}}, {"e", {indvar_i}}};

    auto& block2 = builder.add_block(body_i);
    auto& a = builder.add_access(block2, "a");
    auto& b = builder.add_access(block2, "b");
    auto& c = builder.add_access(block2, "c");
    auto& d = builder.add_access(block2, "d");
    auto& e2 = builder.add_access(block2, "e");
    auto& e3 = builder.add_access(block2, "e");
    auto& tmp1 = builder.add_access(block2, "tmp1");
    auto& tmp2 = builder.add_access(block2, "tmp2");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", subsets.at("a"));
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block2, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, c, "void", tasklet3, "_in2", subsets.at("c"));
    builder.add_memlet(block2, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, tmp2, "void", tasklet4, "_in1", {});
    builder.add_memlet(block2, d, "void", tasklet4, "_in2", subsets.at("d"));
    builder.add_memlet(block2, e2, "void", tasklet4, "_in3", subsets.at("e"));
    builder.add_memlet(block2, tasklet4, "_out", e3, "void", subsets.at("e"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 2);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 7);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(),
              std::vector<std::string>({"_in0", "_in1", "_in2", "_in3", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("e")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 5);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(3), subsets.at(conn2cont.at("_in3"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(4), subsets.at(conn2cont.at("_out"))));
}

TEST(EinsumLift, VectorScaling_4) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", base_desc, true);
    builder.add_container("c", base_desc, true);
    builder.add_container("d", base_desc, true);
    builder.add_container("e", desc, true);
    builder.add_container("tmp1", base_desc);
    builder.add_container("tmp2", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {}}, {"c", {}}, {"d", {}}, {"e", {indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& a = builder.add_access(block1, "a");
    auto& b = builder.add_access(block1, "b");
    auto& c = builder.add_access(block1, "c");
    auto& d = builder.add_access(block1, "d");
    auto& e1 = builder.add_access(block1, "e");
    auto& e2 = builder.add_access(block1, "e");
    auto& tmp1 = builder.add_access(block1, "tmp1");
    auto& tmp2 = builder.add_access(block1, "tmp2");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", e1, "void", subsets.at("e"));
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, a, "void", tasklet2, "_in1", subsets.at("a"));
    builder.add_memlet(block1, b, "void", tasklet2, "_in2", subsets.at("b"));
    builder.add_memlet(block1, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block1, c, "void", tasklet3, "_in2", subsets.at("c"));
    builder.add_memlet(block1, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 =
        builder.add_tasklet(block1, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block1, tmp2, "void", tasklet4, "_in1", {});
    builder.add_memlet(block1, d, "void", tasklet4, "_in2", subsets.at("d"));
    builder.add_memlet(block1, e1, "void", tasklet4, "_in3", subsets.at("e"));
    builder.add_memlet(block1, tasklet4, "_out", e2, "void", subsets.at("e"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_in2", "_in3"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("e")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 4);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(3), subsets.at(conn2cont.at("_in3"))));
}

TEST(EinsumLift, VectorScaling_5) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", base_desc, true);
    builder.add_container("c", base_desc, true);
    builder.add_container("d", base_desc, true);
    builder.add_container("e", desc, true);
    builder.add_container("tmp1", base_desc);
    builder.add_container("tmp2", base_desc);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {}}, {"c", {}}, {"d", {}}, {"e", {indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& a = builder.add_access(block1, "a");
    auto& b = builder.add_access(block1, "b");
    auto& c = builder.add_access(block1, "c");
    auto& d = builder.add_access(block1, "d");
    auto& e = builder.add_access(block1, "e");
    auto& tmp1 = builder.add_access(block1, "tmp1");
    auto& tmp2 = builder.add_access(block1, "tmp2");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, a, "void", tasklet1, "_in1", subsets.at("a"));
    builder.add_memlet(block1, b, "void", tasklet1, "_in2", subsets.at("b"));
    builder.add_memlet(block1, tasklet1, "_out", tmp1, "void", {});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp1, "void", tasklet2, "_in1", {});
    builder.add_memlet(block1, c, "void", tasklet2, "_in2", subsets.at("c"));
    builder.add_memlet(block1, tasklet2, "_out", tmp2, "void", {});
    auto& tasklet3 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp2, "void", tasklet3, "_in1", {});
    builder.add_memlet(block1, d, "void", tasklet3, "_in2", subsets.at("d"));
    builder.add_memlet(block1, tasklet3, "_out", e, "void", subsets.at("e"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 6);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_in1", "_in2", "_in3"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("e")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 4);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_in1"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), subsets.at(conn2cont.at("_in2"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(3), subsets.at(conn2cont.at("_in3"))));
}

TEST(EinsumLift, Means_1) {
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

    auto& block1 = builder.add_block(body_j);
    auto& b1 = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", b1, "void", {indvar_j});

    gen_for(i, I, body_j);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {{"A", {indvar_i, indvar_j}},
                                                                        {"b", {indvar_j}}};

    auto& block2 = builder.add_block(body_i);
    auto& A = builder.add_access(block2, "A");
    auto& b2 = builder.add_access(block2, "b");
    auto& b3 = builder.add_access(block2, "b");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, b2, "void", tasklet2, "_in1", subsets.at("b"));
    builder.add_memlet(block2, A, "void", tasklet2, "_in2", subsets.at("A"));
    builder.add_memlet(block2, tasklet2, "_out", b3, "void", subsets.at("b"));

    auto& block3 = builder.add_block(body_j);
    auto& b4 = builder.add_access(block3, "b");
    auto& b5 = builder.add_access(block3, "b");
    auto& I = builder.add_access(block3, "I");
    auto& tasklet3 = builder.add_tasklet(block3, data_flow::TaskletCode::div, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", sym_desc}});
    builder.add_memlet(block3, b4, "void", tasklet3, "_in1", subsets.at("b"));
    builder.add_memlet(block3, I, "void", tasklet3, "_in2", {});
    builder.add_memlet(block3, tasklet3, "_out", b5, "void", subsets.at("b"));

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i}, block2);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 3);
    auto* block1_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block1_opt);
    EXPECT_EQ(block1_opt, &block1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(1).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 4);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 1);
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).first, indvar_i));
    EXPECT_TRUE(symbolic::eq(einsum_node->map(0).second, bound_i));
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), subsets.at("b")));
    auto conn2cont = get_conn2cont(*block_einsum, *libnode);
    AT_LEAST(einsum_node->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), subsets.at(conn2cont.at("_in0"))));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), subsets.at(conn2cont.at("_out"))));
    auto* block3_opt =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(2).first);
    ASSERT_TRUE(block3_opt);
    EXPECT_EQ(block3_opt, &block3);
}

TEST(EinsumLift, ConstScaling_1) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    auto& block1 = builder.add_block(body_i);
    auto& a = builder.add_access(block1, "a");
    auto& b = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in", base_desc}, {"0.5", base_desc}});
    builder.add_memlet(block1, a, "void", tasklet1, "_in", {indvar_i});
    builder.add_memlet(block1, tasklet1, "_out", b, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 3);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "0.5"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), {indvar_i}));
    AT_LEAST(einsum_node->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), {indvar_i}));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), {}));
}

TEST(EinsumLift, ConstScaling_2) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    auto& block1 = builder.add_block(body_i);
    auto& a = builder.add_access(block1, "a");
    auto& b = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"2", base_desc}, {"_in", base_desc}});
    builder.add_memlet(block1, a, "void", tasklet1, "_in", {indvar_i});
    builder.add_memlet(block1, tasklet1, "_out", b, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 3);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "2"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), {indvar_i}));
    AT_LEAST(einsum_node->in_indices().size(), 2);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), {indvar_i}));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), {}));
}

TEST(EinsumLift, Subtraction) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("a", desc, true);
    builder.add_container("b", desc2, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    auto& block1 = builder.add_block(body_j);
    auto& a = builder.add_access(block1, "a");
    auto& b1 = builder.add_access(block1, "b");
    auto& b2 = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::sub, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, b1, "void", tasklet1, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block1, a, "void", tasklet1, "_in2", {indvar_j});
    builder.add_memlet(block1, tasklet1, "_out", b2, "void", {indvar_i, indvar_j});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block1);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* for_i_opt = dynamic_cast<structured_control_flow::For*>(&root_opt.at(0).first);
    ASSERT_TRUE(for_i_opt);
    EXPECT_EQ(for_i_opt, &for_i);
    AT_LEAST(for_i_opt->root().size(), 1);
    auto* for_j_opt = dynamic_cast<structured_control_flow::For*>(&for_i_opt->root().at(0).first);
    ASSERT_TRUE(for_j_opt);
    EXPECT_EQ(for_j_opt, &for_j);
    AT_LEAST(for_j_opt->root().size(), 1);
    auto* block_einsum =
        dynamic_cast<structured_control_flow::Block*>(&for_j_opt->root().at(0).first);
    ASSERT_TRUE(block_einsum);
    AT_LEAST(block_einsum->dataflow().nodes().size(), 4);
    data_flow::LibraryNode* libnode = nullptr;
    for (auto& node : block_einsum->dataflow().nodes()) {
        if ((libnode = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode);
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(libnode);
    ASSERT_TRUE(einsum_node);
    EXPECT_EQ(einsum_node->outputs(), std::vector<std::string>({"_out"}));
    EXPECT_EQ(einsum_node->inputs(), std::vector<std::string>({"_in0", "-1", "_out"}));
    AT_LEAST(einsum_node->maps().size(), 0);
    EXPECT_TRUE(subsets_eq(einsum_node->out_indices(), {indvar_i, indvar_j}));
    AT_LEAST(einsum_node->in_indices().size(), 3);
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(0), {indvar_j}));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(1), {}));
    EXPECT_TRUE(subsets_eq(einsum_node->in_indices(2), {indvar_i, indvar_j}));
}