#include "sdfg/transformations/einsum_expand.h"

#include <gtest/gtest.h>
#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/code_generators/c_code_generator.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/element.h>
#include <sdfg/function.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sdfg/einsum/einsum_node.h"

using namespace sdfg;

#define gen_for(index, bound, root_)                                              \
    auto indvar_##index = symbolic::symbol(#index);                               \
    auto bound_##index = symbolic::symbol(#bound);                                \
    auto condition_##index = symbolic::Lt(indvar_##index, bound_##index);         \
    auto update_##index = symbolic::add(indvar_##index, symbolic::one());         \
    auto& for_##index = builder.add_for(root_, indvar_##index, condition_##index, \
                                        symbolic::zero(), update_##index);        \
    auto& body_##index = for_##index.root();

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
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block2, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in0", "_in1", "_out"}, false,
            DebugInfo(), {{indvar_j, bound_j}}, subsets.at("C"),
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

    transformations::EinsumExpand transformation(for_k, *einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto sdfg_opt = builder_opt.move();
    codegen::CCodeGenerator generator(*sdfg_opt);
    EXPECT_TRUE(generator.generate());
    std::cout << generator.function_definition() << " {" << std::endl
              << generator.main().str() << "}" << std::endl;
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
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block1, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in0", "_in1", "_out"}, false,
            DebugInfo(), {{indvar_j, bound_j}}, subsets.at("C"),
            {subsets.at("A"), subsets.at("B"), subsets.at("C")});
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

    auto sdfg_opt = builder_opt.move();
    codegen::CCodeGenerator generator(*sdfg_opt);
    EXPECT_TRUE(generator.generate());
    std::cout << generator.function_definition() << " {" << std::endl
              << generator.main().str() << "}" << std::endl;
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
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block2, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in0", "_in1", "_out"}, false,
            DebugInfo(), {{indvar_j, bound_j}, {indvar_k, bound_k}}, subsets.at("C"),
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

    auto sdfg_opt = builder_opt.move();
    codegen::CCodeGenerator generator(*sdfg_opt);
    EXPECT_TRUE(generator.generate());
    std::cout << generator.function_definition() << " {" << std::endl
              << generator.main().str() << "}" << std::endl;
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
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block2, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in0", "_out"}, false, DebugInfo(),
            {}, subsets.at("b"), {subsets.at("A"), subsets.at("b")});
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

    auto sdfg_opt = builder_opt.move();
    codegen::CCodeGenerator generator(*sdfg_opt);
    EXPECT_TRUE(generator.generate());
    std::cout << generator.function_definition() << " {" << std::endl
              << generator.main().str() << "}" << std::endl;
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
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block1, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in0", "_out"}, false, DebugInfo(),
            {}, subsets.at("b"), {subsets.at("A"), subsets.at("b")});
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

    auto sdfg_opt = builder_opt.move();
    codegen::CCodeGenerator generator(*sdfg_opt);
    EXPECT_TRUE(generator.generate());
    std::cout << generator.function_definition() << " {" << std::endl
              << generator.main().str() << "}" << std::endl;
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
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block1, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in0"}, false, DebugInfo(), {},
            subsets.at("b"), {subsets.at("A")});
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

    auto sdfg_opt = builder_opt.move();
    codegen::CCodeGenerator generator(*sdfg_opt);
    EXPECT_TRUE(generator.generate());
    std::cout << generator.function_definition() << " {" << std::endl
              << generator.main().str() << "}" << std::endl;
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
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block2, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in0", "_out"}, false, DebugInfo(),
            {{indvar_i, bound_i}}, subsets.at("b"), {subsets.at("A"), subsets.at("b")});
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

    auto sdfg_opt = builder_opt.move();
    codegen::CCodeGenerator generator(*sdfg_opt);
    EXPECT_TRUE(generator.generate());
    std::cout << generator.function_definition() << " {" << std::endl
              << generator.main().str() << "}" << std::endl;
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
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block1, einsum::LibraryNodeType_Einsum, {"_out"}, {"_in0", "_out"}, false, DebugInfo(),
            {{indvar_i, bound_i}}, subsets.at("b"), {subsets.at("A"), subsets.at("b")});
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

    auto sdfg_opt = builder_opt.move();
    codegen::CCodeGenerator generator(*sdfg_opt);
    EXPECT_TRUE(generator.generate());
    std::cout << generator.function_definition() << " {" << std::endl
              << generator.main().str() << "}" << std::endl;
}
