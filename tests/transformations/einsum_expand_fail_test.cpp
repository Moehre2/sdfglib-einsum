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
#include "sdfg/transformations/einsum_expand.h"

using namespace sdfg;

TEST(EinsumExpandFail, TempValueDependency_1) {
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

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {indvar_i}}, {"c", {indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& a = builder.add_access(block1, "a");
    auto& tmp1 = builder.add_access(block1, "tmp");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in", base_desc}, {"2.0", base_desc}});
    builder.add_memlet(block1, a, "void", tasklet1, "_in", subsets.at("a"));
    builder.add_memlet(block1, tasklet1, "_out", tmp1, "void", {});

    auto& block2 = builder.add_block(body_i);
    auto& b = builder.add_access(block2, "b");
    auto& c = builder.add_access(block2, "c");
    auto& tmp2 = builder.add_access(block2, "tmp");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block2, DebugInfo(), {"_out"}, {"_in1", "_in2"}, {}, subsets.at("c"),
            {subsets.at("b"), {}});
    builder.add_memlet(block2, b, "void", libnode, "_in1", {});
    builder.add_memlet(block2, tmp2, "void", libnode, "_in2", {});
    builder.add_memlet(block2, libnode, "_out", c, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_i, *einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

TEST(EinsumExpandFail, TempValueDependency_2) {
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

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {indvar_i}}, {"c", {indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& a = builder.add_access(block1, "a");
    auto& b = builder.add_access(block1, "b");
    auto& c = builder.add_access(block1, "c");
    auto& tmp = builder.add_access(block1, "tmp");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in", base_desc}, {"2.0", base_desc}});
    builder.add_memlet(block1, a, "void", tasklet1, "_in", subsets.at("a"));
    builder.add_memlet(block1, tasklet1, "_out", tmp, "void", {});
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block1, DebugInfo(), {"_out"}, {"_in1", "_in2"}, {}, subsets.at("c"),
            {subsets.at("b"), {}});
    builder.add_memlet(block1, b, "void", libnode, "_in1", {});
    builder.add_memlet(block1, tmp, "void", libnode, "_in2", {});
    builder.add_memlet(block1, libnode, "_out", c, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_i, *einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

TEST(EinsumExpandFail, TempValueDependency_3) {
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

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {indvar_i}}, {"c", {indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& a = builder.add_access(block1, "a");
    auto& b = builder.add_access(block1, "b");
    auto& tmp1 = builder.add_access(block1, "tmp");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block1, DebugInfo(), {"_out"}, {"_in1", "_in2"}, {}, {},
            {subsets.at("a"), subsets.at("b")});
    builder.add_memlet(block1, a, "void", libnode, "_in1", {});
    builder.add_memlet(block1, b, "void", libnode, "_in2", {});
    builder.add_memlet(block1, libnode, "_out", tmp1, "void", {});

    auto& block2 = builder.add_block(body_i);
    auto& c = builder.add_access(block2, "c");
    auto& tmp2 = builder.add_access(block2, "tmp");
    auto& tasklet1 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in", base_desc}, {"2.0", base_desc}});
    builder.add_memlet(block2, tmp2, "void", tasklet1, "_in", {});
    builder.add_memlet(block2, tasklet1, "_out", c, "void", subsets.at("c"));

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_i, *einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

TEST(EinsumExpandFail, TempValueDependency_4) {
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

    gen_for(i, I, root);

    const std::unordered_map<std::string, data_flow::Subset> subsets = {
        {"a", {indvar_i}}, {"b", {indvar_i}}, {"c", {indvar_i}}};

    auto& block1 = builder.add_block(body_i);
    auto& a = builder.add_access(block1, "a");
    auto& b = builder.add_access(block1, "b");
    auto& c = builder.add_access(block1, "c");
    auto& tmp = builder.add_access(block1, "tmp");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block1, DebugInfo(), {"_out"}, {"_in1", "_in2"}, {}, {},
            {subsets.at("a"), subsets.at("b")});
    builder.add_memlet(block1, a, "void", libnode, "_in1", {});
    builder.add_memlet(block1, b, "void", libnode, "_in2", {});
    builder.add_memlet(block1, libnode, "_out", tmp, "void", {});
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in", base_desc}, {"2.0", base_desc}});
    builder.add_memlet(block1, tmp, "void", tasklet1, "_in", {});
    builder.add_memlet(block1, tasklet1, "_out", c, "void", subsets.at("c"));

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumExpand transformation(for_i, *einsum_node);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}
