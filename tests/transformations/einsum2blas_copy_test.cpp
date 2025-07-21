#include "sdfg/transformations/einsum2blas_copy.h"

#include <gtest/gtest.h>
#include <sdfg/builder/structured_sdfg_builder.h>
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
#include "sdfg/blas/blas_node_copy.h"
#include "sdfg/einsum/einsum_node.h"

using namespace sdfg;

TEST(Einsum2BLASCopy, scopy) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& x = builder.add_access(block, "x");
    auto& y = builder.add_access(block, "y");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0"}, {{indvar_i, bound_i}}, {indvar_i},
            {{indvar_i}});
    builder.add_memlet(block, x, "void", libnode, "_in0", {});
    builder.add_memlet(block, libnode, "_out", y, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    ASSERT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASCopy transformation(*einsum_node);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 3);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeCopy*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_real);
    EXPECT_EQ(blas_node->x(), "_in0");
    EXPECT_EQ(blas_node->y(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_i));
}

TEST(Einsum2BLASCopy, dcopy) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Double);
    types::Pointer desc(base_desc);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& x = builder.add_access(block, "x");
    auto& y = builder.add_access(block, "y");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in0"}, {{indvar_i, bound_i}}, {indvar_i},
            {{indvar_i}});
    builder.add_memlet(block, x, "void", libnode, "_in0", {});
    builder.add_memlet(block, libnode, "_out", y, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    ASSERT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLASCopy transformation(*einsum_node);
    ASSERT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto& root_opt = builder_opt.subject().root();
    AT_LEAST(root_opt.size(), 1);
    auto* block_opt = dynamic_cast<structured_control_flow::Block*>(&root_opt.at(0).first);
    ASSERT_TRUE(block_opt);
    EXPECT_EQ(block_opt, &block);
    AT_LEAST(block_opt->dataflow().nodes().size(), 3);
    data_flow::LibraryNode* libnode_opt = nullptr;
    for (auto& node : block_opt->dataflow().nodes()) {
        if ((libnode_opt = dynamic_cast<data_flow::LibraryNode*>(&node))) break;
    }
    ASSERT_TRUE(libnode_opt);
    auto* blas_node = dynamic_cast<blas::BLASNodeCopy*>(libnode_opt);
    ASSERT_TRUE(blas_node);
    EXPECT_EQ(blas_node->type(), blas::BLASType_double);
    EXPECT_EQ(blas_node->x(), "_in0");
    EXPECT_EQ(blas_node->y(), "_out");
    EXPECT_TRUE(symbolic::eq(blas_node->n(), bound_i));
}