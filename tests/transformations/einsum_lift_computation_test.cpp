#include "sdfg/transformations/einsum_lift_computation.h"

#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

using namespace sdfg;

#define gen_for(index, bound, root_)                                              \
    auto indvar_##index = symbolic::symbol(#index);                               \
    auto bound_##index = symbolic::symbol(#bound);                                \
    auto condition_##index = symbolic::Lt(indvar_##index, bound_##index);         \
    auto update_##index = symbolic::add(indvar_##index, symbolic::one());         \
    auto& for_##index = builder.add_for(root_, indvar_##index, condition_##index, \
                                        symbolic::zero(), update_##index);        \
    auto& body_##index = for_##index.root();

TEST(EinsumLiftComputation, DiagonalExtraction_1) {
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
    auto& A = builder.add_access(block1, "A");
    auto& b = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet1, "_in", {indvar_i, indvar_i});
    builder.add_memlet(block1, tasklet1, "_out", b, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLiftComputation transformation(block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
}

TEST(EinsumLiftComputation, MatrixCopy_1) {
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
    auto& A = builder.add_access(block1, "A");
    auto& B = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet1, "_in", {indvar_i, indvar_j});
    builder.add_memlet(block1, tasklet1, "_out", B, "void", {indvar_i, indvar_j});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLiftComputation transformation(block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
}

TEST(EinsumLiftComputation, MatrixTranspose_1) {
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
    auto& A = builder.add_access(block1, "A");
    auto& B = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"_in", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet1, "_in", {indvar_i, indvar_j});
    builder.add_memlet(block1, tasklet1, "_out", B, "void", {indvar_j, indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLiftComputation transformation(block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
}

TEST(EinsumLiftComputation, MatrixElementwiseMultiplication_1) {
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
    auto& A = builder.add_access(block1, "A");
    auto& B = builder.add_access(block1, "B");
    auto& C = builder.add_access(block1, "C");
    auto& D = builder.add_access(block1, "D");
    auto& tmp = builder.add_access(block1, "tmp");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet1, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block1, B, "void", tasklet1, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block1, tasklet1, "_out", tmp, "void", {});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp, "void", tasklet2, "_in1", {});
    builder.add_memlet(block1, C, "void", tasklet2, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block1, tasklet2, "_out", D, "void", {indvar_i, indvar_j});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLiftComputation transformation(block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
}

TEST(EinsumLiftComputation, VectorScaling_1) {
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
    auto& a = builder.add_access(block1, "a");
    auto& b = builder.add_access(block1, "b");
    auto& c = builder.add_access(block1, "c");
    auto& d = builder.add_access(block1, "d");
    auto& e = builder.add_access(block1, "e");
    auto& tmp1 = builder.add_access(block1, "tmp1");
    auto& tmp2 = builder.add_access(block1, "tmp2");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, a, "void", tasklet1, "_in1", {indvar_i});
    builder.add_memlet(block1, b, "void", tasklet1, "_in2", {});
    builder.add_memlet(block1, tasklet1, "_out", tmp1, "void", {});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp1, "void", tasklet2, "_in1", {});
    builder.add_memlet(block1, c, "void", tasklet2, "_in2", {});
    builder.add_memlet(block1, tasklet2, "_out", tmp2, "void", {});
    auto& tasklet3 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp2, "void", tasklet3, "_in1", {});
    builder.add_memlet(block1, d, "void", tasklet3, "_in2", {});
    builder.add_memlet(block1, tasklet3, "_out", e, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLiftComputation transformation(block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
}
