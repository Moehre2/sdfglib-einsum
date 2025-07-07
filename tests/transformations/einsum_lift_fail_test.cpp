#include <gtest/gtest.h>
#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/function.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include "sdfg/transformations/einsum_lift.h"

using namespace sdfg;

#define gen_for(index, bound, root_)                                              \
    auto indvar_##index = symbolic::symbol(#index);                               \
    auto bound_##index = symbolic::symbol(#bound);                                \
    auto condition_##index = symbolic::Lt(indvar_##index, bound_##index);         \
    auto update_##index = symbolic::add(indvar_##index, symbolic::one());         \
    auto& for_##index = builder.add_for(root_, indvar_##index, condition_##index, \
                                        symbolic::zero(), update_##index);        \
    auto& body_##index = for_##index.root();

TEST(EinsumLiftFail, loops_not_nested) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, root);

    auto& block1 = builder.add_block(root);
    auto& i = builder.add_access(block1, "i");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out", sym_desc},
                                         {{"0", sym_desc}});
    builder.add_memlet(block1, tasklet1, "_out", i, "void", {});

    auto& block2 = builder.add_block(root);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i, for_j}, block2);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

TEST(EinsumLiftFail, comp_block_outside_loops) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("j", sym_desc);
    builder.add_container("J", sym_desc, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    gen_for(j, J, body_i);

    auto& block1 = builder.add_block(root);
    auto& i = builder.add_access(block1, "i");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out", sym_desc},
                                         {{"0", sym_desc}});
    builder.add_memlet(block1, tasklet1, "_out", i, "void", {});

    auto& block2 = builder.add_block(body_i);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i, for_j}, block2);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

TEST(EinsumLiftFail, init_tasklet_not_assign) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    auto& block1 = builder.add_block(root);
    auto& i1 = builder.add_access(block1, "i");
    auto& i2 = builder.add_access(block1, "i");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", sym_desc},
                                         {{"_in", sym_desc}, {"1", sym_desc}});
    builder.add_memlet(block1, i1, "void", tasklet1, "_in", {});
    builder.add_memlet(block1, tasklet1, "_out", i2, "void", {});

    auto& block2 = builder.add_block(root);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block2);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

TEST(EinsumLiftFail, not_zero_initialized) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    auto& root = builder.subject().root();

    auto& block1 = builder.add_block(root);
    auto& i = builder.add_access(block1, "i");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign, {"_out", sym_desc},
                                         {{"1", sym_desc}});
    builder.add_memlet(block1, tasklet1, "_out", i, "void", {});

    auto& block2 = builder.add_block(root);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({}, block2);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

TEST(EinsumLiftFail, loop_in_init) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);

    auto& root = builder.subject().root();

    auto indvar_i_ = symbolic::symbol("i");
    auto bound_i_ = symbolic::symbol("I");
    auto condition_i_ = symbolic::Lt(indvar_i_, bound_i_);
    auto update_i_ = symbolic::add(indvar_i_, symbolic ::one());
    auto& for_i_ = builder.add_for(root, indvar_i_, condition_i_, symbolic::one(), update_i_);
    auto& body_i_ = for_i_.root();

    auto& block1 = builder.add_block(body_i_);
    auto& a = builder.add_access(block1, "a");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", a, "void",
                       {symbolic::sub(indvar_i_, symbolic::one())});

    gen_for(i, I, root);

    auto& block2 = builder.add_block(body_i);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i}, block2);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

TEST(EinsumLiftFail, init_in_loop) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);

    auto& root = builder.subject().root();

    gen_for(i, I, root);

    auto& block1 = builder.add_block(body_i);
    auto& a = builder.add_access(block1, "a");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", a, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i}, block1);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

TEST(EinsumLiftFail, invalid_loop_start) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);
    builder.add_container("z", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("a", base_desc, true);

    auto& root = builder.subject().root();

    auto& block1 = builder.add_block(root);
    auto& a = builder.add_access(block1, "a");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", a, "void", {});

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::symbol("I");
    auto condition_i = symbolic::Lt(indvar_i, bound_i);
    auto update_i = symbolic::add(indvar_i, symbolic::one());
    auto& for_i = builder.add_for(root, indvar_i, condition_i, symbolic::symbol("z"), update_i);
    auto& body_i = for_i.root();

    auto& block2 = builder.add_block(body_i);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i}, block1);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}

TEST(EinsumLiftFail, invalid_loop_condition) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);

    types::Scalar base_desc(types::PrimitiveType::Float);
    builder.add_container("a", base_desc, true);

    auto& root = builder.subject().root();

    auto& block1 = builder.add_block(root);
    auto& a = builder.add_access(block1, "a");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", a, "void", {});

    auto indvar_i = symbolic::symbol("i");
    auto bound_i = symbolic::zero();
    auto condition_i = symbolic::Gt(indvar_i, bound_i);
    auto update_i = symbolic::sub(indvar_i, symbolic::one());
    auto& for_i = builder.add_for(root, indvar_i, condition_i, symbolic::integer(10), update_i);
    auto& body_i = for_i.root();

    auto& block2 = builder.add_block(body_i);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation({for_i}, block1);
    EXPECT_FALSE(transformation.can_be_applied(builder_opt, analysis_manager));
}