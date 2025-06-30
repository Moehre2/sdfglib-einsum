#include "sdfg/transformations/einsum_lift.h"

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

#define gen_map(index, bound, root_)

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

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& C2 = builder.add_access(block2, "C");
    auto& C3 = builder.add_access(block2, "C");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", {indvar_j, indvar_k});
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, C2, "void", tasklet3, "_in1", {indvar_i, indvar_k});
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in2", {});
    builder.add_memlet(block2, tasklet3, "_out", C3, "void", {indvar_i, indvar_k});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_j}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& C2 = builder.add_access(block2, "C");
    auto& C3 = builder.add_access(block2, "C");
    auto& tasklet2 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", {indvar_j, indvar_k});
    builder.add_memlet(block2, C2, "void", tasklet2, "_in3", {indvar_i, indvar_k});
    builder.add_memlet(block2, tasklet2, "_out", C3, "void", {indvar_i, indvar_k});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_j}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_l, indvar_j, indvar_m});
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", {indvar_i, indvar_l, indvar_n});
    builder.add_memlet(block2, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, C, "void", tasklet3, "_in2", {indvar_n, indvar_m, indvar_k});
    builder.add_memlet(block2, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, D2, "void", tasklet4, "_in1", {indvar_i, indvar_j, indvar_k});
    builder.add_memlet(block2, tmp2, "void", tasklet4, "_in2", {});
    builder.add_memlet(block2, tasklet4, "_out", D3, "void", {indvar_i, indvar_j, indvar_k});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_l, for_m, for_n}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_n);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& C = builder.add_access(block2, "C");
    auto& D2 = builder.add_access(block2, "D");
    auto& D3 = builder.add_access(block2, "D");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_l, indvar_j, indvar_m});
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", {indvar_i, indvar_l, indvar_n});
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, C, "void", tasklet3, "_in2", {indvar_n, indvar_m, indvar_k});
    builder.add_memlet(block2, D2, "void", tasklet3, "_in3", {indvar_i, indvar_j, indvar_k});
    builder.add_memlet(block2, tasklet3, "_out", D3, "void", {indvar_i, indvar_j, indvar_k});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_l, for_m, for_n}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", {indvar_j});
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, c2, "void", tasklet3, "_in1", {indvar_i});
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in2", {});
    builder.add_memlet(block2, tasklet3, "_out", c3, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_j}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tasklet2 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", {indvar_j});
    builder.add_memlet(block2, c2, "void", tasklet2, "_in3", {indvar_i});
    builder.add_memlet(block2, tasklet2, "_out", c3, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_j}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_i);
    auto& A = builder.add_access(block2, "A");
    auto& b2 = builder.add_access(block2, "b");
    auto& b3 = builder.add_access(block2, "b");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_i, indvar_i});
    builder.add_memlet(block2, b2, "void", tasklet2, "_in2", {indvar_i});
    builder.add_memlet(block2, tasklet2, "_out", b3, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block1 = builder.add_block(body_i);
    auto& A = builder.add_access(block1, "A");
    auto& b1 = builder.add_access(block1, "b");
    auto& b2 = builder.add_access(block1, "b");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", b1, "void", {indvar_i});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet2, "_in1", {indvar_i, indvar_i});
    builder.add_memlet(block1, b1, "void", tasklet2, "_in2", {indvar_i});
    builder.add_memlet(block1, tasklet2, "_out", b2, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_i);
    auto& A = builder.add_access(block2, "A");
    auto& b2 = builder.add_access(block2, "b");
    auto& b3 = builder.add_access(block2, "b");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_i, indvar_i});
    builder.add_memlet(block2, b2, "void", tasklet2, "_in2", {});
    builder.add_memlet(block2, tasklet2, "_out", b3, "void", {});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_i}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_i);
    auto& A = builder.add_access(block2, "A");
    auto& b2 = builder.add_access(block2, "b");
    auto& b3 = builder.add_access(block2, "b");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_i, indvar_i});
    builder.add_memlet(block2, b2, "void", tasklet2, "_in2", {symbolic::zero()});
    builder.add_memlet(block2, tasklet2, "_out", b3, "void", {symbolic::zero()});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_i}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& B2 = builder.add_access(block2, "B");
    auto& B3 = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block2, B2, "void", tasklet2, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block2, tasklet2, "_out", B3, "void", {indvar_i, indvar_j});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block1 = builder.add_block(body_j);
    auto& A = builder.add_access(block1, "A");
    auto& B1 = builder.add_access(block1, "B");
    auto& B2 = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", B1, "void", {indvar_i, indvar_j});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block1, B1, "void", tasklet2, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block1, tasklet2, "_out", B2, "void", {indvar_i, indvar_j});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_i);
    auto& A = builder.add_access(block2, "A");
    auto& B2 = builder.add_access(block2, "B");
    auto& B3 = builder.add_access(block2, "B");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block2, B2, "void", tasklet2, "_in2", {indvar_j, indvar_i});
    builder.add_memlet(block2, tasklet2, "_out", B3, "void", {indvar_j, indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block1 = builder.add_block(body_i);
    auto& A = builder.add_access(block1, "A");
    auto& B1 = builder.add_access(block1, "B");
    auto& B2 = builder.add_access(block1, "B");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", B1, "void", {indvar_j, indvar_i});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block1, B1, "void", tasklet2, "_in2", {indvar_j, indvar_i});
    builder.add_memlet(block1, tasklet2, "_out", B2, "void", {indvar_j, indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_i);
    auto& a = builder.add_access(block2, "a");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", {indvar_i});
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", {indvar_i});
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, c2, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in2", {});
    builder.add_memlet(block2, tasklet3, "_out", c3, "void", {});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_i}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_i);
    auto& a = builder.add_access(block2, "a");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", {indvar_i});
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", {indvar_i});
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, c2, "void", tasklet3, "_in1", {symbolic::zero()});
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in2", {});
    builder.add_memlet(block2, tasklet3, "_out", c3, "void", {symbolic::zero()});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_i}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_i);
    auto& a = builder.add_access(block2, "a");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tasklet2 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", {indvar_i});
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", {indvar_i});
    builder.add_memlet(block2, c2, "void", tasklet2, "_in3", {});
    builder.add_memlet(block2, tasklet2, "_out", c3, "void", {});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_i}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_i);
    auto& a = builder.add_access(block2, "a");
    auto& b = builder.add_access(block2, "b");
    auto& c2 = builder.add_access(block2, "c");
    auto& c3 = builder.add_access(block2, "c");
    auto& tasklet2 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", {indvar_i});
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", {indvar_i});
    builder.add_memlet(block2, c2, "void", tasklet2, "_in3", {symbolic::zero()});
    builder.add_memlet(block2, tasklet2, "_out", c3, "void", {symbolic::zero()});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {for_i}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block2, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, C, "void", tasklet3, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block2, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, D2, "void", tasklet4, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block2, tmp2, "void", tasklet4, "_in2", {});
    builder.add_memlet(block2, tasklet4, "_out", D3, "void", {indvar_i, indvar_j});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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
    builder.add_memlet(block1, tasklet1, "_out", D1, "void", {indvar_i, indvar_j});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block1, B, "void", tasklet2, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block1, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block1, C, "void", tasklet3, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block1, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, D1, "void", tasklet4, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block1, tmp2, "void", tasklet4, "_in2", {});
    builder.add_memlet(block1, tasklet4, "_out", D2, "void", {indvar_i, indvar_j});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block2 = builder.add_block(body_j);
    auto& A = builder.add_access(block2, "A");
    auto& B = builder.add_access(block2, "B");
    auto& C = builder.add_access(block2, "C");
    auto& D2 = builder.add_access(block2, "D");
    auto& D3 = builder.add_access(block2, "D");
    auto& tmp = builder.add_access(block2, "tmp");
    auto& tasklet2 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block2, B, "void", tasklet2, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block2, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, tmp, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, C, "void", tasklet3, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block2, D2, "void", tasklet3, "_in3", {indvar_i, indvar_j});
    builder.add_memlet(block2, tasklet3, "_out", D3, "void", {indvar_i, indvar_j});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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

    auto& block1 = builder.add_block(body_j);
    auto& A = builder.add_access(block1, "A");
    auto& B = builder.add_access(block1, "B");
    auto& C = builder.add_access(block1, "C");
    auto& D1 = builder.add_access(block1, "D");
    auto& D2 = builder.add_access(block1, "D");
    auto& tmp = builder.add_access(block1, "tmp");
    auto& tasklet1 = builder.add_tasklet(block1, data_flow::TaskletCode::assign,
                                         {"_out", base_desc}, {{"0.0", base_desc}});
    builder.add_memlet(block1, tasklet1, "_out", D1, "void", {indvar_i, indvar_j});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, A, "void", tasklet2, "_in1", {indvar_i, indvar_j});
    builder.add_memlet(block1, B, "void", tasklet2, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block1, tasklet2, "_out", tmp, "void", {});
    auto& tasklet3 =
        builder.add_tasklet(block1, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block1, tmp, "void", tasklet3, "_in1", {});
    builder.add_memlet(block1, C, "void", tasklet3, "_in2", {indvar_i, indvar_j});
    builder.add_memlet(block1, D1, "void", tasklet3, "_in3", {indvar_i, indvar_j});
    builder.add_memlet(block1, tasklet3, "_out", D2, "void", {indvar_i, indvar_j});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", {indvar_i});
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", {});
    builder.add_memlet(block2, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, c, "void", tasklet3, "_in2", {});
    builder.add_memlet(block2, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, tmp2, "void", tasklet4, "_in1", {});
    builder.add_memlet(block2, d, "void", tasklet4, "_in2", {});
    builder.add_memlet(block2, tasklet4, "_out", tmp3, "void", {});
    auto& tasklet5 = builder.add_tasklet(block2, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, e2, "void", tasklet5, "_in1", {indvar_i});
    builder.add_memlet(block2, tmp3, "void", tasklet5, "_in2", {});
    builder.add_memlet(block2, tasklet5, "_out", e3, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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
    builder.add_memlet(block1, tasklet1, "_out", e1, "void", {indvar_i});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, a, "void", tasklet2, "_in1", {indvar_i});
    builder.add_memlet(block1, b, "void", tasklet2, "_in2", {});
    builder.add_memlet(block1, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block1, c, "void", tasklet3, "_in2", {});
    builder.add_memlet(block1, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp2, "void", tasklet4, "_in1", {});
    builder.add_memlet(block1, d, "void", tasklet4, "_in2", {});
    builder.add_memlet(block1, tasklet4, "_out", tmp3, "void", {});
    auto& tasklet5 = builder.add_tasklet(block1, data_flow::TaskletCode::add, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, e1, "void", tasklet5, "_in1", {indvar_i});
    builder.add_memlet(block1, tmp3, "void", tasklet5, "_in2", {});
    builder.add_memlet(block1, tasklet5, "_out", e2, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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
    builder.add_memlet(block2, a, "void", tasklet2, "_in1", {indvar_i});
    builder.add_memlet(block2, b, "void", tasklet2, "_in2", {});
    builder.add_memlet(block2, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block2, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block2, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block2, c, "void", tasklet3, "_in2", {});
    builder.add_memlet(block2, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 =
        builder.add_tasklet(block2, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block2, tmp2, "void", tasklet4, "_in1", {});
    builder.add_memlet(block2, d, "void", tasklet4, "_in2", {});
    builder.add_memlet(block2, e2, "void", tasklet4, "_in3", {indvar_i});
    builder.add_memlet(block2, tasklet4, "_out", e3, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block2);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
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
    builder.add_memlet(block1, tasklet1, "_out", e1, "void", {indvar_i});
    auto& tasklet2 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, a, "void", tasklet2, "_in1", {indvar_i});
    builder.add_memlet(block1, b, "void", tasklet2, "_in2", {});
    builder.add_memlet(block1, tasklet2, "_out", tmp1, "void", {});
    auto& tasklet3 = builder.add_tasklet(block1, data_flow::TaskletCode::mul, {"_out", base_desc},
                                         {{"_in1", base_desc}, {"_in2", base_desc}});
    builder.add_memlet(block1, tmp1, "void", tasklet3, "_in1", {});
    builder.add_memlet(block1, c, "void", tasklet3, "_in2", {});
    builder.add_memlet(block1, tasklet3, "_out", tmp2, "void", {});
    auto& tasklet4 =
        builder.add_tasklet(block1, data_flow::TaskletCode::fma, {"_out", base_desc},
                            {{"_in1", base_desc}, {"_in2", base_desc}, {"_in3", base_desc}});
    builder.add_memlet(block1, tmp2, "void", tasklet4, "_in1", {});
    builder.add_memlet(block1, d, "void", tasklet4, "_in2", {});
    builder.add_memlet(block1, e1, "void", tasklet4, "_in3", {indvar_i});
    builder.add_memlet(block1, tasklet4, "_out", e2, "void", {indvar_i});

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::EinsumLift transformation(tasklet1, {}, block1);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);
}
