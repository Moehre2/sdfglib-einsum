#include "sdfg/transformations/einsum2blas.h"

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

#include "sdfg/einsum/einsum_node.h"

using namespace sdfg;

TEST(Einsum2BLAS, MatrixMatrixMultiplication) {
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
            block, DebugInfo(), {"_out"}, {"_in1", "_in2", "_out"},
            {{i, symbolic::symbol("I")}, {j, symbolic::symbol("J")}, {k, symbolic::symbol("K")}},
            {i, k}, {{i, j}, {j, k}, {i, k}});
    builder.add_memlet(block, A, "void", libnode, "_in1", {});
    builder.add_memlet(block, B, "void", libnode, "_in2", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(&libnode);
    EXPECT_TRUE(einsum_node);

    auto sdfg = builder.move();

    builder::StructuredSDFGBuilder builder_opt(sdfg);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());

    transformations::Einsum2BLAS transformation(*einsum_node);
    EXPECT_TRUE(transformation.can_be_applied(builder_opt, analysis_manager));
    transformation.apply(builder_opt, analysis_manager);

    auto sdfg_opt = builder_opt.move();
    codegen::CCodeGenerator generator(*sdfg_opt);
    EXPECT_TRUE(generator.generate());
    std::cout << generator.function_definition() << " {" << std::endl
              << generator.main().str() << "}" << std::endl;
}
