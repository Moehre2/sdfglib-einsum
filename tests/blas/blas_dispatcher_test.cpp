#include <gtest/gtest.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/code_generators/c_code_generator.h>
#include <sdfg/element.h>
#include <sdfg/function.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <string>
#include <vector>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_gemm.h"

using namespace sdfg;

TEST(BLASDispatcher, MatrixMatrixMultiplication) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("I", sym_desc, true);
    builder.add_container("J", sym_desc, true);
    builder.add_container("K", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto I = symbolic::symbol("I");
    auto J = symbolic::symbol("J");
    auto K = symbolic::symbol("K");

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<blas::BLASNodeGemm, const std::vector<std::string>&,
                                 const std::vector<std::string>&, const blas::BLASType,
                                 symbolic::Expression, symbolic::Expression, symbolic::Expression>(
            block, DebugInfo(), {"_out"}, {"_in1", "_in2", "_out"}, blas::BLASType_real, I, J, K);
    builder.add_memlet(block, A, "void", libnode, "_in1", {});
    builder.add_memlet(block, B, "void", libnode, "_in2", {});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    auto sdfg = builder.move();

    codegen::CCodeGenerator generator(*sdfg);
    EXPECT_TRUE(generator.generate());
    std::cout << generator.function_definition() << " {" << std::endl
              << generator.main().str() << "}" << std::endl;
}
