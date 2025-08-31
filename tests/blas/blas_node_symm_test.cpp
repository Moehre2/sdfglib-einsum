#include "sdfg/blas/blas_node_symm.h"

#include <gtest/gtest.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/element.h>
#include <sdfg/function.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <string>

#include "sdfg/blas/blas_node.h"

using namespace sdfg;

inline void symm_test(const types::PrimitiveType type1, const blas::BLASType type2,
                      const blas::BLASSide side, const blas::BLASTriangular uplo,
                      const std::string expected) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("m", sym_desc, true);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(type1);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("B", desc2, true);
    builder.add_container("C", desc2, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C1 = builder.add_access(block, "C");
    auto& C2 = builder.add_access(block, "C");
    auto& libnode =
        builder.add_library_node<blas::BLASNodeSymm, const blas::BLASType, blas::BLASSide,
                                 blas::BLASTriangular, symbolic::Expression, symbolic::Expression,
                                 std::string, std::string, std::string, std::string>(
            block, DebugInfo(), type2, side, uplo, symbolic::symbol("m"), symbolic::symbol("n"),
            "_alpha", "_A", "_B", "_C");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, A, "void", libnode, "_A", {});
    builder.add_memlet(block, B, "void", libnode, "_B", {});
    builder.add_memlet(block, C1, "void", libnode, "_C", {});
    builder.add_memlet(block, libnode, "_C", C2, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeSymm*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), expected);
}

TEST(BLASNodeSymm, ssymmLL) {
    symm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASSide_Left,
              blas::BLASTriangular_Lower,
              "ssymm('L', 'L', m, n, _alpha, _A, m, _B, m, 1.0, _C, m)");
}

TEST(BLASNodeSymm, ssymmRL) {
    symm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASSide_Right,
              blas::BLASTriangular_Lower,
              "ssymm('R', 'L', m, n, _alpha, _A, n, _B, m, 1.0, _C, m)");
}

TEST(BLASNodeSymm, ssymmLU) {
    symm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASSide_Left,
              blas::BLASTriangular_Upper,
              "ssymm('L', 'U', m, n, _alpha, _A, m, _B, m, 1.0, _C, m)");
}

TEST(BLASNodeSymm, ssymmRU) {
    symm_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASSide_Right,
              blas::BLASTriangular_Upper,
              "ssymm('R', 'U', m, n, _alpha, _A, n, _B, m, 1.0, _C, m)");
}

TEST(BLASNodeSymm, dsymmLL) {
    symm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASSide_Left,
              blas::BLASTriangular_Lower,
              "dsymm('L', 'L', m, n, _alpha, _A, m, _B, m, 1.0, _C, m)");
}

TEST(BLASNodeSymm, dsymmRL) {
    symm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASSide_Right,
              blas::BLASTriangular_Lower,
              "dsymm('R', 'L', m, n, _alpha, _A, n, _B, m, 1.0, _C, m)");
}

TEST(BLASNodeSymm, dsymmLU) {
    symm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASSide_Left,
              blas::BLASTriangular_Upper,
              "dsymm('L', 'U', m, n, _alpha, _A, m, _B, m, 1.0, _C, m)");
}

TEST(BLASNodeSymm, dsymmRU) {
    symm_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASSide_Right,
              blas::BLASTriangular_Upper,
              "dsymm('R', 'U', m, n, _alpha, _A, n, _B, m, 1.0, _C, m)");
}