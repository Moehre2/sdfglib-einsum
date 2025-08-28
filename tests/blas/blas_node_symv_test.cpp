#include "sdfg/blas/blas_node_symv.h"

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

inline void symv_test(const types::PrimitiveType type1, const blas::BLASType type2,
                      const blas::BLASTriangular uplo, const std::string expected) {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("n", sym_desc, true);

    types::Scalar base_desc(type1);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("alpha", base_desc, true);
    builder.add_container("A", desc2, true);
    builder.add_container("x", desc, true);
    builder.add_container("y", desc, true);

    auto& root = builder.subject().root();

    auto& block = builder.add_block(root);
    auto& alpha = builder.add_access(block, "alpha");
    auto& A = builder.add_access(block, "A");
    auto& x = builder.add_access(block, "x");
    auto& y1 = builder.add_access(block, "y");
    auto& y2 = builder.add_access(block, "y");
    auto& libnode = builder.add_library_node<blas::BLASNodeSymv, const blas::BLASType,
                                             blas::BLASTriangular, symbolic::Expression,
                                             std::string, std::string, std::string, std::string>(
        block, DebugInfo(), type2, uplo, symbolic::symbol("n"), "_alpha", "_A", "_x", "_y");
    builder.add_memlet(block, alpha, "void", libnode, "_alpha", {});
    builder.add_memlet(block, A, "void", libnode, "_A", {});
    builder.add_memlet(block, x, "void", libnode, "_x", {});
    builder.add_memlet(block, y1, "void", libnode, "_y", {});
    builder.add_memlet(block, libnode, "_y", y2, "void", {});

    auto* blas_node = dynamic_cast<blas::BLASNodeSymv*>(&libnode);
    ASSERT_TRUE(blas_node);

    EXPECT_EQ(blas_node->toStr(), expected);
}

TEST(BLASNodeSymv, ssymvL) {
    symv_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Lower,
              "ssymv('L', n, _alpha, _A, n, _x, 1, 1.0, _y, 1)");
}

TEST(BLASNodeSymv, ssymvU) {
    symv_test(types::PrimitiveType::Float, blas::BLASType_real, blas::BLASTriangular_Upper,
              "ssymv('U', n, _alpha, _A, n, _x, 1, 1.0, _y, 1)");
}

TEST(BLASNodeSymv, dsymvL) {
    symv_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Lower,
              "dsymv('L', n, _alpha, _A, n, _x, 1, 1.0, _y, 1)");
}

TEST(BLASNodeSymv, dsymvU) {
    symv_test(types::PrimitiveType::Double, blas::BLASType_double, blas::BLASTriangular_Upper,
              "dsymv('U', n, _alpha, _A, n, _x, 1, 1.0, _y, 1)");
}