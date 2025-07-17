#pragma once

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/element.h>
#include <sdfg/function.h>
#include <sdfg/structured_sdfg.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/einsum/einsum_node.h"

using namespace sdfg;

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> matrix_matrix_mult() {
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
            block, DebugInfo(), {"_out"}, {"_out", "_in1", "_in2"},
            {{i, symbolic::symbol("I")}, {j, symbolic::symbol("J")}, {k, symbolic::symbol("K")}},
            {i, k}, {{i, k}, {i, j}, {j, k}});
    builder.add_memlet(block, C1, "void", libnode, "_out", {});
    builder.add_memlet(block, A, "void", libnode, "_in1", {});
    builder.add_memlet(block, B, "void", libnode, "_in2", {});
    builder.add_memlet(block, libnode, "_out", C2, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> tensor_contraction_3d() {
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

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");
    auto k = symbolic::symbol("k");
    auto l = symbolic::symbol("l");
    auto m = symbolic::symbol("m");
    auto n = symbolic::symbol("n");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");
    auto& D1 = builder.add_access(block, "D");
    auto& D2 = builder.add_access(block, "D");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_out", "_in1", "_in2", "_in3"},
            {{i, symbolic::symbol("I")},
             {j, symbolic::symbol("J")},
             {k, symbolic::symbol("K")},
             {l, symbolic::symbol("L")},
             {m, symbolic::symbol("M")},
             {n, symbolic::symbol("N")}},
            {i, j, k}, {{i, j, k}, {l, j, m}, {i, l, n}, {n, m, k}});
    builder.add_memlet(block, D1, "void", libnode, "_out", {});
    builder.add_memlet(block, A, "void", libnode, "_in1", {});
    builder.add_memlet(block, B, "void", libnode, "_in2", {});
    builder.add_memlet(block, C, "void", libnode, "_in3", {});
    builder.add_memlet(block, libnode, "_out", D2, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> matrix_vector_mult() {
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

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "b");
    auto& c1 = builder.add_access(block, "c");
    auto& c2 = builder.add_access(block, "c");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_out", "_in1", "_in2"},
            {{i, symbolic::symbol("I")}, {j, symbolic::symbol("J")}}, {i}, {{i}, {i, j}, {j}});
    builder.add_memlet(block, c1, "void", libnode, "_out", {});
    builder.add_memlet(block, A, "void", libnode, "_in1", {});
    builder.add_memlet(block, b, "void", libnode, "_in2", {});
    builder.add_memlet(block, libnode, "_out", c2, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> diagonal_extraction() {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("b", desc, true);

    auto i = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& b = builder.add_access(block, "b");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in"}, {{i, symbolic::symbol("I")}}, {i}, {{i, i}});
    builder.add_memlet(block, A, "void", libnode, "_in", {});
    builder.add_memlet(block, libnode, "_out", b, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> matrix_trace() {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    types::Pointer desc2(*desc.clone());
    builder.add_container("A", desc2, true);
    builder.add_container("b", desc, true);

    auto i = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& b1 = builder.add_access(block, "b");
    auto& b2 = builder.add_access(block, "b");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_out", "_in"}, {{i, symbolic::symbol("I")}}, {},
            {{}, {i, i}});
    builder.add_memlet(block, b1, "void", libnode, "_out", {});
    builder.add_memlet(block, A, "void", libnode, "_in", {});
    builder.add_memlet(block, libnode, "_out", b2, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> matrix_copy() {
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

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in"},
            {{i, symbolic::symbol("I")}, {j, symbolic::symbol("J")}}, {i, j}, {{i, j}});
    builder.add_memlet(block, A, "void", libnode, "_in", {});
    builder.add_memlet(block, libnode, "_out", B, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> matrix_transpose() {
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

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in"},
            {{i, symbolic::symbol("I")}, {j, symbolic::symbol("J")}}, {j, i}, {{i, j}});
    builder.add_memlet(block, A, "void", libnode, "_in", {});
    builder.add_memlet(block, libnode, "_out", B, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> dot_product() {
    builder::StructuredSDFGBuilder builder("sdfg_1", FunctionType_CPU);

    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    builder.add_container("i", sym_desc);
    builder.add_container("I", sym_desc, true);

    types::Scalar base_desc(types::PrimitiveType::Float);
    types::Pointer desc(base_desc);
    builder.add_container("a", desc, true);
    builder.add_container("b", desc, true);
    builder.add_container("c", desc, true);

    auto i = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c1 = builder.add_access(block, "c");
    auto& c2 = builder.add_access(block, "c");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_out", "_in1", "_in2"}, {{i, symbolic::symbol("I")}},
            {}, {{}, {i}, {i}});
    builder.add_memlet(block, c1, "void", libnode, "_out", {});
    builder.add_memlet(block, a, "void", libnode, "_in1", {});
    builder.add_memlet(block, b, "void", libnode, "_in2", {});
    builder.add_memlet(block, libnode, "_out", c2, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> matrix_elementwise_mult() {
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

    auto i = symbolic::symbol("i");
    auto j = symbolic::symbol("j");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& A = builder.add_access(block, "A");
    auto& B = builder.add_access(block, "B");
    auto& C = builder.add_access(block, "C");
    auto& D = builder.add_access(block, "D");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in1", "_in2", "_in3"},
            {{i, symbolic::symbol("I")}, {j, symbolic::symbol("J")}}, {i, j},
            {{i, j}, {i, j}, {i, j}});
    builder.add_memlet(block, A, "void", libnode, "_in1", {});
    builder.add_memlet(block, B, "void", libnode, "_in2", {});
    builder.add_memlet(block, C, "void", libnode, "_in3", {});
    builder.add_memlet(block, libnode, "_out", D, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

inline std::pair<std::unique_ptr<StructuredSDFG>, einsum::EinsumNode*> vector_scaling() {
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

    auto i = symbolic::symbol("i");

    auto& root = builder.subject().root();
    auto& block = builder.add_block(root);
    auto& a = builder.add_access(block, "a");
    auto& b = builder.add_access(block, "b");
    auto& c = builder.add_access(block, "c");
    auto& d = builder.add_access(block, "d");
    auto& e = builder.add_access(block, "e");
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, DebugInfo(), {"_out"}, {"_in1", "_in2", "_in3", "_in4"},
            {{i, symbolic::symbol("I")}}, {i}, {{i}, {}, {}, {}});
    builder.add_memlet(block, a, "void", libnode, "_in1", {});
    builder.add_memlet(block, b, "void", libnode, "_in2", {});
    builder.add_memlet(block, c, "void", libnode, "_in3", {});
    builder.add_memlet(block, d, "void", libnode, "_in4", {});
    builder.add_memlet(block, libnode, "_out", e, "void", {});

    return std::make_pair(builder.move(), dynamic_cast<einsum::EinsumNode*>(&libnode));
}

// scaling
