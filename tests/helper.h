#pragma once

#include <gtest/gtest.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/symbolic/symbolic.h>

#include <string>
#include <unordered_map>

using namespace sdfg;

#define gen_for(index, bound, root_)                                              \
    auto indvar_##index = symbolic::symbol(#index);                               \
    auto bound_##index = symbolic::symbol(#bound);                                \
    auto condition_##index = symbolic::Lt(indvar_##index, bound_##index);         \
    auto update_##index = symbolic::add(indvar_##index, symbolic::one());         \
    auto& for_##index = builder.add_for(root_, indvar_##index, condition_##index, \
                                        symbolic::zero(), update_##index);        \
    auto& body_##index = for_##index.root();

#define AT_LEAST(val1, val2) \
    EXPECT_EQ(val1, val2);   \
    ASSERT_GE(val1, val2);

inline bool subsets_eq(const data_flow::Subset& subset1, const data_flow::Subset& subset2) {
    if (subset1.size() != subset2.size()) return false;
    for (size_t i = 0; i < subset1.size(); ++i) {
        if (!symbolic::eq(subset1[i], subset2[i])) return false;
    }
    return true;
}

inline std::unordered_map<std::string, std::string> get_conn2cont(
    const structured_control_flow::Block& block, const data_flow::LibraryNode& libnode) {
    std::unordered_map<std::string, std::string> conn2cont;
    for (auto& iedge : block.dataflow().in_edges(libnode)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        conn2cont.insert({iedge.dst_conn(), src.data()});
    }
    return conn2cont;
}