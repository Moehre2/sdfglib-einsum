#include "sdfg/transformations/einsum2blas.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/transformations/transformation.h>

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <set>
#include <string>
#include <unordered_map>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_gemm.h"
#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

Einsum2BLAS::Einsum2BLAS(einsum::EinsumNode& einsum_node) : einsum_node_(einsum_node) {}

std::string Einsum2BLAS::name() const { return "Einsum2BLAS"; }

bool Einsum2BLAS::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                 analysis::AnalysisManager& analysis_manager) {
    // Check inputs
    if (this->einsum_node_.inputs().size() != 3) return false;
    if (this->einsum_node_.input(2) != this->einsum_node_.output(0)) return false;

    // Check maps
    if (this->einsum_node_.maps().size() != 3) return false;
    std::set<std::string> indvars;
    for (auto& map : this->einsum_node_.maps()) indvars.insert(map.first->__str__());

    // Check out indices
    if (this->einsum_node_.out_indices().size() != 2) return false;
    auto i = this->einsum_node_.out_index(0);
    auto j = this->einsum_node_.out_index(1);
    if (!indvars.contains(i->__str__())) return false;
    if (!indvars.contains(j->__str__())) return false;

    // Determine k
    symbolic::Expression k = symbolic::__nullptr__();
    for (auto& map : this->einsum_node_.maps()) {
        if (!symbolic::eq(map.first, i) && !symbolic::eq(map.first, j)) {
            k = map.first;
            break;
        }
    }
    if (symbolic::eq(k, symbolic::__nullptr__())) return false;

    // Check in indices
    if (this->einsum_node_.in_indices().size() != 3) return false;
    if (this->einsum_node_.in_indices(0).size() != 2) return false;
    if (this->einsum_node_.in_indices(1).size() != 2) return false;
    if (this->einsum_node_.in_indices(2).size() != 2) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(2, 0), i)) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(2, 1), j)) return false;
    bool AB = (symbolic::eq(this->einsum_node_.in_index(0, 0), i) &&
               symbolic::eq(this->einsum_node_.in_index(0, 1), k) &&
               symbolic::eq(this->einsum_node_.in_index(1, 0), k) &&
               symbolic::eq(this->einsum_node_.in_index(1, 1), j));
    bool BA = (symbolic::eq(this->einsum_node_.in_index(0, 0), k) &&
               symbolic::eq(this->einsum_node_.in_index(0, 1), j) &&
               symbolic::eq(this->einsum_node_.in_index(1, 0), i) &&
               symbolic::eq(this->einsum_node_.in_index(1, 1), k));
    if (!AB && !BA) return false;

    return true;
}

void Einsum2BLAS::apply(builder::StructuredSDFGBuilder& builder,
                        analysis::AnalysisManager& analysis_manager) {
    // Map map indices to map bounds
    std::unordered_map<std::string, symbolic::Expression> maps;
    for (auto& map : this->einsum_node_.maps()) maps.insert({map.first->__str__(), map.second});

    // Determine i, j, and k
    auto i = this->einsum_node_.out_index(0);
    auto j = this->einsum_node_.out_index(1);
    symbolic::Expression k = symbolic::__nullptr__();
    for (auto& map : this->einsum_node_.maps()) {
        if (!symbolic::eq(map.first, i) && !symbolic::eq(map.first, j)) {
            k = map.first;
            break;
        }
    }

    // Determine bounds
    symbolic::Expression bound_i = maps.at(i->__str__());
    symbolic::Expression bound_j = maps.at(j->__str__());
    symbolic::Expression bound_k = maps.at(k->__str__());

    // Determine order
    // bool AB = (symbolic::eq(this->einsum_node_.in_index(0, 0), i) &&
    //           symbolic::eq(this->einsum_node_.in_index(0, 1), k) &&
    //           symbolic::eq(this->einsum_node_.in_index(1, 0), k) &&
    //           symbolic::eq(this->einsum_node_.in_index(1, 1), j));

    // Get the block in which the einsum node lives
    auto* block =
        dynamic_cast<structured_control_flow::Block*>(this->einsum_node_.get_parent().get_parent());

    // Add the BLAS node
    auto& libnode =
        builder.add_library_node<blas::BLASNodeGemm, const std::vector<std::string>&,
                                 const std::vector<std::string>&, const blas::BLASType,
                                 symbolic::Expression, symbolic::Expression, symbolic::Expression>(
            *block, DebugInfo(), this->einsum_node_.outputs(), this->einsum_node_.inputs(),
            blas::BLASType_real, bound_i, bound_j, bound_k);

    // Copy the memlets
    for (auto& iedge : block->dataflow().in_edges(this->einsum_node_)) {
        builder.add_memlet(*block, iedge.src(), iedge.src_conn(), libnode, iedge.dst_conn(),
                           iedge.subset(), iedge.debug_info());
    }
    for (auto& oedge : block->dataflow().out_edges(this->einsum_node_)) {
        builder.add_memlet(*block, libnode, oedge.src_conn(), oedge.dst(), oedge.dst_conn(),
                           oedge.subset(), oedge.debug_info());
    }

    // Remove the old memlets
    while (block->dataflow().in_edges(this->einsum_node_).begin() !=
           block->dataflow().in_edges(this->einsum_node_).end()) {
        builder.remove_memlet(*block, *block->dataflow().in_edges(this->einsum_node_).begin());
    }
    while (block->dataflow().out_edges(this->einsum_node_).begin() !=
           block->dataflow().out_edges(this->einsum_node_).end()) {
        builder.remove_memlet(*block, *block->dataflow().out_edges(this->einsum_node_).begin());
    }

    // Remove the einsum node
    builder.remove_node(*block, this->einsum_node_);
}

void Einsum2BLAS::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_id"] = this->einsum_node_.element_id();
}

Einsum2BLAS Einsum2BLAS::from_json(builder::StructuredSDFGBuilder& builder,
                                   const nlohmann::json& j) {
    size_t einsum_node_id = j["einsum_node_element_id"].get<size_t>();
    auto einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found.");
    }
    auto einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);

    return Einsum2BLAS(*einsum_node);
}

}  // namespace transformations
}  // namespace sdfg