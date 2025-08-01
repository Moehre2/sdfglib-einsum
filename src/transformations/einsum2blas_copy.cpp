#include "sdfg/transformations/einsum2blas_copy.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/utils.h>

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_copy.h"
#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

Einsum2BLASCopy::Einsum2BLASCopy(einsum::EinsumNode& einsum_node) : einsum_node_(einsum_node) {}

std::string Einsum2BLASCopy::name() const { return "Einsum2BLASCopy"; }

bool Einsum2BLASCopy::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                     analysis::AnalysisManager& analysis_manager) {
    // Check maps
    if (this->einsum_node_.maps().size() != 1) return false;
    symbolic::Symbol indvar = this->einsum_node_.indvar(0);

    // Check out indices
    if (this->einsum_node_.out_indices().size() != 1) return false;
    if (!symbolic::eq(this->einsum_node_.out_index(0), indvar)) return false;

    // Check input
    if (this->einsum_node_.inputs().size() != 1) return false;

    // Check in indices
    if (this->einsum_node_.in_indices(0).size() != 1) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(0, 0), indvar)) return false;

    // Get the data flow graph
    auto& dfg = this->einsum_node_.get_parent();

    // Determine and check the base type of output
    auto& oedge = *dfg.out_edges(this->einsum_node_).begin();
    auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
    const types::IType& dst_type = builder.subject().type(dst.data());
    auto base_type =
        types::infer_type(builder.subject(), dst_type, oedge.subset()).primitive_type();
    if (base_type != types::PrimitiveType::Float && base_type != types::PrimitiveType::Double)
        return false;

    // Check if input has the same base type
    auto& iedge = *dfg.in_edges(this->einsum_node_).begin();
    auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
    const types::IType& src_type = builder.subject().type(src.data());
    if (types::infer_type(builder.subject(), src_type, iedge.subset()).primitive_type() !=
        base_type)
        return false;

    return true;
}

void Einsum2BLASCopy::apply(builder::StructuredSDFGBuilder& builder,
                            analysis::AnalysisManager& analysis_manager) {
    // Get the data flow graph
    auto& dfg = this->einsum_node_.get_parent();

    // Get the block in which the einsum node lives
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());

    // Get the number of iterations (n)
    symbolic::Expression num_iteration = this->einsum_node_.num_iteration(0);

    // Determine the BLAS type
    blas::BLASType type;
    {
        auto& oedge = *dfg.out_edges(this->einsum_node_).begin();
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
        const types::IType& dst_type = builder.subject().type(dst.data());
        if (types::infer_type(builder.subject(), dst_type, oedge.subset()).primitive_type() ==
            types::PrimitiveType::Float) {
            type = blas::BLASType_real;
        } else {
            type = blas::BLASType_double;
        }
    }

    // Add the BLAS node for copy
    auto& libnode = builder.add_library_node<blas::BLASNodeCopy, const blas::BLASType,
                                             symbolic::Expression, std::string, std::string>(
        *block, this->einsum_node_.debug_info(), type, num_iteration, this->einsum_node_.input(0),
        this->einsum_node_.output(0));

    // Copy the memlets
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        builder.add_memlet(*block, iedge.src(), iedge.src_conn(), libnode, iedge.dst_conn(),
                           iedge.subset(), iedge.debug_info());
    }
    for (auto& oedge : dfg.out_edges(this->einsum_node_)) {
        builder.add_memlet(*block, libnode, oedge.src_conn(), oedge.dst(), oedge.dst_conn(),
                           oedge.subset(), oedge.debug_info());
    }

    // Remove the old memlets
    while (dfg.in_edges(this->einsum_node_).begin() != dfg.in_edges(this->einsum_node_).end()) {
        builder.remove_memlet(*block, *dfg.in_edges(this->einsum_node_).begin());
    }
    while (dfg.out_edges(this->einsum_node_).begin() != dfg.out_edges(this->einsum_node_).end()) {
        builder.remove_memlet(*block, *dfg.out_edges(this->einsum_node_).begin());
    }

    // Remove the einsum node
    builder.remove_node(*block, this->einsum_node_);
}

void Einsum2BLASCopy::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_id"] = this->einsum_node_.element_id();
}

Einsum2BLASCopy Einsum2BLASCopy::from_json(builder::StructuredSDFGBuilder& builder,
                                           const nlohmann::json& j) {
    size_t einsum_node_id = j["einsum_node_id"].get<size_t>();
    Element* einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found.");
    }
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);

    return Einsum2BLASCopy(*einsum_node);
}

}  // namespace transformations
}  // namespace sdfg