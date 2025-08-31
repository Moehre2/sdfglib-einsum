#include "sdfg/transformations/einsum2blas_ger.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/element.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/transformations/transformation.h>
#include <sdfg/types/type.h>
#include <sdfg/types/utils.h>

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_ger.h"
#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

Einsum2BLASGer::Einsum2BLASGer(einsum::EinsumNode& einsum_node) : einsum_node_(einsum_node) {}

std::string Einsum2BLASGer::name() const { return "Einsum2BLASGer"; }

bool Einsum2BLASGer::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                    analysis::AnalysisManager& analysis_manager) {
    // Check maps
    if (this->einsum_node_.maps().size() != 2) return false;

    // Check out indices
    symbolic::Symbol indvar_x, indvar_y;
    if (this->einsum_node_.out_indices().size() != 2) return false;
    if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(0)) &&
        symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(1))) {
        indvar_x = this->einsum_node_.indvar(0);
        indvar_y = this->einsum_node_.indvar(1);
    } else if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(1)) &&
               symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(0))) {
        indvar_x = this->einsum_node_.indvar(1);
        indvar_y = this->einsum_node_.indvar(0);
    } else {
        return false;
    }

    // Check inputs
    long long x = -1, y = -1, A = -1;
    if (this->einsum_node_.inputs().size() == 3) {
        A = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 1 &&
                symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_x))
                x = i;
            else if (this->einsum_node_.in_indices(i).size() == 1 &&
                     symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_y))
                y = i;
        }
    } else if (this->einsum_node_.inputs().size() == 4) {
        A = 3;
        long long alpha = -1;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 1 &&
                symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_x))
                x = i;
            else if (this->einsum_node_.in_indices(i).size() == 1 &&
                     symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_y))
                y = i;
            else if (this->einsum_node_.in_indices(i).size() == 0)
                alpha = i;
        }

        // Check alpha
        if (alpha == -1) return false;
        if (this->einsum_node_.in_indices(alpha).size() != 0) return false;
    } else {
        return false;
    }
    if (x == -1 || y == -1) return false;
    if (this->einsum_node_.input(A) != this->einsum_node_.output(0)) return false;

    // Check in indices
    if (this->einsum_node_.in_indices(x).size() != 1) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(x, 0), indvar_x)) return false;

    if (this->einsum_node_.in_indices(y).size() != 1) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(y, 0), indvar_y)) return false;

    if (this->einsum_node_.in_indices(A).size() != 2) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(A, 0), indvar_x)) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(A, 1), indvar_y)) return false;

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

    // Check if all inputs have the same base type
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        const types::IType& src_type = builder.subject().type(src.data());
        if (types::infer_type(builder.subject(), src_type, iedge.subset()).primitive_type() !=
            base_type)
            return false;
    }

    return true;
}

void Einsum2BLASGer::apply(builder::StructuredSDFGBuilder& builder,
                           analysis::AnalysisManager& analysis_manager) {
    // Get the data flow graph
    auto& dfg = this->einsum_node_.get_parent();

    // Get the block in which the einsum node lives
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());

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

    // Determine indvars, m, and n
    symbolic::Symbol indvar_x, indvar_y;
    symbolic::Expression m, n;
    if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(0)) &&
        symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(1))) {
        indvar_x = this->einsum_node_.indvar(0);
        indvar_y = this->einsum_node_.indvar(1);
        m = this->einsum_node_.num_iteration(0);
        n = this->einsum_node_.num_iteration(1);
    } else if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(1)) &&
               symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(0))) {
        indvar_x = this->einsum_node_.indvar(1);
        indvar_y = this->einsum_node_.indvar(0);
        m = this->einsum_node_.num_iteration(1);
        n = this->einsum_node_.num_iteration(0);
    }

    // Determine the input positions
    long long alpha = -1, x = -1, y = -1, A = -1;
    bool has_alpha = false;
    if (this->einsum_node_.inputs().size() == 3) {
        A = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 1 &&
                symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_x))
                x = i;
            else if (this->einsum_node_.in_indices(i).size() == 1 &&
                     symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_y))
                y = i;
        }
    } else {
        A = 3;
        has_alpha = true;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 1 &&
                symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_x))
                x = i;
            else if (this->einsum_node_.in_indices(i).size() == 1 &&
                     symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_y))
                y = i;
            else if (this->einsum_node_.in_indices(i).size() == 0)
                alpha = i;
        }
    }

    // Determine alpha
    std::string alpha_input = has_alpha ? this->einsum_node_.input(alpha)
                                        : ((type == blas::BLASType_real) ? "1.0f" : "1.0");

    // Add the BLAS node for ger
    data_flow::LibraryNode& libnode =
        builder.add_library_node<blas::BLASNodeGer, const blas::BLASType, symbolic::Expression,
                                 symbolic::Expression, std::string, std::string, std::string,
                                 std::string>(
            *block, this->einsum_node_.debug_info(), type, m, n, alpha_input,
            this->einsum_node_.input(x), this->einsum_node_.input(y), this->einsum_node_.input(A));

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

    analysis_manager.invalidate_all();
}

void Einsum2BLASGer::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_id"] = this->einsum_node_.element_id();
}

Einsum2BLASGer Einsum2BLASGer::from_json(builder::StructuredSDFGBuilder& builder,
                                         const nlohmann::json& j) {
    size_t einsum_node_id = j["einsum_node_id"].get<size_t>();
    Element* einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found.");
    }
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);

    return Einsum2BLASGer(*einsum_node);
}

}  // namespace transformations
}  // namespace sdfg