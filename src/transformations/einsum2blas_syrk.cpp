#include "sdfg/transformations/einsum2blas_syrk.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/transformations/transformation.h>
#include <sdfg/types/type.h>
#include <sdfg/types/utils.h>

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_syrk.h"
#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

bool Einsum2BLASSyrk::check_lower(size_t outer_1, size_t outer_2, size_t inner) {
    return !symbolic::uses(this->einsum_node_.num_iteration(outer_1),
                           this->einsum_node_.indvar(outer_2)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(outer_1),
                           this->einsum_node_.indvar(inner)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(inner),
                           this->einsum_node_.indvar(outer_1)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(inner),
                           this->einsum_node_.indvar(outer_2)) &&
           symbolic::eq(this->einsum_node_.num_iteration(outer_2),
                        symbolic::add(this->einsum_node_.indvar(outer_1), symbolic::one()));
}

bool Einsum2BLASSyrk::check_upper(size_t outer_1, size_t outer_2, size_t inner) {
    return !symbolic::uses(this->einsum_node_.num_iteration(outer_2),
                           this->einsum_node_.indvar(outer_1)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(outer_2),
                           this->einsum_node_.indvar(inner)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(inner),
                           this->einsum_node_.indvar(outer_1)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(inner),
                           this->einsum_node_.indvar(outer_2)) &&
           symbolic::eq(this->einsum_node_.num_iteration(outer_1),
                        symbolic::add(this->einsum_node_.indvar(outer_2), symbolic::one()));
}

Einsum2BLASSyrk::Einsum2BLASSyrk(einsum::EinsumNode& einsum_node) : einsum_node_(einsum_node) {}

std::string Einsum2BLASSyrk::name() const { return "Einsum2BLASSyrk"; }

bool Einsum2BLASSyrk::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                     analysis::AnalysisManager& analysis_manager) {
    // Check maps
    if (this->einsum_node_.maps().size() != 3) return false;

    // Check out indices size
    if (this->einsum_node_.out_indices().size() != 2) return false;

    // Check out indices
    long long outer_1 = -1, outer_2 = -1, inner = -1;
    for (size_t i = 0; i < this->einsum_node_.maps().size(); ++i) {
        if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(i)))
            outer_1 = i;
        else if (symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(i)))
            outer_2 = i;
        else
            inner = i;
    }
    if (outer_1 == -1 || outer_2 == -1 || inner == -1 || outer_1 == outer_2 || outer_1 == inner ||
        outer_2 == inner)
        return false;
    symbolic::Symbol indvar_outer_1 = this->einsum_node_.indvar(outer_1);
    symbolic::Symbol indvar_outer_2 = this->einsum_node_.indvar(outer_2);
    symbolic::Symbol indvar_inner = this->einsum_node_.indvar(inner);

    // Check triangular
    if (this->check_lower(outer_1, outer_2, inner) == this->check_upper(outer_1, outer_2, inner))
        return false;

    // Check inputs
    long long A1 = -1, A2 = -1, C = -1;
    if (this->einsum_node_.inputs().size() == 3) {
        C = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() != 2)
                break;
            else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1) ||
                     symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_1))
                A1 = i;
            else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_2) ||
                     symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2))
                A2 = i;
        }
    } else if (this->einsum_node_.inputs().size() == 4) {
        C = 3;
        long long alpha = -1;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 0)
                alpha = i;
            else if (this->einsum_node_.in_indices(i).size() != 2)
                break;
            else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1) ||
                     symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_1))
                A1 = i;
            else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_2) ||
                     symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2))
                A2 = i;
        }

        // Check alpha
        if (alpha == -1) return false;
        if (this->einsum_node_.in_indices(alpha).size() != 0) return false;
    } else {
        return false;
    }
    if (A1 == -1 || A2 == -1 || A1 == A2) return false;
    if (this->einsum_node_.input(C) != this->einsum_node_.output(0)) return false;

    // Check in indices
    bool trans = symbolic::eq(this->einsum_node_.in_index(A1, 1), indvar_outer_1);
    if (trans) {
        if (this->einsum_node_.in_indices(A1).size() != 2) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A1, 0), indvar_inner)) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A1, 1), indvar_outer_1)) return false;

        if (this->einsum_node_.in_indices(A2).size() != 2) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A2, 0), indvar_inner)) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A2, 1), indvar_outer_2)) return false;
    } else {
        if (this->einsum_node_.in_indices(A1).size() != 2) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A1, 0), indvar_outer_1)) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A1, 1), indvar_inner)) return false;

        if (this->einsum_node_.in_indices(A2).size() != 2) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A2, 0), indvar_outer_2)) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A2, 1), indvar_inner)) return false;
    }

    if (this->einsum_node_.in_indices(C).size() != 2) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(C, 0), indvar_outer_1)) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(C, 1), indvar_outer_2)) return false;

    // Get the data flow graph
    auto& dfg = this->einsum_node_.get_parent();

    // Check that x1 and x2 access the same container
    std::string A1_container, A2_container;
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        if (iedge.dst_conn() == this->einsum_node_.input(A1))
            A1_container = dynamic_cast<const data_flow::AccessNode&>(iedge.src()).data();
        else if (iedge.dst_conn() == this->einsum_node_.input(A2))
            A2_container = dynamic_cast<const data_flow::AccessNode&>(iedge.src()).data();
    }
    if (A1_container.empty()) return false;
    if (A2_container.empty()) return false;
    if (A1_container != A2_container) return false;

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

void Einsum2BLASSyrk::apply(builder::StructuredSDFGBuilder& builder,
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

    // Determine indvars
    long long outer_1 = -1, outer_2 = -1, inner = -1;
    for (size_t i = 0; i < this->einsum_node_.maps().size(); ++i) {
        if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(i)))
            outer_1 = i;
        else if (symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(i)))
            outer_2 = i;
        else
            inner = i;
    }
    symbolic::Symbol indvar_outer_1 = this->einsum_node_.indvar(outer_1);
    symbolic::Symbol indvar_outer_2 = this->einsum_node_.indvar(outer_2);
    symbolic::Symbol indvar_inner = this->einsum_node_.indvar(inner);

    // Determine triangular, n, and k
    blas::BLASTriangular uplo;
    symbolic::Expression n, k;
    if (this->check_lower(outer_1, outer_2, inner)) {
        uplo = blas::BLASTriangular_Lower;
        n = this->einsum_node_.num_iteration(outer_1);
    } else {
        uplo = blas::BLASTriangular_Upper;
        n = this->einsum_node_.num_iteration(outer_2);
    }
    k = this->einsum_node_.num_iteration(inner);

    // Determine inputs
    long long alpha = -1, A = -1, A_del = -1, C = -1;
    bool has_alpha = false;
    if (this->einsum_node_.inputs().size() == 3) {
        C = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() != 2)
                break;
            else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1) ||
                     symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_1))
                A = i;
            else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_2) ||
                     symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2))
                A_del = i;
        }
    } else if (this->einsum_node_.inputs().size() == 4) {
        C = 3;
        has_alpha = true;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 0)
                alpha = i;
            else if (this->einsum_node_.in_indices(i).size() != 2)
                break;
            else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1) ||
                     symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_1))
                A = i;
            else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_2) ||
                     symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2))
                A_del = i;
        }
    }

    // Determine transpose
    blas::BLASTranspose trans = symbolic::eq(this->einsum_node_.in_index(A, 1), indvar_outer_1)
                                    ? blas::BLASTranspose_Transpose
                                    : blas::BLASTranspose_No;

    // Determine alpha
    std::string alpha_input = has_alpha ? this->einsum_node_.input(alpha)
                                        : ((type == blas::BLASType_real) ? "1.0f" : "1.0");

    // Add the BLAS node for syrk
    data_flow::LibraryNode& libnode =
        builder.add_library_node<blas::BLASNodeSyrk, const blas::BLASType, blas::BLASTriangular,
                                 blas::BLASTranspose, symbolic::Expression, symbolic::Expression,
                                 std::string, std::string, std::string>(
            *block, this->einsum_node_.debug_info(), type, uplo, trans, n, k, alpha_input,
            this->einsum_node_.input(A), this->einsum_node_.input(C));

    // Copy the memlets
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        if (iedge.dst_conn() == this->einsum_node_.input(A_del)) continue;
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

void Einsum2BLASSyrk::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_id"] = this->einsum_node_.element_id();
}

Einsum2BLASSyrk Einsum2BLASSyrk::from_json(builder::StructuredSDFGBuilder& builder,
                                           const nlohmann::json& j) {
    size_t einsum_node_id = j["einsum_node_element_id"].get<size_t>();
    auto einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found.");
    }
    auto einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);

    return Einsum2BLASSyrk(*einsum_node);
}

}  // namespace transformations
}  // namespace sdfg