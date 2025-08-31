#include "sdfg/transformations/einsum2blas_symm.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/element.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/transformations/transformation.h>
#include <sdfg/types/type.h>
#include <sdfg/types/utils.h>

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_symm.h"
#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

bool Einsum2BLASSymm::check_left_lower(size_t outer_1, size_t outer_2, size_t inner) {
    return !symbolic::uses(this->einsum_node_.num_iteration(outer_1),
                           this->einsum_node_.indvar(outer_2)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(outer_1),
                           this->einsum_node_.indvar(inner)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(outer_2),
                           this->einsum_node_.indvar(outer_1)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(outer_2),
                           this->einsum_node_.indvar(inner)) &&
           symbolic::eq(this->einsum_node_.num_iteration(inner),
                        symbolic::add(this->einsum_node_.indvar(outer_1), symbolic::one()));
}

bool Einsum2BLASSymm::check_left_upper(size_t outer_1, size_t outer_2, size_t inner) {
    return !symbolic::uses(this->einsum_node_.num_iteration(outer_2),
                           this->einsum_node_.indvar(outer_1)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(outer_2),
                           this->einsum_node_.indvar(inner)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(inner),
                           this->einsum_node_.indvar(outer_1)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(inner),
                           this->einsum_node_.indvar(outer_2)) &&
           symbolic::eq(this->einsum_node_.num_iteration(outer_1),
                        symbolic::add(this->einsum_node_.indvar(inner), symbolic::one()));
}

bool Einsum2BLASSymm::check_right_lower(size_t outer_1, size_t outer_2, size_t inner) {
    return !symbolic::uses(this->einsum_node_.num_iteration(outer_1),
                           this->einsum_node_.indvar(outer_2)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(outer_1),
                           this->einsum_node_.indvar(inner)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(inner),
                           this->einsum_node_.indvar(outer_1)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(inner),
                           this->einsum_node_.indvar(outer_2)) &&
           symbolic::eq(this->einsum_node_.num_iteration(outer_2),
                        symbolic::add(this->einsum_node_.indvar(inner), symbolic::one()));
}

bool Einsum2BLASSymm::check_right_upper(size_t outer_1, size_t outer_2, size_t inner) {
    return !symbolic::uses(this->einsum_node_.num_iteration(outer_1),
                           this->einsum_node_.indvar(outer_2)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(outer_1),
                           this->einsum_node_.indvar(inner)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(outer_2),
                           this->einsum_node_.indvar(outer_1)) &&
           !symbolic::uses(this->einsum_node_.num_iteration(outer_2),
                           this->einsum_node_.indvar(inner)) &&
           symbolic::eq(this->einsum_node_.num_iteration(inner),
                        symbolic::add(this->einsum_node_.indvar(outer_2), symbolic::one()));
}

Einsum2BLASSymm::Einsum2BLASSymm(einsum::EinsumNode& einsum_node) : einsum_node_(einsum_node) {}

std::string Einsum2BLASSymm::name() const { return "Einsum2BLASSymm"; }

bool Einsum2BLASSymm::can_be_applied(builder::StructuredSDFGBuilder& builder,
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

    // Check side and triangular
    bool ll = this->check_left_lower(outer_1, outer_2, inner);
    bool lu = this->check_left_upper(outer_1, outer_2, inner);
    bool rl = this->check_right_lower(outer_1, outer_2, inner);
    bool ru = this->check_right_upper(outer_1, outer_2, inner);
    if (ll + lu + rl + ru != 1) return false;
    bool left = ll || lu;

    // Check inputs
    long long A = -1, B = -1, C = -1;
    if (this->einsum_node_.inputs().size() == 3) {
        C = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() != 2) {
                break;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1)) {
                if (left)
                    A = i;
                else
                    B = i;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2)) {
                if (left)
                    B = i;
                else
                    A = i;
            }
        }
    } else if (this->einsum_node_.inputs().size() == 4) {
        C = 3;
        long long alpha = -1;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 0) {
                alpha = i;
            } else if (this->einsum_node_.in_indices(i).size() != 2) {
                break;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1)) {
                if (left)
                    A = i;
                else
                    B = i;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2)) {
                if (left)
                    B = i;
                else
                    A = i;
            }
        }

        // Check alpha
        if (alpha == -1) return false;
        if (this->einsum_node_.in_indices(alpha).size() != 0) return false;
    } else {
        return false;
    }
    if (A == -1 || B == -1 || A == B) return false;
    if (this->einsum_node_.input(C) != this->einsum_node_.output(0)) return false;

    // Check in indices
    if (left) {
        if (this->einsum_node_.in_indices(A).size() != 2) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A, 0), indvar_outer_1)) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A, 1), indvar_inner)) return false;

        if (this->einsum_node_.in_indices(B).size() != 2) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(B, 0), indvar_inner)) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(B, 1), indvar_outer_2)) return false;
    } else {
        if (this->einsum_node_.in_indices(A).size() != 2) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A, 0), indvar_inner)) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(A, 1), indvar_outer_2)) return false;

        if (this->einsum_node_.in_indices(B).size() != 2) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(B, 0), indvar_outer_1)) return false;
        if (!symbolic::eq(this->einsum_node_.in_index(B, 1), indvar_inner)) return false;
    }

    if (this->einsum_node_.in_indices(C).size() != 2) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(C, 0), indvar_outer_1)) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(C, 1), indvar_outer_2)) return false;

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

void Einsum2BLASSymm::apply(builder::StructuredSDFGBuilder& builder,
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

    // Determine side, triangular, m, and n
    bool ll = this->check_left_lower(outer_1, outer_2, inner);
    bool lu = this->check_left_upper(outer_1, outer_2, inner);
    bool rl = this->check_right_lower(outer_1, outer_2, inner);
    blas::BLASSide side = (ll || lu) ? blas::BLASSide_Left : blas::BLASSide_Right;
    blas::BLASTriangular uplo =
        (ll || rl) ? blas::BLASTriangular_Lower : blas::BLASTriangular_Upper;
    symbolic::Expression m, n;
    if (lu)
        m = this->einsum_node_.num_iteration(inner);
    else
        m = this->einsum_node_.num_iteration(outer_1);
    if (rl)
        n = this->einsum_node_.num_iteration(inner);
    else
        n = this->einsum_node_.num_iteration(outer_2);

    // Determine inputs
    long long alpha = -1, A = -1, B = -1, C = -1;
    bool has_alpha = false;
    if (this->einsum_node_.inputs().size() == 3) {
        C = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() != 2) {
                break;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1)) {
                if (side == blas::BLASSide_Left)
                    A = i;
                else
                    B = i;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2)) {
                if (side == blas::BLASSide_Left)
                    B = i;
                else
                    A = i;
            }
        }
    } else if (this->einsum_node_.inputs().size() == 4) {
        C = 3;
        has_alpha = true;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 0) {
                alpha = i;
            } else if (this->einsum_node_.in_indices(i).size() != 2) {
                break;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_outer_1)) {
                if (side == blas::BLASSide_Left)
                    A = i;
                else
                    B = i;
            } else if (symbolic::eq(this->einsum_node_.in_index(i, 1), indvar_outer_2)) {
                if (side == blas::BLASSide_Left)
                    B = i;
                else
                    A = i;
            }
        }
    }

    // Determine alpha
    std::string alpha_input = has_alpha ? this->einsum_node_.input(alpha)
                                        : ((type == blas::BLASType_real) ? "1.0f" : "1.0");

    // Add the BLAS node for symm
    data_flow::LibraryNode& libnode =
        builder.add_library_node<blas::BLASNodeSymm, const blas::BLASType, blas::BLASSide,
                                 blas::BLASTriangular, symbolic::Expression, symbolic::Expression,
                                 std::string, std::string, std::string, std::string>(
            *block, this->einsum_node_.debug_info(), type, side, uplo, m, n, alpha_input,
            this->einsum_node_.input(A), this->einsum_node_.input(B), this->einsum_node_.input(C));

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

void Einsum2BLASSymm::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_id"] = this->einsum_node_.element_id();
}

Einsum2BLASSymm Einsum2BLASSymm::from_json(builder::StructuredSDFGBuilder& builder,
                                           const nlohmann::json& j) {
    size_t einsum_node_id = j["einsum_node_id"].get<size_t>();
    Element* einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found.");
    }
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);

    return Einsum2BLASSymm(*einsum_node);
}

}  // namespace transformations
}  // namespace sdfg