#include "sdfg/transformations/einsum2blas_syr.h"

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
#include "sdfg/blas/blas_node_syr.h"
#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

bool Einsum2BLASSyr::check_L() {
    // Maps: (i, n), (j, i+1)
    if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(0)) &&
        symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(1)) &&
        symbolic::eq(this->einsum_node_.num_iteration(1),
                     symbolic::add(this->einsum_node_.indvar(0), symbolic::one())))
        return true;
    // Maps: (j, i+1), (i, n)
    else if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(1)) &&
             symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(0)) &&
             symbolic::eq(this->einsum_node_.num_iteration(0),
                          symbolic::add(this->einsum_node_.indvar(1), symbolic::one())))
        return true;
    else
        return false;
}

bool Einsum2BLASSyr::check_U() {
    // Maps: (i, j+1), (j, n)
    if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(0)) &&
        symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(1)) &&
        symbolic::eq(this->einsum_node_.num_iteration(0),
                     symbolic::add(this->einsum_node_.indvar(1), symbolic::one())))
        return true;
    // Maps: (j, n), (i, j+1)
    else if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(1)) &&
             symbolic::eq(this->einsum_node_.out_index(1), this->einsum_node_.indvar(0)) &&
             symbolic::eq(this->einsum_node_.num_iteration(1),
                          symbolic::add(this->einsum_node_.indvar(0), symbolic::one())))
        return true;
    else
        return false;
}

Einsum2BLASSyr::Einsum2BLASSyr(einsum::EinsumNode& einsum_node) : einsum_node_(einsum_node) {}

std::string Einsum2BLASSyr::name() const { return "Einsum2BLASSyr"; }

bool Einsum2BLASSyr::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                    analysis::AnalysisManager& analysis_manager) {
    // Check maps
    if (this->einsum_node_.maps().size() != 2) return false;

    // Check out indices size
    if (this->einsum_node_.out_indices().size() != 2) return false;

    // Check triangular
    // Exactly one must be true
    if (this->check_L() + this->check_U() != 1) return false;

    // Check out indices
    symbolic::Symbol indvar_1, indvar_2;
    if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(0))) {
        indvar_1 = this->einsum_node_.indvar(0);
        indvar_2 = this->einsum_node_.indvar(1);
    } else if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(1))) {
        indvar_1 = this->einsum_node_.indvar(1);
        indvar_2 = this->einsum_node_.indvar(0);
    } else {
        return false;
    }

    // Check inputs
    long long x1 = -1, x2 = -1, A = -1;
    if (this->einsum_node_.inputs().size() == 3) {
        A = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 1 &&
                symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_1))
                x1 = i;
            else if (this->einsum_node_.in_indices(i).size() == 1 &&
                     symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_2))
                x2 = i;
        }
    } else if (this->einsum_node_.inputs().size() == 4) {
        A = 3;
        long long alpha = -1;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 1 &&
                symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_1))
                x1 = i;
            else if (this->einsum_node_.in_indices(i).size() == 1 &&
                     symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_2))
                x2 = i;
            else if (this->einsum_node_.in_indices(i).size() == 0)
                alpha = i;
        }

        // Check alpha
        if (alpha == -1) return false;
        if (this->einsum_node_.in_indices(alpha).size() != 0) return false;
    } else {
        return false;
    }
    if (x1 == -1 || x2 == -1 || A == -1 || x1 == x2) return false;
    if (this->einsum_node_.input(A) != this->einsum_node_.output(0)) return false;

    // Check in indices
    if (this->einsum_node_.in_indices(A).size() != 2) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(A, 0), indvar_1)) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(A, 1), indvar_2)) return false;

    if (this->einsum_node_.in_indices(x1).size() != 1) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(x1, 0), indvar_1)) return false;

    if (this->einsum_node_.in_indices(x2).size() != 1) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(x2, 0), indvar_2)) return false;

    // Get the data flow graph
    auto& dfg = this->einsum_node_.get_parent();

    // Check that x1 and x2 access the same container
    std::string x1_container, x2_container;
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        if (iedge.dst_conn() == this->einsum_node_.input(x1))
            x1_container = dynamic_cast<const data_flow::AccessNode&>(iedge.src()).data();
        else if (iedge.dst_conn() == this->einsum_node_.input(x2))
            x2_container = dynamic_cast<const data_flow::AccessNode&>(iedge.src()).data();
    }
    if (x1_container.empty()) return false;
    if (x2_container.empty()) return false;
    if (x1_container != x2_container) return false;

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

void Einsum2BLASSyr::apply(builder::StructuredSDFGBuilder& builder,
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

    // Determine indvars and n
    symbolic::Symbol indvar_1, indvar_2;
    symbolic::Expression n;
    blas::BLASTriangular uplo =
        this->check_L() ? blas::BLASTriangular_Lower : blas::BLASTriangular_Upper;
    if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(0))) {
        indvar_1 = this->einsum_node_.indvar(0);
        indvar_2 = this->einsum_node_.indvar(1);
        if (uplo == blas::BLASTriangular_Lower)
            n = this->einsum_node_.num_iteration(0);
        else
            n = this->einsum_node_.num_iteration(1);
    } else if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(1))) {
        indvar_1 = this->einsum_node_.indvar(1);
        indvar_2 = this->einsum_node_.indvar(0);
        if (uplo == blas::BLASTriangular_Lower)
            n = this->einsum_node_.num_iteration(1);
        else
            n = this->einsum_node_.num_iteration(0);
    }

    // Determine the input positions
    long long alpha = -1, x = -1, x_del = -1, A = -1;
    bool has_alpha = false;
    if (this->einsum_node_.inputs().size() == 3) {
        A = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 1 &&
                symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_1))
                x = i;
            else if (this->einsum_node_.in_indices(i).size() == 1 &&
                     symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_2))
                x_del = i;
        }
    } else if (this->einsum_node_.inputs().size() == 4) {
        A = 3;
        has_alpha = true;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            if (this->einsum_node_.in_indices(i).size() == 1 &&
                symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_1))
                x = i;
            else if (this->einsum_node_.in_indices(i).size() == 1 &&
                     symbolic::eq(this->einsum_node_.in_index(i, 0), indvar_2))
                x_del = i;
            else if (this->einsum_node_.in_indices(i).size() == 0)
                alpha = i;
        }
    }

    // Determine alpha
    std::string alpha_input = has_alpha ? this->einsum_node_.input(alpha)
                                        : ((type == blas::BLASType_real) ? "1.0f" : "1.0");

    // Add the BLAS node for syr
    data_flow::LibraryNode& libnode =
        builder.add_library_node<blas::BLASNodeSyr, const blas::BLASType, blas::BLASTriangular,
                                 symbolic::Expression, std::string, std::string, std::string>(
            *block, this->einsum_node_.debug_info(), type, uplo, n, alpha_input,
            this->einsum_node_.input(x), this->einsum_node_.input(A));

    // Copy the memlets
    for (auto& iedge : dfg.in_edges(this->einsum_node_)) {
        if (iedge.dst_conn() == this->einsum_node_.input(x_del)) continue;
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

void Einsum2BLASSyr::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_id"] = this->einsum_node_.element_id();
}

Einsum2BLASSyr Einsum2BLASSyr::from_json(builder::StructuredSDFGBuilder& builder,
                                         const nlohmann::json& j) {
    size_t einsum_node_id = j["einsum_node_id"].get<size_t>();
    Element* einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found.");
    }
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);

    return Einsum2BLASSyr(*einsum_node);
}

}  // namespace transformations
}  // namespace sdfg