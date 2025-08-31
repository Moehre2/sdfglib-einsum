#include "sdfg/transformations/einsum2blas_gemv.h"

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
#include "sdfg/blas/blas_node_gemv.h"
#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

bool Einsum2BLASGemv::check_matrix_indices(const symbolic::Expression& mat_index1,
                                           const symbolic::Expression& mat_index2,
                                           const symbolic::Symbol& loop_index1,
                                           const symbolic::Symbol& loop_index2) {
    if (symbolic::eq(mat_index1, mat_index2)) return false;
    if (!symbolic::eq(mat_index1, loop_index1) && !symbolic::eq(mat_index1, loop_index2))
        return false;
    if (!symbolic::eq(mat_index2, loop_index1) && !symbolic::eq(mat_index2, loop_index2))
        return false;
    return true;
}

Einsum2BLASGemv::Einsum2BLASGemv(einsum::EinsumNode& einsum_node) : einsum_node_(einsum_node) {}

std::string Einsum2BLASGemv::name() const { return "Einsum2BLASGemv"; }

bool Einsum2BLASGemv::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                     analysis::AnalysisManager& analysis_manager) {
    // Check maps
    if (this->einsum_node_.maps().size() != 2) return false;

    // Check out indices
    symbolic::Symbol indvar_outer, indvar_inner;
    if (this->einsum_node_.out_indices().size() != 1) return false;
    if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(0))) {
        indvar_outer = this->einsum_node_.indvar(0);
        indvar_inner = this->einsum_node_.indvar(1);
    } else if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(1))) {
        indvar_outer = this->einsum_node_.indvar(1);
        indvar_inner = this->einsum_node_.indvar(0);
    } else {
        return false;
    }

    // Check bounds
    if (symbolic::uses(this->einsum_node_.num_iteration(0), indvar_outer)) return false;
    if (symbolic::uses(this->einsum_node_.num_iteration(0), indvar_inner)) return false;
    if (symbolic::uses(this->einsum_node_.num_iteration(1), indvar_outer)) return false;
    if (symbolic::uses(this->einsum_node_.num_iteration(1), indvar_inner)) return false;

    // Check inputs
    long long A = -1, x = -1, y = -1;
    if (this->einsum_node_.inputs().size() == 3) {
        y = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            switch (this->einsum_node_.in_indices(i).size()) {
                case 1:
                    x = i;
                    break;
                case 2:
                    A = i;
                    break;
            }
        }
    } else if (this->einsum_node_.inputs().size() == 4) {
        y = 3;
        long long alpha = -1;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            switch (this->einsum_node_.in_indices(i).size()) {
                case 0:
                    alpha = i;
                    break;
                case 1:
                    x = i;
                    break;
                case 2:
                    A = i;
                    break;
            }
        }

        // Check alpha
        if (alpha == -1) return false;
        if (this->einsum_node_.in_indices(alpha).size() != 0) return false;
    } else {
        return false;
    }
    if (A == -1 || x == -1 || x == y) return false;
    if (this->einsum_node_.input(y) != this->einsum_node_.output(0)) return false;

    // Check in indices
    if (this->einsum_node_.in_indices(A).size() != 2) return false;
    if (!this->check_matrix_indices(this->einsum_node_.in_index(A, 0),
                                    this->einsum_node_.in_index(A, 1), indvar_outer, indvar_inner))
        return false;
    // bool trans = symbolic::eq(this->einsum_node_.in_index(A, 0), indvar_inner);

    if (this->einsum_node_.in_indices(x).size() != 1) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(x, 0), indvar_inner)) return false;

    if (this->einsum_node_.in_indices(y).size() != 1) return false;
    if (!symbolic::eq(this->einsum_node_.in_index(y, 0), indvar_outer)) return false;

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

void Einsum2BLASGemv::apply(builder::StructuredSDFGBuilder& builder,
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

    // Determine the input positions
    long long alpha = -1, A = -1, x = -1, y = -1;
    bool has_alpha = false;
    if (this->einsum_node_.inputs().size() == 3) {
        y = 2;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            switch (this->einsum_node_.in_indices(i).size()) {
                case 1:
                    x = i;
                    break;
                case 2:
                    A = i;
                    break;
            }
        }
    } else {
        y = 3;
        has_alpha = true;
        for (size_t i = 0; i < this->einsum_node_.in_indices().size() - 1; ++i) {
            switch (this->einsum_node_.in_indices(i).size()) {
                case 0:
                    alpha = i;
                    break;
                case 1:
                    x = i;
                    break;
                case 2:
                    A = i;
                    break;
            }
        }
    }

    // Determine m and n and if matrix is accessed in a transposed manner
    size_t outer, inner;
    if (symbolic::eq(this->einsum_node_.out_index(0), this->einsum_node_.indvar(0))) {
        outer = 0;
        inner = 1;
    } else {
        outer = 1;
        inner = 0;
    }
    symbolic::Expression m, n;
    blas::BLASTranspose trans;
    if (symbolic::eq(this->einsum_node_.in_index(A, 0), this->einsum_node_.out_index(0))) {
        m = this->einsum_node_.num_iteration(outer);
        n = this->einsum_node_.num_iteration(inner);
        trans = blas::BLASTranspose_No;
    } else {
        m = this->einsum_node_.num_iteration(inner);
        n = this->einsum_node_.num_iteration(outer);
        trans = blas::BLASTranspose_Transpose;
    }

    // Determine alpha
    std::string alpha_input = has_alpha ? this->einsum_node_.input(alpha)
                                        : ((type == blas::BLASType_real) ? "1.0f" : "1.0");

    // Add the BLAS node for gemv
    data_flow::LibraryNode& libnode =
        builder.add_library_node<blas::BLASNodeGemv, const blas::BLASType, blas::BLASTranspose,
                                 symbolic::Expression, symbolic::Expression, std::string,
                                 std::string, std::string, std::string>(
            *block, this->einsum_node_.debug_info(), type, trans, m, n, alpha_input,
            this->einsum_node_.input(A), this->einsum_node_.input(x), this->einsum_node_.input(y));

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

void Einsum2BLASGemv::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_id"] = this->einsum_node_.element_id();
}

Einsum2BLASGemv Einsum2BLASGemv::from_json(builder::StructuredSDFGBuilder& builder,
                                           const nlohmann::json& j) {
    size_t einsum_node_id = j["einsum_node_id"].get<size_t>();
    Element* einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found.");
    }
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);

    return Einsum2BLASGemv(*einsum_node);
}

}  // namespace transformations
}  // namespace sdfg