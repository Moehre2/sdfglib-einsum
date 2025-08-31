#include "sdfg/transformations/einsum2blas.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/element.h>
#include <sdfg/transformations/transformation.h>

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

Einsum2BLAS::Einsum2BLAS(einsum::EinsumNode& einsum_node)
    : einsum_node_(einsum_node),
      axpy_(einsum_node),
      copy_(einsum_node),
      dot_(einsum_node),
      gemv_(einsum_node),
      symv_(einsum_node),
      ger_(einsum_node),
      syr_(einsum_node),
      gemm_(einsum_node),
      symm_(einsum_node) {}

std::string Einsum2BLAS::name() const { return "Einsum2BLAS"; }

bool Einsum2BLAS::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                 analysis::AnalysisManager& analysis_manager) {
    if (this->axpy_.can_be_applied(builder, analysis_manager)) return true;
    if (this->copy_.can_be_applied(builder, analysis_manager)) return true;
    if (this->dot_.can_be_applied(builder, analysis_manager)) return true;
    if (this->gemv_.can_be_applied(builder, analysis_manager)) return true;
    if (this->symv_.can_be_applied(builder, analysis_manager)) return true;
    if (this->ger_.can_be_applied(builder, analysis_manager)) return true;
    if (this->syr_.can_be_applied(builder, analysis_manager)) return true;
    if (this->gemm_.can_be_applied(builder, analysis_manager)) return true;
    if (this->symm_.can_be_applied(builder, analysis_manager)) return true;
    return false;
}

void Einsum2BLAS::apply(builder::StructuredSDFGBuilder& builder,
                        analysis::AnalysisManager& analysis_manager) {
    if (this->axpy_.can_be_applied(builder, analysis_manager)) {
        this->axpy_.apply(builder, analysis_manager);
        return;
    }
    if (this->copy_.can_be_applied(builder, analysis_manager)) {
        this->copy_.apply(builder, analysis_manager);
        return;
    }
    if (this->dot_.can_be_applied(builder, analysis_manager)) {
        this->dot_.apply(builder, analysis_manager);
        return;
    }
    if (this->gemv_.can_be_applied(builder, analysis_manager)) {
        this->gemv_.apply(builder, analysis_manager);
        return;
    }
    if (this->symv_.can_be_applied(builder, analysis_manager)) {
        this->symv_.apply(builder, analysis_manager);
        return;
    }
    if (this->ger_.can_be_applied(builder, analysis_manager)) {
        this->ger_.apply(builder, analysis_manager);
        return;
    }
    if (this->syr_.can_be_applied(builder, analysis_manager)) {
        this->syr_.apply(builder, analysis_manager);
        return;
    }
    if (this->gemm_.can_be_applied(builder, analysis_manager)) {
        this->gemm_.apply(builder, analysis_manager);
        return;
    }
    if (this->symm_.can_be_applied(builder, analysis_manager)) {
        this->symm_.apply(builder, analysis_manager);
        return;
    }
}

void Einsum2BLAS::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["einsum_node_id"] = this->einsum_node_.element_id();
}

Einsum2BLAS Einsum2BLAS::from_json(builder::StructuredSDFGBuilder& builder,
                                   const nlohmann::json& j) {
    size_t einsum_node_id = j["einsum_node_id"].get<size_t>();
    Element* einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found.");
    }
    auto* einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);

    return Einsum2BLAS(*einsum_node);
}

}  // namespace transformations
}  // namespace sdfg
