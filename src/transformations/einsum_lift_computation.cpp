#include "sdfg/transformations/einsum_lift_computation.h"

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

EinsumLiftComputation::EinsumLiftComputation(structured_control_flow::Block& comp_block)
    : comp_block_(comp_block) {}

std::string EinsumLiftComputation::name() const { return "EinsumLiftComputation"; }

bool EinsumLiftComputation::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                           analysis::AnalysisManager& analysis_manager) {
    return true;
}

void EinsumLiftComputation::apply(builder::StructuredSDFGBuilder& builder,
                                  analysis::AnalysisManager& analysis_manager) {}

void EinsumLiftComputation::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["comp_block_element_id"] = this->comp_block_.element_id();
}

EinsumLiftComputation EinsumLiftComputation::from_json(builder::StructuredSDFGBuilder& builder,
                                                       const nlohmann::json& j) {
    size_t comp_tasklet_id = j["comp_block_element_id"].get<size_t>();
    auto comp_tasklet_element = builder.find_element_by_id(comp_tasklet_id);
    if (!comp_tasklet_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(comp_tasklet_id) + " not found.");
    }
    auto comp_block = dynamic_cast<structured_control_flow::Block*>(comp_tasklet_element);

    return EinsumLiftComputation(*comp_block);
}

}  // namespace transformations
}  // namespace sdfg
