#include "sdfg/transformations/einsum_lift.h"

#include <functional>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

EinsumLift::EinsumLift(
    data_flow::Tasklet& init_tasklet,
    std::vector<std::reference_wrapper<structured_control_flow::StructuredLoop>> loops,
    structured_control_flow::Block& comp_block)
    : init_tasklet_(init_tasklet), loops_(loops), comp_block_(comp_block) {}

std::string EinsumLift::name() const { return "EinsumLift"; }

bool EinsumLift::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) {
    return true;
}

void EinsumLift::apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) {}

void EinsumLift::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["init_tasklet_element_id"] = this->init_tasklet_.element_id();
    j["loops_element_ids"] = nlohmann::json::array();
    for (auto loop : this->loops_) j["loops_element_ids"].push_back(loop.get().element_id());
    j["comp_block_element_id"] = this->comp_block_.element_id();
}

EinsumLift EinsumLift::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    size_t init_tasklet_id = j["init_tasklet_element_id"].get<size_t>();
    auto init_tasklet_element = builder.find_element_by_id(init_tasklet_id);
    if (!init_tasklet_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(init_tasklet_id) + " not found.");
    }
    auto init_tasklet = dynamic_cast<data_flow::Tasklet*>(init_tasklet_element);

    std::vector<std::reference_wrapper<structured_control_flow::StructuredLoop>> loops;
    std::vector<size_t> loop_ids = j["loop_element_id"].get<std::vector<size_t>>();
    for (size_t loop_id : loop_ids) {
        auto loop_element = builder.find_element_by_id(loop_id);
        if (!loop_element) {
            throw InvalidTransformationDescriptionException(
                "Element with ID " + std::to_string(loop_id) + " not found.");
        }
        auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop_element);
        loops.push_back(*loop);
    }

    size_t comp_tasklet_id = j["comp_block_element_id"].get<size_t>();
    auto comp_tasklet_element = builder.find_element_by_id(comp_tasklet_id);
    if (!comp_tasklet_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(comp_tasklet_id) + " not found.");
    }
    auto comp_block = dynamic_cast<structured_control_flow::Block*>(comp_tasklet_element);

    return EinsumLift(*init_tasklet, loops, *comp_block);
}

}  // namespace transformations
}  // namespace sdfg
