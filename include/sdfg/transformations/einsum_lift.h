#pragma once

#include <functional>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class EinsumLift : public Transformation {
    data_flow::Tasklet& init_tasklet_;
    std::vector<std::reference_wrapper<structured_control_flow::StructuredLoop>> loops_;
    structured_control_flow::Block& comp_block_;

   public:
    EinsumLift(data_flow::Tasklet& init_tasklet,
               std::vector<std::reference_wrapper<structured_control_flow::StructuredLoop>> loops,
               structured_control_flow::Block& comp_block);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static EinsumLift from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

}  // namespace transformations
}  // namespace sdfg
