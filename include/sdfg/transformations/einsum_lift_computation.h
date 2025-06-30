#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class EinsumLiftComputation : public Transformation {
    structured_control_flow::Block& comp_block_;

   public:
    EinsumLiftComputation(structured_control_flow::Block& comp_block);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static EinsumLiftComputation from_json(builder::StructuredSDFGBuilder& builder,
                                           const nlohmann::json& j);
};

}  // namespace transformations
}  // namespace sdfg
