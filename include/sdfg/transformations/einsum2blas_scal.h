#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/transformations/transformation.h>

#include <nlohmann/json_fwd.hpp>

#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

class Einsum2BLASScal : public Transformation {
    einsum::EinsumNode& einsum_node_;

   public:
    Einsum2BLASScal(einsum::EinsumNode& einsum_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static Einsum2BLASScal from_json(builder::StructuredSDFGBuilder& builder,
                                     const nlohmann::json& j);
};

}  // namespace transformations
}  // namespace sdfg