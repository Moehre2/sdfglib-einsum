#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/transformations/transformation.h>

#include <nlohmann/json_fwd.hpp>
#include <set>
#include <string>

#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

class EinsumExpand : public Transformation {
    structured_control_flow::StructuredLoop& loop_;
    einsum::EinsumNode& einsum_node_;

    void visitElements(std::set<size_t>& elements,
                       const structured_control_flow::ControlFlowNode& node);
    bool subsetContainsSymbol(const data_flow::Subset& subset, const symbolic::Symbol& symbol);

   public:
    EinsumExpand(structured_control_flow::StructuredLoop& loop, einsum::EinsumNode& einsum_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static EinsumExpand from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

}  // namespace transformations
}  // namespace sdfg