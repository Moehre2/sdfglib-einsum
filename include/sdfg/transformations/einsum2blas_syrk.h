#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/transformations/transformation.h>

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

class Einsum2BLASSyrk : public Transformation {
    einsum::EinsumNode& einsum_node_;

    bool check_lower(size_t outer_1, size_t outer_2, size_t inner);
    bool check_upper(size_t outer_1, size_t outer_2, size_t inner);

   public:
    Einsum2BLASSyrk(einsum::EinsumNode& einsum_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static Einsum2BLASSyrk from_json(builder::StructuredSDFGBuilder& builder,
                                     const nlohmann::json& j);
};

}  // namespace transformations
}  // namespace sdfg