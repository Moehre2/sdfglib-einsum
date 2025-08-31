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

class Einsum2BLASGemm : public Transformation {
    einsum::EinsumNode& einsum_node_;

    bool check_indvars(size_t indvar1, size_t indvar2);
    bool check_matrix_indices(const symbolic::Expression& mat_index1,
                              const symbolic::Expression& mat_index2,
                              const symbolic::Symbol& loop_index1,
                              const symbolic::Symbol& loop_index2);

   public:
    Einsum2BLASGemm(einsum::EinsumNode& einsum_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static Einsum2BLASGemm from_json(builder::StructuredSDFGBuilder& builder,
                                     const nlohmann::json& j);
};

}  // namespace transformations
}  // namespace sdfg