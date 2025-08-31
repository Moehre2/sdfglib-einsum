#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/transformations/transformation.h>

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/einsum/einsum_node.h"
#include "sdfg/transformations/einsum2blas_axpy.h"
#include "sdfg/transformations/einsum2blas_copy.h"
#include "sdfg/transformations/einsum2blas_dot.h"
#include "sdfg/transformations/einsum2blas_gemm.h"
#include "sdfg/transformations/einsum2blas_gemv.h"
#include "sdfg/transformations/einsum2blas_ger.h"
#include "sdfg/transformations/einsum2blas_symm.h"
#include "sdfg/transformations/einsum2blas_symv.h"
#include "sdfg/transformations/einsum2blas_syr.h"

namespace sdfg {
namespace transformations {

class Einsum2BLAS : public Transformation {
    einsum::EinsumNode& einsum_node_;
    Einsum2BLASAxpy axpy_;
    Einsum2BLASCopy copy_;
    Einsum2BLASDot dot_;
    Einsum2BLASGemv gemv_;
    Einsum2BLASSymv symv_;
    Einsum2BLASGer ger_;
    Einsum2BLASSyr syr_;
    Einsum2BLASGemm gemm_;
    Einsum2BLASSymm symm_;

   public:
    Einsum2BLAS(einsum::EinsumNode& einsum_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static Einsum2BLAS from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

}  // namespace transformations
}  // namespace sdfg
