#pragma once

#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/element.h>
#include <sdfg/graph/graph.h>
#include <sdfg/symbolic/symbolic.h>

#include <memory>
#include <string>
#include <vector>

#include "sdfg/blas/blas_node.h"

namespace sdfg {
namespace blas {

inline data_flow::LibraryNodeCode LibraryNodeType_BLAS_gemm("BLAS gemm");

class BLASNodeGemm : public BLASNode {
    symbolic::Expression m_, n_, k_;

   public:
    BLASNodeGemm(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                 data_flow::DataFlowGraph& parent, const std::vector<std::string>& outputs,
                 const std::vector<std::string>& inputs, const BLASType type,
                 symbolic::Expression n, symbolic::Expression m, symbolic::Expression k);

    BLASNodeGemm(const BLASNodeGemm&) = delete;
    BLASNodeGemm& operator=(const BLASNodeGemm&) = delete;

    virtual ~BLASNodeGemm() = default;

    symbolic::Expression m() const;
    symbolic::Expression n() const;
    symbolic::Expression k() const;

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        size_t element_id, const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent) const override;

    virtual std::string toStr() const override;
};

}  // namespace blas
}  // namespace sdfg
