#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/graph/graph.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace blas {

inline data_flow::LibraryNodeCode LibraryNodeType_BLAS_gemm("BLAS sgemm");

class BLASNode : public data_flow::LibraryNode {
    symbolic::Expression m_, n_, k_;

   public:
    BLASNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
             data_flow::DataFlowGraph& parent, const std::vector<std::string>& outputs,
             const std::vector<std::string>& inputs, symbolic::Expression n, symbolic::Expression m,
             symbolic::Expression k);

    BLASNode(const BLASNode&) = delete;
    BLASNode& operator=(const BLASNode&) = delete;

    virtual ~BLASNode() = default;

    symbolic::Expression m() const;
    symbolic::Expression n() const;
    symbolic::Expression k() const;

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        size_t element_id, const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent) const override;

    virtual symbolic::SymbolSet symbols() const override;

    virtual void replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) override;

    virtual std::string toStr() const override;
};

}  // namespace blas
}  // namespace sdfg
