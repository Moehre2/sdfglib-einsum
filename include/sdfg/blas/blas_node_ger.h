#pragma once

#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/element.h>
#include <sdfg/graph/graph.h>
#include <sdfg/symbolic/symbolic.h>

#include <memory>
#include <string>

#include "sdfg/blas/blas_node.h"

namespace sdfg {
namespace blas {

inline data_flow::LibraryNodeCode LibraryNodeType_BLAS_ger("BLAS ger");

class BLASNodeGer : public BLASNode {
    symbolic::Expression m_, n_;

   public:
    BLASNodeGer(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                data_flow::DataFlowGraph& parent, const BLASType type, symbolic::Expression m,
                symbolic::Expression n, std::string alpha, std::string x, std::string y,
                std::string A);

    BLASNodeGer(const BLASNodeGer&) = delete;
    BLASNodeGer& operator=(const BLASNodeGer&) = delete;

    virtual ~BLASNodeGer() = default;

    symbolic::Expression m() const;
    symbolic::Expression n() const;

    std::string alpha() const;
    std::string x() const;
    std::string y() const;
    std::string A() const;

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        size_t element_id, const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent) const override;

    virtual std::string toStr() const override;
};

}  // namespace blas
}  // namespace sdfg