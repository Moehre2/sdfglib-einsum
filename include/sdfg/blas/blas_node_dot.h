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

inline data_flow::LibraryNodeCode LibraryNodeType_BLAS_dot("BLAS dot");

class BLASNodeDot : public BLASNode {
    symbolic::Expression n_;

   public:
    BLASNodeDot(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                data_flow::DataFlowGraph& parent, std::string result, const BLASType type,
                symbolic::Expression n, std::string x, std::string y);

    BLASNodeDot(const BLASNodeDot&) = delete;
    BLASNodeDot& operator=(const BLASNodeDot&) = delete;

    virtual ~BLASNodeDot() = default;

    std::string result() const;

    symbolic::Expression n() const;

    std::string x() const;
    std::string y() const;

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        size_t element_id, const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent) const override;

    virtual std::string toStr() const override;
};

}  // namespace blas
}  // namespace sdfg