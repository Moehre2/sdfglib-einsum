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

inline data_flow::LibraryNodeCode LibraryNodeType_BLAS_syr("BLAS syr");

class BLASNodeSyr : public BLASNode {
    BLASTriangular uplo_;
    symbolic::Expression n_;

   public:
    BLASNodeSyr(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                data_flow::DataFlowGraph& parent, const BLASType type, BLASTriangular uplo,
                symbolic::Expression n, std::string alpha, std::string x, std::string A);

    BLASNodeSyr(const BLASNodeSyr&) = delete;
    BLASNodeSyr& operator=(const BLASNodeSyr&) = delete;

    virtual ~BLASNodeSyr() = default;

    BLASTriangular uplo() const;

    symbolic::Expression n() const;

    std::string alpha() const;
    std::string x() const;
    std::string A() const;

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        size_t element_id, const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent) const override;

    virtual std::string toStr() const override;
};

}  // namespace blas
}  // namespace sdfg