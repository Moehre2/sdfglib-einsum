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

inline data_flow::LibraryNodeCode LibraryNodeType_BLAS_symv("BLAS symv");

class BLASNodeSymv : public BLASNode {
    BLASTriangular uplo_;
    symbolic::Expression n_;

   public:
    BLASNodeSymv(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                 data_flow::DataFlowGraph& parent, const BLASType type, BLASTriangular uplo,
                 symbolic::Expression n, std::string alpha, std::string A, std::string x,
                 std::string y);

    BLASNodeSymv(const BLASNodeSymv&) = delete;
    BLASNodeSymv& operator=(const BLASNodeSymv&) = delete;

    virtual ~BLASNodeSymv() = default;

    BLASTriangular uplo() const;

    symbolic::Expression n() const;

    std::string alpha() const;
    std::string A() const;
    std::string x() const;
    std::string y() const;

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        size_t element_id, const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent) const override;

    virtual std::string toStr() const override;
};

}  // namespace blas
}  // namespace sdfg