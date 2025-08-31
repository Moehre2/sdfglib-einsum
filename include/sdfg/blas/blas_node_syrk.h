#pragma once

#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/element.h>
#include <sdfg/graph/graph.h>
#include <sdfg/symbolic/symbolic.h>

#include <string>

#include "sdfg/blas/blas_node.h"

namespace sdfg {
namespace blas {

inline data_flow::LibraryNodeCode LibraryNodeType_BLAS_syrk("BLAS syrk");

class BLASNodeSyrk : public BLASNode {
    BLASTriangular uplo_;
    BLASTranspose trans_;
    symbolic::Expression n_, k_;

   public:
    BLASNodeSyrk(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                 data_flow::DataFlowGraph& parent, const BLASType type, BLASTriangular uplo,
                 BLASTranspose trans, symbolic::Expression n, symbolic::Expression k,
                 std::string alpha, std::string A, std::string C);

    BLASNodeSyrk(const BLASNodeSyrk&) = delete;
    BLASNodeSyrk& operator=(const BLASNodeSyrk&) = delete;

    virtual ~BLASNodeSyrk() = default;

    BLASTriangular uplo() const;

    BLASTranspose trans() const;

    symbolic::Expression n() const;
    symbolic::Expression k() const;

    std::string alpha() const;
    std::string A() const;
    std::string C() const;

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        size_t element_id, const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent) const override;

    virtual std::string toStr() const override;
};

}  // namespace blas
}  // namespace sdfg
