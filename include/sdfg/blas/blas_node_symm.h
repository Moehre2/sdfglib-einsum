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

inline data_flow::LibraryNodeCode LibraryNodeType_BLAS_symm("BLAS symm");

class BLASNodeSymm : public BLASNode {
    BLASSide side_;
    BLASTriangular uplo_;
    symbolic::Expression m_, n_;

   public:
    BLASNodeSymm(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                 data_flow::DataFlowGraph& parent, const BLASType type, BLASSide side,
                 BLASTriangular uplo, symbolic::Expression m, symbolic::Expression n,
                 std::string alpha, std::string A, std::string B, std::string C);

    BLASNodeSymm(const BLASNodeSymm&) = delete;
    BLASNodeSymm& operator=(const BLASNodeSymm&) = delete;

    virtual ~BLASNodeSymm() = default;

    BLASSide side() const;

    BLASTriangular uplo() const;

    symbolic::Expression m() const;
    symbolic::Expression n() const;

    std::string alpha() const;
    std::string A() const;
    std::string B() const;
    std::string C() const;

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        size_t element_id, const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent) const override;

    virtual std::string toStr() const override;
};

}  // namespace blas
}  // namespace sdfg
