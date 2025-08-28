#include "sdfg/blas/blas_node_symm.h"

#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/element.h>
#include <sdfg/exceptions.h>
#include <sdfg/graph/graph.h>
#include <sdfg/symbolic/symbolic.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "sdfg/blas/blas_node.h"

namespace sdfg {
namespace blas {

BLASNodeSymm::BLASNodeSymm(size_t element_id, const DebugInfo& debug_info,
                           const graph::Vertex vertex, data_flow::DataFlowGraph& parent,
                           const BLASType type, BLASSide side, BLASTriangular uplo,
                           symbolic::Expression m, symbolic::Expression n, std::string alpha,
                           std::string A, std::string B, std::string C)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_symm, {C},
               {alpha, A, B, C}, type),
      side_(side),
      uplo_(uplo),
      m_(m),
      n_(n) {}

BLASSide BLASNodeSymm::side() const { return this->side_; }

BLASTriangular BLASNodeSymm::uplo() const { return this->uplo_; }

symbolic::Expression BLASNodeSymm::m() const { return this->m_; }

symbolic::Expression BLASNodeSymm::n() const { return this->n_; }

std::string BLASNodeSymm::alpha() const { return this->input(0); }

std::string BLASNodeSymm::A() const { return this->input(1); }

std::string BLASNodeSymm::B() const { return this->input(2); }

std::string BLASNodeSymm::C() const { return this->input(3); }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeSymm::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeSymm>(
        element_id, this->debug_info(), vertex, parent, this->type(), this->side(), this->uplo(),
        this->n(), this->m(), this->alpha(), this->A(), this->B(), this->C());
}

std::string BLASNodeSymm::toStr() const {
    std::stringstream stream;

    const std::string m = this->m()->__str__();
    const std::string n = this->n()->__str__();

    stream << blasType2String(this->type()) << "symm(" << blasSide2String(this->side()) << ", "
           << blasTriangular2String(this->uplo()) << ", " << m << ", " << n << ", " << this->alpha()
           << ", " << this->A() << ", ";
    if (this->side() == BLASSide_Left)
        stream << m;
    else
        stream << n;
    stream << ", " << this->B() << ", " << m << ", 1.0, " << this->C() << ", " << m << ")";

    return stream.str();
}

}  // namespace blas
}  // namespace sdfg
