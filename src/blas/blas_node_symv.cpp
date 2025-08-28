#include "sdfg/blas/blas_node_symv.h"

#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/element.h>
#include <sdfg/graph/graph.h>
#include <sdfg/symbolic/symbolic.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>

#include "sdfg/blas/blas_node.h"

namespace sdfg {
namespace blas {

BLASNodeSymv::BLASNodeSymv(size_t element_id, const DebugInfo& debug_info,
                           const graph::Vertex vertex, data_flow::DataFlowGraph& parent,
                           const BLASType type, BLASTriangular uplo, symbolic::Expression n,
                           std::string alpha, std::string A, std::string x, std::string y)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_symv, {y},
               {alpha, A, x, y}, type),
      uplo_(uplo),
      n_(n) {}

BLASTriangular BLASNodeSymv::uplo() const { return this->uplo_; }

symbolic::Expression BLASNodeSymv::n() const { return this->n_; }

std::string BLASNodeSymv::alpha() const { return this->input(0); }

std::string BLASNodeSymv::A() const { return this->input(1); }

std::string BLASNodeSymv::x() const { return this->input(2); }

std::string BLASNodeSymv::y() const { return this->input(3); }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeSymv::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeSymv>(element_id, this->debug_info(), vertex, parent,
                                          this->type(), this->uplo(), this->n(), this->alpha(),
                                          this->A(), this->x(), this->y());
}

std::string BLASNodeSymv::toStr() const {
    std::stringstream stream;

    stream << blasType2String(this->type()) << "symv(" << blasTriangular2String(this->uplo())
           << ", " << this->n()->__str__() << ", " << this->alpha() << ", " << this->A() << ", "
           << this->n()->__str__() << ", " << this->x() << ", 1, 1.0, " << this->y() << ", 1)";

    return stream.str();
}

}  // namespace blas
}  // namespace sdfg