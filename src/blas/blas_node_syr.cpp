#include "sdfg/blas/blas_node_syr.h"

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

BLASNodeSyr::BLASNodeSyr(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                         data_flow::DataFlowGraph& parent, const BLASType type, BLASTriangular uplo,
                         symbolic::Expression n, std::string alpha, std::string x, std::string A)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_syr, {A}, {alpha, x, A},
               type),
      uplo_(uplo),
      n_(n) {}

BLASTriangular BLASNodeSyr::uplo() const { return this->uplo_; }

symbolic::Expression BLASNodeSyr::n() const { return this->n_; }

std::string BLASNodeSyr::alpha() const { return this->input(0); }

std::string BLASNodeSyr::x() const { return this->input(1); }

std::string BLASNodeSyr::A() const { return this->input(2); }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeSyr::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeSyr>(element_id, this->debug_info(), vertex, parent,
                                         this->type(), this->uplo(), this->n(), this->alpha(),
                                         this->x(), this->A());
}

std::string BLASNodeSyr::toStr() const {
    std::stringstream stream;

    stream << blasType2String(this->type()) << "syr(" << blasTriangular2String(this->uplo()) << ", "
           << this->n()->__str__() << ", " << this->alpha() << ", " << this->x() << ", 1, "
           << this->A() << ", " << this->n()->__str__() << ")";

    return stream.str();
}

}  // namespace blas
}  // namespace sdfg