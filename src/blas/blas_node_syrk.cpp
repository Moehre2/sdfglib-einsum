#include "sdfg/blas/blas_node_syrk.h"

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

#include "sdfg/blas/blas_node.h"

namespace sdfg {
namespace blas {

BLASNodeSyrk::BLASNodeSyrk(size_t element_id, const DebugInfo& debug_info,
                           const graph::Vertex vertex, data_flow::DataFlowGraph& parent,
                           const BLASType type, BLASTriangular uplo, BLASTranspose trans,
                           symbolic::Expression n, symbolic::Expression k, std::string alpha,
                           std::string A, std::string C)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_syrk, {C},
               {alpha, A, C}, type),
      uplo_(uplo),
      trans_(trans),
      n_(n),
      k_(k) {}

BLASTriangular BLASNodeSyrk::uplo() const { return this->uplo_; }

BLASTranspose BLASNodeSyrk::trans() const { return this->trans_; }

symbolic::Expression BLASNodeSyrk::n() const { return this->n_; }

symbolic::Expression BLASNodeSyrk::k() const { return this->k_; }

std::string BLASNodeSyrk::alpha() const { return this->input(0); }

std::string BLASNodeSyrk::A() const { return this->input(1); }

std::string BLASNodeSyrk::C() const { return this->input(2); }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeSyrk::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeSyrk>(element_id, this->debug_info(), vertex, parent,
                                          this->type(), this->uplo(), this->trans(), this->n(),
                                          this->k(), this->alpha(), this->A(), this->C());
}

std::string BLASNodeSyrk::toStr() const {
    std::stringstream stream;

    const std::string n = this->n()->__str__();
    const std::string k = this->k()->__str__();

    stream << blasType2String(this->type()) << "syrk(" << blasTriangular2String(this->uplo())
           << ", " << blasTranspose2String(this->trans()) << ", " << n << ", " << k << ", "
           << this->alpha() << ", " << this->A() << ", ";
    if (this->trans() == BLASTranspose_No)
        stream << k;
    else
        stream << n;
    stream << ", 1.0, " << this->C() << ", " << n << ")";

    return stream.str();
}

}  // namespace blas
}  // namespace sdfg
