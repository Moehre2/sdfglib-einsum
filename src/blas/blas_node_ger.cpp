#include "sdfg/blas/blas_node_ger.h"

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

BLASNodeGer::BLASNodeGer(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                         data_flow::DataFlowGraph& parent, const BLASType type,
                         symbolic::Expression m, symbolic::Expression n, std::string alpha,
                         std::string x, std::string y, std::string A)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_ger, {A},
               {alpha, x, y, A}, type),
      m_(m),
      n_(n) {}

symbolic::Expression BLASNodeGer::m() const { return this->m_; }

symbolic::Expression BLASNodeGer::n() const { return this->n_; }

std::string BLASNodeGer::alpha() const { return this->input(0); }

std::string BLASNodeGer::x() const { return this->input(1); }

std::string BLASNodeGer::y() const { return this->input(2); }

std::string BLASNodeGer::A() const { return this->input(3); }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeGer::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeGer>(element_id, this->debug_info(), vertex, parent,
                                         this->type(), this->m(), this->n(), this->alpha(),
                                         this->x(), this->y(), this->A());
}

std::string BLASNodeGer::toStr() const {
    std::stringstream stream;

    stream << blasType2String(this->type()) << "ger(" << this->m()->__str__() << ", "
           << this->n()->__str__() << ", " << this->alpha() << ", " << this->x() << ", 1, "
           << this->y() << ", 1, " << this->A() << ", " << this->n()->__str__() << ")";

    return stream.str();
}

}  // namespace blas
}  // namespace sdfg