#include "sdfg/blas/blas_node_gemv.h"

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

BLASNodeGemv::BLASNodeGemv(size_t element_id, const DebugInfo& debug_info,
                           const graph::Vertex vertex, data_flow::DataFlowGraph& parent,
                           const BLASType type, BLASTranspose trans, symbolic::Expression m,
                           symbolic::Expression n, std::string alpha, std::string A, std::string x,
                           std::string y)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_gemv, {y},
               {alpha, A, x, y}, type),
      trans_(trans),
      m_(m),
      n_(n) {}

BLASTranspose BLASNodeGemv::trans() const { return this->trans_; }

symbolic::Expression BLASNodeGemv::m() const { return this->m_; }

symbolic::Expression BLASNodeGemv::n() const { return this->n_; }

std::string BLASNodeGemv::alpha() const { return this->input(0); }

std::string BLASNodeGemv::A() const { return this->input(1); }

std::string BLASNodeGemv::x() const { return this->input(2); }

std::string BLASNodeGemv::y() const { return this->input(3); }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeGemv::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeGemv>(element_id, this->debug_info(), vertex, parent,
                                          this->type(), this->trans(), this->m(), this->n(),
                                          this->alpha(), this->A(), this->x(), this->y());
}

std::string BLASNodeGemv::toStr() const {
    std::stringstream stream;

    stream << blasType2String(this->type()) << "gemv(" << blasTranspose2String(this->trans())
           << ", " << this->m()->__str__() << ", " << this->n()->__str__() << ", " << this->alpha()
           << ", " << this->A() << ", " << this->m()->__str__() << ", " << this->x() << ", 1, 1.0, "
           << this->y() << ", 1)";

    return stream.str();
}

}  // namespace blas
}  // namespace sdfg