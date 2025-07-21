#include "sdfg/blas/blas_node_scal.h"

#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/element.h>
#include <sdfg/exceptions.h>
#include <sdfg/graph/graph.h>
#include <sdfg/symbolic/symbolic.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "sdfg/blas/blas_node.h"

namespace sdfg {
namespace blas {

BLASNodeScal::BLASNodeScal(size_t element_id, const DebugInfo& debug_info,
                           const graph::Vertex vertex, data_flow::DataFlowGraph& parent,
                           const BLASType type, symbolic::Expression n, std::string alpha,
                           std::string x)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_scal, {x}, {alpha, x},
               type),
      n_(n) {}

symbolic::Expression BLASNodeScal::n() const { return this->n_; }

std::string BLASNodeScal::alpha() const { return this->input(0); }

std::string BLASNodeScal::x() const { return this->input(1); }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeScal::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeScal>(element_id, this->debug_info(), vertex, parent,
                                          this->type(), this->n(), this->alpha(), this->x());
}

std::string BLASNodeScal::toStr() const {
    std::stringstream stream;

    stream << this->output(0) << " = " << blasType2String(this->type()) << "scal("
           << this->n()->__str__() << ", " << this->alpha() << ", " << this->x() << ", 1)";

    return stream.str();
}

}  // namespace blas
}  // namespace sdfg