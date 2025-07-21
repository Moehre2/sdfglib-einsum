#include "sdfg/blas/blas_node_axpy.h"

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

BLASNodeAxpy::BLASNodeAxpy(size_t element_id, const DebugInfo& debug_info,
                           const graph::Vertex vertex, data_flow::DataFlowGraph& parent,
                           const BLASType type, symbolic::Expression n, std::string alpha,
                           std::string x, std::string y)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_axpy, {y},
               {alpha, x, y}, type),
      n_(n) {}

symbolic::Expression BLASNodeAxpy::n() const { return this->n_; }

std::string BLASNodeAxpy::alpha() const { return this->input(0); }

std::string BLASNodeAxpy::x() const { return this->input(1); }

std::string BLASNodeAxpy::y() const { return this->input(2); }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeAxpy::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeAxpy>(element_id, this->debug_info(), vertex, parent,
                                          this->type(), this->n(), this->alpha(), this->x(),
                                          this->y());
}

std::string BLASNodeAxpy::toStr() const {
    std::stringstream stream;

    stream << this->output(0) << " = " << blasType2String(this->type()) << "axpy("
           << this->n()->__str__() << ", " << this->alpha() << ", " << this->x() << ", 1, "
           << this->y() << ", 1)";

    return stream.str();
}

}  // namespace blas
}  // namespace sdfg