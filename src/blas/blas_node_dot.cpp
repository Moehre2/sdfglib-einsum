#include "sdfg/blas/blas_node_dot.h"

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

BLASNodeDot::BLASNodeDot(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                         data_flow::DataFlowGraph& parent, std::string result, const BLASType type,
                         symbolic::Expression n, std::string x, std::string y)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_dot, {result},
               {x, y, result}, type),
      n_(n) {}

std::string BLASNodeDot::result() const { return this->output(0); }

symbolic::Expression BLASNodeDot::n() const { return this->n_; }

std::string BLASNodeDot::x() const { return this->input(0); }

std::string BLASNodeDot::y() const { return this->input(1); }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeDot::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeDot>(element_id, this->debug_info(), vertex, parent,
                                         this->result(), this->type(), this->n(), this->x(),
                                         this->y());
}

std::string BLASNodeDot::toStr() const {
    std::stringstream stream;

    stream << this->result() << " = " << this->result() << " + " << blasType2String(this->type())
           << "dot(" << this->n()->__str__() << ", " << this->x() << ", 1, " << this->y() << ", 1)";

    return stream.str();
}

}  // namespace blas
}  // namespace sdfg