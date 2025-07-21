#include "sdfg/blas/blas_node_copy.h"

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

BLASNodeCopy::BLASNodeCopy(size_t element_id, const DebugInfo& debug_info,
                           const graph::Vertex vertex, data_flow::DataFlowGraph& parent,
                           const BLASType type, symbolic::Expression n, std::string x,
                           std::string y)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_copy, {y}, {x}, type),
      n_(n) {}

symbolic::Expression BLASNodeCopy::n() const { return this->n_; }

std::string BLASNodeCopy::x() const { return this->input(0); }

std::string BLASNodeCopy::y() const { return this->output(0); }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeCopy::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeCopy>(element_id, this->debug_info(), vertex, parent,
                                          this->type(), this->n(), this->x(), this->y());
}

std::string BLASNodeCopy::toStr() const {
    std::stringstream stream;

    stream << blasType2String(this->type()) << "copy(" << this->n()->__str__() << ", " << this->x()
           << ", 1, " << this->y() << ", 1)";

    return stream.str();
}

}  // namespace blas
}  // namespace sdfg