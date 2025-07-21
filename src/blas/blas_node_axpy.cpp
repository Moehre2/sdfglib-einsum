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
                           const std::vector<std::string>& outputs,
                           const std::vector<std::string>& inputs, const BLASType type,
                           symbolic::Expression n)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_axpy, outputs, inputs,
               type),
      n_(n) {
    if (inputs.size() != 3) {
        throw InvalidSDFGException("BLAS axpy can only have exactly three inputs");
    }

    if (inputs[2] != outputs[0]) {
        throw InvalidSDFGException("BLAS axpy 3rd input does not match output");
    }
}

symbolic::Expression BLASNodeAxpy::n() const { return this->n_; }

std::string BLASNodeAxpy::alpha() const { return this->inputs_[0]; }

std::string BLASNodeAxpy::x() const { return this->inputs_[1]; }

std::string BLASNodeAxpy::y() const { return this->inputs_[2]; }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeAxpy::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeAxpy>(element_id, this->debug_info(), vertex, parent,
                                          this->outputs(), this->inputs(), this->type(), this->n());
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