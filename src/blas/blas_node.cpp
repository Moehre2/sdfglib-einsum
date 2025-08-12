#include "sdfg/blas/blas_node.h"

#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/element.h>
#include <sdfg/exceptions.h>
#include <sdfg/graph/graph.h>
#include <sdfg/symbolic/symbolic.h>

#include <string>
#include <vector>

namespace sdfg {
namespace blas {

BLASNode::BLASNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                   data_flow::DataFlowGraph& parent, const data_flow::LibraryNodeCode& code,
                   const std::vector<std::string>& outputs, const std::vector<std::string>& inputs,
                   const BLASType type)
    : data_flow::LibraryNode(element_id, debug_info, vertex, parent, code, outputs, inputs, false),
      type_(type) {
    if (outputs.size() != 1) {
        throw InvalidSDFGException("BLAS node can only have exactly one output");
    }
}

BLASType BLASNode::type() const { return this->type_; }

symbolic::SymbolSet BLASNode::symbols() const { return {}; }

void BLASNode::validate() const {
    // TODO: Implement
}

void BLASNode::replace(const symbolic::Expression& old_expression,
                       const symbolic::Expression& new_expression) {
    // Do nothing
}

}  // namespace blas
}  // namespace sdfg
