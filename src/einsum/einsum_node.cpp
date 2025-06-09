#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace einsum {

EinsumNode::EinsumNode(const DebugInfo& debug_info, const graph::Vertex vertex,
                       data_flow::DataFlowGraph& parent, const data_flow::LibraryNodeCode& code,
                       const std::vector<std::string>& outputs,
                       const std::vector<std::string>& inputs, const bool side_effect)
    : data_flow::LibraryNode(debug_info, vertex, parent, code, outputs, inputs, side_effect) {}

std::unique_ptr<data_flow::DataFlowNode> EinsumNode::clone(const graph::Vertex vertex,
                                                           data_flow::DataFlowGraph& parent) const {
    return std::make_unique<EinsumNode>(this->debug_info(), vertex, parent, this->code(),
                                        this->outputs(), this->inputs(), this->side_effect());
}

void EinsumNode::replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) {
    // Do nothing
}

}  // namespace einsum
}  // namespace sdfg