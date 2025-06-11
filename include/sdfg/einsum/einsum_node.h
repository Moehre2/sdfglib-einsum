#pragma once

#include <sdfg/data_flow/library_node.h>

namespace sdfg {
namespace einsum {

inline constexpr data_flow::LibraryNodeCode LibraryNodeType_Einsum("Einsum");

/**
 * @brief Einsum node
 *
 * This node enables the use of Einstein summation notation as library nodes in
 * the SDFG.
 */
class EinsumNode : public data_flow::LibraryNode {
   private:
    // TODO: Add private members

   public:
    EinsumNode(const DebugInfo& debug_info, const graph::Vertex vertex,
               data_flow::DataFlowGraph& parent, const data_flow::LibraryNodeCode& code,
               const std::vector<std::string>& outputs, const std::vector<std::string>& inputs,
               const bool side_effect);

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    virtual void replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) override;
};

}  // namespace einsum
}  // namespace sdfg
