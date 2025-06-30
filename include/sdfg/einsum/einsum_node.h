#pragma once

#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/symbolic/symbolic.h>

#include <string>
#include <utility>
#include <vector>

namespace sdfg {
namespace einsum {

inline data_flow::LibraryNodeCode LibraryNodeType_Einsum("Einsum");

/**
 * @brief Einsum node
 *
 * This node enables the use of Einstein summation notation as library nodes in
 * the SDFG.
 */
class EinsumNode : public data_flow::LibraryNode {
   private:
    std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> maps_;

    data_flow::Subset out_indices_;
    std::vector<data_flow::Subset> in_indices_;

   public:
    EinsumNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
               data_flow::DataFlowGraph& parent, const data_flow::LibraryNodeCode& code,
               const std::vector<std::string>& outputs, const std::vector<std::string>& inputs,
               const bool side_effect,
               const std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>& maps,
               const data_flow::Subset& out_indices,
               const std::vector<data_flow::Subset>& in_indices);

    EinsumNode(const EinsumNode&) = delete;
    EinsumNode& operator=(const EinsumNode&) = delete;

    virtual ~EinsumNode() = default;

    const std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>& maps() const;

    const std::pair<symbolic::Symbol, symbolic::Expression>& map(size_t index) const;

    const symbolic::Symbol& indvar(size_t index) const;

    const symbolic::Expression& num_iteration(size_t index) const;

    const data_flow::Subset& out_indices() const;

    const symbolic::Expression& out_index(size_t index) const;

    const std::vector<data_flow::Subset>& in_indices() const;

    const data_flow::Subset& in_indices(size_t index) const;

    const symbolic::Expression& in_index(size_t index1, size_t index2) const;

    virtual std::unique_ptr<data_flow::DataFlowNode> clone(
        size_t element_id, const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent) const override;

    virtual symbolic::SymbolSet symbols() const override;

    virtual void replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) override;

    virtual std::string toStr() const override;
};

}  // namespace einsum
}  // namespace sdfg
