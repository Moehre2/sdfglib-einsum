#pragma once

#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/element.h>
#include <sdfg/graph/graph.h>
#include <sdfg/symbolic/symbolic.h>

#include <string>
#include <vector>

namespace sdfg {
namespace blas {

enum BLASType { BLASType_real, BLASType_double };

constexpr const char* blasType2String(const BLASType type) {
    switch (type) {
        case BLASType_real:
            return "s";
        case BLASType_double:
            return "d";
    }
}

class BLASNode : public data_flow::LibraryNode {
    BLASType type_;

   public:
    BLASNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
             data_flow::DataFlowGraph& parent, const data_flow::LibraryNodeCode& code,
             const std::vector<std::string>& outputs, const std::vector<std::string>& inputs,
             const BLASType type);

    BLASNode(const BLASNode&) = delete;
    BLASNode& operator=(const BLASNode&) = delete;

    virtual ~BLASNode() = default;

    BLASType type() const;

    virtual symbolic::SymbolSet symbols() const override;

    virtual void replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) override;
};

}  // namespace blas
}  // namespace sdfg
