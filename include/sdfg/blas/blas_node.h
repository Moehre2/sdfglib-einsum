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

enum BLASTranspose { BLASTranspose_No, BLASTranspose_Transpose };

constexpr const char* blasTranspose2String(const BLASTranspose transpose) {
    switch (transpose) {
        case BLASTranspose_No:
            return "'N'";
        case BLASTranspose_Transpose:
            return "'T'";
    }
}

enum BLASTriangular { BLASTriangular_Upper, BLASTriangular_Lower };

constexpr const char* blasTriangular2String(const BLASTriangular triangular) {
    switch (triangular) {
        case BLASTriangular_Upper:
            return "'U'";
        case BLASTriangular_Lower:
            return "'L'";
    }
}

enum BLASSide { BLASSide_Left, BLASSide_Right };

constexpr const char* blasSide2String(const BLASSide side) {
    switch (side) {
        case BLASSide_Left:
            return "'L'";
        case BLASSide_Right:
            return "'R'";
    }
}

enum BLASImplementation { BLASImplementation_CBLAS, BLASImplementation_CUBLAS };

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

    virtual void validate() const override;

    virtual void replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) override;
};

}  // namespace blas
}  // namespace sdfg
