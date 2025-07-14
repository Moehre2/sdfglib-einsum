#include "sdfg/blas/blas_node.h"

#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/element.h>
#include <sdfg/exceptions.h>
#include <sdfg/graph/graph.h>
#include <sdfg/symbolic/symbolic.h>

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace sdfg {
namespace blas {

BLASNode::BLASNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                   data_flow::DataFlowGraph& parent, const data_flow::LibraryNodeCode& code,
                   const std::vector<std::string>& outputs, const std::vector<std::string>& inputs,
                   const bool side_effect, symbolic::Expression m, symbolic::Expression n,
                   symbolic::Expression k)
    : data_flow::LibraryNode(element_id, debug_info, vertex, parent, code, outputs, inputs,
                             side_effect),
      m_(m),
      n_(n),
      k_(k) {
    if (code.value() != LibraryNodeType_BLAS_gemm.value()) {
        throw InvalidSDFGException("No BLAS library node code: " + code.value());
    }

    if (outputs.size() != 1) {
        throw InvalidSDFGException("BLAS node can only have exactly one output");
    }

    if (inputs.size() != 3) {
        throw InvalidSDFGException("Currently BLAS node can only have exactly three inputs");
    }
}

symbolic::Expression BLASNode::m() const { return this->m_; }

symbolic::Expression BLASNode::n() const { return this->n_; }

symbolic::Expression BLASNode::k() const { return this->k_; }

std::unique_ptr<data_flow::DataFlowNode> BLASNode::clone(size_t element_id,
                                                         const graph::Vertex vertex,
                                                         data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNode>(element_id, this->debug_info(), vertex, parent, this->code(),
                                      this->outputs(), this->inputs(), this->side_effect(),
                                      this->n(), this->m(), this->k());
}

symbolic::SymbolSet BLASNode::symbols() const { return {}; }

void BLASNode::replace(const symbolic::Expression& old_expression,
                       const symbolic::Expression& new_expression) {
    // Do nothing
}

std::string BLASNode::toStr() const {
    if (this->code().value() == LibraryNodeType_BLAS_gemm.value()) {
        std::stringstream stream;

        stream << this->output(0) << " = sgemm('N', 'N', " << this->m()->__str__() << ", "
               << this->n()->__str__() << ", " << this->k()->__str__() << ", 1, " << this->input(0)
               << ", " << this->m()->__str__() << ", " << this->input(1) << ", "
               << this->k()->__str__() << ", 1, " << this->input(2) << ", " << this->m()->__str__()
               << ")";

        return stream.str();
    }
    return this->code().value();
}

}  // namespace blas
}  // namespace sdfg
