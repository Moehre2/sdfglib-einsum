#include "sdfg/blas/blas_node_gemm.h"

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

#include "sdfg/blas/blas_node.h"

namespace sdfg {
namespace blas {

BLASNodeGemm::BLASNodeGemm(size_t element_id, const DebugInfo& debug_info,
                           const graph::Vertex vertex, data_flow::DataFlowGraph& parent,
                           const std::vector<std::string>& outputs,
                           const std::vector<std::string>& inputs, const BLASType type,
                           symbolic::Expression m, symbolic::Expression n, symbolic::Expression k)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_gemm, outputs, inputs,
               type),
      m_(m),
      n_(n),
      k_(k) {
    if (inputs.size() != 3) {
        throw InvalidSDFGException("Currently BLAS node can only have exactly three inputs");
    }
}

symbolic::Expression BLASNodeGemm::m() const { return this->m_; }

symbolic::Expression BLASNodeGemm::n() const { return this->n_; }

symbolic::Expression BLASNodeGemm::k() const { return this->k_; }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeGemm::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeGemm>(element_id, this->debug_info(), vertex, parent,
                                          this->outputs(), this->inputs(), this->type(), this->n(),
                                          this->m(), this->k());
}

std::string BLASNodeGemm::toStr() const {
    if (this->code().value() == LibraryNodeType_BLAS_gemm.value()) {
        std::stringstream stream;

        stream << this->output(0) << " = " << blasType2String(this->type()) << "gemm('N', 'N', "
               << this->m()->__str__() << ", " << this->n()->__str__() << ", "
               << this->k()->__str__() << ", 1, " << this->input(0) << ", " << this->m()->__str__()
               << ", " << this->input(1) << ", " << this->k()->__str__() << ", 1, "
               << this->input(2) << ", " << this->m()->__str__() << ")";

        return stream.str();
    }
    return this->code().value();
}

}  // namespace blas
}  // namespace sdfg
