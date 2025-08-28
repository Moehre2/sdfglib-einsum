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
                           const BLASType type, BLASTranspose transA, BLASTranspose transB,
                           symbolic::Expression m, symbolic::Expression n, symbolic::Expression k,
                           std::string alpha, std::string A, std::string B, std::string C)
    : BLASNode(element_id, debug_info, vertex, parent, LibraryNodeType_BLAS_gemm, {C},
               {alpha, A, B, C}, type),
      transA_(transA),
      transB_(transB),
      m_(m),
      n_(n),
      k_(k) {}

BLASTranspose BLASNodeGemm::transA() const { return this->transA_; }

BLASTranspose BLASNodeGemm::transB() const { return this->transB_; }

symbolic::Expression BLASNodeGemm::m() const { return this->m_; }

symbolic::Expression BLASNodeGemm::n() const { return this->n_; }

symbolic::Expression BLASNodeGemm::k() const { return this->k_; }

std::string BLASNodeGemm::alpha() const { return this->input(0); }

std::string BLASNodeGemm::A() const { return this->input(1); }

std::string BLASNodeGemm::B() const { return this->input(2); }

std::string BLASNodeGemm::C() const { return this->input(3); }

std::unique_ptr<data_flow::DataFlowNode> BLASNodeGemm::clone(
    size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<BLASNodeGemm>(element_id, this->debug_info(), vertex, parent,
                                          this->type(), this->transA(), this->transB(), this->m(),
                                          this->n(), this->k(), this->alpha(), this->A(), this->B(),
                                          this->C());
}

std::string BLASNodeGemm::toStr() const {
    std::stringstream stream;

    stream << blasType2String(this->type()) << "gemm(" << blasTranspose2String(this->transA())
           << ", " << blasTranspose2String(this->transB()) << ", " << this->m()->__str__() << ", "
           << this->n()->__str__() << ", " << this->k()->__str__() << ", " << this->alpha() << ", "
           << this->A() << ", ";
    if (this->transA() == BLASTranspose_No)
        stream << this->k()->__str__();
    else
        stream << this->m()->__str__();
    stream << ", " << this->B() << ", ";
    if (this->transB() == BLASTranspose_No)
        stream << this->n()->__str__();
    else
        stream << this->k()->__str__();
    stream << ", 1.0, " << this->C() << ", " << this->n()->__str__() << ")";

    return stream.str();
}

}  // namespace blas
}  // namespace sdfg
