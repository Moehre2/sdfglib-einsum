#include "sdfg/blas/blas_dispatcher_gemm.h"

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/codegen/language_extension.h>
#include <sdfg/codegen/utils.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/function.h>
#include <sdfg/types/type.h>
#include <sdfg/types/utils.h>

#include "sdfg/blas/blas_node_gemm.h"

namespace sdfg {
namespace blas {

BLASDispatcherGemm::BLASDispatcherGemm(codegen::LanguageExtension& language_extension,
                                       const Function& function,
                                       const data_flow::DataFlowGraph& data_flow_graph,
                                       const data_flow::LibraryNode& node)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void BLASDispatcherGemm::dispatch(codegen::PrettyPrinter& stream) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& blas_node = dynamic_cast<const BLASNodeGemm&>(this->node_);

    // Input connector declarations
    for (auto& iedge : this->data_flow_graph_.in_edges(this->node_)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        const types::IType& src_type = this->function_.type(src.data());

        auto& conn_name = iedge.dst_conn();
        auto& conn_type = types::infer_type(this->function_, src_type, iedge.subset());

        stream << this->language_extension_.declaration(conn_name, conn_type) << " = " << src.data()
               << this->language_extension_.subset(this->function_, src_type, iedge.subset()) << ";"
               << std::endl;
    }

    stream << std::endl;

    // sgemm
    stream << "cblas_" << blasType2String(blas_node.type())
           << "gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, " << blas_node.m()->__str__() << ", "
           << blas_node.n()->__str__() << ", " << blas_node.k()->__str__() << ", 1.0f, "
           << blas_node.input(0) << ", " << blas_node.m()->__str__() << ", " << blas_node.input(1)
           << ", " << blas_node.k()->__str__() << ", 1.0f, " << blas_node.input(2) << ", "
           << blas_node.m()->__str__() << ");" << std::endl;

    stream << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

}  // namespace blas
}  // namespace sdfg