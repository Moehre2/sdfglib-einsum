#include "sdfg/blas/blas_dispatcher_syrk.h"

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/codegen/language_extension.h>
#include <sdfg/codegen/utils.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/function.h>
#include <sdfg/types/type.h>
#include <sdfg/types/utils.h>

#include <string>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_syrk.h"

namespace sdfg {
namespace blas {

BLASDispatcherSyrk::BLASDispatcherSyrk(codegen::LanguageExtension& language_extension,
                                       const Function& function,
                                       const data_flow::DataFlowGraph& data_flow_graph,
                                       const data_flow::LibraryNode& node)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void BLASDispatcherSyrk::dispatch(codegen::PrettyPrinter& stream) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

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

    auto& blas_node = dynamic_cast<const BLASNodeSyrk&>(this->node_);

    const std::string n = blas_node.n()->__str__();
    const std::string k = blas_node.k()->__str__();

    stream << "cblas_" << blasType2String(blas_node.type()) << "syrk(CblasRowMajor, ";
    switch (blas_node.uplo()) {
        case BLASTriangular_Upper:
            stream << "CblasUpper";
            break;
        case BLASTriangular_Lower:
            stream << "CblasLower";
            break;
    }
    stream << ", ";
    switch (blas_node.trans()) {
        case BLASTranspose_No:
            stream << "CblasNoTrans";
            break;
        case BLASTranspose_Transpose:
            stream << "CblasTrans";
            break;
    }
    stream << ", " << n << ", " << k << ", " << blas_node.alpha() << ", " << blas_node.A() << ", ";
    if (blas_node.trans() == BLASTranspose_No)
        stream << k;
    else
        stream << n;
    stream << ", 1.0";
    if (blas_node.type() == BLASType_real) stream << "f";
    stream << ", " << blas_node.C() << ", " << n << ");" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

}  // namespace blas
}  // namespace sdfg