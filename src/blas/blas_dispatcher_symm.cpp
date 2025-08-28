#include "sdfg/blas/blas_dispatcher_symm.h"

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

#include "sdfg/blas/blas_node_symm.h"

namespace sdfg {
namespace blas {

BLASDispatcherSymm::BLASDispatcherSymm(codegen::LanguageExtension& language_extension,
                                       const Function& function,
                                       const data_flow::DataFlowGraph& data_flow_graph,
                                       const data_flow::LibraryNode& node)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void BLASDispatcherSymm::dispatch(codegen::PrettyPrinter& stream) {
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

    auto& blas_node = dynamic_cast<const BLASNodeSymm&>(this->node_);

    const std::string m = blas_node.m()->__str__();
    const std::string n = blas_node.n()->__str__();

    stream << "cblas_" << blasType2String(blas_node.type()) << "symm(CblasRowMajor, ";
    switch (blas_node.side()) {
        case BLASSide_Left:
            stream << "CblasLeft";
            break;
        case BLASSide_Right:
            stream << "CblasRight";
            break;
    }
    stream << ", ";
    switch (blas_node.uplo()) {
        case BLASTriangular_Upper:
            stream << "CblasUpper";
            break;
        case BLASTriangular_Lower:
            stream << "CblasLower";
            break;
    }
    stream << ", " << m << ", " << n << ", " << blas_node.alpha() << ", " << blas_node.A() << ", ";
    if (blas_node.side() == BLASSide_Left)
        stream << m;
    else
        stream << n;
    stream << ", " << blas_node.B() << ", " << m << ", 1.0";
    if (blas_node.type() == BLASType_real) stream << "f";
    stream << ", " << blas_node.C() << ", " << m << ");" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

}  // namespace blas
}  // namespace sdfg