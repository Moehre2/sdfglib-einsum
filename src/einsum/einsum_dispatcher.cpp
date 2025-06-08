#include "sdfg/einsum/einsum_dispatcher.h"

namespace sdfg {
namespace einsum {

EinsumDispatcher::EinsumDispatcher(LanguageExtension& language_extension, const Function& function,
                          const data_flow::DataFlowGraph& data_flow_graph,
                          const data_flow::LibraryNode& node)
        : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void EinsumDispatcher::dispatch(PrettyPrinter& stream) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Connector declarations
    for (auto& iedge : this->data_flow_graph_.in_edges(libnode)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        const types::IType& src_type = this->function_.type(src.data());

        auto& conn_name = iedge.dst_conn();
        auto& conn_type = types::infer_type(function_, src_type, iedge.subset());

        stream << this->language_extension_.declaration(conn_name, conn_type);

        stream << " = " << src.data()
               << this->language_extension_.subset(function_, src_type, iedge.subset()) << ";"
               << std::endl;
    }
    for (auto& oedge : this->data_flow_graph_.out_edges(libnode)) {
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
        const types::IType& dst_type = this->function_.type(dst.data());

        auto& conn_name = oedge.src_conn();
        auto& conn_type = types::infer_type(function_, dst_type, oedge.subset());
        stream << this->language_extension_.declaration(conn_name, conn_type) << ";" << std::endl;
    }

    stream << std::endl;
    
    // TODO: Einsum code goes here

    stream << std::endl;

    // Write back
    for (auto& oedge : this->data_flow_graph_.out_edges(libnode)) {
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
        const types::IType& type = this->function_.type(dst.data());
        stream << dst.data() << this->language_extension_.subset(function_, type, oedge.subset())
               << " = ";
        stream << oedge.src_conn();
        stream << ";" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

}
}