#include "sdfg/einsum/einsum_dispatcher.h"

#include <sdfg/codegen/language_extension.h>
#include <sdfg/codegen/utils.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/function.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/type.h>
#include <sdfg/types/utils.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace einsum {

EinsumDispatcher::EinsumDispatcher(codegen::LanguageExtension& language_extension,
                                   const Function& function,
                                   const data_flow::DataFlowGraph& data_flow_graph,
                                   const data_flow::LibraryNode& node)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void EinsumDispatcher::dispatch(codegen::PrettyPrinter& stream) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    const EinsumNode* einsum_node = dynamic_cast<const EinsumNode*>(&this->node_);

    std::unordered_map<std::string, const types::IType&> src_types;
    size_t outer_maps = 0, inner_maps = 0;

    // Input connector declarations
    for (auto& iedge : this->data_flow_graph_.in_edges(this->node_)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        const types::IType& src_type = this->function_.type(src.data());

        auto& conn_name = iedge.dst_conn();
        if (conn_name == einsum_node->output(0)) continue;
        auto& conn_type = types::infer_type(function_, src_type, iedge.subset());

        src_types.insert({conn_name, src_type});

        stream << this->language_extension_.declaration(conn_name, conn_type);

        stream << " = " << src.data()
               << this->language_extension_.subset(function_, src_type, iedge.subset()) << ";"
               << std::endl;
    }

    stream << std::endl;

    // Create outer maps as for loops
    for (const symbolic::Expression& out_index : einsum_node->out_indices()) {
        for (const std::pair<symbolic::Symbol, symbolic::Expression>& map : einsum_node->maps()) {
            if (map.first->__str__() != out_index->__str__()) continue;
            stream << "for (" << map.first->get_name() << " = 0; " << map.first->get_name() << " < "
                   << this->language_extension_.expression(map.second) << "; "
                   << map.first->get_name() << "++)" << std::endl
                   << "{" << std::endl;
            stream.setIndent(stream.indent() + 4);
            ++outer_maps;
        }
    }

    // Set output connector to previous value / to zero
    auto& oedge = *this->data_flow_graph_.out_edges(this->node_).begin();
    auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
    const types::IType& dst_type = this->function_.type(dst.data());
    auto& conn_name = oedge.src_conn();
    auto& conn_type = types::infer_type(function_, dst_type, einsum_node->out_indices());

    std::string dummy_declaration = this->language_extension_.declaration(dst.data(), conn_type);
    std::string dummy_primitive_type =
        this->language_extension_.primitive_type(conn_type.primitive_type());
    std::string output_container;
    if (dummy_primitive_type.size() + dst.data().size() + 1 >= dummy_declaration.size())
        output_container = dst.data();
    else
        output_container = dummy_declaration.substr(dummy_primitive_type.size() + 1);

    long long iio = einsum_node->getOutInputIndex();

    if (dynamic_cast<const types::Pointer*>(&conn_type))
        stream << this->language_extension_.declaration(conn_name,
                                                        types::Scalar(conn_type.primitive_type()));
    else
        stream << this->language_extension_.declaration(conn_name, conn_type);
    if (iio >= 0)
        stream << " = " << output_container
               << this->language_extension_.subset(this->function_, dst_type,
                                                   einsum_node->out_indices());
    stream << ";" << std::endl;

    stream << std::endl;

    // Create inner maps as for loops
    for (const std::pair<symbolic::Symbol, symbolic::Expression>& map : einsum_node->maps()) {
        bool outer = false;
        for (const symbolic::Expression& out_index : einsum_node->out_indices()) {
            if (map.first->__str__() == out_index->__str__()) {
                outer = true;
                break;
            }
        }
        if (outer) continue;
        stream << "for (" << map.first->get_name() << " = 0; " << map.first->get_name() << " < "
               << this->language_extension_.expression(map.second) << "; " << map.first->get_name()
               << "++)" << std::endl
               << "{" << std::endl;
        stream.setIndent(stream.indent() + 4);
        ++inner_maps;
    }

    // Calculate one entry
    stream << einsum_node->output(0) << " = ";
    if (iio >= 0) stream << einsum_node->output(0) << " + ";
    bool first_mul = false;
    for (size_t i = 0; i < einsum_node->inputs().size(); ++i) {
        if (einsum_node->input(i) == einsum_node->output(0)) continue;
        if (first_mul) stream << " * ";
        first_mul = true;
        if (einsum_node->in_indices(i).size() > 0) {
            stream << einsum_node->input(i);
            stream << this->language_extension_.subset(
                this->function_, src_types.at(einsum_node->input(i)), einsum_node->in_indices(i));
        } else {
            if (dynamic_cast<const types::Pointer*>(&src_types.at(einsum_node->input(i))))
                stream << "*";
            stream << einsum_node->input(i);
        }
    }
    stream << ";" << std::endl;

    // Closing brackets for inner maps
    for (size_t i = 0; i < inner_maps; ++i) {
        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
    }

    stream << std::endl;

    // Write back output connector
    stream << output_container
           << this->language_extension_.subset(this->function_, dst_type,
                                               einsum_node->out_indices())
           << " = " << oedge.src_conn() << ";" << std::endl;

    // Closing brackets for outer maps
    for (size_t i = 0; i < outer_maps; ++i) {
        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

}  // namespace einsum
}  // namespace sdfg