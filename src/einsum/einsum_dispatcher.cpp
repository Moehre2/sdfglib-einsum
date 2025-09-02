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

#include <cstddef>
#include <iostream>
#include <list>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace einsum {

std::vector<size_t> EinsumDispatcher::get_outer_maps(const EinsumNode& einsum_node) {
    std::vector<size_t> result;
    std::set<size_t> used;

    for (auto& out_index : einsum_node.out_indices()) {
        for (size_t i = 0; i < einsum_node.maps().size(); ++i) {
            if (used.contains(i) || !symbolic::uses(out_index, einsum_node.indvar(i))) continue;

            used.insert(i);

            std::list<symbolic::Expression> bounds = {einsum_node.num_iteration(i)};
            std::vector<size_t> deps;
            while (!bounds.empty()) {
                symbolic::Expression current_bound = bounds.front();
                bounds.pop_front();

                for (size_t j = 0; j < einsum_node.maps().size(); ++j) {
                    if (used.contains(j) || !symbolic::uses(current_bound, einsum_node.indvar(j)))
                        continue;

                    used.insert(j);
                    deps.push_back(j);
                    bounds.push_back(einsum_node.num_iteration(j));
                }
            }

            result.insert(result.end(), deps.rbegin(), deps.rend());
            result.push_back(i);
        }
    }

    return result;
}

std::vector<size_t> EinsumDispatcher::get_inner_maps(const EinsumNode& einsum_node) {
    std::vector<size_t> result, outer_maps = this->get_outer_maps(einsum_node);
    std::set<size_t> used(outer_maps.begin(), outer_maps.end());

    for (size_t i = 0; i < einsum_node.maps().size(); ++i) {
        if (used.contains(i)) continue;

        used.insert(i);

        std::list<symbolic::Expression> bounds = {einsum_node.num_iteration(i)};
        std::vector<size_t> deps;
        while (!bounds.empty()) {
            symbolic::Expression current_bound = bounds.front();
            bounds.pop_front();

            for (size_t j = 0; j < einsum_node.maps().size(); ++j) {
                if (used.contains(j) || !symbolic::uses(current_bound, einsum_node.indvar(j)))
                    continue;

                used.insert(j);
                deps.push_back(j);
                bounds.push_back(einsum_node.num_iteration(j));
            }
        }

        result.insert(result.end(), deps.rbegin(), deps.rend());
        result.push_back(i);
    }

    return result;
}

EinsumDispatcher::EinsumDispatcher(codegen::LanguageExtension& language_extension,
                                   const Function& function,
                                   const data_flow::DataFlowGraph& data_flow_graph,
                                   const data_flow::LibraryNode& node)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void EinsumDispatcher::dispatch(codegen::PrettyPrinter& stream) {
    stream << "// Einsum Node" << std::endl;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    const EinsumNode* einsum_node = dynamic_cast<const EinsumNode*>(&this->node_);

    std::unordered_map<std::string, const types::IType&> src_types;

    // Get output container
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

    long long oii = einsum_node->getOutInputIndex();

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

    // Get outer maps
    std::vector<size_t> outer_maps = this->get_outer_maps(*einsum_node);
    size_t num_outer_maps = outer_maps.size();

    // Parallelize loops if possible
    bool reduction = false;
    size_t outer_collapse = 0;
    for (size_t i = 0; i < num_outer_maps; ++i) {
        size_t j;
        for (j = 0; j < i; ++j) {
            if (symbolic::uses(einsum_node->num_iteration(outer_maps[i]),
                               einsum_node->indvar(outer_maps[j])))
                break;
        }
        if (j != i) break;
        ++outer_collapse;
        if (oii >= 0 && !reduction) {
            bool indvar_in_out_indices = false;
            for (auto& index : einsum_node->out_indices()) {
                if (symbolic::eq(index, einsum_node->indvar(outer_maps[i]))) {
                    indvar_in_out_indices = true;
                    break;
                }
            }
            if (!indvar_in_out_indices) reduction = true;
        }
    }
    if (outer_collapse > 0) {
        stream << "#pragma omp parallel for private(";
        for (size_t i = 0; i < einsum_node->maps().size(); ++i) {
            if (i > 0) stream << ", ";
            stream << einsum_node->indvar(i)->__str__();
        }
        stream << ")";
        if (outer_collapse > 1) stream << " collapse(" << outer_collapse << ")";
        if (reduction)
            stream << " reduction(+:" << output_container
                   << this->language_extension_.subset(this->function_, dst_type,
                                                       einsum_node->out_indices())
                   << ")";
        stream << std::endl;
    }

    // Create outer maps as for loops
    for (size_t outer_map : outer_maps) {
        const std::string indvar = einsum_node->indvar(outer_map)->__str__();
        stream << "for (" << indvar << " = 0; " << indvar << " < "
               << this->language_extension_.expression(einsum_node->num_iteration(outer_map))
               << "; " << indvar << "++)" << std::endl
               << "{" << std::endl;
        stream.setIndent(stream.indent() + 4);
    }

    // Set output connector to previous value / to zero
    if (dynamic_cast<const types::Pointer*>(&conn_type))
        stream << this->language_extension_.declaration(conn_name,
                                                        types::Scalar(conn_type.primitive_type()));
    else
        stream << this->language_extension_.declaration(conn_name, conn_type);
    if (oii >= 0)
        stream << " = " << output_container
               << this->language_extension_.subset(this->function_, dst_type,
                                                   einsum_node->out_indices());
    stream << ";" << std::endl;

    stream << std::endl;

    // Get inner maps
    std::vector<size_t> inner_maps = this->get_inner_maps(*einsum_node);
    size_t num_inner_maps = inner_maps.size();

    // Parallelize loops if possible
    if (num_outer_maps == 0) {
        size_t inner_collapse = 0;
        for (size_t i = 0; i < num_inner_maps; ++i) {
            size_t j;
            for (j = 0; j < i; ++j) {
                if (symbolic::uses(einsum_node->num_iteration(inner_maps[i]),
                                   einsum_node->indvar(inner_maps[j])))
                    break;
            }
            if (j != i) break;
            ++inner_collapse;
        }
        if (inner_collapse > 0) {
            stream << "#pragma omp parallel for private(";
            for (size_t i = 0; i < num_inner_maps; ++i) {
                if (i > 0) stream << ", ";
                stream << einsum_node->indvar(inner_maps[i])->__str__();
            }
            stream << ")";
            if (inner_collapse > 1) stream << " collapse(" << inner_collapse << ")";
            stream << " reduction(+:" << einsum_node->output(0) << ")" << std::endl;
        }
    }

    // Create inner maps as for loops
    for (size_t inner_map : inner_maps) {
        const std::string indvar = einsum_node->indvar(inner_map)->__str__();
        stream << "for (" << indvar << " = 0; " << indvar << " < "
               << this->language_extension_.expression(einsum_node->num_iteration(inner_map))
               << "; " << indvar << "++)" << std::endl
               << "{" << std::endl;
        stream.setIndent(stream.indent() + 4);
    }

    // Calculate one entry
    stream << einsum_node->output(0) << " = ";
    if (oii >= 0) stream << einsum_node->output(0) << " + ";
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
            if (src_types.contains(einsum_node->input(i)) &&
                dynamic_cast<const types::Pointer*>(&src_types.at(einsum_node->input(i))))
                stream << "*";
            stream << einsum_node->input(i);
        }
    }
    stream << ";" << std::endl;

    // Closing brackets for inner maps
    for (size_t i = 0; i < num_inner_maps; ++i) {
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
    for (size_t i = 0; i < num_outer_maps; ++i) {
        stream.setIndent(stream.indent() - 4);
        stream << "}" << std::endl;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

}  // namespace einsum
}  // namespace sdfg