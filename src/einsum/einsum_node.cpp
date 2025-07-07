#include "sdfg/einsum/einsum_node.h"

#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/element.h>
#include <sdfg/exceptions.h>
#include <sdfg/graph/graph.h>
#include <sdfg/symbolic/symbolic.h>

#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sdfg {
namespace einsum {

EinsumNode::EinsumNode(size_t element_id, const DebugInfo& debug_info, const graph::Vertex vertex,
                       data_flow::DataFlowGraph& parent, const data_flow::LibraryNodeCode& code,
                       const std::vector<std::string>& outputs,
                       const std::vector<std::string>& inputs, const bool side_effect,
                       const std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>& maps,
                       const data_flow::Subset& out_indices,
                       const std::vector<data_flow::Subset>& in_indices)
    : data_flow::LibraryNode(element_id, debug_info, vertex, parent, code, outputs, inputs,
                             side_effect),
      maps_(maps),
      out_indices_(out_indices),
      in_indices_(in_indices) {
    // Check number of outputs
    if (outputs.size() != 1) {
        throw InvalidSDFGException("Einsum node can only have exactly one output");
    }

    // Check list sizes
    if (inputs.size() != in_indices.size()) {
        throw InvalidSDFGException("Number of input containers != number of input indices");
    }

    // Check if map indices occur at least once as in/out indices
    for (auto& map : maps) {
        bool unused = true;
        for (auto& index : out_indices) {
            if (map.first->__str__() == index->__str__()) {
                unused = false;
                break;
            }
        }
        for (auto& indices : in_indices) {
            for (auto& index : indices) {
                if (map.first->__str__() == index->__str__()) {
                    unused = false;
                    break;
                }
            }
        }
        if (unused) {
            throw InvalidSDFGException("Einsum indvar " + map.first->__str__() +
                                       " does not occur at least once in in/out indices.");
        }
    }

    size_t i;
    for (i = 0; i < inputs.size(); ++i) {
        if (inputs[i] == outputs[0]) break;
    }
    if (i < inputs.size()) {
        if (in_indices[i].size() != out_indices.size()) {
            throw InvalidSDFGException("Out input and output do not have the same indices");
        }
        for (size_t j = 0; j < out_indices.size(); ++j) {
            if (!symbolic::eq(in_indices[i][j], out_indices[j])) {
                throw InvalidSDFGException("Out input and output do not have the same indices");
            }
        }
    }

    // TODO: Check if container exist and types match einsum index access
    // The Problem: For a types::infer_type, I need a sdfg::Function which I am unable to get at
    //              this point
}

const std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>& EinsumNode::maps() const {
    return this->maps_;
}

const std::pair<symbolic::Symbol, symbolic::Expression>& EinsumNode::map(size_t index) const {
    return this->maps_[index];
}

const symbolic::Symbol& EinsumNode::indvar(size_t index) const { return this->maps_[index].first; }

const symbolic::Expression& EinsumNode::num_iteration(size_t index) const {
    return this->maps_[index].second;
}

const data_flow::Subset& EinsumNode::out_indices() const { return this->out_indices_; }

const symbolic::Expression& EinsumNode::out_index(size_t index) const {
    return this->out_indices_[index];
}

const std::vector<data_flow::Subset>& EinsumNode::in_indices() const { return this->in_indices_; }

const data_flow::Subset& EinsumNode::in_indices(size_t index) const {
    return this->in_indices_[index];
}

const symbolic::Expression& EinsumNode::in_index(size_t index1, size_t index2) const {
    return this->in_indices_[index1][index2];
}

std::unique_ptr<data_flow::DataFlowNode> EinsumNode::clone(size_t element_id,
                                                           const graph::Vertex vertex,
                                                           data_flow::DataFlowGraph& parent) const {
    return std::make_unique<EinsumNode>(
        element_id, this->debug_info(), vertex, parent, this->code(), this->outputs(),
        this->inputs(), this->side_effect(), this->maps(), this->out_indices(), this->in_indices());
}

symbolic::SymbolSet EinsumNode::symbols() const {
    // TODO: Implement
    return {};
}

void EinsumNode::replace(const symbolic::Expression& old_expression,
                         const symbolic::Expression& new_expression) {
    // Do nothing
}

std::string EinsumNode::toStr() const {
    std::stringstream stream;

    stream << this->outputs_[0];
    if (this->out_indices_.size() > 0) {
        stream << "[";
        for (size_t i = 0; i < this->out_indices_.size(); ++i) {
            if (i > 0) stream << ",";
            stream << this->out_indices_[i]->__str__();
        }
        stream << "]";
    }
    stream << " = ";
    long long oii = this->getOutInputIndex();
    if (oii >= 0) {
        stream << this->inputs_[oii];
        if (this->in_indices_[oii].size() > 0) {
            stream << "[";
            for (size_t i = 0; i < this->in_indices_[oii].size(); ++i) {
                if (i > 0) stream << ",";
                stream << this->in_indices_[oii][i]->__str__();
            }
            stream << "]";
        }
        stream << " + ";
    }
    bool first_mul = false;
    for (size_t i = 0; i < this->inputs_.size(); ++i) {
        if (this->inputs_[i] == this->outputs_[0]) continue;
        if (first_mul) stream << " * ";
        first_mul = true;
        stream << this->inputs_[i];
        if (this->in_indices_[i].size() > 0) {
            stream << "[";
            for (size_t j = 0; j < this->in_indices_[i].size(); j++) {
                if (j > 0) stream << ",";
                stream << this->in_indices_[i][j]->__str__();
            }
            stream << "]";
        }
    }

    for (auto& map : this->maps_) {
        stream << " for " << map.first->__str__() << " = 0:" << map.second->__str__();
    }

    return stream.str();
}

long long EinsumNode::getOutInputIndex() const {
    for (size_t i = 0; i < this->inputs_.size(); ++i) {
        if (this->inputs_[i] == this->outputs_[0]) return i;
    }
    return -1;
}

}  // namespace einsum
}  // namespace sdfg