#include "sdfg/transformations/einsum_expand.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/data_dependency_analysis.h>
#include <sdfg/analysis/users.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/deepcopy/structured_sdfg_deep_copy.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/transformations/transformation.h>
#include <symengine/basic.h>

#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

void EinsumExpand::visitElements(std::set<size_t>& elements,
                                 const structured_control_flow::ControlFlowNode& node) {
    elements.insert(node.element_id());
    if (auto block = dynamic_cast<const structured_control_flow::Block*>(&node)) {
        for (auto& node : block->dataflow().nodes()) elements.insert(node.element_id());
    } else if (auto sequence = dynamic_cast<const structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); ++i) {
            this->visitElements(elements, sequence->at(i).first);
            elements.insert(sequence->at(i).second.element_id());
        }
    } else if (auto if_else = dynamic_cast<const structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); ++i)
            this->visitElements(elements, if_else->at(i).first);
    } else if (auto while_loop = dynamic_cast<const structured_control_flow::While*>(&node)) {
        this->visitElements(elements, while_loop->root());
    } else if (auto loop = dynamic_cast<const structured_control_flow::For*>(&node)) {
        this->visitElements(elements, loop->root());
    } else if (auto map_node = dynamic_cast<const structured_control_flow::Map*>(&node)) {
        this->visitElements(elements, map_node->root());
    }
}

bool EinsumExpand::subsetContainsSymbol(const data_flow::Subset& subset,
                                        const symbolic::Symbol& symbol) {
    for (auto& expr : subset) {
        if (symbolic::uses(expr, symbol)) return true;
    }
    return false;
}

EinsumExpand::EinsumExpand(structured_control_flow::StructuredLoop& loop,
                           einsum::EinsumNode& einsum_node)
    : loop_(loop), einsum_node_(einsum_node) {}

std::string EinsumExpand::name() const { return "EinsumExpand"; }

bool EinsumExpand::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                  analysis::AnalysisManager& analysis_manager) {
    // Check that the einsum node is in a block in the loop
    structured_control_flow::Block* block_einsum = nullptr;
    size_t loop_root_index;
    for (size_t i = 0; i < this->loop_.root().size(); ++i) {
        if (auto* block =
                dynamic_cast<structured_control_flow::Block*>(&this->loop_.root().at(i).first)) {
            for (auto& node : block->dataflow().nodes()) {
                if (this->einsum_node_.element_id() == node.element_id()) {
                    block_einsum = block;
                    loop_root_index = i;
                    break;
                }
            }
            if (block_einsum) break;
        }
    }
    if (!block_einsum) return false;

    // Check that the loop is of sufficient form
    if (this->loop_.init()->get_type_code() != SymEngine::TypeID::SYMENGINE_INTEGER) return false;
    if (!symbolic::eq(this->loop_.init(), symbolic::zero())) return false;
    if (this->loop_.condition()->get_type_code() != SymEngine::TypeID::SYMENGINE_STRICTLESSTHAN)
        return false;
    if (this->loop_.condition()->get_args().size() != 2) return false;
    if (!symbolic::eq(this->loop_.condition()->get_args().at(0), this->loop_.indvar()))
        return false;
    if (this->loop_.update()->get_type_code() != SymEngine::TypeID::SYMENGINE_ADD) return false;
    if (this->loop_.update()->get_args().size() != 2) return false;
    if (!symbolic::eq(this->loop_.update()->get_args().at(0), this->loop_.indvar()) &&
        !symbolic::eq(this->loop_.update()->get_args().at(1), this->loop_.indvar()))
        return false;
    if (!symbolic::eq(this->loop_.update()->get_args().at(0), symbolic::one()) &&
        !symbolic::eq(this->loop_.update()->get_args().at(1), symbolic::one()))
        return false;

    // Prevent one of the einsum inputs to be the index variable
    for (auto& iedge : block_einsum->dataflow().in_edges(this->einsum_node_)) {
        if (dynamic_cast<const data_flow::AccessNode&>(iedge.src()).data() ==
            this->loop_.indvar()->__str__())
            return false;
    }

    // Check that loop index does not collide with einsum map indices
    for (auto& map : this->einsum_node_.maps()) {
        if (symbolic::eq(map.first, this->loop_.indvar())) return false;
    }

    // Check that the index variable of the loop occurs without a calculation in the einsum in
    // indices
    for (auto& indices : this->einsum_node_.in_indices()) {
        for (auto& index : indices) {
            if (symbolic::uses(index, this->loop_.indvar()) &&
                !symbolic::eq(index, this->loop_.indvar()))
                return false;
        }
    }

    // Check that the out indices contain the index variable of the loop or ...
    if (!this->subsetContainsSymbol(this->einsum_node_.out_indices(), this->loop_.indvar())) {
        // ... the loops are not "correctly" ordered
        bool in_indices_contain_loop_index = false;
        for (auto& indices : this->einsum_node_.in_indices()) {
            if (this->subsetContainsSymbol(indices, this->loop_.indvar())) {
                in_indices_contain_loop_index = true;
                break;
            }
        }
        if (!in_indices_contain_loop_index) return false;
    }

    // Determine the element ids of all elements inside the loop before and after the einsum node
    auto topo_sort = block_einsum->dataflow().topological_sort();
    std::set<size_t> elements_before_einsum, elements_after_einsum;
    for (size_t i = 0; i < this->loop_.root().size(); ++i) {
        if (i < loop_root_index) {
            this->visitElements(elements_before_einsum, this->loop_.root().at(i).first);
            elements_before_einsum.insert(this->loop_.root().at(i).second.element_id());
        } else if (i > loop_root_index) {
            this->visitElements(elements_after_einsum, this->loop_.root().at(i).first);
            elements_after_einsum.insert(this->loop_.root().at(i).second.element_id());
        }
    }
    bool before_einsum = true;
    for (auto* node : topo_sort) {
        if (dynamic_cast<einsum::EinsumNode*>(node) == &this->einsum_node_) {
            before_einsum = false;
        } else if (before_einsum) {
            elements_before_einsum.insert(node->element_id());
        } else {
            elements_after_einsum.insert(node->element_id());
        }
    }

    // Create a map from each input connector to its input container of the einsum node
    std::unordered_map<std::string, std::string> in_containers;
    for (auto& iedge : block_einsum->dataflow().in_edges(this->einsum_node_)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        in_containers.insert({iedge.dst_conn(), src.data()});
        if (block_einsum->dataflow().in_degree(src) == 0) {
            elements_before_einsum.erase(iedge.element_id());
            elements_before_einsum.erase(src.element_id());
        }
    }

    // Create a map from each output connector to its output container of the einsum node
    std::unordered_map<std::string, std::string> out_containers;
    for (auto& oedge : block_einsum->dataflow().out_edges(this->einsum_node_)) {
        auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
        out_containers.insert({oedge.src_conn(), dst.data()});
        if (block_einsum->dataflow().out_degree(dst) == 0) {
            elements_after_einsum.erase(oedge.element_id());
            elements_after_einsum.erase(dst.element_id());
        }
    }

    // Check if all occurrences of the output container in the inputs have the index variable of the
    // loop in their subset
    // E.g., disallow x[i] += ... * x[j] where i is the index variable of the loop
    for (size_t i = 0; i < this->einsum_node_.inputs().size(); ++i) {
        if (!in_containers.contains(this->einsum_node_.input(i))) continue;
        if (this->einsum_node_.input(i) == this->einsum_node_.output(0)) continue;
        if (in_containers.at(this->einsum_node_.input(i)) !=
            out_containers.at(this->einsum_node_.output(0)))
            continue;
        if (!this->subsetContainsSymbol(this->einsum_node_.in_indices(i), this->loop_.indvar()))
            return false;
    }

    // Perform a user analysis
    auto& users = analysis_manager.get<analysis::Users>();
    users.run(analysis_manager);

    // Check for every input container of the einsum node: If it is write accessed inside the loop
    // before the einsum node, the loop index variable has to be in the input indices
    for (size_t i = 0; i < this->einsum_node_.inputs().size(); ++i) {
        if (!in_containers.contains(this->einsum_node_.input(i))) continue;
        for (auto* user : users.uses(in_containers.at(this->einsum_node_.input(i)))) {
            if (user && user->use() == analysis::Use::WRITE && user->element() &&
                elements_before_einsum.contains(user->element()->element_id()) &&
                !this->subsetContainsSymbol(this->einsum_node_.in_indices(i), this->loop_.indvar()))
                return false;
        }
    }

    // Check for the output container of the einsum node: If it is read accessed inside the loop
    // after the einsum node, the loop index variable has to be in the output indices
    if (out_containers.contains(this->einsum_node_.output(0))) {
        for (auto* user : users.uses(out_containers.at(this->einsum_node_.output(0)))) {
            if (user && user->use() == analysis::Use::READ && user->element() &&
                elements_after_einsum.contains(user->element()->element_id()) &&
                !this->subsetContainsSymbol(this->einsum_node_.out_indices(), this->loop_.indvar()))
                return false;
        }
    }

    // Check dependency on symbols of the einsum node inside the loop
    symbolic::SymbolSet map_indvars;
    for (auto& maps : this->einsum_node_.maps()) map_indvars.insert(maps.first);
    for (auto& sym : this->einsum_node_.symbols()) {
        for (auto* user : users.uses(sym->__str__())) {
            if (user && user->use() == analysis::Use::WRITE && user->element() &&
                elements_before_einsum.contains(user->element()->element_id()) &&
                !map_indvars.contains(sym))
                return false;
            if (user && user->use() == analysis::Use::WRITE && user->element() &&
                elements_after_einsum.contains(user->element()->element_id()) &&
                !map_indvars.contains(sym))
                return false;
        }
    }

    return true;
}

void EinsumExpand::apply(builder::StructuredSDFGBuilder& builder,
                         analysis::AnalysisManager& analysis_manager) {
    // Get the block in which the einsum node lives
    structured_control_flow::Block* block_einsum = nullptr;
    size_t loop_root_index;
    for (size_t i = 0; i < this->loop_.root().size(); ++i) {
        if (auto* block =
                dynamic_cast<structured_control_flow::Block*>(&this->loop_.root().at(i).first)) {
            for (auto& node : block->dataflow().nodes()) {
                if (this->einsum_node_.element_id() == node.element_id()) {
                    block_einsum = block;
                    loop_root_index = i;
                    break;
                }
            }
            if (block_einsum) break;
        }
    }

    // Get index variable and number of iterations from the loop
    symbolic::Symbol indvar = this->loop_.indvar();
    symbolic::Expression num_iterations;
    if (symbolic::eq(this->loop_.condition()->get_args().at(0), indvar))
        num_iterations = this->loop_.condition()->get_args().at(1);
    else
        num_iterations = this->loop_.condition()->get_args().at(0);

    // Detect if there is dataflow in the einsum block before/after the einsum node
    bool dataflow_before_einsum = false;
    bool dataflow_after_einsum = false;
    auto topo_sort = block_einsum->dataflow().topological_sort();
    if (topo_sort.size() >
        this->einsum_node_.outputs().size() + this->einsum_node_.inputs().size() + 1) {
        bool einsum_node_occured = false;
        for (auto* node : block_einsum->dataflow().topological_sort()) {
            if (dynamic_cast<einsum::EinsumNode*>(node) == &this->einsum_node_) {
                einsum_node_occured = true;
            } else if (dynamic_cast<data_flow::Tasklet*>(node) ||
                       dynamic_cast<data_flow::LibraryNode*>(node)) {
                if (einsum_node_occured)
                    dataflow_after_einsum = true;
                else
                    dataflow_before_einsum = true;
            }
        }
    }

    // Get the parent node of the loop
    auto& parent = builder.parent(this->loop_);

    // Add a new block after the loop
    auto& new_block_einsum = builder.add_block_after(parent, this->loop_).first;

    // Copy the access to the einsum node from the old block to the new one
    std::unordered_map<std::string, data_flow::AccessNode&> out_access, in_access;
    for (auto& oedge : block_einsum->dataflow().out_edges(this->einsum_node_)) {
        out_access.insert(
            {oedge.src_conn(),
             builder.add_access(new_block_einsum,
                                dynamic_cast<const data_flow::AccessNode&>(oedge.dst()).data())});
    }
    for (auto& iedge : block_einsum->dataflow().in_edges(this->einsum_node_)) {
        in_access.insert(
            {iedge.dst_conn(),
             builder.add_access(new_block_einsum,
                                dynamic_cast<const data_flow::AccessNode&>(iedge.src()).data())});
    }

    // Add the expanded einsum node to the new block after the loop
    auto new_maps = this->einsum_node_.maps();
    new_maps.push_back({indvar, num_iterations});
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            new_block_einsum, this->einsum_node_.debug_info(), this->einsum_node_.outputs(),
            this->einsum_node_.inputs(), new_maps, this->einsum_node_.out_indices(),
            this->einsum_node_.in_indices());

    // Create the memlets in the new einsum block
    for (size_t i = 0; i < this->einsum_node_.outputs().size(); ++i) {
        if (out_access.contains(this->einsum_node_.output(i)))
            builder.add_memlet(new_block_einsum, libnode, this->einsum_node_.output(i),
                               out_access.at(this->einsum_node_.output(i)), "void", {});
    }
    for (size_t i = 0; i < this->einsum_node_.inputs().size(); ++i) {
        if (in_access.contains(this->einsum_node_.input(i)))
            builder.add_memlet(new_block_einsum, in_access.at(this->einsum_node_.input(i)), "void",
                               libnode, this->einsum_node_.input(i), {});
    }

    // Extract and copy dataflow before the original einsum node
    if (dataflow_before_einsum) {
        auto& new_block_before = builder.add_block_before(this->loop_.root(), *block_einsum).first;
        ++loop_root_index;
        std::unordered_map<data_flow::AccessNode*, data_flow::AccessNode*> access_node_map;
        for (auto* access_node : block_einsum->dataflow().data_nodes()) {
            access_node_map.insert({access_node, nullptr});
        }
        for (auto* node : topo_sort) {
            if (dynamic_cast<einsum::EinsumNode*>(node) == &this->einsum_node_) break;
            if (!dynamic_cast<data_flow::CodeNode*>(node)) continue;
            auto* code_node = dynamic_cast<data_flow::CodeNode*>(node);
            data_flow::CodeNode* new_code_node = nullptr;
            if (auto* tasklet = dynamic_cast<data_flow::Tasklet*>(code_node)) {
                new_code_node =
                    &builder.add_tasklet(new_block_before, tasklet->code(), tasklet->output(),
                                         tasklet->inputs(), tasklet->debug_info());
            } else if (auto* libnode = dynamic_cast<data_flow::LibraryNode*>(node)) {
                new_code_node = &builder.copy_library_node(new_block_before, *libnode);
            }
            for (auto& iedge : block_einsum->dataflow().in_edges(*code_node)) {
                auto* access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src());
                if (!access_node_map.at(access_node))
                    access_node_map.at(access_node) = &builder.add_access(
                        new_block_before, access_node->data(), access_node->debug_info());
                builder.add_memlet(new_block_before, *access_node_map.at(access_node),
                                   iedge.src_conn(), *new_code_node, iedge.dst_conn(),
                                   iedge.subset(), iedge.debug_info());
            }
            for (auto& oedge : block_einsum->dataflow().out_edges(*code_node)) {
                auto* access_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
                if (!access_node_map.at(access_node))
                    access_node_map.at(access_node) = &builder.add_access(
                        new_block_before, access_node->data(), access_node->debug_info());
                builder.add_memlet(new_block_before, *new_code_node, oedge.src_conn(),
                                   *access_node_map.at(access_node), oedge.dst_conn(),
                                   oedge.subset(), oedge.debug_info());
            }
        }
    }

    // Create a copy of the loop (second loop) after the new einsum node if necessary
    structured_control_flow::StructuredLoop* new_loop_after = nullptr;
    if (dataflow_after_einsum || loop_root_index + 1 < this->loop_.root().size()) {
        if (auto* for_loop = dynamic_cast<structured_control_flow::For*>(&this->loop_)) {
            new_loop_after = &builder
                                  .add_for_after(parent, new_block_einsum, for_loop->indvar(),
                                                 for_loop->condition(), for_loop->init(),
                                                 for_loop->update(), for_loop->debug_info())
                                  .first;
        } else {
            auto* map = dynamic_cast<structured_control_flow::Map*>(&this->loop_);
            new_loop_after = &builder
                                  .add_map_after(parent, new_block_einsum, map->indvar(),
                                                 map->condition(), map->init(), map->update(),
                                                 map->schedule_type(), {}, map->debug_info())
                                  .first;
        }
    }

    // Extract and copy dataflow after the original einsum node
    if (dataflow_after_einsum) {
        auto& new_block_after =
            builder.add_block(new_loop_after->root(), {}, block_einsum->debug_info());
        std::unordered_map<data_flow::AccessNode*, data_flow::AccessNode*> access_node_map;
        for (auto* access_node : block_einsum->dataflow().data_nodes()) {
            access_node_map.insert({access_node, nullptr});
        }
        bool before_einsum_node = true;
        for (auto* node : topo_sort) {
            if (dynamic_cast<einsum::EinsumNode*>(node) == &this->einsum_node_) {
                before_einsum_node = false;
                continue;
            }
            if (before_einsum_node || !dynamic_cast<data_flow::CodeNode*>(node)) continue;
            auto* code_node = dynamic_cast<data_flow::CodeNode*>(node);
            data_flow::CodeNode* new_code_node = nullptr;
            if (auto* tasklet = dynamic_cast<data_flow::Tasklet*>(code_node)) {
                new_code_node =
                    &builder.add_tasklet(new_block_after, tasklet->code(), tasklet->output(),
                                         tasklet->inputs(), tasklet->debug_info());
            } else if (auto* libnode = dynamic_cast<data_flow::LibraryNode*>(node)) {
                new_code_node = &builder.copy_library_node(new_block_after, *libnode);
            }
            for (auto& iedge : block_einsum->dataflow().in_edges(*code_node)) {
                auto* access_node = dynamic_cast<data_flow::AccessNode*>(&iedge.src());
                if (!access_node_map.at(access_node))
                    access_node_map.at(access_node) = &builder.add_access(
                        new_block_after, access_node->data(), access_node->debug_info());
                builder.add_memlet(new_block_after, *access_node_map.at(access_node),
                                   iedge.src_conn(), *new_code_node, iedge.dst_conn(),
                                   iedge.subset(), iedge.debug_info());
            }
            for (auto& oedge : block_einsum->dataflow().out_edges(*code_node)) {
                auto* access_node = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
                if (!access_node_map.at(access_node))
                    access_node_map.at(access_node) = &builder.add_access(
                        new_block_after, access_node->data(), access_node->debug_info());
                builder.add_memlet(new_block_after, *new_code_node, oedge.src_conn(),
                                   *access_node_map.at(access_node), oedge.dst_conn(),
                                   oedge.subset(), oedge.debug_info());
            }
        }
    }

    // Put all successor nodes of the block with the original einsum node into the second loop
    if (loop_root_index + 1 < this->loop_.root().size()) {
        for (size_t i = loop_root_index + 1; i < this->loop_.root().size(); ++i) {
            deepcopy::StructuredSDFGDeepCopy deep_copy(builder, new_loop_after->root(),
                                                       this->loop_.root().at(i).first);
            deep_copy.copy();
        }
        for (size_t i = this->loop_.root().size() - 1; i > loop_root_index; --i)
            builder.remove_child(this->loop_.root(), i);
    }

    // Remove the original einsum block
    builder.remove_child(this->loop_.root(), loop_root_index);

    // If the loop is empty now, remove it
    if (this->loop_.root().size() == 0) builder.remove_child(parent, this->loop_);

    analysis_manager.invalidate_all();
}

void EinsumExpand::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["loop_element_id"] = this->loop_.element_id();
    j["einsum_node_id"] = this->einsum_node_.element_id();
}

EinsumExpand EinsumExpand::from_json(builder::StructuredSDFGBuilder& builder,
                                     const nlohmann::json& j) {
    size_t loop_id = j["loop_element_id"].get<size_t>();
    auto loop_element = builder.find_element_by_id(loop_id);
    if (!loop_element) {
        throw InvalidTransformationDescriptionException("Element with ID " +
                                                        std::to_string(loop_id) + " not found.");
    }
    auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop_element);

    size_t einsum_node_id = j["einsum_node_element_id"].get<size_t>();
    auto einsum_node_element = builder.find_element_by_id(einsum_node_id);
    if (!einsum_node_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(einsum_node_id) + " not found.");
    }
    auto einsum_node = dynamic_cast<einsum::EinsumNode*>(einsum_node_element);

    return EinsumExpand(*loop, *einsum_node);
}

}  // namespace transformations
}  // namespace sdfg
