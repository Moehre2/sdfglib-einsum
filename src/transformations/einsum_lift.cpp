#include "sdfg/transformations/einsum_lift.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/exceptions.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_control_flow/for.h>
#include <sdfg/structured_control_flow/return.h>
#include <sdfg/structured_control_flow/sequence.h>
#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/transformations/transformation.h>
#include <symengine/basic.h>

#include <functional>
#include <nlohmann/json_fwd.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace transformations {

symbolic::Expression EinsumLift::taskletCode2Expr(const data_flow::TaskletCode code,
                                                  const std::vector<symbolic::Expression>& args) {
    if (code == data_flow::TaskletCode::assign && args.size() == 1) {
        return args[0];
    } else if (code == data_flow::TaskletCode::neg && args.size() == 1) {
        return symbolic::sub(symbolic::zero(), args[0]);
    } else if (code == data_flow::TaskletCode::add && args.size() == 2) {
        return symbolic::add(args[0], args[1]);
    } else if (code == data_flow::TaskletCode::sub && args.size() == 2) {
        return symbolic::sub(args[0], args[1]);
    } else if (code == data_flow::TaskletCode::mul && args.size() == 2) {
        return symbolic::mul(args[0], args[1]);
    } else if (code == data_flow::TaskletCode::fma && args.size() == 3) {
        return symbolic::add(symbolic::mul(args[0], args[1]), args[2]);
    } else if (code == data_flow::TaskletCode::max && args.size() == 2) {
        return symbolic::max(args[0], args[1]);
    } else if (code == data_flow::TaskletCode::min && args.size() == 2) {
        return symbolic::min(args[0], args[1]);
    } else if ((code == data_flow::TaskletCode::pow || code == data_flow::TaskletCode::powf ||
                code == data_flow::TaskletCode::powl) &&
               args.size() == 2) {
        return symbolic::pow(args[0], args[1]);
    }
    return symbolic::__nullptr__();
}

std::string EinsumLift::createAccessExpr(const std::string& name, const data_flow::Subset& subset) {
    std::stringstream cont;

    cont << name;
    for (auto& sym : subset) cont << "[" << sym->__str__() << "]";

    return cont.str();
}

bool EinsumLift::checkMulExpr(const symbolic::Expression expr) {
    if (expr->get_type_code() == SymEngine::TypeID::SYMENGINE_SYMBOL) return true;
    if (expr->get_type_code() == SymEngine::TypeID::SYMENGINE_MUL && expr->get_args().size() > 0) {
        for (symbolic::Expression mul : expr->get_args()) {
            if (!this->checkMulExpr(mul)) return false;
        }
        return true;
    }
    return false;
}

EinsumLift::EinsumLift(
    std::vector<std::reference_wrapper<structured_control_flow::StructuredLoop>> loops,
    structured_control_flow::Block& comp_block)
    : loops_(loops), comp_block_(comp_block) {}

std::string EinsumLift::name() const { return "EinsumLift"; }

bool EinsumLift::can_be_applied(builder::StructuredSDFGBuilder& builder,
                                analysis::AnalysisManager& analysis_manager) {
    // Check if loops are nested in each other with computation block inside
    if (this->loops_.size() > 0) {
        size_t i;
        for (i = 0; i < this->loops_.size() - 1; ++i) {
            if (this->loops_[i].get().root().size() != 1 ||
                this->loops_[i].get().root().at(0).first.element_id() !=
                    this->loops_[i + 1].get().element_id())
                return false;
        }
        if (this->loops_[i].get().root().size() != 1 ||
            this->loops_[i].get().root().at(0).first.element_id() != this->comp_block_.element_id())
            return false;
    }

    // Check that each loop is of sufficient form
    for (auto loop : this->loops_) {
        if (loop.get().init()->get_type_code() != SymEngine::TypeID::SYMENGINE_INTEGER)
            return false;
        if (loop.get().init()->__str__() != "0") return false;
        if (loop.get().condition()->get_type_code() !=
                SymEngine::TypeID::SYMENGINE_STRICTLESSTHAN &&
            loop.get().condition()->get_type_code() != SymEngine::TypeID::SYMENGINE_LESSTHAN)
            return false;
        if (loop.get().condition()->get_args().size() != 2) return false;
        if (loop.get().condition()->get_args().at(0)->__str__() != loop.get().indvar()->__str__())
            return false;
        if (loop.get().condition()->get_args().at(1)->get_type_code() !=
                SymEngine::TypeID::SYMENGINE_INTEGER &&
            loop.get().condition()->get_args().at(1)->get_type_code() !=
                SymEngine::TypeID::SYMENGINE_SYMBOL)
            return false;
        if (loop.get().update()->get_type_code() != SymEngine::TypeID::SYMENGINE_ADD) return false;
        if (loop.get().update()->get_args().size() != 2) return false;
        if (loop.get().update()->get_args().at(0)->__str__() != loop.get().indvar()->__str__() &&
            loop.get().update()->get_args().at(1)->__str__() != loop.get().indvar()->__str__())
            return false;
        if (loop.get().update()->get_args().at(0)->__str__() != "1" &&
            loop.get().update()->get_args().at(1)->__str__() != "1")
            return false;
        for (auto other_loop : this->loops_) {
            if (other_loop.get().element_id() == loop.get().element_id()) continue;
            if (other_loop.get().indvar()->__str__() == loop.get().indvar()->__str__())
                return false;
        }
    }

    // Capture all supported calculations from all tasklets in comp block
    std::vector<std::string> outputs, inputs;
    std::vector<data_flow::Subset> out_indicess, in_indices;
    auto& comp_dfg = this->comp_block_.dataflow();
    std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> comps, comps_static_assign;
    for (auto tasklet : comp_dfg.tasklets()) {
        switch (tasklet->code()) {
            case data_flow::TaskletCode::assign:
            case data_flow::TaskletCode::neg:
            case data_flow::TaskletCode::add:
            case data_flow::TaskletCode::sub:
            case data_flow::TaskletCode::mul:
            case data_flow::TaskletCode::fma:
            case data_flow::TaskletCode::max:
            case data_flow::TaskletCode::min:
            case data_flow::TaskletCode::pow:
            case data_flow::TaskletCode::powf:
            case data_flow::TaskletCode::powl:
                break;
            default:
                return false;
        }
        auto& oedge = *comp_dfg.out_edges(*tasklet).begin();
        std::string out_cont = dynamic_cast<const data_flow::AccessNode&>(oedge.dst()).data();
        symbolic::Symbol new_comp_out =
            symbolic::symbol(this->createAccessExpr(out_cont, oedge.subset()));
        outputs.push_back(out_cont);
        out_indicess.push_back(oedge.subset());
        std::unordered_map<std::string, symbolic::Expression> input_map;
        for (auto& iedge : comp_dfg.in_edges(*tasklet)) {
            std::string in_cont = dynamic_cast<const data_flow::AccessNode&>(iedge.src()).data();
            input_map.insert({iedge.dst_conn(),
                              symbolic::symbol(this->createAccessExpr(in_cont, iedge.subset()))});
            inputs.push_back(in_cont);
            in_indices.push_back(iedge.subset());
        }
        std::vector<symbolic::Expression> comp_ins;
        for (auto tasklet_input : tasklet->inputs()) {
            if (input_map.contains(tasklet_input.first)) {
                comp_ins.push_back(input_map.at(tasklet_input.first));
            } else if (tasklet_input.first == "0" || tasklet_input.first == "0.0") {
                comp_ins.push_back(symbolic::zero());
            } else if (tasklet_input.first == "1" || tasklet_input.first == "1.0") {
                comp_ins.push_back(symbolic::one());
            } else {
                comp_ins.push_back(symbolic::symbol(tasklet_input.first));
                inputs.push_back(tasklet_input.first);
                in_indices.push_back({});
            }
        }
        symbolic::Expression new_comp = this->taskletCode2Expr(tasklet->code(), comp_ins);
        if (symbolic::eq(new_comp, symbolic::__nullptr__())) return false;
        if (tasklet->code() == data_flow::TaskletCode::assign && input_map.size() == 0)
            comps_static_assign.push_back({new_comp_out, new_comp});
        else
            comps.push_back({new_comp_out, new_comp});
    }

    // Perform a fixed point iteration to join all captured calculations to one "dummy" calculation
    if (comps.size() == 0) return false;
    std::vector<bool> comps_used(comps.size(), false),
        comps_static_assign_used(comps_static_assign.size(), false);
    symbolic::Symbol comp_out = comps[0].first;
    symbolic::Expression comp = comps[0].second;
    comps_used[0] = true;
    bool applied = false;
    do {
        applied = false;
        for (size_t i = 0; i < comps_static_assign.size(); ++i) {
            if (symbolic::uses(comp, comps_static_assign[i].first)) {
                comp = symbolic::subs(comp, comps_static_assign[i].first,
                                      comps_static_assign[i].second);
                applied = true;
                comps_static_assign_used[i] = true;
            }
        }
        for (size_t i = 0; i < comps.size(); ++i) {
            if (symbolic::eq(comp_out, comps[i].first)) continue;
            if (symbolic::uses(comp, comps[i].first)) {
                comp = symbolic::subs(comp, comps[i].first, comps[i].second);
                applied = true;
                comps_used[i] = true;
            }
            if (symbolic::uses(comps[i].second, comp_out)) {
                comp = symbolic::subs(comps[i].second, comp_out, comp);
                comp_out = comps[i].first;
                applied = true;
                comps_used[i] = true;
            }
        }
    } while (applied);

    // Check if all captured calculations were used
    for (bool used : comps_used) {
        if (!used) return false;
    }
    for (bool used : comps_static_assign_used) {
        if (!used) return false;
    }

    // Check that we captured the output container with its subset
    std::string output;
    data_flow::Subset out_indices;
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (comp_out->get_name() == this->createAccessExpr(outputs[i], out_indicess[i])) {
            output = outputs[i];
            out_indices = out_indicess[i];
            break;
        }
    }
    if (output.empty()) return false;

    // Check that the output container does not use an index variable of the loops
    for (auto& index : out_indices) {
        for (auto& loop : this->loops_) {
            if (symbolic::uses(index, loop.get().indvar())) return false;
        }
    }

    // Simplify "dummy" calculation and check if it can be represented in Einstein notation
    symbolic::Expression scomp = symbolic::simplify(comp);
    if (scomp->get_type_code() == SymEngine::TypeID::SYMENGINE_ADD) {
        symbolic::Expression scomp_mul;
        if (scomp->get_args().at(0)->get_type_code() == SymEngine::TypeID::SYMENGINE_SYMBOL &&
            symbolic::eq(scomp->get_args().at(0), comp_out))
            scomp_mul = scomp->get_args().at(1);
        else if (scomp->get_args().at(1)->get_type_code() == SymEngine::TypeID::SYMENGINE_SYMBOL &&
                 symbolic::eq(scomp->get_args().at(1), comp_out))
            scomp_mul = scomp->get_args().at(0);
        else
            return false;
        if (symbolic::uses(scomp_mul, comp_out) || !this->checkMulExpr(scomp_mul)) return false;
    } else if (scomp->get_type_code() == SymEngine::TypeID::SYMENGINE_MUL) {
        if (symbolic::uses(scomp, comp_out) || !this->checkMulExpr(scomp)) return false;
    } else if (scomp->get_type_code() == SymEngine::TypeID::SYMENGINE_SYMBOL) {
        if (symbolic::uses(scomp, comp_out)) return false;
    } else {
        return false;
    }

    // Reduce inputs and in_indices to the ones occurring in the simplified "dummy" calculation
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!symbolic::uses(scomp, this->createAccessExpr(inputs[i], in_indices[i]))) {
            inputs.erase(inputs.begin() + i);
            in_indices.erase(in_indices.begin() + i);
            --i;
        }
    }

    // Check that the index variables of the loops occur without a calculation in the in indices
    for (auto& indices : in_indices) {
        for (auto& index : indices) {
            for (auto& loop : this->loops_) {
                if (symbolic::uses(index, loop.get().indvar()) &&
                    !symbolic::eq(index, loop.get().indvar()))
                    return false;
            }
        }
    }

    return true;
}

void EinsumLift::apply(builder::StructuredSDFGBuilder& builder,
                       analysis::AnalysisManager& analysis_manager) {
    // Construct maps
    std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> maps;
    for (auto& loop : this->loops_) {
        if (loop.get().condition()->get_args().at(0)->__str__() == loop.get().indvar()->__str__())
            maps.push_back({loop.get().indvar(), loop.get().condition()->get_args().at(1)});
        else
            maps.push_back({loop.get().indvar(), loop.get().condition()->get_args().at(0)});
    }

    // Capture all supported calculations from all tasklets in comp block
    std::vector<std::string> outputs, inputs;
    std::vector<data_flow::Subset> out_indicess, in_indices;
    auto& comp_dfg = this->comp_block_.dataflow();
    std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> comps, comps_static_assign;
    for (auto tasklet : comp_dfg.tasklets()) {
        auto& oedge = *comp_dfg.out_edges(*tasklet).begin();
        std::string out_cont = dynamic_cast<const data_flow::AccessNode&>(oedge.dst()).data();
        symbolic::Symbol new_comp_out =
            symbolic::symbol(this->createAccessExpr(out_cont, oedge.subset()));
        outputs.push_back(out_cont);
        out_indicess.push_back(oedge.subset());
        std::unordered_map<std::string, symbolic::Expression> input_map;
        for (auto& iedge : comp_dfg.in_edges(*tasklet)) {
            std::string in_cont = dynamic_cast<const data_flow::AccessNode&>(iedge.src()).data();
            input_map.insert({iedge.dst_conn(),
                              symbolic::symbol(this->createAccessExpr(in_cont, iedge.subset()))});
            inputs.push_back(in_cont);
            in_indices.push_back(iedge.subset());
        }
        std::vector<symbolic::Expression> comp_ins;
        for (auto tasklet_input : tasklet->inputs()) {
            if (input_map.contains(tasklet_input.first)) {
                comp_ins.push_back(input_map.at(tasklet_input.first));
            } else if (tasklet_input.first == "0" || tasklet_input.first == "0.0") {
                comp_ins.push_back(symbolic::zero());
            } else if (tasklet_input.first == "1" || tasklet_input.first == "1.0") {
                comp_ins.push_back(symbolic::one());
            } else {
                comp_ins.push_back(symbolic::symbol(tasklet_input.first));
                inputs.push_back(tasklet_input.first);
                in_indices.push_back({});
            }
        }
        symbolic::Expression new_comp = this->taskletCode2Expr(tasklet->code(), comp_ins);
        if (tasklet->code() == data_flow::TaskletCode::assign && input_map.size() == 0)
            comps_static_assign.push_back({new_comp_out, new_comp});
        else
            comps.push_back({new_comp_out, new_comp});
    }

    // Perform a fixed point iteration to join all captured calculations to one "dummy" calculation
    symbolic::Symbol comp_out = comps[0].first;
    symbolic::Expression comp = comps[0].second;
    bool applied = false;
    do {
        applied = false;
        for (size_t i = 0; i < comps_static_assign.size(); ++i) {
            if (symbolic::uses(comp, comps_static_assign[i].first)) {
                comp = symbolic::subs(comp, comps_static_assign[i].first,
                                      comps_static_assign[i].second);
                applied = true;
            }
        }
        for (size_t i = 0; i < comps.size(); ++i) {
            if (symbolic::eq(comp_out, comps[i].first)) continue;
            if (symbolic::uses(comp, comps[i].first)) {
                comp = symbolic::subs(comp, comps[i].first, comps[i].second);
                applied = true;
                break;
            }
            if (symbolic::uses(comps[i].second, comp_out)) {
                comp = symbolic::subs(comps[i].second, comp_out, comp);
                comp_out = comps[i].first;
                applied = true;
                break;
            }
        }
    } while (applied);

    // Determine the output container with its subset
    std::string output;
    data_flow::Subset out_indices;
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (comp_out->get_name() == this->createAccessExpr(outputs[i], out_indicess[i])) {
            output = outputs[i];
            out_indices = out_indicess[i];
            break;
        }
    }

    // Simplify "dummy" calculation
    symbolic::Expression scomp = symbolic::simplify(comp);

    // Determine if simplified "dummy" calculation contains output container with subsets
    bool out_in_scomp = symbolic::uses(scomp, this->createAccessExpr(output, out_indices));

    // Reduce inputs and in_indices to the ones occurring in the simplified "dummy" calculation
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!symbolic::uses(scomp, this->createAccessExpr(inputs[i], in_indices[i]))) {
            inputs.erase(inputs.begin() + i);
            in_indices.erase(in_indices.begin() + i);
            --i;
        } else if (symbolic::uses(comp_out, this->createAccessExpr(inputs[i], in_indices[i]))) {
            inputs.erase(inputs.begin() + i);
            in_indices.erase(in_indices.begin() + i);
            --i;
        }
    }

    // Get the most outer node and its parent node
    structured_control_flow::ControlFlowNode* most_outer_node;
    if (this->loops_.size() == 0)
        most_outer_node = &this->comp_block_;
    else
        most_outer_node = &this->loops_[0].get();
    auto& parent = builder.parent(*most_outer_node);

    // Add a new block after the most outer node and remove the most outer node
    auto block_and_transition = builder.add_block_after(parent, *most_outer_node);
    auto& block = block_and_transition.first;
    builder.remove_child(parent, *most_outer_node);

    // Put output back into inputs and in_indices if it was there before
    if (out_in_scomp) {
        inputs.push_back(output);
        in_indices.push_back(out_indices);
    }

    // Add access nodes
    data_flow::AccessNode& out_access = builder.add_access(block, output);
    std::vector<std::reference_wrapper<data_flow::AccessNode>> in_access;
    for (auto& input : inputs) in_access.push_back(builder.add_access(block, input));

    // Create connectors
    std::vector<std::string> in_conns;
    for (size_t i = 0; i < inputs.size(); ++i) in_conns.push_back("_in" + std::to_string(i));
    if (out_in_scomp) in_conns[inputs.size() - 1] = "_out";

    // Add einsum node as library node
    auto& libnode =
        builder.add_library_node<einsum::EinsumNode,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            block, einsum::LibraryNodeType_Einsum, {"_out"}, in_conns, false, DebugInfo(), maps,
            out_indices, in_indices);

    // Add memlets
    builder.add_memlet(block, libnode, "_out", out_access, "void", {});
    for (size_t i = 0; i < inputs.size(); ++i)
        builder.add_memlet(block, in_access[i], "void", libnode, in_conns[i], {});
}

void EinsumLift::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["loops_element_ids"] = nlohmann::json::array();
    for (auto loop : this->loops_) j["loops_element_ids"].push_back(loop.get().element_id());
    j["comp_block_element_id"] = this->comp_block_.element_id();
}

EinsumLift EinsumLift::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    std::vector<std::reference_wrapper<structured_control_flow::StructuredLoop>> loops;
    std::vector<size_t> loop_ids = j["loop_element_id"].get<std::vector<size_t>>();
    for (size_t loop_id : loop_ids) {
        auto loop_element = builder.find_element_by_id(loop_id);
        if (!loop_element) {
            throw InvalidTransformationDescriptionException(
                "Element with ID " + std::to_string(loop_id) + " not found.");
        }
        auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop_element);
        loops.push_back(*loop);
    }

    size_t comp_tasklet_id = j["comp_block_element_id"].get<size_t>();
    auto comp_tasklet_element = builder.find_element_by_id(comp_tasklet_id);
    if (!comp_tasklet_element) {
        throw InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(comp_tasklet_id) + " not found.");
    }
    auto comp_block = dynamic_cast<structured_control_flow::Block*>(comp_tasklet_element);

    return EinsumLift(loops, *comp_block);
}

}  // namespace transformations
}  // namespace sdfg
