#include "sdfg/einsum/einsum_serializer.h"

#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/symbolic/symbolic.h>
#include <symengine/expression.h>

#include <string>
#include <utility>
#include <vector>

namespace sdfg {
namespace einsum {

std::string EinsumSerializer::expression(const symbolic::Expression& expr) {
    serializer::JSONSymbolicPrinter printer;
    return printer.apply(expr);
}

nlohmann::json EinsumSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    if (library_node.code() != LibraryNodeType_Einsum) {
        throw std::runtime_error("Invalid library node type");
    }

    const auto& einsum_node = static_cast<const EinsumNode&>(library_node);

    nlohmann::json j;
    j["type"] = "library_node";
    j["code"] = std::string(LibraryNodeType_Einsum.value());
    j["side_effect"] = einsum_node.side_effect();

    j["inputs"] = nlohmann::json::array();
    for (const auto& input : einsum_node.inputs()) {
        j["inputs"].push_back(input);
    }

    j["outputs"] = nlohmann::json::array();
    for (const auto& output : einsum_node.outputs()) {
        j["outputs"].push_back(output);
    }

    j["maps"] = nlohmann::json::array();
    for (const auto& map : einsum_node.maps()) {
        nlohmann::json mapj = nlohmann::json::array();
        mapj.push_back(this->expression(map.first));
        mapj.push_back(this->expression(map.second));
        j["maps"].push_back(mapj);
    }

    j["out_indices"] = nlohmann::json::array();
    for (const auto& index : einsum_node.out_indices()) {
        j["out_indices"].push_back(this->expression(index));
    }

    j["in_indices"] = nlohmann::json::array();
    for (const auto& indices : einsum_node.in_indices()) {
        nlohmann::json indicesj = nlohmann::json::array();
        for (const auto& index : indices) {
            indicesj.push_back(this->expression(index));
        }
        j["in_indices"].push_back(indicesj);
    }

    return j;
}

data_flow::LibraryNode& EinsumSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::structured_control_flow::Block& parent) {
    if (j["type"] != "library_node") {
        throw std::runtime_error("Invalid library node type");
    }

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Einsum.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    auto inputs = j["inputs"].get<std::vector<std::string>>();
    auto outputs = j["outputs"].get<std::vector<std::string>>();

    std::vector<std::pair<symbolic::Symbol, symbolic::Expression>> maps;
    auto maps_str = j["maps"].get<std::vector<std::vector<std::string>>>();
    for (auto& map_str : maps_str) {
        if (map_str.size() != 2) {
            throw std::runtime_error("Invalid number of map arguments");
        }
        maps.push_back({symbolic::symbol(map_str[0]), SymEngine::Expression(map_str[1])});
    }

    data_flow::Subset out_indices;
    auto out_indices_str = j["out_indices"].get<std::vector<std::string>>();
    for (auto& index_str : out_indices_str) {
        out_indices.push_back(SymEngine::Expression(index_str));
    }

    std::vector<data_flow::Subset> in_indices;
    auto in_indices_str = j["in_indices"].get<std::vector<std::vector<std::string>>>();
    for (auto& indices_str : in_indices_str) {
        data_flow::Subset subset;
        for (auto& index_str : indices_str) {
            subset.push_back(SymEngine::Expression(index_str));
        }
        in_indices.push_back(subset);
    }

    auto& einsum_node =
        builder.add_library_node<EinsumNode, const std::vector<std::string>&,
                                 const std::vector<std::string>&,
                                 std::vector<std::pair<symbolic::Symbol, symbolic::Expression>>,
                                 data_flow::Subset, std::vector<data_flow::Subset>>(
            parent, DebugInfo(), outputs, inputs, maps, out_indices, in_indices);

    return einsum_node;
}

}  // namespace einsum
}  // namespace sdfg