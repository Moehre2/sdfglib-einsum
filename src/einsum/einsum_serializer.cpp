#include "sdfg/einsum/einsum_serializer.h"

namespace sdfg {
namespace einsum {

nlohmann::json EinsumSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    if (library_node.type() != LibraryNodeType_Einsum) {
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

    // TODO: Add more fields to the EinsumNode

    return j;
}

data_flow::LibraryNode& EinsumSerializer::deserialize(const nlohmann::json& j,
                                                      sdfg::builder::StructuredSDFGBuilder& builder,
                                                      sdfg::structured_control_flow::Block& parent) {
    if (j["type"] != "library_node") {
        throw std::runtime_error("Invalid library node type");
    }

    auto code = j["code"].get<std::string_view>();
    if (code != LibraryNodeType_Einsum.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    auto side_effect = j["side_effect"].get<bool>();
    auto inputs = j["inputs"].get<std::vector<std::string>>();
    auto outputs = j["outputs"].get<std::vector<std::string>>();

    auto& einsum_node = builder.add_library_node<EinsumNode>(parent, code, outputs, inputs, side_effect);

    // TODO: Add more fields to the EinsumNode

    return einsum_node;
}

}
}