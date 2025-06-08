#pragma once

#include <sdfg/serializer/json_serializer.h>
#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace einsum {

/**
 * @brief Serializer for Einsum nodes
 * 
 * This serializer is used to serialize and deserialize Einsum nodes.
 * It is registered with the LibraryNodeSerializerRegistry.
 */
class EinsumSerializer : public serializer::LibraryNodeSerializer {
public:
    virtual nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    virtual data_flow::LibraryNode& deserialize(const nlohmann::json& j,
                                                sdfg::builder::StructuredSDFGBuilder& builder,
                                                sdfg::structured_control_flow::Block& parent) override;
};

// This function must be called by the application using the plugin
inline void register_einsum_serializer() {
    serializer::LibraryNodeSerializerRegistry::instance().register_library_node_serializer(
        LibraryNodeType_Einsum,
        []() { return std::make_unique<EinsumSerializer>(); }
    );
}

}
}
