#pragma once

#include <sdfg/codegen/dispatchers/block_dispatcher.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>

#include <memory>

#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace einsum {

class EinsumDispatcher : public codegen::LibraryNodeDispatcher {
   public:
    EinsumDispatcher(codegen::LanguageExtension& language_extension, const Function& function,
                     const data_flow::DataFlowGraph& data_flow_graph,
                     const data_flow::LibraryNode& node);

    virtual void dispatch(codegen::PrettyPrinter& stream) override;
};

// This function must be called by the application using the plugin
inline void register_einsum_dispatcher() {
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        LibraryNodeType_Einsum.value(),
        [](codegen::LanguageExtension& language_extension, const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph, const data_flow::LibraryNode& node) {
            return std::make_unique<EinsumDispatcher>(language_extension, function, data_flow_graph,
                                                      node);
        });
}

}  // namespace einsum
}  // namespace sdfg