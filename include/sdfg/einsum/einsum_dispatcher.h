#pragma once

#include <sdfg/codegen/dispatchers/block_dispatcher.h>
#include "sdfg/einsum/einsum_node.h"

namespace sdfg {
namespace einsum {

class EinsumDispatcher : public codegen::LibraryNodeDispatcher {
   public:
    EinsumDispatcher(LanguageExtension& language_extension, const Function& function,
                          const data_flow::DataFlowGraph& data_flow_graph,
                          const data_flow::LibraryNode& node);

    virtual void dispatch(PrettyPrinter& stream) override;
};

// This function must be called by the application using the plugin
inline void register_einsum_dispatcher() {
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        LibraryNodeType_Einsum,
        []() { return std::make_unique<EinsumDispatcher>(); }
    );
}

}
}