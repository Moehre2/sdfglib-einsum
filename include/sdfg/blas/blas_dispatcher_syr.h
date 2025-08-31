#pragma once

#include <sdfg/codegen/dispatchers/block_dispatcher.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/codegen/language_extension.h>
#include <sdfg/codegen/utils.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/function.h>

#include "sdfg/blas/blas_node_syr.h"

namespace sdfg {
namespace blas {

class BLASDispatcherSyr : public codegen::LibraryNodeDispatcher {
   public:
    BLASDispatcherSyr(codegen::LanguageExtension& language_extension, const Function& function,
                      const data_flow::DataFlowGraph& data_flow_graph,
                      const data_flow::LibraryNode& node);

    virtual void dispatch(codegen::PrettyPrinter& stream) override;
};

inline void register_blas_dispatcher_syr() {
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        LibraryNodeType_BLAS_syr.value(),
        [](codegen::LanguageExtension& language_extension, const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph, const data_flow::LibraryNode& node) {
            return std::make_unique<BLASDispatcherSyr>(language_extension, function,
                                                       data_flow_graph, node);
        });
}

}  // namespace blas
}  // namespace sdfg