#pragma once

#include <sdfg/codegen/dispatchers/block_dispatcher.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/codegen/language_extension.h>
#include <sdfg/codegen/utils.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/function.h>

#include "sdfg/blas/blas_node_copy.h"

namespace sdfg {
namespace blas {

class BLASDispatcherCopy : public codegen::LibraryNodeDispatcher {
   public:
    BLASDispatcherCopy(codegen::LanguageExtension& language_extension, const Function& function,
                       const data_flow::DataFlowGraph& data_flow_graph,
                       const data_flow::LibraryNode& node);

    virtual void dispatch(codegen::PrettyPrinter& stream) override;
};

inline void register_blas_dispatcher_copy() {
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        LibraryNodeType_BLAS_copy.value(),
        [](codegen::LanguageExtension& language_extension, const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph, const data_flow::LibraryNode& node) {
            return std::make_unique<BLASDispatcherCopy>(language_extension, function,
                                                        data_flow_graph, node);
        });
}

}  // namespace blas
}  // namespace sdfg