#pragma once

#include <memory>

#include "sdfg/blas/blas_node.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/dispatchers/library_nodes/library_node_dispatcher.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/function.h"

namespace sdfg {
namespace blas {

class BLASDispatcher : public codegen::LibraryNodeDispatcher {
   public:
    BLASDispatcher(codegen::LanguageExtension& language_extension, const Function& function,
                   const data_flow::DataFlowGraph& data_flow_graph,
                   const data_flow::LibraryNode& node);

    virtual void dispatch(codegen::PrettyPrinter& stream) override;
};

// This function must be called by the application using the plugin
inline void register_blas_dispatcher() {
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        LibraryNodeType_BLAS_gemm.value(),
        [](codegen::LanguageExtension& language_extension, const Function& function,
           const data_flow::DataFlowGraph& data_flow_graph, const data_flow::LibraryNode& node) {
            return std::make_unique<BLASDispatcher>(language_extension, function, data_flow_graph,
                                                    node);
        });
}

}  // namespace blas
}  // namespace sdfg