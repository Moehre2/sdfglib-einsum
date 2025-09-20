#pragma once

#include <sdfg/codegen/dispatchers/block_dispatcher.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/codegen/language_extension.h>
#include <sdfg/codegen/utils.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/function.h>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_axpy.h"

namespace sdfg {
namespace blas {

class BLASDispatcherAxpy : public codegen::LibraryNodeDispatcher {
   private:
    const BLASImplementation impl_;

    void dispatchCBLAS(codegen::PrettyPrinter& stream, const BLASNodeAxpy& blas_node);
    void dispatchCUBLAS(codegen::PrettyPrinter& stream, const BLASNodeAxpy& blas_node);

   public:
    BLASDispatcherAxpy(codegen::LanguageExtension& language_extension, const Function& function,
                       const data_flow::DataFlowGraph& data_flow_graph,
                       const data_flow::LibraryNode& node, const BLASImplementation impl);

    virtual void dispatch(codegen::PrettyPrinter& stream) override;
};

inline void register_blas_dispatcher_axpy(BLASImplementation impl) {
    codegen::LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        LibraryNodeType_BLAS_axpy.value(),
        [impl](codegen::LanguageExtension& language_extension, const Function& function,
               const data_flow::DataFlowGraph& data_flow_graph,
               const data_flow::LibraryNode& node) {
            return std::make_unique<BLASDispatcherAxpy>(language_extension, function,
                                                        data_flow_graph, node, impl);
        });
}

}  // namespace blas
}  // namespace sdfg