#include "sdfg/blas/blas_dispatcher_axpy.h"

#include <sdfg/codegen/dispatchers/block_dispatcher.h>
#include <sdfg/codegen/language_extension.h>
#include <sdfg/codegen/utils.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/function.h>

#include <string>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_axpy.h"

namespace sdfg {
namespace blas {

void BLASDispatcherAxpy::dispatchCBLAS(codegen::PrettyPrinter& stream,
                                       const BLASNodeAxpy& blas_node) {
    stream << "cblas_" << blasType2String(blas_node.type()) << "axpy(" << blas_node.n()->__str__()
           << ", " << blas_node.alpha() << ", " << blas_node.x() << ", 1, " << blas_node.y()
           << ", 1);" << std::endl;
}

void BLASDispatcherAxpy::dispatchCUBLAS(codegen::PrettyPrinter& stream,
                                        const BLASNodeAxpy& blas_node) {
    std::string type, type2;
    switch (blas_node.type()) {
        case BLASType_real:
            type = "float ";
            type2 = "S";
            break;
        case BLASType_double:
            type = "double";
            type2 = "D";
            break;
    }
    const std::string n = blas_node.n()->__str__();
    const std::string alpha = blas_node.alpha();
    const std::string x = blas_node.x();
    const std::string dx = "d" + x;
    const std::string y = blas_node.y();
    const std::string dy = "d" + y;

    stream << "#ifndef CUDA_CHECK" << std::endl
           << "#define CUDA_CHECK(X) X" << std::endl
           << "#endif" << std::endl
           << "#ifndef CUBLAS_CHECK" << std::endl
           << "#define CUBLAS_CHECK(X) X" << std::endl
           << "#endif" << std::endl
           << std::endl
           << "cublasHandle_t handle;" << std::endl
           << "CUBLAS_CHECK(cublasCreate(&handle));" << std::endl
           << std::endl
           << type << " *" << dx << ", *" << dy << ";" << std::endl
           << std::endl
           << "CUDA_CHECK(cudaMalloc(&" << dx << ", " << n << " * sizeof(" << type << ")));"
           << std::endl
           << "CUDA_CHECK(cudaMalloc(&" << dy << ", " << n << " * sizeof(" << type << ")));"
           << std::endl
           << std::endl
           << type << " alpha = " << alpha << ";" << std::endl
           << "CUBLAS_CHECK(cublasSetVector(" << n << ", sizeof(" << type << "), " << x << ", 1, "
           << dx << ", 1));" << std::endl
           << "CUBLAS_CHECK(cublasSetVector(" << n << ", sizeof(" << type << "), " << y << ", 1, "
           << dy << ", 1));" << std::endl
           << std::endl
           << "CUBLAS_CHECK(cublas" << type2 << "axpy(handle, " << n << ", &alpha, " << dx
           << ", 1, " << dy << ", 1));" << std::endl
           << std::endl
           << "CUDA_CHECK(cudaDeviceSynchronize());" << std::endl
           << std::endl
           << "CUBLAS_CHECK(cublasGetVector(" << n << ", sizeof(" << type << "), " << dy << ", 1, "
           << y << ", 1));" << std::endl
           << std::endl
           << "CUDA_CHECK(cudaFree(" << dx << "));" << std::endl
           << "CUDA_CHECK(cudaFree(" << dy << "));" << std::endl
           << std::endl
           << "CUBLAS_CHECK(cublasDestroy(handle));" << std::endl;
}

BLASDispatcherAxpy::BLASDispatcherAxpy(codegen::LanguageExtension& language_extension,
                                       const Function& function,
                                       const data_flow::DataFlowGraph& data_flow_graph,
                                       const data_flow::LibraryNode& node,
                                       const BLASImplementation impl)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node),
      impl_(impl) {}

void BLASDispatcherAxpy::dispatch(codegen::PrettyPrinter& stream) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    for (auto& iedge : this->data_flow_graph_.in_edges(this->node_)) {
        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        const types::IType& src_type = this->function_.type(src.data());

        auto& conn_name = iedge.dst_conn();
        auto& conn_type = types::infer_type(this->function_, src_type, iedge.subset());

        stream << this->language_extension_.declaration(conn_name, conn_type) << " = " << src.data()
               << this->language_extension_.subset(this->function_, src_type, iedge.subset()) << ";"
               << std::endl;
    }
    stream << std::endl;

    auto& blas_node = dynamic_cast<const BLASNodeAxpy&>(this->node_);

    switch (this->impl_) {
        case BLASImplementation_CBLAS:
            this->dispatchCBLAS(stream, blas_node);
            break;
        case BLASImplementation_CUBLAS:
            this->dispatchCUBLAS(stream, blas_node);
            break;
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

}  // namespace blas
}  // namespace sdfg