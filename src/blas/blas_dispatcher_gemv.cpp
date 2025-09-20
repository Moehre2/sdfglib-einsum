#include "sdfg/blas/blas_dispatcher_gemv.h"

#include <sdfg/codegen/dispatchers/block_dispatcher.h>
#include <sdfg/codegen/language_extension.h>
#include <sdfg/codegen/utils.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/function.h>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_gemv.h"

namespace sdfg {
namespace blas {

void BLASDispatcherGemv::dispatchCBLAS(codegen::PrettyPrinter& stream,
                                       const BLASNodeGemv& blas_node) {
    stream << "cblas_" << blasType2String(blas_node.type()) << "gemv(CblasRowMajor, ";
    switch (blas_node.trans()) {
        case BLASTranspose_No:
            stream << "CblasNoTrans";
            break;
        case BLASTranspose_Transpose:
            stream << "CblasTrans";
            break;
    }
    stream << ", " << blas_node.m()->__str__() << ", " << blas_node.n()->__str__() << ", "
           << blas_node.alpha() << ", " << blas_node.A() << ", " << blas_node.n()->__str__() << ", "
           << blas_node.x() << ", 1, 1.0";
    if (blas_node.type() == BLASType_real) stream << "f";
    stream << ", " << blas_node.y() << ", 1);" << std::endl;
}

void BLASDispatcherGemv::dispatchCUBLAS(codegen::PrettyPrinter& stream,
                                        const BLASNodeGemv& blas_node) {
    std::string type, type2, beta;
    switch (blas_node.type()) {
        case BLASType_real:
            type = "float ";
            type2 = "S";
            beta = "1.0f";
            break;
        case BLASType_double:
            type = "double";
            type2 = "D";
            beta = "1.0";
            break;
    }
    const std::string m = blas_node.m()->__str__();
    const std::string n = blas_node.n()->__str__();
    std::string trans, x_size, y_size;
    switch (blas_node.trans()) {
        case BLASTranspose_No:
            trans = "CUBLAS_OP_T";
            x_size = m;
            y_size = n;
            break;
        case BLASTranspose_Transpose:
            trans = "CUBLAS_OP_N";
            x_size = n;
            y_size = m;
            break;
    }
    const std::string alpha = blas_node.alpha();
    const std::string A = blas_node.A();
    const std::string dA = "d" + A;
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
           << type << " *" << dA << ", *" << dx << ", *" << dy << ";" << std::endl
           << std::endl
           << "CUDA_CHECK(cudaMalloc(&" << dA << ", " << m << " * " << n << " * sizeof(" << type
           << ")));" << std::endl
           << "CUDA_CHECK(cudaMalloc(&" << dx << ", " << x_size << " * sizeof(" << type << ")));"
           << std::endl
           << "CUDA_CHECK(cudaMalloc(&" << dy << ", " << y_size << " * sizeof(" << type << ")));"
           << std::endl
           << std::endl
           << type << " alpha = " << alpha << ";" << std::endl
           << type << " beta = " << beta << ";" << std::endl
           << "CUBLAS_CHECK(cublasSetMatrix(" << m << ", " << n << ", sizeof(" << type << "), " << A
           << ", " << m << ", " << dA << ", " << m << "));" << std::endl
           << "CUBLAS_CHECK(cublasSetVector(" << x_size << ", sizeof(" << type << "), " << x
           << ", 1, " << dx << ", 1));" << std::endl
           << "CUBLAS_CHECK(cublasSetVector(" << y_size << ", sizeof(" << type << "), " << y
           << ", 1, " << dy << ", 1));" << std::endl
           << std::endl
           << "CUBLAS_CHECK(cublas" << type2 << "gemv(handle, " << trans << ", " << n << ", " << m
           << ", &alpha, " << dA << ", " << n << ", " << dx << ", 1, &beta, " << dy << ", 1));"
           << std::endl
           << std::endl
           << "CUDA_CHECK(cudaDeviceSynchronize());" << std::endl
           << std::endl
           << "CUBLAS_CHECK(cublasGetVector(" << y_size << ", sizeof(" << type << "), " << dy
           << ", 1, " << y << ", 1));" << std::endl
           << std::endl
           << "CUDA_CHECK(cudaFree(" << dA << "));" << std::endl
           << "CUDA_CHECK(cudaFree(" << dx << "));" << std::endl
           << "CUDA_CHECK(cudaFree(" << dy << "));" << std::endl
           << std::endl
           << "CUBLAS_CHECK(cublasDestroy(handle));" << std::endl;
}

BLASDispatcherGemv::BLASDispatcherGemv(codegen::LanguageExtension& language_extension,
                                       const Function& function,
                                       const data_flow::DataFlowGraph& data_flow_graph,
                                       const data_flow::LibraryNode& node,
                                       const BLASImplementation impl)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node),
      impl_(impl) {}

void BLASDispatcherGemv::dispatch(codegen::PrettyPrinter& stream) {
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

    auto& blas_node = dynamic_cast<const BLASNodeGemv&>(this->node_);

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