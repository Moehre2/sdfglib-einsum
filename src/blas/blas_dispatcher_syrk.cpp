#include "sdfg/blas/blas_dispatcher_syrk.h"

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/codegen/language_extension.h>
#include <sdfg/codegen/utils.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/data_flow_graph.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/function.h>
#include <sdfg/types/type.h>
#include <sdfg/types/utils.h>

#include <string>

#include "sdfg/blas/blas_node.h"
#include "sdfg/blas/blas_node_syrk.h"

namespace sdfg {
namespace blas {

void BLASDispatcherSyrk::dispatchCBLAS(codegen::PrettyPrinter& stream,
                                       const BLASNodeSyrk& blas_node) {
    const std::string n = blas_node.n()->__str__();
    const std::string k = blas_node.k()->__str__();

    stream << "cblas_" << blasType2String(blas_node.type()) << "syrk(CblasRowMajor, ";
    switch (blas_node.uplo()) {
        case BLASTriangular_Upper:
            stream << "CblasUpper";
            break;
        case BLASTriangular_Lower:
            stream << "CblasLower";
            break;
    }
    stream << ", ";
    switch (blas_node.trans()) {
        case BLASTranspose_No:
            stream << "CblasNoTrans";
            break;
        case BLASTranspose_Transpose:
            stream << "CblasTrans";
            break;
    }
    stream << ", " << n << ", " << k << ", " << blas_node.alpha() << ", " << blas_node.A() << ", ";
    if (blas_node.trans() == BLASTranspose_No)
        stream << k;
    else
        stream << n;
    stream << ", 1.0";
    if (blas_node.type() == BLASType_real) stream << "f";
    stream << ", " << blas_node.C() << ", " << n << ");" << std::endl;
}

void BLASDispatcherSyrk::dispatchCUBLAS(codegen::PrettyPrinter& stream,
                                        const BLASNodeSyrk& blas_node) {
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
    std::string uplo;
    switch (blas_node.uplo()) {
        case BLASTriangular_Upper:
            uplo = "CUBLAS_FILL_MODE_LOWER";
            break;
        case BLASTriangular_Lower:
            uplo = "CUBLAS_FILL_MODE_UPPER";
            break;
    }
    const std::string n = blas_node.n()->__str__();
    const std::string k = blas_node.k()->__str__();
    std::string trans, ldA;
    switch (blas_node.trans()) {
        case BLASTranspose_No:
            trans = "CUBLAS_OP_T";
            ldA = k;
            break;
        case BLASTranspose_Transpose:
            trans = "CUBLAS_OP_N";
            ldA = n;
            break;
    }
    const std::string alpha = blas_node.alpha();
    const std::string A = blas_node.A();
    const std::string dA = "d" + A;
    const std::string C = blas_node.C();
    const std::string dC = "d" + C;

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
           << type << " *" << dA << ", *" << dC << ";" << std::endl
           << std::endl
           << "CUDA_CHECK(cudaMalloc(&" << dA << ", " << n << " * " << k << " * sizeof(" << type
           << ")));" << std::endl
           << "CUDA_CHECK(cudaMalloc(&" << dC << ", " << n << " * " << n << " * sizeof(" << type
           << ")));" << std::endl
           << std::endl
           << type << " alpha = " << alpha << ";" << std::endl
           << type << " beta = " << beta << ";" << std::endl
           << "CUBLAS_CHECK(cublasSetMatrix(" << n << ", " << k << ", sizeof(" << type << "), " << A
           << ", " << n << ", " << dA << ", " << n << "));" << std::endl
           << "CUBLAS_CHECK(cublasSetMatrix(" << n << ", " << n << ", sizeof(" << type << "), " << C
           << ", " << n << ", " << dC << ", " << n << "));" << std::endl
           << std::endl
           << "CUBLAS_CHECK(cublas" << type2 << "syrk(handle, " << uplo << ", " << trans << ", "
           << n << ", " << k << ", &alpha, " << dA << ", " << ldA << ", &beta, " << dC << ", " << n
           << "));" << std::endl
           << std::endl
           << "CUDA_CHECK(cudaDeviceSynchronize());" << std::endl
           << std::endl
           << "CUBLAS_CHECK(cublasGetMatrix(" << n << ", " << n << ", sizeof(" << type << "), "
           << dC << ", " << n << ", " << C << ", " << n << "));" << std::endl
           << std::endl
           << "CUDA_CHECK(cudaFree(" << dA << "));" << std::endl
           << "CUDA_CHECK(cudaFree(" << dC << "));" << std::endl
           << std::endl
           << "CUBLAS_CHECK(cublasDestroy(handle));" << std::endl;
}

BLASDispatcherSyrk::BLASDispatcherSyrk(codegen::LanguageExtension& language_extension,
                                       const Function& function,
                                       const data_flow::DataFlowGraph& data_flow_graph,
                                       const data_flow::LibraryNode& node,
                                       const BLASImplementation impl)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node),
      impl_(impl) {}

void BLASDispatcherSyrk::dispatch(codegen::PrettyPrinter& stream) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Input connector declarations
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

    auto& blas_node = dynamic_cast<const BLASNodeSyrk&>(this->node_);

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