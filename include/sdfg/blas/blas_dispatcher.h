#pragma once

#include "sdfg/blas/blas_dispatcher_axpy.h"
#include "sdfg/blas/blas_dispatcher_copy.h"
#include "sdfg/blas/blas_dispatcher_dot.h"
#include "sdfg/blas/blas_dispatcher_gemm.h"
#include "sdfg/blas/blas_dispatcher_gemv.h"

namespace sdfg {
namespace blas {

// This function must be called by the application using the plugin
inline void register_blas_dispatchers() {
    register_blas_dispatcher_axpy();
    register_blas_dispatcher_copy();
    register_blas_dispatcher_dot();
    register_blas_dispatcher_gemv();
    register_blas_dispatcher_gemm();
}

}  // namespace blas
}  // namespace sdfg