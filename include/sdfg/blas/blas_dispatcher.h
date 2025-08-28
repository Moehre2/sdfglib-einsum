#pragma once

#include "sdfg/blas/blas_dispatcher_axpy.h"
#include "sdfg/blas/blas_dispatcher_copy.h"
#include "sdfg/blas/blas_dispatcher_dot.h"
#include "sdfg/blas/blas_dispatcher_gemm.h"
#include "sdfg/blas/blas_dispatcher_gemv.h"
#include "sdfg/blas/blas_dispatcher_ger.h"
#include "sdfg/blas/blas_dispatcher_symm.h"
#include "sdfg/blas/blas_dispatcher_symv.h"
#include "sdfg/blas/blas_dispatcher_syr.h"
#include "sdfg/blas/blas_dispatcher_syrk.h"

namespace sdfg {
namespace blas {

// This function must be called by the application using the plugin
inline void register_blas_dispatchers() {
    register_blas_dispatcher_axpy();
    register_blas_dispatcher_copy();
    register_blas_dispatcher_dot();
    register_blas_dispatcher_gemv();
    register_blas_dispatcher_symv();
    register_blas_dispatcher_ger();
    register_blas_dispatcher_syr();
    register_blas_dispatcher_gemm();
    register_blas_dispatcher_symm();
    register_blas_dispatcher_syrk();
}

}  // namespace blas
}  // namespace sdfg