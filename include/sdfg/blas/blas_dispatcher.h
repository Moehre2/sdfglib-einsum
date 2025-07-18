#pragma once

#include "sdfg/blas/blas_dispatcher_gemm.h"

namespace sdfg {
namespace blas {

inline void register_blas_dispatchers() { register_blas_dispatcher_gemm(); }

}  // namespace blas
}  // namespace sdfg