#include <gtest/gtest.h>

#include "sdfg/blas/blas_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/einsum/einsum_dispatcher.h"

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    sdfg::codegen::register_default_dispatchers();
    sdfg::einsum::register_einsum_dispatcher();
    sdfg::blas::register_blas_dispatchers();
    return RUN_ALL_TESTS();
}
