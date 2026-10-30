#pragma once

#include <mcblas/mcblas.h>

#include "infini_train/src/kernels/common/gemm.h"

namespace infini_train::kernels::maca {

void Gemm(Device device, GemmParams params);

// Developer reference path: FP32 inputs/output with FP64 GEMM temporaries.
void GemmFp64Accum(Device device, GemmParams params, mcblasHandle_t blas_handle);

} // namespace infini_train::kernels::maca
