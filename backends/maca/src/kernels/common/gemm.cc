#include "kernels/common/gemm.h"

#include <mcblas/mcblas.h>

#include "glog/logging.h"

#include "common/common_maca.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "runtime/maca_runtime_common.h"

namespace infini_train::kernels::maca {
namespace {

mcblasOperation_t ToMcblasOperation(GemmTranspose transpose) {
    switch (transpose) {
    case GemmTranspose::kNoTranspose:
        return MCBLAS_OP_N;
    case GemmTranspose::kTranspose:
        return MCBLAS_OP_T;
    }
    LOG(FATAL) << "Gemm: unsupported transpose flag " << static_cast<int>(transpose);
    return MCBLAS_OP_N;
}

macaDataType ToMacaDataType(DataType dtype) {
    switch (dtype) {
    case DataType::kFLOAT32:
        return MACA_R_32F;
    case DataType::kBFLOAT16:
        return MACA_R_16BF;
    default:
        LOG(FATAL) << "Gemm: unsupported DataType " << static_cast<int>(dtype);
        return MACA_R_32F;
    }
}

} // namespace

void Gemm(Device device, GemmParams params) {
    const mcblasHandle_t blas_handle
        = dynamic_cast<core::maca::MacaBlasHandle *>(core::GetDeviceGuardImpl(device.type())->GetBlasHandle(device))
              ->mcblas_handle();

    if (params.batch_count == 1) {
        DCHECK_EQ(params.stride_a, 0LL);
        DCHECK_EQ(params.stride_b, 0LL);
        DCHECK_EQ(params.stride_c, 0LL);
    }

    const mcblasOperation_t trans_a = ToMcblasOperation(params.trans_a);
    const mcblasOperation_t trans_b = ToMcblasOperation(params.trans_b);
    const macaDataType type_a = ToMacaDataType(params.input_dtype);
    const macaDataType type_b = ToMacaDataType(params.input_dtype);
    const macaDataType type_c = ToMacaDataType(params.output_dtype);

    if (params.batch_count == 1) {
        MCBLAS_CHECK(mcblasGemmEx(blas_handle, trans_a, trans_b, params.m, params.n, params.k, &params.alpha, params.A,
                                  type_a, params.lda, params.B, type_b, params.ldb, &params.beta, params.C, type_c,
                                  params.ldc, MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT));
    } else {
        MCBLAS_CHECK(mcblasGemmStridedBatchedEx(
            blas_handle, trans_a, trans_b, params.m, params.n, params.k, &params.alpha, params.A, type_a, params.lda,
            params.stride_a, params.B, type_b, params.ldb, params.stride_b, &params.beta, params.C, type_c, params.ldc,
            params.stride_c, params.batch_count, MCBLAS_COMPUTE_32F, MCBLAS_GEMM_DEFAULT));
    }
}

REGISTER_KERNEL(Device::DeviceType::kPrivateUse1, Gemm, infini_train::kernels::maca::Gemm)

} // namespace infini_train::kernels::maca
