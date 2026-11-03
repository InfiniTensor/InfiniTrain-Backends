#include <cstddef>
#include <memory>

#include <cnnl.h>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "common/common_mlu.h"
#include "runtime/mlu_guard_impl.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::kernels::mlu {
namespace {

cnnlDataType_t ToCnnlDType(DataType dtype) {
    switch (dtype) {
    case DataType::kBOOL:
        return CNNL_DTYPE_BOOL;
    case DataType::kUINT8:
        return CNNL_DTYPE_UINT8;
    case DataType::kINT8:
        return CNNL_DTYPE_INT8;
    case DataType::kUINT16:
        return CNNL_DTYPE_UINT16;
    case DataType::kINT16:
        return CNNL_DTYPE_INT16;
    case DataType::kUINT32:
        return CNNL_DTYPE_UINT32;
    case DataType::kINT32:
        return CNNL_DTYPE_INT32;
    case DataType::kUINT64:
        return CNNL_DTYPE_UINT64;
    case DataType::kINT64:
        return CNNL_DTYPE_INT64;
    case DataType::kBFLOAT16:
        return CNNL_DTYPE_BFLOAT16;
    case DataType::kFLOAT16:
        return CNNL_DTYPE_HALF;
    case DataType::kFLOAT32:
        return CNNL_DTYPE_FLOAT;
    case DataType::kFLOAT64:
        return CNNL_DTYPE_DOUBLE;
    default:
        LOG(FATAL) << "MLU Cast: unsupported dtype " << static_cast<int>(dtype);
    }
}

// Conversion selectors from the cnnlCastDataType_t table in cnnl.h. Returns
// false for pairs CNNL 2.2 does not provide (extend as the framework needs).
bool CastSelector(DataType src, DataType dst, cnnlCastDataType_t *selector) {
    const auto s = static_cast<int>(src);
    const auto d = static_cast<int>(dst);
    switch (s) {
    case static_cast<int>(DataType::kFLOAT32):
        switch (d) {
        case static_cast<int>(DataType::kFLOAT16):
            *selector = CNNL_CAST_FLOAT_TO_HALF;
            return true;
        case static_cast<int>(DataType::kBFLOAT16):
            *selector = CNNL_CAST_FLOAT_TO_BFLOAT16;
            return true;
        case static_cast<int>(DataType::kFLOAT64):
            *selector = CNNL_CAST_FLOAT_TO_DOUBLE;
            return true;
        case static_cast<int>(DataType::kINT32):
            *selector = CNNL_CAST_FLOAT_TO_INT32;
            return true;
        case static_cast<int>(DataType::kUINT32):
            *selector = CNNL_CAST_FLOAT_TO_UINT32;
            return true;
        case static_cast<int>(DataType::kINT16):
            *selector = CNNL_CAST_FLOAT_TO_INT16;
            return true;
        case static_cast<int>(DataType::kINT8):
            *selector = CNNL_CAST_FLOAT_TO_INT8;
            return true;
        case static_cast<int>(DataType::kUINT8):
            *selector = CNNL_CAST_FLOAT_TO_UINT8;
            return true;
        case static_cast<int>(DataType::kBOOL):
            *selector = CNNL_CAST_FLOAT_TO_BOOL;
            return true;
        case static_cast<int>(DataType::kINT64):
            *selector = CNNL_CAST_FLOAT_TO_INT64;
            return true;
        case static_cast<int>(DataType::kUINT64):
            *selector = CNNL_CAST_FLOAT_TO_UINT64;
            return true;
        }
        return false;
    case static_cast<int>(DataType::kFLOAT16):
        switch (d) {
        case static_cast<int>(DataType::kFLOAT32):
            *selector = CNNL_CAST_HALF_TO_FLOAT;
            return true;
        case static_cast<int>(DataType::kBFLOAT16):
            *selector = CNNL_CAST_HALF_TO_BFLOAT16;
            return true;
        case static_cast<int>(DataType::kFLOAT64):
            *selector = CNNL_CAST_HALF_TO_DOUBLE;
            return true;
        case static_cast<int>(DataType::kINT32):
            *selector = CNNL_CAST_HALF_TO_INT32;
            return true;
        case static_cast<int>(DataType::kINT16):
            *selector = CNNL_CAST_HALF_TO_INT16;
            return true;
        case static_cast<int>(DataType::kINT8):
            *selector = CNNL_CAST_HALF_TO_INT8;
            return true;
        case static_cast<int>(DataType::kUINT8):
            *selector = CNNL_CAST_HALF_TO_UINT8;
            return true;
        case static_cast<int>(DataType::kUINT32):
            *selector = CNNL_CAST_HALF_TO_UINT32;
            return true;
        case static_cast<int>(DataType::kUINT64):
            *selector = CNNL_CAST_HALF_TO_UINT64;
            return true;
        case static_cast<int>(DataType::kINT64):
            *selector = CNNL_CAST_HALF_TO_INT64;
            return true;
        case static_cast<int>(DataType::kBOOL):
            *selector = CNNL_CAST_HALF_TO_BOOL;
            return true;
        }
        return false;
    case static_cast<int>(DataType::kBFLOAT16):
        switch (d) {
        case static_cast<int>(DataType::kFLOAT32):
            *selector = CNNL_CAST_BFLOAT16_TO_FLOAT;
            return true;
        case static_cast<int>(DataType::kFLOAT16):
            *selector = CNNL_CAST_BFLOAT16_TO_HALF;
            return true;
        case static_cast<int>(DataType::kFLOAT64):
            *selector = CNNL_CAST_BFLOAT16_TO_DOUBLE;
            return true;
        case static_cast<int>(DataType::kINT8):
            *selector = CNNL_CAST_BFLOAT16_TO_INT8;
            return true;
        case static_cast<int>(DataType::kUINT8):
            *selector = CNNL_CAST_BFLOAT16_TO_UINT8;
            return true;
        case static_cast<int>(DataType::kINT16):
            *selector = CNNL_CAST_BFLOAT16_TO_INT16;
            return true;
        case static_cast<int>(DataType::kINT32):
            *selector = CNNL_CAST_BFLOAT16_TO_INT32;
            return true;
        case static_cast<int>(DataType::kUINT32):
            *selector = CNNL_CAST_BFLOAT16_TO_UINT32;
            return true;
        case static_cast<int>(DataType::kINT64):
            *selector = CNNL_CAST_BFLOAT16_TO_INT64;
            return true;
        case static_cast<int>(DataType::kUINT64):
            *selector = CNNL_CAST_BFLOAT16_TO_UINT64;
            return true;
        case static_cast<int>(DataType::kBOOL):
            *selector = CNNL_CAST_BFLOAT16_TO_BOOL;
            return true;
        }
        return false;
    case static_cast<int>(DataType::kFLOAT64):
        switch (d) {
        case static_cast<int>(DataType::kFLOAT32):
            *selector = CNNL_CAST_DOUBLE_TO_FLOAT;
            return true;
        case static_cast<int>(DataType::kFLOAT16):
            *selector = CNNL_CAST_DOUBLE_TO_HALF;
            return true;
        case static_cast<int>(DataType::kBFLOAT16):
            *selector = CNNL_CAST_DOUBLE_TO_BFLOAT16;
            return true;
        case static_cast<int>(DataType::kINT64):
            *selector = CNNL_CAST_DOUBLE_TO_INT64;
            return true;
        }
        return false;
    case static_cast<int>(DataType::kINT32):
        switch (d) {
        case static_cast<int>(DataType::kFLOAT32):
            *selector = CNNL_CAST_INT32_TO_FLOAT;
            return true;
        case static_cast<int>(DataType::kFLOAT16):
            *selector = CNNL_CAST_INT32_TO_HALF;
            return true;
        case static_cast<int>(DataType::kBFLOAT16):
            *selector = CNNL_CAST_INT32_TO_BFLOAT16;
            return true;
        case static_cast<int>(DataType::kINT8):
            *selector = CNNL_CAST_INT32_TO_INT8;
            return true;
        case static_cast<int>(DataType::kINT16):
            *selector = CNNL_CAST_INT32_TO_INT16;
            return true;
        case static_cast<int>(DataType::kUINT8):
            *selector = CNNL_CAST_INT32_TO_UINT8;
            return true;
        case static_cast<int>(DataType::kUINT32):
            *selector = CNNL_CAST_INT32_TO_UINT32;
            return true;
        case static_cast<int>(DataType::kUINT64):
            *selector = CNNL_CAST_INT32_TO_UINT64;
            return true;
        case static_cast<int>(DataType::kINT64):
            *selector = CNNL_CAST_INT32_TO_INT64;
            return true;
        case static_cast<int>(DataType::kBOOL):
            *selector = CNNL_CAST_INT32_TO_BOOL;
            return true;
        }
        return false;
    case static_cast<int>(DataType::kINT16):
        switch (d) {
        case static_cast<int>(DataType::kFLOAT32):
            *selector = CNNL_CAST_INT16_TO_FLOAT;
            return true;
        case static_cast<int>(DataType::kFLOAT16):
            *selector = CNNL_CAST_INT16_TO_HALF;
            return true;
        case static_cast<int>(DataType::kBFLOAT16):
            *selector = CNNL_CAST_INT16_TO_BFLOAT16;
            return true;
        case static_cast<int>(DataType::kINT64):
            *selector = CNNL_CAST_INT16_TO_INT64;
            return true;
        case static_cast<int>(DataType::kUINT64):
            *selector = CNNL_CAST_INT16_TO_UINT64;
            return true;
        case static_cast<int>(DataType::kINT32):
            *selector = CNNL_CAST_INT16_TO_INT32;
            return true;
        case static_cast<int>(DataType::kUINT32):
            *selector = CNNL_CAST_INT16_TO_UINT32;
            return true;
        case static_cast<int>(DataType::kINT8):
            *selector = CNNL_CAST_INT16_TO_INT8;
            return true;
        case static_cast<int>(DataType::kBOOL):
            *selector = CNNL_CAST_INT16_TO_BOOL;
            return true;
        case static_cast<int>(DataType::kUINT8):
            *selector = CNNL_CAST_INT16_TO_UINT8;
            return true;
        }
        return false;
    case static_cast<int>(DataType::kINT8):
        switch (d) {
        case static_cast<int>(DataType::kFLOAT32):
            *selector = CNNL_CAST_INT8_TO_FLOAT;
            return true;
        case static_cast<int>(DataType::kFLOAT16):
            *selector = CNNL_CAST_INT8_TO_HALF;
            return true;
        case static_cast<int>(DataType::kBFLOAT16):
            *selector = CNNL_CAST_INT8_TO_BFLOAT16;
            return true;
        case static_cast<int>(DataType::kUINT8):
            *selector = CNNL_CAST_INT8_TO_UINT8;
            return true;
        case static_cast<int>(DataType::kINT32):
            *selector = CNNL_CAST_INT8_TO_INT32;
            return true;
        case static_cast<int>(DataType::kUINT32):
            *selector = CNNL_CAST_INT8_TO_UINT32;
            return true;
        case static_cast<int>(DataType::kINT64):
            *selector = CNNL_CAST_INT8_TO_INT64;
            return true;
        case static_cast<int>(DataType::kUINT64):
            *selector = CNNL_CAST_INT8_TO_UINT64;
            return true;
        case static_cast<int>(DataType::kBOOL):
            *selector = CNNL_CAST_INT8_TO_BOOL;
            return true;
        case static_cast<int>(DataType::kINT16):
            *selector = CNNL_CAST_INT8_TO_INT16;
            return true;
        }
        return false;
    case static_cast<int>(DataType::kUINT8):
        switch (d) {
        case static_cast<int>(DataType::kFLOAT32):
            *selector = CNNL_CAST_UINT8_TO_FLOAT;
            return true;
        case static_cast<int>(DataType::kFLOAT16):
            *selector = CNNL_CAST_UINT8_TO_HALF;
            return true;
        case static_cast<int>(DataType::kBFLOAT16):
            *selector = CNNL_CAST_UINT8_TO_BFLOAT16;
            return true;
        case static_cast<int>(DataType::kBOOL):
            *selector = CNNL_CAST_UINT8_TO_BOOL;
            return true;
        case static_cast<int>(DataType::kINT8):
            *selector = CNNL_CAST_UINT8_TO_INT8;
            return true;
        case static_cast<int>(DataType::kINT16):
            *selector = CNNL_CAST_UINT8_TO_INT16;
            return true;
        case static_cast<int>(DataType::kINT32):
            *selector = CNNL_CAST_UINT8_TO_INT32;
            return true;
        case static_cast<int>(DataType::kUINT32):
            *selector = CNNL_CAST_UINT8_TO_UINT32;
            return true;
        case static_cast<int>(DataType::kUINT64):
            *selector = CNNL_CAST_UINT8_TO_UINT64;
            return true;
        case static_cast<int>(DataType::kINT64):
            *selector = CNNL_CAST_UINT8_TO_INT64;
            return true;
        }
        return false;
    case static_cast<int>(DataType::kINT64):
        switch (d) {
        case static_cast<int>(DataType::kINT32):
            *selector = CNNL_CAST_INT64_TO_INT32;
            return true;
        case static_cast<int>(DataType::kINT16):
            *selector = CNNL_CAST_INT64_TO_INT16;
            return true;
        case static_cast<int>(DataType::kINT8):
            *selector = CNNL_CAST_INT64_TO_INT8;
            return true;
        case static_cast<int>(DataType::kBOOL):
            *selector = CNNL_CAST_INT64_TO_BOOL;
            return true;
        case static_cast<int>(DataType::kFLOAT32):
            *selector = CNNL_CAST_INT64_TO_FLOAT;
            return true;
        case static_cast<int>(DataType::kFLOAT16):
            *selector = CNNL_CAST_INT64_TO_HALF;
            return true;
        case static_cast<int>(DataType::kBFLOAT16):
            *selector = CNNL_CAST_INT64_TO_BFLOAT16;
            return true;
        case static_cast<int>(DataType::kFLOAT64):
            *selector = CNNL_CAST_INT64_TO_DOUBLE;
            return true;
        case static_cast<int>(DataType::kUINT64):
            *selector = CNNL_CAST_INT64_TO_UINT64;
            return true;
        case static_cast<int>(DataType::kUINT32):
            *selector = CNNL_CAST_INT64_TO_UINT32;
            return true;
        case static_cast<int>(DataType::kUINT8):
            *selector = CNNL_CAST_INT64_TO_UINT8;
            return true;
        }
        return false;
    case static_cast<int>(DataType::kUINT64):
        switch (d) {
        case static_cast<int>(DataType::kFLOAT16):
            *selector = CNNL_CAST_UINT64_TO_HALF;
            return true;
        case static_cast<int>(DataType::kBFLOAT16):
            *selector = CNNL_CAST_UINT64_TO_BFLOAT16;
            return true;
        case static_cast<int>(DataType::kFLOAT32):
            *selector = CNNL_CAST_UINT64_TO_FLOAT;
            return true;
        case static_cast<int>(DataType::kUINT8):
            *selector = CNNL_CAST_UINT64_TO_UINT8;
            return true;
        case static_cast<int>(DataType::kBOOL):
            *selector = CNNL_CAST_UINT64_TO_BOOL;
            return true;
        case static_cast<int>(DataType::kUINT32):
            *selector = CNNL_CAST_UINT64_TO_UINT32;
            return true;
        }
        return false;
    case static_cast<int>(DataType::kUINT32):
        switch (d) {
        case static_cast<int>(DataType::kFLOAT32):
            *selector = CNNL_CAST_UINT32_TO_FLOAT;
            return true;
        case static_cast<int>(DataType::kFLOAT16):
            *selector = CNNL_CAST_UINT32_TO_HALF;
            return true;
        case static_cast<int>(DataType::kBFLOAT16):
            *selector = CNNL_CAST_UINT32_TO_BFLOAT16;
            return true;
        case static_cast<int>(DataType::kUINT8):
            *selector = CNNL_CAST_UINT32_TO_UINT8;
            return true;
        case static_cast<int>(DataType::kINT64):
            *selector = CNNL_CAST_UINT32_TO_INT64;
            return true;
        case static_cast<int>(DataType::kUINT64):
            *selector = CNNL_CAST_UINT32_TO_UINT64;
            return true;
        case static_cast<int>(DataType::kBOOL):
            *selector = CNNL_CAST_UINT32_TO_BOOL;
            return true;
        }
        return false;
    case static_cast<int>(DataType::kBOOL):
        switch (d) {
        case static_cast<int>(DataType::kFLOAT32):
            *selector = CNNL_CAST_BOOL_TO_FLOAT;
            return true;
        case static_cast<int>(DataType::kFLOAT16):
            *selector = CNNL_CAST_BOOL_TO_HALF;
            return true;
        case static_cast<int>(DataType::kBFLOAT16):
            *selector = CNNL_CAST_BOOL_TO_BFLOAT16;
            return true;
        case static_cast<int>(DataType::kINT8):
            *selector = CNNL_CAST_BOOL_TO_INT8;
            return true;
        case static_cast<int>(DataType::kUINT8):
            *selector = CNNL_CAST_BOOL_TO_UINT8;
            return true;
        case static_cast<int>(DataType::kINT16):
            *selector = CNNL_CAST_BOOL_TO_INT16;
            return true;
        case static_cast<int>(DataType::kINT32):
            *selector = CNNL_CAST_BOOL_TO_INT32;
            return true;
        case static_cast<int>(DataType::kINT64):
            *selector = CNNL_CAST_BOOL_TO_INT64;
            return true;
        case static_cast<int>(DataType::kUINT32):
            *selector = CNNL_CAST_BOOL_TO_UINT32;
            return true;
        case static_cast<int>(DataType::kUINT64):
            *selector = CNNL_CAST_BOOL_TO_UINT64;
            return true;
        }
        return false;
    default:
        return false;
    }
}
} // namespace

std::shared_ptr<Tensor> Cast(std::shared_ptr<Tensor> input, DataType dtype) {
    auto dst_tensor = std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());
    auto device = input->GetDevice();
    auto *stream = core::GetDeviceGuardImpl(device.type())->GetStream(device);
    const auto &mlu_queue = dynamic_cast<core::mlu::MluStream *>(stream)->mlu_queue();
    core::mlu::MluBlasHandle blas_handle(stream);

    if (input->Dtype() == dtype) { // identity cast: plain device-to-device copy
        core::GetDeviceGuardImpl(device.type())
            ->Memcpy(dst_tensor->DataPtr(), input->DataPtr(), input->NumElements() * kDataTypeToSize.at(dtype),
                     core::MemcpyKind::kD2D);
        return dst_tensor;
    }

    cnnlCastDataType_t selector = CNNL_CAST_FLOAT_TO_HALF;
    CHECK(CastSelector(input->Dtype(), dtype, &selector))
        << "MLU Cast: no CNNL conversion from " << static_cast<int>(input->Dtype()) << " to "
        << static_cast<int>(dtype);

    cnnlTensorDescriptor_t x_desc = nullptr;
    cnnlTensorDescriptor_t y_desc = nullptr;
    CNNL_CHECK(cnnlCreateTensorDescriptor(&x_desc));
    CNNL_CHECK(cnnlCreateTensorDescriptor(&y_desc));
    CHECK_LE(input->NumElements(), 0x7fffffffull) << "MLU Cast: tensor too large";
    int dims[1] = {static_cast<int>(input->NumElements())};
    CNNL_CHECK(cnnlSetTensorDescriptor(x_desc, CNNL_LAYOUT_ARRAY, ToCnnlDType(input->Dtype()), 1, dims));
    CNNL_CHECK(cnnlSetTensorDescriptor(y_desc, CNNL_LAYOUT_ARRAY, ToCnnlDType(dtype), 1, dims));
    CNNL_CHECK(
        cnnlCastDataType(blas_handle.cnnl_handle(), x_desc, input->DataPtr(), selector, y_desc, dst_tensor->DataPtr()));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(x_desc));
    CNNL_CHECK(cnnlDestroyTensorDescriptor(y_desc));

    return dst_tensor;
}
} // namespace infini_train::kernels::mlu

namespace infini_train::kernels::mlu {

REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, Cast, infini_train::kernels::mlu::Cast);

} // namespace infini_train::kernels::mlu
