#include <cstddef>
#include <cstring>
#include <memory>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/scalar.h"
#include "infini_train/include/tensor.h"

#include "runtime/mlu_guard_impl.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::kernels::mlu {
namespace {

extern "C" void infini_train_mlu_fill_pattern(void *dst, unsigned int elemsize, unsigned long long pattern,
                                              unsigned int num_elements, cnrtDim3_t dim, cnrtFunctionType_t func_type,
                                              cnrtQueue_t queue);

// Element bit pattern for the fill constant, converted with the same
// Scalar::to<T>() semantics the CPU kernel uses.
unsigned long long BuildPattern(const Scalar &scalar, DataType dtype) {
    unsigned long long pattern = 0;
    switch (dtype) {
    case DataType::kBOOL: {
        unsigned char v = scalar.to<bool>() ? 1 : 0;
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kUINT8: {
        auto v = scalar.to<uint8_t>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kINT8: {
        auto v = scalar.to<int8_t>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kUINT16: {
        auto v = scalar.to<uint16_t>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kINT16: {
        auto v = scalar.to<int16_t>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kUINT32: {
        auto v = scalar.to<uint32_t>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kINT32: {
        auto v = scalar.to<int32_t>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kUINT64: {
        auto v = scalar.to<uint64_t>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kINT64: {
        auto v = scalar.to<int64_t>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kBFLOAT16: {
        BF16 v = scalar.to<BF16>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kFLOAT16: {
        FP16 v = scalar.to<FP16>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kFLOAT32: {
        float v = scalar.to<float>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    case DataType::kFLOAT64: {
        double v = scalar.to<double>();
        std::memcpy(&pattern, &v, sizeof(v));
        break;
    }
    default:
        LOG(FATAL) << "MLU Fill: unsupported dtype " << static_cast<int>(dtype);
    }
    return pattern;
}
} // namespace

void Fill(std::shared_ptr<Tensor> tensor, Scalar scalar) {
    auto device = tensor->GetDevice();
    auto *stream = core::GetDeviceGuardImpl(device.type())->GetStream(device);
    const auto &mlu_stream = dynamic_cast<core::mlu::MluStream *>(stream)->mlu_queue();

    const size_t elemsize = kDataTypeToSize.at(tensor->Dtype());
    CHECK_LE(tensor->NumElements(), 0xffffffffull) << "MLU Fill: tensor too large";

    unsigned long long pattern = BuildPattern(scalar, tensor->Dtype());
    cnrtDim3_t dim = {4, 1, 1};
    infini_train_mlu_fill_pattern(tensor->DataPtr(), (unsigned int)elemsize, pattern,
                                  (unsigned int)tensor->NumElements(), dim, cnrtFuncTypeUnion1, mlu_stream);
}
} // namespace infini_train::kernels::mlu

namespace infini_train::kernels::mlu {

REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, Fill, infini_train::kernels::mlu::Fill);

} // namespace infini_train::kernels::mlu
