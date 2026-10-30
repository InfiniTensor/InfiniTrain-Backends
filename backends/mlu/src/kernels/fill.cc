// 对应上游 src/kernels/cuda/fill.cu 的 host 集成部分；设备 kernel 在 fill.mlu。

#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/scalar.h"
#include "infini_train/include/tensor.h"

#include "kernels/mlu_kernel_api.h"
#include "runtime/mlu_dispatch.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::kernels::mlu {

namespace {

// Scalar::to<T>() 对 MLU 原生 16 位类型不能直接用：host 侧 __bang_bfloat16
// 没有 double 构造函数（static_cast 在 float/int 等整型 ctor 间歧义）。
// 按 scalar.h 中 TODO 的建议，后端自定义 scalar_cast，半精度统一走
// double -> float -> half/bf16 两级转换，与框架 to<BF16>() 的舍入路径一致。
template <typename T> T MluScalarCast(const Scalar &scalar) {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, bfloat16_t>) {
        return T(scalar.to<float>());
    } else {
        return scalar.to<T>();
    }
}

} // namespace

void Fill(std::shared_ptr<Tensor> tensor, Scalar scalar) {
    auto device = tensor->GetDevice();
    const auto queue = dynamic_cast<infini_train::core::mlu::MluStream *>(
                           infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                           ->mlu_queue();

    core::mlu::DispatchMluFunc<INFINI_MLU_ALL_NUMERIC_TYPES>(
        tensor->Dtype(),
        [=]<typename T>() {
            const T value = MluScalarCast<T>(scalar);
            static_assert(sizeof(T) <= sizeof(uint64_t));
            uint64_t value_bits = 0;
            std::memcpy(&value_bits, &value, sizeof(T));
            // 零填充快路径：单核标量 kernel 写 154MB（如 embedding 梯度表）要跑分钟级，
            // 实测会触发设备错误把队列搞崩；交零直接用 cnrtMemset（同步）
            if (value_bits == 0) {
                CNRT_CHECK(cnrtMemset(tensor->DataPtr(), 0, static_cast<size_t>(tensor->NumElements()) * sizeof(T)));
                return;
            }
            FillLaunch(tensor->DataPtr(), value_bits, MluTypeTag<T>(), static_cast<int64_t>(tensor->NumElements()),
                       queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "Fill");
        },
        "MLU Fill");
}
} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_FILL_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_FILL_KERNEL(Fill)

#undef REGISTER_MLU_FILL_KERNEL
