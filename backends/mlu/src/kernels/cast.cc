// 对应上游 src/kernels/cuda/cast.cu 的 host 集成部分；设备 kernel 在 cast.mlu。

#include <cstdint>
#include <memory>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype_dispatch.h"
#include "infini_train/include/tensor.h"

#include "kernels/mlu_kernel_api.h"
#include "runtime/mlu_dispatch.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::kernels::mlu {

std::shared_ptr<Tensor> Cast(std::shared_ptr<Tensor> input, DataType dtype) {
    auto device = input->GetDevice();
    auto dst_tensor = std::make_shared<Tensor>(input->Dims(), dtype, device);
    const auto queue = dynamic_cast<infini_train::core::mlu::MluStream *>(
                           infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                           ->mlu_queue();

    core::mlu::DispatchMluFunc<DataTypeList<INFINI_MLU_ALL_NUMERIC_TYPES, INFINI_LOGICAL_TYPES>,
                               DataTypeList<INFINI_MLU_ALL_NUMERIC_TYPES, INFINI_LOGICAL_TYPES>>(
        {dtype, input->Dtype()},
        [=]<typename Tdst, typename Tsrc>() {
            CastLaunch(dst_tensor->DataPtr(), input->DataPtr(), MluTypeTag<Tdst>(), MluTypeTag<Tsrc>(),
                       static_cast<int64_t>(input->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "Cast");
        },
        "MLU Cast");

    return {dst_tensor};
}
} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_CAST_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_CAST_KERNEL(Cast)

#undef REGISTER_MLU_CAST_KERNEL
