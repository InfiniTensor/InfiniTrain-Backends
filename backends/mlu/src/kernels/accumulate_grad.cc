// 对应上游 src/kernels/cuda/accumulate_grad.cu 的 host 集成部分；设备 kernel 在 accumulate_grad.mlu。

#include <cmath>
#include <cstdint>
#include <memory>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "kernels/mlu_kernel_api.h"
#include "runtime/mlu_dispatch.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::kernels::mlu {

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    auto device = tensor->GetDevice();
    const auto queue = dynamic_cast<infini_train::core::mlu::MluStream *>(
                           infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                           ->mlu_queue();

    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        gradient->Dtype(),
        [=]<typename T>() {
            AccumulateGradLaunch(gradient->DataPtr(), rate, tensor->DataPtr(), MluTypeTag<T>(),
                                 static_cast<int64_t>(gradient->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "AccumulateGrad");
        },
        "MLU AccumulateGrad");
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

    auto device = grad->GetDevice();
    const auto queue = dynamic_cast<infini_train::core::mlu::MluStream *>(
                           infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                           ->mlu_queue();

    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        grad->Dtype(),
        [=]<typename T>() {
            AdamAccumulateGradLaunch(grad->DataPtr(), param->DataPtr(), m->DataPtr(), v->DataPtr(), learning_rate,
                                     beta1, beta2, eps, bias_correction_m, bias_correction_v, MluTypeTag<T>(),
                                     static_cast<int64_t>(grad->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "AdamAccumulateGrad");
        },
        "MLU AdamAccumulateGrad");
}
} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_MLU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_MLU_ACCUMULATE_GRAD_KERNEL
