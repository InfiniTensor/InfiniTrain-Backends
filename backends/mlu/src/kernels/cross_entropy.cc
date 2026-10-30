// 对应上游 src/kernels/cuda/cross_entropy.cu 的 host 集成部分；设备 kernel 在 cross_entropy.mlu。
// 与 CUDA 版一致：kernel 只算每样本 loss，批均值在 host 侧 FP64 累加（含 D2H/H2D）。

#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "kernels/mlu_kernel_api.h"
#include "runtime/mlu_debug.h"
#include "runtime/mlu_dispatch.h"
#include "runtime/mlu_runtime_common.h"

namespace infini_train::kernels::mlu {
namespace {
cnrtQueue_t CurrentQueue(const Device &device) {
    return dynamic_cast<infini_train::core::mlu::MluStream *>(
               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->mlu_queue();
}
} // namespace

std::shared_ptr<Tensor> CrossEntropyForward(const std::shared_ptr<Tensor> &input,
                                            const std::shared_ptr<Tensor> &target) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t num_classes = *input_dims.rbegin();

    auto batched_output = std::make_shared<Tensor>(std::vector<int64_t>{bs}, input->Dtype(), input->GetDevice());
    const auto queue = CurrentQueue(input->GetDevice());

    return core::mlu::DispatchMluFunc<DataTypeList<DataType::kUINT8, DataType::kINT64>,
                                      DataTypeList<INFINI_MLU_FLOATING_TYPES>>(
        {target->Dtype(), input->Dtype()},
        [=]<typename Ttarget, typename Tinput>() {
            CrossEntropyForwardLaunch(input->DataPtr(), target->DataPtr(), batched_output->DataPtr(), bs, num_classes,
                                      MluTypeTag<Ttarget>(), MluTypeTag<Tinput>(), queue);
            // 探针放在 Launch 之后、第一次 To(Device())（含 D2H）之前；
            // 原先写在 return 之后属死代码，导致这个最重的 kernel 从未被覆盖。
            ::infini_train::core::mlu::DebugSyncQueue(queue, "CrossEntropyForward");
            ::infini_train::core::mlu::DebugCheck<Tinput>(batched_output, "CrossEntropyForward");

            // 与 CUDA 版一致：host 侧 FP64 累加求均值（含 D2H/H2D 同步）
            auto loss_cpu = batched_output->To(Device());
            auto loss = std::make_shared<Tensor>(std::vector<int64_t>{}, input->Dtype(), Device());
            const auto *loss_cpu_ptr = static_cast<const Tinput *>(loss_cpu.DataPtr());
            const double loss_sum
                = std::accumulate(loss_cpu_ptr, loss_cpu_ptr + bs, 0.0,
                                  [](double acc, const Tinput &val) { return acc + static_cast<double>(val); });
            const float loss_mean = static_cast<float>(loss_sum / static_cast<double>(bs));
            static_cast<Tinput *>(loss->DataPtr())[0] = static_cast<Tinput>(loss_mean);

            return std::make_shared<Tensor>(loss->To(input->GetDevice()));
        },
        "MLU CrossEntropyForward");
}

std::shared_ptr<Tensor> CrossEntropyBackward(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &target,
                                             const std::shared_ptr<Tensor> &grad_output) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t num_classes = *input_dims.rbegin();

    auto input_casted = std::make_shared<Tensor>(input->To(grad_output->Dtype()));
    CHECK_EQ(grad_output->Dims().size(), 0);
    auto grad_input = std::make_shared<Tensor>(input_casted->Dims(), input_casted->Dtype(), grad_output->GetDevice());

    const auto queue = CurrentQueue(target->GetDevice());
    core::mlu::DispatchMluFunc<DataTypeList<DataType::kUINT8, DataType::kINT64>,
                               DataTypeList<INFINI_MLU_FLOATING_TYPES>>(
        {target->Dtype(), input_casted->Dtype()},
        [=]<typename Ttarget, typename Tinput>() {
            CrossEntropyBackwardLaunch(input_casted->DataPtr(), grad_input->DataPtr(), target->DataPtr(),
                                       grad_output->DataPtr(), bs, num_classes, MluTypeTag<Ttarget>(),
                                       MluTypeTag<Tinput>(), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "CrossEntropyBackward");
        },
        "MLU CrossEntropyBackward");
    return {grad_input};
}
} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_CROSS_ENTROPY_KERNEL(kernel_name)                                                                 \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_CROSS_ENTROPY_KERNEL(CrossEntropyForward)
REGISTER_MLU_CROSS_ENTROPY_KERNEL(CrossEntropyBackward)

#undef REGISTER_MLU_CROSS_ENTROPY_KERNEL
