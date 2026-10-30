// 对应上游 src/kernels/cuda/elementwise.cu 的 host 集成部分；设备 kernel 在 elementwise.mlu。
// 只保留 GPT-2 训练路径需要的算子（一元 Neg/Reciprocal/Sin/Cos/Tanh/Pow/Rsqrt/Exp/Log/Sigmoid、
// 二元 Add/Sub/Mul/Div 及 Add/Mul 标量变体、Equals/EqualsScalar）。

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/common/common.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/scalar.h"
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

// 对应 CUDA 版 MakeBroadcastMeta：a/b 的 shape 右对齐到 out 的 ndim
MluBroadcastMeta MakeBroadcastMeta(const std::vector<int64_t> &a_dims, const std::vector<int64_t> &b_dims,
                                   const std::vector<int64_t> &out_dims) {
    MluBroadcastMeta m{};
    const int ndim = static_cast<int>(out_dims.size());
    CHECK_LE(ndim, MLU_MAX_BROADCAST_DIMS) << "Broadcast ndim exceeds MLU_MAX_BROADCAST_DIMS";
    m.ndim = ndim;

    std::vector<int64_t> a_shape(ndim, 1), b_shape(ndim, 1);
    std::copy_backward(a_dims.begin(), a_dims.end(), a_shape.end());
    std::copy_backward(b_dims.begin(), b_dims.end(), b_shape.end());

    const auto a_str = ComputeStrides(a_shape);
    const auto b_str = ComputeStrides(b_shape);
    const auto out_str = ComputeStrides(out_dims);

    for (int i = 0; i < ndim; ++i) {
        m.a_strides[i] = a_str[i];
        m.b_strides[i] = b_str[i];
        m.out_strides[i] = out_str[i];
        m.a_shape[i] = a_shape[i];
        m.b_shape[i] = b_shape[i];
    }
    return m;
}

bool ShapesEqual(const std::vector<int64_t> &a, const std::vector<int64_t> &b) { return a == b; }

// dtype 提升（与 CUDA 版一致：按 PromoteDataTypes 结果 Cast）
std::shared_ptr<Tensor> Promote(const std::shared_ptr<Tensor> &t, DataType promoted) {
    return t->Dtype() == promoted ? t : std::make_shared<Tensor>(t->To(promoted));
}

std::shared_ptr<Tensor> UnaryForwardImpl(const std::shared_ptr<Tensor> &input, MluUnaryOp op) {
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            UnaryForwardLaunch(output->DataPtr(), input->DataPtr(), op, MluTypeTag<T>(),
                               static_cast<int64_t>(input->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "UnaryForward");
            ::infini_train::core::mlu::DebugCheck<T>(output, "UnaryForward");
        },
        "MLU UnaryForward");
    return output;
}

// aux 语义见 mlu_kernel_api.h UnaryBackwardLaunch 注释
std::shared_ptr<Tensor> UnaryBackwardImpl(const std::shared_ptr<Tensor> &grad_output,
                                          const std::shared_ptr<Tensor> &aux, MluUnaryOp op) {
    const auto dtype = grad_output->Dtype();
    const auto aux_dtype = aux ? aux->Dtype() : dtype;
    const DataType promoted = PromoteDataTypes(dtype, aux_dtype);
    auto g = Promote(grad_output, promoted);
    auto x = aux ? Promote(aux, promoted) : nullptr;
    auto grad_in = std::make_shared<Tensor>(grad_output->Dims(), promoted, grad_output->GetDevice());
    const auto queue = CurrentQueue(grad_output->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        promoted,
        [=]<typename T>() {
            UnaryBackwardLaunch(grad_in->DataPtr(), g->DataPtr(), x ? x->DataPtr() : g->DataPtr(), op, MluTypeTag<T>(),
                                static_cast<int64_t>(g->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "UnaryBackward");
        },
        "MLU UnaryBackward");
    return grad_in;
}

std::shared_ptr<Tensor> BinaryForwardImpl(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b,
                                          MluBinaryOp op) {
    const DataType promoted = PromoteDataTypes(a->Dtype(), b->Dtype());
    auto ap = Promote(a, promoted);
    auto bp = Promote(b, promoted);
    // 与 CUDA 版一致：只假设 b -> a 单向广播
    CHECK(a->NumElements() >= b->NumElements() && a->NumElements() % b->NumElements() == 0);

    auto output = std::make_shared<Tensor>(a->Dims(), promoted, a->GetDevice());
    const auto meta = MakeBroadcastMeta(ap->Dims(), bp->Dims(), output->Dims());
    const auto queue = CurrentQueue(a->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        promoted,
        [=]<typename T>() {
            BinaryForwardLaunch(output->DataPtr(), ap->DataPtr(), bp->DataPtr(), op, MluTypeTag<T>(), meta,
                                static_cast<int64_t>(output->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "BinaryForward");
            ::infini_train::core::mlu::DebugCheck<T>(output, "BinaryForward");
        },
        "MLU BinaryForward");
    return output;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
BinaryBackwardImpl(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &a,
                   const std::shared_ptr<Tensor> &b, const std::vector<int64_t> &a_dims,
                   const std::vector<int64_t> &b_dims, MluBinaryOp op) {
    auto dtype = grad_output->Dtype();
    const auto a_dtype = a ? a->Dtype() : dtype;
    const auto b_dtype = b ? b->Dtype() : dtype;
    // 与 CUDA 版一致：计算 dtype 由前向保存的张量决定，而不是 grad_output
    const DataType promoted = PromoteDataTypes(a_dtype, b_dtype);
    auto g = Promote(grad_output, promoted);
    auto ap = a ? Promote(a, promoted) : nullptr;
    auto bp = b ? Promote(b, promoted) : nullptr;
    // Add/Sub 的反向不需要 a/b 的值（grad_a=g, grad_b=±g），调用方传 nullptr；
    // 用 g 占位保证 kernel 内索引不越界（公式本身不会读到）
    if (!ap) {
        ap = g;
    }
    if (!bp) {
        bp = g;
    }

    auto device = grad_output->GetDevice();
    auto grad_a = std::make_shared<Tensor>(a_dims, promoted, device);
    auto grad_b = std::make_shared<Tensor>(b_dims, promoted, device);

    const auto meta = MakeBroadcastMeta(a_dims, b_dims, g->Dims());
    const auto queue = CurrentQueue(device);
    const int64_t numel_a = grad_a->NumElements();
    const int64_t numel_b = grad_b->NumElements();
    const int64_t numel_out = g->NumElements();
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        promoted,
        [=]<typename T>() {
            BinaryBackwardLaunch(grad_a->DataPtr(), grad_b->DataPtr(), g->DataPtr(), ap->DataPtr(), bp->DataPtr(), op,
                                 MluTypeTag<T>(), meta, numel_a, numel_b, numel_out, queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "BinaryBackward");
        },
        "MLU BinaryBackward");
    return {grad_a, grad_b};
}

std::shared_ptr<Tensor> ScalarForwardImpl(const std::shared_ptr<Tensor> &input, float value, MluBinaryOp op) {
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            ScalarForwardLaunch(output->DataPtr(), input->DataPtr(), value, op, MluTypeTag<T>(),
                                static_cast<int64_t>(input->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "ScalarForward");
            if (std::getenv("MLU_DEBUG_CHECK")) {
                char tag[96];
                std::snprintf(tag, sizeof(tag), "ScalarForward(op=%d,scalar=%g) in", static_cast<int>(op), value);
                ::infini_train::core::mlu::DebugCheck<T>(input, tag);
                std::snprintf(tag, sizeof(tag), "ScalarForward(op=%d,scalar=%g) out", static_cast<int>(op), value);
                ::infini_train::core::mlu::DebugCheck<T>(output, tag);
            }
        },
        "MLU ScalarForward");
    return output;
}

std::shared_ptr<Tensor> ScalarBackwardImpl(const std::shared_ptr<Tensor> &grad_output, float value, MluBinaryOp op) {
    auto grad_in = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    const auto queue = CurrentQueue(grad_output->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        grad_output->Dtype(),
        [=]<typename T>() {
            ScalarBackwardLaunch(grad_in->DataPtr(), grad_output->DataPtr(), value, op, MluTypeTag<T>(),
                                 static_cast<int64_t>(grad_output->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "ScalarBackward");
        },
        "MLU ScalarBackward");
    return grad_in;
}

} // namespace

// ----------------------------- 一元 -----------------------------

std::shared_ptr<Tensor> NegForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForwardImpl(input, MluUnaryOp::kNeg);
}
std::shared_ptr<Tensor> NegBackward(const std::shared_ptr<Tensor> &grad_output) {
    return UnaryBackwardImpl(grad_output, nullptr, MluUnaryOp::kNeg);
}
std::shared_ptr<Tensor> ReciprocalForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForwardImpl(input, MluUnaryOp::kReciprocal);
}
std::shared_ptr<Tensor> ReciprocalBackward(const std::shared_ptr<Tensor> &grad_output,
                                           const std::shared_ptr<Tensor> &input) {
    return UnaryBackwardImpl(grad_output, input, MluUnaryOp::kReciprocal);
}
std::shared_ptr<Tensor> SinForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForwardImpl(input, MluUnaryOp::kSin);
}
std::shared_ptr<Tensor> SinBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input) {
    return UnaryBackwardImpl(grad_output, input, MluUnaryOp::kSin);
}
std::shared_ptr<Tensor> CosForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForwardImpl(input, MluUnaryOp::kCos);
}
std::shared_ptr<Tensor> CosBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input) {
    return UnaryBackwardImpl(grad_output, input, MluUnaryOp::kCos);
}
std::shared_ptr<Tensor> TanhForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForwardImpl(input, MluUnaryOp::kTanh);
}
std::shared_ptr<Tensor> TanhBackward(const std::shared_ptr<Tensor> &grad_output,
                                     const std::shared_ptr<Tensor> &output) {
    return UnaryBackwardImpl(grad_output, output, MluUnaryOp::kTanh);
}
std::shared_ptr<Tensor> RsqrtForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForwardImpl(input, MluUnaryOp::kRsqrt);
}
std::shared_ptr<Tensor> RsqrtBackward(const std::shared_ptr<Tensor> &grad_output,
                                      const std::shared_ptr<Tensor> &input) {
    return UnaryBackwardImpl(grad_output, input, MluUnaryOp::kRsqrt);
}
std::shared_ptr<Tensor> ExpForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForwardImpl(input, MluUnaryOp::kExp);
}
std::shared_ptr<Tensor> ExpBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &output) {
    return UnaryBackwardImpl(grad_output, output, MluUnaryOp::kExp);
}
std::shared_ptr<Tensor> LogForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForwardImpl(input, MluUnaryOp::kLog);
}
std::shared_ptr<Tensor> LogBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input) {
    return UnaryBackwardImpl(grad_output, input, MluUnaryOp::kLog);
}
std::shared_ptr<Tensor> SigmoidForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForwardImpl(input, MluUnaryOp::kSigmoid);
}
// 注意调用方参数顺序是 (output, grad_output)
std::shared_ptr<Tensor> SigmoidBackward(const std::shared_ptr<Tensor> &output,
                                        const std::shared_ptr<Tensor> &grad_output) {
    return UnaryBackwardImpl(grad_output, output, MluUnaryOp::kSigmoid);
}

std::shared_ptr<Tensor> PowForward(const std::shared_ptr<Tensor> &input, float exponent, bool scalar_is_base) {
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    const float e = exponent;
    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            PowForwardLaunch(output->DataPtr(), input->DataPtr(), e, scalar_is_base, MluTypeTag<T>(),
                             static_cast<int64_t>(input->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "PowForward");
            ::infini_train::core::mlu::DebugCheck<T>(output, "PowForward");
        },
        "MLU PowForward");
    return output;
}

std::shared_ptr<Tensor> PowBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    float exponent, bool scalar_is_base) {
    const DataType promoted = PromoteDataTypes(grad_output->Dtype(), input->Dtype());
    auto g = Promote(grad_output, promoted);
    auto x = Promote(input, promoted);
    auto grad_in = std::make_shared<Tensor>(g->Dims(), promoted, g->GetDevice());
    const float e = exponent;
    const auto queue = CurrentQueue(g->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_FLOATING_TYPES>(
        promoted,
        [=]<typename T>() {
            PowBackwardLaunch(grad_in->DataPtr(), g->DataPtr(), x->DataPtr(), e, scalar_is_base, MluTypeTag<T>(),
                              static_cast<int64_t>(g->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "PowBackward");
        },
        "MLU PowBackward");
    return grad_in;
}

// ----------------------------- 比较 -----------------------------

std::shared_ptr<Tensor> EqualsForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    CHECK(a->Dtype() == b->Dtype());
    auto output = std::make_shared<Tensor>(a->Dims(), DataType::kBOOL, a->GetDevice());
    const auto meta = MakeBroadcastMeta(a->Dims(), b->Dims(), output->Dims());
    const auto queue = CurrentQueue(a->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_ALL_NUMERIC_TYPES>(
        a->Dtype(),
        [=]<typename T>() {
            EqualsForwardLaunch(output->DataPtr(), a->DataPtr(), b->DataPtr(), MluTypeTag<T>(), meta,
                                static_cast<int64_t>(output->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "EqualsForward");
        },
        "MLU EqualsForward");
    return output;
}

std::shared_ptr<Tensor> EqualsScalarForward(const std::shared_ptr<Tensor> &input, float scalar) {
    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kBOOL, input->GetDevice());
    const double value = static_cast<double>(scalar);
    const auto queue = CurrentQueue(input->GetDevice());
    core::mlu::DispatchMluFunc<INFINI_MLU_ALL_NUMERIC_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            EqualsScalarForwardLaunch(output->DataPtr(), input->DataPtr(), value, MluTypeTag<T>(),
                                      static_cast<int64_t>(input->NumElements()), queue);
            ::infini_train::core::mlu::DebugSyncQueue(queue, "EqualsScalarForward");
        },
        "MLU EqualsScalarForward");
    return output;
}

// ----------------------------- 二元 -----------------------------

std::shared_ptr<Tensor> AddForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForwardImpl(a, b, MluBinaryOp::kAdd);
}
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> AddBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::vector<int64_t> &a_dims,
                                                                        const std::vector<int64_t> &b_dims) {
    return BinaryBackwardImpl(grad_output, nullptr, nullptr, a_dims, b_dims, MluBinaryOp::kAdd);
}
std::shared_ptr<Tensor> AddScalarForward(const std::shared_ptr<Tensor> &input, float scalar) {
    return ScalarForwardImpl(input, scalar, MluBinaryOp::kAdd);
}
std::shared_ptr<Tensor> AddScalarBackward(const std::shared_ptr<Tensor> &grad_output) {
    return ScalarBackwardImpl(grad_output, 0.0f, MluBinaryOp::kAdd);
}

std::shared_ptr<Tensor> SubForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForwardImpl(a, b, MluBinaryOp::kSub);
}
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> SubBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::vector<int64_t> &a_dims,
                                                                        const std::vector<int64_t> &b_dims) {
    return BinaryBackwardImpl(grad_output, nullptr, nullptr, a_dims, b_dims, MluBinaryOp::kSub);
}

std::shared_ptr<Tensor> MulForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForwardImpl(a, b, MluBinaryOp::kMul);
}
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MulBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::shared_ptr<Tensor> &a,
                                                                        const std::shared_ptr<Tensor> &b) {
    return BinaryBackwardImpl(grad_output, a, b, a->Dims(), b->Dims(), MluBinaryOp::kMul);
}
std::shared_ptr<Tensor> MulScalarForward(const std::shared_ptr<Tensor> &input, float scalar) {
    return ScalarForwardImpl(input, scalar, MluBinaryOp::kMul);
}
std::shared_ptr<Tensor> MulScalarBackward(const std::shared_ptr<Tensor> &grad_output, float scalar) {
    return ScalarBackwardImpl(grad_output, scalar, MluBinaryOp::kMul);
}

std::shared_ptr<Tensor> DivForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForwardImpl(a, b, MluBinaryOp::kDiv);
}
std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DivBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::shared_ptr<Tensor> &a,
                                                                        const std::shared_ptr<Tensor> &b) {
    return BinaryBackwardImpl(grad_output, a, b, a->Dims(), b->Dims(), MluBinaryOp::kDiv);
}

} // namespace infini_train::kernels::mlu

#define REGISTER_MLU_ELEMENTWISE_KERNEL(kernel_name)                                                                   \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kPrivateUse1, kernel_name,                                       \
                    infini_train::kernels::mlu::kernel_name)

REGISTER_MLU_ELEMENTWISE_KERNEL(NegForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(NegBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(ReciprocalForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(ReciprocalBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(SinForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(SinBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(CosForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(CosBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(TanhForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(TanhBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(PowForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(PowBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(RsqrtForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(RsqrtBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(ExpForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(ExpBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(LogForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(LogBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(SigmoidForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(SigmoidBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(EqualsForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(EqualsScalarForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(AddForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(AddBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(AddScalarForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(AddScalarBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(SubForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(SubBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(MulForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(MulBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(MulScalarForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(MulScalarBackward)
REGISTER_MLU_ELEMENTWISE_KERNEL(DivForward)
REGISTER_MLU_ELEMENTWISE_KERNEL(DivBackward)

#undef REGISTER_MLU_ELEMENTWISE_KERNEL
