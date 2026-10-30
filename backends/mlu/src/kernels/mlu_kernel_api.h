#pragma once

// MLU kernel 启动器接口头。
//
// 关键约束：cncc 基于 clang 11（C++17 级别），编译不了框架的 C++20 头文件
// （std::format、模板 lambda 等），所以 .mlu 文件只允许包含本头 + cnrt/bang
// 头，保持 C++11 干净；框架集成（dispatch、REGISTER_KERNEL）全部放在由 g++
// 编译的同名 .cc 里，通过这里声明的 *Launch 函数调起设备 kernel。

#include <stdint.h>

#include <cnrt.h>

namespace infini_train::kernels::mlu {

// 与 mlu/runtime/mlu_dispatch.h 的 MluTypeTag<T>() 一一对应
enum class MluScalarType : int {
    kBool = 0,
    kU8,
    kI8,
    kU16,
    kI16,
    kU32,
    kI32,
    kU64,
    kI64,
    kF16,
    kBF16,
    kF32,
    kF64,
};

// 填充：value_bits 按 type 位宽取低字节解释
void FillLaunch(void *data, uint64_t value_bits, MluScalarType type, int64_t num_elements, cnrtQueue_t queue);

void CastLaunch(void *dst, const void *src, MluScalarType dst_type, MluScalarType src_type, int64_t num_elements,
                cnrtQueue_t queue);

// 梯度累加：tensor += grad * rate
void AccumulateGradLaunch(const void *grad, float rate, void *tensor, MluScalarType type, int64_t num_elements,
                          cnrtQueue_t queue);

// Adam 更新（bias correction 在 host 侧算好传入）
void AdamAccumulateGradLaunch(const void *grad, void *param, void *m, void *v, float learning_rate, float beta1,
                              float beta2, float eps, float bias_correction_m, float bias_correction_v,
                              MluScalarType type, int64_t num_elements, cnrtQueue_t queue);

// -----------------------------------------------------------------------------
// elementwise
// -----------------------------------------------------------------------------

// Broadcast 元数据（POD，按值传入 kernel）。对应 CUDA 版 elementwise.cu 的
// BroadcastMeta：shape 右对齐到 out 的 ndim，strides 为连续布局。
#define MLU_MAX_BROADCAST_DIMS 8
struct MluBroadcastMeta {
    int32_t ndim;
    int64_t a_strides[MLU_MAX_BROADCAST_DIMS];
    int64_t b_strides[MLU_MAX_BROADCAST_DIMS];
    int64_t out_strides[MLU_MAX_BROADCAST_DIMS];
    int64_t a_shape[MLU_MAX_BROADCAST_DIMS];
    int64_t b_shape[MLU_MAX_BROADCAST_DIMS];
};

enum class MluUnaryOp : int {
    kNeg = 0,
    kReciprocal,
    kTanh,
    kRsqrt,
    kExp,
    kLog,
    kSin,
    kCos,
    kSigmoid,
};

enum class MluBinaryOp : int {
    kAdd = 0,
    kSub,
    kMul,
    kDiv,
};

// 一元前向：out[i] = op(in[i])；16 位浮点在 kernel 内经 float 中转
void UnaryForwardLaunch(void *out, const void *in, MluUnaryOp op, MluScalarType type, int64_t num_elements,
                        cnrtQueue_t queue);

// 一元反向：grad_in[i] = g(grad_out[i], aux[i])；aux 按算子约定为 forward 的
// output（tanh/exp/sigmoid）或 input（reciprocal/rsqrt/log/sin/cos）
void UnaryBackwardLaunch(void *grad_in, const void *grad_out, const void *aux, MluUnaryOp op, MluScalarType type,
                         int64_t num_elements, cnrtQueue_t queue);

// Pow：scalar_is_base=false 时 out=in^exponent；否则 out=exponent^in
void PowForwardLaunch(void *out, const void *in, float exponent, bool scalar_is_base, MluScalarType type,
                      int64_t num_elements, cnrtQueue_t queue);
void PowBackwardLaunch(void *grad_in, const void *grad_out, const void *in, float exponent, bool scalar_is_base,
                       MluScalarType type, int64_t num_elements, cnrtQueue_t queue);

// 二元前向（广播）：out[i] = op(a[bcast_a(i)], b[bcast_b(i)])
void BinaryForwardLaunch(void *out, const void *a, const void *b, MluBinaryOp op, MluScalarType type,
                         MluBroadcastMeta meta, int64_t num_elements, cnrtQueue_t queue);

// 二元反向：同形状时一次扫描直接写 grad_a/grad_b；有广播时对每个 grad 元素
// 串行 gather 累加 collapsed 维（FP32 累加，确定性顺序，不用 atomic）
void BinaryBackwardLaunch(void *grad_a, void *grad_b, const void *grad_out, const void *a, const void *b,
                          MluBinaryOp op, MluScalarType type, MluBroadcastMeta meta, int64_t numel_a, int64_t numel_b,
                          int64_t numel_out, cnrtQueue_t queue);

// 标量二元（仅 Add/Mul；Sub/Div 标量在框架层已折合为 Add/Mul）
void ScalarForwardLaunch(void *out, const void *in, float scalar, MluBinaryOp op, MluScalarType type,
                         int64_t num_elements, cnrtQueue_t queue);
void ScalarBackwardLaunch(void *grad_in, const void *grad_out, float scalar, MluBinaryOp op, MluScalarType type,
                          int64_t num_elements, cnrtQueue_t queue);

// 相等比较（输出 bool）；用于 causal mask 生成等。反向无意义（框架层 FATAL）
void EqualsForwardLaunch(void *out, const void *a, const void *b, MluScalarType type, MluBroadcastMeta meta,
                         int64_t num_elements, cnrtQueue_t queue);
void EqualsScalarForwardLaunch(void *out, const void *in, double scalar, MluScalarType type, int64_t num_elements,
                               cnrtQueue_t queue);

// -----------------------------------------------------------------------------
// embedding
// -----------------------------------------------------------------------------

// input 为 int64 token ids；越界 id 直接跳过（与 CUDA 版一致）
void EmbeddingForwardLaunch(const void *input, void *output, const void *weight, int64_t num_tokens, int64_t embed_dim,
                            int64_t vocab_size, MluScalarType type, cnrtQueue_t queue);
// grad_weight 调用前必须已清零（kernel 内为 atomic 累加）
void EmbeddingBackwardLaunch(const void *input, const void *grad_output, void *grad_weight, int64_t num_tokens,
                             int64_t embed_dim, int64_t vocab_size, MluScalarType type, cnrtQueue_t queue);

// -----------------------------------------------------------------------------
// layernorm / softmax / cross_entropy（行/样本级归约，全程 FP32 中间精度）
// -----------------------------------------------------------------------------

// input: (num_tokens, embed_dim)；mean_out/rstd_out 为 FP32，必须非空
void LayerNormForwardLaunch(const void *input, const void *weight, const void *bias, float *mean_out, float *rstd_out,
                            void *output, float eps, int64_t num_tokens, int64_t embed_dim, MluScalarType type,
                            cnrtQueue_t queue);
void LayerNormBackwardLaunch(const void *input, const void *grad_output, const float *mean, const float *rstd,
                             const void *weight, void *grad_input, void *grad_weight, void *grad_bias,
                             int64_t num_tokens, int64_t embed_dim, MluScalarType type, cnrtQueue_t queue);

// 沿 dim 的 softmax：outer/axis/inner 由 host 按 dims 拆解
void SoftmaxForwardLaunch(void *output, const void *input, int64_t outer_size, int64_t axis_size, int64_t inner_size,
                          MluScalarType type, cnrtQueue_t queue);
void SoftmaxBackwardLaunch(void *grad_input, const void *grad_output, const void *output, int64_t outer_size,
                           int64_t axis_size, int64_t inner_size, MluScalarType type, cnrtQueue_t queue);

// target_type ∈ {kU8, kI64}；input_type 为浮点。loss 为每样本值（bs,），
// 批均值由 host 侧 FP64 累加完成（与 CUDA 版一致）
void CrossEntropyForwardLaunch(const void *input, const void *target, void *loss, int64_t bs, int64_t num_classes,
                               MluScalarType target_type, MluScalarType input_type, cnrtQueue_t queue);
void CrossEntropyBackwardLaunch(const void *input, void *grad_input, const void *target, const void *grad_output,
                                int64_t bs, int64_t num_classes, MluScalarType target_type, MluScalarType input_type,
                                cnrtQueue_t queue);

// -----------------------------------------------------------------------------
// transform（transpose / slice / tril / triu / mask）
// -----------------------------------------------------------------------------

// 通用索引元数据（POD，按值传入 kernel；对应 CUDA 版的 device 端 dims/strides 缓冲）
#define MLU_MAX_META_DIMS 8
struct MluIndexMeta {
    int32_t ndim;
    int64_t dims[MLU_MAX_META_DIMS]; // slice: new_dims（输出形状）
    int64_t in_strides[MLU_MAX_META_DIMS];
    int64_t out_strides[MLU_MAX_META_DIMS];
    int64_t starts[MLU_MAX_META_DIMS]; // 仅 slice 使用
    int64_t steps[MLU_MAX_META_DIMS];  // 仅 slice 使用
};

// Tril/Triu 前向反向同形（保留或置 0），upper=false 为 tril
void TrilTriuLaunch(const void *input, void *output, int64_t rows, int64_t cols, int64_t diagonal, bool upper,
                    MluScalarType type, cnrtQueue_t queue);

// meta.in_strides 为输入步长，meta.out_strides 为输出步长；反向即 dim0/dim1 互换再调一次
void TransposeLaunch(const void *input, void *output, MluIndexMeta meta, int64_t dim0, int64_t dim1,
                     int64_t num_elements, MluScalarType type, cnrtQueue_t queue);

// meta 需填 dims(=new_dims)/in_strides/out_strides/starts/steps；反向的 grad_input 调用前必须已清零
void SliceForwardLaunch(const void *input, void *output, MluIndexMeta meta, int64_t num_elements, MluScalarType type,
                        cnrtQueue_t queue);
void SliceBackwardLaunch(const void *grad_output, void *grad_input, MluIndexMeta meta, int64_t num_elements,
                         MluScalarType type, cnrtQueue_t queue);

// MaskedFill：lead=true 时 mask 覆盖前导维（mask[i/inner]），否则尾部（mask[i%mask_extent]）；
// mask 在 host 侧已 Cast 成与 input 相同 dtype，按 == 1 判定
void MaskForwardLaunch(const void *input, const void *mask, void *output, float value, bool lead, int64_t mask_extent,
                       int64_t inner, int64_t num_elements, MluScalarType type, cnrtQueue_t queue);
void MaskBackwardLaunch(const void *grad_output, const void *mask, void *grad_input, bool lead, int64_t mask_extent,
                        int64_t inner, int64_t num_elements, MluScalarType type, cnrtQueue_t queue);

} // namespace infini_train::kernels::mlu
