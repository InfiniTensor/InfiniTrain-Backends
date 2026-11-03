#pragma once

// 对应 master 上 src/core/runtime/cuda/cuda_dispatch.h。
//
// 本头由 host 侧（g++）编译单元使用：bang_fp16.h / bang_bf16.h 对
// __GNUC__ 提供 host 侧 half/bfloat16_t 支持。这两个头是 vendored 拷贝
// （third_party/bang_host/，原因见该目录 README），不是 cncc 资源目录里的。
// 注意 .mlu 文件不要 include 本头（见 kernels/mlu_kernel_api.h 的说明）。

#include <utility>
#include <vector>

#include <bang_bf16.h>
#include <bang_fp16.h>

#include "infini_train/include/core/backend_type_map.h"
#include "infini_train/include/dtype_dispatch.h"

#include "kernels/mlu_kernel_api.h"

// -----------------------------------------------------------------------------
// MLU 低精度类型映射：FP16 -> __half, BF16 -> bfloat16_t（BANG C 类型）
// 注意：host 侧 bang_fp16.h 不定义 half 别名，只能用 __half（bang_bf16.h
// 有 bfloat16_t 的 host typedef）。
// -----------------------------------------------------------------------------
namespace infini_train::core {
template <> struct BackendTypeMap<Device::DeviceType::kPrivateUse1, DataType::kFLOAT16> {
    using type = __half;
};

template <> struct BackendTypeMap<Device::DeviceType::kPrivateUse1, DataType::kBFLOAT16> {
    using type = bfloat16_t;
};
} // namespace infini_train::core

// 注册标准（非低精度）dtype；FP16/BF16 已在上面显式特化。
INFINI_REGISTER_STANDARD_BACKEND_TYPES(infini_train::Device::DeviceType::kPrivateUse1)

// -----------------------------------------------------------------------------
// MLU dispatch 类型表
// MLU590 硬件/toolchain 不支持 FP64 设备运算（double 比较需软件运行时
// __nedf2，cncc 后端无法选择），因此 MLU kernel dispatch 一律排除 kFLOAT64；
// 真被请求时由 dispatch 在运行期报清晰错误。
// -----------------------------------------------------------------------------
#define INFINI_MLU_FLOATING_TYPES DataType::kFLOAT32, DataType::kFLOAT16, DataType::kBFLOAT16
#define INFINI_MLU_ALL_NUMERIC_TYPES INFINI_MLU_FLOATING_TYPES, INFINI_ALL_INTEGRAL_TYPES
#define INFINI_MLU_ALL_NUMERIC_AND_LOGICAL_TYPES INFINI_MLU_ALL_NUMERIC_TYPES, INFINI_LOGICAL_TYPES

namespace infini_train::core::mlu {

template <DataType DType> struct MluTypeMap : BackendTypeMap<Device::DeviceType::kPrivateUse1, DType> {};

// -----------------------------------------------------------------------------
// MLU dispatch helpers
// -----------------------------------------------------------------------------

template <DataType... AllowedDTypes, typename Functor, typename... Args>
auto DispatchMluFunc(DataType dtype, Functor &&func, std::string_view context_identifier = "", Args &&...args) {
    return infini_train::DispatchByTypeMap<MluTypeMap, AllowedDTypes...>(
        dtype, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

template <typename... AllowedTypeLists, typename Functor, typename... Args>
auto DispatchMluFunc(const std::vector<DataType> &dtypes, Functor &&func, std::string_view context_identifier = "",
                     Args &&...args) {
    return infini_train::DispatchByTypeMap<MluTypeMap, AllowedTypeLists...>(
        dtypes, std::forward<Functor>(func), context_identifier, std::forward<Args>(args)...);
}

} // namespace infini_train::core::mlu

// -----------------------------------------------------------------------------
// MluTypeTag<T>(): C++ 类型 -> kernels::mlu::MluScalarType（给启动器用）
// -----------------------------------------------------------------------------
namespace infini_train::kernels::mlu {

template <typename T> constexpr MluScalarType MluTypeTag();
template <> constexpr MluScalarType MluTypeTag<bool>() { return MluScalarType::kBool; }
template <> constexpr MluScalarType MluTypeTag<uint8_t>() { return MluScalarType::kU8; }
template <> constexpr MluScalarType MluTypeTag<int8_t>() { return MluScalarType::kI8; }
template <> constexpr MluScalarType MluTypeTag<uint16_t>() { return MluScalarType::kU16; }
template <> constexpr MluScalarType MluTypeTag<int16_t>() { return MluScalarType::kI16; }
template <> constexpr MluScalarType MluTypeTag<uint32_t>() { return MluScalarType::kU32; }
template <> constexpr MluScalarType MluTypeTag<int32_t>() { return MluScalarType::kI32; }
template <> constexpr MluScalarType MluTypeTag<uint64_t>() { return MluScalarType::kU64; }
template <> constexpr MluScalarType MluTypeTag<int64_t>() { return MluScalarType::kI64; }
template <> constexpr MluScalarType MluTypeTag<__half>() { return MluScalarType::kF16; }
template <> constexpr MluScalarType MluTypeTag<bfloat16_t>() { return MluScalarType::kBF16; }
template <> constexpr MluScalarType MluTypeTag<float>() { return MluScalarType::kF32; }
template <> constexpr MluScalarType MluTypeTag<double>() { return MluScalarType::kF64; }

} // namespace infini_train::kernels::mlu
