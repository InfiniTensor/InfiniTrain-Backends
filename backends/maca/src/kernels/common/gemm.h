#pragma once

#include "infini_train/src/kernels/common/gemm.h"

namespace infini_train::kernels::maca {

void Gemm(Device device, GemmParams params);

} // namespace infini_train::kernels::maca
