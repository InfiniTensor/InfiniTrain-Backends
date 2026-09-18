#pragma once

#include <cub/cub.cuh>

namespace infini_train::kernels::maca {

// MACA ships a CUB compatible with the pre-2.8 API (cub::Sum/Max/Min).
// Match the CUDA reduction aliases used by the shared kernel conventions.
using CubSumOp = cub::Sum;
using CubMaxOp = cub::Max;
using CubMinOp = cub::Min;

} // namespace infini_train::kernels::maca
