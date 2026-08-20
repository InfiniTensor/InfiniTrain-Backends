#include "kernels/register_maca_kernels.h"

namespace infini_train::kernels::maca {

void RegisterAccumulateGradKernels();
void RegisterCastKernels();
void RegisterCommKernels();
void RegisterConcatKernels();
void RegisterCrossEntropyKernels();
void RegisterElementwiseKernels();
void RegisterEmbeddingKernels();
void RegisterFillKernels();
void RegisterGatherKernels();
void RegisterLayerNormKernels();
void RegisterLinearKernels();
void RegisterNoOpKernels();
void RegisterOuterKernels();
void RegisterReductionKernels();
void RegisterScatterKernels();
void RegisterSliceKernels();
void RegisterSoftmaxKernels();
void RegisterSplitKernels();
void RegisterStackKernels();
void RegisterTopKKernels();
void RegisterTransformKernels();
void RegisterVocabParallelCrossEntropyKernels();

void RegisterMacaKernels() {
  RegisterAccumulateGradKernels();
  RegisterCastKernels();
  RegisterCommKernels();
  RegisterConcatKernels();
  RegisterCrossEntropyKernels();
  RegisterElementwiseKernels();
  RegisterEmbeddingKernels();
  RegisterFillKernels();
  RegisterGatherKernels();
  RegisterLayerNormKernels();
  RegisterLinearKernels();
  RegisterNoOpKernels();
  RegisterOuterKernels();
  RegisterReductionKernels();
  RegisterScatterKernels();
  RegisterSliceKernels();
  RegisterSoftmaxKernels();
  RegisterSplitKernels();
  RegisterStackKernels();
  RegisterTopKKernels();
  RegisterTransformKernels();
  RegisterVocabParallelCrossEntropyKernels();
}

} // namespace infini_train::kernels::maca
