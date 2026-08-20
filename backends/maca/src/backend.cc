#include "infini_train_maca/backend.h"

#include <mutex>

#include "infini_train/include/core/privateuse1_backend.h"

#include "kernels/register_maca_kernels.h"
#include "runtime/maca_guard_impl.h"
#ifdef USE_MCCL
#include "ccl/mccl_impl.h"
#endif

namespace infini_train::maca {

void RegisterBackend() {
  static std::once_flag once;
  std::call_once(once, []() {
    core::PrivateUse1BackendRegistration registration;
    registration.name = "maca";
    registration.register_runtime = &core::maca::RegisterMacaRuntime;
    registration.register_kernels = &kernels::maca::RegisterMacaKernels;
#ifdef USE_MCCL
    registration.register_ccl = &core::maca::RegisterMcclBackend;
#endif
    core::RegisterPrivateUse1Backend(registration);
  });
}

} // namespace infini_train::maca
