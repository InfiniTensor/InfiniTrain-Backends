#include "infini_train_maca/backend.h"

#include <mutex>

#include "infini_train/include/core/privateuse1_backend.h"

#include "runtime/maca_guard_impl.h"

namespace infini_train::maca {

void RegisterBackend() {
    static std::once_flag once;
    std::call_once(once, []() {
        core::PrivateUse1BackendRegistration registration;
        registration.name = "maca";
        registration.default_autocast_dtype = DataType::kBFLOAT16;
        core::RegisterPrivateUse1Backend(registration);
    });
}

} // namespace infini_train::maca
