#include "infini_train_mlu/backend.h"

#include <mutex>

#include "infini_train/include/core/privateuse1_backend.h"

namespace infini_train::mlu {

void RegisterBackend() {
    static std::once_flag once;
    std::call_once(once, []() {
        core::PrivateUse1BackendRegistration registration;
        registration.name = "mlu";
        registration.default_autocast_dtype = DataType::kBFLOAT16;
        core::RegisterPrivateUse1Backend(registration);
    });
}

} // namespace infini_train::mlu
