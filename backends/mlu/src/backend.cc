#include "infini_train_mlu/backend.h"

#include <mutex>

#include "infini_train/include/core/privateuse1_backend.h"
#include "infini_train/include/datatype.h"

namespace infini_train::backends::mlu {

void RegisterBackend() {
    static std::once_flag once;
    std::call_once(once, [] {
        // 前提（RegisterPrivateUse1Backend 内部 CHECK）：kPrivateUse1 的
        // DeviceGuardImpl 和 Cast/Fill/NoOpForward/NoOpBackward 四个必需 kernel
        // 已通过静态注册就位——这依赖最终可执行文件以 --whole-archive 方式
        // 链接 infini_train_mlu / infini_train_mlu_kernels（由 provider 的 MLUExecutable 接口负责 whole-archive
        // 链接）。
        core::PrivateUse1BackendRegistration registration;
        registration.name = "mlu";
        registration.default_autocast_dtype = DataType::kBFLOAT16;
        core::RegisterPrivateUse1Backend(registration);
    });
}

} // namespace infini_train::backends::mlu
