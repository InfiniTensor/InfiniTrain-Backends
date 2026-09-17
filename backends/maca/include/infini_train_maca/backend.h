#pragma once

#include "gflags/gflags_declare.h"

DECLARE_bool(maca_multithread_workarounds);
DECLARE_bool(maca_retain_async_pool);

namespace infini_train::maca {

// Registers MACA as the process-wide PrivateUse1 provider. Device runtime
// initialization remains lazy until the first device operation.
void RegisterBackend();

} // namespace infini_train::maca
